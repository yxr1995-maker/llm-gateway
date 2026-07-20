"""对外 OpenAI 兼容路由：POST /v1/chat/completions、GET /v1/models。

- 鉴权：master_key 非空时强制 Bearer
- 模型解析：别名 / provider/model / 已配置模型名
- 故障转移：上游非 2xx 或网络错误时标记 key 失败，池内轮询下一个 key
  （最多试 len(keys) 次），全部失败返回 502 OpenAI 错误格式
- 流式：StreamingResponse 边收边发（text/event-stream）
- 统计 / 限流：duck-typing 从 app.state 取，缺失则静默跳过
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import AsyncIterator

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .providers import UpstreamError, build_providers

logger = logging.getLogger("llm-gateway.router")

router = APIRouter()


# ------------------------------------------------------------------ 工具
def _error(status: int, message: str, err_type: str = "server_error",
           code: str | None = None) -> JSONResponse:
    """OpenAI 错误格式响应。"""
    return JSONResponse(
        status_code=status,
        content={
            "error": {"message": message, "type": err_type, "param": None, "code": code}
        },
    )


def _caller_key(request: Request) -> str:
    """调用方 Bearer token（统计/限流维度）。"""
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def _check_auth(request: Request) -> JSONResponse | None:
    """master_key 非空时强制鉴权；通过返回 None。"""
    master = request.app.state.config.master_key
    if not master:
        return None
    if _caller_key(request) != master:
        return _error(
            401,
            "Invalid or missing API key",
            "authentication_error",
            "invalid_api_key",
        )
    return None


def _refresh_runtime(request: Request) -> None:
    """按 mtime 热重载配置；变化时重建 providers 并同步密钥池。"""
    cfg = request.app.state.config
    try:
        if cfg.maybe_reload():
            request.app.state.providers = build_providers(cfg.providers)
            request.app.state.pool.sync(cfg.providers)
            request.app.state.master_key = cfg.master_key  # 同步给 admin 鉴权
    except Exception:
        logger.exception("配置热重载失败，沿用旧配置")


async def _record_stats(request: Request, *, api_key: str, model: str, provider: str,
                        prompt_tokens: int = 0, completion_tokens: int = 0,
                        total_tokens: int = 0, latency_ms: int = 0, status: int = 200,
                        stream: int = 0) -> None:
    """duck-typing 调用 app.state.stats.record；缺失/出错静默跳过。"""
    stats = getattr(request.app.state, "stats", None)
    if stats is None:
        return
    ts = time.time()
    try:
        await stats.record(
            ts=ts, api_key=api_key, model=model, provider=provider,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            total_tokens=total_tokens, latency_ms=latency_ms,
            status=status, stream=stream,
        )
    except TypeError:
        # 位置参数兜底（防御签名差异）
        try:
            await stats.record(ts, api_key, model, provider, prompt_tokens,
                               completion_tokens, total_tokens, latency_ms,
                               status, stream)
        except Exception:
            pass
    except Exception:
        pass


async def _ratelimit_allow(request: Request, key: str) -> bool:
    """duck-typing 调用 app.state.ratelimiter.allow；不存在或出错则放行。"""
    limiter = getattr(request.app.state, "ratelimiter", None)
    if limiter is None:
        return True
    try:
        result = limiter.allow(key)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)
    except Exception:
        logger.exception("ratelimiter.allow 出错，放行")
        return True


# ------------------------------------------------------------------ 路由
@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _refresh_runtime(request)
    auth_err = _check_auth(request)
    if auth_err is not None:
        return auth_err

    try:
        body = await request.json()
    except Exception:
        return _error(400, "Invalid JSON body", "invalid_request_error")
    if not isinstance(body, dict):
        return _error(400, "Invalid request body", "invalid_request_error")

    model = str(body.get("model") or "")
    stream = bool(body.get("stream"))
    cfg = request.app.state.config

    try:
        provider_name, real_model = cfg.resolve_model(model)
    except KeyError:
        return _error(
            404,
            f"The model `{model}` does not exist",
            "invalid_request_error",
            "model_not_found",
        )

    provider = request.app.state.providers.get(provider_name)
    if provider is None:
        return _error(500, f"Provider `{provider_name}` 初始化失败", "server_error")
    pool = request.app.state.pool
    caller = _caller_key(request)

    # 限流：按调用方 key + 模型
    if not await _ratelimit_allow(request, f"{caller}:{model}"):
        return _error(
            429, "Rate limit exceeded", "rate_limit_exceeded", "rate_limit_exceeded"
        )

    t0 = time.monotonic()
    attempts = max(1, pool.size(provider_name))
    last_msg = f"provider `{provider_name}` 无可用上游 key"

    async def record_stream_done(nchars: int) -> None:
        """流式结束后的用量记录：字符数/4 估算。"""
        est = nchars // 4 if nchars else 0
        await _record_stats(
            request, api_key=caller, model=model, provider=provider_name,
            prompt_tokens=0, completion_tokens=est, total_tokens=est,
            latency_ms=int((time.monotonic() - t0) * 1000), status=200, stream=1,
        )

    async def counted_stream(first: bytes | None,
                             agen: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        """边收边发，同时累计字符数用于估算用量。"""
        nchars = 0
        if first is not None:
            nchars += len(first.decode("utf-8", "ignore"))
            yield first
        async for chunk in agen:
            nchars += len(chunk.decode("utf-8", "ignore"))
            yield chunk
        await record_stream_done(nchars)

    for _ in range(attempts):
        key = pool.acquire(provider_name)
        if key is None:
            break  # 池内已无未冷却的 key
        try:
            result = await provider.chat_completions(real_model, body, key, stream)
            # openai_like 非流式直接返回上游响应，这里统一判定状态码
            if isinstance(result, httpx.Response) and result.status_code >= 400:
                raise UpstreamError(result.status_code, result.text[:300])

            if not stream:
                data = result.json()
                pool.report_success(provider_name, key)
                usage = data.get("usage") or {}
                await _record_stats(
                    request, api_key=caller, model=model, provider=provider_name,
                    prompt_tokens=usage.get("prompt_tokens", 0) or 0,
                    completion_tokens=usage.get("completion_tokens", 0) or 0,
                    total_tokens=usage.get("total_tokens", 0) or 0,
                    latency_ms=int((time.monotonic() - t0) * 1000),
                    status=200, stream=0,
                )
                return JSONResponse(content=data)

            # 流式：先取首个 chunk，确认上游 2xx 且开始产出后再响应客户端，
            # 否则仍可在池内故障转移（此时响应头尚未发出）
            try:
                first = await result.__anext__()
            except StopAsyncIteration:
                first = None
            pool.report_success(provider_name, key)
            return StreamingResponse(
                counted_stream(first, result),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        except UpstreamError as exc:
            pool.report_failure(provider_name, key)
            last_msg = f"上游错误 ({provider_name}): HTTP {exc.status_code} {exc.message[:200]}"
            logger.warning("key 故障转移: %s", last_msg)
        except Exception as exc:  # 网络/解析等错误，同样转移
            pool.report_failure(provider_name, key)
            last_msg = f"上游请求失败 ({provider_name}): {exc!r}"[:300]
            logger.warning("key 故障转移: %s", last_msg)

    # 全部 key 失败：502 OpenAI 错误格式
    await _record_stats(
        request, api_key=caller, model=model, provider=provider_name,
        latency_ms=int((time.monotonic() - t0) * 1000), status=502, stream=int(stream),
    )
    return _error(502, last_msg, "upstream_error", "upstream_error")


@router.get("/v1/models")
async def list_models(request: Request):
    """返回所有已配置模型 + 别名（OpenAI models list 格式）。"""
    _refresh_runtime(request)
    auth_err = _check_auth(request)
    if auth_err is not None:
        return auth_err

    cfg = request.app.state.config
    created = int(time.time())
    data: list[dict] = []
    for alias in cfg.aliases:
        data.append(
            {"id": str(alias), "object": "model", "created": created, "owned_by": "alias"}
        )
    for pname, pcfg in cfg.providers.items():
        for m in (pcfg or {}).get("models") or []:
            data.append(
                {"id": str(m), "object": "model", "created": created, "owned_by": pname}
            )
    return {"object": "list", "data": data}


def _usage_triplet(data: dict) -> tuple[int, int, int]:
    """从响应 JSON 提取 (prompt, completion, total)；兼容 chat 与 Responses 两种 usage 命名。"""
    usage = (data or {}).get("usage") or {}
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    total = usage.get("total_tokens") or (prompt + completion)
    return prompt, completion, total


@router.post("/v1/responses")
async def responses(request: Request):
    """OpenAI Responses API 透传（Codex 等 Responses-only 客户端）。

    与 /v1/chat/completions 相同的鉴权 / 别名解析 / 密钥池故障转移，
    但转发到上游 /responses 端点。仅 openai_like provider 支持
    （上游需原生实现 Responses 协议，如火山 Ark / agnes）；
    anthropic / gemini provider 不支持时返回 501。
    """
    _refresh_runtime(request)
    auth_err = _check_auth(request)
    if auth_err is not None:
        return auth_err

    try:
        body = await request.json()
    except Exception:
        return _error(400, "Invalid JSON body", "invalid_request_error")
    if not isinstance(body, dict):
        return _error(400, "Invalid request body", "invalid_request_error")

    model = str(body.get("model") or "")
    stream = bool(body.get("stream"))
    cfg = request.app.state.config

    try:
        provider_name, real_model = cfg.resolve_model(model)
    except KeyError:
        return _error(
            404,
            f"The model `{model}` does not exist",
            "invalid_request_error",
            "model_not_found",
        )

    provider = request.app.state.providers.get(provider_name)
    if provider is None:
        return _error(500, f"Provider `{provider_name}` 初始化失败", "server_error")
    call = getattr(provider, "responses", None)
    if not callable(call):
        return _error(
            501,
            f"Provider `{provider_name}` ({provider.type}) 不支持 Responses API",
            "invalid_request_error",
            "unsupported_endpoint",
        )

    pool = request.app.state.pool
    caller = _caller_key(request)

    if not await _ratelimit_allow(request, f"{caller}:{model}"):
        return _error(
            429, "Rate limit exceeded", "rate_limit_exceeded", "rate_limit_exceeded"
        )

    t0 = time.monotonic()
    attempts = max(1, pool.size(provider_name))
    last_msg = f"provider `{provider_name}` 无可用上游 key"

    async def record_stream_done(nchars: int) -> None:
        est = nchars // 4 if nchars else 0
        await _record_stats(
            request, api_key=caller, model=model, provider=provider_name,
            prompt_tokens=0, completion_tokens=est, total_tokens=est,
            latency_ms=int((time.monotonic() - t0) * 1000), status=200, stream=1,
        )

    async def counted_stream(first: bytes | None,
                             agen: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        nchars = 0
        if first is not None:
            nchars += len(first.decode("utf-8", "ignore"))
            yield first
        async for chunk in agen:
            nchars += len(chunk.decode("utf-8", "ignore"))
            yield chunk
        await record_stream_done(nchars)

    for _ in range(attempts):
        key = pool.acquire(provider_name)
        if key is None:
            break
        try:
            result = await call(real_model, body, key, stream)
            if isinstance(result, httpx.Response) and result.status_code >= 400:
                raise UpstreamError(result.status_code, result.text[:300])

            if not stream:
                data = result.json()
                pool.report_success(provider_name, key)
                prompt_t, completion_t, total_t = _usage_triplet(data)
                await _record_stats(
                    request, api_key=caller, model=model, provider=provider_name,
                    prompt_tokens=prompt_t, completion_tokens=completion_t,
                    total_tokens=total_t,
                    latency_ms=int((time.monotonic() - t0) * 1000),
                    status=200, stream=0,
                )
                return JSONResponse(content=data)

            try:
                first = await result.__anext__()
            except StopAsyncIteration:
                first = None
            pool.report_success(provider_name, key)
            return StreamingResponse(
                counted_stream(first, result),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        except UpstreamError as exc:
            pool.report_failure(provider_name, key)
            last_msg = f"上游错误 ({provider_name}): HTTP {exc.status_code} {exc.message[:200]}"
            logger.warning("key 故障转移: %s", last_msg)
        except Exception as exc:
            pool.report_failure(provider_name, key)
            last_msg = f"上游请求失败 ({provider_name}): {exc!r}"[:300]
            logger.warning("key 故障转移: %s", last_msg)

    await _record_stats(
        request, api_key=caller, model=model, provider=provider_name,
        latency_ms=int((time.monotonic() - t0) * 1000), status=502, stream=int(stream),
    )
    return _error(502, last_msg, "upstream_error", "upstream_error")


@router.post("/v1/embeddings")
async def embeddings(request: Request):
    """向量 embedding 透传（OpenAI 兼容 /v1/embeddings）。

    仅 openai_like provider 支持；故障转移同 chat_completions。
    """
    _refresh_runtime(request)
    auth_err = _check_auth(request)
    if auth_err is not None:
        return auth_err

    try:
        body = await request.json()
    except Exception:
        return _error(400, "Invalid JSON body", "invalid_request_error")
    if not isinstance(body, dict):
        return _error(400, "Invalid request body", "invalid_request_error")

    model = str(body.get("model") or "")
    cfg = request.app.state.config
    try:
        provider_name, real_model = cfg.resolve_model(model)
    except KeyError:
        return _error(
            404,
            f"The model `{model}` does not exist",
            "invalid_request_error",
            "model_not_found",
        )

    provider = request.app.state.providers.get(provider_name)
    if provider is None:
        return _error(500, f"Provider `{provider_name}` 初始化失败", "server_error")
    embed_fn = getattr(provider, "embeddings", None)
    if not callable(embed_fn):
        return _error(
            501,
            f"Provider `{provider_name}` 不支持 embeddings",
            "unsupported_endpoint",
        )
    pool = request.app.state.pool
    caller = _caller_key(request)

    if not await _ratelimit_allow(request, f"{caller}:{model}"):
        return _error(
            429, "Rate limit exceeded", "rate_limit_exceeded", "rate_limit_exceeded"
        )

    t0 = time.monotonic()
    attempts = max(1, pool.size(provider_name))
    last_msg = f"provider `{provider_name}` 无可用上游 key"

    for _ in range(attempts):
        key = pool.acquire(provider_name)
        if key is None:
            break
        try:
            resp = await embed_fn(real_model, body, key)
            if isinstance(resp, httpx.Response) and resp.status_code >= 400:
                raise UpstreamError(resp.status_code, resp.text[:300])
            data = resp.json()
            pool.report_success(provider_name, key)
            usage = data.get("usage") or {}
            await _record_stats(
                request, api_key=caller, model=model, provider=provider_name,
                prompt_tokens=usage.get("prompt_tokens", 0) or 0,
                completion_tokens=0,
                total_tokens=usage.get("total_tokens", 0) or 0,
                latency_ms=int((time.monotonic() - t0) * 1000),
                status=200, stream=0,
            )
            return JSONResponse(content=data)
        except UpstreamError as exc:
            pool.report_failure(provider_name, key)
            last_msg = f"上游错误 ({provider_name}): HTTP {exc.status_code} {exc.message[:200]}"
            logger.warning("embeddings 故障转移: %s", last_msg)
        except Exception as exc:
            pool.report_failure(provider_name, key)
            last_msg = f"上游请求失败 ({provider_name}): {exc!r}"[:300]
            logger.warning("embeddings 故障转移: %s", last_msg)

    await _record_stats(
        request, api_key=caller, model=model, provider=provider_name,
        latency_ms=int((time.monotonic() - t0) * 1000), status=502, stream=0,
    )
    return _error(502, last_msg, "upstream_error", "upstream_error")
