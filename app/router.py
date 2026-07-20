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
from .protocol import (
    responses_req_to_chat, chat_resp_to_responses, chat_stream_to_responses,
    anthropic_req_to_chat, chat_resp_to_anthropic, chat_stream_to_anthropic,
)
from .moa import is_moa, moa_pipeline_name, run_moa

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
    if is_moa(model):
        return await _handle_moa(request, "chat", body, model)
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
                data = _sanitize_chat_output(result.json())
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
            return _sse_response(_safe_stream(_chat_ensure_done(counted_stream(first, result)), _chat_error_tail()))
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
    for mname in (cfg.raw.get("moa") or {}):
        data.append(
            {"id": f"moa:{mname}", "object": "model", "created": created, "owned_by": "moa"}
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
    """OpenAI Responses API（统一枢纽）。

    - 上游原生支持 Responses（supports_responses=True，如火山 Ark / agnes）-> 透传
    - 其他上游（anthropic / gemini / 仅 chat 的 openai_like）-> responses->chat->原生，
      响应再 chat->responses 转回，流式同理
    鉴权 / 别名解析 / 密钥池故障转移 / 限流 / 统计与 chat_completions 一致。
    工具调用经 normalize，非法 JSON 兜底；流式中途异常合成合法收尾，不断连。
    """
    return await _dispatch_unified(request, wire="responses")


@router.post("/v1/messages")
async def messages(request: Request):
    """Anthropic Messages API 输入端点（统一枢纽）。

    anthropic 输入 -> chat -> 任意上游 -> chat -> anthropic 输出。
    让只说 Anthropic 协议的客户端也能聚合到任意 provider。
    """
    return await _dispatch_unified(request, wire="anthropic")


async def _handle_moa(request: Request, wire: str, body: dict, model: str):
    """MOA 请求处理：归一化为 chat -> run_moa -> 转回客户端 wire 格式。"""
    _refresh_runtime(request)
    caller = _caller_key(request)
    name = moa_pipeline_name(model)
    stream = bool(body.get("stream"))

    if wire == "responses":
        chat_body = responses_req_to_chat(body)
    elif wire == "anthropic":
        chat_body = anthropic_req_to_chat(body)
    else:
        chat_body = dict(body)

    t0 = time.monotonic()
    try:
        result = await run_moa(request.app.state.config, request.app.state.providers,
                               request.app.state.pool, chat_body, name, stream)
    except KeyError:
        return _error(404, f"MOA pipeline `{name}` does not exist",
                      "invalid_request_error", "model_not_found")
    except Exception as exc:
        await _record_stats(request, api_key=caller, model=model, provider="moa",
                            latency_ms=int((time.monotonic() - t0) * 1000), status=502, stream=int(stream))
        return _error(502, f"MOA failed: {exc!r}"[:300], "upstream_error", "upstream_error")

    if not stream:
        chat_data = result.json() if isinstance(result, httpx.Response) else result
        if wire == "responses":
            out = chat_resp_to_responses(chat_data, model)
        elif wire == "anthropic":
            out = chat_resp_to_anthropic(chat_data, model)
        else:
            out = _sanitize_chat_output(chat_data)
        pt = out.get("usage", {}).get("input_tokens", 0) if wire != "chat" else 0
        ct = out.get("usage", {}).get("output_tokens", 0) if wire != "chat" else 0
        await _record_stats(request, api_key=caller, model=model, provider="moa",
                            prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct,
                            latency_ms=int((time.monotonic() - t0) * 1000), status=200, stream=0)
        return JSONResponse(content=out)

    # 流式：aggregator 的 chat 流 -> 客户端 wire
    if wire == "responses":
        conv = chat_stream_to_responses(result, model, True)
        tail = _responses_error_tail(model)
    elif wire == "anthropic":
        conv = chat_stream_to_anthropic(result, model)
        tail = _anthropic_error_tail()
    else:
        conv = _chat_ensure_done(result)
        tail = _chat_error_tail()
    await _record_stats(request, api_key=caller, model=model, provider="moa",
                        latency_ms=int((time.monotonic() - t0) * 1000), status=200, stream=1)
    return _sse_response(_safe_stream(conv, tail))


async def _dispatch_unified(request: Request, wire: str) -> JSONResponse | StreamingResponse:
    """responses / anthropic 两种输入面的统一分发。

    wire="responses"  : 输入/输出 OpenAI Responses
    wire="anthropic"  : 输入/输出 Anthropic Messages
    内部一律转 chat 调 provider.chat_completions（复用各家 chat<->原生转换）。
    supports_responses=True 的 openai_like 上游在 responses 面走原生透传快路径。
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
    if is_moa(model):
        return await _handle_moa(request, wire, body, model)
    cfg = request.app.state.config

    try:
        provider_name, real_model = cfg.resolve_model(model)
    except KeyError:
        return _error(
            404, f"The model `{model}` does not exist",
            "invalid_request_error", "model_not_found",
        )

    provider = request.app.state.providers.get(provider_name)
    if provider is None:
        return _error(500, f"Provider `{provider_name}` 初始化失败", "server_error")

    # responses 面且上游原生支持 -> 透传快路径
    passthrough = wire == "responses" and provider.supports_responses and callable(getattr(provider, "responses", None))

    pool = request.app.state.pool
    caller = _caller_key(request)
    if not await _ratelimit_allow(request, f"{caller}:{model}"):
        return _error(429, "Rate limit exceeded", "rate_limit_exceeded", "rate_limit_exceeded")

    t0 = time.monotonic()
    attempts = max(1, pool.size(provider_name))
    last_msg = f"provider `{provider_name}` 无可用上游 key"

    want_usage = wire == "responses"  # responses 流尽量带 usage

    for _ in range(attempts):
        key = pool.acquire(provider_name)
        if key is None:
            break
        try:
            if passthrough:
                result = await provider.responses(real_model, body, key, stream)
                if isinstance(result, httpx.Response) and result.status_code >= 400:
                    raise UpstreamError(result.status_code, result.text[:300])
                if not stream:
                    data = _sanitize_responses_output(result.json())
                    pool.report_success(provider_name, key)
                    pt, ct, tt = _usage_triplet(data)
                    await _record_stats(request, api_key=caller, model=model,
                        provider=provider_name, prompt_tokens=pt, completion_tokens=ct,
                        total_tokens=tt, latency_ms=int((time.monotonic()-t0)*1000), status=200, stream=0)
                    return JSONResponse(content=data)
                try:
                    first = await result.__anext__()
                except StopAsyncIteration:
                    first = None
                pool.report_success(provider_name, key)
                return _sse_response(_safe_stream(result, _responses_error_tail(model)))

            # 转换路径：input -> chat
            if wire == "responses":
                chat_body = responses_req_to_chat(body)
            else:
                chat_body = anthropic_req_to_chat(body)
            chat_body["model"] = real_model
            chat_body["stream"] = stream
            result = await provider.chat_completions(real_model, chat_body, key, stream)
            if isinstance(result, httpx.Response) and result.status_code >= 400:
                raise UpstreamError(result.status_code, result.text[:300])

            if not stream:
                chat_data = result.json()
                pool.report_success(provider_name, key)
                if wire == "responses":
                    out = chat_resp_to_responses(chat_data, model)
                else:
                    out = chat_resp_to_anthropic(chat_data, model)
                pt = out.get("usage", {}).get("input_tokens", 0)
                ct = out.get("usage", {}).get("output_tokens", 0)
                await _record_stats(request, api_key=caller, model=model,
                    provider=provider_name, prompt_tokens=pt, completion_tokens=ct,
                    total_tokens=pt+ct, latency_ms=int((time.monotonic()-t0)*1000), status=200, stream=0)
                return JSONResponse(content=out)

            # 流式转换
            try:
                first = await result.__anext__()
            except StopAsyncIteration:
                first = None
            pool.report_success(provider_name, key)
            if wire == "responses":
                conv = chat_stream_to_responses(_prepend(first, result), model, want_usage)
                tail = _responses_error_tail(model)
            else:
                conv = chat_stream_to_anthropic(_prepend(first, result), model)
                tail = _anthropic_error_tail()
            return _sse_response(_safe_stream(conv, tail))
        except UpstreamError as exc:
            pool.report_failure(provider_name, key)
            last_msg = f"上游错误 ({provider_name}): HTTP {exc.status_code} {exc.message[:200]}"
            logger.warning("key 故障转移: %s", last_msg)
        except Exception as exc:
            pool.report_failure(provider_name, key)
            last_msg = f"上游请求失败 ({provider_name}): {exc!r}"[:300]
            logger.warning("key 故障转移: %s", last_msg)

    await _record_stats(request, api_key=caller, model=model, provider=provider_name,
        latency_ms=int((time.monotonic()-t0)*1000), status=502, stream=int(stream))
    return _error(502, last_msg, "upstream_error", "upstream_error")


def _sanitize_responses_output(data: dict) -> dict:
    """透传 responses 输出时修复 function_call 非法 arguments，避免客户端解析崩。"""
    from .protocol import repair_arguments
    for item in data.get("output") or []:
        if isinstance(item, dict) and item.get("type") == "function_call":
            item["arguments"] = repair_arguments(item.get("arguments"))
    return data


async def _prepend(first: bytes | None, agen: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    if first is not None:
        yield first
    async for chunk in agen:
        yield chunk


def _sse_response(agen: AsyncIterator[bytes]) -> StreamingResponse:
    return StreamingResponse(agen, media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


async def _safe_stream(agen: AsyncIterator[bytes], error_tail: bytes) -> AsyncIterator[bytes]:
    """流式中途上游异常时合成合法收尾，避免客户端会话因断流而中断。"""
    try:
        async for chunk in agen:
            yield chunk
    except Exception as exc:
        logger.warning("流式中途异常，合成收尾不断连: %r", exc)
        if error_tail:
            yield error_tail


async def _chat_ensure_done(agen: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """chat 流：若上游提前结束且未发 [DONE]，补一个 finish chunk + [DONE]，
    避免客户端因缺少终止符而挂起/中断会话。"""
    saw_done = False
    async for chunk in agen:
        if b"[DONE]" in chunk:
            saw_done = True
        yield chunk
    if not saw_done:
        yield _chat_error_tail()


def _sanitize_chat_output(data: dict) -> dict:
    """chat 输出工具调用规整：补 id、修非法 arguments。"""
    from .protocol import normalize_tool_calls
    for ch in data.get("choices") or []:
        msg = (ch or {}).get("message") or {}
        if msg.get("tool_calls"):
            msg["tool_calls"] = normalize_tool_calls(msg["tool_calls"])
    return data


def _chat_error_tail() -> bytes:
    """chat 流式中断时的合法收尾：一个 finish chunk + [DONE]。"""
    import json as _json, time as _time
    chunk = {"object": "chat.completion.chunk", "created": int(_time.time()),
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    return ("data: " + _json.dumps(chunk, ensure_ascii=False) + "\n\ndata: [DONE]\n\n").encode()


def _responses_error_tail(model: str) -> bytes:
    import json as _json, uuid as _uuid, time as _time
    rid = f"resp_{_uuid.uuid4().hex[:24]}"
    payload = {"type": "response.completed", "response": {
        "id": rid, "object": "response", "created_at": int(_time.time()),
        "status": "completed", "model": model, "output": []}}
    return ("event: response.completed\ndata: " + _json.dumps(payload, ensure_ascii=False) + "\n\n").encode()


def _anthropic_error_tail() -> bytes:
    import json as _json
    out = ("event: message_delta\ndata: " + _json.dumps(
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
         "usage": {"output_tokens": 0}}) + "\n\n").encode()
    out += ("event: message_stop\ndata: " + _json.dumps({"type": "message_stop"}) + "\n\n").encode()
    return out


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
