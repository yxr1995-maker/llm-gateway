"""OpenAI-compatible routes：POST /v1/chat/completions、GET /v1/models。

- auth: enforce Bearer when master_key is non-empty
- model resolution: alias / provider/model / configured model name
- failover: on upstream non-2xx or network error, mark the key failed and rotate to the next key in the pool
   (up to len(keys) attempts); all-fail returns 502 in OpenAI error format
- streaming: StreamingResponse streams as it receives (text/event-stream)
- stats / rate-limit: duck-typed from app.state, silently skipped if absent
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


# ------------------------------------------------------------------ helpers
def _error(status: int, message: str, err_type: str = "server_error",
           code: str | None = None) -> JSONResponse:
    """OpenAI-format error response."""
    return JSONResponse(
        status_code=status,
        content={
            "error": {"message": message, "type": err_type, "param": None, "code": code}
        },
    )


def _caller_key(request: Request) -> str:
    """Caller Bearer token (stats/rate-limit dimension)."""
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def _check_auth(request: Request) -> JSONResponse | None:
    """Enforce auth when master_key is non-empty; return None if passed."""
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
    """Hot-reload config by mtime; rebuild providers and sync the key pool on change."""
    cfg = request.app.state.config
    try:
        if cfg.maybe_reload():
            request.app.state.providers = build_providers(cfg.providers)
            request.app.state.pool.sync(cfg.providers)
            request.app.state.master_key = cfg.master_key  # synced for admin auth
    except Exception:
        logger.exception("config hot-reload failed, keeping previous config")


async def _record_stats(request: Request, *, api_key: str, model: str, provider: str,
                        prompt_tokens: int = 0, completion_tokens: int = 0,
                        total_tokens: int = 0, latency_ms: int = 0, status: int = 200,
                        stream: int = 0) -> None:
    """Duck-typed call to app.state.stats.record; silently skipped if absent/erroring."""
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
        # positional-arg fallback (defends against signature differences)
        try:
            await stats.record(ts, api_key, model, provider, prompt_tokens,
                               completion_tokens, total_tokens, latency_ms,
                               status, stream)
        except Exception:
            pass
    except Exception:
        pass


async def _ratelimit_allow(request: Request, key: str) -> bool:
    """Duck-typed call to app.state.ratelimiter.allow; allow if absent/erroring."""
    limiter = getattr(request.app.state, "ratelimiter", None)
    if limiter is None:
        return True
    try:
        result = limiter.allow(key)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)
    except Exception:
        logger.exception("ratelimiter.allow errored, allowing")
        return True


# ------------------------------------------------------------------ route
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
        return _error(500, f"Provider `{provider_name}` failed to initialize", "server_error")
    pool = request.app.state.pool
    caller = _caller_key(request)

    # rate-limit: per caller key + model
    if not await _ratelimit_allow(request, f"{caller}:{model}"):
        return _error(
            429, "Rate limit exceeded", "rate_limit_exceeded", "rate_limit_exceeded"
        )

    t0 = time.monotonic()
    attempts = max(1, pool.size(provider_name))
    last_msg = f"provider `{provider_name}` has no available upstream key"

    async def record_stream_done(nchars: int) -> None:
        """Usage record after streaming: estimated as chars/4."""
        est = nchars // 4 if nchars else 0
        await _record_stats(
            request, api_key=caller, model=model, provider=provider_name,
            prompt_tokens=0, completion_tokens=est, total_tokens=est,
            latency_ms=int((time.monotonic() - t0) * 1000), status=200, stream=1,
        )

    async def counted_stream(first: bytes | None,
                             agen: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        """Stream as received while accumulating chars to estimate usage."""
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
            break  # no un-cooled key left in the pool
        try:
            result = await provider.chat_completions(real_model, body, key, stream)
            # openai_like non-stream returns the upstream response directly; judge status here
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

            # streaming: fetch the first chunk, confirm upstream 2xx and producing, then respond to the client;
            # otherwise we can still fail over in the pool (response headers not yet sent)
            try:
                first = await result.__anext__()
            except StopAsyncIteration:
                first = None
            pool.report_success(provider_name, key)
            return _sse_response(_safe_stream(_chat_ensure_done(counted_stream(first, result)), _chat_error_tail()))
        except UpstreamError as exc:
            pool.report_failure(provider_name, key)
            last_msg = f"upstream error ({provider_name}): HTTP {exc.status_code} {exc.message[:200]}"
            logger.warning("key failover: %s", last_msg)
        except Exception as exc:  # network/parse errors, fail over too
            pool.report_failure(provider_name, key)
            last_msg = f"upstream request failed ({provider_name}): {exc!r}"[:300]
            logger.warning("key failover: %s", last_msg)

    # all keys failed: 502 in OpenAI error format
    await _record_stats(
        request, api_key=caller, model=model, provider=provider_name,
        latency_ms=int((time.monotonic() - t0) * 1000), status=502, stream=int(stream),
    )
    return _error(502, last_msg, "upstream_error", "upstream_error")


@router.get("/v1/models")
async def list_models(request: Request):
    """Return all configured models + aliases (OpenAI models list format)."""
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
            if isinstance(m, dict):
                mid = m.get("name"); ctx = m.get("context") or m.get("context_window")
            else:
                mid = m; ctx = None
            item = {"id": str(mid), "object": "model", "created": created, "owned_by": pname}
            if ctx:
                item["context_window"] = int(ctx)
            data.append(item)
    for mname in (cfg.raw.get("moa") or {}):
        data.append(
            {"id": f"moa:{mname}", "object": "model", "created": created, "owned_by": "moa"}
        )
    return {"object": "list", "data": data}


def _usage_triplet(data: dict) -> tuple[int, int, int]:
    """Extract (prompt, completion, total) from the response JSON; supports both chat and Responses usage key names."""
    usage = (data or {}).get("usage") or {}
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    total = usage.get("total_tokens") or (prompt + completion)
    return prompt, completion, total


@router.post("/v1/responses")
async def responses(request: Request):
    """OpenAI Responses API (unified hub).

    - Upstream natively supports Responses (supports_responses=True, e.g. Volcano Ark / agnes) -> passthrough
    - Other upstreams (anthropic / gemini / chat-only openai_like) -> responses->chat->native, 
      response converted back chat->responses, streaming likewise
    Auth / alias resolution / key-pool failover / rate-limit / stats are identical to chat_completions.
    Tool calls are normalized with invalid-JSON fallback; mid-stream errors get a valid synthesized terminator so the connection stays alive.
    """
    return await _dispatch_unified(request, wire="responses")


@router.post("/v1/messages")
async def messages(request: Request):
    """Anthropic Messages API input endpoint (unified hub).

    anthropic input -> chat -> any upstream -> chat -> anthropic output.
    Lets Anthropic-protocol-only clients aggregate to any provider.
    """
    return await _dispatch_unified(request, wire="anthropic")


async def _handle_moa(request: Request, wire: str, body: dict, model: str):
    """MOA request handler: normalize to chat -> run_moa -> convert back to the client wire format."""
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

    # streaming: aggregator's chat stream -> client wire
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
    """Unified dispatch for the responses / anthropic input faces.

    wire="responses"  : input/output OpenAI Responses
    wire="anthropic"  : input/output Anthropic Messages
    Internally everything is converted to chat and calls provider.chat_completions (reusing each vendor's chat<->native conversion).
    openai_like upstreams with supports_responses=True take the native passthrough fast path on the responses face.
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
        return _error(500, f"Provider `{provider_name}` failed to initialize", "server_error")

    # responses face and upstream natively supports -> passthrough fast path
    passthrough = wire == "responses" and provider.supports_responses and callable(getattr(provider, "responses", None))

    pool = request.app.state.pool
    caller = _caller_key(request)
    if not await _ratelimit_allow(request, f"{caller}:{model}"):
        return _error(429, "Rate limit exceeded", "rate_limit_exceeded", "rate_limit_exceeded")

    t0 = time.monotonic()
    attempts = max(1, pool.size(provider_name))
    last_msg = f"provider `{provider_name}` has no available upstream key"

    want_usage = wire == "responses"  # responses stream tries to include usage

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

            # conversion path: input -> chat
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

            # streaming convert
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
            last_msg = f"upstream error ({provider_name}): HTTP {exc.status_code} {exc.message[:200]}"
            logger.warning("key failover: %s", last_msg)
        except Exception as exc:
            pool.report_failure(provider_name, key)
            last_msg = f"upstream request failed ({provider_name}): {exc!r}"[:300]
            logger.warning("key failover: %s", last_msg)

    await _record_stats(request, api_key=caller, model=model, provider=provider_name,
        latency_ms=int((time.monotonic()-t0)*1000), status=502, stream=int(stream))
    return _error(502, last_msg, "upstream_error", "upstream_error")


def _sanitize_responses_output(data: dict) -> dict:
    """When passthrough responses output, repair invalid function_call arguments to avoid client parse crashes."""
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
    """On mid-stream upstream errors, synthesize a valid terminator to prevent the client session from breaking on a dropped stream."""
    try:
        async for chunk in agen:
            yield chunk
    except Exception as exc:
        logger.warning("mid-stream error, synthesizing terminator to keep connection alive: %r", exc)
        if error_tail:
            yield error_tail


async def _chat_ensure_done(agen: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """chat stream: if the upstream ends early without [DONE], append a finish chunk + [DONE]; 
    prevents the client from hanging or breaking the session on a missing terminator."""
    saw_done = False
    async for chunk in agen:
        if b"[DONE]" in chunk:
            saw_done = True
        yield chunk
    if not saw_done:
        yield _chat_error_tail()


def _sanitize_chat_output(data: dict) -> dict:
    """Normalize chat output tool calls: fill ids, repair invalid arguments."""
    from .protocol import normalize_tool_calls
    for ch in data.get("choices") or []:
        msg = (ch or {}).get("message") or {}
        if msg.get("tool_calls"):
            msg["tool_calls"] = normalize_tool_calls(msg["tool_calls"])
    return data


def _chat_error_tail() -> bytes:
    """Valid terminator for a broken chat stream: a finish chunk + [DONE]."""
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
    """Embeddings passthrough (OpenAI-compatible /v1/embeddings).

    Only openai_like providers support it; failover is the same as chat_completions.
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
        return _error(500, f"Provider `{provider_name}` failed to initialize", "server_error")
    embed_fn = getattr(provider, "embeddings", None)
    if not callable(embed_fn):
        return _error(
            501,
            f"Provider `{provider_name}` does not support embeddings",
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
    last_msg = f"provider `{provider_name}` has no available upstream key"

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
            last_msg = f"upstream error ({provider_name}): HTTP {exc.status_code} {exc.message[:200]}"
            logger.warning("embeddings failover: %s", last_msg)
        except Exception as exc:
            pool.report_failure(provider_name, key)
            last_msg = f"upstream request failed ({provider_name}): {exc!r}"[:300]
            logger.warning("embeddings failover: %s", last_msg)

    await _record_stats(
        request, api_key=caller, model=model, provider=provider_name,
        latency_ms=int((time.monotonic() - t0) * 1000), status=502, stream=0,
    )
    return _error(502, last_msg, "upstream_error", "upstream_error")
