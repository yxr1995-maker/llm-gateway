"""Google Gemini API ↔ OpenAI 格式双向转换（含流式 SSE）。

- 端点：:generateContent（非流式）/ :streamGenerateContent?alt=sse（流式）
- 鉴权：API key 走 query 参数 ?key=
- 请求：OpenAI messages → contents（user/model 角色）+ systemInstruction
- 响应：candidates → OpenAI choices / chunk
"""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator
from urllib.parse import quote

import httpx

from . import ProviderBase, UpstreamError, get_client, register

# Gemini finishReason → OpenAI finish_reason
_FINISH_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "BLOCKLIST": "content_filter",
    "SPII": "content_filter",
}


def _map_finish(reason) -> str:
    return _FINISH_MAP.get(reason or "", "stop")


# ================================================================== 请求转换
def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(str(p.get("text", "")))
        elif isinstance(p, str):
            parts.append(p)
    return "\n".join(parts)


def _user_parts(content) -> list[dict]:
    """OpenAI 用户 content → Gemini parts（文本 / inline_data 图片）。"""
    if isinstance(content, str) or content is None:
        return [{"text": content or ""}]
    parts: list[dict] = []
    for p in content:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text":
            parts.append({"text": str(p.get("text", ""))})
        elif p.get("type") == "image_url":
            url = (p.get("image_url") or {}).get("url") or ""
            if url.startswith("data:") and ";base64," in url:
                mime, data = url[5:].split(";base64,", 1)
                parts.append({"inline_data": {"mime_type": mime, "data": data}})
            # 远程 URL 图片忽略（Gemini 需 file_data，此处不展开），不报错
    return parts or [{"text": ""}]


def _assistant_parts(msg: dict) -> list[dict]:
    """assistant 消息 → Gemini parts（文本 + functionCall）。"""
    parts: list[dict] = []
    text = _content_to_text(msg.get("content"))
    if text:
        parts.append({"text": text})
    for tc in msg.get("tool_calls") or []:
        fn = (tc or {}).get("function") or {}
        args = fn.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        parts.append(
            {"functionCall": {"name": fn.get("name", ""), "args": args if isinstance(args, dict) else {}}}
        )
    return parts or [{"text": ""}]


def _convert_tools(body: dict) -> list[dict]:
    """OpenAI tools → Gemini functionDeclarations。"""
    decls = []
    for t in body.get("tools") or []:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function":
            fn = t.get("function") or {}
            decl = {"name": fn.get("name", ""), "description": fn.get("description", "")}
            if fn.get("parameters"):
                decl["parameters"] = fn["parameters"]
            decls.append(decl)
        elif t.get("name"):
            decls.append(t)  # 已接近 Gemini 格式则透传
    return [{"functionDeclarations": decls}] if decls else []


def convert_request(body: dict) -> dict:
    """OpenAI chat.completions 请求体 → Gemini generateContent 请求体。"""
    contents: list[dict] = []
    system_parts: list[str] = []
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            text = _content_to_text(msg.get("content"))
            if text:
                system_parts.append(text)
        elif role == "user":
            contents.append({"role": "user", "parts": _user_parts(msg.get("content"))})
        elif role == "assistant":
            contents.append({"role": "model", "parts": _assistant_parts(msg)})
        elif role == "tool":
            # OpenAI tool 结果 → Gemini functionResponse（放在 user 回合）
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": msg.get("name") or "tool",
                                "response": {"result": _content_to_text(msg.get("content"))},
                            }
                        }
                    ],
                }
            )
        # 其他角色忽略，不报错

    payload: dict = {"contents": contents}
    if system_parts:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

    generation: dict = {}
    if body.get("temperature") is not None:
        generation["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        generation["topP"] = body["top_p"]
    if body.get("top_k") is not None:
        generation["topK"] = body["top_k"]
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
    if max_tokens:
        generation["maxOutputTokens"] = max_tokens
    stop = body.get("stop")
    if stop:
        generation["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)
    if generation:
        payload["generationConfig"] = generation

    tools = _convert_tools(body)
    if tools:
        payload["tools"] = tools
    return payload


# ================================================================== 响应转换
def _candidate_to_message_and_finish(candidate: dict) -> tuple[dict, str]:
    """candidate → (OpenAI message, finish_reason)。"""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    content = candidate.get("content") or {}
    for part in content.get("parts") or []:
        if not isinstance(part, dict):
            continue
        if "text" in part:
            text_parts.append(str(part.get("text") or ""))
        elif "functionCall" in part:
            fc = part.get("functionCall") or {}
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args") or {}, ensure_ascii=False),
                    },
                }
            )
    message: dict = {"role": "assistant", "content": "".join(text_parts)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    finish = _map_finish(candidate.get("finishReason"))
    if tool_calls and candidate.get("finishReason") in (None, "STOP"):
        finish = "tool_calls"
    return message, finish


def convert_response(data: dict, model: str) -> dict:
    """Gemini generateContent 响应 JSON → OpenAI chat.completion JSON。"""
    candidates = data.get("candidates") or []
    if candidates:
        message, finish = _candidate_to_message_and_finish(candidates[0])
    else:
        # 无候选（如 prompt 被拦截）
        block = (data.get("promptFeedback") or {}).get("blockReason")
        message = {"role": "assistant", "content": ""}
        finish = "content_filter" if block else "stop"

    meta = data.get("usageMetadata") or {}
    prompt_tokens = meta.get("promptTokenCount", 0) or 0
    completion_tokens = meta.get("candidatesTokenCount", 0) or 0
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": meta.get("totalTokenCount", prompt_tokens + completion_tokens) or 0,
        },
    }


# ================================================================== 流式转换
def _chunk(chat_id: str, created: int, model: str, delta: dict | None = None,
           finish: str | None = None) -> bytes:
    obj = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
    }
    return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")


async def convert_stream(lines: AsyncIterator[str], model: str,
                         want_usage: bool = False) -> AsyncIterator[bytes]:
    """Gemini streamGenerateContent?alt=sse 行流 → OpenAI chunk 字节流。"""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    role_sent = False
    finish_sent = False
    tool_index = -1
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    async for line in lines:
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if not data_str:
            continue
        try:
            evt = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        if not role_sent:
            role_sent = True
            yield _chunk(chat_id, created, model, {"role": "assistant"})

        meta = evt.get("usageMetadata") or {}
        prompt_tokens = meta.get("promptTokenCount", prompt_tokens) or 0
        completion_tokens = meta.get("candidatesTokenCount", completion_tokens) or 0
        total_tokens = meta.get("totalTokenCount", total_tokens) or 0

        candidates = evt.get("candidates") or []
        if not candidates:
            continue
        cand = candidates[0] or {}
        content = cand.get("content") or {}
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            if "text" in part:
                text = str(part.get("text") or "")
                if text:
                    yield _chunk(chat_id, created, model, {"content": text})
            elif "functionCall" in part:
                fc = part.get("functionCall") or {}
                tool_index += 1
                yield _chunk(
                    chat_id, created, model,
                    {
                        "tool_calls": [
                            {
                                "index": tool_index,
                                "id": f"call_{uuid.uuid4().hex[:24]}",
                                "type": "function",
                                "function": {
                                    "name": fc.get("name", ""),
                                    "arguments": json.dumps(
                                        fc.get("args") or {}, ensure_ascii=False
                                    ),
                                },
                            }
                        ]
                    },
                )
        if cand.get("finishReason") and not finish_sent:
            finish_sent = True
            yield _chunk(chat_id, created, model, {}, _map_finish(cand.get("finishReason")))

    if role_sent and not finish_sent:
        yield _chunk(chat_id, created, model, {}, "stop")
    if want_usage and (prompt_tokens or completion_tokens or total_tokens):
        obj = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens or prompt_tokens + completion_tokens,
            },
        }
        yield ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")
    yield b"data: [DONE]\n\n"


# ================================================================== Provider
@register("gemini")
class GeminiProvider(ProviderBase):
    """Google Gemini provider（key 走 query 参数）。"""

    def _url(self, model: str, stream: bool, api_key: str) -> str:
        action = "streamGenerateContent" if stream else "generateContent"
        base = self.base_url
        # 兼容 base_url 已带 /v1beta（或 /v1）后缀的写法，避免重复拼接
        if not base.endswith(("/v1beta", "/v1")):
            base = f"{base}/v1beta"
        url = f"{base}/models/{quote(model, safe='')}:{action}"
        # 流式必须 alt=sse；key 一律走 query 参数
        return f"{url}?{'alt=sse&' if stream else ''}key={api_key}"

    async def chat_completions(
        self, model: str, body: dict, api_key: str, stream: bool
    ) -> "httpx.Response | AsyncIterator[bytes]":
        payload = convert_request(body)
        headers = {"content-type": "application/json"}

        if not stream:
            resp = await get_client().post(
                self._url(model, False, api_key),
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            if resp.status_code >= 400:
                raise UpstreamError(resp.status_code, resp.text[:500])
            return httpx.Response(
                200,
                json=convert_response(resp.json(), model),
                headers={"content-type": "application/json"},
            )

        want_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        return self._stream(model, payload, headers, api_key, want_usage)

    async def _stream(self, model: str, payload: dict, headers: dict, api_key: str,
                      want_usage: bool) -> AsyncIterator[bytes]:
        client = get_client()
        async with client.stream(
            "POST",
            self._url(model, True, api_key),
            json=payload,
            headers=headers,
            timeout=self.timeout,
        ) as resp:
            if resp.status_code >= 400:
                detail = (await resp.aread())[:500].decode("utf-8", "replace")
                raise UpstreamError(resp.status_code, detail)
            async for chunk in convert_stream(resp.aiter_lines(), model, want_usage):
                yield chunk
