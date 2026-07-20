"""Anthropic Messages API ↔ OpenAI 格式双向转换（含流式 SSE）。

- 请求：OpenAI chat.completions 格式 → Anthropic /v1/messages 格式
  （system 提取为顶层字段、max_tokens 必填默认 4096、tool_calls → tool_use、
  tool 消息 → tool_result）
- 响应：Anthropic messages 响应 / 流事件（message_start、content_block_start、
  content_block_delta、message_delta、message_stop）→ OpenAI chunk 格式
- 鉴权头：x-api-key + anthropic-version
"""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator

import httpx

from . import ProviderBase, UpstreamError, get_client, register

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096  # Anthropic 要求 max_tokens 必填

# Anthropic stop_reason → OpenAI finish_reason
_STOP_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "stop",
}


# ================================================================== 请求转换
def _content_to_text(content) -> str:
    """OpenAI content（str 或 parts 列表）→ 纯文本。"""
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


def _user_parts_to_blocks(content) -> list[dict]:
    """OpenAI 用户消息 content parts → Anthropic content blocks（文本/图片）。"""
    if isinstance(content, str) or content is None:
        return [{"type": "text", "text": content or ""}]
    blocks: list[dict] = []
    for p in content:
        if not isinstance(p, dict):
            continue
        ptype = p.get("type")
        if ptype == "text":
            blocks.append({"type": "text", "text": str(p.get("text", ""))})
        elif ptype == "image_url":
            url = (p.get("image_url") or {}).get("url") or ""
            if url.startswith("data:") and ";base64," in url:
                # data:image/png;base64,... → base64 源
                media_type, data = url[5:].split(";base64,", 1)
                blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": data},
                    }
                )
            elif url:
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        # 其他类型（音频等）忽略，不报错
    return blocks or [{"type": "text", "text": ""}]


def _assistant_blocks(msg: dict) -> list[dict]:
    """assistant 消息（可能带 tool_calls）→ Anthropic content blocks。"""
    blocks: list[dict] = []
    text = _content_to_text(msg.get("content"))
    if text:
        blocks.append({"type": "text", "text": text})
    for tc in msg.get("tool_calls") or []:
        fn = (tc or {}).get("function") or {}
        args = fn.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}  # 非法 JSON 兜底，不报错
        blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:12]}",
                "name": fn.get("name", ""),
                "input": args if isinstance(args, dict) else {},
            }
        )
    return blocks or [{"type": "text", "text": ""}]


def _to_blocks(content) -> list[dict]:
    if isinstance(content, list):
        return list(content)
    if content in (None, ""):
        return []
    return [{"type": "text", "text": str(content)}]


def _merge_messages(messages: list[dict]) -> list[dict]:
    """合并连续同角色消息（Anthropic 要求 user/assistant 交替）。"""
    merged: list[dict] = []
    for m in messages:
        if merged and merged[-1]["role"] == m["role"]:
            blocks = _to_blocks(merged[-1]["content"]) + _to_blocks(m["content"])
            if blocks and all(b.get("type") == "text" for b in blocks):
                merged[-1]["content"] = "\n".join(b.get("text", "") for b in blocks)
            else:
                merged[-1]["content"] = blocks
        else:
            merged.append(dict(m))
    return merged


def _convert_tools(body: dict) -> list[dict]:
    """OpenAI tools → Anthropic tools；未知结构原样透传，不报错。"""
    tools = []
    for t in body.get("tools") or []:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function":
            fn = t.get("function") or {}
            tools.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters")
                    or {"type": "object", "properties": {}},
                }
            )
        elif t.get("name"):
            tools.append(t)  # 已是 Anthropic 格式
    return tools


def _convert_tool_choice(tc):
    """OpenAI tool_choice → Anthropic tool_choice。"""
    if tc is None:
        return None
    if isinstance(tc, str):
        return {
            "auto": {"type": "auto"},
            "none": {"type": "none"},
            "required": {"type": "any"},
        }.get(tc, {"type": "auto"})
    if isinstance(tc, dict):
        if tc.get("type") == "function":
            return {"type": "tool", "name": (tc.get("function") or {}).get("name", "")}
        return tc  # 假定已是 Anthropic 格式
    return None


def convert_request(body: dict) -> dict:
    """OpenAI chat.completions 请求体 → Anthropic Messages 请求体（model/stream 由调用方补）。"""
    system_parts: list[str] = []
    messages: list[dict] = []
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            # system 提取为顶层字段
            text = _content_to_text(msg.get("content"))
            if text:
                system_parts.append(text)
        elif role == "assistant":
            messages.append({"role": "assistant", "content": _assistant_blocks(msg)})
        elif role == "tool":
            # OpenAI tool 结果 → Anthropic user 消息里的 tool_result
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": _content_to_text(msg.get("content")),
                        }
                    ],
                }
            )
        elif role == "user":
            messages.append(
                {"role": "user", "content": _user_parts_to_blocks(msg.get("content"))}
            )
        # 其他角色忽略，不报错

    payload: dict = {
        "messages": _merge_messages(messages),
        # max_tokens 必填：取 max_tokens / max_completion_tokens，默认 4096
        "max_tokens": body.get("max_tokens")
        or body.get("max_completion_tokens")
        or DEFAULT_MAX_TOKENS,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")):
        if body.get(src) is not None:
            payload[dst] = body[src]
    stop = body.get("stop")
    if stop:
        payload["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    tools = _convert_tools(body)
    if tools:
        payload["tools"] = tools
        choice = _convert_tool_choice(body.get("tool_choice"))
        if choice:
            payload["tool_choice"] = choice
    return payload


# ================================================================== 响应转换
def convert_response(data: dict, model: str) -> dict:
    """Anthropic Messages 响应 JSON → OpenAI chat.completion JSON。"""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in data.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(
                            block.get("input") or {}, ensure_ascii=False
                        ),
                    },
                }
            )
    message: dict = {"role": "assistant", "content": "".join(text_parts)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage_in = data.get("usage") or {}
    prompt_tokens = usage_in.get("input_tokens", 0) or 0
    completion_tokens = usage_in.get("output_tokens", 0) or 0
    return {
        "id": data.get("id") or f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _STOP_MAP.get(data.get("stop_reason"), "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
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


def _usage_chunk(chat_id: str, created: int, model: str,
                 prompt_tokens: int, completion_tokens: int) -> bytes:
    obj = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")


async def convert_stream(lines: AsyncIterator[str], model: str,
                         want_usage: bool = False) -> AsyncIterator[bytes]:
    """把 Anthropic SSE 行流转为 OpenAI chunk 字节流（逐事件产出，不整段缓冲）。

    lines: 上游 aiter_lines() 的行迭代器。
    want_usage: 客户端要求 stream_options.include_usage 时，结束前追加 usage chunk。
    """
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    role_sent = False
    finish_sent = False
    tool_index = -1
    input_tokens = 0
    output_tokens = 0

    async for line in lines:
        if not line.startswith("data:"):
            continue  # event:/ping/空行等忽略
        data_str = line[5:].strip()
        if not data_str:
            continue
        try:
            evt = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        etype = evt.get("type")

        if etype == "message_start":
            msg = evt.get("message") or {}
            chat_id = msg.get("id") or chat_id
            usage = msg.get("usage") or {}
            input_tokens = usage.get("input_tokens", input_tokens) or 0
            output_tokens = usage.get("output_tokens", output_tokens) or 0
            if not role_sent:
                role_sent = True
                yield _chunk(chat_id, created, model, {"role": "assistant"})
        elif etype == "content_block_start":
            block = evt.get("content_block") or {}
            if block.get("type") == "tool_use":
                tool_index += 1
                yield _chunk(
                    chat_id, created, model,
                    {
                        "tool_calls": [
                            {
                                "index": tool_index,
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": "",
                                },
                            }
                        ]
                    },
                )
            # text 块开始无需产出
        elif etype == "content_block_delta":
            delta = evt.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                yield _chunk(chat_id, created, model, {"content": delta.get("text", "")})
            elif dtype == "input_json_delta" and tool_index >= 0:
                yield _chunk(
                    chat_id, created, model,
                    {
                        "tool_calls": [
                            {
                                "index": tool_index,
                                "function": {
                                    "arguments": delta.get("partial_json", "")
                                },
                            }
                        ]
                    },
                )
            # thinking_delta 等忽略
        elif etype == "message_delta":
            delta = evt.get("delta") or {}
            usage = evt.get("usage") or {}
            output_tokens = usage.get("output_tokens", output_tokens) or 0
            stop = delta.get("stop_reason")
            if stop and not finish_sent:
                finish_sent = True
                yield _chunk(chat_id, created, model, {}, _STOP_MAP.get(stop, "stop"))
        elif etype == "message_stop":
            break
        elif etype == "error":
            # 流中途上游报错：无法故障转移，结束流
            break
        # ping 等事件忽略

    if role_sent and not finish_sent:
        # 异常收尾的保底 finish，避免客户端挂起
        yield _chunk(chat_id, created, model, {}, "stop")
    if want_usage and (input_tokens or output_tokens):
        yield _usage_chunk(chat_id, created, model, input_tokens, output_tokens)
    yield b"data: [DONE]\n\n"


# ================================================================== Provider
@register("anthropic")
class AnthropicProvider(ProviderBase):
    """Anthropic Messages API provider。"""

    API_PATH = "/v1/messages"

    async def chat_completions(
        self, model: str, body: dict, api_key: str, stream: bool
    ) -> "httpx.Response | AsyncIterator[bytes]":
        url = f"{self.base_url}{self.API_PATH}"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        payload = convert_request(body)
        payload["model"] = model
        payload["stream"] = bool(stream)

        if not stream:
            resp = await get_client().post(
                url, json=payload, headers=headers, timeout=self.timeout
            )
            if resp.status_code >= 400:
                raise UpstreamError(resp.status_code, resp.text[:500])
            # 转换为 OpenAI 格式后构造响应返回
            return httpx.Response(
                200,
                json=convert_response(resp.json(), model),
                headers={"content-type": "application/json"},
            )

        want_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        return self._stream(url, payload, headers, model, want_usage)

    async def _stream(self, url: str, payload: dict, headers: dict, model: str,
                      want_usage: bool) -> AsyncIterator[bytes]:
        client = get_client()
        async with client.stream(
            "POST", url, json=payload, headers=headers, timeout=self.timeout
        ) as resp:
            if resp.status_code >= 400:
                detail = (await resp.aread())[:500].decode("utf-8", "replace")
                raise UpstreamError(resp.status_code, detail)
            async for chunk in convert_stream(resp.aiter_lines(), model, want_usage):
                yield chunk
