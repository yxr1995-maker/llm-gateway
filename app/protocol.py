"""Protocol conversion core: chat.completions is the internal execution layer; Responses / Anthropic Messages
act as input/output faces.

Conversion matrix (all include non-stream + stream):
  responses <-> chat        （lets /v1/responses reach any upstream)
  anthropic <-> chat        （lets /v1/messages reach any upstream; reuses the providers'
                              chat->anthropic; this file adds the anthropic->chat direction)

Tool robustness:
  - normalize_tool_calls: fill ids, ensure arguments is valid JSON
  - repair_arguments: invalid JSON falls back to {"_raw": ...} to avoid client parse crashes
"""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator, Iterable

# ============================================================ tool robustness
def repair_arguments(args) -> str:
    """Ensure tool_call.arguments is a valid JSON string."""
    if args is None:
        return "{}"
    if isinstance(args, (dict, list)):
        return json.dumps(args, ensure_ascii=False)
    s = str(args)
    try:
        json.loads(s)
        return s
    except (json.JSONDecodeError, TypeError):
        # invalid JSON: wrap as _raw so the client gets a parseable object instead of crashing on JSON parse
        return json.dumps({"_raw": s}, ensure_ascii=False)


def normalize_tool_calls(tool_calls: list | None) -> list[dict]:
    """Normalize OpenAI tool_calls: fill ids, repair arguments."""
    out: list[dict] = []
    for i, tc in enumerate(tool_calls or []):
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        out.append({
            "id": tc.get("id") or f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": fn.get("name", ""),
                "arguments": repair_arguments(fn.get("arguments")),
            },
        })
    return out


# ============================================================ responses -> chat
def responses_req_to_chat(body: dict) -> dict:
    """OpenAI Responses request -> chat.completions request."""
    out: dict = {"model": body.get("model")}
    messages: list[dict] = []

    instr = body.get("instructions")
    if isinstance(instr, str) and instr.strip():
        messages.append({"role": "system", "content": instr})

    inp = body.get("input")
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            role = item.get("role")
            if itype == "function_call_output" or (role == "tool"):
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or item.get("tool_call_id") or "",
                    "content": item.get("output") or item.get("content") or "",
                })
            elif itype == "function_call":
                messages.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": repair_arguments(item.get("arguments")),
                        },
                    }],
                })
            elif role in ("user", "assistant", "system"):
                content = _parts_to_text(item.get("content"), role == "assistant")
                msg = {"role": role, "content": content}
                # assistant messages may embed function_call
                calls = [c for c in (item.get("content") or [])
                         if isinstance(c, dict) and c.get("type") == "function_call"]
                if role == "assistant" and calls:
                    msg["tool_calls"] = [{
                        "id": c.get("call_id") or c.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {"name": c.get("name", ""),
                                     "arguments": repair_arguments(c.get("arguments"))},
                    } for c in calls]
                    if not content:
                        msg.pop("content", None)
                messages.append(msg)

    out["messages"] = messages

    if body.get("max_output_tokens") is not None:
        out["max_tokens"] = body["max_output_tokens"]
    for k in ("temperature", "top_p", "stream", "tool_choice", "seed"):
        if k in body:
            out[k] = body[k]
    if body.get("tools"):
        out["tools"] = [{
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("parameters") or {},
            },
        } for t in body["tools"] if isinstance(t, dict)]
    return out


def _parts_to_text(content, assistant_text_only: bool = False) -> str:
    """Responses content parts -> plain text (function_call ignored; handled separately by the caller)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if not isinstance(p, dict):
            continue
        ptype = p.get("type")
        if ptype in ("input_text", "output_text", "text"):
            parts.append(str(p.get("text", "")))
        elif ptype == "input_image":
            # images can't be carried on the text channel; skipped (upper layers may take the native path)
            continue
    return "".join(parts)


# ============================================================ chat -> responses
def chat_resp_to_responses(data: dict, model: str) -> dict:
    """chat.completion response -> Responses response."""
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content")
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    tool_calls = normalize_tool_calls(msg.get("tool_calls"))

    output: list[dict] = []
    if text:
        output.append({
            "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text}],
        })
    for tc in tool_calls:
        output.append({
            "type": "function_call",
            "id": tc["id"], "call_id": tc["id"],
            "name": tc["function"]["name"],
            "arguments": tc["function"]["arguments"],
            "status": "completed",
        })

    usage = data.get("usage") or {}
    return {
        "id": data.get("id") or f"resp_{uuid.uuid4().hex[:24]}",
        "object": "response",
        "created_at": data.get("created") or int(time.time()),
        "status": "completed",
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0) or 0,
            "output_tokens": usage.get("completion_tokens", 0) or 0,
            "total_tokens": usage.get("total_tokens", 0) or 0,
        },
    }


async def chat_stream_to_responses(agen: AsyncIterator[bytes], model: str,
                                   want_usage: bool) -> AsyncIterator[bytes]:
    """chat SSE stream -> Responses SSE event stream."""
    rid = f"resp_{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def ev(etype: str, payload: dict) -> bytes:
        payload = {"type": etype, **payload}
        return ("event: " + etype + "\ndata: " +
                json.dumps(payload, ensure_ascii=False) + "\n\n").encode()

    base_resp = {"id": rid, "object": "response", "status": "in_progress",
                 "model": model, "created_at": created, "output": []}
    yield ev("response.created", {"response": base_resp})

    msg_added = False
    text_started = False
    tool_buffers: dict[int, dict] = {}  # index -> {id,name,args,started}
    prompt_tokens = completion_tokens = total_tokens = 0
    finish_reason = None

    async for raw in agen:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", prompt_tokens) or 0
            completion_tokens = usage.get("completion_tokens", completion_tokens) or 0
            total_tokens = usage.get("total_tokens", total_tokens) or 0
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            # text
            content = delta.get("content")
            if content:
                if not msg_added:
                    msg_added = True
                    yield ev("response.output_item.added",
                             {"output_index": 0,
                              "item": {"type": "message", "role": "assistant",
                                       "status": "in_progress", "content": []}})
                    yield ev("response.content_part.added",
                             {"output_index": 0, "content_index": 0,
                              "part": {"type": "output_text", "text": ""}})
                    text_started = True
                yield ev("response.output_text.delta",
                         {"output_index": 0, "content_index": 0, "delta": content})
            # tool calls
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                buf = tool_buffers.setdefault(idx, {
                    "id": tc.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                    "name": (tc.get("function") or {}).get("name", ""),
                    "args": "", "started": False,
                })
                if tc.get("id"):
                    buf["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    buf["name"] = fn["name"]
                if not buf["started"]:
                    buf["started"] = True
                    out_idx = 1 + idx
                    yield ev("response.output_item.added",
                             {"output_index": out_idx,
                              "item": {"type": "function_call", "status": "in_progress",
                                       "call_id": buf["id"], "id": buf["id"],
                                       "name": buf["name"], "arguments": ""}})
                if fn.get("arguments"):
                    buf["args"] += fn["arguments"]
                    yield ev("response.function_call_arguments.delta",
                             {"output_index": 1 + idx, "delta": fn["arguments"]})
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    # finalize
    if text_started:
        yield ev("response.output_text.done", {"output_index": 0, "content_index": 0, "text": ""})
        yield ev("response.content_part.done", {"output_index": 0, "content_index": 0,
                                                "part": {"type": "output_text", "text": ""}})
        yield ev("response.output_item.done", {"output_index": 0,
                                               "item": {"type": "message", "role": "assistant",
                                                        "status": "completed", "content": []}})
    for idx, buf in tool_buffers.items():
        yield ev("response.function_call_arguments.done",
                 {"output_index": 1 + idx, "arguments": buf["args"]})
        yield ev("response.output_item.done",
                 {"output_index": 1 + idx,
                  "item": {"type": "function_call", "status": "completed",
                           "call_id": buf["id"], "id": buf["id"],
                           "name": buf["name"], "arguments": buf["args"]}})

    final_resp = dict(base_resp)
    final_resp["status"] = "completed"
    final_resp["output"] = []
    if want_usage:
        final_resp["usage"] = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens or (prompt_tokens + completion_tokens),
        }
    yield ev("response.completed", {"response": final_resp})


# ============================================================ anthropic -> chat
def anthropic_req_to_chat(body: dict) -> dict:
    """Anthropic Messages request -> chat.completions request."""
    out: dict = {"model": body.get("model")}
    messages: list[dict] = []
    sys = body.get("system")
    if isinstance(sys, str) and sys.strip():
        messages.append({"role": "system", "content": sys})
    elif isinstance(sys, list):
        text = "".join(b.get("text", "") for b in sys
                       if isinstance(b, dict) and b.get("type") == "text")
        if text.strip():
            messages.append({"role": "system", "content": text})

    for m in body.get("messages") or []:
        role = m.get("role")
        content = m.get("content")
        if role == "user":
            messages.append({"role": "user", "content": _anthropic_content_to_text(content)})
        elif role == "assistant":
            text, tool_calls = _anthropic_assistant_parts(content)
            msg: dict = {"role": "assistant"}
            if text:
                msg["content"] = text
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
    out["messages"] = messages

    if body.get("max_tokens") is not None:
        out["max_tokens"] = body["max_tokens"]
    for k in ("temperature", "top_p", "stream", "tool_choice", "stop"):
        if k in body:
            out[k] = body[k]
    if body.get("tools"):
        out["tools"] = [{
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {},
            },
        } for t in body["tools"] if isinstance(t, dict)]
    return out


def _anthropic_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif isinstance(b, dict) and b.get("type") == "tool_result":
            parts.append(_anthropic_content_to_text(b.get("content")))
    return "".join(parts)


def _anthropic_assistant_parts(content):
    text_parts, tool_calls = [], []
    for b in content or []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            text_parts.append(b.get("text", ""))
        elif b.get("type") == "tool_use":
            tool_calls.append({
                "id": b.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": b.get("name", ""),
                    "arguments": repair_arguments(b.get("input")),
                },
            })
    # tool_result appears in a user turn; not handled here
    return "".join(text_parts), tool_calls


# ============================================================ chat -> anthropic
def chat_resp_to_anthropic(data: dict, model: str) -> dict:
    """chat.completion response -> Anthropic Messages response."""
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content")
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    blocks: list[dict] = []
    if text:
        blocks.append({"type": "text", "text": text})
    for tc in normalize_tool_calls(msg.get("tool_calls")):
        try:
            inp = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, TypeError):
            inp = {"_raw": tc["function"]["arguments"]}
        blocks.append({"type": "tool_use", "id": tc["id"],
                       "name": tc["function"]["name"], "input": inp})

    finish = choice.get("finish_reason")
    stop = {"stop": "end_turn", "length": "max_tokens",
            "tool_calls": "tool_use", "content_filter": "end_turn"}.get(finish, "end_turn")
    usage = data.get("usage") or {}
    return {
        "id": data.get("id") or f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message", "role": "assistant", "model": model,
        "content": blocks or [{"type": "text", "text": ""}],
        "stop_reason": stop, "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0) or 0,
            "output_tokens": usage.get("completion_tokens", 0) or 0,
        },
    }


async def chat_stream_to_anthropic(agen: AsyncIterator[bytes], model: str) -> AsyncIterator[bytes]:
    """chat SSE stream -> Anthropic Messages SSE event stream."""
    mid = f"msg_{uuid.uuid4().hex[:24]}"

    def ev(etype: str, payload: dict) -> bytes:
        return ("event: " + etype + "\ndata: " +
                json.dumps(payload, ensure_ascii=False) + "\n\n").encode()

    yield ev("message_start", {"type": "message_start", "message": {
        "id": mid, "type": "message", "role": "assistant", "model": model,
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0}}})
    yield ev("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}})

    block_idx = 0
    tool_buffers: dict[int, dict] = {}
    input_tokens = output_tokens = 0
    stop_reason = "end_turn"

    async for raw in agen:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            input_tokens = usage.get("prompt_tokens", input_tokens) or 0
            output_tokens = usage.get("completion_tokens", output_tokens) or 0
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                yield ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                 "delta": {"type": "text_delta",
                                                           "text": delta["content"]}})
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                buf = tool_buffers.setdefault(idx, {
                    "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": (tc.get("function") or {}).get("name", ""),
                    "args": "", "block_index": None,
                })
                if tc.get("id"):
                    buf["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    buf["name"] = fn["name"]
                if buf["block_index"] is None:
                    # close the current text block, open a tool_use block
                    yield ev("content_block_stop", {"type": "content_block_stop", "index": block_idx})
                    block_idx += 1
                    buf["block_index"] = block_idx
                    yield ev("content_block_start", {
                        "type": "content_block_start", "index": block_idx,
                        "content_block": {"type": "tool_use", "id": buf["id"],
                                          "name": buf["name"], "input": {}}})
                if fn.get("arguments"):
                    buf["args"] += fn["arguments"]
                    yield ev("content_block_delta", {
                        "type": "content_block_delta", "index": block_idx,
                        "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]}})
            if choice.get("finish_reason"):
                fr = choice["finish_reason"]
                stop_reason = {"stop": "end_turn", "length": "max_tokens",
                               "tool_calls": "tool_use"}.get(fr, "end_turn")

    yield ev("content_block_stop", {"type": "content_block_stop", "index": block_idx})
    for buf in tool_buffers.values():
        yield ev("content_block_stop", {"type": "content_block_stop", "index": buf["block_index"]})
    yield ev("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                               "usage": {"output_tokens": output_tokens}})
    yield ev("message_stop", {"type": "message_stop"})


