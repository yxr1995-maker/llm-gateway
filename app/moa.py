"""Mixture-of-Agents / multimodal hybrid pipeline.

Two composable patterns per pipeline (model trigger: `moa:<name>` / `moa/<name>`):

1. Parallel MOA (text synthesis): `proposers` answer in parallel, `aggregator`
   synthesizes. Each agent may set `reasoning_effort` or `default` to inherit the
   pipeline `default_reasoning_effort`.

2. Staged multimodal pipeline: `stages` run in order; each stage declares a
   `modality` (text / vision / image / video) and its output feeds the next stage.
   Consecutive stages with `parallel: true` fan out concurrently and their text
   outputs are merged. This lets users mix vision, image-gen and video-gen models
   into a hybrid LLM for complex workflows.

TPS-oriented scheduling:
- `max_concurrency` semaphore bounds in-flight upstream calls (over-subscription
  throttles each stream and tanks per-stream TPS).
- Parallel stages / proposers run via asyncio.gather (lower wall-clock latency).
- The final text/vision stage is streamed to the client (first token sooner,
  higher perceived TPS); earlier stages complete before it starts.
- Per-stage `timeout`; a failed `optional` stage is skipped (artifact becomes a
  note) instead of aborting the whole pipeline.
- Shared httpx connection pool is reused across all calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import AsyncIterator

import httpx

from . import cache as _cache
from .providers import UpstreamError, parse_retry_after

logger = logging.getLogger("llm-gateway.moa")

DEFAULT_AGG_PROMPT = (
    "You are the lead AI assistant. Several assistant models have answered the same "
    "user request. Synthesize their answers: correct errors, remove redundancy, and "
    "produce a single high-quality final answer. Keep tool calls only if clearly "
    "necessary; otherwise respond to the user directly."
)


def moa_pipeline_name(model: str) -> str | None:
    if model.startswith("moa:"):
        return model[4:]
    if model.startswith("moa/"):
        return model[4:]
    return None


def is_moa(model: str) -> bool:
    return moa_pipeline_name(model) is not None


def _msg_text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return str(c or "")


def _extract_text(chat_data: dict) -> str:
    ch = (chat_data.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return _msg_text(msg)


def _effective_effort(agent: dict, default: str | None) -> str | None:
    e = agent.get("reasoning_effort")
    if not e or e == "default":
        e = default
    return e if e and e != "none" else None


def _jaccard(a: str, b: str) -> float:
    sa = set((a or "").lower().split()); sb = set((b or "").lower().split())
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def _dedup(outs: list[str], threshold: float = 0.85) -> list[str]:
    """Drop near-duplicate proposer outputs (keep first of each cluster)."""
    kept: list[str] = []
    for o in outs:
        if not any(_jaccard(o, k) >= threshold for k in kept):
            kept.append(o)
    return kept


async def _compress(text: str, pipe: dict, providers, pool, sem) -> str:
    """Compress a verbose intermediate result before feeding the aggregator/planner.
    Truncates to max_chars, or summarizes with a cheap model if configured."""
    cfg = pipe.get("compress") or {}
    if not cfg.get("enabled") or not text:
        return text
    max_chars = int(cfg.get("max_chars", 600))
    if len(text) <= max_chars:
        return text
    summ = cfg.get("summarizer")
    if summ and summ.get("provider"):
        prov = providers.get(summ["provider"])
        if prov is not None:
            try:
                cb = {"model": summ["model"], "messages": [
                    {"role": "system", "content": "Summarize the following; keep key facts; be concise."},
                    {"role": "user", "content": text}], "stream": False}
                res = await _call_chat(prov, summ["provider"], summ["model"], cb, pool, False, sem)
                d = res.json() if isinstance(res, httpx.Response) else res
                out = _extract_text(d)
                if out:
                    return out
            except Exception:
                pass
    return text[:max_chars] + " …"


async def _with_sem(sem, coro_fn):
    async with sem:
        return await coro_fn()


async def _call_chat(provider, provider_name, model, chat_body, pool, stream, sem=None):
    attempts = max(1, pool.size(provider_name))
    last: Exception | None = None
    cache_key = _cache.make_key(provider_name, model, chat_body) if (not stream and _cache.ENABLED) else None
    if cache_key is not None:
        hit = await _cache.get(cache_key)
        if hit is not None:
            return httpx.Response(200, json=hit, headers={"content-type": "application/json", "x-cache": "HIT"})
    for _ in range(attempts):
        key = pool.acquire(provider_name)
        if key is None:
            break
        try:
            coro = lambda: provider.chat_completions(model, chat_body, key, stream)
            result = await (coro() if sem is None else _with_sem(sem, coro))
            if isinstance(result, httpx.Response) and result.status_code >= 400:
                raise UpstreamError(result.status_code, result.text[:300], retry_after=parse_retry_after(result.headers))
            pool.report_success(provider_name, key)
            if cache_key is not None and isinstance(result, httpx.Response) and result.status_code < 400:
                try:
                    await _cache.set(cache_key, result.json())
                except Exception:
                    pass
            return result
        except UpstreamError as exc:
            outcome = pool.report_failure(provider_name, key, exc.status_code, exc.retry_after); last = exc
            if outcome == "caller":
                break  # request itself is bad (4xx non-429); rotating keys cannot help
        except Exception as exc:
            pool.report_failure(provider_name, key); last = exc
    raise last or RuntimeError(f"provider `{provider_name}` has no available upstream key")


async def _call_media(provider, provider_name, model, body, pool, kind, sem=None):
    fn = getattr(provider, f"{kind}_gen", None)
    if not callable(fn):
        raise RuntimeError(f"provider `{provider_name}` does not support {kind} generation")
    attempts = max(1, pool.size(provider_name))
    last: Exception | None = None
    for _ in range(attempts):
        key = pool.acquire(provider_name)
        if key is None:
            break
        try:
            coro = lambda: fn(model, body, key)
            result = await (coro() if sem is None else _with_sem(sem, coro))
            if isinstance(result, httpx.Response) and result.status_code >= 400:
                raise UpstreamError(result.status_code, result.text[:300], retry_after=parse_retry_after(result.headers))
            pool.report_success(provider_name, key)
            return result
        except UpstreamError as exc:
            outcome = pool.report_failure(provider_name, key, exc.status_code, exc.retry_after); last = exc
            if outcome == "caller":
                break  # request itself is bad (4xx non-429); rotating keys cannot help
        except Exception as exc:
            pool.report_failure(provider_name, key); last = exc
    raise last or RuntimeError(f"provider `{provider_name}` has no available upstream key")


def _last_user_input(chat_body: dict) -> dict:
    """Extract the last user message as {text, images} for stage seeding."""
    text, images = "", []
    for m in (chat_body.get("messages") or [])[::-1]:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                for p in c:
                    if not isinstance(p, dict):
                        continue
                    if p.get("type") == "text":
                        text += p.get("text", "")
                    elif p.get("type") == "image_url":
                        u = (p.get("image_url") or {}).get("url") or ""
                        if u:
                            images.append(u)
            elif isinstance(c, str):
                text = c
            break
    return {"text": text, "images": images}


def _chat_dict(model: str, content: str, usage: dict | None = None) -> dict:
    u = usage or {}
    return {
        "id": f"chatcmpl-moa-{uuid.uuid4().hex[:24]}", "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": u.get("prompt_tokens", 0) or 0,
                  "completion_tokens": u.get("completion_tokens", 0) or 0,
                  "total_tokens": u.get("total_tokens", 0) or 0},
    }


async def run_moa(cfg, providers, pool, chat_body: dict, pipeline_name: str, stream: bool):
    """Dispatch a pipeline. Returns a chat-completion dict (non-stream) or an
    AsyncIterator of chat SSE bytes (stream)."""
    pipe = (cfg.raw.get("moa") or {}).get(pipeline_name)
    if not pipe:
        raise KeyError(pipeline_name)
    if pipe.get("stages"):
        return await _run_staged(cfg, providers, pool, chat_body, pipe, pipeline_name, stream)
    return await _run_proposers(cfg, providers, pool, chat_body, pipe, pipeline_name, stream)


# ---------------------------------------------------------------- parallel MOA
async def _run_proposers(cfg, providers, pool, chat_body, pipe, name, stream):
    proposers = pipe.get("proposers") or []
    agg = pipe.get("aggregator") or {}
    agg_prompt = pipe.get("aggregator_prompt") or DEFAULT_AGG_PROMPT
    default_effort = pipe.get("default_reasoning_effort")
    max_conc = int(pipe.get("max_concurrency") or 4)
    sem = asyncio.Semaphore(max_conc)
    if not proposers or not agg.get("provider") or not agg.get("model"):
        raise ValueError(f"MOA pipeline `{name}` requires proposers and aggregator")

    prop_body = dict(chat_body)
    prop_body["stream"] = False
    prop_body.pop("tools", None)
    prop_body.pop("tool_choice", None)

    async def one(p: dict) -> str:
        prov = providers.get(p.get("provider"))
        if prov is None:
            return f"[proposer {p.get('provider')} unavailable]"
        try:
            pb = dict(prop_body)
            _e = _effective_effort(p, default_effort)
            if _e:
                pb["reasoning_effort"] = _e
            res = await _call_chat(prov, p["provider"], p["model"], pb, pool, False, sem)
            data = res.json() if isinstance(res, httpx.Response) else res
            return _extract_text(data)
        except Exception as exc:
            logger.warning("MOA proposer %s/%s failed: %r", p.get("provider"), p.get("model"), exc)
            return f"[proposer {p.get('provider')}/{p.get('model')} failed: {exc!r}]"

    if pipe.get("early_stop"):
        min_sim = float(pipe.get("early_stop_similarity", 0.7))
        outputs = []
        for p in proposers:
            outputs.append(await one(p))
            if len(outputs) >= 2 and any(_jaccard(outputs[-1], k) >= min_sim for k in outputs[:-1]):
                break  # consensus reached, skip the rest
    else:
        outputs = await asyncio.gather(*[one(p) for p in proposers])
    if pipe.get("dedup", True):
        outputs = _dedup(outputs, float(pipe.get("dedup_threshold", 0.85)))
    if (pipe.get("compress") or {}).get("enabled"):
        outputs = await asyncio.gather(*[_compress(o, pipe, providers, pool, sem) for o in outputs])
    sections = [f"### Assistant {i}:\n{o}" for i, o in enumerate(outputs, 1)]
    sys_content = agg_prompt + "\n\nThe assistants' answers follow:\n\n" + "\n\n".join(sections)
    agg_messages = [{"role": "system", "content": sys_content}] + list(chat_body.get("messages") or [])
    agg_body = {"model": agg["model"], "messages": agg_messages, "stream": stream}
    for k in ("max_tokens", "temperature", "top_p", "tools", "tool_choice"):
        if k in chat_body:
            agg_body[k] = chat_body[k]
    _ae = _effective_effort(agg, default_effort)
    if _ae:
        agg_body["reasoning_effort"] = _ae
    prov = providers.get(agg["provider"])
    if prov is None:
        raise RuntimeError(f"aggregator provider `{agg['provider']}` unavailable")
    return await _call_chat(prov, agg["provider"], agg["model"], agg_body, pool, stream, sem)


# ------------------------------------------------------ staged multimodal pipeline
async def _run_staged(cfg, providers, pool, chat_body, pipe, name, stream):
    stages = pipe.get("stages") or []
    if not stages:
        raise ValueError(f"MOA pipeline `{name}` has empty stages")
    default_effort = pipe.get("default_reasoning_effort")
    max_conc = int(pipe.get("max_concurrency") or 4)
    sem = asyncio.Semaphore(max_conc)
    init = _last_user_input(chat_body)
    artifact = {"kind": "text", "text": init["text"], "images": init["images"]}

    async def run_stage(st: dict, art: dict, is_last: bool):
        modality = st.get("modality", "text")
        prov = providers.get(st.get("provider"))
        if prov is None:
            raise RuntimeError(f"stage provider `{st.get('provider')}` unavailable")
        model = st.get("model")
        timeout = float(st.get("timeout") or 120)
        prompt = st.get("prompt")
        prev_text = art.get("text") if art.get("kind") == "text" else ""
        inp_text = prompt if prompt else prev_text

        if modality in ("text", "vision"):
            cb = dict(chat_body)
            if modality == "vision":
                parts = []
                if inp_text:
                    parts.append({"type": "text", "text": inp_text})
                if art.get("kind") == "image" and art.get("url"):
                    imgs = [art["url"]]
                else:
                    imgs = art.get("images") or []
                for u in imgs:
                    parts.append({"type": "image_url", "image_url": {"url": u}})
                cb["messages"] = [{"role": "user", "content": parts or [{"type": "text", "text": inp_text}]}]
            else:
                cb["messages"] = [{"role": "user", "content": inp_text}]
            cb["model"] = model
            cb["stream"] = bool(stream and is_last)
            _e = _effective_effort(st, default_effort)
            if _e:
                cb["reasoning_effort"] = _e
            res = await asyncio.wait_for(_call_chat(prov, st["provider"], model, cb, pool, cb["stream"], sem), timeout)
            if cb["stream"]:
                return ("stream", res)  # async iter handed to caller
            data = res.json() if isinstance(res, httpx.Response) else res
            return {"kind": "text", "text": _extract_text(data), "usage": (data.get("usage") or {}), "_chat": data}

        if modality in ("image", "video"):
            body = {"model": model, "prompt": inp_text}
            for k in ("n", "size", "quality", "duration"):
                if k in st:
                    body[k] = st[k]
            kind = "image" if modality == "image" else "video"
            res = await asyncio.wait_for(_call_media(prov, st["provider"], model, body, pool, kind, sem), timeout)
            data = res.json() if isinstance(res, httpx.Response) else res
            item = ((data.get("data") or [{}])[0])
            url = item.get("url") or item.get("b64_json") or item.get("video_url") or ""
            return {"kind": kind, "url": url, "text": url}
        raise ValueError(f"unknown modality {modality!r}")

    # walk stages with parallel grouping
    i = 0
    n = len(stages)
    while i < n:
        group = [stages[i]]
        j = i
        while j + 1 < n and stages[j].get("parallel") and stages[j + 1].get("parallel"):
            j += 1
            group.append(stages[j])
        is_last_group = (j == n - 1)
        if len(group) == 1:
            try:
                r = await run_stage(group[0], artifact, is_last_group)
            except asyncio.TimeoutError:
                if group[0].get("optional"):
                    artifact = {"kind": "text", "text": artifact.get("text", "") + "\n[stage timed out, skipped]"}
                    i = j + 1; continue
                raise
            except Exception as exc:
                if group[0].get("optional"):
                    logger.warning("optional stage failed, skipped: %r", exc)
                    artifact = {"kind": "text", "text": artifact.get("text", "") + f"\n[stage failed, skipped: {exc!r}]"}
                    i = j + 1; continue
                raise
            if is_last_group and isinstance(r, tuple) and r[0] == "stream":
                return r[1]  # async iter
            if is_last_group:
                return _final_dict(r, group[0].get("model", name))
            artifact = r
        else:
            # parallel fan-out; merge text outputs (non-stream)
            async def grun(st):
                try:
                    return await run_stage(st, artifact, False)
                except Exception as exc:
                    if st.get("optional"):
                        return {"kind": "text", "text": f"[stage failed, skipped: {exc!r}]"}
                    raise
            results = await asyncio.gather(*[grun(st) for st in group])
            merged = "\n\n".join(r.get("text", "") for r in results if r.get("kind") == "text")
            artifact = {"kind": "text", "text": merged}
        i = j + 1

    # if we exit the loop without returning (e.g. last group was parallel), emit final
    return _final_dict(artifact, stages[-1].get("model", name))


def _final_dict(r: dict, model: str) -> dict:
    if r.get("kind") in ("image", "video"):
        url = r.get("url", "")
        md = f"![{r['kind']}]({url})" if r["kind"] == "image" else f"[video]({url})"
        return _chat_dict(model, md or "(no media)")
    if r.get("_chat"):
        return r["_chat"]
    return _chat_dict(model, r.get("text", ""))
