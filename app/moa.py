"""Mixture-of-Agents (MOA): multiple proposers in parallel -> aggregator synthesis.

config (config.yaml):
  moa:
    default:
      proposers:
        - { provider: volcano, model: glm-5.2 }
        - { provider: kimi, model: kimi-for-coding }
      aggregator: { provider: volcano, model: glm-5.2 }
      aggregator_prompt: "..."   # optional, overrides the default synthesis prompt

Trigger: request model is `moa:<name>` or `moa/<name>`.
Runs in chat space: proposers fetch answers in parallel (non-stream), aggregator synthesizes (stream or non-stream).
A single proposer failure doesn't break the run (recorded as a note and skipped).
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import httpx

from .providers import UpstreamError

logger = logging.getLogger("llm-gateway.moa")

DEFAULT_AGG_PROMPT = (
    "You are the lead AI assistant. Several assistant models have answered the same user request."
    "Synthesize their answers: correct errors, remove redundancy, and produce a single high-quality final answer."
    "Keep tool calls only if clearly necessary; otherwise respond to the user directly."
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


async def _call_chat(provider, provider_name: str, model: str, chat_body: dict,
                     pool, stream: bool):
    """Single provider call + key-pool failover. Returns a chat response (httpx.Response or async iter)."""
    attempts = max(1, pool.size(provider_name))
    last: Exception | None = None
    for _ in range(attempts):
        key = pool.acquire(provider_name)
        if key is None:
            break
        try:
            result = await provider.chat_completions(model, chat_body, key, stream)
            if isinstance(result, httpx.Response) and result.status_code >= 400:
                raise UpstreamError(result.status_code, result.text[:300])
            pool.report_success(provider_name, key)
            return result
        except UpstreamError as exc:
            pool.report_failure(provider_name, key)
            last = exc
        except Exception as exc:  # network/parse
            pool.report_failure(provider_name, key)
            last = exc
    raise last or RuntimeError(f"provider `{provider_name}` has no available upstream key")


def _effective_effort(agent: dict, default: str | None) -> str | None:
    """Resolve an agent's reasoning effort: per-agent value wins, else the pipeline
    default; "default" means inherit the pipeline default. Returns None if none."""
    e = agent.get("reasoning_effort")
    if not e or e == "default":
        e = default
    return e if e and e != "none" else None


async def run_moa(cfg, providers, pool, chat_body: dict, pipeline_name: str,
                  stream: bool):
    """Run the MOA pipeline; return the aggregator's chat response (dict or async iter)."""
    pipe = (cfg.raw.get("moa") or {}).get(pipeline_name)
    if not pipe:
        raise KeyError(pipeline_name)
    proposers = pipe.get("proposers") or []
    agg = pipe.get("aggregator") or {}
    agg_prompt = pipe.get("aggregator_prompt") or DEFAULT_AGG_PROMPT
    default_effort = pipe.get("default_reasoning_effort")
    if not proposers or not agg.get("provider") or not agg.get("model"):
        raise ValueError(f"MOA pipeline `{pipeline_name}` requires proposers and aggregator")

    # proposers use non-stream, no tools (pure suggestions) to reduce tool noise
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
            res = await _call_chat(prov, p["provider"], p["model"], pb, pool, False)
            data = res.json() if isinstance(res, httpx.Response) else res
            return _extract_text(data)
        except Exception as exc:
            logger.warning("MOA proposer %s/%s failed: %r", p.get("provider"), p.get("model"), exc)
            return f"[proposer {p.get('provider')}/{p.get('model')} failed: {exc!r}]"

    outputs = await asyncio.gather(*[one(p) for p in proposers])

    sections = [f"### Assistant {i} ({p.get('provider')}/{p.get('model')}):\n{o}"
                for i, (p, o) in enumerate(zip(proposers, outputs), 1)]
    sys_content = agg_prompt + "\n\nThe assistants' answers follow:\n\n" + "\n\n".join(sections)

    agg_messages = [{"role": "system", "content": sys_content}]
    agg_messages += list(chat_body.get("messages") or [])

    agg_body: dict = {
        "model": agg["model"],
        "messages": agg_messages,
        "stream": stream,
    }
    for k in ("max_tokens", "temperature", "top_p", "tools", "tool_choice"):
        if k in chat_body:
            agg_body[k] = chat_body[k]

    _ae = _effective_effort(agg, default_effort)
    if _ae:
        agg_body["reasoning_effort"] = _ae
    prov = providers.get(agg["provider"])
    if prov is None:
        raise RuntimeError(f"aggregator provider `{agg['provider']}` unavailable")
    return await _call_chat(prov, agg["provider"], agg["model"], agg_body, pool, stream)
