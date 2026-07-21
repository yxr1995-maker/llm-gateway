"""Cascade pattern (effort-driven), independent of MOA / Planner-Worker.

The tier is chosen by the USER's reasoning-effort selection in the client
(e.g. Codex `model_reasoning_effort` -> sent as `reasoning.effort` in the
Responses API). Effort maps to a tier: low = cheap/fast, high = T1/expensive.
The user controls the cost/quality trade-off directly; the gateway just runs
the chosen tier (no auto-router overhead, no auto-escalation). An optional
`t1_verify` (strict) has T1 review/rewrite a cheap-tier answer.

Config (top-level `cascade`), trigger with model `cascade:<name>`:

  cascade:
    solve:
      tiers:
        - {name: L0, provider: agnes,   model: agnes-2.0-flash,  reasoning_effort: none}
        - {name: L1, provider: kimi,    model: kimi-for-coding,  reasoning_effort: low}
        - {name: L2, provider: volcano, model: glm-5.2,          reasoning_effort: high}   # T1
      effort_map: {none: 0, low: 0, medium: 1, high: 2, very_high: 2}   # user effort -> tier index
      default_tier: 2          # when the request carries no effort (0-based; default = top/T1)
      consensus_k: 1           # samples + majority vote at the chosen tier (stability)
      t1_verify: false         # strict: T1 verifies/rewrites a non-top-tier answer
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter

import httpx

from .moa import _call_chat, _chat_dict, _effective_effort, _extract_text, _last_user_input

logger = logging.getLogger("llm-gateway.cascade")

_DEFAULT_EFFORT_MAP = {"none": 0, "low": 0, "minimal": 0, "medium": 1, "high": 2, "very_high": 2}

_T1_VERIFY_PROMPT = (
    "You are the final reviewer. Given the task and a candidate answer, if the candidate is "
    "correct and complete, output it unchanged. Otherwise output a corrected, complete answer. "
    "Output ONLY the final answer, nothing else."
)


def cascade_name(model: str) -> str | None:
    if model.startswith("cascade:"):
        return model[len("cascade:"):]
    if model.startswith("cascade/"):
        return model[len("cascade/"):]
    return None


def is_cascade(model: str) -> bool:
    return cascade_name(model) is not None


def _text_of(res) -> str:
    d = res.json() if isinstance(res, httpx.Response) else res
    return _extract_text(d)


def _vote(outs: list[str]) -> str:
    c = Counter(o.strip() for o in outs if o and o.strip())
    return c.most_common(1)[0][0] if c else ""


async def _call_text(agent, providers, pool, messages, sem, default_effort, want_stream=False):
    prov = providers.get(agent.get("provider"))
    if prov is None:
        raise RuntimeError(f"provider `{agent.get('provider')}` unavailable")
    cb = {"model": agent["model"], "messages": messages, "stream": want_stream}
    _e = _effective_effort(agent, default_effort)
    if _e:
        cb["reasoning_effort"] = _e
    return await _call_chat(prov, agent["provider"], agent["model"], cb, pool, want_stream, sem)


async def run_cascade(cfg, providers, pool, chat_body: dict, name: str, stream: bool, effort: str | None = None):
    """Run the tier chosen by the user's reasoning effort. Returns a chat dict."""
    pipe = (cfg.raw.get("cascade") or {}).get(name)
    if not pipe:
        raise KeyError(name)
    tiers = pipe.get("tiers") or []
    if not tiers:
        raise ValueError(f"cascade `{name}` requires tiers")
    k = max(1, int(pipe.get("consensus_k") or 1))
    default_effort = pipe.get("default_reasoning_effort")
    max_conc = int(pipe.get("max_concurrency") or 4)
    sem = asyncio.Semaphore(max_conc)
    user_text = _last_user_input(chat_body)["text"]
    used = {"in": 0, "out": 0}

    def _acc(res):
        d = res.json() if isinstance(res, httpx.Response) else res
        u = (d.get("usage") or {}) if isinstance(d, dict) else {}
        used["in"] += int(u.get("prompt_tokens", u.get("input_tokens", 0)) or 0)
        used["out"] += int(u.get("completion_tokens", u.get("output_tokens", 0)) or 0)

    async def tier_answer(agent):
        if k <= 1:
            res = await _call_text(agent, providers, pool, [{"role": "user", "content": user_text}], sem, default_effort)
            _acc(res)
            return _text_of(res)
        outs = await asyncio.gather(*[
            _call_text(agent, providers, pool, [{"role": "user", "content": user_text}], sem, default_effort)
            for _ in range(k)])
        for r in outs:
            _acc(r)
        return _vote([_text_of(r) for r in outs])

    # 1) pick the tier from the user's reasoning effort
    effort_map = pipe.get("effort_map") or _DEFAULT_EFFORT_MAP
    if effort:
        idx = effort_map.get(str(effort).lower())
        if idx is None:
            idx = len(tiers) - 1
    else:
        idx = int(pipe.get("default_tier", len(tiers) - 1))
    idx = max(0, min(int(idx), len(tiers) - 1))
    agent = tiers[idx]

    # 2) run that tier
    try:
        cand = await tier_answer(agent)
    except Exception as exc:
        logger.warning("cascade tier %s (%s) failed: %r", agent.get("name") or idx, effort, exc)
        cand = ""

    # 3) optional T1 verify/rewrite (strict) when a non-top tier was used
    t1_verify = bool(pipe.get("t1_verify")) or pipe.get("strictness") == "strict"
    if t1_verify and idx < len(tiers) - 1 and cand:
        try:
            top = tiers[-1]
            res = await _call_text(top, providers, pool,
                [{"role": "system", "content": _T1_VERIFY_PROMPT},
                 {"role": "user", "content": f"Task: {user_text}\n\nCandidate:\n{cand}"}],
                sem, default_effort)
            _acc(res)
            verified = _text_of(res)
            if verified:
                cand = verified
        except Exception as exc:
            logger.warning("cascade t1_verify failed, keeping the tier answer: %r", exc)

    return _chat_dict(agent["model"], cand or "",
                      {"prompt_tokens": used["in"], "completion_tokens": used["out"],
                       "total_tokens": used["in"] + used["out"]})
