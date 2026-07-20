"""Cascade pattern (independent of MOA / Planner-Worker).

Tiered by reasoning level: L0 (fast/cheap) -> ... -> Ln (T1, slow/expensive).
A cheap router picks the starting tier; each tier samples k answers (self-
consistency) and a cheap self-verifier decides whether to accept or escalate.
Most requests stop at a cheap tier; T1 is only a quality floor. Quality is
~T1 on average (not a per-query hard guarantee); set strictness=strict to have
T1 verify low-confidence results.

Config (top-level `cascade`), trigger with model `cascade:<name>`:

  cascade:
    solve:
      router: { provider: agnes, model: agnes-2.0-flash }
      tiers:
        - {name: L0, provider: agnes, model: agnes-2.0-flash, reasoning_effort: none}
        - {name: L1, provider: kimi, model: kimi-for-coding, reasoning_effort: low}
        - {name: L2, provider: volcano, model: glm-5.2, reasoning_effort: high}
      consensus_k: 3
      strictness: balanced
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

_ROUTE_PROMPT = (
    "You are a router. Given the user task, output the minimum 0-based tier index needed "
    "(0 = trivial, higher = harder). Respond with ONLY JSON: {\"tier\":0}."
)
_VERIFY_PROMPT = (
    "You are a verifier. Given the task and a candidate answer, judge whether the answer is "
    "correct and complete. Respond with ONLY JSON: {\"confident\":true/false,\"score\":0.0-1.0,"
    "\"issues\":\"...\"}."
)


def cascade_name(model: str) -> str | None:
    if model.startswith("cascade:"):
        return model[len("cascade:"):]
    if model.startswith("cascade/"):
        return model[len("cascade/"):]
    return None


def is_cascade(model: str) -> bool:
    return cascade_name(model) is not None


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _text_of(res) -> str:
    d = res.json() if isinstance(res, httpx.Response) else res
    return _extract_text(d)


def _vote(outs: list[str]) -> str:
    c = Counter(o.strip() for o in outs if o and o.strip())
    if c:
        return c.most_common(1)[0][0]
    return ""


async def _call_text(agent, providers, pool, messages, sem, default_effort, want_stream=False):
    prov = providers.get(agent.get("provider"))
    if prov is None:
        raise RuntimeError(f"provider `{agent.get('provider')}` unavailable")
    cb = {"model": agent["model"], "messages": messages, "stream": want_stream}
    _e = _effective_effort(agent, default_effort)
    if _e:
        cb["reasoning_effort"] = _e
    return await _call_chat(prov, agent["provider"], agent["model"], cb, pool, want_stream, sem)


async def run_cascade(cfg, providers, pool, chat_body: dict, name: str, stream: bool):
    """Returns a chat-completion dict (non-stream). Streaming degrades to a
    non-stream JSON response (the cascade votes internally)."""
    pipe = (cfg.raw.get("cascade") or {}).get(name)
    if not pipe:
        raise KeyError(name)
    tiers = pipe.get("tiers") or []
    router = pipe.get("router")
    k = max(1, int(pipe.get("consensus_k") or 1))
    default_effort = pipe.get("default_reasoning_effort")
    max_conc = int(pipe.get("max_concurrency") or 4)
    sem = asyncio.Semaphore(max_conc)
    if not tiers:
        raise ValueError(f"cascade `{name}` requires tiers")
    user_text = _last_user_input(chat_body)["text"]
    verifier = router or tiers[0]

    async def tier_answer(agent):
        if k <= 1:
            res = await _call_text(agent, providers, pool, [{"role": "user", "content": user_text}], sem, default_effort)
            return _text_of(res)
        outs = await asyncio.gather(*[
            _call_text(agent, providers, pool, [{"role": "user", "content": user_text}], sem, default_effort)
            for _ in range(k)])
        return _vote([_text_of(r) for r in outs])

    async def verify(cand):
        try:
            res = await _call_text(verifier, providers, pool,
                                   [{"role": "system", "content": _VERIFY_PROMPT},
                                    {"role": "user", "content": f"Task: {user_text}\n\nAnswer: {cand}"}],
                                   sem, default_effort)
            d = _parse_json(_text_of(res)) or {}
            if d.get("confident") is True:
                return True
            score = d.get("score")
            try:
                return float(score) >= 0.7
            except (TypeError, ValueError):
                return False
        except Exception as exc:
            logger.warning("cascade verifier failed, assuming ok: %r", exc)
            return True

    # 1) upfront route
    start = 0
    if router:
        try:
            res = await _call_text(router, providers, pool,
                                   [{"role": "system", "content": _ROUTE_PROMPT},
                                    {"role": "user", "content": user_text}], sem, default_effort)
            d = _parse_json(_text_of(res)) or {}
            t = d.get("tier")
            if isinstance(t, int):
                start = max(0, min(t, len(tiers) - 1))
        except Exception as exc:
            logger.warning("cascade router failed, starting at L0: %r", exc)

    # 2) cascade: cheap -> T1
    last_cand = ""
    for i in range(start, len(tiers)):
        agent = tiers[i]
        is_top = (i == len(tiers) - 1)
        try:
            last_cand = await tier_answer(agent)
        except Exception as exc:
            logger.warning("cascade tier %s failed: %r", agent.get("name") or i, exc)
            last_cand = ""
        if is_top:
            break  # T1 floor: accept
        if await verify(last_cand):
            break  # accepted at a cheap tier
        # else escalate
    return _chat_dict(tiers[-1]["model"], last_cand or "")
