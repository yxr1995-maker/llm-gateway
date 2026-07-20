"""Planner-Worker pattern (independent of MOA).

A strong planner model decomposes the task into subtasks; cheap worker models
execute them in parallel; the planner synthesizes a final answer (optionally
re-planning for up to max_rounds). Expensive tokens are spent only on planning,
grunt work goes to cheap models.

Config (top-level `planner_worker`), trigger with model `pw:<name>`:

  planner_worker:
    solve:
      planner: { provider: volcano, model: glm-5.2, reasoning_effort: high }
      workers:
        - { provider: kimi, model: kimi-for-coding, reasoning_effort: low }
        - { provider: agnes, model: agnes-2.0-flash }
      max_rounds: 1
      max_concurrency: 4
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

import httpx

from .moa import _call_chat, _compress, _effective_effort, _extract_text, _last_user_input

logger = logging.getLogger("llm-gateway.planner_worker")

_DECOMPOSE_PROMPT = (
    "You are the planner. Decompose the user's task into independent subtasks that "
    "cheaper worker models will execute in parallel. Respond with ONLY a JSON object: "
    '{"subtasks":[{"id":1,"description":"...","worker":0}]}. "worker" is an optional 0-based '
    "worker index. If no decomposition is needed, return a single subtask with the original task."
)
_CONTINUE_PROMPT = (
    "You are the planner. Given the task and the worker results so far, decide whether another "
    "round of subtasks is needed. Respond with ONLY JSON: "
    '{"continue":true/false,"subtasks":[{"id":1,"description":"..."}]}.'
)
_SYNTH_PROMPT = (
    "You are the lead assistant. Cheaper worker models executed the subtasks of the user's "
    "request. Their results follow. Synthesize one coherent final answer; resolve conflicts "
    "and fill gaps."
)


def pw_name(model: str) -> str | None:
    if model.startswith("pw:"):
        return model[3:]
    if model.startswith("pw/"):
        return model[3:]
    return None


def is_pw(model: str) -> bool:
    return pw_name(model) is not None


def _parse_json_obj(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _parse_subtasks(text: str) -> list[dict]:
    d = _parse_json_obj(text) or {}
    out: list[dict] = []
    for s in d.get("subtasks") or []:
        if isinstance(s, dict) and s.get("description"):
            out.append({"description": str(s["description"]), "worker": s.get("worker")})
        elif isinstance(s, str) and s.strip():
            out.append({"description": s, "worker": None})
    return out


async def run_planner_worker(cfg, providers, pool, chat_body: dict, name: str, stream: bool):
    """Returns a chat-completion dict (non-stream) or an AsyncIterator of chat SSE
    bytes (streamed final synthesis)."""
    pipe = (cfg.raw.get("planner_worker") or {}).get(name)
    if not pipe:
        raise KeyError(name)
    planner = pipe.get("planner") or {}
    workers = pipe.get("workers") or []
    max_rounds = max(1, int(pipe.get("max_rounds") or 1))
    max_conc = int(pipe.get("max_concurrency") or 4)
    sem = asyncio.Semaphore(max_conc)
    default_effort = pipe.get("default_reasoning_effort")
    if not planner.get("provider") or not planner.get("model"):
        raise ValueError(f"planner_worker `{name}` requires a planner")
    if not workers:
        raise ValueError(f"planner_worker `{name}` requires at least one worker")
    user_text = _last_user_input(chat_body)["text"]

    async def call_text(agent, messages, want_stream):
        prov = providers.get(agent.get("provider"))
        if prov is None:
            raise RuntimeError(f"provider `{agent.get('provider')}` unavailable")
        cb = {"model": agent["model"], "messages": messages, "stream": want_stream}
        _e = _effective_effort(agent, default_effort)
        if _e:
            cb["reasoning_effort"] = _e
        return await _call_chat(prov, agent["provider"], agent["model"], cb, pool, want_stream, sem)

    def text_of(res):
        d = res.json() if isinstance(res, httpx.Response) else res
        return _extract_text(d)

    async def run_workers(subtasks):
        async def work(i, sub):
            widx = sub.get("worker")
            if not isinstance(widx, int) or widx < 0 or widx >= len(workers):
                widx = i % len(workers)
            w = workers[widx]
            try:
                res = await call_text(w, [{"role": "user", "content": sub["description"]}], False)
                return f"[subtask {i+1}] {sub['description']}\n=> {text_of(res)}"
            except Exception as exc:
                logger.warning("planner_worker worker %s/%s failed: %r", w.get("provider"), w.get("model"), exc)
                return f"[subtask {i+1}] {sub['description']}\n=> [worker failed: {exc!r}]"
        return await asyncio.gather(*[work(i, s) for i, s in enumerate(subtasks)])

    # round 0: decompose
    plan_res = await call_text(planner, [{"role": "system", "content": _DECOMPOSE_PROMPT},
                                         {"role": "user", "content": user_text}], False)
    subtasks = _parse_subtasks(text_of(plan_res)) or [{"description": user_text, "worker": None}]
    all_results = list(await run_workers(subtasks))

    # extra rounds: planner decides whether to continue
    for _ in range(max_rounds - 1):
        cont_res = await call_text(planner, [{"role": "system", "content": _CONTINUE_PROMPT},
                                             {"role": "user", "content": user_text + "\n\nResults so far:\n\n" + "\n\n".join(all_results)}], False)
        cd = _parse_json_obj(text_of(cont_res)) or {}
        if not cd.get("continue"):
            break
        more = _parse_subtasks(json.dumps(cd))
        if not more:
            break
        all_results += list(await run_workers(more))

    # final synthesis (streamed if the client asked for stream)
    if (pipe.get("compress") or {}).get("enabled"):
        all_results = list(await asyncio.gather(*[_compress(r, pipe, providers, pool, sem) for r in all_results]))
    syn_messages = [{"role": "system", "content": _SYNTH_PROMPT + "\n\nWorker results:\n\n" + "\n\n".join(all_results)},
                    {"role": "user", "content": user_text}]
    return await call_text(planner, syn_messages, bool(stream))
