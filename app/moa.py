"""Mixture-of-Agents（MOA）：多 proposer 并行 -> aggregator 综合。

配置（config.yaml）：
  moa:
    default:
      proposers:
        - { provider: volcano, model: glm-5.2 }
        - { provider: kimi, model: kimi-for-coding }
      aggregator: { provider: volcano, model: glm-5.2 }
      aggregator_prompt: "..."   # 可选，覆盖默认综合提示

触发：请求 model 为 `moa:<name>` 或 `moa/<name>`。
内部在 chat 空间执行：proposer 并行取回答（非流式），aggregator 流式/非流式综合。
单 proposer 失败不影响整体（记为错误注记后继续）。
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import httpx

from .providers import UpstreamError

logger = logging.getLogger("llm-gateway.moa")

DEFAULT_AGG_PROMPT = (
    "你是首席 AI 助手。下方多个助手模型对同一用户请求给出了回答。"
    "请综合它们的回答，修正错误、去除冗余，输出一份高质量最终答复。"
    "如确有必要可保留工具调用，否则直接回应用户。"
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
    """单 provider 调用 + 密钥池故障转移。返回 chat 响应（httpx.Response 或 async iter）。"""
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
        except Exception as exc:  # 网络/解析
            pool.report_failure(provider_name, key)
            last = exc
    raise last or RuntimeError(f"provider `{provider_name}` 无可用上游 key")


async def run_moa(cfg, providers, pool, chat_body: dict, pipeline_name: str,
                  stream: bool):
    """执行 MOA 流水线，返回 aggregator 的 chat 响应（dict 或 async iter）。"""
    pipe = (cfg.raw.get("moa") or {}).get(pipeline_name)
    if not pipe:
        raise KeyError(pipeline_name)
    proposers = pipe.get("proposers") or []
    agg = pipe.get("aggregator") or {}
    agg_prompt = pipe.get("aggregator_prompt") or DEFAULT_AGG_PROMPT
    if not proposers or not agg.get("provider") or not agg.get("model"):
        raise ValueError(f"MOA pipeline `{pipeline_name}` 需配置 proposers 与 aggregator")

    # proposer 用非流式、不带工具（纯建议），降低工具噪声
    prop_body = dict(chat_body)
    prop_body["stream"] = False
    prop_body.pop("tools", None)
    prop_body.pop("tool_choice", None)

    async def one(p: dict) -> str:
        prov = providers.get(p.get("provider"))
        if prov is None:
            return f"[proposer {p.get('provider')} 不可用]"
        try:
            res = await _call_chat(prov, p["provider"], p["model"], prop_body, pool, False)
            data = res.json() if isinstance(res, httpx.Response) else res
            return _extract_text(data)
        except Exception as exc:
            logger.warning("MOA proposer %s/%s 失败: %r", p.get("provider"), p.get("model"), exc)
            return f"[proposer {p.get('provider')}/{p.get('model')} 失败: {exc!r}]"

    outputs = await asyncio.gather(*[one(p) for p in proposers])

    sections = [f"### 助手 {i} ({p.get('provider')}/{p.get('model')}):\n{o}"
                for i, (p, o) in enumerate(zip(proposers, outputs), 1)]
    sys_content = agg_prompt + "\n\n以下是各助手的回答：\n\n" + "\n\n".join(sections)

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

    prov = providers.get(agg["provider"])
    if prov is None:
        raise RuntimeError(f"aggregator provider `{agg['provider']}` 不可用")
    return await _call_chat(prov, agg["provider"], agg["model"], agg_body, pool, stream)
