"""Difficulty router (RouteLLM-style): pick the cascade start tier by query difficulty
instead of relying on the user's manual reasoning_effort.

Two backends:
  - rule (default, zero-dep, runnable now): heuristic P(strong needed) from query signals.
  - mf  (plug-in): loads a trained Matrix-Factorization scorer (RouteLLM, arXiv:2406.18665)
    over query embeddings; needs an embedder + a .pkl produced from preference data.

Integration into cascade.run_cascade (guarded, default off):

    from .difficulty_router import DifficultyRouter
    # inside run_cascade, where idx is currently chosen via effort_map:
    rtr = DifficultyRouter.from_config(pipe.get("router") or {})
    if rtr.enabled and effort is None:        # only auto-route when user gave no effort
        dec = await rtr.predict(chat_body["messages"])
        idx = dec.tier_idx
        # dec.p_strong can be logged to stats for threshold tuning

Config (under cascade.<name>):

    router:
      enabled: true
      backend: rule            # rule | mf
      high_thr: 0.7            # p_strong >= high_thr -> top tier
      low_thr: 0.3             # p_strong <= low_thr -> bottom tier
      fallback_tier: 2         # on error / mf unavailable
      # mf backend:
      # model_path: data/router_mf.pkl
      # embedder: {provider: kimi, model: kimi-for-coding}   # any embedding-capable upstream
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("llm-gateway.difficulty_router")

# Default tier roster (matches your real providers/aliases). Override via config.
DEFAULT_TIERS = [
    {"name": "L0", "provider": "agnes", "model": "agnes-2.0-flash"},      # weak/cheap
    {"name": "L1", "provider": "kimi", "model": "kimi-for-coding"},       # mid
    {"name": "L2", "provider": "volcano", "model": "glm-5.2"},            # strong/T1
]

# Heuristic signals that push a query toward the strong tier.
_CODE_FENCE = re.compile(r"```|`[^`]{4,}`")
_REASON_MARKERS = re.compile(
    r"\b(debug|fix|bug|refactor|optimi[sz]e|architect|design|algorithm|prove|"
    r"derive|calculate|analy[sz]e|why|reason|step[- ]by[- ]step|edge case|"
    r"race condition|concurrency|security|vulnerab)\b", re.I
)
_EASY_MARKERS = re.compile(r"^(hi|hello|hey|你好|在吗|thanks|谢谢|ok|好的)\b", re.I)


@dataclass
class RouterDecision:
    tier_idx: int
    p_strong: float
    source: str            # "rule" | "mf" | "fallback"


class RuleRouter:
    """Heuristic scorer. Good enough to ship; collect p_strong to train the MF later."""

    def score(self, messages: list[dict]) -> float:
        text = _last_user_text(messages)
        if not text:
            return 0.5
        p = 0.20
        if _CODE_FENCE.search(text):
            p += 0.30
        if len(text) > 500:
            p += 0.20
        elif len(text) < 60 and not _REASON_MARKERS.search(text):
            p -= 0.10
        if _REASON_MARKERS.search(text):
            p += 0.20
        if _EASY_MARKERS.match(text.strip()):
            p -= 0.25
        return max(0.0, min(1.0, p))


class MFRouter:
    """Trained Matrix-Factorization scorer (RouteLLM). Loads a .pkl exposing
    `score(embedding: list[float]) -> float`. Embeddings are produced by an
    upstream embedder passed in `set_embedder`. Untrained -> falls back."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None
        self._embedder = None
        try:
            import pickle
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)
        except Exception as exc:
            logger.warning("MF router model not loaded (%s); will fallback", exc)

    def set_embedder(self, embed_fn):
        """embed_fn: async (text) -> list[float]"""
        self._embedder = embed_fn

    async def score(self, messages: list[dict]) -> float:
        if self._model is None or self._embedder is None:
            return -1.0  # signal fallback
        text = _last_user_text(messages)
        emb = await self._embedder(text)
        try:
            return float(self._model.score(emb))
        except Exception:
            return -1.0


class DifficultyRouter:
    def __init__(self, cfg: dict, tiers: list[dict] | None = None):
        self.enabled = bool(cfg.get("enabled"))
        self.backend = cfg.get("backend", "rule")
        self.high_thr = float(cfg.get("high_thr", 0.7))
        self.low_thr = float(cfg.get("low_thr", 0.3))
        self.fallback_tier = int(cfg.get("fallback_tier", 2))
        self.tiers = tiers or DEFAULT_TIERS
        self._rule = RuleRouter()
        self._mf = MFRouter(cfg["model_path"]) if self.backend == "mf" and cfg.get("model_path") else None

    @classmethod
    def from_config(cls, cfg: dict, tiers: list[dict] | None = None) -> "DifficultyRouter":
        return cls(cfg, tiers)

    def _tier_for_p(self, p: float) -> int:
        n = len(self.tiers)
        if n == 0:
            return self.fallback_tier
        if p >= self.high_thr:
            return n - 1                       # top tier
        if p <= self.low_thr:
            return 0                           # bottom tier
        # linear interp across middle tiers
        return max(1, min(n - 2, int((p - self.low_thr) / (self.high_thr - self.low_thr) * (n - 1))))

    async def predict(self, messages: list[dict]) -> RouterDecision:
        if not self.enabled:
            return RouterDecision(self.fallback_tier, 0.5, "fallback")
        try:
            if self._mf is not None:
                p = await self._mf.score(messages)
                if p >= 0:
                    return RouterDecision(self._tier_for_p(p), p, "mf")
            p = self._rule.score(messages)
            return RouterDecision(self._tier_for_p(p), p, "rule")
        except Exception as exc:
            logger.warning("router predict failed, fallback: %r", exc)
            return RouterDecision(self.fallback_tier, 0.5, "fallback")


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):  # multimodal: concat text parts
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return ""


if __name__ == "__main__":
    import asyncio

    async def demo():
        r = DifficultyRouter({"enabled": True, "backend": "rule"})
        samples = [
            [{"role": "user", "content": "你好"}],
            [{"role": "user", "content": "解释一下什么是 REST API"}],
            [{"role": "user", "content": "```python\ndef f(x): return x\n```\n这段代码有个并发竞态 bug，帮我定位并修复，要考虑边界情况"}],
        ]
        for msgs in samples:
            d = await r.predict(msgs)
            tier = r.tiers[d.tier_idx] if d.tier_idx < len(r.tiers) else {}
            print(f"p={d.p_strong:.2f} tier={d.tier_idx}({tier.get('name')}) src={d.source} | "
                  f"{_last_user_text(msgs)[:40]!r}")

    asyncio.run(demo())
