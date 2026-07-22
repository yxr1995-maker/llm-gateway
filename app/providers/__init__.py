"""Provider abstract base + registry + module-level shared httpx.AsyncClient.

Contract (SPEC):
    class ProviderBase:
        type: str
        def __init__(self, name: str, base_url: str, keys: list[str]): ...
        async def chat_completions(self, model, body, api_key, stream)
            -> httpx.Response | AsyncIterator[bytes]

- Non-streaming returns httpx.Response (body is OpenAI-format JSON; either the real upstream response, 
  or a converted/constructed response).
- Streaming returns AsyncIterator[bytes], yielding SSE bytes already converted to OpenAI chunk format
  （including the trailing "data: [DONE]"）；On upstream non-2xx, raise UpstreamError before the first yield,
  so the router can fail over within the key pool.
"""

from __future__ import annotations

from typing import AsyncIterator

import httpx

# default timeout (SPEC: all network calls have a timeout, default 60s)
DEFAULT_TIMEOUT = 60.0

# module-level shared async client (lazy; the whole gateway reuses one connection pool)
_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(DEFAULT_TIMEOUT),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=40),
        )
    return _client


async def close_client() -> None:
    """Close the shared client (called on app shutdown)."""
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        finally:
            _client = None


class UpstreamError(Exception):
    """Upstream returned non-2xx or the request failed; used by the router for key-pool failover.

    Carries `retry_after` (seconds, parsed from the upstream Retry-After header) so the pool
    can apply an accurate quota cooldown on 429 responses.
    """

    def __init__(self, status_code: int, message: str,
                 retry_after: float | None = None, headers: dict | None = None):
        super().__init__(f"upstream {status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.retry_after = retry_after
        self.headers = headers or {}


def parse_retry_after(headers) -> float | None:
    """Parse a Retry-After header (delta-seconds or HTTP-date) to seconds; None if absent/unparseable."""
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        return None
    if not raw:
        return None
    raw = raw.strip()
    try:
        v = float(raw)
        return v if v > 0 else None
    except ValueError:
        pass
    # HTTP-date fallback (RFC 7231)
    try:
        from email.utils import parsedate_to_datetime
        import time as _t
        dt = parsedate_to_datetime(raw)
        delta = dt.timestamp() - _t.time()
        return delta if delta > 0 else None
    except Exception:
        return None


def upstream_error_from_response(resp, max_bytes: int = 500) -> "UpstreamError":
    """Build an UpstreamError from an httpx.Response, parsing Retry-After from its headers."""
    try:
        detail = resp.text[:max_bytes]
    except Exception:
        detail = ""
    return UpstreamError(resp.status_code, detail, retry_after=parse_retry_after(resp.headers))


class ProviderBase:
    """Base class for all providers."""

    type: str = "base"

    def __init__(self, name: str, base_url: str, keys: list[str]):
        self.name = name
        self.base_url = (base_url or "").rstrip("/")
        self.keys = list(keys or [])
        # create_provider timeout and model list can be overridden per provider config
        self.timeout: float = DEFAULT_TIMEOUT
        self.models: list[str] = []
        self.supports_responses: bool = False  # whether the upstream natively supports /v1/responses
        self.image_path: str = "images/generations"  # media generation endpoint paths
        self.video_path: str = "videos/generations"
        self.video_poll_interval: float = 5.0   # async video task poll interval (s)
        self.video_max_wait: float = 180.0       # async video task max wait (s)

    async def chat_completions(
        self, model: str, body: dict, api_key: str, stream: bool
    ) -> "httpx.Response | AsyncIterator[bytes]":
        raise NotImplementedError


# ------------------------------------------------------------------ registry
_REGISTRY: dict[str, type[ProviderBase]] = {}


def register(type_name: str):
    """Decorator that registers a provider type."""

    def deco(cls: type[ProviderBase]) -> type[ProviderBase]:
        cls.type = type_name
        _REGISTRY[type_name] = cls
        return cls

    return deco


def create_provider(name: str, cfg: dict) -> ProviderBase:
    """Create a provider instance from config."""
    cfg = cfg or {}
    type_name = cfg.get("type") or "openai_like"
    cls = _REGISTRY.get(type_name)
    if cls is None:
        raise ValueError(f"unknown provider type: {type_name!r} (provider={name!r})")
    provider = cls(name=name, base_url=cfg.get("base_url") or "", keys=cfg.get("keys") or [])
    provider.models = list(cfg.get("models") or [])
    provider.supports_responses = bool(cfg.get("supports_responses"))
    if cfg.get("image_path"): provider.image_path = cfg["image_path"]
    if cfg.get("video_path"): provider.video_path = cfg["video_path"]
    if cfg.get("video_poll_interval"): provider.video_poll_interval = float(cfg["video_poll_interval"])
    if cfg.get("video_max_wait"): provider.video_max_wait = float(cfg["video_max_wait"])
    if cfg.get("timeout"):
        try:
            provider.timeout = float(cfg["timeout"])
        except (TypeError, ValueError):
            pass
    return provider


def build_providers(providers_cfg: dict) -> dict[str, ProviderBase]:
    """Build {name: instance} from the full providers config; a single failure doesn't affect others."""
    out: dict[str, ProviderBase] = {}
    for name, pcfg in (providers_cfg or {}).items():
        try:
            out[name] = create_provider(name, pcfg)
        except Exception:
            import logging

            logging.getLogger("llm-gateway.providers").exception(
                "provider failed to initialize: %s", name
            )
    return out


def known_types() -> list[str]:
    return sorted(_REGISTRY)


# import built-in providers to register them
from . import anthropic, gemini, openai_like  # noqa: E402,F401
