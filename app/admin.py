"""Admin API: /admin/api/*

Dependency-injection convention (wired in main.py):
    app.state.config      config (a dict or dict-convertible object; see config.example.yaml)
    app.state.providers   provider registry (optional; fallback when config has no providers)
    app.state.stats       StatsStore instance (may be uninit; endpoints return empty data)
    app.state.master_key  master key; when non-empty all /admin/api/* require Bearer auth

Mounting: the router registers both "/api/*" and "/admin/api/*" as equivalent path sets,
so main.py, whether `include_router(admin.router)` or
`include_router(admin.router, prefix="/admin")`，/admin/api/* both work.
"""

from __future__ import annotations

import asyncio
import hmac
import time
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request

router = APIRouter(include_in_schema=False)

_MASK = "****"
_HEALTH_TIMEOUT = 5.0   # health probe timeout (seconds)
_TEST_TIMEOUT = 60.0    # model test timeout (seconds)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _mask_key(key: Any) -> str:
    """Mask: show only the first 6 chars of a key + **** (fully masked if shorter than 6)."""
    s = str(key or "")
    if not s:
        return ""
    return s[:6] + _MASK if len(s) > 6 else _MASK


def _to_plain(obj: Any) -> Any:
    """Recursively convert a config object (dict / pydantic / plain object) into pure dict/list/scalars."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_plain(v) for v in obj]
    if hasattr(obj, "model_dump"):           # pydantic v2
        return _to_plain(obj.model_dump())
    if hasattr(obj, "dict") and callable(obj.dict):  # pydantic v1
        try:
            return _to_plain(obj.dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return _to_plain(vars(obj))
    return str(obj)


def _get_config(request: Request) -> dict:
    cfg = getattr(request.app.state, "config", None)
    plain = _to_plain(cfg)
    return plain if isinstance(plain, dict) else {}


def _get_master_key(request: Request, config: dict | None = None) -> str:
    key = getattr(request.app.state, "master_key", None)
    if key:
        return str(key)
    if config is None:
        config = _get_config(request)
    server = config.get("server") or {}
    return str(server.get("master_key") or "")


def _get_stats(request: Request):
    return getattr(request.app.state, "stats", None)


def _provider_items(request: Request, config: dict) -> list[tuple[str, dict]]:
    """Get the provider config list from config (preferred) or app.state.providers (fallback)."""
    providers = config.get("providers")
    if isinstance(providers, dict) and providers:
        return [(str(name), p if isinstance(p, dict) else {}) for name, p in providers.items()]
    if isinstance(providers, list) and providers:
        return [
            (str(p.get("name", f"provider{i}")), p)
            for i, p in enumerate(providers) if isinstance(p, dict)
        ]
    # fallback: the runtime provider registry (ProviderBase instances)
    registry = getattr(request.app.state, "providers", None)
    items: list[tuple[str, dict]] = []
    if isinstance(registry, dict):
        for name, obj in registry.items():
            items.append((
                str(name),
                {
                    "type": getattr(obj, "type", ""),
                    "base_url": getattr(obj, "base_url", ""),
                    "keys": list(getattr(obj, "keys", []) or []),
                    "models": list(getattr(obj, "models", []) or []),
                },
            ))
    return items


# ----------------------------------------------------------------------
# auth
# ----------------------------------------------------------------------
async def require_auth(request: Request) -> None:
    """When master_key is non-empty, verify Authorization: Bearer <master_key>."""
    config = _get_config(request)
    master_key = _get_master_key(request, config)
    if not master_key:
        return
    auth = request.headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not token or not hmac.compare_digest(token.encode(), master_key.encode()):
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "unauthorized: master key missing or wrong", "type": "authentication_error"}},
        )


# ----------------------------------------------------------------------
# Config overview (masked)
# ----------------------------------------------------------------------
def _build_masked_config(request: Request) -> dict:
    config = _get_config(request)
    server = dict(config.get("server") or {})
    if server.get("master_key"):
        server["master_key"] = _mask_key(server["master_key"])
    safe_providers: dict[str, dict] = {}
    for name, p in _provider_items(request, config):
        safe: dict[str, Any] = {
            "type": p.get("type", "openai_like"),
            "base_url": p.get("base_url", ""),
            "keys": [_mask_key(k) for k in (p.get("keys") or [])],
            "models": list(p.get("models") or []),
        }
        for k, v in p.items():
            if k not in safe and k not in ("name", "keys", "api_key", "key"):
                safe[k] = v
        safe_providers[name] = safe
    return {
        "server": server,
        "providers": safe_providers,
        "aliases": dict(config.get("aliases") or {}),
        "rate_limit": dict(config.get("rate_limit") or {}),
        "moa": dict(config.get("moa") or {}),
    }


async def get_config(request: Request, _: None = Depends(require_auth)):
    return _build_masked_config(request)


async def get_config_raw(request: Request, _: None = Depends(require_auth)):
    """Full unmasked config (for the admin editor; protected by require_auth)."""
    return _to_plain(_get_config(request))


def _apply_runtime(request: Request) -> None:
    """Rebuild providers / key pool / master_key after saving config."""
    cfg = request.app.state.config
    from .providers import build_providers
    request.app.state.providers = build_providers(cfg.providers)
    request.app.state.pool.sync(cfg.providers)
    request.app.state.master_key = cfg.master_key


async def put_config(request: Request, _: None = Depends(require_auth)):
    """Save the full config (called after admin edits)."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={"error": {"message": "request body must be JSON", "type": "invalid_request_error"}})
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail={"error": {"message": "config must be a JSON object", "type": "invalid_request_error"}})
    body.setdefault("server", {})
    body.setdefault("providers", {})
    body.setdefault("aliases", {})
    body.setdefault("rate_limit", {})
    body.setdefault("moa", {})
    cfg = request.app.state.config
    old_server = cfg.server
    body["server"].setdefault("host", old_server.get("host", "0.0.0.0"))
    body["server"].setdefault("port", old_server.get("port", 8080))
    cfg.save(body)
    _apply_runtime(request)
    return _build_masked_config(request)


# ----------------------------------------------------------------------
# usage stats (return empty data if stats is uninit; never 500)
# ----------------------------------------------------------------------
def _empty_summary(days: int) -> dict:
    return {
        "days": days,
        "since": time.time() - days * 86400,
        "totals": {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "avg_latency_ms": 0.0,
            "success_rate": 0.0,
        },
        "by_model": [],
        "by_provider": [],
    }


async def usage_summary(
    request: Request,
    days: int = Query(7, ge=1, le=365, description="stats over the last N days"),
    _: None = Depends(require_auth),
):
    stats = _get_stats(request)
    if stats is None:
        return _empty_summary(days)
    try:
        return await stats.summary(days=days)
    except Exception:
        return _empty_summary(days)


async def usage_recent(
    request: Request,
    limit: int = Query(50, ge=1, le=1000, description="return the last N rows"),
    _: None = Depends(require_auth),
):
    stats = _get_stats(request)
    if stats is None:
        return []
    try:
        return await stats.recent(limit=limit)
    except Exception:
        return []


# ----------------------------------------------------------------------
# connectivity probe
# ----------------------------------------------------------------------
def _probe_target(ptype: str, base_url: str, key: str) -> tuple[str, dict]:
    """Build a minimal probe request (GET /models level) by provider type."""
    if ptype == "anthropic":
        headers = {"anthropic-version": "2023-06-01"}
        if key:
            headers["x-api-key"] = key
        return base_url + "/v1/models", headers
    if ptype == "gemini":
        url = base_url + "/models" if "/v1beta" in base_url else base_url + "/v1beta/models"
        if key:
            url += ("&" if "?" in url else "?") + "key=" + key
        return url, {}
    # openai_like and other OpenAI-compatible protocols
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return base_url + "/models", headers


async def _probe_one(client: httpx.AsyncClient, name: str, pconf: dict) -> tuple[str, dict]:
    ptype = str(pconf.get("type") or "openai_like")
    base_url = str(pconf.get("base_url") or "").rstrip("/")
    keys = pconf.get("keys") or []
    key = str(keys[0]) if keys else ""
    if not base_url:
        return name, {"ok": False, "latency_ms": 0, "error": "base_url not configured"}
    url, headers = _probe_target(ptype, base_url, key)
    start = time.monotonic()
    try:
        resp = await client.get(url, headers=headers)
        latency = int((time.monotonic() - start) * 1000)
        ok = resp.status_code < 400
        result: dict[str, Any] = {"ok": ok, "latency_ms": latency}
        if not ok:
            result["error"] = f"HTTP {resp.status_code}"
        return name, result
    except Exception as exc:  # timeout / connection failure / DNS etc.
        latency = int((time.monotonic() - start) * 1000)
        return name, {"ok": False, "latency_ms": latency, "error": f"{type(exc).__name__}: {exc}"}


async def health(request: Request, _: None = Depends(require_auth)):
    """Send a minimal probe request to each provider (5s timeout), concurrently."""
    config = _get_config(request)
    items = _provider_items(request, config)
    if not items:
        return {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(_HEALTH_TIMEOUT)) as client:
        pairs = await asyncio.gather(*(_probe_one(client, name, p) for name, p in items))
    return {name: result for name, result in pairs}


# ----------------------------------------------------------------------
# model test: goes through the gateway's own /v1/chat/completions (loopback)
# ----------------------------------------------------------------------
async def test_model(request: Request, _: None = Depends(require_auth)):
    try:
        body = await request.json()
    except Exception:
        body = None
    model = (body or {}).get("model") if isinstance(body, dict) else None
    if not model or not isinstance(model, str):
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "request body must be JSON and include a model field", "type": "invalid_request_error"}},
        )

    config = _get_config(request)
    master_key = _get_master_key(request, config)
    base = str(request.base_url).rstrip("/")
    url = base + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if master_key:
        headers["Authorization"] = f"Bearer {master_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hi in one word"}],
        "stream": False,
    }

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(_TEST_TIMEOUT)) as client:
            resp = await client.post(url, json=payload, headers=headers)
        latency = int((time.monotonic() - start) * 1000)
    except Exception as exc:
        return {
            "ok": False,
            "latency_ms": int((time.monotonic() - start) * 1000),
            "error": f"{type(exc).__name__}: {exc}",
        }

    if resp.status_code >= 400:
        message = resp.text[:500]
        try:
            err = resp.json().get("error", {})
            if isinstance(err, dict):
                message = err.get("message") or message
            else:
                message = str(err)
        except Exception:
            pass
        return {"ok": False, "latency_ms": latency, "error": f"HTTP {resp.status_code}: {message}"}

    try:
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
    except Exception:
        return {"ok": False, "latency_ms": latency, "error": "response parse failed: " + resp.text[:300]}
    return {"ok": True, "latency_ms": latency, "reply": reply}


# ----------------------------------------------------------------------
# route registration: mount both /api/* and /admin/api/*, tolerating whether include_router adds prefix="/admin"
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# key-pool runtime state (which keys are cooled / flagged needs-reauth)
# ----------------------------------------------------------------------
async def pool_status(request: Request, _: None = Depends(require_auth)):
    """Per-provider key-pool runtime state: total / available / cooled counts and
    the masked keys flagged needs-reauth (returned 401/403). Lets the dashboard
    tell the operator which keys to rotate."""
    pool = getattr(request.app.state, "pool", None)
    config = _get_config(request)
    items = _provider_items(request, config)
    if pool is None or not items:
        return {}
    out: dict[str, Any] = {}
    for name, pconf in items:
        keys = pconf.get("keys") or []
        total = len(keys)
        available = pool.available(name) if hasattr(pool, "available") else total
        reauth = pool.needs_reauth(name) if hasattr(pool, "needs_reauth") else []
        out[name] = {
            "total": total,
            "available": available,
            "cooled": max(0, total - available),
            "needs_reauth": [_mask_key(k) for k in reauth],
        }
    return out


# ----------------------------------------------------------------------
# Codex config injection (wire the gateway into Codex as its provider)
# ----------------------------------------------------------------------
def _codex_config_path(explicit: str | None = None) -> str:
    import os
    if explicit:
        return explicit
    home = os.environ.get("CODEX_HOME") or os.path.expanduser("~/.codex")
    import os.path
    return os.path.join(home, "config.toml")


def _gateway_base_url(config: dict) -> str:
    server = config.get("server") or {}
    host = str(server.get("host") or "127.0.0.1")
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    port = server.get("port") or 8080
    return f"http://{host}:{port}/v1"


async def codex_inject_ep(
    request: Request,
    path: str | None = Query(None, description="Codex config.toml path (default: $CODEX_HOME/config.toml)"),
    _: None = Depends(require_auth),
):
    """Inject the gateway into Codex config as its model provider (idempotent, backed up)."""
    from .codex_inject import inject as _inject
    config = _get_config(request)
    return _inject(_codex_config_path(path), _gateway_base_url(config))


async def codex_restore_ep(
    request: Request,
    path: str | None = Query(None, description="Codex config.toml path"),
    _: None = Depends(require_auth),
):
    """Undo the injection (restores from backup)."""
    from .codex_inject import restore as _restore
    return _restore(_codex_config_path(path))


async def codex_status_ep(
    request: Request,
    path: str | None = Query(None, description="Codex config.toml path"),
    _: None = Depends(require_auth),
):
    """Report whether Codex config currently points at the gateway."""
    from .codex_inject import status as _status
    return _status(_codex_config_path(path))


for _prefix in ("", "/admin"):
    router.add_api_route(_prefix + "/api/config", get_config, methods=["GET"])
    router.add_api_route(_prefix + "/api/config/raw", get_config_raw, methods=["GET"])
    router.add_api_route(_prefix + "/api/config", put_config, methods=["PUT"])
    router.add_api_route(_prefix + "/api/usage/summary", usage_summary, methods=["GET"])
    router.add_api_route(_prefix + "/api/usage/recent", usage_recent, methods=["GET"])
    router.add_api_route(_prefix + "/api/health", health, methods=["GET"])
    router.add_api_route(_prefix + "/api/test", test_model, methods=["POST"])
    router.add_api_route(_prefix + "/api/pool", pool_status, methods=["GET"])
    router.add_api_route(_prefix + "/api/codex/inject", codex_inject_ep, methods=["POST"])
    router.add_api_route(_prefix + "/api/codex/restore", codex_restore_ep, methods=["POST"])
    router.add_api_route(_prefix + "/api/codex/status", codex_status_ep, methods=["GET"])
