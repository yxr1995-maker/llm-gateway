"""FastAPI assembly and entry point.

Run with:
    uvicorn app.main:app
    python -m app.main           # reads server.host/port from config

Config file: defaults to ./config.yaml; override with the GATEWAY_CONFIG env var.
"""

from __future__ import annotations

import inspect
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import GatewayConfig
from .pool import KeyPool
from .providers import build_providers, close_client
from .router import router

logger = logging.getLogger("llm-gateway")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

DEFAULT_CONFIG_PATH = "./config.yaml"


# ------------------------------------------------------------- Optional module wiring
def _try_construct(candidates: list):
    """Try instantiating each candidate (callable, arg-combo list) in turn; return None if all fail."""
    for fn, arg_options in candidates:
        if not callable(fn):
            continue
        for args in arg_options:
            try:
                obj = fn(*args)
            except Exception:
                continue
            if inspect.iscoroutine(obj):
                obj.close()  # don't await during sync wiring; close to avoid RuntimeWarning
                continue
            if obj is not None:
                return obj
    return None


def _wire_stats(app: FastAPI, config: GatewayConfig) -> None:
    """Duck-typed wiring of the usage stats module (extras; silently skipped on failure)."""
    try:
        from app import stats as stats_mod
    except ImportError:
        return
    try:
        Path("data").mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    candidates = []
    for fname in ("create", "init", "get_stats", "default"):
        fn = getattr(stats_mod, fname, None)
        if callable(fn):
            candidates.append((fn, ((config,), tuple())))
    for cname in ("Stats", "UsageStats", "StatsDB", "StatsStore"):
        cls = getattr(stats_mod, cname, None)
        if cls is not None:
            candidates.append((cls, (("data/usage.db",), (config,), tuple())))
    obj = _try_construct(candidates)
    if obj is not None:
        app.state.stats = obj
        logger.info("stats module wired: %s", type(obj).__name__)
    else:
        logger.warning("app.stats exists but cannot be instantiated; skipping usage stats")


def _wire_ratelimiter(app: FastAPI, config: GatewayConfig) -> None:
    """Duck-typed wiring of the rate limiter (extras; skipped if requests_per_minute<=0 or on failure)."""
    try:
        rpm = int((config.rate_limit or {}).get("requests_per_minute") or 0)
    except (TypeError, ValueError):
        rpm = 0
    if rpm <= 0:
        return  # 0 = unlimited
    try:
        from app import ratelimit as rl_mod
    except ImportError:
        return
    candidates = []
    for cname in ("RateLimiter", "TokenBucket", "Limiter"):
        cls = getattr(rl_mod, cname, None)
        if cls is not None:
            candidates.append((cls, ((rpm,), tuple())))
    for fname in ("create", "create_limiter", "get_limiter", "default"):
        fn = getattr(rl_mod, fname, None)
        if callable(fn):
            candidates.append((fn, ((rpm,), (config,), tuple())))
    obj = _try_construct(candidates)
    if obj is not None:
        app.state.ratelimiter = obj
        logger.info("ratelimiter wired: %s (rpm=%s)", type(obj).__name__, rpm)
    else:
        logger.warning("app.ratelimit exists but cannot be instantiated; skipping rate limit")


async def _call_optional(obj, names: list[str]) -> None:
    """Optionally call the obj's init/close hooks (sync or async)."""
    if obj is None:
        return
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                res = fn()
                if inspect.isawaitable(res):
                    await res
            except Exception:
                logger.warning("optional hook %s.%s call failed", type(obj).__name__, name, exc_info=True)
            return


# ------------------------------------------------------------------ wire
def create_app(config_path: str | None = None) -> FastAPI:
    config_path = config_path or os.environ.get("GATEWAY_CONFIG") or DEFAULT_CONFIG_PATH
    config = GatewayConfig(config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # startup: optional stats table init etc.
        await _call_optional(getattr(app.state, "stats", None),
                             ["init", "setup", "connect", "open", "start"])
        yield
        # shutdown: optional module cleanup + shared httpx client
        await _call_optional(getattr(app.state, "stats", None),
                             ["close", "aclose", "stop", "shutdown"])
        await close_client()

    app = FastAPI(title="llm-gateway", version="0.1.0", lifespan=lifespan)
    app.state.config = config
    app.state.master_key = config.master_key  # for admin etc. to authenticate against
    app.state.pool = KeyPool()
    app.state.pool.sync(config.providers)
    app.state.providers = build_providers(config.providers)

    # optional: stats / rate-limit (duck-typed; absence doesn't affect core forwarding)
    _wire_stats(app, config)
    _wire_ratelimiter(app, config)

    app.include_router(router)

    # admin API (provided by extras; skipped if absent)
    try:
        from app.admin import router as admin_router

        app.include_router(admin_router)
        logger.info("admin router mounted")
    except ImportError:
        pass

    # static dir (static/admin.html etc.); mounted last to avoid overriding API routes
    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.is_dir():
        from fastapi.responses import RedirectResponse

        @app.get("/", include_in_schema=False)
        async def _root():  # root redirects to the admin console
            return RedirectResponse("/static/admin.html")

        app.mount("/static", StaticFiles(directory=str(static_dir), html=True), name="static")
        logger.info("static dir mounted: %s (URL prefix /static)", static_dir)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    server = app.state.config.server
    host = str(server.get("host") or "0.0.0.0")
    port = int(server.get("port") or 8080)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
