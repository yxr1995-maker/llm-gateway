"""FastAPI 装配与启动入口。

启动方式：
    uvicorn app.main:app
    python -m app.main           # 读取 config 里的 server.host/port

配置文件：默认 ./config.yaml，可用环境变量 GATEWAY_CONFIG 覆盖。
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


# ------------------------------------------------------------- 可选模块装配
def _try_construct(candidates: list):
    """按候选 (可调用对象, 参数组合列表) 依次尝试实例化，全部失败返回 None。"""
    for fn, arg_options in candidates:
        if not callable(fn):
            continue
        for args in arg_options:
            try:
                obj = fn(*args)
            except Exception:
                continue
            if inspect.iscoroutine(obj):
                obj.close()  # 同步装配阶段不 await，关闭避免 RuntimeWarning
                continue
            if obj is not None:
                return obj
    return None


def _wire_stats(app: FastAPI, config: GatewayConfig) -> None:
    """duck-typing 装配用量统计模块（extras 提供；失败静默跳过）。"""
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
        logger.info("stats 模块已装配: %s", type(obj).__name__)
    else:
        logger.warning("app.stats 存在但无法实例化，跳过用量统计")


def _wire_ratelimiter(app: FastAPI, config: GatewayConfig) -> None:
    """duck-typing 装配限流器（extras 提供；requests_per_minute<=0 或失败则跳过）。"""
    try:
        rpm = int((config.rate_limit or {}).get("requests_per_minute") or 0)
    except (TypeError, ValueError):
        rpm = 0
    if rpm <= 0:
        return  # 0 = 不限
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
        logger.info("ratelimiter 已装配: %s (rpm=%s)", type(obj).__name__, rpm)
    else:
        logger.warning("app.ratelimit 存在但无法实例化，跳过限流")


async def _call_optional(obj, names: list[str]) -> None:
    """可选地调用 obj 的初始化/关闭钩子（同步异步均可）。"""
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
                logger.warning("可选钩子 %s.%s 调用失败", type(obj).__name__, name, exc_info=True)
            return


# ------------------------------------------------------------------ 装配
def create_app(config_path: str | None = None) -> FastAPI:
    config_path = config_path or os.environ.get("GATEWAY_CONFIG") or DEFAULT_CONFIG_PATH
    config = GatewayConfig(config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 启动：可选 stats 建表等初始化
        await _call_optional(getattr(app.state, "stats", None),
                             ["init", "setup", "connect", "open", "start"])
        yield
        # 关闭：可选模块清理 + 共享 httpx client
        await _call_optional(getattr(app.state, "stats", None),
                             ["close", "aclose", "stop", "shutdown"])
        await close_client()

    app = FastAPI(title="llm-gateway", version="0.1.0", lifespan=lifespan)
    app.state.config = config
    app.state.master_key = config.master_key  # 供 admin 等模块鉴权使用
    app.state.pool = KeyPool()
    app.state.pool.sync(config.providers)
    app.state.providers = build_providers(config.providers)

    # 可选：统计 / 限流（duck-typing，缺失不影响核心转发）
    _wire_stats(app, config)
    _wire_ratelimiter(app, config)

    app.include_router(router)

    # 管理 API（由 extras 提供，缺失则跳过）
    try:
        from app.admin import router as admin_router

        app.include_router(admin_router)
        logger.info("admin 路由已挂载")
    except ImportError:
        pass

    # 静态目录（static/admin.html 等），最后挂载避免覆盖 API 路由
    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.is_dir():
        from fastapi.responses import RedirectResponse

        @app.get("/", include_in_schema=False)
        async def _root():  # 根路径跳转到管理页
            return RedirectResponse("/static/admin.html")

        app.mount("/static", StaticFiles(directory=str(static_dir), html=True), name="static")
        logger.info("静态目录已挂载: %s (URL 前缀 /static)", static_dir)

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
