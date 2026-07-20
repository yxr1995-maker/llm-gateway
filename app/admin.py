"""管理 API：/admin/api/*

依赖注入约定（main.py 装配）：
    app.state.config      配置（dict 或可转 dict 的对象，见 config.example.yaml）
    app.state.providers   Provider registry（可选，config 缺 providers 时兜底）
    app.state.stats       StatsStore 实例（可能未初始化，接口需兜底返回空数据）
    app.state.master_key  管理密钥；非空时所有 /admin/api/* 需 Bearer 鉴权

挂载方式：router 同时注册了 "/api/*" 与 "/admin/api/*" 两组等价路径，
因此 main.py 无论 `include_router(admin.router)` 还是
`include_router(admin.router, prefix="/admin")`，/admin/api/* 都可用。
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
_HEALTH_TIMEOUT = 5.0   # 健康探测超时（秒）
_TEST_TIMEOUT = 60.0    # 模型测试超时（秒）


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def _mask_key(key: Any) -> str:
    """脱敏：key 只显示前 6 位 + ****（不足 6 位整体打码）。"""
    s = str(key or "")
    if not s:
        return ""
    return s[:6] + _MASK if len(s) > 6 else _MASK


def _to_plain(obj: Any) -> Any:
    """把配置对象（dict / pydantic / 普通对象）递归转成纯 dict/list/标量。"""
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
    """从 config（优先）或 app.state.providers（兜底）取 provider 配置列表。"""
    providers = config.get("providers")
    if isinstance(providers, dict) and providers:
        return [(str(name), p if isinstance(p, dict) else {}) for name, p in providers.items()]
    if isinstance(providers, list) and providers:
        return [
            (str(p.get("name", f"provider{i}")), p)
            for i, p in enumerate(providers) if isinstance(p, dict)
        ]
    # 兜底：运行时的 provider registry（ProviderBase 实例）
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
# 鉴权
# ----------------------------------------------------------------------
async def require_auth(request: Request) -> None:
    """master_key 非空时校验 Authorization: Bearer <master_key>。"""
    config = _get_config(request)
    master_key = _get_master_key(request, config)
    if not master_key:
        return
    auth = request.headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not token or not hmac.compare_digest(token.encode(), master_key.encode()):
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "未授权：master key 缺失或错误", "type": "authentication_error"}},
        )


# ----------------------------------------------------------------------
# 配置总览（脱敏）
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
    """未脱敏的完整配置（供管理页编辑用；受 require_auth 保护）。"""
    return _to_plain(_get_config(request))


def _apply_runtime(request: Request) -> None:
    """配置保存后重建 providers / 密钥池 / master_key。"""
    cfg = request.app.state.config
    from .providers import build_providers
    request.app.state.providers = build_providers(cfg.providers)
    request.app.state.pool.sync(cfg.providers)
    request.app.state.master_key = cfg.master_key


async def put_config(request: Request, _: None = Depends(require_auth)):
    """保存整份配置（管理页编辑后调用）。"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={"error": {"message": "请求体需为 JSON", "type": "invalid_request_error"}})
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail={"error": {"message": "配置需为 JSON 对象", "type": "invalid_request_error"}})
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
# 用量统计（stats 未初始化时返回空数据，绝不 500）
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
    days: int = Query(7, ge=1, le=365, description="统计最近 N 天"),
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
    limit: int = Query(50, ge=1, le=1000, description="返回最近 N 条"),
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
# 连通性探测
# ----------------------------------------------------------------------
def _probe_target(ptype: str, base_url: str, key: str) -> tuple[str, dict]:
    """按 provider 类型构造最小探测请求（GET /models 级别）。"""
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
    # openai_like 及其他 OpenAI 兼容协议
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return base_url + "/models", headers


async def _probe_one(client: httpx.AsyncClient, name: str, pconf: dict) -> tuple[str, dict]:
    ptype = str(pconf.get("type") or "openai_like")
    base_url = str(pconf.get("base_url") or "").rstrip("/")
    keys = pconf.get("keys") or []
    key = str(keys[0]) if keys else ""
    if not base_url:
        return name, {"ok": False, "latency_ms": 0, "error": "未配置 base_url"}
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
    except Exception as exc:  # 超时/连接失败/DNS 等
        latency = int((time.monotonic() - start) * 1000)
        return name, {"ok": False, "latency_ms": latency, "error": f"{type(exc).__name__}: {exc}"}


async def health(request: Request, _: None = Depends(require_auth)):
    """逐个 provider 发最小探测请求（超时 5s），并发执行。"""
    config = _get_config(request)
    items = _provider_items(request, config)
    if not items:
        return {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(_HEALTH_TIMEOUT)) as client:
        pairs = await asyncio.gather(*(_probe_one(client, name, p) for name, p in items))
    return {name: result for name, result in pairs}


# ----------------------------------------------------------------------
# 模型测试：走网关自身的 /v1/chat/completions（环回请求）
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
            detail={"error": {"message": "请求体需为 JSON 且包含 model 字段", "type": "invalid_request_error"}},
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
        return {"ok": False, "latency_ms": latency, "error": "响应解析失败: " + resp.text[:300]}
    return {"ok": True, "latency_ms": latency, "reply": reply}


# ----------------------------------------------------------------------
# 路由注册：同时挂 /api/* 与 /admin/api/*，兼容 include_router 是否加 prefix="/admin"
# ----------------------------------------------------------------------
for _prefix in ("", "/admin"):
    router.add_api_route(_prefix + "/api/config", get_config, methods=["GET"])
    router.add_api_route(_prefix + "/api/config/raw", get_config_raw, methods=["GET"])
    router.add_api_route(_prefix + "/api/config", put_config, methods=["PUT"])
    router.add_api_route(_prefix + "/api/usage/summary", usage_summary, methods=["GET"])
    router.add_api_route(_prefix + "/api/usage/recent", usage_recent, methods=["GET"])
    router.add_api_route(_prefix + "/api/health", health, methods=["GET"])
    router.add_api_route(_prefix + "/api/test", test_model, methods=["POST"])
