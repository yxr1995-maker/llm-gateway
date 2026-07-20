"""Provider 抽象基类 + registry + 模块级共享 httpx.AsyncClient。

契约（SPEC）：
    class ProviderBase:
        type: str
        def __init__(self, name: str, base_url: str, keys: list[str]): ...
        async def chat_completions(self, model, body, api_key, stream)
            -> httpx.Response | AsyncIterator[bytes]

- 非流式返回 httpx.Response（body 已是 OpenAI 格式 JSON；可能是真实上游响应，
  也可能是转换后构造的响应）。
- 流式返回 AsyncIterator[bytes]，产出已转换为 OpenAI chunk 格式的 SSE 字节流
  （含结尾 "data: [DONE]"）；上游非 2xx 时在首次 yield 前抛 UpstreamError，
  以便路由层在密钥池内故障转移。
"""

from __future__ import annotations

from typing import AsyncIterator

import httpx

# 默认超时（SPEC：所有网络调用带超时，默认 60s）
DEFAULT_TIMEOUT = 60.0

# 模块级共享异步 client（惰性创建，全网关复用连接池）
_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(DEFAULT_TIMEOUT),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _client


async def close_client() -> None:
    """关闭共享 client（应用退出时调用）。"""
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        finally:
            _client = None


class UpstreamError(Exception):
    """上游返回非 2xx 或请求失败，供路由层做密钥池故障转移。"""

    def __init__(self, status_code: int, message: str):
        super().__init__(f"upstream {status_code}: {message}")
        self.status_code = status_code
        self.message = message


class ProviderBase:
    """所有 provider 的基类。"""

    type: str = "base"

    def __init__(self, name: str, base_url: str, keys: list[str]):
        self.name = name
        self.base_url = (base_url or "").rstrip("/")
        self.keys = list(keys or [])
        # create_provider 可根据 provider 配置覆盖超时与模型列表
        self.timeout: float = DEFAULT_TIMEOUT
        self.models: list[str] = []
        self.supports_responses: bool = False  # 上游是否原生支持 /v1/responses

    async def chat_completions(
        self, model: str, body: dict, api_key: str, stream: bool
    ) -> "httpx.Response | AsyncIterator[bytes]":
        raise NotImplementedError


# ------------------------------------------------------------------ registry
_REGISTRY: dict[str, type[ProviderBase]] = {}


def register(type_name: str):
    """注册 provider 类型的装饰器。"""

    def deco(cls: type[ProviderBase]) -> type[ProviderBase]:
        cls.type = type_name
        _REGISTRY[type_name] = cls
        return cls

    return deco


def create_provider(name: str, cfg: dict) -> ProviderBase:
    """按配置创建 provider 实例。"""
    cfg = cfg or {}
    type_name = cfg.get("type") or "openai_like"
    cls = _REGISTRY.get(type_name)
    if cls is None:
        raise ValueError(f"未知 provider 类型: {type_name!r} (provider={name!r})")
    provider = cls(name=name, base_url=cfg.get("base_url") or "", keys=cfg.get("keys") or [])
    provider.models = list(cfg.get("models") or [])
    provider.supports_responses = bool(cfg.get("supports_responses"))
    if cfg.get("timeout"):
        try:
            provider.timeout = float(cfg["timeout"])
        except (TypeError, ValueError):
            pass
    return provider


def build_providers(providers_cfg: dict) -> dict[str, ProviderBase]:
    """按整份 providers 配置构建 {名称: 实例}；单个失败不影响其他。"""
    out: dict[str, ProviderBase] = {}
    for name, pcfg in (providers_cfg or {}).items():
        try:
            out[name] = create_provider(name, pcfg)
        except Exception:
            import logging

            logging.getLogger("llm-gateway.providers").exception(
                "provider 初始化失败: %s", name
            )
    return out


def known_types() -> list[str]:
    return sorted(_REGISTRY)


# 导入内置 provider 完成注册
from . import anthropic, gemini, openai_like  # noqa: E402,F401
