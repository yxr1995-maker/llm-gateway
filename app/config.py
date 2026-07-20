"""YAML 配置加载 + 模型别名解析 + 按文件 mtime 热重载。

配置格式见 SPEC.md / config.example.yaml：
- server: host/port/master_key
- providers: 各 provider 的 type/base_url/keys/models
- aliases: 模型别名 → "provider/真实模型名"
- rate_limit: requests_per_minute
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger("llm-gateway.config")

# 配置文件不存在 / 解析失败时的兜底空配置
_EMPTY: dict = {"server": {}, "providers": {}, "aliases": {}, "rate_limit": {}}


class GatewayConfig:
    """网关配置对象，按 mtime 检测变更并热重载。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._mtime: float | None = None
        self._data: dict = dict(_EMPTY)
        self.load()

    # ------------------------------------------------------------------ 加载
    def load(self) -> None:
        """读取并解析 YAML；失败时保留上一份可用配置。"""
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            # 配置文件缺失：首次启动给空配置，之后保留旧配置
            self._mtime = None
            logger.warning("配置文件不存在: %s", self.path)
            return
        try:
            data = yaml.safe_load(text) or {}
            if not isinstance(data, dict):
                raise ValueError("配置顶层必须是 mapping")
        except Exception as exc:  # YAML 语法错误等
            logger.error("配置解析失败，保留旧配置: %s", exc)
            return
        self._data = data
        self._mtime = self.path.stat().st_mtime
        logger.info("配置已加载: %s", self.path)

    def maybe_reload(self) -> bool:
        """若文件 mtime 变化则重载，返回是否发生了重载。"""
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            return False
        if self._mtime is None or mtime != self._mtime:
            self.load()
            return True
        return False

    # ------------------------------------------------------------------ 访问
    @property
    def server(self) -> dict:
        return self._data.get("server") or {}

    @property
    def providers(self) -> dict:
        return self._data.get("providers") or {}

    @property
    def aliases(self) -> dict:
        return self._data.get("aliases") or {}

    @property
    def rate_limit(self) -> dict:
        return self._data.get("rate_limit") or {}

    @property
    def master_key(self) -> str:
        return str(self.server.get("master_key") or "").strip()

    @property
    def raw(self) -> dict:
        return self._data

    def dict(self) -> dict:
        """pydantic v1 风格导出（供 admin 等模块把配置对象转纯 dict）。"""
        return dict(self._data)

    # ------------------------------------------------------------- 模型解析
    def resolve_model(self, model: str) -> tuple[str, str]:
        """把请求中的模型名解析为 (provider 名, 真实模型名)。

        支持三种写法（SPEC 接口契约）：
        1. 别名：aliases 里配置的短名（可链式，防环）
        2. "provider/model" 显式指定
        3. 已配置 provider 的 models 列表中的模型名（自动匹配）
        未命中抛 KeyError。
        """
        if not model:
            raise KeyError(model)

        # 1. 别名解析（允许别名指向别名，防环）
        seen: set[str] = set()
        cur = model
        while cur in self.aliases and cur not in seen:
            seen.add(cur)
            cur = str(self.aliases[cur])
        if cur in self.aliases:  # 有环，停在原地
            cur = model

        # 2. provider/model 显式写法
        if "/" in cur:
            provider_name, real = cur.split("/", 1)
            if provider_name in self.providers and real:
                return provider_name, real
            # 前缀不是已配置 provider → 继续按模型名查找，最后未命中会抛 KeyError

        # 3. 在各 provider 的 models 列表里查找
        for provider_name, pcfg in self.providers.items():
            models = (pcfg or {}).get("models") or []
            if cur in models:
                return provider_name, cur

        raise KeyError(model)
