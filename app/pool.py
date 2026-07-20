"""密钥池：轮询 + 失败冷却 + 成功恢复（SPEC 契约）。

    pool = KeyPool()
    pool.sync(config.providers)          # 按配置同步各 provider 的 key 列表
    key = pool.acquire("openai")         # 轮询取一个未冷却的 key，无可用返回 None
    pool.report_failure("openai", key)   # 标记失败，默认冷却 60s
    pool.report_success("openai", key)   # 成功，立即解除冷却
"""

from __future__ import annotations

import time

# 失败后的默认冷却时长（秒）
DEFAULT_COOLDOWN = 60.0


class KeyPool:
    def __init__(self) -> None:
        self._keys: dict[str, list[str]] = {}            # provider -> key 列表
        self._idx: dict[str, int] = {}                   # provider -> 轮询游标
        self._cooldown: dict[tuple[str, str], float] = {}  # (provider, key) -> 冷却截止时刻(monotonic)

    # ------------------------------------------------------------------ 同步
    def register(self, provider: str, keys: list[str]) -> None:
        """注册/更新单个 provider 的 key 列表。"""
        keys = list(keys or [])
        if self._keys.get(provider) != keys:
            self._keys[provider] = keys
            self._idx[provider] = 0
            # 清理已移除 key 的冷却记录
            for pk in [pk for pk in self._cooldown if pk[0] == provider and pk[1] not in keys]:
                del self._cooldown[pk]

    def sync(self, providers_cfg: dict) -> None:
        """按整份 providers 配置同步（配置热重载后调用）。"""
        providers_cfg = providers_cfg or {}
        for name, pcfg in providers_cfg.items():
            self.register(name, (pcfg or {}).get("keys") or [])
        # 移除配置里已不存在的 provider
        for name in [n for n in self._keys if n not in providers_cfg]:
            del self._keys[name]
            self._idx.pop(name, None)
            for pk in [pk for pk in self._cooldown if pk[0] == name]:
                del self._cooldown[pk]

    # ------------------------------------------------------------------ 查询
    def size(self, provider: str) -> int:
        """该 provider 配置的 key 总数（含冷却中的）。"""
        return len(self._keys.get(provider) or [])

    def available(self, provider: str) -> int:
        """当前未冷却的 key 数。"""
        now = time.monotonic()
        return sum(
            1
            for k in self._keys.get(provider) or []
            if self._cooldown.get((provider, k), 0.0) <= now
        )

    # ------------------------------------------------------------------ 契约
    def acquire(self, provider: str) -> str | None:
        """轮询取 key，跳过冷却中的；全部冷却或未配置返回 None。"""
        keys = self._keys.get(provider) or []
        if not keys:
            return None
        now = time.monotonic()
        start = self._idx.get(provider, 0) % len(keys)
        for i in range(len(keys)):
            pos = (start + i) % len(keys)
            key = keys[pos]
            if self._cooldown.get((provider, key), 0.0) <= now:
                self._idx[provider] = (pos + 1) % len(keys)
                return key
        return None

    def report_failure(self, provider: str, key: str, cooldown: float = DEFAULT_COOLDOWN) -> None:
        """标记 key 失败，进入冷却。"""
        self._cooldown[(provider, key)] = time.monotonic() + cooldown

    def report_success(self, provider: str, key: str) -> None:
        """标记 key 成功，立即解除冷却。"""
        self._cooldown.pop((provider, key), None)
