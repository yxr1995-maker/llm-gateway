"""Key pool: round-robin + failure cooldown + success recovery (SPEC contract).

    pool = KeyPool()
    pool.sync(config.providers)          # sync each provider's key list from config
    key = pool.acquire("openai")         # round-robin an un-cooled key; None if none available
    pool.report_failure("openai", key)   # mark failed, default cooldown 60s
    pool.report_success("openai", key)   # success, clear cooldown immediately
"""

from __future__ import annotations

import time

# default cooldown after failure (seconds)
DEFAULT_COOLDOWN = 60.0


class KeyPool:
    def __init__(self) -> None:
        self._keys: dict[str, list[str]] = {}            # provider -> key list
        self._idx: dict[str, int] = {}                   # provider -> round-robin cursor
        self._cooldown: dict[tuple[str, str], float] = {}  # (provider, key) -> cooldown deadline (monotonic)

    # ------------------------------------------------------------------ sync
    def register(self, provider: str, keys: list[str]) -> None:
        """Register/update a single provider's key list."""
        keys = list(keys or [])
        if self._keys.get(provider) != keys:
            self._keys[provider] = keys
            self._idx[provider] = 0
            # drop cooldown records for removed keys
            for pk in [pk for pk in self._cooldown if pk[0] == provider and pk[1] not in keys]:
                del self._cooldown[pk]

    def sync(self, providers_cfg: dict) -> None:
        """Sync from the full providers config (called after hot reload)."""
        providers_cfg = providers_cfg or {}
        for name, pcfg in providers_cfg.items():
            self.register(name, (pcfg or {}).get("keys") or [])
        # remove providers no longer in config
        for name in [n for n in self._keys if n not in providers_cfg]:
            del self._keys[name]
            self._idx.pop(name, None)
            for pk in [pk for pk in self._cooldown if pk[0] == name]:
                del self._cooldown[pk]

    # ------------------------------------------------------------------ queries
    def size(self, provider: str) -> int:
        """Total keys configured for this provider (including cooled ones)."""
        return len(self._keys.get(provider) or [])

    def available(self, provider: str) -> int:
        """Number of currently un-cooled keys."""
        now = time.monotonic()
        return sum(
            1
            for k in self._keys.get(provider) or []
            if self._cooldown.get((provider, k), 0.0) <= now
        )

    # ------------------------------------------------------------------ contract
    def acquire(self, provider: str) -> str | None:
        """Round-robin a key, skipping cooled ones; None if all cooled or unconfigured."""
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
        """mark the key failed and start cooldown."""
        self._cooldown[(provider, key)] = time.monotonic() + cooldown

    def report_success(self, provider: str, key: str) -> None:
        """mark the key successful; clear cooldown immediately."""
        self._cooldown.pop((provider, key), None)
