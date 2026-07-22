"""Key pool: round-robin + failure-classified cooldown + success recovery.

    pool = KeyPool()
    pool.sync(config.providers)          # sync each provider's key list from config
    key = pool.acquire("openai")         # round-robin an un-cooled key; None if none available
    outcome = pool.report_failure("openai", key, status_code=429, retry_after=30)
    pool.report_success("openai", key)   # success, clear cooldown / fail-count / reauth flag

Failure classification (mirrors the policy popularized by opencodex's routing):
  - 401/403  credential  -> long cooldown (won't self-heal in 60s) + mark needs-reauth
  - 429      quota       -> respect Retry-After, hard cooldown
  - 4xx*     caller      -> request itself is bad; do NOT cool the key, signal the router to stop
                            (* 429 excluded; rotating keys cannot fix a malformed request)
  - 5xx / network / None transient -> escalating soft backoff [30s, 2m, 10m, 30m]

report_failure() returns the outcome class ("credential" | "quota" | "caller" | "transient")
so the router can decide whether to keep rotating keys or bail out immediately.
"""

from __future__ import annotations

import time

# --- cooldown policy constants (seconds) ---
DEFAULT_COOLDOWN = 60.0
# credential (401/403): the key is invalid/revoked; it will not recover in 60s
CREDENTIAL_COOLDOWN = 3600.0
# quota (429): floor / ceiling when no usable Retry-After
QUOTA_COOLDOWN_MIN = 60.0
QUOTA_COOLDOWN_MAX = 86400.0
# transient (5xx / network / timeout): escalating ladder by consecutive failure count
TRANSIENT_LADDER = (30.0, 120.0, 600.0, 1800.0)  # 30s -> 2m -> 10m -> 30m


class KeyPool:
    def __init__(self) -> None:
        self._keys: dict[str, list[str]] = {}            # provider -> key list
        self._idx: dict[str, int] = {}                   # provider -> round-robin cursor
        self._cooldown: dict[tuple[str, str], float] = {}  # (provider, key) -> cooldown deadline (monotonic)
        self._fail_count: dict[tuple[str, str], int] = {}  # (provider, key) -> consecutive transient failures
        self._needs_reauth: set[tuple[str, str]] = set()   # keys flagged 401/403 (for admin surfacing)

    # ------------------------------------------------------------------ sync
    def register(self, provider: str, keys: list[str]) -> None:
        """Register/update a single provider's key list."""
        keys = list(keys or [])
        if self._keys.get(provider) != keys:
            self._keys[provider] = keys
            self._idx[provider] = 0
            # drop per-key state for removed keys
            for pk in [pk for pk in self._cooldown if pk[0] == provider and pk[1] not in keys]:
                del self._cooldown[pk]
            for pk in [pk for pk in self._fail_count if pk[0] == provider and pk[1] not in keys]:
                del self._fail_count[pk]
            self._needs_reauth = {pk for pk in self._needs_reauth
                                  if not (pk[0] == provider and pk[1] not in keys)}

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
            for pk in [pk for pk in self._fail_count if pk[0] == name]:
                del self._fail_count[pk]
            self._needs_reauth = {pk for pk in self._needs_reauth if pk[0] != name}

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

    def needs_reauth(self, provider: str) -> list[str]:
        """Keys flagged 401/403 since their last success (for admin surfacing)."""
        return [k for k in self._keys.get(provider) or [] if (provider, k) in self._needs_reauth]

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

    def report_failure(self, provider: str, key: str, status_code: int | None = None,
                       retry_after: float | None = None) -> str:
        """Classify the failure, apply the matching cooldown, and return the outcome class.

        Returns one of: "credential" | "quota" | "caller" | "transient".
        "caller" means the request itself is bad (4xx, non-429): the key is left untouched
        and the caller should stop rotating instead of burning the whole pool.
        """
        pk = (provider, key)
        now = time.monotonic()

        # credential error: key is invalid / revoked - won't heal in 60s
        if status_code in (401, 403):
            self._cooldown[pk] = now + CREDENTIAL_COOLDOWN
            self._needs_reauth.add(pk)
            self._fail_count.pop(pk, None)
            return "credential"

        # quota / rate-limit: honor Retry-After when present
        if status_code == 429:
            cd = QUOTA_COOLDOWN_MIN
            if retry_after and retry_after > 0:
                cd = min(max(retry_after, QUOTA_COOLDOWN_MIN), QUOTA_COOLDOWN_MAX)
            self._cooldown[pk] = now + cd
            self._fail_count.pop(pk, None)
            return "quota"

        # caller error (400/404/422/...): the request is malformed, not the key.
        # Do not punish the key; signal the router to stop failover.
        if status_code is not None and 400 <= status_code < 500:
            self._fail_count.pop(pk, None)
            return "caller"

        # transient (5xx / network / timeout / unknown): escalating backoff
        n = self._fail_count.get(pk, 0)
        self._fail_count[pk] = n + 1
        self._cooldown[pk] = now + TRANSIENT_LADDER[min(n, len(TRANSIENT_LADDER) - 1)]
        return "transient"

    def report_success(self, provider: str, key: str) -> None:
        """mark the key successful; clear cooldown / fail-count / reauth flag immediately."""
        pk = (provider, key)
        self._cooldown.pop(pk, None)
        self._fail_count.pop(pk, None)
        self._needs_reauth.discard(pk)
