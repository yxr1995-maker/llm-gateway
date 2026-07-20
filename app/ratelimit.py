"""In-memory token-bucket rate limiter (per key).

Usage (wired by main.py):

    app.state.ratelimiter = RateLimiter(config["rate_limit"]["requests_per_minute"])

    if not app.state.ratelimiter.allow(key):
        raise HTTPException(status_code=429, ...)

notes：
- bucket capacity = rate_per_minute, i.e. allows a burst of up to one minute's worth;
- refills at rate_per_minute / 60 tokens per second;
- rate_per_minute <= 0 means unlimited (config 0 = unlimited);
- pure in-process implementation; fine for a single-process service; counters reset on restart.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

_CLEANUP_THRESHOLD = 10000   # prune lazily when the bucket count exceeds this
_STALE_SECONDS = 600.0       # idle full buckets older than 10 minutes can be pruned


class RateLimiter:
    """Per-key token-bucket rate limiter."""

    def __init__(self, rate_per_minute: float, now: Callable[[], float] | None = None):
        """
        :param rate_per_minute: requests per minute per key; <= 0 means unlimited.
        :param now: time function (tests only; defaults to time.monotonic).
        """
        self.rate_per_minute = float(rate_per_minute or 0)
        self.capacity = self.rate_per_minute if self.rate_per_minute > 0 else 0.0
        self.refill_per_second = self.rate_per_minute / 60.0
        self._now = now or time.monotonic
        # key -> [tokens left, last refill time]
        self._buckets: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Try to consume a token for key. True if allowed, False if over the limit."""
        if self.rate_per_minute <= 0:
            return True  # 0 = unlimited
        now = self._now()
        with self._lock:
            self._maybe_cleanup(now)
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = [self.capacity, now]
                self._buckets[key] = bucket
            # refill evenly over elapsed time (capped at capacity)
            elapsed = now - bucket[1]
            if elapsed > 0:
                bucket[0] = min(self.capacity, bucket[0] + elapsed * self.refill_per_second)
                bucket[1] = now
            if bucket[0] >= 1.0:
                bucket[0] -= 1.0
                return True
            return False

    def retry_after(self, key: str) -> float:
        """Estimated seconds until key can pass again (for a 429 Retry-After; optional)."""
        if self.rate_per_minute <= 0:
            return 0.0
        now = self._now()
        with self._lock:
            bucket = self._buckets.get(key)
            tokens = self.capacity if bucket is None else bucket[0]
            if tokens >= 1.0:
                return 0.0
            return (1.0 - tokens) / self.refill_per_second

    def _maybe_cleanup(self, now: float) -> None:
        """When the bucket count is large, prune long-idle full buckets to bound memory."""
        if len(self._buckets) <= _CLEANUP_THRESHOLD:
            return
        stale_keys = [
            k for k, b in self._buckets.items()
            if b[0] >= self.capacity and now - b[1] > _STALE_SECONDS
        ]
        for k in stale_keys:
            del self._buckets[k]
