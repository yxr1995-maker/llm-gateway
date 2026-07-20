"""内存令牌桶限流（按 key 维度）。

用法（与 main.py 的装配方式）：

    app.state.ratelimiter = RateLimiter(config["rate_limit"]["requests_per_minute"])

    if not app.state.ratelimiter.allow(key):
        raise HTTPException(status_code=429, ...)

说明：
- 桶容量 = rate_per_minute，即允许瞬时突发最多一分钟的量；
- 匀速回填：rate_per_minute / 60 个令牌每秒；
- rate_per_minute <= 0 表示不限流（对应配置里的 0 = 不限）；
- 纯进程内存实现，单进程服务够用，重启后计数清零。
"""

from __future__ import annotations

import threading
import time
from typing import Callable

_CLEANUP_THRESHOLD = 10000   # 桶数量超过该值时触发惰性清理
_STALE_SECONDS = 600.0       # 空闲且已满的桶超过 10 分钟可被清理


class RateLimiter:
    """按 key 的令牌桶限流器。"""

    def __init__(self, rate_per_minute: float, now: Callable[[], float] | None = None):
        """
        :param rate_per_minute: 每个 key 每分钟允许的请求数；<= 0 表示不限流。
        :param now: 时间函数（仅测试用，默认 time.monotonic）。
        """
        self.rate_per_minute = float(rate_per_minute or 0)
        self.capacity = self.rate_per_minute if self.rate_per_minute > 0 else 0.0
        self.refill_per_second = self.rate_per_minute / 60.0
        self._now = now or time.monotonic
        # key -> [剩余令牌数, 上次回填时间]
        self._buckets: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """尝试为 key 消耗一个令牌。允许返回 True，超限返回 False。"""
        if self.rate_per_minute <= 0:
            return True  # 0 = 不限
        now = self._now()
        with self._lock:
            self._maybe_cleanup(now)
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = [self.capacity, now]
                self._buckets[key] = bucket
            # 按经过的时间匀速回填（不超过容量）
            elapsed = now - bucket[1]
            if elapsed > 0:
                bucket[0] = min(self.capacity, bucket[0] + elapsed * self.refill_per_second)
                bucket[1] = now
            if bucket[0] >= 1.0:
                bucket[0] -= 1.0
                return True
            return False

    def retry_after(self, key: str) -> float:
        """预计多少秒后 key 可以再次通过（用于构造 429 的 Retry-After，可选）。"""
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
        """桶数量过大时清理长时间空闲且已充满的桶，防止内存无限增长。"""
        if len(self._buckets) <= _CLEANUP_THRESHOLD:
            return
        stale_keys = [
            k for k, b in self._buckets.items()
            if b[0] >= self.capacity and now - b[1] > _STALE_SECONDS
        ]
        for k in stale_keys:
            del self._buckets[k]
