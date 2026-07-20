"""SQLite usage stats (async wrapper over aiosqlite).

Schema (see SPEC.md; do not change):

    CREATE TABLE IF NOT EXISTS usage (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts REAL, api_key TEXT, model TEXT, provider TEXT,
      prompt_tokens INT, completion_tokens INT, total_tokens INT,
      latency_ms INT, status INT, stream INT
    );

Usage (wired by main.py):

    stats = StatsStore("data/usage.db")
    await stats.init()                       # or await stats.init("data/usage.db")
    app.state.stats = stats

    await stats.record(ts=time.time(), api_key="sk-...", model="gpt-4o",
                       provider="openai", prompt_tokens=10, completion_tokens=20,
                       total_tokens=30, latency_ms=350, status=200, stream=0)
    data = await stats.summary(days=7)
    rows = await stats.recent(limit=50)
"""

from __future__ import annotations

import asyncio
import os
import time

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL, api_key TEXT, model TEXT, provider TEXT,
  prompt_tokens INT, completion_tokens INT, total_tokens INT,
  latency_ms INT, status INT, stream INT
);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage (ts);
"""

_DEFAULT_DB_PATH = os.path.join("data", "usage.db")


class StatsStore:
    """Usage stats store.

    Single connection + asyncio.Lock: all reads/writes are serialized under one lock,
    so record() can be called concurrently by any number of coroutines.
    the DB also enables WAL for read-only access by other processes/tools.
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or _DEFAULT_DB_PATH
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    async def init(self, db_path: str | None = None) -> None:
        """Open the connection and create tables. Idempotent."""
        if db_path:
            if self._db is not None and os.path.abspath(db_path) != os.path.abspath(self.db_path):
                await self.close()
            self.db_path = db_path
        if self._db is not None:
            return
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        try:
            await self._db.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass  # fall back to default logging if WAL is unavailable; functionality unaffected
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        """Close the connection (called on shutdown)."""
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def initialized(self) -> bool:
        return self._db is not None

    def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("StatsStore not initialized; await stats.init(db_path) first")
        return self._db

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------
    async def record(
        self,
        ts,
        api_key,
        model,
        provider,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        latency_ms,
        status,
        stream,
    ) -> None:
        """Async-write one call record (coroutine-safe; concurrently callable)."""
        db = self._require_db()
        async with self._lock:
            await db.execute(
                "INSERT INTO usage (ts, api_key, model, provider,"
                " prompt_tokens, completion_tokens, total_tokens,"
                " latency_ms, status, stream)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    float(ts),
                    str(api_key or ""),
                    str(model or ""),
                    str(provider or ""),
                    _to_int(prompt_tokens),
                    _to_int(completion_tokens),
                    _to_int(total_tokens),
                    _to_int(latency_ms),
                    _to_int(status),
                    1 if stream else 0,
                ),
            )
            await db.commit()

    # ------------------------------------------------------------------
    # aggregate
    # ------------------------------------------------------------------
    async def summary(self, days: int = 7) -> dict:
        """Aggregate usage over the last N days by model and provider.

        Returns:
            {
              "days": 7,
              "since": <start timestamp>,
              "totals":      {calls, prompt_tokens, completion_tokens,
                              total_tokens, avg_latency_ms, success_rate},
              "by_model":    [{model, same aggregate fields...}, ...],   # desc by call count
              "by_provider": [{provider, same aggregate fields...}, ...]
            }
        """
        db = self._require_db()
        days = max(1, _to_int(days, default=7))
        since = time.time() - days * 86400
        async with self._lock:
            by_model = await self._aggregate(db, "model", since)
            by_provider = await self._aggregate(db, "provider", since)
            totals = await self._aggregate(db, None, since)
        return {
            "days": days,
            "since": since,
            "totals": totals,
            "by_model": by_model,
            "by_provider": by_provider,
        }

    async def _aggregate(self, db: aiosqlite.Connection, group: str | None, since: float):
        select_extra = f"{group}, " if group else ""
        sql = (
            f"SELECT {select_extra}"
            "COUNT(*) AS calls,"
            " COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,"
            " COALESCE(SUM(completion_tokens), 0) AS completion_tokens,"
            " COALESCE(SUM(total_tokens), 0) AS total_tokens,"
            " AVG(latency_ms) AS avg_latency_ms,"
            " SUM(CASE WHEN status BETWEEN 200 AND 299 THEN 1 ELSE 0 END) * 1.0"
            " / COUNT(*) AS success_rate"
            " FROM usage WHERE ts >= ?"
        )
        if group:
            sql += f" GROUP BY {group} ORDER BY calls DESC, {group} ASC"
        async with db.execute(sql, (since,)) as cur:
            rows = await cur.fetchall()

        if group is None:
            if not rows or rows[0]["calls"] == 0:
                return {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "avg_latency_ms": 0.0,
                    "success_rate": 0.0,
                }
            return self._row_to_agg(rows[0])

        result = []
        for row in rows:
            item = {group: row[group]}
            item.update(self._row_to_agg(row))
            result.append(item)
        return result

    @staticmethod
    def _row_to_agg(row: aiosqlite.Row) -> dict:
        return {
            "calls": int(row["calls"]),
            "prompt_tokens": int(row["prompt_tokens"] or 0),
            "completion_tokens": int(row["completion_tokens"] or 0),
            "total_tokens": int(row["total_tokens"] or 0),
            "avg_latency_ms": round(float(row["avg_latency_ms"] or 0.0), 1),
            "success_rate": round(float(row["success_rate"] or 0.0), 4),
        }

    # ------------------------------------------------------------------
    # recent
    # ------------------------------------------------------------------
    async def recent(self, limit: int = 50) -> list:
        """Recent call details (reverse insertion order)."""
        db = self._require_db()
        limit = min(max(1, _to_int(limit, default=50)), 1000)
        async with self._lock:
            async with db.execute(
                "SELECT id, ts, api_key, model, provider, prompt_tokens,"
                " completion_tokens, total_tokens, latency_ms, status, stream"
                " FROM usage ORDER BY id DESC LIMIT ?",
                (limit,),
            ) as cur:
                rows = await cur.fetchall()
        return [
            {
                "id": row["id"],
                "ts": row["ts"],
                "api_key": row["api_key"],
                "model": row["model"],
                "provider": row["provider"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "total_tokens": row["total_tokens"],
                "latency_ms": row["latency_ms"],
                "status": row["status"],
                "stream": row["stream"],
            }
            for row in rows
        ]


def _to_int(value, default: int = 0) -> int:
    """Leniently coerce input to int (accepts None/empty/float)."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


# ----------------------------------------------------------------------
# convenience functions over a module-level default instance (same signatures as instance methods, for simple use)
# ----------------------------------------------------------------------
default_store = StatsStore()


async def init(db_path: str | None = None) -> None:
    await default_store.init(db_path)


async def record(
    ts,
    api_key,
    model,
    provider,
    prompt_tokens,
    completion_tokens,
    total_tokens,
    latency_ms,
    status,
    stream,
) -> None:
    await default_store.record(
        ts, api_key, model, provider,
        prompt_tokens, completion_tokens, total_tokens,
        latency_ms, status, stream,
    )


async def summary(days: int = 7) -> dict:
    return await default_store.summary(days)


async def recent(limit: int = 50) -> list:
    return await default_store.recent(limit)
