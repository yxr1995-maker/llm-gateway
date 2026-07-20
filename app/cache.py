"""Result cache for upstream LLM calls (cross-cutting; used by MOA / PW / cascade).

Keyed by (provider, model, normalized request body). Non-streaming only.
Backed by SQLite (aiosqlite). Enabled via config `cache: {enabled: true, ttl: 3600}`
and wired in main.py lifespan.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time

import aiosqlite

ENABLED = False
TTL = 3600
_DB_PATH = "data/cache.db"

_lock = asyncio.Lock()
_db: aiosqlite.Connection | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS resp_cache (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cache_exp ON resp_cache (expires_at);
"""


async def init(db_path: str | None = None, enabled: bool | None = None, ttl: int | None = None):
    global _db, ENABLED, TTL, _DB_PATH
    if enabled is not None:
        ENABLED = bool(enabled)
    if ttl is not None:
        TTL = int(ttl)
    if db_path:
        _DB_PATH = db_path
    if not ENABLED:
        return
    _db = await aiosqlite.connect(_DB_PATH)
    await _db.executescript(_SCHEMA)
    await _db.commit()


async def close():
    global _db
    if _db is not None:
        await _db.close()
        _db = None


def make_key(provider: str, model: str, body: dict) -> str:
    b = dict(body or {})
    # strip non-deterministic / transport fields
    for k in ("stream", "stream_options", "seed", "user"):
        b.pop(k, None)
    try:
        norm = json.dumps({"p": provider, "m": model, "b": b}, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        norm = f"{provider}:{model}:{body}"
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


async def get(key: str):
    if not ENABLED or _db is None:
        return None
    try:
        async with _lock:
            cur = await _db.execute("SELECT value, expires_at FROM resp_cache WHERE key=?", (key,))
            row = await cur.fetchone()
            await cur.close()
        if not row:
            return None
        if row[1] < time.time():
            return None  # expired
        return json.loads(row[0])
    except Exception:
        return None


async def set(key: str, value, ttl: int | None = None):
    if not ENABLED or _db is None:
        return
    try:
        async with _lock:
            await _db.execute(
                "INSERT OR REPLACE INTO resp_cache (key, value, expires_at) VALUES (?, ?, ?)",
                (key, json.dumps(value, ensure_ascii=False), time.time() + (ttl or TTL)),
            )
            await _db.commit()
    except Exception:
        pass
