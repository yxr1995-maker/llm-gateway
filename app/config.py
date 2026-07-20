"""YAML config loading + model alias resolution + hot reload by file mtime.

Config format see SPEC.md / config.example.yaml:
- server: host/port/master_key
- providers: each provider's type/base_url/keys/models
- aliases: model alias -> "provider/real-model"
- rate_limit: requests_per_minute
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger("llm-gateway.config")

# Fallback empty config when the config file is missing / fails to parse
_EMPTY: dict = {"server": {}, "providers": {}, "aliases": {}, "rate_limit": {}}


class GatewayConfig:
    """Gateway config object; detects changes by mtime and hot-reloads."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._mtime: float | None = None
        self._data: dict = dict(_EMPTY)
        self.load()

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        """Read and parse YAML; keep the last valid config on failure."""
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            # Config file missing: empty config on first start, otherwise keep the previous one
            self._mtime = None
            logger.warning("config file not found: %s", self.path)
            return
        try:
            data = yaml.safe_load(text) or {}
            if not isinstance(data, dict):
                raise ValueError("config top level must be a mapping")
        except Exception as exc:  # YAML syntax error etc.
            logger.error("config parse failed, keeping previous config: %s", exc)
            return
        self._data = data
        self._mtime = self.path.stat().st_mtime
        logger.info("config loaded: %s", self.path)

    def save(self, data: dict) -> None:
        """Write the full config back to config.yaml and reload (called after admin edits)."""
        text = yaml.safe_dump(data or {}, allow_unicode=True, sort_keys=False)
        self.path.write_text(text, encoding="utf-8")
        try:
            self.path.chmod(0o600)
        except OSError:
            pass
        self.load()

    def maybe_reload(self) -> bool:
        """Reload if the file mtime changed; return whether a reload happened."""
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            return False
        if self._mtime is None or mtime != self._mtime:
            self.load()
            return True
        return False

    # ------------------------------------------------------------------ access
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
        """pydantic v1-style export (lets admin etc. turn the config object into a plain dict)."""
        return dict(self._data)

    # ------------------------------------------------------------- model resolution
    def resolve_model(self, model: str) -> tuple[str, str]:
        """Resolve the request model name to (provider name, real model name).

        Supports three forms (SPEC contract):
        1. alias: short name configured in aliases (chainable, cycle-safe)
        2. "provider/model" explicit
        3. a model name in a configured provider's models list (auto-matched)
        raises KeyError if unmatched.
        """
        if not model:
            raise KeyError(model)

        # 1. alias resolution (aliases may point to aliases, cycle-safe)
        seen: set[str] = set()
        cur = model
        while cur in self.aliases and cur not in seen:
            seen.add(cur)
            cur = str(self.aliases[cur])
        if cur in self.aliases:  # cycle; stop here
            cur = model

        # 2. provider/model explicit form
        if "/" in cur:
            provider_name, real = cur.split("/", 1)
            if provider_name in self.providers and real:
                return provider_name, real
            # prefix is not a configured provider -> continue by model name; raises KeyError if unmatched

        # 3. look up in each provider's models list
        for provider_name, pcfg in self.providers.items():
            models = (pcfg or {}).get("models") or []
            if cur in models:
                return provider_name, cur

        raise KeyError(model)
