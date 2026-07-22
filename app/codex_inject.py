"""Codex config injection: wire the gateway into Codex as its model provider.

Mirrors opencodex's injection approach (Design B) adapted for this gateway:
  - adds a marker-tagged ``[model_providers.<name>]`` table pointing at the gateway
    (wire_api = "responses"), and sets the root ``model_provider`` to it;
  - idempotent (re-inject refreshes base_url in place), EOL-safe (preserves CRLF/LF),
    and fully reversible (backup + restore);
  - operates on a config *path* so it is unit-testable on temp copies without
    touching the live ``~/.codex/config.toml``.

Auth mirrors the proven CC Switch pattern used in the live config
(``requires_openai_auth = true`` + ``experimental_bearer_token = "PROXY_MANAGED"``);
pair with ``server.trust_loopback: true`` in the gateway so loopback Codex requests
are accepted without a separate master key (just like a local proxy should).
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

MARKER_START = "# >>> llm-gateway proxy (auto-injected; restore: python -m app.codex_inject restore)"
MARKER_END = "# <<< llm-gateway proxy"
DEFAULT_PROVIDER = "llm-gateway"
DEFAULT_NAME = "LLM Gateway"
BACKUP_SUFFIX = ".llm-gateway-bak"

_ROOT_MODEL_PROVIDER_RE = re.compile(r"^[ \t]*model_provider[ \t]*=", re.MULTILINE)


# ------------------------------------------------------------------ EOL handling
def detect_eol(text: str) -> str:
    crlf = text.count("\r\n")
    if crlf == 0:
        return "\n"
    return "\r\n" if crlf >= (text.count("\n") - crlf) else "\n"


def _to_lf(text: str) -> str:
    return text.replace("\r\n", "\n")


def _apply_eol(text: str, eol: str) -> str:
    lf = _to_lf(text)
    return lf if eol == "\n" else lf.replace("\n", "\r\n")


# ------------------------------------------------------------------ marker blocks
def _strip_marker_blocks(text: str) -> str:
    """Remove every marker-wrapped block (MARKER_START .. MARKER_END, inclusive)."""
    lines = _to_lf(text).split("\n")
    out: list[str] = []
    skipping = False
    for line in lines:
        if line.strip() == MARKER_START:
            skipping = True
            continue
        if skipping:
            if line.strip() == MARKER_END:
                skipping = False
            continue
        out.append(line)
    # collapse trailing blank lines left by removal, keep a single trailing newline
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + ("\n" if out else "")


def _marker_blocks(text: str) -> list[str]:
    """Return the contents of every marker-wrapped block."""
    blocks: list[str] = []
    cur: list[str] | None = None
    for line in _to_lf(text).split("\n"):
        if line.strip() == MARKER_START:
            cur = []
        elif line.strip() == MARKER_END:
            if cur is not None:
                blocks.append("\n".join(cur))
            cur = None
        elif cur is not None:
            cur.append(line)
    return blocks


def is_injected(text: str) -> bool:
    return MARKER_START in text


def _replace_root_model_provider(root_lines: list[str], provider: str) -> list[str]:
    """Replace the first root ``model_provider =`` line with a marker-wrapped block.
    If none exists, the caller inserts the block at the top."""
    block = [MARKER_START, f'model_provider = "{provider}"', MARKER_END]
    for i, line in enumerate(root_lines):
        if _ROOT_MODEL_PROVIDER_RE.match(line):
            return root_lines[:i] + block + root_lines[i + 1:]
    return None  # not found


def _provider_table_block(provider: str, name: str, base_url: str, wire_api: str,
                          requires_openai_auth: bool, bearer: str | None) -> str:
    lines = [MARKER_START, f"[model_providers.{provider}]", f'name = "{name}"',
             f'base_url = "{base_url}"', f'wire_api = "{wire_api}"']
    if requires_openai_auth:
        lines.append("requires_openai_auth = true")
    if bearer:
        lines.append(f'experimental_bearer_token = "{bearer}"')
    lines.append(MARKER_END)
    return "\n".join(lines)


# ------------------------------------------------------------------ public API
def inject(config_path: str | Path, base_url: str, *,
           provider: str = DEFAULT_PROVIDER, name: str = DEFAULT_NAME,
           wire_api: str = "responses", requires_openai_auth: bool = True,
           bearer: str | None = "PROXY_MANAGED") -> dict:
    """Inject the gateway as Codex's model provider. Idempotent + backed up.

    Returns a dict: {ok, path, injected, provider, base_url, backed_up, backup_path}.
    """
    path = Path(config_path).expanduser()
    text = path.read_bytes().decode("utf-8") if path.exists() else ""
    eol = detect_eol(text) if text else "\n"
    backup_path = path.with_suffix(path.suffix + BACKUP_SUFFIX)

    was_injected = is_injected(text)
    # back up the clean (non-injected) state so restore is lossless
    if not was_injected and text.strip():
        backup_path.write_bytes(text.encode("utf-8"))

    body = _to_lf(text)
    body = _strip_marker_blocks(body)

    # root model_provider: replace in place if present, else insert at top
    lines = body.split("\n")
    replaced = _replace_root_model_provider(lines, provider)
    if replaced is not None:
        lines = replaced
    else:
        lines = [MARKER_START, f'model_provider = "{provider}"', MARKER_END, ""] + lines
    body = "\n".join(lines)
    if not body.endswith("\n"):
        body += "\n"

    # append the provider table at EOF
    table = _provider_table_block(provider, name, base_url, wire_api,
                                  requires_openai_auth, bearer)
    body = body.rstrip("\n") + "\n\n" + table + "\n"

    path.write_bytes(_apply_eol(body, eol).encode("utf-8"))
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return {
        "ok": True, "path": str(path), "injected": True, "provider": provider,
        "base_url": base_url, "backed_up": (not was_injected) and bool(text.strip()),
        "backup_path": str(backup_path),
    }


def restore(config_path: str | Path) -> dict:
    """Undo the injection. Prefers the full backup; otherwise strips marker blocks."""
    path = Path(config_path).expanduser()
    backup_path = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if backup_path.exists():
        shutil.copy2(backup_path, path)
        return {"ok": True, "path": str(path), "restored": True, "via": "backup",
                "backup_path": str(backup_path)}
    # no backup: best-effort strip (model_provider original value is unrecoverable)
    text = path.read_bytes().decode("utf-8") if path.exists() else ""
    if not is_injected(text):
        return {"ok": False, "path": str(path), "restored": False, "via": "none",
                "error": "not injected and no backup found"}
    eol = detect_eol(text)
    body = _apply_eol(_strip_marker_blocks(text), eol)
    path.write_bytes(body.encode("utf-8"))
    return {"ok": True, "path": str(path), "restored": True, "via": "strip",
            "warning": "no backup; original model_provider value not recovered"}


def status(config_path: str | Path) -> dict:
    """Report whether the config is currently injected, and the injected base_url."""
    path = Path(config_path).expanduser()
    backup_path = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    text = path.read_bytes().decode("utf-8") if path.exists() else ""
    injected = is_injected(text)
    base_url = provider = None
    if injected:
        for blk in _marker_blocks(text):
            mp = re.search(r'\[model_providers\.([^\]]+)\]', blk)
            if mp:
                provider = mp.group(1)
                mb = re.search(r'base_url[ \t]*=[ \t]*"([^"]+)"', blk)
                base_url = mb.group(1) if mb else None
                break
    return {"path": str(path), "exists": path.exists(), "injected": injected,
            "provider": provider, "base_url": base_url,
            "backup_exists": backup_path.exists(), "backup_path": str(backup_path)}


# ------------------------------------------------------------------ CLI
def _default_codex_config() -> str:
    import os
    home = os.environ.get("CODEX_HOME") or os.path.expanduser("~/.codex")
    return str(Path(home) / "config.toml")


def main(argv: list[str] | None = None) -> int:
    import argparse, json
    p = argparse.ArgumentParser(prog="python -m app.codex_inject",
                                description="Inject/restore the gateway as Codex's provider")
    p.add_argument("action", choices=["inject", "restore", "status"])
    p.add_argument("-c", "--config", default=_default_codex_config(),
                   help="Codex config.toml path (default: $CODEX_HOME/config.toml)")
    p.add_argument("--base-url", default="http://127.0.0.1:8080/v1",
                   help="gateway base URL to inject (default: http://127.0.0.1:8080/v1)")
    p.add_argument("--provider", default=DEFAULT_PROVIDER)
    args = p.parse_args(argv)
    if args.action == "inject":
        print(json.dumps(inject(args.config, args.base_url, provider=args.provider), indent=2))
    elif args.action == "restore":
        print(json.dumps(restore(args.config), indent=2))
    else:
        print(json.dumps(status(args.config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
