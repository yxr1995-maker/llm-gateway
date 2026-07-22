"""Tests for app.codex_inject (inject/restore/status on temp config copies).

Run: PYTHONPATH=. .venv/bin/python tests/test_codex_inject.py
"""
from __future__ import annotations
import pathlib, tempfile
from app import codex_inject as ci

_results: list = []
def check(name, cond, detail=""):
    _results.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"  -> {detail}"))

SAMPLE = '''model_provider = "custom"
model = "glm-5.2"
approval_policy = "never"

[[skills.config]]
path = "/x/SKILL.md"
enabled = false

[model_providers.custom]
name = "ark_agentplan"
base_url = "http://127.0.0.1:15721/v1"
wire_api = "responses"
requires_openai_auth = true
experimental_bearer_token = "PROXY_MANAGED"

[features]
goals = true
'''


def _write(tmp, text):
    p = pathlib.Path(tmp) / "config.toml"
    p.write_bytes(text.encode("utf-8"))
    return p


def test_inject_swaps_provider_and_preserves_rest():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(tmp, SAMPLE)
        r = ci.inject(p, "http://127.0.0.1:8080/v1")
        check("inject ok", r["ok"] and r["backed_up"], r)
        t = p.read_bytes().decode("utf-8")
        check("model_provider -> llm-gateway", 'model_provider = "llm-gateway"' in t, t.splitlines()[0])
        check("old custom provider table kept", "[model_providers.custom]" in t)
        check("new gateway table added", "[model_providers.llm-gateway]" in t)
        check("base_url = 8080", 'base_url = "http://127.0.0.1:8080/v1"' in t)
        check("wire_api responses", 'wire_api = "responses"' in t)
        check("marker present", ci.MARKER_START in t)
        check("model kept", 'model = "glm-5.2"' in t)
        check("skills kept", "[[skills.config]]" in t)
        check("features kept", "[features]" in t)
        check("single model_provider line", t.count("model_provider = ") == 1, str(t.count("model_provider = ")))


def test_idempotent_reinject_no_duplicates():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(tmp, SAMPLE)
        ci.inject(p, "http://127.0.0.1:8080/v1")
        r = ci.inject(p, "http://127.0.0.1:9090/v1")
        check("reinject not backed_up again", r["backed_up"] is False, r)
        t = p.read_bytes().decode("utf-8")
        check("single model_provider after reinject", t.count("model_provider = ") == 1)
        check("single gateway table", t.count("[model_providers.llm-gateway]") == 1)
        check("base_url refreshed to 9090", 'base_url = "http://127.0.0.1:9090/v1"' in t)
        check("single marker pair", t.count(ci.MARKER_START) == 2)


def test_restore_from_backup():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(tmp, SAMPLE)
        ci.inject(p, "http://127.0.0.1:8080/v1")
        r = ci.restore(p)
        check("restore ok via backup", r["ok"] and r["via"] == "backup", r)
        t = p.read_bytes().decode("utf-8")
        check("restored model_provider=custom", 'model_provider = "custom"' in t, t.splitlines()[0])
        check("restored base_url=15721", 'base_url = "http://127.0.0.1:15721/v1"' in t)
        check("no gateway markers after restore", ci.MARKER_START not in t)
        check("no gateway table after restore", "[model_providers.llm-gateway]" not in t)


def test_status():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(tmp, SAMPLE)
        s0 = ci.status(p)
        check("status not injected", s0["injected"] is False and s0["backup_exists"] is False, s0)
        ci.inject(p, "http://127.0.0.1:8080/v1")
        s1 = ci.status(p)
        check("status injected", s1["injected"] is True, s1)
        check("status base_url", s1["base_url"] == "http://127.0.0.1:8080/v1", s1)
        check("status provider", s1["provider"] == "llm-gateway", s1)
        check("status backup exists", s1["backup_exists"] is True, s1)


def test_eol_preserved_crlf():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(tmp, SAMPLE.replace("\n", "\r\n"))
        ci.inject(p, "http://127.0.0.1:8080/v1")
        t = p.read_bytes().decode("utf-8")
        check("CRLF preserved", t.count("\r\n") == t.count("\n"), "expected all newlines to be CRLF")
        check("no bare LF mixing", "\n" not in t.replace("\r\n", ""), "found a bare LF not part of CRLF")


def test_inject_on_missing_config_creates_it():
    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "config.toml"
        r = ci.inject(p, "http://127.0.0.1:8080/v1")
        check("inject on missing ok", r["ok"] and r["backed_up"] is False, r)
        t = p.read_bytes().decode("utf-8")
        check("created with model_provider", 'model_provider = "llm-gateway"' in t)
        check("created with table", "[model_providers.llm-gateway]" in t)


def main():
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]:
        fn()
    failed = [r for r in _results if not r[1]]
    print(f"\n=== {len(_results)-len(failed)}/{len(_results)} passed ===")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
