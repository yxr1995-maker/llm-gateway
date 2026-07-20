"""End-to-end smoke test (SPEC acceptance item 2).

Starts a local mock upstream + gateway and black-box verifies: auth, model list, three resolution forms, 
three provider protocol conversions, streaming, failover, 502, Responses passthrough, 
embeddings passthrough, admin API. No real API key needed.

Usage: .venv/bin/python tests/smoke_test.py
"""
from __future__ import annotations

import httpx
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import tempfile
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
VENV_PY = str(REPO / ".venv" / "bin" / "python")
MOCK_PORT = 9100
GW_PORT = 18099
BASE = f"http://127.0.0.1:{GW_PORT}/v1"
ADMIN = f"http://127.0.0.1:{GW_PORT}/admin/api"
KEY = "sk-test-master"

results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append((name, cond, detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"  -> {detail}"))


def wait_port(port: int, timeout: float = 15.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), 1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="gwtest-")
    cfg = pathlib.Path(tmp) / "config.yaml"
    cfg.write_text(f"""
server:
  host: 127.0.0.1
  port: {GW_PORT}
  master_key: "{KEY}"
providers:
  openai:
    type: openai_like
    base_url: http://127.0.0.1:{MOCK_PORT}/v1
    keys: ["sk-fake1", "sk-good1"]
    models:
      - name: gpt-4o
        context: 128000
      - text-embedding-3-small
    supports_responses: true
  anthropic:
    type: anthropic
    base_url: http://127.0.0.1:{MOCK_PORT}
    keys: ["sk-ant-fake1", "sk-ant-good1"]
    models: ["claude-sonnet-4-5"]
  gemini:
    type: gemini
    base_url: http://127.0.0.1:{MOCK_PORT}
    keys: ["AIzaFake1", "AIzaGood1"]
    models: ["gemini-2.5-flash"]
aliases:
  gpt: openai/gpt-4o
  claude: anthropic/claude-sonnet-4-5
  gem: gemini/gemini-2.5-flash
moa:
  default:
    proposers:
      - provider: openai
        model: gpt-4o
        reasoning_effort: low
      - provider: anthropic
        model: claude-sonnet-4-5
        reasoning_effort: default
    default_reasoning_effort: high
    aggregator:
      provider: openai
      model: gpt-4o
      reasoning_effort: medium
rate_limit:
  requests_per_minute: 0
""")

    mock = subprocess.Popen([VENV_PY, str(REPO / "tests" / "mock_upstream.py")])
    gw = subprocess.Popen(
        [VENV_PY, "-m", "app.main"],
        cwd=tmp,
        env={**os.environ, "GATEWAY_CONFIG": str(cfg), "PYTHONPATH": str(REPO)},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_port(MOCK_PORT):
            print("FATAL mock not started"); return 2
        if not wait_port(GW_PORT):
            print("FATAL gateway not started"); return 2
        time.sleep(0.5)
        h = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

        # 1-2 auth
        check("no-auth 401", httpx.get(f"{BASE}/models").status_code == 401)
        check("wrong key 401",
              httpx.get(f"{BASE}/models", headers={"Authorization": "Bearer wrong"}).status_code == 401)

        # 3 model list
        r = httpx.get(f"{BASE}/models", headers=h)
        check("/v1/models 200", r.status_code == 200)
        ids = {x["id"] for x in r.json()["data"]}
        check("models include aliases", {"gpt", "claude", "gem"} <= ids, str(ids))
        check("models include base models",
              {"gpt-4o", "claude-sonnet-4-5", "gemini-2.5-flash"} <= ids, str(ids))
        ctx = {x["id"]: x.get("context_window") for x in r.json()["data"]}
        check("model context_window exposed", ctx.get("gpt-4o") == 128000, str(ctx))
        check("models include moa pipeline", "moa:default" in ids, str(ids))

        # 4 openai non-stream (first key sk-fake1 fails -> auto-failover sk-good1)
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "gpt", "messages": [{"role": "user", "content": "hi"}]}).json()
        check("openai alias chat + failover",
              r["choices"][0]["message"]["content"] == "Hello from mock openai!")

        # 5 anthropic convert
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "claude", "messages": [{"role": "user", "content": "hi"}]}).json()
        check("anthropic conversion chat",
              r["choices"][0]["message"]["content"] == "Hi from mock claude!")

        # 6 gemini convert
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "gem", "messages": [{"role": "user", "content": "hi"}]}).json()
        check("gemini conversion chat",
              r["choices"][0]["message"]["content"] == "Yo from mock gemini!")

        # 7 streaming
        with httpx.stream("POST", f"{BASE}/chat/completions", headers=h,
                          json={"model": "gpt", "messages": [{"role": "user", "content": "hi"}],
                                "stream": True}) as s:
            txt = b"".join(s.iter_bytes()).decode("utf-8", "ignore")
        check("stream has [DONE]", "data: [DONE]" in txt, txt[:100])
        check("stream has chunk", "chat.completion.chunk" in txt)

        # 8 unknown model 404
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "no-such", "messages": [{"role": "user", "content": "hi"}]})
        check("unknown model 404", r.status_code == 404, str(r.status_code))

        # 9 provider/model explicit
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
        check("provider/model explicit", r.status_code == 200, str(r.status_code))

        # 10 Responses passthrough (openai_like native supports_responses)
        r = httpx.post(f"{BASE}/responses", headers=h,
                       json={"model": "gpt", "input": "hi"}).json()
        check("responses passthrough completed", r.get("status") == "completed", str(r)[:120])

        # 10b Responses converted to reach the anthropic upstream
        r = httpx.post(f"{BASE}/responses", headers=h,
                       json={"model": "claude", "input": "hi"}).json()
        check("responses->anthropic converted completed", r.get("status") == "completed", str(r)[:120])

        # 10c Responses converted to reach the gemini upstream
        r = httpx.post(f"{BASE}/responses", headers=h,
                       json={"model": "gem", "input": "hi"}).json()
        check("responses->gemini converted completed", r.get("status") == "completed", str(r)[:120])

        # 10d Responses streaming (convert path, anthropic)
        with httpx.stream("POST", f"{BASE}/responses", headers=h,
                          json={"model": "claude", "input": "hi", "stream": True}) as st:
            txt = b"".join(st.iter_bytes()).decode("utf-8", "ignore")
        check("responses stream has completed", "response.completed" in txt, txt[:120])

        # 10e /v1/messages (Anthropic input) reaches the openai upstream
        r = httpx.post(f"{BASE}/messages", headers=h,
                       json={"model": "gpt", "max_tokens": 50,
                             "messages": [{"role": "user", "content": "hi"}]})
        check("messages->openai 200", r.status_code == 200, str(r.status_code) + r.text[:100])
        if r.status_code == 200:
            d = r.json()
            check("messages returns anthropic format",
                  d.get("type") == "message" and d.get("content"), str(d)[:120])

        # 10f /v1/messages reaches the anthropic upstream (anthropic->chat->anthropic)
        r = httpx.post(f"{BASE}/messages", headers=h,
                       json={"model": "claude", "max_tokens": 50,
                             "messages": [{"role": "user", "content": "hi"}]})
        check("messages->anthropic 200", r.status_code == 200, str(r.status_code))

        # 11 tool repair: tool-bad upstream returns invalid arguments; the gateway should normalize to valid JSON
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "openai/tool-bad", "messages": [{"role": "user", "content": "x"}]})
        check("tool-bad 200", r.status_code == 200, str(r.status_code))
        if r.status_code == 200:
            args = r.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            import json as _j
            try:
                _j.loads(args); ok = True
            except Exception:
                ok = False
            check("tool arguments repaired to valid JSON", ok, args[:80])

        # 11b broken stream stays open: stream-break sends one chunk then drops; the gateway should synthesize [DONE]
        with httpx.stream("POST", f"{BASE}/chat/completions", headers=h,
                          json={"model": "openai/stream-break", "stream": True,
                                "messages": [{"role": "user", "content": "x"}]}) as st:
            txt = b"".join(st.iter_bytes()).decode("utf-8", "ignore")
        check("broken stream synthesizes [DONE]", "data: [DONE]" in txt, txt[:120])

        # 11c MOA (chat face, non-stream)
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "moa:default",
                             "messages": [{"role": "user", "content": "hi"}]})
        check("MOA chat 200", r.status_code == 200, str(r.status_code) + r.text[:120])
        if r.status_code == 200:
            c = r.json()["choices"][0]["message"].get("content")
            check("MOA has synthesized content", bool(c), str(c)[:80])

        # 11d MOA (responses face, non-stream)
        r = httpx.post(f"{BASE}/responses", headers=h,
                       json={"model": "moa:default", "input": "hi"})
        check("MOA responses 200 completed", r.status_code == 200 and r.json().get("status") == "completed",
              str(r.status_code) + r.text[:100])

        # 11e MOA (chat face, streaming)
        with httpx.stream("POST", f"{BASE}/chat/completions", headers=h,
                          json={"model": "moa:default", "stream": True,
                                "messages": [{"role": "user", "content": "hi"}]}) as st:
            txt = b"".join(st.iter_bytes()).decode("utf-8", "ignore")
        check("MOA stream has [DONE]", "data: [DONE]" in txt, txt[:100])

        # 11f MOA unknown pipeline -> 404
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "moa:nope", "messages": [{"role": "user", "content": "hi"}]})
        check("MOA unknown pipeline 404", r.status_code == 404, str(r.status_code))

        # 12 embeddings passthrough
        r = httpx.post(f"{BASE}/embeddings", headers=h,
                       json={"model": "openai/text-embedding-3-small", "input": "hello"})
        check("embeddings 200", r.status_code == 200, str(r.status_code) + r.text[:120])
        if r.status_code == 200:
            d = r.json()
            check("embeddings shape", isinstance(d.get("data"), list) and "embedding" in d["data"][0])

        # 13 all keys fail -> 502 (boom model always 500)
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "openai/boom", "messages": [{"role": "user", "content": "hi"}]})
        check("all keys fail 502", r.status_code == 502, str(r.status_code))

        # 14 admin health
        r = httpx.get(f"{ADMIN}/health", headers=h)
        check("admin health 200", r.status_code == 200)
        if r.status_code == 200:
            ok = {k: v.get("ok") for k, v in r.json().items()}
            check("admin health all ok", all(ok.values()), str(ok))

        # 14a admin auth: no key should 401 when master_key is non-empty
        check("admin no-key 401", httpx.get(f"{ADMIN}/config").status_code == 401)

        # 14b config edit: PUT /admin/api/config adds alias, hot-reload
        raw = httpx.get(f"{ADMIN}/config/raw", headers=h).json()
        raw.setdefault("aliases", {})["extra"] = "openai/gpt-4o"
        r = httpx.put(f"{ADMIN}/config", headers=h, json=raw)
        check("PUT config 200", r.status_code == 200, str(r.status_code) + r.text[:120])
        ids2 = {x["id"] for x in httpx.get(f"{BASE}/models", headers=h).json()["data"]}
        check("config edit hot-reload (new alias visible)", "extra" in ids2, str(ids2))

        # 15 admin summary
        r = httpx.get(f"{ADMIN}/usage/summary?days=1", headers=h)
        check("admin summary calls>0", r.json().get("totals", {}).get("calls", 0) > 0, r.text[:120])
    finally:
        for p in (gw, mock):
            try:
                p.terminate(); p.wait(5)
            except Exception:
                try: p.kill()
                except Exception: pass

    npass = sum(1 for _, c, _ in results if c)
    print(f"\n=== {npass}/{len(results)} passed ===")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
