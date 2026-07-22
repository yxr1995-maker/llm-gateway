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
  multimodal:
    max_concurrency: 2
    default_reasoning_effort: low
    stages:
      - provider: openai
        model: gpt-4o
        modality: image
        prompt: a cat
      - provider: openai
        model: gpt-4o
        modality: vision
      - provider: openai
        model: gpt-4o
        modality: text
  imggen:
    stages:
      - provider: openai
        model: gpt-4o
        modality: image
        prompt: a dog
  opt:
    dedup: true
    early_stop: true
    early_stop_similarity: 0.5
    compress:
      enabled: true
      max_chars: 50
    proposers:
      - provider: openai
        model: gpt-4o
      - provider: openai
        model: gpt-4o
    aggregator:
      provider: openai
      model: gpt-4o
planner_worker:
  solve:
    planner:
      provider: openai
      model: gpt-4o
      reasoning_effort: high
    workers:
      - provider: openai
        model: gpt-4o
      - provider: anthropic
        model: claude-sonnet-4-5
    max_rounds: 1
    max_concurrency: 2
cascade:
  solve:
    router:
      provider: openai
      model: gpt-4o
    tiers:
      - name: L0
        provider: openai
        model: gpt-4o
        reasoning_effort: none
      - name: L1
        provider: openai
        model: gpt-4o
        reasoning_effort: low
      - name: L2
        provider: anthropic
        model: claude-sonnet-4-5
        reasoning_effort: high
    consensus_k: 1
  solve_strict:
    router:
      provider: openai
      model: gpt-4o
    tiers:
      - name: L0
        provider: openai
        model: gpt-4o
        reasoning_effort: none
      - name: L2
        provider: anthropic
        model: claude-sonnet-4-5
        reasoning_effort: high
    consensus_k: 1
    t1_verify: true
cache:
  enabled: true
  ttl: 60
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
        check("models include pw pipeline", "pw:solve" in ids, str(ids))
        check("models include cascade", "cascade:solve" in ids, str(ids))

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

        # 11g planner-worker: planner decomposes -> workers execute -> synthesize
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "pw:solve", "messages": [{"role": "user", "content": "build a demo"}]})
        check("PW chat 200", r.status_code == 200, str(r.status_code)+r.text[:120])
        if r.status_code == 200:
            check("PW synthesized output", r.json()["choices"][0]["message"].get("content") == "SYNTHESIZED", r.text[:160])

        # 11g2 planner-worker via responses face
        r = httpx.post(f"{BASE}/responses", headers=h, json={"model": "pw:solve", "input": "build a demo"})
        check("PW responses completed", r.status_code == 200 and r.json().get("status") == "completed", str(r.status_code)+r.text[:120])

        # 11g3 unknown pw pipeline -> 404
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "pw:nope", "messages": [{"role": "user", "content": "x"}]})
        check("PW unknown pipeline 404", r.status_code == 404, str(r.status_code))

        # 11h cascade is effort-driven: effort=low -> L0(openai); no effort -> default top(anthropic)
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "cascade:solve", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "low"})
        check("cascade effort=low -> L0", r.status_code == 200 and r.json()["choices"][0]["message"].get("content") == "Hello from mock openai!", r.text[:160])
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "cascade:solve", "messages": [{"role": "user", "content": "hi"}]})
        check("cascade no effort -> top T1", r.status_code == 200 and r.json()["choices"][0]["message"].get("content") == "Hi from mock claude!", r.text[:160])
        r = httpx.post(f"{BASE}/responses", headers=h,
                       json={"model": "cascade:solve", "input": "hi", "reasoning": {"effort": "low"}})
        check("cascade responses effort=low completed", r.status_code == 200 and r.json().get("status") == "completed", r.text[:120])

        # 11h1 cascade streams the chosen tier
        with httpx.stream("POST", f"{BASE}/chat/completions", headers=h,
                          json={"model": "cascade:solve", "stream": True, "reasoning_effort": "low",
                                "messages": [{"role": "user", "content": "hi"}]}) as st:
            txt = b"".join(st.iter_bytes()).decode("utf-8", "ignore")
        check("cascade stream has chunk+DONE", "chat.completion.chunk" in txt and "data: [DONE]" in txt, txt[:100])

        # 11h2 cascade unknown -> 404
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "cascade:nope", "messages": [{"role": "user", "content": "x"}]})
        check("cascade unknown 404", r.status_code == 404, str(r.status_code))

        # 11i result cache: identical call -> second is HIT
        body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "cachecheck"}]}
        r1 = httpx.post(f"{BASE}/chat/completions", headers=h, json=body)
        r2 = httpx.post(f"{BASE}/chat/completions", headers=h, json=body)
        check("cache first MISS", r1.headers.get("x-cache") == "MISS", r1.headers.get("x-cache"))
        check("cache second HIT", r2.headers.get("x-cache") == "HIT", r2.headers.get("x-cache"))

        # 11j cascade t1_verify: effort=low (non-top) -> T1 (anthropic) rewrites
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "cascade:solve_strict", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "low"})
        check("cascade t1_verify 200", r.status_code == 200, str(r.status_code)+r.text[:120])
        if r.status_code == 200:
            check("cascade t1_verify used T1", r.json()["choices"][0]["message"].get("content") == "Hi from mock claude!", r.text[:160])

        # 11k MOA dedup + early_stop + compress (two identical proposers)
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "moa:opt", "messages": [{"role": "user", "content": "hi"}]})
        check("MOA opt (dedup/early_stop/compress) 200", r.status_code == 200, str(r.status_code)+r.text[:120])

        # 11f1 multimodal staged pipeline (image -> vision -> text)
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "moa:multimodal", "messages": [{"role": "user", "content": "describe"}]})
        check("MOA multimodal staged 200", r.status_code == 200, str(r.status_code)+r.text[:120])
        if r.status_code == 200:
            check("MOA multimodal has content", bool(r.json()["choices"][0]["message"].get("content")))

        # 11f2 image-gen pipeline returns markdown image
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "moa:imggen", "messages": [{"role": "user", "content": "x"}]})
        check("MOA imggen 200", r.status_code == 200, str(r.status_code)+r.text[:120])
        if r.status_code == 200:
            check("MOA imggen markdown image", "![image]" in r.json()["choices"][0]["message"].get("content",""), r.text[:120])

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

        # 13a caller error (4xx non-429) -> surface upstream 400, not a misleading 502,
        # and do not burn the whole key pool retrying a malformed request (runs before the
        # boom test, which cools every key via transient backoff)
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "openai/badreq", "messages": [{"role": "user", "content": "hi"}]})
        check("caller error 400 (not 502)", r.status_code == 400, str(r.status_code) + r.text[:120])

        # 13b all keys fail -> 502 (boom model always 500)
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "openai/boom", "messages": [{"role": "user", "content": "hi"}]})
        check("all keys fail 502", r.status_code == 502, str(r.status_code))

        # 14 admin health
        r = httpx.get(f"{ADMIN}/health", headers=h)
        check("admin health 200", r.status_code == 200)
        if r.status_code == 200:
            ok = {k: v.get("ok") for k, v in r.json().items()}
            check("admin health all ok", all(ok.values()), str(ok))

        # 14c key-pool runtime state: openai keys were cooled by the boom test (transient backoff)
        r = httpx.get(f"{ADMIN}/pool", headers=h)
        check("admin pool 200", r.status_code == 200, str(r.status_code))
        if r.status_code == 200:
            pdata = r.json()
            check("pool has openai", "openai" in pdata, str(list(pdata.keys())))
            oi = pdata.get("openai", {})
            check("pool openai cooled>0 after boom", oi.get("cooled", 0) > 0, str(oi))
            check("pool exposes needs_reauth list", isinstance(oi.get("needs_reauth"), list), str(oi))

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
