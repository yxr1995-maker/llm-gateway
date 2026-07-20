"""端到端冒烟测试（SPEC 验收第 2 条）。

启动本地 mock 上游 + 网关，黑盒验证：鉴权、模型列表、三种寻址、
三种 provider 协议转换、流式、故障转移、502、Responses 透传、
embeddings 透传、管理 API。无需真实 API key。

用法：.venv/bin/python tests/smoke_test.py
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
    models: ["gpt-4o", "text-embedding-3-small"]
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
            print("FATAL mock 未启动"); return 2
        if not wait_port(GW_PORT):
            print("FATAL 网关未启动"); return 2
        time.sleep(0.5)
        h = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

        # 1-2 鉴权
        check("无鉴权 401", httpx.get(f"{BASE}/models").status_code == 401)
        check("错误 key 401",
              httpx.get(f"{BASE}/models", headers={"Authorization": "Bearer wrong"}).status_code == 401)

        # 3 模型列表
        r = httpx.get(f"{BASE}/models", headers=h)
        check("/v1/models 200", r.status_code == 200)
        ids = {x["id"] for x in r.json()["data"]}
        check("models 含别名", {"gpt", "claude", "gem"} <= ids, str(ids))
        check("models 含底层模型",
              {"gpt-4o", "claude-sonnet-4-5", "gemini-2.5-flash"} <= ids, str(ids))

        # 4 openai 非流式（首个 key sk-fake1 故障 -> 自动转移 sk-good1）
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "gpt", "messages": [{"role": "user", "content": "hi"}]}).json()
        check("openai 别名 chat + 故障转移",
              r["choices"][0]["message"]["content"] == "Hello from mock openai!")

        # 5 anthropic 转换
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "claude", "messages": [{"role": "user", "content": "hi"}]}).json()
        check("anthropic 转换 chat",
              r["choices"][0]["message"]["content"] == "Hi from mock claude!")

        # 6 gemini 转换
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "gem", "messages": [{"role": "user", "content": "hi"}]}).json()
        check("gemini 转换 chat",
              r["choices"][0]["message"]["content"] == "Yo from mock gemini!")

        # 7 流式
        with httpx.stream("POST", f"{BASE}/chat/completions", headers=h,
                          json={"model": "gpt", "messages": [{"role": "user", "content": "hi"}],
                                "stream": True}) as s:
            txt = b"".join(s.iter_bytes()).decode("utf-8", "ignore")
        check("流式含 [DONE]", "data: [DONE]" in txt, txt[:100])
        check("流式含 chunk", "chat.completion.chunk" in txt)

        # 8 未知模型 404
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "no-such", "messages": [{"role": "user", "content": "hi"}]})
        check("未知模型 404", r.status_code == 404, str(r.status_code))

        # 9 provider/model 显式
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
        check("provider/model 显式寻址", r.status_code == 200, str(r.status_code))

        # 10 Responses 透传（openai_like）
        r = httpx.post(f"{BASE}/responses", headers=h,
                       json={"model": "gpt", "input": "hi"}).json()
        check("responses 透传 status=completed", r.get("status") == "completed", str(r)[:120])

        # 11 Responses 在 anthropic 不支持 -> 501
        r = httpx.post(f"{BASE}/responses", headers=h, json={"model": "claude", "input": "hi"})
        check("responses anthropic 501", r.status_code == 501, str(r.status_code))

        # 12 embeddings 透传
        r = httpx.post(f"{BASE}/embeddings", headers=h,
                       json={"model": "openai/text-embedding-3-small", "input": "hello"})
        check("embeddings 200", r.status_code == 200, str(r.status_code) + r.text[:120])
        if r.status_code == 200:
            d = r.json()
            check("embeddings 结构", isinstance(d.get("data"), list) and "embedding" in d["data"][0])

        # 13 全部 key 失败 -> 502（boom 模型恒 500）
        r = httpx.post(f"{BASE}/chat/completions", headers=h,
                       json={"model": "openai/boom", "messages": [{"role": "user", "content": "hi"}]})
        check("全 key 失败 502", r.status_code == 502, str(r.status_code))

        # 14 admin health
        r = httpx.get(f"{ADMIN}/health", headers=h)
        check("admin health 200", r.status_code == 200)
        if r.status_code == 200:
            ok = {k: v.get("ok") for k, v in r.json().items()}
            check("admin health 三家 ok", all(ok.values()), str(ok))

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
