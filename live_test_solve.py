"""Live 测试：单进程内启动网关(router 开启) -> 发 model=solve -> 校验路由日志+真实上游响应。
会真实调用最便宜上游（易查询 -> L0 agnes）。运行：.venv/bin/python live_test_solve.py
"""
import os, sys, time, json, subprocess, urllib.request, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, ".venv", "bin", "python")
TEST_CFG = "/tmp/gw_test_config.yaml"
LOG = "/tmp/gw_live.log"
PORT = 8099
BASE = f"http://127.0.0.1:{PORT}"


def make_test_cfg():
    c = yaml.safe_load(open(os.path.join(HERE, "config.yaml"), encoding="utf-8"))
    c["server"]["host"] = "127.0.0.1"
    c["server"]["port"] = PORT
    c["server"]["master_key"] = ""          # 本地测试免鉴权
    c["server"]["trust_loopback"] = True
    c["cascade"]["solve"]["router"]["enabled"] = True
    yaml.safe_dump(c, open(TEST_CFG, "w"), allow_unicode=True, sort_keys=False)


def wait_health(timeout=30):
    for _ in range(timeout):
        try:
            with urllib.request.urlopen(BASE + "/v1/models", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def send_solve(text):
    body = json.dumps({"model": "solve", "messages": [{"role": "user", "content": text}], "stream": False}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())


def main():
    make_test_cfg()
    open(LOG, "w").close()
    env = dict(os.environ, GATEWAY_CONFIG=TEST_CFG)
    proc = subprocess.Popen([VENV, "-m", "app.main"], cwd=HERE, env=env,
                            stdout=open(LOG, "a"), stderr=subprocess.STDOUT)
    print(f"gateway pid={proc.pid}, waiting health...")
    try:
        if not wait_health():
            print("[FAIL] gateway not healthy"); print(open(LOG).read()[-1500:]); return
        print("[OK] gateway healthy on", BASE)
        print("sending model=solve (易查询, 预期 L0/agnes)...")
        resp = send_solve("你好，一句话自我介绍")
        print("  响应 model :", resp.get("model"))
        print("  内容       :", (resp.get("choices", [{}])[0].get("message", {}).get("content", ""))[:200])
        print("  usage      :", resp.get("usage"))
        log = open(LOG).read()
        rlines = [l for l in log.splitlines() if "difficulty-router" in l]
        print("  路由日志    :", rlines[-1].split("llm-gateway.cascade:")[1].strip() if rlines else "(无)")
        ok = bool(rlines) and bool(resp.get("choices"))
        print("\nLIVE TEST:", "PASS ✅" if ok else "需检查 ⚠️")
    except Exception as e:
        print("[ERROR]", repr(e))
        print(open(LOG).read()[-1500:])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        if os.path.exists(TEST_CFG):
            os.remove(TEST_CFG)
        print("(gateway stopped, temp config removed)")


if __name__ == "__main__":
    main()
