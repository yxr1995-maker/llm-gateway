"""冒烟测试：solve 别名 -> cascade 派发 -> 难度路由器选 tier（全链路）。
不调用真实上游（_call_text 桩掉）。运行：.venv/bin/python smoke_cascade_router.py
"""
import asyncio
from app import cascade
from app.config import GatewayConfig
from app.cascade import is_cascade

# ---- 桩掉上游，无需网络/key ----
_picked = []
async def _fake_call_text(agent, providers, pool, messages, sem, default_effort, want_stream=False):
    _picked.append(agent.get("name"))
    return {"choices": [{"message": {"role": "assistant", "content": f"FAKE[{agent.get('name')}]"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": agent.get("model")}
cascade._call_text = _fake_call_text

cfg = GatewayConfig("config.yaml")
pipe = dict(cfg.raw["cascade"]["solve"])
pipe["router"] = dict(pipe.get("router") or {})
pipe["router"]["enabled"] = True                       # 测试时开启路由
tiers = pipe["tiers"]

class _Wrap:
    def __init__(self, raw): self.raw = raw
wrap = _Wrap({"cascade": {"solve": pipe}})

# 制造一个明确 >500 字符 + 代码块 + 推理标记的难查询（命中 L2）
_hard = ("```python\n" + "def process(data):\n    return data\n" * 18 +
         "```\n这段代码存在并发竞态条件 bug，请帮我详细分析定位并修复，"
         "要考虑所有边界情况、竞态窗口、失败回滚与安全性，给出逐步推理和完整测试用例。")

CASES = [
    ("easy/你好", "你好", "L0"),
    ("easy/解释", "简单解释一下 REST API 是什么", "L0"),
    ("medium/短代码", "```js\nconsole.log(1)\n```\n这段代码做什么", "L1"),
    ("hard/复杂bug", _hard, "L2"),
]


async def run_case(label, text, expect):
    _picked.clear()
    body = {"model": "cascade:solve", "messages": [{"role": "user", "content": text}], "stream": False}
    await cascade.run_cascade(wrap, {}, None, body, "solve", stream=False, effort=None)
    got = _picked[0] if _picked else None
    ok = got == expect
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<14} -> {got} (期望 {expect})")
    return ok


async def main():
    print("1) 别名展开")
    expanded = cfg.resolve_alias("solve")
    print(f"   resolve_alias('solve') = {expanded!r}  is_cascade = {is_cascade(expanded)}")
    print(f"   resolve_alias('kimi')  = {cfg.resolve_alias('kimi')!r}  (现有别名不受影响)")
    print("2) 路由选 tier（router 开启，上游已桩）")
    results = [await run_case(*c) for c in CASES]
    print(f"\n{'全部 PASS' if all(results) else '存在 FAIL'}  ({sum(results)}/{len(results)})")


if __name__ == "__main__":
    asyncio.run(main())
