#!/usr/bin/env python3
"""Compute current CPT (strong-model call ratio) from data/usage.db and project
RouteLLM-style savings. Runnable now (reports empty DB); useful once traffic flows.

Usage:
  python3 analyze_cpt.py
  python3 analyze_cpt.py --days 7 --target-cpt 0.30

Model -> tier mapping defaults to your real providers. Edit PRICE_TABLE with your
account's actual list prices (values below are APPROXIMATE public prices, drift-prone).
"""
from __future__ import annotations
import argparse, sqlite3, os

# tier: 0=weak(cheap), 1=mid, 2=strong(T1). strong_tier>=2.
MODEL_TIER = {
    "agnes-2.0-flash": 0,
    "deepseek-v4-flash": 0,
    "kimi-for-coding": 1, "kimi-for-coding-highspeed": 1, "k3": 1,
    "glm-5.2": 2, "kimi-k2.7-code": 2, "minimax-m3": 2, "deepseek-v4-pro": 2,
}
STRONG_TIER = 2  # tier index considered "strong"

# APPROXIMATE USD per 1M tokens (in, out). VERIFY against your real bill.
PRICE_TABLE = {
    "agnes-2.0-flash": (0.10, 0.40),
    "deepseek-v4-flash": (0.14, 0.55),
    "kimi-for-coding": (0.60, 2.50),
    "k3": (0.60, 2.50),
    "glm-5.2": (0.50, 2.00),
    "kimi-k2.7-code": (0.60, 2.50),
    "minimax-m3": (1.00, 4.00),
    "deepseek-v4-pro": (1.40, 5.50),
}


def tier_of(model: str) -> int:
    return MODEL_TIER.get(model, 1)


def cost_of(model: str, pin: int, pout: int) -> float:
    pi, po = PRICE_TABLE.get(model, (0.5, 2.0))
    return pin / 1e6 * pi + pout / 1e6 * po


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/usage.db")
    ap.add_argument("--days", type=int, default=0, help="0 = all time")
    ap.add_argument("--target-cpt", type=float, default=0.30, help="projected strong-call ratio")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"[!] DB not found: {args.db}"); return
    con = sqlite3.connect(args.db)
    where = "" if not args.days else f"WHERE ts >= strftime('%s','now','-{args.days} days')"
    rows = con.execute(
        f"SELECT model, COUNT(*), SUM(COALESCE(prompt_tokens,0)), SUM(COALESCE(completion_tokens,0)) "
        f"FROM usage {where} GROUP BY model ORDER BY 2 DESC"
    ).fetchall()

    print("=" * 64)
    print("LLM 网关 CPT 分析" + (f" (近 {args.days} 天)" if args.days else " (全部)"))
    print("=" * 64)
    if not rows or (len(rows) == 1 and rows[0][1] <= 1):
        print("[i] usage.db 基本为空，暂无真实流量可算 CPT。")
        print("[i] 流量接入后重跑此脚本即可得到实际强模型调用占比与节省。")
    else:
        total_calls = sum(r[1] for r in rows)
        strong_calls = sum(r[1] for r in rows if tier_of(r[0]) >= STRONG_TIER)
        cpt = strong_calls / total_calls if total_calls else 0
        print(f"{'model':<26}{'calls':>8}{'tier':>6}{'cost$':>10}")
        tot_cost = 0.0
        for m, n, pin, pout in rows:
            c = cost_of(m, pin or 0, pout or 0); tot_cost += c
            print(f"{m:<26}{n:>8}{tier_of(m):>6}{c:>10.4f}")
        print("-" * 64)
        print(f"总调用: {total_calls}  强模型调用: {strong_calls}  当前 CPT: {cpt:.1%}")
        print(f"当前估算成本: ${tot_cost:.4f}")
        # projection: if we routed to hit target_cpt, with same total output volume,
        # blended strong cost = target*strong_out_price + (1-target)*weak_out_price
        weak_out = PRICE_TABLE.get("agnes-2.0-flash", (0, 0.4))[1]
        strong_out = PRICE_TABLE.get("glm-5.2", (0, 2.0))[1]
        tot_out = sum((r[3] or 0) for r in rows)
        blended_now = cpt * strong_out + (1 - cpt) * weak_out
        blended_tgt = args.target_cpt * strong_out + (1 - args.target_cpt) * weak_out
        proj_cost = tot_out / 1e6 * blended_tgt
        print(f"\n[投测] 目标 CPT={args.target_cpt:.0%}: 混合输出价 ${blended_tgt:.3f}/M "
              f"(当前 ${blended_now:.3f}/M)")
        save = 1 - blended_tgt / blended_now if blended_now else 0
        print(f"       预计相对当前节省 ~{save:.0%}" + (f"，约 ${tot_cost-proj_cost:.4f}" if tot_cost else ""))
    con.close()


if __name__ == "__main__":
    main()
