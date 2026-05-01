#!/usr/bin/env python3
"""Quick simulation: what if Cap$37 was active all week?"""
import json

with open("data/gold_trade_log.json", encoding="utf-8") as f:
    trades = json.load(f)

week_start = "2026-04-28"
closes = [t for t in trades
          if t.get("action") in ("CLOSE", "CLOSE_DETECTED")
          and t.get("time", "") >= week_start]

total_pnl = 0
cap_trades = []
non_cap_trades = []

for t in closes:
    pnl = t.get("profit", 0)
    total_pnl += pnl
    reason = t.get("reason", "")
    if "MaxLoss Cap" in reason:
        cap_trades.append(t)
    else:
        non_cap_trades.append(t)

print("=" * 70)
print(f"本周交易汇总 (4/28 - 4/30)  总笔数: {len(closes)}")
print("=" * 70)

non_cap_total = sum(t["profit"] for t in non_cap_trades)
print(f"\n非 Cap 交易 ({len(non_cap_trades)} 笔):  PnL = ${non_cap_total:.2f}")

print(f"\nMaxLoss Cap 截断的交易 ({len(cap_trades)} 笔):")
print("-" * 70)

actual_cap_total = 0
cap37_total = 0

for t in cap_trades:
    pnl = t["profit"]
    actual_loss = abs(pnl)
    lots = t.get("lots", 0)
    time_str = t["time"][:16]
    strat = t.get("strategy", "?")
    direction = t.get("direction", "?")
    actual_cap_total += pnl

    if actual_loss <= 37:
        cap37_pnl = pnl
        note = "Cap37 也触发(同结果)"
    else:
        cap37_pnl = -37.0 * lots / 0.05 if lots != 0.05 else -37.0
        # Actually: Cap is absolute $, not per-lot. Just use -37
        cap37_pnl = -37.0
        note = f"原亏${actual_loss:.0f} -> Cap37省${actual_loss - 37:.0f}"

    cap37_total += cap37_pnl
    print(f"  {time_str} | {strat:>8} {direction:>4} {lots}lot | "
          f"实际: ${pnl:>8.2f} | Cap37: ${cap37_pnl:>8.2f} | {note}")

print("-" * 70)
print(f"  Cap交易实际合计:  ${actual_cap_total:.2f}")
print(f"  Cap37模拟合计:    ${cap37_total:.2f}")

sim_total = non_cap_total + cap37_total
print(f"\n{'='*70}")
print(f"  实际总 PnL:       ${total_pnl:>10.2f}")
print(f"  Cap$37 模拟 PnL:  ${sim_total:>10.2f}")
print(f"  差额:             ${sim_total - total_pnl:>10.2f}")
print(f"{'='*70}")
