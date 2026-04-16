#!/usr/bin/env python3
"""
最近15天 L5.1 交易模拟 — 逐笔输出
数据范围: 2026-03-29 ~ 2026-04-09 (本地数据截止)
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS
from datetime import datetime

print("=" * 80)
print("L5.1 最近15天交易模拟")
print("=" * 80)

data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

# 最近15天
START = "2026-03-25"
END = "2026-04-10"

data_slice = data.slice(START, END)

kw = {**LIVE_PARITY_KWARGS}
s = run_variant(data_slice, "L51_recent15d", verbose=False, spread_cost=0.30, **kw)

print(f"\n--- 总体统计 ({START} -> {END}) ---")
print(f"  交易数: {s['n']}")
print(f"  Sharpe: {s['sharpe']:.2f}")
print(f"  PnL: ${s['total_pnl']:.2f}")
print(f"  胜率: {s['win_rate']:.1%}")
print(f"  MaxDD: ${s['max_dd']:.2f}")

trades = s.get('_trades', [])
if trades:
    print(f"\n--- 逐笔交易明细 ({len(trades)} 笔) ---")
    print(f"  {'#':>3} {'入场时间':<18} {'方向':<5} {'策略':<12} {'入场价':>10} {'出场价':>10} "
          f"{'PnL':>8} {'出场原因':<20} {'持仓bars':>6} {'MFE':>8}")
    print("  " + "-" * 120)
    
    cumulative = 0
    wins = 0
    losses = 0
    for i, t in enumerate(trades):
        pnl = t.pnl
        cumulative += pnl
        if pnl > 0:
            wins += 1
        else:
            losses += 1
        
        direction = getattr(t, 'direction', '?')
        strategy = getattr(t, 'strategy', '?') or '?'
        entry_time = str(getattr(t, 'entry_time', ''))[:16]
        entry_price = getattr(t, 'entry_price', 0)
        exit_price = getattr(t, 'exit_price', 0)
        exit_reason = getattr(t, 'exit_reason', '') or ''
        bars_held = getattr(t, 'bars_held', 0)
        mfe = getattr(t, 'max_favorable', 0) or 0
        
        pnl_str = f"${pnl:>+7.2f}"
        marker = "✅" if pnl > 0 else "❌"
        
        print(f"  {i+1:>3} {entry_time:<18} {direction:<5} {strategy:<12} "
              f"{entry_price:>10.2f} {exit_price:>10.2f} {pnl_str} "
              f"{exit_reason:<20} {bars_held:>6} ${mfe:>7.2f} {marker}")
    
    print("  " + "-" * 120)
    print(f"  累计: ${cumulative:.2f}  |  胜: {wins}  负: {losses}  |  胜率: {wins/(wins+losses):.1%}")
    
    # 按天汇总
    from collections import defaultdict
    daily = defaultdict(lambda: {"pnl": 0, "n": 0, "wins": 0})
    for t in trades:
        day = str(getattr(t, 'entry_time', ''))[:10]
        daily[day]["pnl"] += t.pnl
        daily[day]["n"] += 1
        if t.pnl > 0:
            daily[day]["wins"] += 1
    
    print(f"\n--- 逐日汇总 ---")
    print(f"  {'日期':<12} {'交易数':>6} {'胜率':>6} {'日PnL':>10} {'累计PnL':>10}")
    cum = 0
    for day in sorted(daily.keys()):
        d = daily[day]
        cum += d["pnl"]
        wr = d["wins"] / d["n"] if d["n"] > 0 else 0
        marker = "📈" if d["pnl"] > 0 else "📉" if d["pnl"] < 0 else "➖"
        print(f"  {day:<12} {d['n']:>6} {wr:>6.0%} ${d['pnl']:>+9.2f} ${cum:>+9.2f} {marker}")
    
    # 按出场原因汇总
    from collections import Counter
    exit_reasons = Counter()
    exit_pnl = defaultdict(float)
    for t in trades:
        reason = (getattr(t, 'exit_reason', '') or '').split(':')[0][:25]
        exit_reasons[reason] += 1
        exit_pnl[reason] += t.pnl
    
    print(f"\n--- 出场原因分布 ---")
    print(f"  {'出场原因':<25} {'次数':>5} {'总PnL':>10} {'均PnL':>10}")
    for reason, count in exit_reasons.most_common():
        avg = exit_pnl[reason] / count
        print(f"  {reason:<25} {count:>5} ${exit_pnl[reason]:>+9.2f} ${avg:>+9.2f}")

else:
    print("\n  ⚠️ 没有交易记录")

# 也跑一下 L6 对比
print(f"\n\n{'=' * 80}")
print("L6 候选版本 同期对比")
print("=" * 80)

ULTRA2 = {
    'low': {'trail_act': 0.30, 'trail_dist': 0.06},
    'normal': {'trail_act': 0.20, 'trail_dist': 0.04},
    'high': {'trail_act': 0.08, 'trail_dist': 0.01},
}

L6_kw = {**LIVE_PARITY_KWARGS,
         "regime_config": ULTRA2,
         "trailing_activate_atr": 0.20,
         "trailing_distance_atr": 0.04}

s6 = run_variant(data_slice, "L6_recent15d", verbose=False, spread_cost=0.30, **L6_kw)

print(f"\n--- L6 总体统计 ({START} -> {END}) ---")
print(f"  交易数: {s6['n']}")
print(f"  Sharpe: {s6['sharpe']:.2f}")
print(f"  PnL: ${s6['total_pnl']:.2f}")
print(f"  胜率: {s6['win_rate']:.1%}")
print(f"  MaxDD: ${s6['max_dd']:.2f}")

trades6 = s6.get('_trades', [])
if trades6:
    print(f"\n--- L6 逐笔交易 ({len(trades6)} 笔) ---")
    print(f"  {'#':>3} {'入场时间':<18} {'方向':<5} {'PnL':>8} {'出场原因':<20} {'bars':>5}")
    print("  " + "-" * 70)
    for i, t in enumerate(trades6):
        direction = getattr(t, 'direction', '?')
        entry_time = str(getattr(t, 'entry_time', ''))[:16]
        exit_reason = (getattr(t, 'exit_reason', '') or '')[:20]
        bars_held = getattr(t, 'bars_held', 0)
        marker = "✅" if t.pnl > 0 else "❌"
        print(f"  {i+1:>3} {entry_time:<18} {direction:<5} ${t.pnl:>+7.2f} {exit_reason:<20} {bars_held:>5} {marker}")

print(f"\n--- L5.1 vs L6 对比 ---")
print(f"  {'指标':<12} {'L5.1':>10} {'L6':>10} {'Delta':>10}")
print(f"  {'交易数':<12} {s['n']:>10} {s6['n']:>10} {s6['n']-s['n']:>+10}")
print(f"  {'Sharpe':<12} {s['sharpe']:>10.2f} {s6['sharpe']:>10.2f} {s6['sharpe']-s['sharpe']:>+10.2f}")
print(f"  {'PnL':<12} ${s['total_pnl']:>9.2f} ${s6['total_pnl']:>9.2f} ${s6['total_pnl']-s['total_pnl']:>+9.2f}")
print(f"  {'胜率':<12} {s['win_rate']:>10.1%} {s6['win_rate']:>10.1%}")
print(f"  {'MaxDD':<12} ${s['max_dd']:>9.2f} ${s6['max_dd']:>9.2f}")
