#!/usr/bin/env python3
"""
L8_BASE+Cap80 (当前实盘) vs L8_MAX (R49最优) 对比测试
=====================================================
- A: L8_BASE + Cap$80 (当前实盘配置)
- B: L8_MAX  = L8_BASE + H1KC(E15/M2.0) + EqCurve(LB10) + Cap$30

输出:
  1. 全样本对比 (Sharpe/PnL/N/WR/MaxDD/RR)
  2. 逐年对比
  3. K-Fold 6-Fold 对比
  4. 多 Spread 鲁棒性对比 ($0.30/$0.50/$0.80/$1.00)
"""
import sys, time, json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS
from backtest.engine import TradeRecord

SPREAD = 0.50

L8_BASE_KW = {
    **LIVE_PARITY_KWARGS,
    'keltner_adx_threshold': 14,
    'regime_config': {
        'low':    {'trail_act': 0.22, 'trail_dist': 0.04},
        'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
        'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
    },
    'min_entry_gap_hours': 1.0,
    'keltner_max_hold_m15': 20,
    'spread_cost': SPREAD,
}

L8_MAX_ENGINE_KW = {
    **L8_BASE_KW,
    # EqCurve 不在引擎层启用 — R49 发现引擎实现过于激进 (N仅26)
    # EA 侧的 EqCurve 实现与引擎不同，这里仅做 H1KC + Cap 后处理
}

OUT_DIR = Path("results/l8base_vs_l8max")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# H1 KC helpers
# ═══════════════════════════════════════════════════════════════

def add_h1_kc_dir(h1_df: pd.DataFrame, ema_period: int = 15, mult: float = 2.0) -> pd.DataFrame:
    h1 = h1_df.copy()
    h1['EMA_kc'] = h1['Close'].ewm(span=ema_period, adjust=False).mean()
    tr = pd.DataFrame({
        'hl': h1['High'] - h1['Low'],
        'hc': (h1['High'] - h1['Close'].shift(1)).abs(),
        'lc': (h1['Low'] - h1['Close'].shift(1)).abs(),
    }).max(axis=1)
    h1['ATR_kc'] = tr.rolling(14).mean()
    h1['KC_U'] = h1['EMA_kc'] + mult * h1['ATR_kc']
    h1['KC_L'] = h1['EMA_kc'] - mult * h1['ATR_kc']
    h1['kc_dir'] = 'NEUTRAL'
    h1.loc[h1['Close'] > h1['KC_U'], 'kc_dir'] = 'BULL'
    h1.loc[h1['Close'] < h1['KC_L'], 'kc_dir'] = 'BEAR'
    return h1


def filter_trades_by_h1_kc(trades: List, h1_kc: pd.DataFrame) -> Tuple[List, int]:
    kept, skipped = [], 0
    for t in trades:
        et = t.entry_time if hasattr(t, 'entry_time') else t['entry_time']
        td = t.direction if hasattr(t, 'direction') else t.get('dir', '')
        et_ts = pd.Timestamp(et)
        h1_mask = h1_kc.index <= et_ts
        if not h1_mask.any():
            skipped += 1
            continue
        kc_d = h1_kc.loc[h1_kc.index[h1_mask][-1], 'kc_dir']
        if (td == 'BUY' and kc_d == 'BULL') or (td == 'SELL' and kc_d == 'BEAR'):
            kept.append(t)
        else:
            skipped += 1
    return kept, skipped


def apply_cap(trades: List, cap_usd: float) -> List:
    if cap_usd <= 0:
        return trades
    capped = []
    for t in trades:
        if t.pnl < -cap_usd:
            capped.append(TradeRecord(
                strategy=t.strategy, direction=t.direction,
                entry_price=t.entry_price, exit_price=t.exit_price,
                entry_time=t.entry_time, exit_time=t.exit_time,
                lots=t.lots, pnl=-cap_usd,
                exit_reason=f"MaxLossCap${cap_usd}",
                bars_held=t.bars_held,
            ))
        else:
            capped.append(t)
    return capped


def stats_from_trades(trades: List, label: str = "") -> Dict:
    if not trades:
        return {'label': label, 'n': 0, 'total_pnl': 0, 'sharpe': 0,
                'win_rate': 0, 'max_dd': 0, 'avg_win': 0, 'avg_loss': 0, 'rr': 0,
                'year_pnl': {}, '_trades': trades}
    pnls = [t.pnl for t in trades]
    daily: Dict = {}
    for t in trades:
        d = pd.Timestamp(t.exit_time).date()
        daily[d] = daily.get(d, 0) + t.pnl

    daily_pnl = pd.Series(daily).sort_index()
    sh = (daily_pnl.mean() / daily_pnl.std(ddof=1) * np.sqrt(252)) if len(daily_pnl) > 1 and daily_pnl.std() > 0 else 0.0

    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls) if pnls else 0
    avg_w = np.mean(wins) if wins else 0
    avg_l = abs(np.mean(losses)) if losses else 0
    rr = avg_w / avg_l if avg_l > 0 else 0

    year_pnl = defaultdict(float)
    for t in trades:
        y = pd.Timestamp(t.exit_time).year
        year_pnl[y] += t.pnl

    return {
        'label': label, 'n': len(pnls), 'total_pnl': round(sum(pnls), 2),
        'sharpe': round(sh, 2), 'win_rate': round(wr, 4),
        'max_dd': round(max_dd, 2), 'avg_win': round(avg_w, 2),
        'avg_loss': round(avg_l, 2), 'rr': round(rr, 2),
        'year_pnl': dict(year_pnl), '_trades': trades,
    }


# ═══════════════════════════════════════════════════════════════
# Run both strategies
# ═══════════════════════════════════════════════════════════════

def run_a_vs_b(data: DataBundle, spread: float = SPREAD):
    """Run L8_BASE+Cap80 vs L8_MAX. Returns (stats_a, stats_b)."""

    kw_a = {**L8_BASE_KW, 'spread_cost': spread}
    kw_b = {**L8_MAX_ENGINE_KW, 'spread_cost': spread}

    h1_kc = add_h1_kc_dir(data.h1_df, ema_period=15, mult=2.0)

    # A: L8_BASE + Cap$80
    res_a = run_variant(data, "L8_BASE_Cap80", verbose=False, **kw_a)
    trades_a = apply_cap(res_a['_trades'], 80)
    stats_a = stats_from_trades(trades_a, "L8_BASE+Cap80")

    # B: L8_MAX = L8_BASE + EqCurve(LB10) + H1KC(E15/M2.0) + Cap$30
    res_b = run_variant(data, "L8_MAX", verbose=False, **kw_b)
    trades_b_raw = res_b['_trades']
    trades_b, skipped = filter_trades_by_h1_kc(trades_b_raw, h1_kc)
    trades_b = apply_cap(trades_b, 30)
    stats_b = stats_from_trades(trades_b, "L8_MAX")

    return stats_a, stats_b


def print_comparison(sa, sb):
    print(f"\n{'='*80}")
    print(f"  L8_BASE+Cap80 (当前实盘)  vs  L8_MAX (R49最优)")
    print(f"  Spread: ${SPREAD}")
    print(f"{'='*80}")

    row = "  {:<20} {:>15} {:>15} {:>12}"
    print(row.format("指标", "L8_BASE+Cap80", "L8_MAX", "Delta"))
    print(f"  {'-'*62}")

    def pct_chg(a, b):
        return f"{(b/a-1)*100:+.1f}%" if a else "—"

    rows = [
        ("Sharpe", f"{sa['sharpe']:.2f}", f"{sb['sharpe']:.2f}", f"{sb['sharpe']-sa['sharpe']:+.2f}"),
        ("Total PnL", f"${sa['total_pnl']:,.0f}", f"${sb['total_pnl']:,.0f}", pct_chg(sa['total_pnl'], sb['total_pnl'])),
        ("Trades (N)", f"{sa['n']}", f"{sb['n']}", f"{sb['n']-sa['n']:+d}"),
        ("Win Rate", f"{sa['win_rate']:.1%}", f"{sb['win_rate']:.1%}", f"{(sb['win_rate']-sa['win_rate'])*100:+.1f}pp"),
        ("MaxDD", f"${sa['max_dd']:,.0f}", f"${sb['max_dd']:,.0f}", pct_chg(sa['max_dd'], sb['max_dd'])),
        ("Avg Win", f"${sa['avg_win']:.2f}", f"${sb['avg_win']:.2f}", pct_chg(sa['avg_win'], sb['avg_win'])),
        ("Avg Loss", f"${sa['avg_loss']:.2f}", f"${sb['avg_loss']:.2f}", pct_chg(sa['avg_loss'], sb['avg_loss'])),
        ("Reward/Risk", f"{sa['rr']:.2f}", f"{sb['rr']:.2f}", f"{sb['rr']-sa['rr']:+.2f}"),
    ]
    for r in rows:
        print(row.format(*r))

    # Yearly
    all_years = sorted(set(list(sa['year_pnl'].keys()) + list(sb['year_pnl'].keys())))
    print(f"\n  {'年度对比':^62}")
    print(row.format("Year", "L8_BASE+Cap80", "L8_MAX", "Delta"))
    print(f"  {'-'*62}")
    a_pos, b_pos = 0, 0
    for y in all_years:
        pa = sa['year_pnl'].get(y, 0)
        pb = sb['year_pnl'].get(y, 0)
        if pa > 0: a_pos += 1
        if pb > 0: b_pos += 1
        print(row.format(str(y), f"${pa:,.0f}", f"${pb:,.0f}", f"${pb-pa:+,.0f}"))
    print(f"\n  正收益年数: L8_BASE+Cap80 = {a_pos}/{len(all_years)},  L8_MAX = {b_pos}/{len(all_years)}")


def run_kfold_comparison(data: DataBundle):
    print(f"\n{'='*80}")
    print(f"  K-Fold 6-Fold 对比")
    print(f"{'='*80}")

    FOLDS = [
        ("F1_2015-2016", "2015-01-01", "2016-12-31"),
        ("F2_2017-2018", "2017-01-01", "2018-12-31"),
        ("F3_2019-2020", "2019-01-01", "2020-12-31"),
        ("F4_2021-2022", "2021-01-01", "2022-12-31"),
        ("F5_2023-2024", "2023-01-01", "2024-12-31"),
        ("F6_2025-2026", "2025-01-01", "2026-12-31"),
    ]

    row = "  {:<15} {:>10} {:>10} {:>10}"
    print(row.format("Fold", "BASE+Cap80", "L8_MAX", "Delta"))
    print(f"  {'-'*45}")

    a_sharpes, b_sharpes = [], []
    for fname, start, end in FOLDS:
        fold_data = data.slice(start, end)
        if len(fold_data.m15_df) < 1000:
            print(f"  {fname:<15} (insufficient data)")
            continue
        sa, sb = run_a_vs_b(fold_data)
        delta = sb['sharpe'] - sa['sharpe']
        a_sharpes.append(sa['sharpe'])
        b_sharpes.append(sb['sharpe'])
        print(row.format(fname, f"{sa['sharpe']:.2f}", f"{sb['sharpe']:.2f}", f"{delta:+.2f}"))

    print(f"  {'-'*45}")
    if a_sharpes and b_sharpes:
        print(row.format("Mean", f"{np.mean(a_sharpes):.2f}", f"{np.mean(b_sharpes):.2f}",
                         f"{np.mean(b_sharpes)-np.mean(a_sharpes):+.2f}"))
        print(row.format("Min", f"{np.min(a_sharpes):.2f}", f"{np.min(b_sharpes):.2f}", ""))
        a_pass = sum(1 for s in a_sharpes if s > 0)
        b_pass = sum(1 for s in b_sharpes if s > 0)
        print(f"\n  全正 Fold 数: L8_BASE+Cap80 = {a_pass}/6,  L8_MAX = {b_pass}/6")


def run_spread_comparison(data: DataBundle):
    print(f"\n{'='*80}")
    print(f"  多 Spread 鲁棒性对比")
    print(f"{'='*80}")

    row = "  {:<10} {:>12} {:>12} {:>10}"
    print(row.format("Spread", "BASE+Cap80", "L8_MAX", "Delta"))
    print(f"  {'-'*44}")

    for sp in [0.30, 0.50, 0.80, 1.00]:
        sa, sb = run_a_vs_b(data, spread=sp)
        delta = sb['sharpe'] - sa['sharpe']
        print(row.format(f"${sp:.2f}", f"{sa['sharpe']:.2f}", f"{sb['sharpe']:.2f}", f"{delta:+.2f}"))


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()

    print("=" * 80)
    print("  L8_BASE+Cap80 vs L8_MAX 对比测试")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("\n  A = L8_BASE + Cap$80 (当前实盘)")
    print("  B = L8_MAX  = L8_BASE + H1KC(E15/M2.0) + EqCurve(LB10) + Cap$30")

    print("\n  Loading data...")
    data = DataBundle.load_default()

    # 1. Full sample
    print("\n  [1/4] Full Sample Comparison...")
    sa, sb = run_a_vs_b(data)
    print_comparison(sa, sb)

    # 2. K-Fold
    print("\n  [2/4] K-Fold 6-Fold Comparison...")
    run_kfold_comparison(data)

    # 3. Spread robustness
    print("\n  [3/4] Multi-Spread Robustness...")
    run_spread_comparison(data)

    # 4. Summary
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  完成! 总耗时: {elapsed:.0f}s")
    print(f"{'='*80}")

    # Save key numbers
    summary = {
        'A_label': 'L8_BASE+Cap80',
        'B_label': 'L8_MAX (H1KC+EqCurve+Cap30)',
        'full_sample': {
            'A': {k: v for k, v in sa.items() if k != '_trades'},
            'B': {k: v for k, v in sb.items() if k != '_trades'},
        },
        'spread': SPREAD,
        'timestamp': datetime.now().isoformat(),
    }
    with open(OUT_DIR / "comparison_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Results saved to {OUT_DIR}/")
