"""
手数对比测试: 0.03 vs 0.04 固定手数
====================================
验证手数从 0.03 提升到 0.04 对策略表现的影响。
通过设置 min_lot_size = max_lot_size 来模拟固定手数。
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS
from backtest.engine import TradeRecord

SPREAD = 0.50

BASE_KWARGS = {
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
    'initial_capital': 5000,
    'risk_per_trade': 125,
}


def run_fixed_lot(data, lot_size, label):
    """Run backtest with fixed lot size by clamping min=max=lot_size."""
    kwargs = {
        **BASE_KWARGS,
        'min_lot_size': lot_size,
        'max_lot_size': lot_size,
    }
    return run_variant(data, label, **kwargs)


def print_comparison(results):
    """Print side-by-side comparison."""
    print("\n" + "=" * 70)
    print("  手数对比: 0.03 vs 0.04 (固定手数, L8_BASE+Cap80 参数)")
    print("=" * 70)

    headers = ["指标", "0.03 手", "0.04 手", "变化"]
    row_fmt = "  {:<22} {:>12} {:>12} {:>12}"
    print(row_fmt.format(*headers))
    print("  " + "-" * 60)

    r03, r04 = results[0], results[1]

    rows = [
        ("交易笔数", f"{r03['n']}", f"{r04['n']}", "—"),
        ("总 PnL", f"${r03['total_pnl']:,.2f}", f"${r04['total_pnl']:,.2f}",
         f"{(r04['total_pnl']/r03['total_pnl']-1)*100:+.1f}%" if r03['total_pnl'] else "—"),
        ("Sharpe", f"{r03['sharpe']:.2f}", f"{r04['sharpe']:.2f}",
         f"{r04['sharpe']-r03['sharpe']:+.2f}"),
        ("胜率", f"{r03['win_rate']:.1%}", f"{r04['win_rate']:.1%}",
         f"{(r04['win_rate']-r03['win_rate'])*100:+.1f}pp"),
        ("MaxDD", f"${r03['max_dd']:,.2f}", f"${r04['max_dd']:,.2f}",
         f"{(r04['max_dd']/r03['max_dd']-1)*100:+.1f}%" if r03['max_dd'] else "—"),
        ("MaxDD%", f"{r03['max_dd_pct']:.2%}", f"{r04['max_dd_pct']:.2%}",
         f"{(r04['max_dd_pct']-r03['max_dd_pct'])*100:+.2f}pp"),
        ("盈亏比", f"{r03['rr']:.2f}", f"{r04['rr']:.2f}",
         f"{r04['rr']-r03['rr']:+.2f}"),
        ("Avg Win", f"${r03['avg_win']:,.2f}", f"${r04['avg_win']:,.2f}",
         f"{(r04['avg_win']/r03['avg_win']-1)*100:+.1f}%" if r03['avg_win'] else "—"),
        ("Avg Loss", f"${r03['avg_loss']:,.2f}", f"${r04['avg_loss']:,.2f}",
         f"{(r04['avg_loss']/r03['avg_loss']-1)*100:+.1f}%" if r03['avg_loss'] else "—"),
    ]

    for row in rows:
        print(row_fmt.format(*row))

    # Per-year breakdown
    if 'year_pnl' in r03 and 'year_pnl' in r04:
        print("\n  年度 PnL 对比:")
        print(row_fmt.format("年份", "0.03 手", "0.04 手", "变化"))
        print("  " + "-" * 60)
        all_years = sorted(set(list(r03['year_pnl'].keys()) + list(r04['year_pnl'].keys())))
        for y in all_years:
            p03 = r03['year_pnl'].get(y, 0)
            p04 = r04['year_pnl'].get(y, 0)
            delta = f"{(p04/p03-1)*100:+.1f}%" if p03 else "—"
            print(row_fmt.format(str(y), f"${p03:,.2f}", f"${p04:,.2f}", delta))

    print("\n  理论关系: 固定手数 0.04/0.03 = +33.3% (PnL 和 MaxDD 应等比放大)")
    print("  Sharpe 和胜率应保持不变 (仅缩放，不改变信号)")
    print("=" * 70)


if __name__ == "__main__":
    t0 = time.time()

    print("Loading data...")
    data = DataBundle.load_default()

    results = []
    for lot, label in [(0.03, "Fixed_0.03"), (0.04, "Fixed_0.04")]:
        stats = run_fixed_lot(data, lot, label)
        results.append(stats)

    print_comparison(results)

    print(f"\nTotal time: {time.time()-t0:.1f}s")
