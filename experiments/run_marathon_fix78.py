#!/usr/bin/env python3
"""
Marathon Phase 7/8 补跑 — 修复 trades key (_trades not trades)
"""
import sys, os, time, numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "marathon_results"


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def phase7_recent_analysis(p):
    p("=" * 80)
    p("PHASE 7 (FIXED): 近期行情专项分析 (2024-2026)")
    p("=" * 80)

    from backtest import DataBundle, run_variant
    from backtest.runner import LIVE_PARITY_KWARGS
    import pandas as pd

    L4_STAR = {**LIVE_PARITY_KWARGS, "time_decay_tp": False}

    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    recent = data.slice("2024-01-01", "2026-04-10")
    s = run_variant(recent, "L4star_2024_2026", verbose=False, spread_cost=0.30, **L4_STAR)

    trades = s.get('_trades', [])
    p(f"\n  Overall 2024-2026: N={s['n']}, Sharpe={s['sharpe']:.2f}, "
      f"PnL={fmt(s['total_pnl'])}, WR={s['win_rate']:.1f}%, MaxDD={fmt(s['max_dd'])}")
    p(f"  Trade records: {len(trades)}")

    if not trades:
        p("  ERROR: Still no trade records!")
        return

    # Monthly PnL
    p(f"\n--- Monthly PnL ---")
    p(f"{'Month':<10s}  {'N':>5s}  {'PnL':>10s}  {'WR%':>6s}  {'$/t':>8s}")
    p("-" * 45)
    monthly = {}
    for t in trades:
        month = pd.Timestamp(t.exit_time).strftime('%Y-%m')
        if month not in monthly:
            monthly[month] = {'n': 0, 'wins': 0, 'pnl': 0.0}
        monthly[month]['n'] += 1
        monthly[month]['pnl'] += t.pnl
        if t.pnl > 0:
            monthly[month]['wins'] += 1
    neg_months = 0
    for month in sorted(monthly.keys()):
        m = monthly[month]
        wr = m['wins'] / m['n'] * 100 if m['n'] > 0 else 0
        avg = m['pnl'] / m['n'] if m['n'] > 0 else 0
        marker = " ***" if m['pnl'] < 0 else ""
        if m['pnl'] < 0:
            neg_months += 1
        p(f"  {month:<8s}  {m['n']:>5d}  {fmt(m['pnl'])}  {wr:>5.1f}%  ${avg:>7.2f}{marker}")
    p(f"\n  Negative months: {neg_months}/{len(monthly)}")

    # Weekly PnL for 2026
    p(f"\n--- Weekly PnL (2026) ---")
    p(f"{'Week':<12s}  {'N':>5s}  {'PnL':>10s}  {'WR%':>6s}")
    p("-" * 40)
    weekly = {}
    for t in trades:
        ts = pd.Timestamp(t.exit_time)
        if ts.year < 2026:
            continue
        week = ts.strftime('%Y-W%V')
        if week not in weekly:
            weekly[week] = {'n': 0, 'wins': 0, 'pnl': 0.0}
        weekly[week]['n'] += 1
        weekly[week]['pnl'] += t.pnl
        if t.pnl > 0:
            weekly[week]['wins'] += 1
    for week in sorted(weekly.keys()):
        w = weekly[week]
        wr = w['wins'] / w['n'] * 100 if w['n'] > 0 else 0
        marker = " ***" if w['pnl'] < 0 else ""
        p(f"  {week:<10s}  {w['n']:>5d}  {fmt(w['pnl'])}  {wr:>5.1f}%{marker}")

    # Exit type distribution
    p(f"\n--- Exit Type Distribution (2024-2026) ---")
    p(f"{'Exit':<20s}  {'N':>5s}  {'PnL':>10s}  {'WR%':>6s}  {'$/t':>8s}  {'Bars':>6s}")
    p("-" * 60)
    by_exit = {}
    for t in trades:
        reason = t.exit_reason or 'unknown'
        if reason not in by_exit:
            by_exit[reason] = {'n': 0, 'wins': 0, 'pnl': 0.0, 'bars': []}
        by_exit[reason]['n'] += 1
        by_exit[reason]['pnl'] += t.pnl
        if t.pnl > 0:
            by_exit[reason]['wins'] += 1
        by_exit[reason]['bars'].append(t.bars_held)
    for reason in sorted(by_exit.keys(), key=lambda r: -by_exit[r]['n']):
        e = by_exit[reason]
        wr = e['wins'] / e['n'] * 100 if e['n'] > 0 else 0
        avg = e['pnl'] / e['n'] if e['n'] > 0 else 0
        avg_bars = np.mean(e['bars']) if e['bars'] else 0
        p(f"  {reason:<18s}  {e['n']:>5d}  {fmt(e['pnl'])}  {wr:>5.1f}%  ${avg:>7.2f}  {avg_bars:>5.1f}")

    # Consecutive loss analysis
    p(f"\n--- 连续亏损分析 ---")
    streaks = []
    current_streak = 0
    current_loss = 0.0
    for t in trades:
        if t.pnl < 0:
            current_streak += 1
            current_loss += t.pnl
        else:
            if current_streak > 0:
                streaks.append((current_streak, current_loss))
            current_streak = 0
            current_loss = 0.0
    if current_streak > 0:
        streaks.append((current_streak, current_loss))
    if streaks:
        max_streak = max(streaks, key=lambda x: x[0])
        max_loss = min(streaks, key=lambda x: x[1])
        p(f"  最长连亏: {max_streak[0]} 笔, 总亏损 {fmt(max_streak[1])}")
        p(f"  最大连亏金额: {max_loss[0]} 笔, 总亏损 {fmt(max_loss[1])}")
        streak_counts = {}
        for s, _ in streaks:
            streak_counts[s] = streak_counts.get(s, 0) + 1
        p(f"  连亏分布:")
        for k in sorted(streak_counts.keys()):
            p(f"    {k}连亏: {streak_counts[k]}次")

    # Strategy breakdown
    p(f"\n--- Strategy Breakdown (2024-2026) ---")
    by_strat = {}
    for t in trades:
        strat = t.strategy or 'unknown'
        if strat not in by_strat:
            by_strat[strat] = {'n': 0, 'wins': 0, 'pnl': 0.0}
        by_strat[strat]['n'] += 1
        by_strat[strat]['pnl'] += t.pnl
        if t.pnl > 0:
            by_strat[strat]['wins'] += 1
    for strat, d in sorted(by_strat.items(), key=lambda x: -x[1]['n']):
        wr = d['wins'] / d['n'] * 100 if d['n'] > 0 else 0
        avg = d['pnl'] / d['n'] if d['n'] > 0 else 0
        p(f"  {strat}: N={d['n']}, PnL={fmt(d['pnl'])}, WR={wr:.1f}%, $/t=${avg:.2f}")

    # Day-of-week analysis
    p(f"\n--- Day-of-Week (2024-2026) ---")
    by_dow = {}
    dow_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
    for t in trades:
        dow = pd.Timestamp(t.entry_time).dayofweek
        name = dow_names.get(dow, str(dow))
        if name not in by_dow:
            by_dow[name] = {'n': 0, 'pnl': 0.0, 'wins': 0}
        by_dow[name]['n'] += 1
        by_dow[name]['pnl'] += t.pnl
        if t.pnl > 0:
            by_dow[name]['wins'] += 1
    for name in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']:
        if name in by_dow:
            d = by_dow[name]
            wr = d['wins'] / d['n'] * 100 if d['n'] > 0 else 0
            avg = d['pnl'] / d['n'] if d['n'] > 0 else 0
            p(f"  {name}: N={d['n']}, PnL={fmt(d['pnl'])}, WR={wr:.1f}%, $/t=${avg:.2f}")

    # Hour distribution
    p(f"\n--- Entry Hour (2024-2026) ---")
    by_hour = {}
    for t in trades:
        hour = pd.Timestamp(t.entry_time).hour
        if hour not in by_hour:
            by_hour[hour] = {'n': 0, 'pnl': 0.0, 'wins': 0}
        by_hour[hour]['n'] += 1
        by_hour[hour]['pnl'] += t.pnl
        if t.pnl > 0:
            by_hour[hour]['wins'] += 1
    for h in sorted(by_hour.keys()):
        d = by_hour[h]
        wr = d['wins'] / d['n'] * 100 if d['n'] > 0 else 0
        avg = d['pnl'] / d['n'] if d['n'] > 0 else 0
        bar = '#' * max(1, int(d['n'] / 10))
        p(f"  {h:>2d}:00  N={d['n']:>4d}  PnL={fmt(d['pnl'])}  WR={wr:>5.1f}%  $/t=${avg:.2f}  {bar}")


def phase8_monte_carlo(p):
    p("\n" + "=" * 80)
    p("PHASE 8 (FIXED): Monte Carlo Bootstrap 稳定性")
    p("  随机抽取 80% 交易样本，重复 500 次")
    p("=" * 80)

    from backtest import DataBundle, run_variant
    from backtest.runner import LIVE_PARITY_KWARGS

    L4_STAR = {**LIVE_PARITY_KWARGS, "time_decay_tp": False}

    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    s = run_variant(data, "MC_base", verbose=False, spread_cost=0.30, **L4_STAR)
    trades = s.get('_trades', [])

    p(f"\n  Full sample: N={s['n']}, Sharpe={s['sharpe']:.2f}")
    p(f"  Trade records: {len(trades)}")

    if not trades:
        p("  ERROR: No trade records!")
        return

    pnls = np.array([t.pnl for t in trades])
    n_total = len(pnls)
    n_sample = int(n_total * 0.8)
    n_bootstrap = 500

    p(f"  Total trades: {n_total}")
    p(f"  Sample size: {n_sample} (80%)")
    p(f"  Bootstrap iterations: {n_bootstrap}")

    np.random.seed(42)
    sharpes = []
    maxdds = []
    total_pnls = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_total, n_sample, replace=True)
        sample = pnls[idx]
        std = sample.std(ddof=1)
        if std > 0:
            sharpe = sample.mean() / std * np.sqrt(252 * 6)
            sharpes.append(sharpe)
        total_pnls.append(sample.sum())
        cum = np.cumsum(sample)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        maxdds.append(dd.max())

    sharpes = np.array(sharpes)
    maxdds = np.array(maxdds)
    total_pnls = np.array(total_pnls)

    p(f"\n--- Sharpe Distribution ---")
    p(f"  Mean:   {sharpes.mean():.2f}")
    p(f"  Median: {np.median(sharpes):.2f}")
    p(f"  Std:    {sharpes.std():.2f}")
    p(f"  5th %%:  {np.percentile(sharpes, 5):.2f}")
    p(f"  25th %%: {np.percentile(sharpes, 25):.2f}")
    p(f"  75th %%: {np.percentile(sharpes, 75):.2f}")
    p(f"  95th %%: {np.percentile(sharpes, 95):.2f}")
    p(f"  Min:    {sharpes.min():.2f}")
    p(f"  Max:    {sharpes.max():.2f}")
    p(f"  P(Sharpe > 0):   {(sharpes > 0).mean()*100:.1f}%")
    p(f"  P(Sharpe > 1.0): {(sharpes > 1.0).mean()*100:.1f}%")
    p(f"  P(Sharpe > 2.0): {(sharpes > 2.0).mean()*100:.1f}%")
    p(f"  P(Sharpe > 3.0): {(sharpes > 3.0).mean()*100:.1f}%")

    p(f"\n  Histogram (bin width 0.2):")
    bins = np.arange(max(0, sharpes.min() - 0.1), sharpes.max() + 0.3, 0.2)
    hist, edges = np.histogram(sharpes, bins=bins)
    max_h = max(hist) if len(hist) > 0 else 1
    for i in range(len(hist)):
        bar_len = int(hist[i] / max_h * 40)
        bar = '#' * bar_len
        p(f"  [{edges[i]:>5.1f}, {edges[i+1]:>5.1f}): {hist[i]:>3d} {bar}")

    p(f"\n--- MaxDD Distribution ---")
    p(f"  Mean:   {fmt(maxdds.mean())}")
    p(f"  Median: {fmt(np.median(maxdds))}")
    p(f"  5th %%:  {fmt(np.percentile(maxdds, 5))}")
    p(f"  25th %%: {fmt(np.percentile(maxdds, 25))}")
    p(f"  75th %%: {fmt(np.percentile(maxdds, 75))}")
    p(f"  95th %%: {fmt(np.percentile(maxdds, 95))}")
    p(f"  Worst:  {fmt(maxdds.max())}")
    p(f"  P(MaxDD < $500): {(maxdds < 500).mean()*100:.1f}%")
    p(f"  P(MaxDD < $1000): {(maxdds < 1000).mean()*100:.1f}%")

    p(f"\n--- PnL Distribution ---")
    p(f"  Mean:   {fmt(total_pnls.mean())}")
    p(f"  Median: {fmt(np.median(total_pnls))}")
    p(f"  5th %%:  {fmt(np.percentile(total_pnls, 5))}")
    p(f"  95th %%: {fmt(np.percentile(total_pnls, 95))}")
    p(f"  P(PnL > 0): {(total_pnls > 0).mean()*100:.1f}%")


def phase10_trade_stats(p):
    p("\n" + "=" * 80)
    p("PHASE 10 (FIXED): L4* 交易统计")
    p("=" * 80)

    from backtest import DataBundle, run_variant
    from backtest.runner import LIVE_PARITY_KWARGS

    L4_STAR = {**LIVE_PARITY_KWARGS, "time_decay_tp": False}

    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    s = run_variant(data, "L4star_stats", verbose=False, spread_cost=0.30, **L4_STAR)
    trades = s.get('_trades', [])

    p(f"\n  Full sample: N={s['n']}, Sharpe={s['sharpe']:.2f}")

    if not trades:
        p("  ERROR: No trade records!")
        return

    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl < 0]
    flat = [t.pnl for t in trades if t.pnl == 0]

    p(f"  Win:  {len(wins)} ({len(wins)/len(trades)*100:.1f}%)")
    p(f"  Loss: {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")
    p(f"  Flat: {len(flat)} ({len(flat)/len(trades)*100:.1f}%)")
    p(f"  Avg win:      ${np.mean(wins):.2f}")
    p(f"  Avg loss:     ${np.mean(losses):.2f}")
    p(f"  Median win:   ${np.median(wins):.2f}")
    p(f"  Median loss:  ${np.median(losses):.2f}")
    p(f"  Max win:      ${max(wins):.2f}")
    p(f"  Max loss:     ${min(losses):.2f}")
    p(f"  Profit factor: {sum(wins)/abs(sum(losses)):.2f}")
    p(f"  Expectancy:   ${np.mean([t.pnl for t in trades]):.2f}/trade")

    # Win/loss by size bucket
    p(f"\n--- PnL Distribution Buckets ---")
    all_pnls = [t.pnl for t in trades]
    buckets = [(-999, -50), (-50, -20), (-20, -10), (-10, -5), (-5, 0),
               (0, 1), (1, 3), (3, 5), (5, 10), (10, 20), (20, 50), (50, 999)]
    for lo, hi in buckets:
        count = sum(1 for x in all_pnls if lo <= x < hi)
        if count > 0:
            total = sum(x for x in all_pnls if lo <= x < hi)
            label = f"[${lo}, ${hi})" if hi < 999 else f"[${lo}+)"
            if lo == -999:
                label = f"(<${abs(lo)})"
                label = f"(< -$50)"
            p(f"  {label:<15s}: {count:>5d} trades  ({count/len(trades)*100:>5.1f}%)  Total: {fmt(total)}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    phases = [
        ("phase07_recent_FIXED.txt", "Phase 7 (FIXED)", phase7_recent_analysis),
        ("phase08_montecarlo_FIXED.txt", "Phase 8 (FIXED)", phase8_monte_carlo),
        ("phase10_stats_FIXED.txt", "Phase 10 Trade Stats (FIXED)", phase10_trade_stats),
    ]

    for filename, desc, func in phases:
        filepath = os.path.join(OUTPUT_DIR, filename)
        print(f"--- STARTING: {desc} ---", flush=True)
        t0 = time.time()

        with open(filepath, 'w', encoding='utf-8') as f:
            def p(msg=""):
                print(msg, flush=True)
                f.write(msg + "\n")
                f.flush()

            p(f"# {desc}")
            p(f"# Started: {datetime.now()}")
            p("")
            try:
                func(p)
            except Exception as e:
                import traceback
                p(f"\n# ERROR: {e}")
                p(traceback.format_exc())
            elapsed = time.time() - t0
            p(f"\n# Completed: {datetime.now()}")
            p(f"# Elapsed: {elapsed/60:.1f} minutes")

        print(f"--- COMPLETED: {desc} [{(time.time()-t0)/60:.1f} min] ---", flush=True)


if __name__ == "__main__":
    main()
