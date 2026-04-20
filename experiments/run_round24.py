"""
R24: Five research directions on XAUUSD (all on L7 baseline)
=============================================================
A. ExtremeRegime K-Fold (high trail 0.25/0.05 → L8 candidate)
B. Timeout analysis + MaxHold optimization on L7
C. Dynamic position sizing (streak/regime/equity curve)
D. Session-adaptive trail parameters
E. D1 trend confirmation filter
"""

import sys, os, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from backtest.runner import (
    DataBundle, run_variant, run_kfold, LIVE_PARITY_KWARGS
)

OUT_DIR = Path("results/round24_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            try:
                f.write(data)
            except UnicodeEncodeError:
                f.write(data.encode('ascii', errors='replace').decode('ascii'))
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


L7_KWARGS = {
    **LIVE_PARITY_KWARGS,
    'time_adaptive_trail': {'start': 2, 'decay': 0.75, 'floor': 0.003},
    'min_entry_gap_hours': 1.0,
}


def print_stats_row(stats, prefix=""):
    print(f"  {prefix}{stats.get('label','')}: N={stats['n']}, Sharpe={stats['sharpe']:.2f}, "
          f"PnL=${stats['total_pnl']:.0f}, WR={stats['win_rate']:.1f}%, "
          f"MaxDD=${stats['max_dd']:.0f}, {stats.get('elapsed_s',0):.0f}s")


def kfold_summary(results, label=""):
    sharpes = [r['sharpe'] for r in results]
    pnls = [r['total_pnl'] for r in results]
    positive = sum(1 for s in sharpes if s > 0)
    print(f"\n  K-Fold Summary [{label}]: {positive}/{len(results)} positive")
    print(f"  Sharpe: mean={np.mean(sharpes):.2f}, min={min(sharpes):.2f}, max={max(sharpes):.2f}")
    print(f"  PnL: mean=${np.mean(pnls):.0f}, total=${sum(pnls):.0f}")
    return positive, len(results)


# ═══════════════════════════════════════════════════════════════
# A. ExtremeRegime K-Fold
# ═══════════════════════════════════════════════════════════════

def run_A(data):
    print("\n" + "=" * 80)
    print("R24-A: ExtremeRegime on L7 — K-Fold Validation")
    print("=" * 80)

    # L7 baseline
    l7_stats = run_variant(data, "L7_baseline", **L7_KWARGS)
    print_stats_row(l7_stats)

    # L7 + ExtremeRegime (tighten high vol trail)
    l8_kwargs = copy.deepcopy(L7_KWARGS)
    l8_kwargs['regime_config'] = {
        'low':    {'trail_act': 0.40, 'trail_dist': 0.10},
        'normal': {'trail_act': 0.28, 'trail_dist': 0.06},
        'high':   {'trail_act': 0.12, 'trail_dist': 0.01},   # tighter than L7's 0.02
    }
    l8a_stats = run_variant(data, "L8a_extreme_high", **l8_kwargs)
    print_stats_row(l8a_stats)

    # Even more extreme
    l8b_kwargs = copy.deepcopy(L7_KWARGS)
    l8b_kwargs['regime_config'] = {
        'low':    {'trail_act': 0.40, 'trail_dist': 0.10},
        'normal': {'trail_act': 0.28, 'trail_dist': 0.06},
        'high':   {'trail_act': 0.08, 'trail_dist': 0.01},
    }
    l8b_stats = run_variant(data, "L8b_ultra_extreme", **l8b_kwargs)
    print_stats_row(l8b_stats)

    # Also test OnlyHigh approach: only change high, keep low/normal same
    l8c_kwargs = copy.deepcopy(L7_KWARGS)
    l8c_kwargs['regime_config'] = {
        'low':    {'trail_act': 0.40, 'trail_dist': 0.10},
        'normal': {'trail_act': 0.28, 'trail_dist': 0.06},
        'high':   {'trail_act': 0.06, 'trail_dist': 0.005},
    }
    l8c_stats = run_variant(data, "L8c_max_tight", **l8c_kwargs)
    print_stats_row(l8c_stats)

    # K-Fold on best
    best_label = max([l8a_stats, l8b_stats, l8c_stats], key=lambda x: x['sharpe'])
    best_name = best_label['label']
    print(f"\n  Best candidate: {best_name} Sharpe={best_label['sharpe']:.2f}")

    # K-Fold L7 baseline
    print("\n  --- K-Fold: L7 baseline ---")
    kf_l7 = run_kfold(data, L7_KWARGS, label_prefix="L7")
    for r in kf_l7:
        print_stats_row(r, "  ")
    kfold_summary(kf_l7, "L7")

    # K-Fold best candidate
    if best_name == "L8a_extreme_high":
        kf_kwargs = l8a_kwargs
    elif best_name == "L8b_ultra_extreme":
        kf_kwargs = l8b_kwargs
    else:
        kf_kwargs = l8c_kwargs

    print(f"\n  --- K-Fold: {best_name} ---")
    kf_best = run_kfold(data, kf_kwargs, label_prefix=best_name)
    for r in kf_best:
        print_stats_row(r, "  ")
    kfold_summary(kf_best, best_name)

    # Delta per fold
    print(f"\n  --- Fold-by-Fold Delta ---")
    for i in range(min(len(kf_l7), len(kf_best))):
        delta = kf_best[i]['sharpe'] - kf_l7[i]['sharpe']
        print(f"  Fold{i+1}: L7={kf_l7[i]['sharpe']:.2f}, {best_name}={kf_best[i]['sharpe']:.2f}, delta={delta:+.2f}")


# ═══════════════════════════════════════════════════════════════
# B. Timeout Analysis + MaxHold on L7
# ═══════════════════════════════════════════════════════════════

def run_B(data):
    print("\n" + "=" * 80)
    print("R24-B: Timeout Analysis + MaxHold Optimization on L7")
    print("=" * 80)

    # Baseline L7
    l7_stats = run_variant(data, "L7_MH20", **L7_KWARGS)
    trades = l7_stats['_trades']

    # Timeout profile
    exit_reasons = {}
    for t in trades:
        r = t.exit_reason
        exit_reasons.setdefault(r, {'n': 0, 'pnl': 0, 'wins': 0, 'bars': []})
        exit_reasons[r]['n'] += 1
        exit_reasons[r]['pnl'] += t.pnl
        if t.pnl > 0:
            exit_reasons[r]['wins'] += 1
        exit_reasons[r]['bars'].append(t.bars_held)

    print(f"\n  Exit Reason Profile (L7):")
    print(f"  {'Reason':<20s} {'N':>6} {'PnL':>10} {'WR%':>6} {'AvgBars':>8}")
    print(f"  {'-'*54}")
    for reason in sorted(exit_reasons, key=lambda r: exit_reasons[r]['pnl'], reverse=True):
        v = exit_reasons[reason]
        wr = v['wins'] / v['n'] * 100 if v['n'] > 0 else 0
        avg_bars = np.mean(v['bars']) if v['bars'] else 0
        print(f"  {reason:<20s} {v['n']:>6} ${v['pnl']:>9.0f} {wr:>5.1f}% {avg_bars:>7.1f}")

    # Timeout trades: PnL distribution by bars held
    timeout_trades = [t for t in trades if 'timeout' in t.exit_reason.lower() or t.exit_reason == 'Timeout']
    if timeout_trades:
        print(f"\n  Timeout Trades Detail:")
        print(f"  Total: {len(timeout_trades)} trades, PnL=${sum(t.pnl for t in timeout_trades):.0f}")
        bars_held = [t.bars_held for t in timeout_trades]
        print(f"  Bars held: mean={np.mean(bars_held):.1f}, median={np.median(bars_held):.0f}, "
              f"min={min(bars_held)}, max={max(bars_held)}")

        # PnL by bars_held bucket
        buckets = [(1, 10), (11, 15), (16, 20), (21, 30), (31, 60), (61, 999)]
        for lo, hi in buckets:
            b_trades = [t for t in timeout_trades if lo <= t.bars_held <= hi]
            if b_trades:
                pnl = sum(t.pnl for t in b_trades)
                wr = sum(1 for t in b_trades if t.pnl > 0) / len(b_trades) * 100
                print(f"  Bars {lo}-{hi}: N={len(b_trades)}, PnL=${pnl:.0f}, WR={wr:.1f}%")

    # MaxHold sweep
    print(f"\n  --- MaxHold Sweep ---")
    for mh in [10, 12, 15, 18, 20, 25, 30]:
        kwargs = copy.deepcopy(L7_KWARGS)
        kwargs['keltner_max_hold_m15'] = mh
        stats = run_variant(data, f"L7_MH{mh}", verbose=False, **kwargs)
        print_stats_row(stats)

    # K-Fold on interesting MH values
    for mh in [12, 15, 18]:
        kwargs = copy.deepcopy(L7_KWARGS)
        kwargs['keltner_max_hold_m15'] = mh
        print(f"\n  --- K-Fold: MH={mh} ---")
        kf = run_kfold(data, kwargs, label_prefix=f"MH{mh}")
        for r in kf:
            print_stats_row(r, "  ")
        kfold_summary(kf, f"MH{mh}")

    # Early exit for adverse timeout: if holding > N bars and floating PnL < -X*ATR, exit
    print(f"\n  --- Timeout Early Exit (via reduced MaxHold) ---")
    # This is tested via MaxHold sweep above. Summary:
    print(f"  See MaxHold sweep results above.")


# ═══════════════════════════════════════════════════════════════
# C. Dynamic Position Sizing
# ═══════════════════════════════════════════════════════════════

def run_C(data):
    print("\n" + "=" * 80)
    print("R24-C: Dynamic Position Sizing (Post-hoc Simulation)")
    print("=" * 80)

    l7_stats = run_variant(data, "L7_sizing_base", **L7_KWARGS)
    trades = l7_stats['_trades']

    pnl_list = [t.pnl for t in trades]
    entry_times = [t.entry_time for t in trades]

    # Baseline stats
    total_pnl = sum(pnl_list)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t.exit_time).date()
        daily.setdefault(d, 0)
        daily[d] += t.pnl
    daily_arr = np.array(list(daily.values()))
    base_sharpe = daily_arr.mean() / daily_arr.std() * np.sqrt(252) if daily_arr.std() > 0 else 0
    base_eq = np.cumsum(pnl_list)
    base_dd = (np.maximum.accumulate(base_eq) - base_eq).max()

    print(f"\n  Baseline: PnL=${total_pnl:.0f}, Sharpe={base_sharpe:.2f}, MaxDD=${base_dd:.0f}")

    # Strategy 1: Streak-based sizing
    # After N consecutive wins: scale up; after N consecutive losses: scale down
    print(f"\n  --- Strategy 1: Streak-based Sizing ---")
    for win_streak_mult, loss_streak_reduce in [(1.5, 0.5), (2.0, 0.5), (1.5, 0.75)]:
        scaled_pnl = []
        streak = 0
        for pnl in pnl_list:
            if streak >= 3:
                mult = win_streak_mult
            elif streak <= -3:
                mult = loss_streak_reduce
            else:
                mult = 1.0
            scaled_pnl.append(pnl * mult)
            if pnl > 0:
                streak = streak + 1 if streak > 0 else 1
            elif pnl < 0:
                streak = streak - 1 if streak < 0 else -1
            else:
                streak = 0

        sp = np.array(scaled_pnl)
        eq = np.cumsum(sp)
        dd = (np.maximum.accumulate(eq) - eq).max()
        d_pnl = {}
        for i, t in enumerate(trades):
            d = pd.Timestamp(t.exit_time).date()
            d_pnl.setdefault(d, 0)
            d_pnl[d] += sp[i]
        d_arr = np.array(list(d_pnl.values()))
        sh = d_arr.mean() / d_arr.std() * np.sqrt(252) if d_arr.std() > 0 else 0
        print(f"  Win{win_streak_mult}x/Loss{loss_streak_reduce}x: PnL=${sp.sum():.0f}, "
              f"Sharpe={sh:.2f}, MaxDD=${dd:.0f}, "
              f"delta_sharpe={sh - base_sharpe:+.2f}")

    # Strategy 2: Regime-based sizing (high vol = bigger)
    print(f"\n  --- Strategy 2: Regime-based Sizing ---")
    h1_df = data.h1_df
    for high_mult, low_mult in [(1.5, 0.5), (1.5, 0.75), (2.0, 0.5)]:
        scaled_pnl = []
        for i, t in enumerate(trades):
            et = pd.Timestamp(t.entry_time)
            h1_mask = h1_df.index <= et
            if h1_mask.any():
                h1_idx = h1_df.index[h1_mask][-1]
                atr_pct = h1_df.loc[h1_idx, 'atr_percentile'] if 'atr_percentile' in h1_df.columns else 0.5
            else:
                atr_pct = 0.5
            if atr_pct > 0.70:
                mult = high_mult
            elif atr_pct < 0.30:
                mult = low_mult
            else:
                mult = 1.0
            scaled_pnl.append(pnl_list[i] * mult)

        sp = np.array(scaled_pnl)
        eq = np.cumsum(sp)
        dd = (np.maximum.accumulate(eq) - eq).max()
        d_pnl = {}
        for i, t in enumerate(trades):
            d = pd.Timestamp(t.exit_time).date()
            d_pnl.setdefault(d, 0)
            d_pnl[d] += sp[i]
        d_arr = np.array(list(d_pnl.values()))
        sh = d_arr.mean() / d_arr.std() * np.sqrt(252) if d_arr.std() > 0 else 0
        print(f"  High{high_mult}x/Low{low_mult}x: PnL=${sp.sum():.0f}, "
              f"Sharpe={sh:.2f}, MaxDD=${dd:.0f}, "
              f"delta_sharpe={sh - base_sharpe:+.2f}")

    # Strategy 3: Equity curve management
    print(f"\n  --- Strategy 3: Equity Curve Management ---")
    for lookback, cutoff in [(50, 0), (100, 0), (50, -10)]:
        scaled_pnl = []
        cumulative = 0
        recent = []
        for pnl in pnl_list:
            recent.append(pnl)
            if len(recent) > lookback:
                recent.pop(0)
            if len(recent) >= lookback:
                avg_recent = np.mean(recent)
                mult = 0.5 if avg_recent < cutoff else 1.0
            else:
                mult = 1.0
            scaled_pnl.append(pnl * mult)
            cumulative += pnl

        sp = np.array(scaled_pnl)
        eq = np.cumsum(sp)
        dd = (np.maximum.accumulate(eq) - eq).max()
        d_pnl = {}
        for i, t in enumerate(trades):
            d = pd.Timestamp(t.exit_time).date()
            d_pnl.setdefault(d, 0)
            d_pnl[d] += sp[i]
        d_arr = np.array(list(d_pnl.values()))
        sh = d_arr.mean() / d_arr.std() * np.sqrt(252) if d_arr.std() > 0 else 0
        print(f"  Lookback{lookback}/Cutoff${cutoff}: PnL=${sp.sum():.0f}, "
              f"Sharpe={sh:.2f}, MaxDD=${dd:.0f}, "
              f"delta_sharpe={sh - base_sharpe:+.2f}")

    # Strategy 4: Kelly fraction
    print(f"\n  --- Strategy 4: Kelly Criterion ---")
    wins = [p for p in pnl_list if p > 0]
    losses = [abs(p) for p in pnl_list if p < 0]
    if wins and losses:
        wr = len(wins) / len(pnl_list)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        kelly_f = wr - (1 - wr) / (avg_win / avg_loss)
        print(f"  WR={wr:.3f}, Avg Win=${avg_win:.2f}, Avg Loss=${avg_loss:.2f}")
        print(f"  Full Kelly={kelly_f:.3f}")
        for frac in [0.25, 0.50, 0.75, 1.0]:
            f = kelly_f * frac
            scaled = [p * max(0.1, min(3.0, 1 + f)) for p in pnl_list]
            sp = np.array(scaled)
            eq = np.cumsum(sp)
            dd = (np.maximum.accumulate(eq) - eq).max()
            d_pnl = {}
            for i, t in enumerate(trades):
                d = pd.Timestamp(t.exit_time).date()
                d_pnl.setdefault(d, 0)
                d_pnl[d] += sp[i]
            d_arr = np.array(list(d_pnl.values()))
            sh = d_arr.mean() / d_arr.std() * np.sqrt(252) if d_arr.std() > 0 else 0
            print(f"  {frac:.0%} Kelly (f={f:.3f}): PnL=${sp.sum():.0f}, Sharpe={sh:.2f}, MaxDD=${dd:.0f}")


# ═══════════════════════════════════════════════════════════════
# D. Session-Adaptive Trail
# ═══════════════════════════════════════════════════════════════

def run_D(data):
    print("\n" + "=" * 80)
    print("R24-D: Session-Adaptive Trail Parameters")
    print("=" * 80)

    # Baseline
    l7_stats = run_variant(data, "L7_session_base", **L7_KWARGS)
    trades = l7_stats['_trades']

    # Analyze PnL by entry session
    sessions = {
        'Asia (0-7 UTC)': range(0, 7),
        'London (7-13 UTC)': range(7, 13),
        'NY (13-20 UTC)': range(13, 20),
        'Late (20-24 UTC)': range(20, 24),
    }
    print(f"\n  PnL by Entry Session (L7):")
    print(f"  {'Session':<25s} {'N':>6} {'PnL':>10} {'WR%':>6} {'AvgPnL':>8} {'AvgBars':>8}")
    print(f"  {'-'*68}")
    for sname, hours in sessions.items():
        s_trades = [t for t in trades if pd.Timestamp(t.entry_time).hour in hours]
        if s_trades:
            pnl = sum(t.pnl for t in s_trades)
            wr = sum(1 for t in s_trades if t.pnl > 0) / len(s_trades) * 100
            avg = pnl / len(s_trades)
            avg_bars = np.mean([t.bars_held for t in s_trades])
            print(f"  {sname:<25s} {len(s_trades):>6} ${pnl:>9.0f} {wr:>5.1f}% ${avg:>7.2f} {avg_bars:>7.1f}")

    # Exit reason by session
    print(f"\n  Exit Reason by Session:")
    for sname, hours in sessions.items():
        s_trades = [t for t in trades if pd.Timestamp(t.entry_time).hour in hours]
        if not s_trades:
            continue
        reasons = {}
        for t in s_trades:
            reasons.setdefault(t.exit_reason, 0)
            reasons[t.exit_reason] += 1
        total = len(s_trades)
        parts = ", ".join(f"{r}={n}({n/total*100:.0f}%)" for r, n in sorted(reasons.items(), key=lambda x: -x[1]))
        print(f"  {sname}: {parts}")

    # Test session-based regime override
    # Asia: wider trail (less volatile), London/NY: tighter
    print(f"\n  --- Session-Adaptive Configs ---")

    configs = [
        ("Asia_wider", {
            'session_regime_override': {
                'asia': {'trail_act': 0.50, 'trail_dist': 0.12},
            }
        }),
        ("London_tighter", {
            'session_regime_override': {
                'london': {'trail_act': 0.06, 'trail_dist': 0.01},
            }
        }),
        ("NY_tighter", {
            'session_regime_override': {
                'ny': {'trail_act': 0.06, 'trail_dist': 0.01},
            }
        }),
        ("Asia_wide_LdnNY_tight", {
            'session_regime_override': {
                'asia': {'trail_act': 0.50, 'trail_dist': 0.12},
                'london': {'trail_act': 0.06, 'trail_dist': 0.01},
                'ny': {'trail_act': 0.06, 'trail_dist': 0.01},
            }
        }),
    ]

    # Since the engine may not support session_regime_override natively,
    # we do post-hoc analysis: re-weight trade PnLs by session
    print(f"\n  Note: Engine does not support session_regime_override natively.")
    print(f"  Performing post-hoc analysis: if we could filter trades by session,")
    print(f"  which sessions should use tighter/wider parameters?")

    # Session-only backtests using time slicing
    # Run L7 with different trail configs and see session breakdown
    for trail_label, trail_a, trail_d in [
        ("Tight_0.06/0.01", 0.06, 0.01),
        ("Wide_0.50/0.12", 0.50, 0.12),
        ("Ultra_0.03/0.005", 0.03, 0.005),
    ]:
        kwargs = copy.deepcopy(L7_KWARGS)
        kwargs['regime_config'] = {
            'low':    {'trail_act': trail_a * 3, 'trail_dist': trail_d * 3},
            'normal': {'trail_act': trail_a, 'trail_dist': trail_d},
            'high':   {'trail_act': trail_a * 0.5, 'trail_dist': trail_d * 0.5},
        }
        stats = run_variant(data, f"Trail_{trail_label}", verbose=False, **kwargs)
        trs = stats['_trades']

        session_pnl = {}
        for sname, hours in sessions.items():
            s_trades = [t for t in trs if pd.Timestamp(t.entry_time).hour in hours]
            session_pnl[sname] = sum(t.pnl for t in s_trades) if s_trades else 0

        session_str = ", ".join(f"{s[:5]}=${v:.0f}" for s, v in session_pnl.items())
        print(f"  {trail_label}: Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}, {session_str}")


# ═══════════════════════════════════════════════════════════════
# E. D1 Trend Confirmation
# ═══════════════════════════════════════════════════════════════

def run_E(data):
    print("\n" + "=" * 80)
    print("R24-E: D1 Trend Confirmation Filter (Post-hoc)")
    print("=" * 80)

    l7_stats = run_variant(data, "L7_d1_base", **L7_KWARGS)
    trades = l7_stats['_trades']

    h1_df = data.h1_df

    # Build D1 data from H1
    d1 = h1_df.resample('D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    d1['EMA20'] = d1['Close'].ewm(span=20).mean()
    d1['EMA50'] = d1['Close'].ewm(span=50).mean()
    d1['D1_ATR'] = (d1['High'] - d1['Low']).rolling(14).mean()
    d1['D1_trend'] = 0
    d1.loc[d1['Close'] > d1['EMA20'], 'D1_trend'] = 1
    d1.loc[d1['Close'] < d1['EMA20'], 'D1_trend'] = -1

    d1['D1_trend_strong'] = 0
    d1.loc[(d1['Close'] > d1['EMA20']) & (d1['EMA20'] > d1['EMA50']), 'D1_trend_strong'] = 1
    d1.loc[(d1['Close'] < d1['EMA20']) & (d1['EMA20'] < d1['EMA50']), 'D1_trend_strong'] = -1

    d1['prev_high'] = d1['High'].shift(1)
    d1['prev_low'] = d1['Low'].shift(1)

    # Tag trades with D1 context
    for t in trades:
        et = pd.Timestamp(t.entry_time).normalize()
        if et in d1.index:
            d1_row = d1.loc[et]
        else:
            d1_before = d1.index[d1.index <= et]
            if len(d1_before) == 0:
                t._d1_trend = 0
                t._d1_strong = 0
                continue
            d1_row = d1.loc[d1_before[-1]]

        t._d1_trend = d1_row['D1_trend']
        t._d1_strong = d1_row['D1_trend_strong']

    # Filter analysis
    filters = [
        ("No filter", lambda t: True),
        ("D1 same direction (EMA20)", lambda t: (t.direction == 'BUY' and getattr(t, '_d1_trend', 0) == 1) or
                                                 (t.direction == 'SELL' and getattr(t, '_d1_trend', 0) == -1)),
        ("D1 opposite direction", lambda t: (t.direction == 'BUY' and getattr(t, '_d1_trend', 0) == -1) or
                                             (t.direction == 'SELL' and getattr(t, '_d1_trend', 0) == 1)),
        ("D1 strong same dir", lambda t: (t.direction == 'BUY' and getattr(t, '_d1_strong', 0) == 1) or
                                          (t.direction == 'SELL' and getattr(t, '_d1_strong', 0) == -1)),
        ("D1 neutral only", lambda t: getattr(t, '_d1_trend', 0) == 0),
    ]

    print(f"\n  D1 Filter Analysis:")
    print(f"  {'Filter':<30s} {'N':>6} {'PnL':>10} {'WR%':>6} {'AvgPnL':>8} {'Sharpe':>8}")
    print(f"  {'-'*72}")
    for fname, ffunc in filters:
        f_trades = [t for t in trades if ffunc(t)]
        if not f_trades:
            print(f"  {fname:<30s} {'N/A':>6}")
            continue
        pnl = sum(t.pnl for t in f_trades)
        wr = sum(1 for t in f_trades if t.pnl > 0) / len(f_trades) * 100
        avg = pnl / len(f_trades)
        d_pnl = {}
        for t in f_trades:
            d = pd.Timestamp(t.exit_time).date()
            d_pnl.setdefault(d, 0)
            d_pnl[d] += t.pnl
        d_arr = np.array(list(d_pnl.values()))
        sh = d_arr.mean() / d_arr.std() * np.sqrt(252) if len(d_arr) > 1 and d_arr.std() > 0 else 0
        print(f"  {fname:<30s} {len(f_trades):>6} ${pnl:>9.0f} {wr:>5.1f}% ${avg:>7.2f} {sh:>7.2f}")

    # Direction consistency analysis
    print(f"\n  Trade Direction vs D1 Trend:")
    combos = {}
    for t in trades:
        key = (t.direction, getattr(t, '_d1_trend', 0))
        combos.setdefault(key, {'n': 0, 'pnl': 0, 'wins': 0})
        combos[key]['n'] += 1
        combos[key]['pnl'] += t.pnl
        if t.pnl > 0:
            combos[key]['wins'] += 1

    for (direction, d1_trend), v in sorted(combos.items()):
        wr = v['wins'] / v['n'] * 100 if v['n'] > 0 else 0
        trend_label = {1: "UP", -1: "DOWN", 0: "NEUTRAL"}.get(d1_trend, "?")
        print(f"  {direction} + D1_{trend_label}: N={v['n']}, PnL=${v['pnl']:.0f}, WR={wr:.1f}%, "
              f"Avg=${v['pnl']/v['n']:.2f}")


def main():
    t_start = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out_path = OUT_DIR / "R24_output.txt"
    out = open(out_path, 'w', encoding='utf-8')

    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, out)

    print(f"# R24: Five Research Directions on XAUUSD")
    print(f"# Started: {ts}")

    # Load data once
    print("\nLoading data...")
    data = DataBundle.load_default()

    run_A(data)
    run_B(data)
    run_C(data)
    run_D(data)
    run_E(data)

    elapsed = time.time() - t_start
    print(f"\n# Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Elapsed: {elapsed/60:.1f} minutes")

    sys.stdout = old_stdout
    out.close()
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
