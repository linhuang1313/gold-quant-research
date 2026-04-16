#!/usr/bin/env python3
"""
Post-Fix Experiment Batch — 6 experiments on fixed engine + LIVE_PARITY
========================================================================
EXP-A: ORB Ablation (on/off + K-Fold)
EXP-B: Global Timeout sweep (24/32/40/48/60 bars for all strategies)
EXP-C: Time Decay TP parameter grid
EXP-D: M15 Mean Reversion (independent RSI2 strategy, bypass choppy gate)
EXP-E: KC Squeeze→Expansion confidence scoring
EXP-F: Multi-KC ensemble (20/1.5 + 25/1.2 + 30/1.0 signal overlap)
"""
import sys, os, time, gc
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine, TradeRecord

OUTPUT_FILE = "exp_batch_postfix_output.txt"

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-01"),
]

# Use MaxHold=20 as validated baseline
BASE = {**LIVE_PARITY_KWARGS, "keltner_max_hold_m15": 20}


class TeeOutput:
    def __init__(self, fp):
        self.file = open(fp, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, d):
        self.stdout.write(d)
        self.file.write(d)
        self.file.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()


tee = TeeOutput(OUTPUT_FILE)
sys.stdout = tee


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def run_kfold(data, label, base_kwargs, test_kwargs, spread=0.30):
    """Run K-Fold and return (wins, details)."""
    wins = 0
    details = []
    for fold_name, start, end in FOLDS:
        fold_data = data.slice(start, end)
        if len(fold_data.m15_df) < 1000 or len(fold_data.h1_df) < 200:
            continue
        sb = run_variant(fold_data, f"B_{fold_name}", verbose=False,
                         **base_kwargs, spread_cost=spread)
        st = run_variant(fold_data, f"T_{fold_name}", verbose=False,
                         **test_kwargs, spread_cost=spread)
        delta = st['sharpe'] - sb['sharpe']
        won = delta > 0
        if won:
            wins += 1
        details.append((fold_name, sb['sharpe'], st['sharpe'], delta, won))
    return wins, details


def print_kfold(label, wins, details):
    for fn, sb, st, delta, won in details:
        print(f"    {fn}: Base={sb:>6.2f}  {label}={st:>6.2f}  "
              f"delta={delta:>+.2f} {'V' if won else 'X'}")
    result = "PASS" if wins >= 5 else "FAIL"
    print(f"    Result: {wins}/{len(details)} {result}")
    return wins >= 5


print("=" * 80)
print("POST-FIX EXPERIMENT BATCH (6 experiments)")
print(f"Started: {datetime.now()}")
print(f"Base: LIVE_PARITY + MaxHold=20")
print("=" * 80)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)


# ══════════════════════════════════════════════════════════════════════════════
# EXP-A: ORB ABLATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("EXP-A: ORB ABLATION — Does ORB strategy help or hurt?")
print("  Test: disable ORB by filtering out orb signals post-hoc")
print("=" * 80)

import research_config as cfg

print(f"\n{'Config':<25s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'WR%':>5s}  {'$/t':>7s}  {'MaxDD':>11s}")
print("-" * 80)

for spread in [0.30, 0.50]:
    # With ORB (default)
    s_with = run_variant(data, f"A_withORB_sp{spread}", verbose=False,
                         **BASE, spread_cost=spread)

    # Without ORB: disable via config flag
    old_orb = cfg.ORB_ENABLED
    cfg.ORB_ENABLED = False
    s_without = run_variant(data, f"A_noORB_sp{spread}", verbose=False,
                            **BASE, spread_cost=spread)
    cfg.ORB_ENABLED = old_orb

    for label, s in [("With ORB", s_with), ("No ORB", s_without)]:
        n = s['n']
        avg = s['total_pnl'] / n if n > 0 else 0
        print(f"  {label:<21s} sp${spread:.2f}  {n:>5d}  {s['sharpe']:>6.2f}  "
              f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  "
              f"${avg:>6.2f}  {fmt(s['max_dd'])}")

    delta = s_without['sharpe'] - s_with['sharpe']
    print(f"  -> Delta (No ORB - With ORB): Sharpe {delta:>+.2f}")
    print()

# K-Fold if removing ORB improves $0.30
print("--- K-Fold: No ORB vs With ORB @ $0.30 ---")
cfg_orb_backup = cfg.ORB_ENABLED

def _run_no_orb_kfold(data):
    wins = 0
    details = []
    for fold_name, start, end in FOLDS:
        fold_data = data.slice(start, end)
        if len(fold_data.m15_df) < 1000 or len(fold_data.h1_df) < 200:
            continue
        sb = run_variant(fold_data, f"A_B_{fold_name}", verbose=False,
                         **BASE, spread_cost=0.30)
        cfg.ORB_ENABLED = False
        st = run_variant(fold_data, f"A_T_{fold_name}", verbose=False,
                         **BASE, spread_cost=0.30)
        cfg.ORB_ENABLED = True
        delta = st['sharpe'] - sb['sharpe']
        won = delta > 0
        if won:
            wins += 1
        details.append((fold_name, sb['sharpe'], st['sharpe'], delta, won))
    return wins, details

cfg.ORB_ENABLED = True
wins_a, det_a = _run_no_orb_kfold(data)
print_kfold("NoORB", wins_a, det_a)
cfg.ORB_ENABLED = cfg_orb_backup
gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# EXP-B: GLOBAL TIMEOUT SWEEP
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("EXP-B: GLOBAL TIMEOUT SWEEP")
print("  Keltner MaxHold already set to 20 (validated)")
print("  This tests the fallback max_hold for other strategies (ORB, RSI, gap)")
print("  via config.STRATEGIES[x]['max_hold_bars'] override")
print("  Current default: 15 H1 bars = 60 M15 bars (15 hours)")
print("  Testing: 6/8/10/12/15 H1 bars (= 24/32/40/48/60 M15)")
print("=" * 80)

TIMEOUT_VALUES = [6, 8, 10, 12, 15]  # H1 bars

print(f"\n{'Timeout(H1)':>11s}  {'M15':>4s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'WR%':>5s}  {'$/t':>7s}  {'MaxDD':>11s}")
print("-" * 75)

# Save original config
orig_strategies = {}
for sname in cfg.STRATEGIES:
    orig_strategies[sname] = cfg.STRATEGIES[sname].get('max_hold_bars', 15)

timeout_results = {}
for mh_h1 in TIMEOUT_VALUES:
    # Override max_hold_bars for all strategies
    for sname in cfg.STRATEGIES:
        cfg.STRATEGIES[sname]['max_hold_bars'] = mh_h1

    s = run_variant(data, f"B_TO{mh_h1}h", verbose=False, **BASE, spread_cost=0.30)
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    marker = " <-- current" if mh_h1 == 15 else ""
    print(f"  {mh_h1:>9d}  {mh_h1*4:>4d}  {n:>5d}  {s['sharpe']:>6.2f}  "
          f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  "
          f"${avg:>6.2f}  {fmt(s['max_dd'])}{marker}")
    timeout_results[mh_h1] = s

# Restore
for sname in cfg.STRATEGIES:
    cfg.STRATEGIES[sname]['max_hold_bars'] = orig_strategies.get(sname, 15)

gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# EXP-C: TIME DECAY TP PARAMETER GRID
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("EXP-C: TIME DECAY TP PARAMETER GRID")
print("  Current: start_hour=1.0, atr_start=0.30, atr_step=0.10")
print("  Grid: start_hour(0.5/1.0/1.5/2.0) x atr_start(0.20/0.30/0.40) x atr_step(0.05/0.10/0.15)")
print("=" * 80)

TD_START_HOURS = [0.5, 1.0, 1.5, 2.0]
TD_ATR_STARTS = [0.20, 0.30, 0.40]
TD_ATR_STEPS = [0.05, 0.10, 0.15]

# Also test TD off
print("\n  --- Time Decay OFF vs ON (current) ---")
s_off = run_variant(data, "C_TD_off", verbose=False,
                    **{**BASE, "time_decay_tp": False}, spread_cost=0.30)
s_on = run_variant(data, "C_TD_on", verbose=False,
                   **BASE, spread_cost=0.30)
for label, s in [("TD OFF", s_off), ("TD ON (current)", s_on)]:
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    print(f"  {label:<20s}  {n:>5d}  {s['sharpe']:>6.2f}  "
          f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  ${avg:>6.2f}  {fmt(s['max_dd'])}")
print(f"  TD contribution: Sharpe {s_on['sharpe'] - s_off['sharpe']:>+.2f}")

print(f"\n{'start_h':>7s} {'atr_s':>5s} {'atr_step':>8s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'$/t':>7s}  {'MaxDD':>11s}")
print("-" * 75)

td_results = {}
for sh in TD_START_HOURS:
    for ast in TD_ATR_STARTS:
        for astp in TD_ATR_STEPS:
            kwargs = {
                **BASE,
                "time_decay_start_hour": sh,
                "time_decay_atr_start": ast,
                "time_decay_atr_step": astp,
            }
            s = run_variant(data, f"C_{sh}_{ast}_{astp}", verbose=False,
                            **kwargs, spread_cost=0.30)
            n = s['n']
            avg = s['total_pnl'] / n if n > 0 else 0
            marker = " <--" if (sh == 1.0 and ast == 0.30 and astp == 0.10) else ""
            print(f"  {sh:>5.1f} {ast:>5.2f} {astp:>8.2f}  {n:>5d}  {s['sharpe']:>6.2f}  "
                  f"{fmt(s['total_pnl'])}  ${avg:>6.2f}  {fmt(s['max_dd'])}{marker}")
            td_results[(sh, ast, astp)] = s

ranked_td = sorted(td_results.items(), key=lambda x: -x[1]['sharpe'])
print(f"\n  Top-3: {[(k, round(v['sharpe'],2)) for k,v in ranked_td[:3]]}")
print(f"  Current (1.0, 0.30, 0.10): Sharpe={td_results.get((1.0, 0.30, 0.10), {}).get('sharpe', 'N/A'):.2f}")

gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# EXP-D: M15 MEAN REVERSION (INDEPENDENT, NO CHOPPY GATE)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("EXP-D: M15 MEAN REVERSION — Independent RSI2 Strategy")
print("  Current M15 RSI: 6 trades in 11 years under Adaptive (choppy gate kills it)")
print("  Test: run RSI with NO intraday_adaptive gating, various thresholds")
print("  RSI2 factor IC=-0.031 (strongest factor)")
print("=" * 80)

RSI_BUY_TH = [5, 10, 15, 20]
RSI_SELL_TH = [80, 85, 90, 95]

print(f"\n{'RSI_buy':>7s} {'RSI_sell':>8s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'WR%':>5s}  {'$/t':>7s}  {'MaxDD':>11s}")
print("-" * 75)

rsi_results = {}
for buy_th in RSI_BUY_TH:
    for sell_th in RSI_SELL_TH:
        kwargs = {
            **LIVE_PARITY_KWARGS,
            "keltner_max_hold_m15": 20,
            "intraday_adaptive": False,
            "rsi_buy_threshold": buy_th,
            "rsi_sell_threshold": sell_th,
            "rsi_max_hold_m15": 8,
        }
        s = run_variant(data, f"D_RSI{buy_th}_{sell_th}", verbose=False,
                        **kwargs, spread_cost=0.30)
        n = s['n']
        avg = s['total_pnl'] / n if n > 0 else 0
        print(f"  {buy_th:>5d} {sell_th:>8d}  {n:>5d}  {s['sharpe']:>6.2f}  "
              f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  "
              f"${avg:>6.2f}  {fmt(s['max_dd'])}")
        rsi_results[(buy_th, sell_th)] = s

# Also test: Adaptive ON (current) but with wider RSI thresholds
print("\n  --- With Adaptive (current behavior) but wider thresholds ---")
for buy_th, sell_th in [(10, 90), (15, 85), (20, 80)]:
    kwargs = {
        **BASE,
        "rsi_buy_threshold": buy_th,
        "rsi_sell_threshold": sell_th,
    }
    s = run_variant(data, f"D_Adpt_RSI{buy_th}_{sell_th}", verbose=False,
                    **kwargs, spread_cost=0.30)
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    print(f"  Adaptive+RSI({buy_th}/{sell_th})  {n:>5d}  {s['sharpe']:>6.2f}  "
          f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  "
          f"${avg:>6.2f}  {fmt(s['max_dd'])}")

gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# EXP-E: KC SQUEEZE → EXPANSION SIGNAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("EXP-E: KC SQUEEZE -> EXPANSION CONFIDENCE SCORING")
print("  Squeeze: Bollinger Band (20,2) fits inside Keltner Channel")
print("  Hypothesis: breakout after squeeze is more reliable")
print("  Implementation: add squeeze detection, test as lot multiplier (1.2x when squeeze)")
print("=" * 80)

# We need to add BB to the data for squeeze detection
# Then run a custom engine wrapper

def add_bollinger(df, period=20, std_mult=2.0):
    """Add Bollinger Bands to dataframe."""
    df = df.copy()
    df['BB_mid'] = df['Close'].rolling(period).mean()
    df['BB_std'] = df['Close'].rolling(period).std()
    df['BB_upper'] = df['BB_mid'] + std_mult * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - std_mult * df['BB_std']
    df['squeeze'] = (df['BB_lower'] > df['KC_lower']) & (df['BB_upper'] < df['KC_upper'])
    df['squeeze_bars'] = 0
    squeeze_count = 0
    squeeze_bars_list = []
    for sq in df['squeeze']:
        if sq:
            squeeze_count += 1
        else:
            squeeze_count = 0
        squeeze_bars_list.append(squeeze_count)
    df['squeeze_bars'] = squeeze_bars_list
    df['squeeze_release'] = (~df['squeeze']) & (df['squeeze'].shift(1) == True)
    return df

# Add BB to both timeframes
data_sq = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
data_sq.h1_df = add_bollinger(data_sq.h1_df)
data_sq.m15_df = add_bollinger(data_sq.m15_df)

# Count squeeze events
h1_squeezes = data_sq.h1_df['squeeze'].sum()
h1_releases = data_sq.h1_df['squeeze_release'].sum()
print(f"\n  H1: {h1_squeezes} squeeze bars, {h1_releases} squeeze releases")
print(f"  H1: squeeze rate = {h1_squeezes/len(data_sq.h1_df)*100:.1f}%")

# Run baseline vs squeeze-filtered
# Can't easily modify engine to use squeeze, so do post-hoc analysis:
# Run baseline, then analyze which trades occurred during/after squeeze

s_base = run_variant(data_sq, "E_baseline", verbose=True,
                     **BASE, spread_cost=0.30)
trades = s_base.get('_trades', [])

# Match trades to squeeze state
squeeze_trades = []
nonsqueeze_trades = []
for t in trades:
    if t.strategy != 'keltner':
        continue
    entry_time = t.entry_time
    # Find H1 bar at entry
    h1_mask = data_sq.h1_df.index <= pd.Timestamp(entry_time)
    if h1_mask.any():
        h1_idx = h1_mask.sum() - 1
        if h1_idx >= 0 and h1_idx < len(data_sq.h1_df):
            row = data_sq.h1_df.iloc[h1_idx]
            # Was there a recent squeeze (within last 5 bars)?
            start_idx = max(0, h1_idx - 5)
            recent_squeeze = data_sq.h1_df.iloc[start_idx:h1_idx+1]['squeeze_release'].any()
            if recent_squeeze:
                squeeze_trades.append(t)
            else:
                nonsqueeze_trades.append(t)

sq_pnl = sum(t.pnl for t in squeeze_trades) if squeeze_trades else 0
nsq_pnl = sum(t.pnl for t in nonsqueeze_trades) if nonsqueeze_trades else 0
sq_n = len(squeeze_trades)
nsq_n = len(nonsqueeze_trades)
sq_avg = sq_pnl / sq_n if sq_n > 0 else 0
nsq_avg = nsq_pnl / nsq_n if nsq_n > 0 else 0
sq_wr = sum(1 for t in squeeze_trades if t.pnl > 0) / sq_n * 100 if sq_n > 0 else 0
nsq_wr = sum(1 for t in nonsqueeze_trades if t.pnl > 0) / nsq_n * 100 if nsq_n > 0 else 0

print(f"\n  Post-squeeze trades (release within 5 bars): N={sq_n}, "
      f"PnL=${sq_pnl:.0f}, $/t=${sq_avg:.2f}, WR={sq_wr:.1f}%")
print(f"  Non-squeeze trades:                          N={nsq_n}, "
      f"PnL=${nsq_pnl:.0f}, $/t=${nsq_avg:.2f}, WR={nsq_wr:.1f}%")

if sq_n >= 30 and nsq_n >= 30:
    print(f"  Squeeze edge: $/t difference = ${sq_avg - nsq_avg:+.2f}")
else:
    print(f"  WARNING: squeeze sample too small ({sq_n}) for reliable conclusion")

gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# EXP-F: MULTI-KC ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("EXP-F: MULTI-KC ENSEMBLE — Signal Overlap Scoring")
print("  Run 3 KC configs: (20,1.5), (25,1.2), (30,1.0)")
print("  Post-hoc: analyze if trades signaled by 2+ configs perform better")
print("=" * 80)

KC_CONFIGS = [
    (20, 1.5, "KC20_15"),
    (25, 1.2, "KC25_12"),
    (30, 1.0, "KC30_10"),
]

kc_trade_sets = {}
for ema, mult, name in KC_CONFIGS:
    d = DataBundle.load_custom(kc_ema=ema, kc_mult=mult)
    s = run_variant(d, f"F_{name}", verbose=False, **BASE, spread_cost=0.30)
    kc_trades = [t for t in s.get('_trades', []) if t.strategy == 'keltner']
    kc_trade_sets[name] = {
        'stats': s,
        'trades': kc_trades,
        'entry_keys': set()
    }
    # Create entry keys (rounded time + direction) for overlap detection
    for t in kc_trades:
        key = (pd.Timestamp(t.entry_time).floor('h'), t.direction)
        kc_trade_sets[name]['entry_keys'].add(key)

    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    kc_n = len(kc_trades)
    kc_pnl = sum(t.pnl for t in kc_trades)
    kc_avg = kc_pnl / kc_n if kc_n > 0 else 0
    print(f"\n  {name}: Total N={n}, Sharpe={s['sharpe']:.2f}, "
          f"KC trades={kc_n}, KC $/t=${kc_avg:.2f}")

# Overlap analysis on the primary (25,1.2) trades
primary = kc_trade_sets["KC25_12"]
overlap_2 = []
overlap_3 = []
single = []

for t in primary['trades']:
    key = (pd.Timestamp(t.entry_time).floor('h'), t.direction)
    count = sum(1 for name in kc_trade_sets if key in kc_trade_sets[name]['entry_keys'])
    if count >= 3:
        overlap_3.append(t)
    elif count >= 2:
        overlap_2.append(t)
    else:
        single.append(t)

for label, trades in [("3-config overlap", overlap_3), ("2-config overlap", overlap_2), ("Single config", single)]:
    if not trades:
        print(f"\n  {label}: N=0")
        continue
    n = len(trades)
    pnl = sum(t.pnl for t in trades)
    avg = pnl / n
    wr = sum(1 for t in trades if t.pnl > 0) / n * 100
    print(f"\n  {label}: N={n}, PnL=${pnl:.0f}, $/t=${avg:.2f}, WR={wr:.1f}%")

gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

elapsed = time.time() - t_total
print(f"\n  Total runtime: {elapsed/60:.1f} minutes")
print(f"  Completed: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nResults saved to {OUTPUT_FILE}")
