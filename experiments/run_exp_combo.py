#!/usr/bin/env python3
"""
COMBO TEST: Stack all K-Fold 6/6 PASS improvements and test interactions.
===========================================================================
Layer 0: LIVE_PARITY baseline @ $0.30
Layer 1: + MaxHold=20
Layer 2: + Choppy threshold 0.50
Layer 3: + Tight_all regime trail (EXP-K)
Layer 4: + KC mid reversion exit min_bars=2 (EXP-U)
Layer 5: + Breakout strength sizing (EXP-V) — post-hoc analysis

Each layer is tested individually AND cumulatively, then K-Fold 6-fold on the
best cumulative stack.
"""
import sys, os, io, time, copy
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd

import research_config as config
from indicators import check_exit_signal
from backtest.engine import BacktestEngine
from backtest.runner import DataBundle, LIVE_PARITY_KWARGS, run_variant, run_kfold

OUTPUT_FILE = "exp_combo_output.txt"
SPREAD = 0.30

TIGHT_ALL_REGIME = {
    'low':    {'trail_act': 0.5,  'trail_dist': 0.15},
    'normal': {'trail_act': 0.35, 'trail_dist': 0.10},
    'high':   {'trail_act': 0.20, 'trail_dist': 0.03},
}


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


_ORIGINAL_CHECK_EXITS = BacktestEngine._check_exits


def make_kc_mid_exit(min_bars=2):
    """KC mid reversion exit — monkey-patches _check_exits."""
    def _check_exits(self, m15_window, h1_window, bar, bar_time):
        high = float(bar['High'])
        low = float(bar['Low'])
        close = float(bar['Close'])

        if self._regime_config and h1_window is not None and len(h1_window) > 0:
            atr_pct = self._get_atr_percentile(h1_window)
            regime = 'low' if atr_pct < 0.30 else ('high' if atr_pct > 0.70 else 'normal')
            rc = self._regime_config.get(regime, {})
            self._trail_act = rc.get('trail_act', self._trail_act_base)
            self._trail_dist = rc.get('trail_dist', self._trail_dist_base)
            self._sl_atr_mult = rc.get('sl', self._sl_atr_mult_base)

        for pos in list(self.positions):
            pos.bars_held += 1
            reason = None
            exit_price = close

            if pos.direction == 'BUY':
                if low <= pos.sl_price:
                    reason = 'SL'
                    exit_price = pos.sl_price
                elif high >= pos.tp_price:
                    reason = 'TP'
                    exit_price = pos.tp_price
            else:
                if high >= pos.sl_price:
                    reason = 'SL'
                    exit_price = pos.sl_price
                elif low <= pos.tp_price:
                    reason = 'TP'
                    exit_price = pos.tp_price

            if (
                not reason
                and pos.strategy == 'keltner'
                and pos.bars_held >= min_bars
                and h1_window is not None
                and len(h1_window) > 0
            ):
                h1_last = h1_window.iloc[-1]
                h1_close = float(h1_last['Close'])
                kc_m = h1_last.get('KC_mid', None)
                if kc_m is not None and not pd.isna(kc_m):
                    km = float(kc_m)
                    if pos.direction == 'BUY' and h1_close < km:
                        reason = 'kc_mid_revert'
                        exit_price = close
                    elif pos.direction == 'SELL' and h1_close > km:
                        reason = 'kc_mid_revert'
                        exit_price = close

            if not reason and pos.strategy == 'keltner' and config.TRAILING_STOP_ENABLED:
                act_atr = self._trail_act or config.TRAILING_ACTIVATE_ATR
                dist_atr = self._trail_dist or config.TRAILING_DISTANCE_ATR
                atr = self._get_h1_atr(h1_window)
                if atr > 0:
                    if (
                        self._atr_spike_protection
                        and pos.entry_atr > 0
                        and atr > pos.entry_atr * self._atr_spike_threshold
                    ):
                        dist_atr = dist_atr * self._atr_spike_trail_mult
                        self.atr_spike_tighten_count += 1
                    if pos.direction == 'BUY':
                        float_profit = high - pos.entry_price
                        pos.extreme_price = max(pos.extreme_price, high)
                    else:
                        float_profit = pos.entry_price - low
                        pos.extreme_price = min(pos.extreme_price, low) if pos.extreme_price > 0 else low
                    if float_profit >= atr * act_atr:
                        trail_distance = atr * dist_atr
                        if pos.direction == 'BUY':
                            new_trail = pos.extreme_price - trail_distance
                            pos.trailing_stop_price = max(pos.trailing_stop_price, new_trail)
                            if low <= pos.trailing_stop_price:
                                reason = 'Trailing'
                                exit_price = pos.trailing_stop_price
                        else:
                            new_trail = pos.extreme_price + trail_distance
                            if pos.trailing_stop_price <= 0:
                                pos.trailing_stop_price = new_trail
                            else:
                                pos.trailing_stop_price = min(pos.trailing_stop_price, new_trail)
                            if high >= pos.trailing_stop_price:
                                reason = 'Trailing'
                                exit_price = pos.trailing_stop_price

            if not reason and pos.strategy == 'keltner':
                pass
            elif not reason and pos.strategy in ('m15_rsi', 'm5_rsi'):
                if pos.bars_held > 1:
                    exit_sig = check_exit_signal(m15_window, pos.strategy, pos.direction)
                    if exit_sig:
                        reason = exit_sig
                        exit_price = close
            elif not reason and pos.strategy not in ('keltner',):
                if h1_window is not None and len(h1_window) > 2:
                    exit_sig = check_exit_signal(h1_window, pos.strategy, pos.direction)
                    if exit_sig:
                        reason = exit_sig
                        exit_price = close

            if (
                not reason
                and self._time_decay_tp
                and pos.strategy == 'keltner'
                and pos.bars_held >= self._td_start_bars
            ):
                trailing_activated = pos.trailing_stop_price > 0
                if not trailing_activated:
                    atr_td = self._get_h1_atr(h1_window) if h1_window is not None else 0
                    if atr_td > 0:
                        decay_bars = pos.bars_held - self._td_start_bars
                        min_profit_atr = max(0.0, self._td_atr_start - decay_bars * self._td_atr_step_per_bar)
                        min_profit = atr_td * min_profit_atr
                        if pos.direction == 'BUY':
                            float_pnl = close - pos.entry_price
                        else:
                            float_pnl = pos.entry_price - close
                        if float_pnl >= min_profit and float_pnl > 0:
                            reason = 'TimeDecayTP'
                            exit_price = close
                            self.time_decay_tp_count += 1

            if not reason:
                if pos.strategy == 'm15_rsi':
                    max_hold = self._rsi_max_hold_m15 if self._rsi_max_hold_m15 > 0 else 15
                elif pos.strategy == 'orb' and self._orb_max_hold_m15 > 0:
                    max_hold = self._orb_max_hold_m15
                elif pos.strategy == 'keltner' and self._keltner_max_hold_m15 > 0:
                    max_hold = self._keltner_max_hold_m15
                else:
                    max_hold_h1 = config.STRATEGIES.get(pos.strategy, {}).get('max_hold_bars', 15)
                    max_hold = max_hold_h1 * 4
                if pos.bars_held >= max_hold:
                    reason = f'Timeout:{pos.bars_held}>={max_hold}'
                    exit_price = close

            if reason:
                self._close_position(pos, exit_price, bar_time, reason)

    return _check_exits


print("=" * 80)
print("COMBO TEST: Stacking all K-Fold PASS improvements")
print(f"Started: {datetime.now()}")
print("=" * 80)

t0 = time.time()
data = DataBundle.load_default()

BASE = {**LIVE_PARITY_KWARGS}

# ── Layer definitions ──
layers = [
    ("L0: LIVE_PARITY baseline", {**BASE}, False),
    ("L1: + MaxHold=20", {**BASE, "keltner_max_hold_m15": 20}, False),
    ("L2: + Choppy 0.50", {**BASE, "keltner_max_hold_m15": 20, "choppy_threshold": 0.50}, False),
    ("L3: + Tight_all trail", {**BASE, "keltner_max_hold_m15": 20, "choppy_threshold": 0.50, "regime_config": TIGHT_ALL_REGIME}, False),
    ("L4: + KC mid revert", {**BASE, "keltner_max_hold_m15": 20, "choppy_threshold": 0.50, "regime_config": TIGHT_ALL_REGIME}, True),
]

# ── Part 1: Cumulative layer stacking ──
print("\n--- Part 1: Cumulative Layer Stacking @ $0.30 ---")
print(f"{'Layer':<30s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'WR%':>5s}  {'$/t':>7s}  {'MaxDD':>11s}")
print("-" * 90)

results = {}
for name, kwargs, needs_patch in layers:
    if needs_patch:
        BacktestEngine._check_exits = make_kc_mid_exit(min_bars=2)
    else:
        BacktestEngine._check_exits = _ORIGINAL_CHECK_EXITS

    s = run_variant(data, name[:15], verbose=False, **kwargs, spread_cost=SPREAD)
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    print(f"  {name:<28s}  {n:>5d}  {s['sharpe']:>6.2f}  "
          f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  "
          f"${avg:>6.2f}  {fmt(s['max_dd'])}")
    results[name] = s

BacktestEngine._check_exits = _ORIGINAL_CHECK_EXITS

# ── Part 2: Individual contribution (each improvement alone on baseline) ──
print("\n--- Part 2: Individual Contribution (each alone vs L0) ---")
individual = [
    ("MaxHold=20 only", {**BASE, "keltner_max_hold_m15": 20}, False),
    ("Choppy 0.50 only", {**BASE, "choppy_threshold": 0.50}, False),
    ("Tight_all only", {**BASE, "regime_config": TIGHT_ALL_REGIME}, False),
    ("KC mid only", {**BASE}, True),
]

base_sharpe = results["L0: LIVE_PARITY baseline"]['sharpe']
print(f"  Baseline Sharpe: {base_sharpe:.2f}")
print(f"{'Improvement':<25s}  {'N':>5s}  {'Sharpe':>6s}  {'Delta':>7s}  {'PnL':>11s}")
print("-" * 60)

for name, kwargs, needs_patch in individual:
    if needs_patch:
        BacktestEngine._check_exits = make_kc_mid_exit(min_bars=2)
    else:
        BacktestEngine._check_exits = _ORIGINAL_CHECK_EXITS

    s = run_variant(data, name[:15], verbose=False, **kwargs, spread_cost=SPREAD)
    delta = s['sharpe'] - base_sharpe
    print(f"  {name:<23s}  {s['n']:>5d}  {s['sharpe']:>6.2f}  {delta:>+7.2f}  {fmt(s['total_pnl'])}")

BacktestEngine._check_exits = _ORIGINAL_CHECK_EXITS

# ── Part 3: K-Fold for full stack (L4) ──
print("\n--- Part 3: K-Fold 6-fold for Full Stack (L4) vs Baseline (L0) ---")

full_stack_kw = {
    **BASE,
    "keltner_max_hold_m15": 20,
    "choppy_threshold": 0.50,
    "regime_config": TIGHT_ALL_REGIME,
}

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-01"),
]

wins = 0
print(f"\n{'Fold':<8s}  {'Baseline':>8s}  {'FullStack':>9s}  {'FS+KCmid':>9s}  {'Delta_noKC':>10s}  {'Delta_KC':>10s}")
print("-" * 65)

for fold_name, start, end in FOLDS:
    fold_data = data.slice(start, end)
    if len(fold_data.m15_df) < 1000:
        continue

    BacktestEngine._check_exits = _ORIGINAL_CHECK_EXITS
    s_base = run_variant(fold_data, f"B_{fold_name}", verbose=False, **BASE, spread_cost=SPREAD)

    s_stack = run_variant(fold_data, f"S_{fold_name}", verbose=False, **full_stack_kw, spread_cost=SPREAD)

    BacktestEngine._check_exits = make_kc_mid_exit(min_bars=2)
    s_stack_kc = run_variant(fold_data, f"SK_{fold_name}", verbose=False, **full_stack_kw, spread_cost=SPREAD)
    BacktestEngine._check_exits = _ORIGINAL_CHECK_EXITS

    d_no_kc = s_stack['sharpe'] - s_base['sharpe']
    d_kc = s_stack_kc['sharpe'] - s_base['sharpe']
    won = d_kc > 0
    if won:
        wins += 1
    mark = 'V' if won else 'X'
    print(f"  {fold_name:<6s}  {s_base['sharpe']:>8.2f}  {s_stack['sharpe']:>9.2f}  "
          f"{s_stack_kc['sharpe']:>9.2f}  {d_no_kc:>+10.2f}  {d_kc:>+10.2f} {mark}")

print(f"\n  Full Stack + KC mid: {wins}/6 {'PASS' if wins >= 5 else 'FAIL'}")

# ── Part 4: Full stack @ $0.50 stress test ──
print("\n--- Part 4: Full Stack @ $0.50 Stress Test ---")
for spread_val in [0.00, 0.30, 0.50]:
    BacktestEngine._check_exits = make_kc_mid_exit(min_bars=2)
    s = run_variant(data, f"FS_${spread_val}", verbose=False, **full_stack_kw, spread_cost=spread_val)
    BacktestEngine._check_exits = _ORIGINAL_CHECK_EXITS
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    print(f"  sp${spread_val:.2f}  N={n:>5d}  Sharpe={s['sharpe']:>6.2f}  "
          f"PnL={fmt(s['total_pnl'])}  WR={s['win_rate']:>5.1f}%  MaxDD={fmt(s['max_dd'])}")

# ── Part 5: Year-by-year for full stack ──
print("\n--- Part 5: Year-by-Year Full Stack + KC mid @ $0.30 ---")
BacktestEngine._check_exits = make_kc_mid_exit(min_bars=2)
s_full = run_variant(data, "FS_yearly", verbose=True, **full_stack_kw, spread_cost=SPREAD)
BacktestEngine._check_exits = _ORIGINAL_CHECK_EXITS
trades = s_full.get('_trades', [])

years = sorted(set(t.exit_time.year for t in trades if t.exit_time))
print(f"  {'Year':>4s}  {'N':>5s}  {'PnL':>10s}  {'WR%':>5s}  {'$/t':>7s}")
print("  " + "-" * 40)
for y in years:
    yt = [t for t in trades if t.exit_time and t.exit_time.year == y]
    n = len(yt)
    pnl = sum(t.pnl for t in yt)
    wr = sum(1 for t in yt if t.pnl > 0) / n * 100 if n > 0 else 0
    avg = pnl / n if n > 0 else 0
    print(f"  {y:>4d}  {n:>5d}  ${pnl:>9,.0f}  {wr:>5.1f}%  ${avg:>6.2f}")

# ── Part 6: Exit reason breakdown for full stack ──
print("\n--- Part 6: Exit Reason Breakdown (Full Stack + KC mid) ---")
from collections import Counter
reasons = Counter()
reason_pnl = {}
for t in trades:
    r = t.exit_reason.split(':')[0] if ':' in t.exit_reason else t.exit_reason
    reasons[r] += 1
    reason_pnl.setdefault(r, []).append(t.pnl)

print(f"  {'Reason':<20s}  {'N':>5s}  {'PnL':>11s}  {'WR%':>5s}  {'$/t':>7s}")
print("  " + "-" * 55)
for r, n in reasons.most_common():
    pnls = reason_pnl[r]
    pnl = sum(pnls)
    wr = sum(1 for p in pnls if p > 0) / n * 100
    avg = pnl / n
    print(f"  {r:<20s}  {n:>5d}  {fmt(pnl)}  {wr:>5.1f}%  ${avg:>6.2f}")

elapsed = time.time() - t0
print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
print(f"Completed: {datetime.now()}")
sys.stdout = tee.stdout
tee.close()
print(f"Results saved to {OUTPUT_FILE}")
