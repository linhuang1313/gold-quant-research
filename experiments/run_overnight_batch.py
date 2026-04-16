#!/usr/bin/env python3
"""
OVERNIGHT BATCH: Wait for combo to finish, then launch follow-up experiments.
==============================================================================
Exp 1: TDTP OFF K-Fold verification (EXP-C showed Sharpe -0.22)
Exp 2: Full stack + EXP-L trend weights (0.2/0.3/0.2/0.3)
Exp 3: Full stack WITHOUT KC mid (L3 only) — separate K-Fold
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
from backtest.runner import DataBundle, LIVE_PARITY_KWARGS, run_variant

OUTPUT_FILE = "overnight_batch_output.txt"
SPREAD = 0.30

TIGHT_ALL_REGIME = {
    'low':    {'trail_act': 0.5,  'trail_dist': 0.15},
    'normal': {'trail_act': 0.35, 'trail_dist': 0.10},
    'high':   {'trail_act': 0.20, 'trail_dist': 0.03},
}

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-10"),
]

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
    """KC mid reversion exit."""
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
                    reason, exit_price = 'SL', pos.sl_price
                elif high >= pos.tp_price:
                    reason, exit_price = 'TP', pos.tp_price
            else:
                if high >= pos.sl_price:
                    reason, exit_price = 'SL', pos.sl_price
                elif low <= pos.tp_price:
                    reason, exit_price = 'TP', pos.tp_price

            if (not reason and pos.strategy == 'keltner'
                and pos.bars_held >= min_bars
                and h1_window is not None and len(h1_window) > 0):
                h1_last = h1_window.iloc[-1]
                kc_m = h1_last.get('KC_mid', None)
                if kc_m is not None and not pd.isna(kc_m):
                    km = float(kc_m)
                    h1_close = float(h1_last['Close'])
                    if (pos.direction == 'BUY' and h1_close < km) or \
                       (pos.direction == 'SELL' and h1_close > km):
                        reason, exit_price = 'kc_mid_revert', close

            if not reason and pos.strategy == 'keltner' and config.TRAILING_STOP_ENABLED:
                act_atr = self._trail_act or config.TRAILING_ACTIVATE_ATR
                dist_atr = self._trail_dist or config.TRAILING_DISTANCE_ATR
                atr = self._get_h1_atr(h1_window)
                if atr > 0:
                    if (self._atr_spike_protection and pos.entry_atr > 0
                        and atr > pos.entry_atr * self._atr_spike_threshold):
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
                                reason, exit_price = 'Trailing', pos.trailing_stop_price
                        else:
                            new_trail = pos.extreme_price + trail_distance
                            if pos.trailing_stop_price <= 0:
                                pos.trailing_stop_price = new_trail
                            else:
                                pos.trailing_stop_price = min(pos.trailing_stop_price, new_trail)
                            if high >= pos.trailing_stop_price:
                                reason, exit_price = 'Trailing', pos.trailing_stop_price

            if not reason and pos.strategy == 'keltner':
                pass
            elif not reason and pos.strategy in ('m15_rsi', 'm5_rsi'):
                if pos.bars_held > 1:
                    exit_sig = check_exit_signal(m15_window, pos.strategy, pos.direction)
                    if exit_sig:
                        reason, exit_price = exit_sig, close
            elif not reason and pos.strategy not in ('keltner',):
                if h1_window is not None and len(h1_window) > 2:
                    exit_sig = check_exit_signal(h1_window, pos.strategy, pos.direction)
                    if exit_sig:
                        reason, exit_price = exit_sig, close

            if (not reason and self._time_decay_tp
                and pos.strategy == 'keltner'
                and pos.bars_held >= self._td_start_bars):
                trailing_activated = pos.trailing_stop_price > 0
                if not trailing_activated:
                    atr_td = self._get_h1_atr(h1_window) if h1_window is not None else 0
                    if atr_td > 0:
                        decay_bars = pos.bars_held - self._td_start_bars
                        min_profit_atr = max(0.0, self._td_atr_start - decay_bars * self._td_atr_step_per_bar)
                        min_profit = atr_td * min_profit_atr
                        float_pnl = (close - pos.entry_price) if pos.direction == 'BUY' else (pos.entry_price - close)
                        if float_pnl >= min_profit and float_pnl > 0:
                            reason, exit_price = 'TimeDecayTP', close
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

# ==============================================================================
print("=" * 80)
print("OVERNIGHT BATCH EXPERIMENTS")
print(f"Started: {datetime.now()}")
print("=" * 80)

t0 = time.time()
data = DataBundle.load_default()

BASE = {**LIVE_PARITY_KWARGS}

# Full stack WITHOUT KC mid (L3)
L3_KW = {
    **BASE,
    "keltner_max_hold_m15": 20,
    "choppy_threshold": 0.50,
    "regime_config": TIGHT_ALL_REGIME,
}

# ==============================================================================
# EXP-1: TDTP OFF K-Fold
# EXP-C showed: TD OFF Sharpe=2.85 vs TD ON=2.62 (delta=-0.22)
# Need K-Fold to confirm whether TDTP should be disabled
# ==============================================================================
print("\n" + "=" * 80)
print("EXP-1: TIME DECAY TP OFF — K-Fold Verification")
print("  Hypothesis: TDTP hurts Sharpe by -0.22, should we disable it?")
print("  Test: L3 stack with TDTP OFF vs L3 stack with TDTP ON")
print("=" * 80)

L3_TDTP_OFF = {**L3_KW, "time_decay_tp": False}

# Full sample first
BacktestEngine._check_exits = _ORIGINAL_CHECK_EXITS
s_on = run_variant(data, "TDTP_ON", verbose=False, **L3_KW, spread_cost=SPREAD)
s_off = run_variant(data, "TDTP_OFF", verbose=False, **L3_TDTP_OFF, spread_cost=SPREAD)
print(f"\n  Full sample:")
print(f"    TDTP ON:   N={s_on['n']:>5d}  Sharpe={s_on['sharpe']:>6.2f}  PnL={fmt(s_on['total_pnl'])}  WR={s_on['win_rate']:>5.1f}%")
print(f"    TDTP OFF:  N={s_off['n']:>5d}  Sharpe={s_off['sharpe']:>6.2f}  PnL={fmt(s_off['total_pnl'])}  WR={s_off['win_rate']:>5.1f}%")
print(f"    Delta: Sharpe {s_off['sharpe'] - s_on['sharpe']:>+.2f}")

# K-Fold
wins = 0
print(f"\n  K-Fold:")
for fold_name, start, end in FOLDS:
    fold_data = data.slice(start, end)
    if len(fold_data.m15_df) < 1000:
        continue
    s_f_on = run_variant(fold_data, f"TD_ON_{fold_name}", verbose=False, **L3_KW, spread_cost=SPREAD)
    s_f_off = run_variant(fold_data, f"TD_OFF_{fold_name}", verbose=False, **L3_TDTP_OFF, spread_cost=SPREAD)
    delta = s_f_off['sharpe'] - s_f_on['sharpe']
    won = delta > 0
    if won: wins += 1
    print(f"    {fold_name}: ON={s_f_on['sharpe']:>6.2f}  OFF={s_f_off['sharpe']:>6.2f}  delta={delta:>+.2f} {'V' if won else 'X'}")
print(f"    Result: {wins}/6 {'PASS' if wins >= 5 else 'FAIL'}")

# ==============================================================================
# EXP-2: L3 stack K-Fold (WITHOUT KC mid) — separate validation
# This tests: MaxHold=20 + Choppy 0.50 + Tight_all trail
# ==============================================================================
print("\n" + "=" * 80)
print("EXP-2: L3 STACK (no KC mid) — K-Fold Verification")
print("  Tests cumulative: MaxHold=20 + Choppy 0.50 + Tight_all trail")
print("=" * 80)

BacktestEngine._check_exits = _ORIGINAL_CHECK_EXITS

# Full sample
s_l3 = run_variant(data, "L3_full", verbose=False, **L3_KW, spread_cost=SPREAD)
s_baseline = run_variant(data, "L0_full", verbose=False, **BASE, spread_cost=SPREAD)
print(f"\n  Full sample:")
print(f"    L0 Baseline: Sharpe={s_baseline['sharpe']:>6.2f}  PnL={fmt(s_baseline['total_pnl'])}")
print(f"    L3 Stack:    Sharpe={s_l3['sharpe']:>6.2f}  PnL={fmt(s_l3['total_pnl'])}")

wins = 0
print(f"\n  K-Fold:")
for fold_name, start, end in FOLDS:
    fold_data = data.slice(start, end)
    if len(fold_data.m15_df) < 1000:
        continue
    sb = run_variant(fold_data, f"B_{fold_name}", verbose=False, **BASE, spread_cost=SPREAD)
    sl = run_variant(fold_data, f"L3_{fold_name}", verbose=False, **L3_KW, spread_cost=SPREAD)
    delta = sl['sharpe'] - sb['sharpe']
    won = delta > 0
    if won: wins += 1
    print(f"    {fold_name}: Base={sb['sharpe']:>6.2f}  L3={sl['sharpe']:>6.2f}  delta={delta:>+.2f} {'V' if won else 'X'}")
print(f"    Result: {wins}/6 {'PASS' if wins >= 5 else 'FAIL'}")

# Year-by-year for L3
print(f"\n  Year-by-year L3:")
s_l3_detail = run_variant(data, "L3_yearly", verbose=True, **L3_KW, spread_cost=SPREAD)
trades = s_l3_detail.get('_trades', [])
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

# ==============================================================================
# EXP-3: L3 + Trend Weights optimization (EXP-L)
# Best weights: (0.2, 0.3, 0.2, 0.3) — K-Fold 6/6 PASS standalone
# Does it still help when stacked on L3?
# ==============================================================================
print("\n" + "=" * 80)
print("EXP-3: L3 + TREND WEIGHTS (0.2/0.3/0.2/0.3)")
print("  EXP-L showed K-Fold 6/6 PASS standalone. Test stacking on L3.")
print("=" * 80)

L3_WEIGHTS = {
    **L3_KW,
    "trend_weights": (0.2, 0.3, 0.2, 0.3),
}

s_l3_w = run_variant(data, "L3+W_full", verbose=False, **L3_WEIGHTS, spread_cost=SPREAD)
print(f"\n  Full sample:")
print(f"    L3:          Sharpe={s_l3['sharpe']:>6.2f}  PnL={fmt(s_l3['total_pnl'])}  N={s_l3['n']}")
print(f"    L3+Weights:  Sharpe={s_l3_w['sharpe']:>6.2f}  PnL={fmt(s_l3_w['total_pnl'])}  N={s_l3_w['n']}")
print(f"    Delta: Sharpe {s_l3_w['sharpe'] - s_l3['sharpe']:>+.2f}")

wins = 0
print(f"\n  K-Fold:")
for fold_name, start, end in FOLDS:
    fold_data = data.slice(start, end)
    if len(fold_data.m15_df) < 1000:
        continue
    sl = run_variant(fold_data, f"L3_{fold_name}", verbose=False, **L3_KW, spread_cost=SPREAD)
    sw = run_variant(fold_data, f"L3W_{fold_name}", verbose=False, **L3_WEIGHTS, spread_cost=SPREAD)
    delta = sw['sharpe'] - sl['sharpe']
    won = delta > 0
    if won: wins += 1
    print(f"    {fold_name}: L3={sl['sharpe']:>6.2f}  L3+W={sw['sharpe']:>6.2f}  delta={delta:>+.2f} {'V' if won else 'X'}")
print(f"    Result: {wins}/6 {'PASS' if wins >= 5 else 'FAIL'}")

# ==============================================================================
# EXP-4: Stress test — L3 at $0.50 spread
# ==============================================================================
print("\n" + "=" * 80)
print("EXP-4: L3 STACK STRESS TEST @ various spread levels")
print("=" * 80)

BacktestEngine._check_exits = _ORIGINAL_CHECK_EXITS
for sp in [0.00, 0.20, 0.30, 0.40, 0.50, 0.60]:
    s = run_variant(data, f"L3_sp{sp}", verbose=False, **L3_KW, spread_cost=sp)
    print(f"  sp${sp:.2f}: N={s['n']:>5d}  Sharpe={s['sharpe']:>6.2f}  "
          f"PnL={fmt(s['total_pnl'])}  WR={s['win_rate']:>5.1f}%  MaxDD={fmt(s['max_dd'])}")

# ==============================================================================
elapsed = time.time() - t0
print(f"\n{'=' * 80}")
print(f"Total runtime: {elapsed/60:.1f} minutes")
print(f"Completed: {datetime.now()}")
print(f"{'=' * 80}")

sys.stdout = tee.stdout
tee.close()
print(f"Results saved to {OUTPUT_FILE}")
