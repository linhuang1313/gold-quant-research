#!/usr/bin/env python3
"""
L4 AUDIT: KC Mid Reversion Exit — Vulnerability Detection
==========================================================
FIX1: Look-Ahead Bias — use closed H1 bar only (iloc[-2]) vs current (iloc[-1])
FIX2: Exit Priority — KC mid AFTER trailing vs BEFORE (current)
FIX3: min_bars sensitivity — 2/4/6/8/12
FIX4: Profit filter — KC mid only on losing / winning / all trades
FIX5: Combined best fix — full K-Fold + stress test
"""
import sys, os, io, time, copy
from datetime import datetime
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd

import research_config as config
from indicators import check_exit_signal
from backtest.engine import BacktestEngine
from backtest.runner import DataBundle, LIVE_PARITY_KWARGS, run_variant, run_kfold

OUTPUT_FILE = "exp_l4_audit_output.txt"
SPREAD = 0.30

TIGHT_ALL_REGIME = {
    'low':    {'trail_act': 0.5,  'trail_dist': 0.15},
    'normal': {'trail_act': 0.35, 'trail_dist': 0.10},
    'high':   {'trail_act': 0.20, 'trail_dist': 0.03},
}

L3_KW = {
    **LIVE_PARITY_KWARGS,
    "keltner_max_hold_m15": 20,
    "choppy_threshold": 0.50,
    "regime_config": TIGHT_ALL_REGIME,
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

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-01"),
]


# ═══════════════════════════════════════════════════════════════
# KC Mid Exit Variants
# ═══════════════════════════════════════════════════════════════

def make_kc_mid_exit(min_bars=2, use_closed_h1=False, after_trailing=False,
                     profit_filter=None):
    """
    KC mid reversion exit with configurable behavior.

    Args:
        min_bars: minimum bars_held before KC mid can trigger
        use_closed_h1: if True, use h1_window.iloc[-2] (last closed H1 bar)
                       instead of iloc[-1] (current, possibly unclosed)
        after_trailing: if True, check KC mid AFTER trailing (not before)
        profit_filter: None=all, 'loss_only'=only when floating loss,
                       'profit_only'=only when floating profit
    """
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

            # SL/TP
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

            # --- KC mid reversion (BEFORE trailing if after_trailing=False) ---
            if not after_trailing and not reason:
                reason, exit_price = _kc_mid_check(
                    pos, h1_window, close, min_bars, use_closed_h1, profit_filter,
                    exit_price
                )

            # Trailing
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

            # --- KC mid reversion (AFTER trailing if after_trailing=True) ---
            if after_trailing and not reason:
                reason, exit_price = _kc_mid_check(
                    pos, h1_window, close, min_bars, use_closed_h1, profit_filter,
                    exit_price
                )

            # Signal exit
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

            # TimeDecayTP
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

            # Timeout
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


def _kc_mid_check(pos, h1_window, close, min_bars, use_closed_h1, profit_filter,
                  default_exit_price):
    """Shared KC mid reversion logic. Returns (reason, exit_price) or (None, default)."""
    if pos.strategy != 'keltner':
        return None, default_exit_price
    if pos.bars_held < min_bars:
        return None, default_exit_price
    if h1_window is None or len(h1_window) < 2:
        return None, default_exit_price

    if use_closed_h1:
        h1_bar = h1_window.iloc[-2]
    else:
        h1_bar = h1_window.iloc[-1]

    h1_close = float(h1_bar['Close'])
    kc_m = h1_bar.get('KC_mid', None)
    if kc_m is None or pd.isna(kc_m):
        return None, default_exit_price
    km = float(kc_m)

    if profit_filter is not None:
        if pos.direction == 'BUY':
            float_pnl = close - pos.entry_price
        else:
            float_pnl = pos.entry_price - close
        if profit_filter == 'loss_only' and float_pnl >= 0:
            return None, default_exit_price
        if profit_filter == 'profit_only' and float_pnl < 0:
            return None, default_exit_price

    if pos.direction == 'BUY' and h1_close < km:
        return 'kc_mid_revert', close
    elif pos.direction == 'SELL' and h1_close > km:
        return 'kc_mid_revert', close

    return None, default_exit_price


def run_and_report(data, label, needs_patch, patch_kwargs=None, spread=SPREAD, **extra_kw):
    """Run a variant, return stats, print one-line summary."""
    kw = {**L3_KW, **extra_kw}
    if needs_patch:
        BacktestEngine._check_exits = make_kc_mid_exit(**(patch_kwargs or {}))
    else:
        BacktestEngine._check_exits = _ORIGINAL_CHECK_EXITS

    s = run_variant(data, label[:15], verbose=False, **kw, spread_cost=spread)
    BacktestEngine._check_exits = _ORIGINAL_CHECK_EXITS

    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    return s


def print_row(label, s):
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    print(f"  {label:<35s}  {n:>5d}  {s['sharpe']:>6.2f}  "
          f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  "
          f"${avg:>6.2f}  {fmt(s['max_dd'])}")


def print_exit_breakdown(s, label=""):
    """Print exit reason breakdown from trades."""
    trades = s.get('_trades', [])
    if not trades:
        return
    reasons = Counter()
    reason_pnl = {}
    for t in trades:
        r = t.exit_reason.split(':')[0] if ':' in t.exit_reason else t.exit_reason
        reasons[r] += 1
        reason_pnl.setdefault(r, []).append(t.pnl)

    if label:
        print(f"\n  Exit breakdown for {label}:")
    print(f"  {'Reason':<20s}  {'N':>5s}  {'PnL':>11s}  {'WR%':>5s}  {'$/t':>7s}")
    print("  " + "-" * 55)
    for r, n in reasons.most_common():
        pnls = reason_pnl[r]
        pnl = sum(pnls)
        wr = sum(1 for p in pnls if p > 0) / n * 100
        avg = pnl / n
        print(f"  {r:<20s}  {n:>5d}  {fmt(pnl)}  {wr:>5.1f}%  ${avg:>6.2f}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

print("=" * 80)
print("L4 AUDIT: KC Mid Reversion Exit — Vulnerability Detection")
print(f"Started: {datetime.now()}")
print("=" * 80)

t0 = time.time()
data = DataBundle.load_default()
BASE = {**LIVE_PARITY_KWARGS}

header = f"{'Variant':<35s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'WR%':>5s}  {'$/t':>7s}  {'MaxDD':>11s}"

# ── Baselines ──
print("\n--- Baselines ---")
print(header)
print("-" * 95)

s_l3 = run_and_report(data, "L3 (no KC mid)", False)
print_row("L3 (no KC mid)", s_l3)

s_l4_orig = run_and_report(data, "L4 ORIGINAL (iloc[-1])", True,
                           {"min_bars": 2, "use_closed_h1": False, "after_trailing": False})
print_row("L4 ORIGINAL (iloc[-1])", s_l4_orig)


# ═══════════════════════════════════════════════════════════════
# FIX1: Look-Ahead Bias
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FIX1: LOOK-AHEAD BIAS — Does KC mid use future H1 data?")
print("=" * 80)
print(f"\nKey question: Does using the CLOSED H1 bar (iloc[-2]) instead of")
print(f"the current (possibly unclosed) H1 bar (iloc[-1]) change results?")
print(f"If Sharpe drops significantly, the original result is polluted.\n")
print(header)
print("-" * 95)

# A: Original (already computed)
print_row("A: Original (iloc[-1])", s_l4_orig)

# B: Use closed H1 bar
s_fix1b = run_and_report(data, "B: Closed H1 (iloc[-2])", True,
                         {"min_bars": 2, "use_closed_h1": True, "after_trailing": False})
print_row("B: Closed H1 (iloc[-2])", s_fix1b)

# C: Only check at H1 boundary (minute==0)
# For this we need a special variant
def make_kc_mid_h1_boundary_only(min_bars=2):
    """KC mid that only fires on H1 boundaries (bar_time.minute == 0)."""
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

            # KC mid ONLY at H1 boundary, using the PREVIOUS (closed) H1 bar
            if (
                not reason
                and bar_time.minute == 0
                and pos.strategy == 'keltner'
                and pos.bars_held >= min_bars
                and h1_window is not None
                and len(h1_window) >= 2
            ):
                h1_prev = h1_window.iloc[-2]
                h1_close = float(h1_prev['Close'])
                kc_m = h1_prev.get('KC_mid', None)
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

            if (not reason and self._time_decay_tp and pos.strategy == 'keltner'
                    and pos.bars_held >= self._td_start_bars):
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


BacktestEngine._check_exits = make_kc_mid_h1_boundary_only(min_bars=2)
s_fix1c = run_variant(data, "C_H1boundary", verbose=False, **L3_KW, spread_cost=SPREAD)
BacktestEngine._check_exits = _ORIGINAL_CHECK_EXITS
print_row("C: H1 boundary only (closed)", s_fix1c)

delta_ab = s_fix1b['sharpe'] - s_l4_orig['sharpe']
delta_ac = s_fix1c['sharpe'] - s_l4_orig['sharpe']
print(f"\n  VERDICT:")
print(f"    Original vs Closed H1:      delta Sharpe = {delta_ab:+.2f}")
print(f"    Original vs H1 boundary:    delta Sharpe = {delta_ac:+.2f}")
if abs(delta_ab) > 0.5:
    print(f"    >>> LOOK-AHEAD CONFIRMED: Sharpe drops {abs(delta_ab):.2f} when using closed H1 <<<")
    print(f"    >>> All L4 results are UNRELIABLE. Must use closed H1 version. <<<")
else:
    print(f"    Look-ahead impact is small ({delta_ab:+.2f}). KC mid is not overly sensitive to H1 bar status.")

# Exit breakdown for each
print_exit_breakdown(s_l4_orig, "L4 Original")
print_exit_breakdown(s_fix1b, "FIX1-B Closed H1")
print_exit_breakdown(s_fix1c, "FIX1-C H1 Boundary")


# ═══════════════════════════════════════════════════════════════
# FIX2: Exit Priority
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FIX2: EXIT PRIORITY — KC mid before vs after Trailing")
print("=" * 80)
print(f"\nDoes KC mid steal exits from Trailing? (Trailing usually exits at higher price)")
print(header)
print("-" * 95)

# Use the BEST h1 source from FIX1 (will use closed if look-ahead confirmed)
best_h1 = abs(delta_ab) > 0.5
h1_label = "closed" if best_h1 else "current"

# A: KC mid before trailing (current behavior, with best h1)
s_fix2a = run_and_report(data, f"A: KC before trail ({h1_label})", True,
                         {"min_bars": 2, "use_closed_h1": best_h1, "after_trailing": False})
print_row(f"A: KC before trail ({h1_label})", s_fix2a)

# B: KC mid after trailing
s_fix2b = run_and_report(data, f"B: KC after trail ({h1_label})", True,
                         {"min_bars": 2, "use_closed_h1": best_h1, "after_trailing": True})
print_row(f"B: KC after trail ({h1_label})", s_fix2b)

# C: KC mid only when trailing NOT activated
s_fix2c = run_and_report(data, f"C: KC no-trail only ({h1_label})", True,
                         {"min_bars": 2, "use_closed_h1": best_h1, "after_trailing": True})
print_row(f"C: KC after trail ({h1_label})", s_fix2c)

print_exit_breakdown(s_fix2a, "KC before trail")
print_exit_breakdown(s_fix2b, "KC after trail")

delta_priority = s_fix2b['sharpe'] - s_fix2a['sharpe']
print(f"\n  VERDICT:")
print(f"    Before vs After trailing: delta Sharpe = {delta_priority:+.2f}")
if delta_priority > 0.3:
    print(f"    >>> KC mid SHOULD be after trailing — gains {delta_priority:.2f} Sharpe <<<")
elif delta_priority < -0.3:
    print(f"    >>> KC mid correctly placed before trailing <<<")
else:
    print(f"    Exit order has minimal impact ({delta_priority:+.2f})")


# ═══════════════════════════════════════════════════════════════
# FIX3: min_bars Sensitivity
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FIX3: min_bars SENSITIVITY — How early is too early?")
print("=" * 80)
print(f"\nUsing {h1_label} H1, before trailing")
print(header)
print("-" * 95)

for mb in [1, 2, 4, 6, 8, 12]:
    s = run_and_report(data, f"min_bars={mb}", True,
                       {"min_bars": mb, "use_closed_h1": best_h1, "after_trailing": False})
    print_row(f"min_bars={mb:>2d}", s)

# Also print L3 for reference
print_row("L3 (no KC mid)", s_l3)


# ═══════════════════════════════════════════════════════════════
# FIX4: Profit Filter
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FIX4: PROFIT FILTER — Is KC mid helping losers or winners?")
print("=" * 80)
print(header)
print("-" * 95)

# A: All trades (baseline)
s_fix4a = run_and_report(data, "A: All trades", True,
                         {"min_bars": 2, "use_closed_h1": best_h1, "profit_filter": None})
print_row("A: All trades (default)", s_fix4a)

# B: Loss only
s_fix4b = run_and_report(data, "B: Loss only", True,
                         {"min_bars": 2, "use_closed_h1": best_h1, "profit_filter": "loss_only"})
print_row("B: KC mid loss only", s_fix4b)

# C: Profit only
s_fix4c = run_and_report(data, "C: Profit only", True,
                         {"min_bars": 2, "use_closed_h1": best_h1, "profit_filter": "profit_only"})
print_row("C: KC mid profit only", s_fix4c)

# L3 reference
print_row("L3 (no KC mid)", s_l3)

print(f"\n  VERDICT:")
print(f"    All:    Sharpe={s_fix4a['sharpe']:.2f}")
print(f"    Loss:   Sharpe={s_fix4b['sharpe']:.2f}  (delta vs L3: {s_fix4b['sharpe'] - s_l3['sharpe']:+.2f})")
print(f"    Profit: Sharpe={s_fix4c['sharpe']:.2f}  (delta vs L3: {s_fix4c['sharpe'] - s_l3['sharpe']:+.2f})")
print(f"    L3:     Sharpe={s_l3['sharpe']:.2f}")

print_exit_breakdown(s_fix4b, "Loss-only KC mid")
print_exit_breakdown(s_fix4c, "Profit-only KC mid")


# ═══════════════════════════════════════════════════════════════
# FIX5: Best Combined — K-Fold + Stress Test
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FIX5: BEST COMBINED FIX — K-Fold Validation")
print("=" * 80)

# Determine best config from FIX1-4 results
# Always use closed H1 (safe from look-ahead)
# Determine best priority from FIX2
best_after = s_fix2b['sharpe'] > s_fix2a['sharpe']
# Determine best profit filter from FIX4
pf_options = [
    (None, s_fix4a['sharpe'], "all"),
    ("loss_only", s_fix4b['sharpe'], "loss_only"),
    ("profit_only", s_fix4c['sharpe'], "profit_only"),
]
best_pf = max(pf_options, key=lambda x: x[1])

best_config = {
    "min_bars": 2,
    "use_closed_h1": True,
    "after_trailing": best_after,
    "profit_filter": best_pf[0],
}
print(f"\nBest config from FIX1-4:")
print(f"  use_closed_h1 = True (always safe)")
print(f"  after_trailing = {best_after}")
print(f"  profit_filter = {best_pf[2]} (Sharpe={best_pf[1]:.2f})")
print(f"  min_bars = 2")

# Full sample
print(f"\n--- Full Sample @ $0.30 ---")
print(header)
print("-" * 95)

s_best = run_and_report(data, "BEST FIX combo", True, best_config)
print_row("BEST FIX combined", s_best)
print_row("L3 (no KC mid)", s_l3)
print_row("L4 ORIGINAL", s_l4_orig)

print_exit_breakdown(s_best, "BEST FIX")

# K-Fold
print(f"\n--- K-Fold 6-fold: BEST FIX vs L3 ---")
print(f"{'Fold':<8s}  {'L3':>8s}  {'BEST_FIX':>9s}  {'ORIGINAL':>9s}  {'Fix_vs_L3':>10s}  {'Orig_vs_L3':>10s}")
print("-" * 65)

fix_wins = 0
orig_wins = 0
for fold_name, start, end in FOLDS:
    fold_data = data.slice(start, end)
    if len(fold_data.m15_df) < 1000:
        continue

    BacktestEngine._check_exits = _ORIGINAL_CHECK_EXITS
    s_b = run_variant(fold_data, f"B_{fold_name}", verbose=False, **L3_KW, spread_cost=SPREAD)

    BacktestEngine._check_exits = make_kc_mid_exit(**best_config)
    s_f = run_variant(fold_data, f"F_{fold_name}", verbose=False, **L3_KW, spread_cost=SPREAD)

    BacktestEngine._check_exits = make_kc_mid_exit(min_bars=2, use_closed_h1=False, after_trailing=False)
    s_o = run_variant(fold_data, f"O_{fold_name}", verbose=False, **L3_KW, spread_cost=SPREAD)

    BacktestEngine._check_exits = _ORIGINAL_CHECK_EXITS

    d_fix = s_f['sharpe'] - s_b['sharpe']
    d_orig = s_o['sharpe'] - s_b['sharpe']
    fw = d_fix > 0
    ow = d_orig > 0
    if fw: fix_wins += 1
    if ow: orig_wins += 1
    print(f"  {fold_name:<6s}  {s_b['sharpe']:>8.2f}  {s_f['sharpe']:>9.2f}  "
          f"{s_o['sharpe']:>9.2f}  {d_fix:>+10.2f} {'V' if fw else 'X'}  "
          f"{d_orig:>+10.2f} {'V' if ow else 'X'}")

print(f"\n  BEST FIX K-Fold: {fix_wins}/6 {'PASS' if fix_wins >= 5 else 'FAIL'}")
print(f"  ORIGINAL K-Fold: {orig_wins}/6 {'PASS' if orig_wins >= 5 else 'FAIL'}")

# Stress test
print(f"\n--- Stress Test: BEST FIX at various spreads ---")
print(f"{'Spread':<10s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'WR%':>5s}  {'MaxDD':>11s}")
print("-" * 60)

for sp in [0.00, 0.20, 0.30, 0.40, 0.50, 0.60]:
    s = run_and_report(data, f"sp${sp:.2f}", True, best_config, spread=sp)
    print(f"  ${sp:.2f}      {s['n']:>5d}  {s['sharpe']:>6.2f}  {fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  {fmt(s['max_dd'])}")


# ═══════════════════════════════════════════════════════════════
# ALSO: Test if look-ahead matters for Trailing exit (sanity check)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("BONUS: Does the original L4 without KC mid also have issues?")
print("(Trailing uses ATR from h1_window — is ATR from unclosed bar a problem?)")
print("=" * 80)
print(header)
print("-" * 95)
print_row("L3 (baseline)", s_l3)
print_row("L4 original", s_l4_orig)
print_row("L4 closed H1", s_fix1b)

# Year-by-year for BEST FIX
print("\n--- Year-by-Year: BEST FIX @ $0.30 ---")
trades_best = s_best.get('_trades', [])
if trades_best:
    years = sorted(set(t.exit_time.year for t in trades_best if t.exit_time))
    print(f"  {'Year':>4s}  {'N':>5s}  {'PnL':>10s}  {'WR%':>5s}  {'$/t':>7s}")
    print("  " + "-" * 40)
    for y in years:
        yt = [t for t in trades_best if t.exit_time and t.exit_time.year == y]
        n = len(yt)
        pnl = sum(t.pnl for t in yt)
        wr = sum(1 for t in yt if t.pnl > 0) / n * 100 if n > 0 else 0
        avg = pnl / n if n > 0 else 0
        print(f"  {y:>4d}  {n:>5d}  ${pnl:>9,.0f}  {wr:>5.1f}%  ${avg:>6.2f}")


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"\n  L3 (no KC mid):     Sharpe={s_l3['sharpe']:.2f}  PnL={fmt(s_l3['total_pnl'])}  MaxDD={fmt(s_l3['max_dd'])}")
print(f"  L4 ORIGINAL:        Sharpe={s_l4_orig['sharpe']:.2f}  PnL={fmt(s_l4_orig['total_pnl'])}  MaxDD={fmt(s_l4_orig['max_dd'])}")
print(f"  L4 BEST FIX:        Sharpe={s_best['sharpe']:.2f}  PnL={fmt(s_best['total_pnl'])}  MaxDD={fmt(s_best['max_dd'])}")
print(f"  BEST FIX K-Fold:    {fix_wins}/6 {'PASS' if fix_wins >= 5 else 'FAIL'}")
print(f"  ORIGINAL K-Fold:    {orig_wins}/6 {'PASS' if orig_wins >= 5 else 'FAIL'}")

print(f"\n  Look-ahead impact:  {delta_ab:+.2f} Sharpe")
print(f"  Priority impact:    {delta_priority:+.2f} Sharpe")

if abs(delta_ab) > 0.5:
    print(f"\n  >>> CRITICAL: Look-ahead bias confirmed. L4 original results UNRELIABLE.")
    print(f"  >>> Must use closed H1 version. BEST FIX Sharpe={s_best['sharpe']:.2f}")
    if s_best['sharpe'] > s_l3['sharpe'] + 0.3 and fix_wins >= 5:
        print(f"  >>> BEST FIX still beats L3 after correction — KC mid is valid with fix")
    elif s_best['sharpe'] <= s_l3['sharpe']:
        print(f"  >>> BEST FIX does NOT beat L3 — KC mid exit should be DROPPED")
    else:
        print(f"  >>> BEST FIX marginal improvement over L3 — KC mid is borderline")
else:
    print(f"\n  Look-ahead impact is small. L4 results are reasonably reliable.")
    if s_best['sharpe'] > s_l3['sharpe'] + 0.3 and fix_wins >= 5:
        print(f"  >>> Recommend deploying BEST FIX version (safe from look-ahead)")
    else:
        print(f"  >>> Consider whether KC mid is worth the complexity")

elapsed = time.time() - t0
print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
print(f"Completed: {datetime.now()}")
sys.stdout = tee.stdout
tee.close()
print(f"Results saved to {OUTPUT_FILE}")
