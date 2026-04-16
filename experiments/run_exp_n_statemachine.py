#!/usr/bin/env python3
"""
EXP-N: Keltner StateMachine vs Simple Breakout
================================================
signals.py has a full KeltnerStateMachine (4-phase: pullback->armed->window->entry)
but scan_all_signals uses simple check_keltner_signal instead.
Comment says "simple version backtests better" — but that was on the OLD engine!
Re-test on fixed engine + LIVE_PARITY.
Also test: hybrid (SM in high ADX only, simple in low ADX).
"""
import sys, os, time, gc
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine
import indicators as signals_mod

OUTPUT_FILE = "exp_n_statemachine_output.txt"
BASE = {**LIVE_PARITY_KWARGS, "keltner_max_hold_m15": 20}

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-01"),
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

print("=" * 80)
print("EXP-N: KELTNER STATE MACHINE vs SIMPLE BREAKOUT")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

# ── Part 1: Simple Breakout (current, baseline) ──
print("\n--- Part 1: Simple Breakout (current production) ---")
s_simple = run_variant(data, "N_simple", verbose=True, **BASE, spread_cost=0.30)
trades_simple = s_simple.get('_trades', [])
kc_simple = [t for t in trades_simple if t.strategy == 'keltner']
print(f"  Simple: N={s_simple['n']}, Sharpe={s_simple['sharpe']:.2f}, KC_trades={len(kc_simple)}")

# ── Part 2: StateMachine Breakout ──
print("\n--- Part 2: StateMachine Breakout ---")
print("  Replace check_keltner_signal with KeltnerStateMachine.update()")

# Save original
_orig_check = signals_mod.check_keltner_signal

# Create a global state machine
_sm = signals_mod.KeltnerStateMachine()
_sm_call_count = 0

def _sm_keltner_signal(df):
    """Wrapper: use StateMachine instead of simple breakout."""
    global _sm_call_count
    _sm_call_count += 1
    result = _sm.update(df)
    return result

# Patch
signals_mod.check_keltner_signal = _sm_keltner_signal

try:
    s_sm = run_variant(data, "N_statemachine", verbose=True, **BASE, spread_cost=0.30)
    trades_sm = s_sm.get('_trades', [])
    kc_sm = [t for t in trades_sm if t.strategy == 'keltner']
    print(f"  StateMachine: N={s_sm['n']}, Sharpe={s_sm['sharpe']:.2f}, KC_trades={len(kc_sm)}")
    print(f"  SM calls: {_sm_call_count}")
except Exception as e:
    print(f"  StateMachine ERROR: {e}")
    s_sm = None
    kc_sm = []
finally:
    signals_mod.check_keltner_signal = _orig_check

# ── Part 3: Compare signal overlap ──
if s_sm:
    print("\n--- Part 3: Signal Overlap Analysis ---")
    # Compare entry times
    simple_times = set(pd.Timestamp(t.entry_time).floor('h') for t in kc_simple)
    sm_times = set(pd.Timestamp(t.entry_time).floor('h') for t in kc_sm)
    
    overlap = simple_times & sm_times
    only_simple = simple_times - sm_times
    only_sm = sm_times - simple_times
    
    print(f"  Simple only: {len(only_simple)} entries")
    print(f"  SM only: {len(only_sm)} entries")
    print(f"  Both: {len(overlap)} entries")
    print(f"  Overlap rate: {len(overlap)/max(len(simple_times),1)*100:.1f}%")
    
    # PnL comparison by overlap category
    if overlap:
        overlap_simple_pnl = [t.pnl for t in kc_simple if pd.Timestamp(t.entry_time).floor('h') in overlap]
        only_simple_pnl = [t.pnl for t in kc_simple if pd.Timestamp(t.entry_time).floor('h') in only_simple]
        only_sm_pnl = [t.pnl for t in kc_sm if pd.Timestamp(t.entry_time).floor('h') in only_sm]
        
        for label, pnls in [("Overlap trades (Simple side)", overlap_simple_pnl),
                            ("Simple-only trades", only_simple_pnl),
                            ("SM-only trades", only_sm_pnl)]:
            if pnls:
                n = len(pnls)
                total = sum(pnls)
                avg = total / n
                wr = sum(1 for p in pnls if p > 0) / n * 100
                print(f"    {label}: N={n}, PnL=${total:,.0f}, $/t=${avg:.2f}, WR={wr:.1f}%")

    # ── Part 4: Head-to-head at multiple spreads ──
    print("\n--- Part 4: Simple vs SM at Multiple Spreads ---")
    print(f"  {'Config':<12s}  {'N_simple':>8s}  {'Sh_simple':>9s}  {'N_SM':>6s}  {'Sh_SM':>6s}  {'Delta':>7s}")
    print("  " + "-" * 55)
    
    for sp in [0.00, 0.30, 0.50]:
        s1 = run_variant(data, f"N_sim_sp{sp}", verbose=False, **BASE, spread_cost=sp)
        
        _sm2 = signals_mod.KeltnerStateMachine()
        def _sm2_fn(df):
            return _sm2.update(df)
        signals_mod.check_keltner_signal = _sm2_fn
        s2 = run_variant(data, f"N_sm_sp{sp}", verbose=False, **BASE, spread_cost=sp)
        signals_mod.check_keltner_signal = _orig_check
        
        delta = s2['sharpe'] - s1['sharpe']
        print(f"  sp${sp:.2f}      {s1['n']:>8d}  {s1['sharpe']:>9.2f}  {s2['n']:>6d}  {s2['sharpe']:>6.2f}  {delta:>+7.2f}")

    # ── Part 5: K-Fold if SM wins ──
    if s_sm and s_sm['sharpe'] > s_simple['sharpe'] + 0.10:
        print(f"\n--- Part 5: K-Fold StateMachine vs Simple @ $0.30 ---")
        wins = 0
        for fold_name, start, end in FOLDS:
            fold_data = data.slice(start, end)
            if len(fold_data.m15_df) < 1000:
                continue
            sb = run_variant(fold_data, f"N_B_{fold_name}", verbose=False, **BASE, spread_cost=0.30)
            
            _sm3 = signals_mod.KeltnerStateMachine()
            def _sm3_fn(df):
                return _sm3.update(df)
            signals_mod.check_keltner_signal = _sm3_fn
            st = run_variant(fold_data, f"N_T_{fold_name}", verbose=False, **BASE, spread_cost=0.30)
            signals_mod.check_keltner_signal = _orig_check
            
            delta = st['sharpe'] - sb['sharpe']
            won = delta > 0
            if won: wins += 1
            print(f"    {fold_name}: Simple={sb['sharpe']:>6.2f}  SM={st['sharpe']:>6.2f}  delta={delta:>+.2f} {'V' if won else 'X'}")
        print(f"    Result: {wins}/6 {'PASS' if wins >= 5 else 'FAIL'}")
    else:
        print("\n  SM does not beat Simple by >0.10 Sharpe, skip K-Fold.")

elapsed = time.time() - t_total
print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
print(f"Completed: {datetime.now()}")
sys.stdout = tee.stdout
tee.close()
print(f"Results saved to {OUTPUT_FILE}")
