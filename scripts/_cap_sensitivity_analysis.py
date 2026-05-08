#!/usr/bin/env python3
"""
Cap Sensitivity Analysis: find stable cap_atr_multiple range for dynamic Cap.
Tests cap = lots × PV × cap_atr_multiple × ATR across a grid.
"""
import sys, os, io, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

PV = 100

STRATEGIES = {
    'PSAR':        {'lots': 0.09, 'sl_atr': 4.0, 'current_cap': 60},
    'TSMOM':       {'lots': 0.15, 'sl_atr': 6.0, 'current_cap': 60},
    'SESS_BO':     {'lots': 0.13, 'sl_atr': 4.5, 'current_cap': 60},
    'Chandelier':  {'lots': 0.08, 'sl_atr': 4.0, 'current_cap': 25},
    'Dual_Thrust': {'lots': 0.04, 'sl_atr': 3.5, 'current_cap': 18},
    'Keltner':     {'lots': 0.02, 'sl_atr': 3.5, 'current_cap': 35},
}

CAP_ATR_MULTIPLES = [3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 10.0]


def main():
    from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS

    print(f"\n{'='*70}")
    print(f"  Cap Sensitivity Analysis")
    print(f"  Formula: cap = lots × PV × cap_atr_multiple × ATR")
    print(f"{'='*70}\n")

    data = DataBundle.load_default()
    h1 = data.h1_df
    recent_atr = float(h1['ATR'].dropna().iloc[-60:].mean())
    print(f"  Recent avg ATR (60 bars): {recent_atr:.2f}")

    print(f"\n  Dynamic Cap values at current ATR={recent_atr:.2f}:")
    print(f"  {'Strategy':>14} {'Lots':>5} {'SL_ATR':>7} {'Curr$':>7}  ", end='')
    for mult in CAP_ATR_MULTIPLES:
        print(f"{mult:>6.1f}x", end='')
    print()
    print(f"  {'-'*14} {'-'*5} {'-'*7} {'-'*7}  " + '-' * (7 * len(CAP_ATR_MULTIPLES)))

    for name, cfg in STRATEGIES.items():
        lots = cfg['lots']
        sl_atr = cfg['sl_atr']
        current_cap = cfg['current_cap']
        print(f"  {name:>14} {lots:>5.2f} {sl_atr:>7.1f} ${current_cap:>5}  ", end='')
        for mult in CAP_ATR_MULTIPLES:
            cap_value = lots * PV * mult * recent_atr
            marker = '*' if mult * recent_atr < sl_atr * recent_atr else ' '
            print(f"${cap_value:>5.0f}{marker}", end='')
        print()

    print(f"\n  * = cap_atr_multiple < SL multiple (cap tighter than SL, will over-trigger)")

    print(f"\n  Price tolerance ($/oz) at current ATR:")
    print(f"  {'Strategy':>14} {'Curr':>6}  ", end='')
    for mult in CAP_ATR_MULTIPLES:
        print(f"{mult:>6.1f}x", end='')
    print()
    print(f"  {'-'*14} {'-'*6}  " + '-' * (7 * len(CAP_ATR_MULTIPLES)))

    for name, cfg in STRATEGIES.items():
        lots = cfg['lots']
        current_cap = cfg['current_cap']
        current_tol = current_cap / (lots * PV) if lots * PV > 0 else 0
        print(f"  {name:>14} ${current_tol:>4.1f}  ", end='')
        for mult in CAP_ATR_MULTIPLES:
            tol = mult * recent_atr
            print(f"${tol:>5.1f}", end=' ')
        print()

    # Cap trigger rate simulation
    print(f"\n\n  Cap Trigger Rate Simulation (backtest L8_MAX as proxy):")
    print(f"  Testing how often floating loss exceeds cap for different multiples\n")

    kw = {**LIVE_PARITY_KWARGS, 'min_lot_size': 0.02, 'max_lot_size': 0.02}

    for mult in [5.0, 6.0, 7.0, 8.0]:
        cap = 0.02 * PV * mult * recent_atr
        kw_cap = {**kw, 'maxloss_cap': cap}
        result = run_variant(data, f"Cap_{mult}x", verbose=False, **kw_cap)
        trades = result.get('n_trades', 0)
        sharpe = result.get('sharpe', 0)
        pnl = result.get('total_pnl', 0)
        cap_exits = result.get('maxloss_cap_exits', 0)
        cap_rate = cap_exits / trades * 100 if trades > 0 else 0
        print(f"    {mult:.1f}x ATR (cap=${cap:.0f}): Sharpe={sharpe:.3f}, "
              f"PnL=${pnl:.0f}, Cap exits={cap_exits}/{trades} ({cap_rate:.1f}%)")

    print(f"\n  Recommended range: 6.0-7.0x ATR")
    print(f"  - Must be > SL multiple (max 6.0x for TSMOM)")
    print(f"  - Cap trigger rate should be 5-15%")
    print(f"  - Current fixed-dollar caps approximate 6-7x ATR at current levels")


if __name__ == '__main__':
    main()
