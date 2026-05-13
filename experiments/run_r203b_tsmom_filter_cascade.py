"""TSMOM Filter Cascade Diagnostic
====================================
Building on r203, this script simulates the FULL EA filter cascade
to find which filter blocks each signal.

EA filters (in order):
  1. Rule B: skip 8 H1 bars after 3-sigma ATR spike (60-bar lookback)
  2. MinEntryGapHours: 2.0 h between entries
  3. ATR floor: atr < 0.1 -> reject
  4. Crossover: score>0 && prev<=0 (BUY) or score<0 && prev>=0 (SELL)
  5. (No existing position)

For each crossover signal in last 6 months, report which filter (if any)
would have blocked entry in live trading.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from backtest.runner import load_csv, H1_CSV_PATH


FAST_LB = 480
SLOW_LB = 720
RULEB_SIGMA = 3.0
RULEB_LB = 60
RULEB_SKIP_BARS = 8
MIN_ENTRY_GAP_HOURS = 2.0
ATR_FLOOR = 0.1
ATR_PERIOD = 14
MAX_HOLD_BARS = 20
SL_ATR_MULT = 4.5
TP_ATR_MULT = 6.0


def compute_atr(h1: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Wilder's ATR (matches MT4 iATR)."""
    high = h1['High'].values
    low = h1['Low'].values
    close = h1['Close'].values
    tr = np.zeros(len(h1))
    for i in range(1, len(h1)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr = np.zeros(len(h1))
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, len(h1)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def compute_score(close: np.ndarray) -> np.ndarray:
    n = len(close)
    score = np.zeros(n)
    for i in range(n):
        s = 0.0
        if i >= FAST_LB and close[i - FAST_LB] > 0:
            ret = close[i] / close[i - FAST_LB] - 1.0
            s += 0.5 * (1.0 if ret > 0 else (-1.0 if ret < 0 else 0.0))
        if i >= SLOW_LB and close[i - SLOW_LB] > 0:
            ret = close[i] / close[i - SLOW_LB] - 1.0
            s += 0.5 * (1.0 if ret > 0 else (-1.0 if ret < 0 else 0.0))
        score[i] = s
    return score


def simulate_ea(h1: pd.DataFrame, score: np.ndarray, atr: np.ndarray,
                start_idx: int) -> pd.DataFrame:
    """Replay EA logic, recording filter outcome at every crossover.

    Returns: DataFrame of all crossover events with filter status.
    """
    n = len(h1)
    rows = []

    rule_b_skip = 0
    last_entry_time = pd.Timestamp('1970-01-01', tz='UTC')
    in_position = False
    pos_entry_idx = -1
    pos_entry_atr = 0.0
    pos_extreme = 0.0
    pos_dir = 0
    pos_bars = 0
    pos_entry_price = 0.0

    close = h1['Close'].values
    high = h1['High'].values
    low = h1['Low'].values
    ts = h1.index

    for i in range(max(start_idx, SLOW_LB + 2), n):
        cur_atr = atr[i]
        if cur_atr <= 0:
            continue

        # Manage open position (matches EA ManageOpenTrade)
        if in_position:
            pos_bars += 1
            c = close[i]
            tp_dist = TP_ATR_MULT * pos_entry_atr
            sl_dist = SL_ATR_MULT * pos_entry_atr
            pnl = (c - pos_entry_price) if pos_dir == 1 else (pos_entry_price - c)

            if pnl >= tp_dist or pnl <= -sl_dist or pos_bars >= MAX_HOLD_BARS:
                in_position = False
            else:
                if pos_dir == 1 and c > pos_extreme:
                    pos_extreme = c
                if pos_dir == -1 and c < pos_extreme:
                    pos_extreme = c
                cur_score = score[i]
                if pos_dir == 1 and cur_score < 0:
                    in_position = False
                if pos_dir == -1 and cur_score > 0:
                    in_position = False

        # Rule B check (on new H1 bar)
        if i >= RULEB_LB:
            window = atr[i - RULEB_LB:i]
            window = window[window > 0]
            if len(window) >= 10:
                mean = window.mean()
                std = max(window.std(), 1e-6)
                if cur_atr > mean + RULEB_SIGMA * std:
                    rule_b_skip = RULEB_SKIP_BARS
                elif rule_b_skip > 0:
                    rule_b_skip -= 1

        # Detect crossover (only record at signal events)
        if i < 1:
            continue
        s_now = score[i]
        s_prev = score[i - 1]
        sig = None
        if s_now > 0 and s_prev <= 0:
            sig = 'BUY'
        elif s_now < 0 and s_prev >= 0:
            sig = 'SELL'
        if sig is None:
            continue

        # Apply filters in order and record outcome
        blocker = None
        if in_position:
            blocker = 'OPEN_POSITION'
        elif rule_b_skip > 0:
            blocker = f'RULE_B (skip {rule_b_skip} more bars)'
        elif (ts[i] - last_entry_time).total_seconds() < MIN_ENTRY_GAP_HOURS * 3600:
            gap_h = (ts[i] - last_entry_time).total_seconds() / 3600
            blocker = f'MIN_GAP ({gap_h:.1f}h < {MIN_ENTRY_GAP_HOURS}h)'
        elif cur_atr < ATR_FLOOR:
            blocker = f'ATR_FLOOR ({cur_atr:.4f} < {ATR_FLOOR})'

        rows.append({
            'time': ts[i],
            'side': sig,
            'score': s_now,
            'prev_score': s_prev,
            'atr': cur_atr,
            'rule_b_skip': rule_b_skip,
            'in_position': in_position,
            'pos_bars': pos_bars if in_position else 0,
            'pos_dir': pos_dir if in_position else 0,
            'blocker': blocker,
            'would_enter': blocker is None,
        })

        if blocker is None:
            in_position = True
            pos_dir = 1 if sig == 'BUY' else -1
            pos_entry_atr = cur_atr
            pos_extreme = close[i]
            pos_entry_price = close[i]
            pos_bars = 0
            last_entry_time = ts[i]

    return pd.DataFrame(rows)


def main():
    print('=' * 70)
    print('TSMOM EA Filter Cascade Diagnostic')
    print('=' * 70)
    print(f'Loading H1: {H1_CSV_PATH}')
    h1 = load_csv(str(H1_CSV_PATH))
    print(f'  Bars: {len(h1):,}  {h1.index[0]} -> {h1.index[-1]}')

    print('Computing ATR (Wilder, period=14)...')
    atr = compute_atr(h1, ATR_PERIOD)
    print(f'  ATR range: [{atr[atr>0].min():.2f}, {atr[atr>0].max():.2f}]  mean={atr[atr>0].mean():.2f}')

    print('Computing TSMOM score...')
    score = compute_score(h1['Close'].values)

    print('Simulating EA filter cascade (from 2025-01-01)...')
    start_ts = pd.Timestamp('2025-01-01', tz='UTC')
    start_idx = h1.index.searchsorted(start_ts)
    sigs = simulate_ea(h1, score, atr, start_idx)
    print(f'  Total crossovers: {len(sigs)}')

    if sigs.empty:
        print('No signals — exit.')
        return

    print('\n--- Filter Outcome Breakdown ---')
    sigs['outcome'] = sigs.apply(
        lambda r: 'ENTERED' if r['would_enter'] else r['blocker'].split(' ')[0], axis=1
    )
    breakdown = sigs['outcome'].value_counts()
    print(breakdown.to_string())
    print(f'\n  Pass rate: {breakdown.get("ENTERED", 0)}/{len(sigs)} '
          f'({100*breakdown.get("ENTERED",0)/len(sigs):.1f}%)')

    print('\n--- Per-Month Outcome ---')
    sigs['ym'] = sigs['time'].dt.strftime('%Y-%m')
    pivot = sigs.pivot_table(index='ym', columns='outcome', aggfunc='size', fill_value=0)
    pivot['TOTAL'] = pivot.sum(axis=1)
    print(pivot.to_string())

    print('\n--- Last 60 Days (Live Window) Detail ---')
    last_60d = h1.index[-1] - pd.Timedelta(days=60)
    recent = sigs[sigs['time'] >= last_60d].copy()
    if recent.empty:
        print('  (no signals in last 60d)')
    else:
        recent_display = recent[['time', 'side', 'atr', 'in_position', 'blocker', 'would_enter']]
        print(recent_display.to_string(index=False))

    print('\n--- Recent BUY signals blocked by RULE_B ---')
    rb_blocked = sigs[sigs['blocker'].fillna('').str.startswith('RULE_B')].tail(10)
    if not rb_blocked.empty:
        print(rb_blocked[['time', 'side', 'atr', 'rule_b_skip']].to_string(index=False))

    print('\n' + '=' * 70)
    print('Summary')
    print('=' * 70)
    n_entered = breakdown.get('ENTERED', 0)
    n_rule_b = breakdown.get('RULE_B', 0)
    n_open = breakdown.get('OPEN_POSITION', 0)
    n_gap = breakdown.get('MIN_GAP', 0)
    n_atr = breakdown.get('ATR_FLOOR', 0)
    print(f'  Total signals (since 2025-01-01): {len(sigs)}')
    print(f'  Entries simulated:                 {n_entered}')
    print(f'  Blocked by RULE_B (ATR 3sigma):    {n_rule_b}')
    print(f'  Blocked by OPEN_POSITION:          {n_open}')
    print(f'  Blocked by MIN_GAP (<2h):          {n_gap}')
    print(f'  Blocked by ATR_FLOOR:              {n_atr}')

    out = Path('results/r203_tsmom_signal_diag')
    out.mkdir(parents=True, exist_ok=True)
    sigs.to_csv(out / 'filter_cascade.csv', index=False)
    print(f'\n  Full cascade saved to: {out / "filter_cascade.csv"}')


if __name__ == '__main__':
    main()
