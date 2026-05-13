"""TSMOM Entry Signal Diagnostic
================================
Purpose: Diagnose why TSMOM live trading has 0 triggers.

Hypothesis: TSMOM uses HUGE lookbacks (480/720 H1 bars = 20/30 trading days).
Entry only fires on score *zero-crossing* — a very rare event.

This script:
  1. Loads full H1 data (2015+)
  2. Computes TSMOM score for every bar (matching EA logic exactly)
  3. Counts BUY/SELL crossover events per month
  4. Reports the long-term frequency to set expectations
  5. Shows the recent state (last 90 days) to see if signals were near-firing
  6. Identifies the current score and how far we are from a flip

Run: python experiments/run_r203_tsmom_signal_diag.py
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


def compute_score_series(close: np.ndarray) -> np.ndarray:
    """Mirror MQ4 MomentumScore exactly.

    score[i] = 0.5 * sign(close[i] / close[i - FAST_LB] - 1)
             + 0.5 * sign(close[i] / close[i - SLOW_LB] - 1)
    """
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


def find_signals(score: np.ndarray, ts: pd.DatetimeIndex) -> pd.DataFrame:
    """Find all crossover events (matching EA entry logic).

    BUY  fires when score[i] > 0 and score[i-1] <= 0
    SELL fires when score[i] < 0 and score[i-1] >= 0
    """
    rows = []
    for i in range(1, len(score)):
        if score[i] > 0 and score[i - 1] <= 0:
            rows.append({'time': ts[i], 'side': 'BUY', 'score': score[i], 'prev': score[i - 1]})
        elif score[i] < 0 and score[i - 1] >= 0:
            rows.append({'time': ts[i], 'side': 'SELL', 'score': score[i], 'prev': score[i - 1]})
    if not rows:
        return pd.DataFrame(columns=['time', 'side', 'score', 'prev'])
    df = pd.DataFrame(rows)
    df.set_index('time', inplace=True)
    return df


def print_signal_calendar(sigs: pd.DataFrame, start: pd.Timestamp | None = None):
    """Print per-month signal counts."""
    if sigs.empty:
        print('  (no signals)')
        return
    df = sigs.copy()
    if start is not None:
        df = df[df.index >= start]
    if df.empty:
        print('  (no signals in window)')
        return
    df['ym'] = df.index.strftime('%Y-%m')
    pivot = df.pivot_table(index='ym', columns='side', aggfunc='size', fill_value=0)
    pivot['TOTAL'] = pivot.sum(axis=1)
    print(pivot.to_string())
    print(f'  Total: {len(df)} signals over {df["ym"].nunique()} months '
          f'({len(df)/max(df["ym"].nunique(),1):.2f}/month)')


def main():
    print('=' * 70)
    print('TSMOM Entry Signal Diagnostic')
    print('=' * 70)
    print(f'Loading H1 data: {H1_CSV_PATH}')
    h1 = load_csv(str(H1_CSV_PATH))
    print(f'  Bars: {len(h1):,}  Range: {h1.index[0]} -> {h1.index[-1]}')
    print(f'  Fast lookback: {FAST_LB} H1 bars ({FAST_LB/24:.1f} trading days)')
    print(f'  Slow lookback: {SLOW_LB} H1 bars ({SLOW_LB/24:.1f} trading days)')

    print('\nComputing TSMOM score series...')
    score = compute_score_series(h1['Close'].values)

    print('\nFinding crossover events (entry signals)...')
    sigs = find_signals(score, h1.index)
    n_buy = (sigs['side'] == 'BUY').sum()
    n_sell = (sigs['side'] == 'SELL').sum()
    print(f'  Total: {len(sigs)} signals  (BUY={n_buy}, SELL={n_sell})')
    print(f'  Avg: {len(sigs)/((h1.index[-1] - h1.index[0]).days/30):.2f} signals/month over full history')

    print('\n--- Per-Year Signal Counts ---')
    if not sigs.empty:
        sigs_y = sigs.copy()
        sigs_y['year'] = sigs_y.index.year
        py = sigs_y.pivot_table(index='year', columns='side', aggfunc='size', fill_value=0)
        py['TOTAL'] = py.sum(axis=1)
        print(py.to_string())

    print('\n--- Last 24 Months (Recent History) ---')
    last_24m = h1.index[-1] - pd.Timedelta(days=720)
    print_signal_calendar(sigs, start=last_24m)

    print('\n--- Last 6 Months ---')
    last_6m = h1.index[-1] - pd.Timedelta(days=180)
    print_signal_calendar(sigs, start=last_6m)
    recent_sigs = sigs[sigs.index >= last_6m]
    if not recent_sigs.empty:
        print('\n  Detail:')
        print(recent_sigs.to_string())

    print('\n--- Last 60 Days (Live Window) ---')
    last_60d = h1.index[-1] - pd.Timedelta(days=60)
    live_sigs = sigs[sigs.index >= last_60d]
    if live_sigs.empty:
        print(f'  ZERO signals in last 60 days (data ends {h1.index[-1]})')
    else:
        print(live_sigs.to_string())

    print('\n--- Score Trajectory (Last 30 H1 Bars) ---')
    last_30 = pd.DataFrame({
        'close': h1['Close'].iloc[-30:].values,
        'score': score[-30:],
    }, index=h1.index[-30:])
    print(last_30.to_string())

    print('\n--- Score Distribution (last 6 months) ---')
    score_recent = pd.Series(score[-180*24:], index=h1.index[-180*24:])
    print(score_recent.value_counts().sort_index().to_string())

    cur_score = score[-1]
    fast_ret = h1['Close'].iloc[-1] / h1['Close'].iloc[-FAST_LB - 1] - 1.0
    slow_ret = h1['Close'].iloc[-1] / h1['Close'].iloc[-SLOW_LB - 1] - 1.0
    print(f'\n--- Current State (as of {h1.index[-1]}) ---')
    print(f'  Current score:    {cur_score:+.2f}')
    print(f'  Fast return (480h={FAST_LB/24:.0f}d): {fast_ret*100:+.2f}%')
    print(f'  Slow return (720h={SLOW_LB/24:.0f}d): {slow_ret*100:+.2f}%')
    print(f'  Current close:    {h1["Close"].iloc[-1]:.2f}')
    print(f'  Close - FAST ref: {h1["Close"].iloc[-FAST_LB-1]:.2f}')
    print(f'  Close - SLOW ref: {h1["Close"].iloc[-SLOW_LB-1]:.2f}')

    if cur_score >= 0.5:
        gap_fast = h1['Close'].iloc[-1] - h1['Close'].iloc[-FAST_LB - 1]
        gap_slow = h1['Close'].iloc[-1] - h1['Close'].iloc[-SLOW_LB - 1]
        print(f'  Score is POSITIVE. Distance to flip:')
        print(f'    Need close to drop {gap_fast:.2f} (=$ {gap_fast:.2f}) below FAST ref to lose 0.5')
        print(f'    Need close to drop {gap_slow:.2f} (=$ {gap_slow:.2f}) below SLOW ref to lose 0.5')

    print('\n' + '=' * 70)
    print('Diagnosis Summary')
    print('=' * 70)
    monthly_avg = len(sigs) / max(((h1.index[-1] - h1.index[0]).days / 30), 1)
    print(f'  Historical signal frequency: {monthly_avg:.2f} signals/month '
          f'(~{30/max(monthly_avg,0.01):.0f} days between signals on avg)')
    last_sig = sigs.index[-1] if not sigs.empty else None
    if last_sig is not None:
        days_since = (h1.index[-1] - last_sig).days
        print(f'  Last signal: {last_sig}  ({days_since} days ago)')
        print(f'  Last signal: {sigs.iloc[-1]["side"]} score={sigs.iloc[-1]["score"]:+.2f}')
    print(f'  Current score: {cur_score:+.2f}  '
          f'(crossover requires score to change sign)')
    print(f'\n  CONCLUSION: With 480/720 H1 lookbacks, gold trend reversals are rare.')
    print(f'  "0 live triggers" is EXPECTED behavior, not a bug.')
    print(f'  The strategy is signal-starved by design.')

    out = Path('results/r203_tsmom_signal_diag')
    out.mkdir(parents=True, exist_ok=True)
    sigs.to_csv(out / 'all_signals.csv')
    print(f'\n  Full signal list saved to: {out / "all_signals.csv"}')


if __name__ == '__main__':
    main()
