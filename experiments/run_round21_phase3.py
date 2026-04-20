"""
Round 21 Phase 3: K-Fold Validation & Portfolio Analysis
=========================================================
Validates the two promising strategies from Phase 2:
  - S1: Squeeze Straddle (best: SqzB3_T0.2/0.04_MH20, Sharpe=1.67)
  - S3: Overnight Hold (NYclose_to_LDNopen, Sharpe=0.88)
  - S3b: OffHours session (Sharpe=2.47 from return analysis)

Also runs L7 baseline for correlation analysis.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Dict

OUT_DIR = Path("results/round21_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.30

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-10"),
]


def run_s1_kfold(h1_df):
    """K-Fold validation of best S1 config: SqzB3, Trail 0.2/0.04, MH20."""
    print("\n" + "=" * 70)
    print("S1 K-Fold: SqzB3_T0.2/0.04_MH20")
    print("=" * 70)

    min_squeeze_bars = 3
    trail_act, trail_dist = 0.2, 0.04
    max_hold = 20

    full_trades = _run_s1_single(h1_df, min_squeeze_bars, trail_act, trail_dist, max_hold)

    fold_results = []
    for fold_name, start, end in FOLDS:
        ts = pd.Timestamp(start, tz='UTC')
        te = pd.Timestamp(end, tz='UTC')
        fold_h1 = h1_df[(h1_df.index >= ts) & (h1_df.index < te)]
        if len(fold_h1) < 500:
            continue

        trades = _run_s1_single(fold_h1, min_squeeze_bars, trail_act, trail_dist, max_hold)
        pnls = [t['pnl'] for t in trades]
        n = len(pnls)
        total = sum(pnls)
        wr = sum(1 for p in pnls if p > 0) / n if n > 0 else 0
        sharpe = np.mean(pnls) / np.std(pnls, ddof=1) * np.sqrt(n / 2) if n > 1 and np.std(pnls, ddof=1) > 0 else 0

        print(f"  {fold_name} ({start}~{end}): N={n}, PnL=${total:.0f}, "
              f"WR={wr:.1%}, Sharpe={sharpe:.2f}")
        fold_results.append({
            'fold': fold_name, 'n': n, 'pnl': total, 'wr': wr,
            'sharpe': sharpe, 'start': start, 'end': end
        })

    positive_folds = sum(1 for r in fold_results if r['pnl'] > 0)
    print(f"\n  K-Fold result: {positive_folds}/{len(fold_results)} positive folds")

    return full_trades, fold_results


def _run_s1_single(h1_df, min_squeeze_bars, trail_act, trail_dist, max_hold):
    df = h1_df
    atr = df['ATR'].values
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values

    if 'squeeze' in df.columns:
        squeeze = df['squeeze'].values
    else:
        bb_mid = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        squeeze = ((bb_upper < df['KC_upper']) & (bb_lower > df['KC_lower'])).astype(float).values

    times = df.index
    trades = []
    squeeze_count = 0
    in_trade = False
    buy_pos = sell_pos = None

    for i in range(50, len(df) - 1):
        cur_atr = atr[i]
        if cur_atr <= 0:
            continue

        if squeeze[i] == 1:
            squeeze_count += 1
        else:
            if squeeze_count >= min_squeeze_bars and not in_trade:
                entry_price = close[i]
                sl_dist = 1.5 * cur_atr
                buy_pos = {'entry_time': times[i], 'entry_price': entry_price,
                           'sl': entry_price - sl_dist, 'trail_stop': 0,
                           'extreme': entry_price, 'bars': 0, 'closed': False}
                sell_pos = {'entry_time': times[i], 'entry_price': entry_price,
                            'sl': entry_price + sl_dist, 'trail_stop': 999999,
                            'extreme': entry_price, 'bars': 0, 'closed': False}
                in_trade = True
            squeeze_count = 0

        if not in_trade:
            continue

        for pos, direction in [(buy_pos, 'BUY'), (sell_pos, 'SELL')]:
            if pos is None or pos['closed']:
                continue
            pos['bars'] += 1
            h, l, c = high[i], low[i], close[i]
            a = cur_atr
            exit_price = None
            reason = ""

            if direction == 'BUY':
                if l <= pos['sl']:
                    exit_price = pos['sl']; reason = "SL"
                else:
                    pos['extreme'] = max(pos['extreme'], h)
                    if h - pos['entry_price'] >= a * trail_act:
                        trail = pos['extreme'] - a * trail_dist
                        pos['trail_stop'] = max(pos['trail_stop'], trail)
                        if l <= pos['trail_stop']:
                            exit_price = pos['trail_stop']; reason = "Trail"
                    if pos['bars'] >= max_hold and exit_price is None:
                        exit_price = c; reason = "Timeout"
            else:
                if h >= pos['sl']:
                    exit_price = pos['sl']; reason = "SL"
                else:
                    pos['extreme'] = min(pos['extreme'], l)
                    if pos['entry_price'] - l >= a * trail_act:
                        trail = pos['extreme'] + a * trail_dist
                        pos['trail_stop'] = min(pos['trail_stop'], trail)
                        if h >= pos['trail_stop']:
                            exit_price = pos['trail_stop']; reason = "Trail"
                    if pos['bars'] >= max_hold and exit_price is None:
                        exit_price = c; reason = "Timeout"

            if exit_price is not None:
                pnl = (exit_price - pos['entry_price'] - SPREAD) if direction == 'BUY' else (pos['entry_price'] - exit_price - SPREAD)
                trades.append({
                    'entry_time': pos['entry_time'], 'exit_time': times[i],
                    'direction': direction, 'pnl': pnl, 'reason': reason,
                    'bars': pos['bars']
                })
                pos['closed'] = True

        if (buy_pos and buy_pos['closed']) and (sell_pos and sell_pos['closed']):
            in_trade = False
            buy_pos = sell_pos = None

    return trades


def run_s3_kfold(h1_df):
    """K-Fold validation of S3 overnight hold (NYclose to LDN open)."""
    print("\n" + "=" * 70)
    print("S3 K-Fold: NYclose_to_LDNopen (BUY)")
    print("=" * 70)

    full_trades = _run_s3_single(h1_df, entry_hour=21, exit_hour=7, direction='BUY')

    fold_results = []
    for fold_name, start, end in FOLDS:
        ts = pd.Timestamp(start, tz='UTC')
        te = pd.Timestamp(end, tz='UTC')
        fold_h1 = h1_df[(h1_df.index >= ts) & (h1_df.index < te)]
        if len(fold_h1) < 500:
            continue

        trades = _run_s3_single(fold_h1, entry_hour=21, exit_hour=7, direction='BUY')
        pnls = [t['pnl'] for t in trades]
        n = len(pnls)
        total = sum(pnls)
        wr = sum(1 for p in pnls if p > 0) / n if n > 0 else 0
        sharpe = np.mean(pnls) / np.std(pnls, ddof=1) * np.sqrt(252) if n > 1 and np.std(pnls, ddof=1) > 0 else 0

        print(f"  {fold_name} ({start}~{end}): N={n}, PnL=${total:.0f}, "
              f"WR={wr:.1%}, Sharpe={sharpe:.2f}")
        fold_results.append({
            'fold': fold_name, 'n': n, 'pnl': total, 'wr': wr,
            'sharpe': sharpe, 'start': start, 'end': end
        })

    positive_folds = sum(1 for r in fold_results if r['pnl'] > 0)
    print(f"\n  K-Fold result: {positive_folds}/{len(fold_results)} positive folds")

    return full_trades, fold_results


def _run_s3_single(h1_df, entry_hour, exit_hour, direction):
    df = h1_df
    trades = []
    dates_seen = set()

    hours = df.index.hour
    dates = df.index.date
    close_vals = df['Close'].values

    for i in range(1, len(df)):
        h = hours[i]
        d = dates[i]

        if h == entry_hour and d not in dates_seen:
            entry_price = close_vals[i]
            entry_time = df.index[i]

            for j in range(i + 1, min(i + 30, len(df))):
                if hours[j] == exit_hour:
                    exit_price = close_vals[j]
                    if direction == 'BUY':
                        pnl = exit_price - entry_price - SPREAD
                    else:
                        pnl = entry_price - exit_price - SPREAD
                    trades.append({
                        'entry_time': entry_time, 'exit_time': df.index[j],
                        'direction': direction, 'pnl': pnl,
                        'reason': 'SessionClose', 'bars': j - i
                    })
                    dates_seen.add(d)
                    break

    return trades


def run_l7_baseline(data):
    """Run L7 for correlation analysis."""
    print("\n" + "=" * 70)
    print("L7 Baseline (for correlation)")
    print("=" * 70)
    from backtest.runner import run_variant, LIVE_PARITY_KWARGS

    l7_kwargs = {**LIVE_PARITY_KWARGS}
    l7_kwargs['time_adaptive_trail'] = {'start': 2, 'decay': 0.75, 'floor': 0.003}
    l7_kwargs['min_entry_gap_hours'] = 1.0

    stats = run_variant(data, "L7_baseline", **l7_kwargs)
    return stats


def correlation_analysis(l7_trades, s1_trades, s3_trades, h1_df):
    """Compute daily PnL correlation between strategies."""
    print("\n" + "=" * 70)
    print("Portfolio Correlation Analysis")
    print("=" * 70)

    def daily_pnl_series(trades, name):
        if not trades:
            return pd.Series(dtype=float, name=name)
        data = []
        for t in trades:
            if isinstance(t, dict):
                data.append({'date': pd.Timestamp(t['exit_time']).date(), 'pnl': t['pnl']})
            else:
                data.append({'date': pd.Timestamp(t.exit_time).date(), 'pnl': t.pnl})
        df = pd.DataFrame(data)
        return df.groupby('date')['pnl'].sum().rename(name)

    l7_daily = daily_pnl_series(l7_trades, 'L7')
    s1_daily = daily_pnl_series(s1_trades, 'S1')
    s3_daily = daily_pnl_series(s3_trades, 'S3')

    all_dates = sorted(set(l7_daily.index) | set(s1_daily.index) | set(s3_daily.index))
    combined = pd.DataFrame(index=all_dates)
    combined['L7'] = l7_daily.reindex(all_dates).fillna(0)
    combined['S1'] = s1_daily.reindex(all_dates).fillna(0)
    combined['S3'] = s3_daily.reindex(all_dates).fillna(0)
    combined['L7+S1'] = combined['L7'] + combined['S1']
    combined['L7+S3'] = combined['L7'] + combined['S3']
    combined['L7+S1+S3'] = combined['L7'] + combined['S1'] + combined['S3']

    print("\n  Daily PnL Correlation Matrix:")
    corr = combined[['L7', 'S1', 'S3']].corr()
    print(corr.to_string())

    print("\n  Portfolio Stats:")
    for col in ['L7', 'S1', 'S3', 'L7+S1', 'L7+S3', 'L7+S1+S3']:
        s = combined[col]
        total = s.sum()
        sharpe = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0
        # MaxDD
        equity = s.cumsum()
        peak = equity.cummax()
        dd = (peak - equity).max()
        print(f"    {col:12s}: PnL=${total:8.0f}, Sharpe={sharpe:.2f}, MaxDD=${dd:.0f}")

    # Year-by-year for the combined portfolio
    combined['year'] = pd.to_datetime(combined.index).year
    print("\n  L7+S1+S3 Year-by-Year:")
    for y, grp in combined.groupby('year'):
        yp = grp['L7+S1+S3'].sum()
        l7p = grp['L7'].sum()
        s1p = grp['S1'].sum()
        s3p = grp['S3'].sum()
        print(f"    {y}: Total=${yp:.0f} (L7=${l7p:.0f}, S1=${s1p:.0f}, S3=${s3p:.0f})")


def main():
    t_start = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out_path = OUT_DIR / "R21_phase3_kfold.txt"
    out = open(out_path, 'w', encoding='utf-8')

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

    import sys as _sys
    old_stdout = _sys.stdout
    _sys.stdout = Tee(old_stdout, out)

    print(f"# R21 Phase 3: K-Fold Validation & Portfolio Analysis")
    print(f"# Started: {ts}")

    from backtest.runner import DataBundle
    data = DataBundle.load_default()
    h1_df = data.h1_df

    # S1 K-Fold
    s1_trades, s1_folds = run_s1_kfold(h1_df)

    # S3 K-Fold
    s3_trades, s3_folds = run_s3_kfold(h1_df)

    # L7 baseline
    l7_stats = run_l7_baseline(data)
    l7_trades = l7_stats['_trades']
    # Convert TradeRecord to dict-like for correlation
    l7_trade_dicts = []
    for t in l7_trades:
        l7_trade_dicts.append({
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'pnl': t.pnl, 'direction': t.direction
        })

    # Correlation & portfolio
    correlation_analysis(l7_trade_dicts, s1_trades, s3_trades, h1_df)

    elapsed = time.time() - t_start
    print(f"\n# Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Elapsed: {elapsed/60:.1f} minutes")

    _sys.stdout = old_stdout
    out.close()
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
