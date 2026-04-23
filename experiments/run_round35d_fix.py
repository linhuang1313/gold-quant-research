"""
R35-D Fix: SuperTrend / PSAR Deep Validation
Bug fix: mask.values -> mask (numpy array doesn't have .values)
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from backtest.runner import (DataBundle, run_variant, LIVE_PARITY_KWARGS)
from experiments.run_round32b_fix_phaseBC import compute_supertrend, compute_psar, backtest_generic_v2

OUT_DIR = Path("results/round35_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            try: f.write(data)
            except: pass
            f.flush()
    def flush(self):
        for f in self.files: f.flush()


L7_MH8 = {
    **LIVE_PARITY_KWARGS,
    'time_adaptive_trail': {'start': 2, 'decay': 0.75, 'floor': 0.003},
    'min_entry_gap_hours': 1.0,
    'keltner_max_hold_m15': 8,
}


def make_daily(trades):
    daily = {}
    for t in trades:
        exit_t = t['exit_time']
        d = pd.Timestamp(exit_t).date()
        daily.setdefault(d, 0); daily[d] += t['pnl']
    return pd.Series(daily).sort_index()


def main():
    t0 = time.time()
    out_path = OUT_DIR / "R35D_output.txt"
    out = open(out_path, 'w', encoding='utf-8', buffering=1)
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, out)

    print(f"# R35-D Fix: SuperTrend / PSAR Deep Validation")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()

    close = h1_df['Close'].values; high = h1_df['High'].values; low = h1_df['Low'].values
    times = h1_df.index
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr14 = pd.Series(tr).rolling(14).mean().values

    # D1: SuperTrend K-Fold
    print("=" * 80)
    print("Phase D: SuperTrend / PSAR Deep Validation")
    print("=" * 80)

    print(f"\n  --- D1: SuperTrend K-Fold (P20/F3.0, SL5/TP12/MH30) ---")
    atr_p20 = pd.Series(tr).rolling(20).mean().values
    st_dir = compute_supertrend(high, low, close, atr_p20, period=20, factor=3.0)
    signals = [None] * len(close)
    for i in range(1, len(close)):
        if st_dir[i] == -1 and st_dir[i-1] == 1: signals[i] = 'BUY'
        elif st_dir[i] == 1 and st_dir[i-1] == -1: signals[i] = 'SELL'

    folds = [
        ("Fold1", "2015-01-01", "2017-01-01"), ("Fold2", "2017-01-01", "2019-01-01"),
        ("Fold3", "2019-01-01", "2021-01-01"), ("Fold4", "2021-01-01", "2023-01-01"),
        ("Fold5", "2023-01-01", "2025-01-01"), ("Fold6", "2025-01-01", "2026-04-01"),
    ]
    pass_count = 0
    for fname, start, end in folds:
        mask = (h1_df.index >= pd.Timestamp(start, tz='UTC')) & (h1_df.index < pd.Timestamp(end, tz='UTC'))
        idx = np.where(np.array(mask))[0]
        if len(idx) < 500: continue
        s, e = idx[0], idx[-1]+1
        r = backtest_generic_v2(close[s:e], high[s:e], low[s:e], atr14[s:e], times[s:e],
                                signals[s:e], f"ST_{fname}", sl_atr=5.0, tp_atr=12.0, max_hold=30)
        if r['sharpe'] > 0: pass_count += 1
        print(f"  {fname}: Sharpe={r['sharpe']:.2f}, N={r['n']}, PnL=${r['total_pnl']:.0f}")
    print(f"  K-Fold pass: {pass_count}/6")

    # D2: PSAR K-Fold
    print(f"\n  --- D2: PSAR K-Fold (AF=0.01/Max=0.10) ---")
    _, psar_dir = compute_psar(high, low, close, af_start=0.01, af_max=0.10)
    signals_p = [None] * len(close)
    for i in range(1, len(close)):
        if psar_dir[i] == 1 and psar_dir[i-1] == -1: signals_p[i] = 'BUY'
        elif psar_dir[i] == -1 and psar_dir[i-1] == 1: signals_p[i] = 'SELL'

    pass_count_p = 0
    for fname, start, end in folds:
        mask = (h1_df.index >= pd.Timestamp(start, tz='UTC')) & (h1_df.index < pd.Timestamp(end, tz='UTC'))
        idx = np.where(np.array(mask))[0]
        if len(idx) < 500: continue
        s, e = idx[0], idx[-1]+1
        r = backtest_generic_v2(close[s:e], high[s:e], low[s:e], atr14[s:e], times[s:e],
                                signals_p[s:e], f"PSAR_{fname}", sl_atr=3.5, tp_atr=8.0, max_hold=50)
        if r['sharpe'] > 0: pass_count_p += 1
        print(f"  {fname}: Sharpe={r['sharpe']:.2f}, N={r['n']}, PnL=${r['total_pnl']:.0f}")
    print(f"  K-Fold pass: {pass_count_p}/6")

    # D3: Correlation with L7
    print(f"\n  --- D3: Correlation with L7 ---")
    l7 = run_variant(data, "D3_l7", verbose=False, **L7_MH8)
    l7_trades_d = {}
    for t in l7['_trades']:
        exit_t = t.exit_time if hasattr(t, 'exit_time') else t['exit_time']
        pnl = t.pnl if hasattr(t, 'pnl') else t['pnl']
        d = pd.Timestamp(exit_t).date()
        l7_trades_d.setdefault(d, 0); l7_trades_d[d] += pnl
    daily_l7 = pd.Series(l7_trades_d).sort_index()

    r_st = backtest_generic_v2(close, high, low, atr14, times, signals,
                               "ST_full", sl_atr=5.0, tp_atr=12.0, max_hold=30)
    r_psar = backtest_generic_v2(close, high, low, atr14, times, signals_p,
                                 "PSAR_full", sl_atr=3.5, tp_atr=8.0, max_hold=50)

    daily_st = make_daily(r_st['_trades'])
    daily_psar = make_daily(r_psar['_trades'])
    all_idx = sorted(set(daily_l7.index) | set(daily_st.index) | set(daily_psar.index))

    corr_df = pd.DataFrame({
        'L7': daily_l7.reindex(all_idx, fill_value=0),
        'SuperTrend': daily_st.reindex(all_idx, fill_value=0),
        'PSAR': daily_psar.reindex(all_idx, fill_value=0),
    })
    print(corr_df.corr().to_string())
    print(f"\n  ST full: Sharpe={r_st['sharpe']:.2f}, N={r_st['n']}, PnL=${r_st['total_pnl']:.0f}")
    print(f"  PSAR full: Sharpe={r_psar['sharpe']:.2f}, N={r_psar['n']}, PnL=${r_psar['total_pnl']:.0f}")

    # D4: SuperTrend cliff test
    print(f"\n  --- D4: SuperTrend Parameter Cliff ---")
    print(f"  {'Period':>7} {'Factor':>7} {'Sharpe':>7} {'N':>6} {'PnL':>9}")
    for period in [10, 14, 20, 25, 30]:
        for factor in [2.0, 2.5, 3.0, 3.5, 4.0]:
            atr_p2 = pd.Series(tr).rolling(period).mean().values
            st2 = compute_supertrend(high, low, close, atr_p2, period=period, factor=factor)
            sig2 = [None] * len(close)
            for i in range(1, len(close)):
                if st2[i] == -1 and st2[i-1] == 1: sig2[i] = 'BUY'
                elif st2[i] == 1 and st2[i-1] == -1: sig2[i] = 'SELL'
            r2 = backtest_generic_v2(close, high, low, atr14, times, sig2,
                                     f"ST_{period}_{factor}", sl_atr=5.0, tp_atr=12.0, max_hold=30)
            print(f"  {period:>7} {factor:>7.1f} {r2['sharpe']:>7.2f} {r2['n']:>6} ${r2['total_pnl']:>8.0f}")

    # D5: PSAR cliff test
    print(f"\n  --- D5: PSAR Parameter Cliff ---")
    print(f"  {'AF_start':>9} {'AF_max':>7} {'Sharpe':>7} {'N':>6} {'PnL':>9}")
    for af_s in [0.005, 0.01, 0.02, 0.03, 0.05]:
        for af_m in [0.05, 0.10, 0.15, 0.20, 0.30]:
            _, pdir = compute_psar(high, low, close, af_start=af_s, af_max=af_m)
            sig_p = [None] * len(close)
            for i in range(1, len(close)):
                if pdir[i] == 1 and pdir[i-1] == -1: sig_p[i] = 'BUY'
                elif pdir[i] == -1 and pdir[i-1] == 1: sig_p[i] = 'SELL'
            r_p = backtest_generic_v2(close, high, low, atr14, times, sig_p,
                                      f"PSAR_{af_s}_{af_m}", sl_atr=3.5, tp_atr=8.0, max_hold=50)
            print(f"  {af_s:>9.3f} {af_m:>7.2f} {r_p['sharpe']:>7.2f} {r_p['n']:>6} ${r_p['total_pnl']:>8.0f}")

    elapsed = time.time() - t0
    print(f"\n# Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Elapsed: {elapsed/60:.1f} minutes")

    sys.stdout = old_stdout
    out.close()
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
