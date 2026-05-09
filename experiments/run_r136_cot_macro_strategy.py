#!/usr/bin/env python3
"""
R136 — COT + Macro Independent Strategy v2
============================================
Phases:
  1. Load COT data + macro data (aligned_daily.csv)
  2. COT signal: net speculative z-score (20-week), contrarian
  3. Macro composite: VIX z, DXY z, real yield (US10Y - breakeven), GVZ
  4. D1-level strategy: COT contrarian + macro alignment (>=2 of 3 agree)
  5. Parameter sweep: z_threshold, macro_agree, SL/TP grid
  6. K-Fold 5-fold on D1 data
  7. Correlation with H1 strategies (target < 0.1)
  8. Walk-Forward validation
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import load_csv, DataBundle, run_variant, LIVE_PARITY_KWARGS

OUTPUT_DIR = Path("results/r136_cot_macro_strategy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01

H1_CANDIDATES = [
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-27.csv"),
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv"),
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-03-25.csv"),
]

FOLDS = [
    ("Fold1", "2015-01-01", "2017-03-01"),
    ("Fold2", "2017-03-01", "2019-06-01"),
    ("Fold3", "2019-06-01", "2021-09-01"),
    ("Fold4", "2021-09-01", "2023-12-01"),
    ("Fold5", "2023-12-01", "2027-01-01"),
]

WF_WINDOWS = [
    ("WF1", "2015-01-01", "2019-01-01", "2019-01-01", "2021-01-01"),
    ("WF2", "2016-01-01", "2020-01-01", "2020-01-01", "2022-01-01"),
    ("WF3", "2017-01-01", "2021-01-01", "2021-01-01", "2023-01-01"),
    ("WF4", "2018-01-01", "2022-01-01", "2022-01-01", "2024-01-01"),
    ("WF5", "2019-01-01", "2023-01-01", "2023-01-01", "2025-01-01"),
]

t0 = time.time()


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _mk(pos, exit_px, exit_time, reason, bar_i, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_px,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_i - pos['bar']}


def _run_exit_d1(pos, i, hi, lo, cl, spread, lot, pv, times,
                 sl_atr, tp_atr, max_hold):
    """Simplified D1 exit: SL, TP, and time-based only (no trailing on daily bars)."""
    atr = pos['atr']
    sl = atr * sl_atr; tp = atr * tp_atr
    bars = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_now = (cl - pos['entry'] - spread) * lot * pv
        if lo <= pos['entry'] - sl:
            return _mk(pos, pos['entry'] - sl, times[i], "SL", i, -sl * lot * pv)
        if hi >= pos['entry'] + tp:
            return _mk(pos, pos['entry'] + tp, times[i], "TP", i, tp * lot * pv)
        if bars >= max_hold:
            return _mk(pos, cl, times[i], "TimeExit", i, pnl_now)
    else:
        pnl_now = (pos['entry'] - cl - spread) * lot * pv
        if hi >= pos['entry'] + sl:
            return _mk(pos, pos['entry'] + sl, times[i], "SL", i, -sl * lot * pv)
        if lo <= pos['entry'] - tp:
            return _mk(pos, pos['entry'] - tp, times[i], "TP", i, tp * lot * pv)
        if bars >= max_hold:
            return _mk(pos, cl, times[i], "TimeExit", i, pnl_now)
    return None


def _trades_to_daily(trades):
    if not trades:
        return np.array([0.0])
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    if not daily:
        return np.array([0.0])
    return np.array([daily[k] for k in sorted(daily.keys())])


def _sharpe(arr):
    if len(arr) < 10 or np.std(arr, ddof=1) == 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))


def _max_dd(arr):
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 1 else 0.0


def _compute_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0, 'calmar': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    sh = _sharpe(daily)
    dd = _max_dd(daily)
    pnl = sum(pnls)
    return {
        'n': n, 'sharpe': round(sh, 3), 'pnl': round(pnl, 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1) if n else 0,
        'max_dd': round(dd, 2),
        'calmar': round(pnl / dd, 2) if dd > 0 else 9999,
    }


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_h1_and_resample_d1():
    """Load H1 CSV, resample to daily OHLCV."""
    csv_path = next((p for p in H1_CANDIDATES if p.exists()), H1_CANDIDATES[0])
    print(f"  Loading H1: {csv_path}", flush=True)
    h1 = load_csv(str(csv_path))
    print(f"  H1: {len(h1)} bars ({h1.index[0]} → {h1.index[-1]})", flush=True)

    d1 = h1.resample('D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum',
    }).dropna()
    if d1.index.tz is not None:
        d1.index = d1.index.tz_localize(None)
    d1['ATR14'] = compute_atr(d1, 14)
    d1 = d1.dropna(subset=['ATR14'])
    print(f"  D1 resampled: {len(d1)} days ({d1.index[0].date()} → {d1.index[-1].date()})", flush=True)
    return h1, d1


def load_cot():
    cot_path = DATA_DIR / "cot_gold_weekly.csv"
    if not cot_path.exists():
        print(f"  WARNING: {cot_path} not found", flush=True)
        return None
    cot = pd.read_csv(cot_path, index_col=0, parse_dates=True)
    if cot.index.tz is not None:
        cot.index = cot.index.tz_localize(None)
    print(f"  COT: {len(cot)} rows ({cot.index[0].date()} → {cot.index[-1].date()})", flush=True)
    print(f"  COT columns: {list(cot.columns)}", flush=True)
    return cot


def load_macro():
    macro_path = DATA_DIR / "external" / "aligned_daily.csv"
    if not macro_path.exists():
        print(f"  WARNING: {macro_path} not found", flush=True)
        return None
    macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    if macro.index.tz is not None:
        macro.index = macro.index.tz_localize(None)
    print(f"  Macro: {len(macro)} rows ({macro.index[0].date()} → {macro.index[-1].date()})", flush=True)
    avail = [c for c in ['VIX', 'DXY', 'US10Y', 'US2Y', 'GVZ', 'crude_wti', 'copper', 'gld_holdings']
             if c in macro.columns]
    print(f"  Macro available: {avail}", flush=True)
    return macro


def compute_cot_z(cot, window=20):
    """Compute rolling z-score of net speculative positions (weekly data, window in weeks)."""
    if cot is None:
        return None
    if 'net_spec' not in cot.columns:
        for alt in ['noncomm_net', 'noncomm_long', 'net_long']:
            if alt in cot.columns:
                if alt == 'noncomm_long' and 'noncomm_short' in cot.columns:
                    cot['net_spec'] = cot['noncomm_long'] - cot['noncomm_short']
                else:
                    cot['net_spec'] = cot[alt]
                break
    if 'net_spec' not in cot.columns:
        print("  WARNING: Cannot find net speculative column in COT data", flush=True)
        return None

    rm = cot['net_spec'].rolling(window).mean()
    rs = cot['net_spec'].rolling(window).std()
    cot['cot_z'] = (cot['net_spec'] - rm) / rs.replace(0, np.nan)
    return cot


def compute_macro_signals(macro, d1_index):
    """Build macro composite signals aligned to D1 index."""
    if macro is None:
        return pd.DataFrame(index=d1_index)

    aligned = macro.reindex(d1_index, method='ffill')
    out = pd.DataFrame(index=d1_index)

    if 'VIX' in aligned.columns:
        rm = aligned['VIX'].rolling(50).mean()
        rs = aligned['VIX'].rolling(50).std()
        out['vix_z'] = (aligned['VIX'] - rm) / rs.replace(0, np.nan)
        out['vix_bullish'] = (out['vix_z'] > 1.0).astype(int)
        out['vix_bearish'] = (out['vix_z'] < -0.5).astype(int)

    if 'DXY' in aligned.columns:
        rm = aligned['DXY'].rolling(50).mean()
        rs = aligned['DXY'].rolling(50).std()
        out['dxy_z'] = (aligned['DXY'] - rm) / rs.replace(0, np.nan)
        out['dxy_gold_bull'] = (out['dxy_z'] < -0.5).astype(int)
        out['dxy_gold_bear'] = (out['dxy_z'] > 1.0).astype(int)

    if 'US10Y' in aligned.columns and 'US2Y' in aligned.columns:
        out['real_yield_proxy'] = aligned['US10Y'] - aligned['US2Y']
        rm = out['real_yield_proxy'].rolling(50).mean()
        rs = out['real_yield_proxy'].rolling(50).std()
        out['yield_z'] = (out['real_yield_proxy'] - rm) / rs.replace(0, np.nan)
        out['yield_gold_bull'] = (out['yield_z'] < -0.5).astype(int)
        out['yield_gold_bear'] = (out['yield_z'] > 1.0).astype(int)

    if 'GVZ' in aligned.columns:
        rm = aligned['GVZ'].rolling(50).mean()
        rs = aligned['GVZ'].rolling(50).std()
        out['gvz_z'] = (aligned['GVZ'] - rm) / rs.replace(0, np.nan)
        out['gvz_high'] = (out['gvz_z'] > 1.0).astype(int)

    return out


# ═══════════════════════════════════════════════════════════════
# Strategy backtest
# ═══════════════════════════════════════════════════════════════

def backtest_cot_macro(d1, cot_daily, macro_signals, z_threshold, macro_agree,
                       sl_mult, tp_mult, max_hold=20, lot=UNIT_LOT, spread=SPREAD,
                       start=None, end=None):
    """
    D1 strategy: COT contrarian + macro alignment.
    - BUY when cot_z < -z_threshold AND >= macro_agree bullish factors
    - SELL when cot_z > z_threshold AND >= macro_agree bearish factors
    """
    c = d1['Close'].values; h = d1['High'].values; lo = d1['Low'].values
    atr_arr = d1['ATR14'].values; times = d1.index; n = len(d1)
    dates_str = [str(t.date()) for t in d1.index]

    cot_z = cot_daily.reindex(d1.index, method='ffill')['cot_z'].values \
        if cot_daily is not None and 'cot_z' in cot_daily.columns \
        else np.full(n, np.nan)

    macro_aligned = macro_signals.reindex(d1.index, method='ffill')

    bull_cols = [c for c in macro_aligned.columns if c.endswith('_bull') or c == 'gvz_high']
    bear_cols = [c for c in macro_aligned.columns if c.endswith('_bear')]

    bull_count = macro_aligned[bull_cols].sum(axis=1).values if bull_cols else np.zeros(n)
    bear_count = macro_aligned[bear_cols].sum(axis=1).values if bear_cols else np.zeros(n)

    trades = []; pos = None; last_exit = -999

    for i in range(1, n):
        if start and dates_str[i] < start:
            continue
        if end and dates_str[i] > end:
            break

        if pos is not None:
            result = _run_exit_d1(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                  sl_mult, tp_mult, max_hold)
            if result:
                trades.append(result); pos = None; last_exit = i
            continue

        if i - last_exit < 2:
            continue
        if np.isnan(atr_arr[i]) or atr_arr[i] < 0.1:
            continue
        if np.isnan(cot_z[i]):
            continue

        if cot_z[i] < -z_threshold and bull_count[i] >= macro_agree:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i,
                   'time': times[i], 'atr': atr_arr[i]}
        elif cot_z[i] > z_threshold and bear_count[i] >= macro_agree:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i,
                   'time': times[i], 'atr': atr_arr[i]}

    return trades


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80, flush=True)
    print("  R136 — COT + Macro Independent Strategy v2", flush=True)
    print("=" * 80, flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Load data
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 1: Load Data", flush=True)
    print("=" * 70, flush=True)

    h1, d1 = load_h1_and_resample_d1()
    cot_raw = load_cot()
    macro_raw = load_macro()

    all_results = {
        'experiment': 'R136 COT + Macro Independent Strategy v2',
        'd1_days': len(d1),
    }

    if cot_raw is None:
        print("  FATAL: COT data not available.", flush=True)
        all_results['error'] = 'COT data not found'
        with open(OUTPUT_DIR / "r136_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        return

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: COT signal
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: COT Signal — Net Speculative Z-Score (20-week)", flush=True)
    print("=" * 70, flush=True)

    cot = compute_cot_z(cot_raw, window=20)
    if cot is None or 'cot_z' not in cot.columns:
        print("  FATAL: Could not compute COT z-score.", flush=True)
        all_results['error'] = 'COT z-score computation failed'
        with open(OUTPUT_DIR / "r136_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        return

    valid_z = cot['cot_z'].dropna()
    print(f"  COT z-score: {len(valid_z)} valid weeks", flush=True)
    print(f"  Range: [{valid_z.min():.2f}, {valid_z.max():.2f}]", flush=True)
    print(f"  Current: {valid_z.iloc[-1]:.2f}", flush=True)
    print(f"  % below -1.5: {(valid_z < -1.5).mean()*100:.1f}%", flush=True)
    print(f"  % above +1.5: {(valid_z > 1.5).mean()*100:.1f}%", flush=True)

    cot_daily = cot.reindex(d1.index, method='ffill')

    all_results['phase2_cot'] = {
        'valid_weeks': len(valid_z),
        'z_range': [round(valid_z.min(), 2), round(valid_z.max(), 2)],
        'pct_below_neg15': round((valid_z < -1.5).mean()*100, 1),
        'pct_above_pos15': round((valid_z > 1.5).mean()*100, 1),
    }

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: Macro composite
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3: Macro Composite Signals", flush=True)
    print("=" * 70, flush=True)

    macro_signals = compute_macro_signals(macro_raw, d1.index)
    print(f"  Macro signal columns: {list(macro_signals.columns)}", flush=True)

    bull_cols = [c for c in macro_signals.columns if c.endswith('_bull') or c == 'gvz_high']
    bear_cols = [c for c in macro_signals.columns if c.endswith('_bear')]
    print(f"  Bullish factors: {bull_cols}", flush=True)
    print(f"  Bearish factors: {bear_cols}", flush=True)

    if bull_cols:
        bull_sum = macro_signals[bull_cols].sum(axis=1)
        print(f"\n  Bull factor distribution:", flush=True)
        for v in range(int(bull_sum.max()) + 1):
            pct = (bull_sum == v).mean() * 100
            print(f"    {v} factors active: {pct:.1f}%", flush=True)

    if bear_cols:
        bear_sum = macro_signals[bear_cols].sum(axis=1)
        print(f"\n  Bear factor distribution:", flush=True)
        for v in range(int(bear_sum.max()) + 1):
            pct = (bear_sum == v).mean() * 100
            print(f"    {v} factors active: {pct:.1f}%", flush=True)

    all_results['phase3_macro'] = {
        'bull_factors': bull_cols, 'bear_factors': bear_cols,
        'columns': list(macro_signals.columns),
    }

    # ═══════════════════════════════════════════════════════════════
    # Phase 4-5: Strategy backtest + parameter sweep
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 4-5: Parameter Sweep", flush=True)
    print("=" * 70, flush=True)

    z_thresholds = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
    macro_agrees = [1, 2, 3]
    sl_mults = [1.5, 2.0, 3.0]
    tp_mults = [3.0, 4.0, 6.0]
    max_hold_d1 = 20

    grid_results = []
    total = len(z_thresholds) * len(macro_agrees) * len(sl_mults) * len(tp_mults)
    print(f"  Grid size: {total} combos", flush=True)
    print(f"\n  {'Config':<42s}  {'n':>5s}  {'Sharpe':>7s}  {'PnL':>10s}  {'WR':>6s}  {'MaxDD':>8s}", flush=True)
    print("  " + "-" * 85, flush=True)

    done = 0
    for zt, ma, sl_m, tp_m in product(z_thresholds, macro_agrees, sl_mults, tp_mults):
        trades = backtest_cot_macro(
            d1, cot_daily, macro_signals, z_threshold=zt, macro_agree=ma,
            sl_mult=sl_m, tp_mult=tp_m, max_hold=max_hold_d1,
        )
        st = _compute_stats(trades)
        label = f"z={zt}/agree={ma}/sl={sl_m}/tp={tp_m}"
        grid_results.append({'z_threshold': zt, 'macro_agree': ma,
                             'sl_mult': sl_m, 'tp_mult': tp_m, **st})
        if st['sharpe'] > 0.2 and st['n'] >= 10:
            print(f"  {label:<42s}  {st['n']:5d}  {st['sharpe']:7.3f}  ${st['pnl']:>9.0f}  "
                  f"{st['wr']:5.1f}%  ${st['max_dd']:>7.0f}", flush=True)
        done += 1
        if done % 50 == 0:
            print(f"  ... {done}/{total} done ({time.time()-t0:.0f}s)", flush=True)

    grid_results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Top 10 by Sharpe:", flush=True)
    for i, g in enumerate(grid_results[:10]):
        print(f"    #{i+1}: z={g['z_threshold']}/agree={g['macro_agree']}/"
              f"sl={g['sl_mult']}/tp={g['tp_mult']} -> Sharpe={g['sharpe']:.3f}, "
              f"n={g['n']}, PnL=${g['pnl']:.0f}, WR={g['wr']:.1f}%", flush=True)

    all_results['phase5_param_sweep'] = grid_results[:30]

    if not grid_results or grid_results[0]['n'] < 3:
        print("\n  WARNING: No viable parameter set. Using defaults.", flush=True)
        best = {'z_threshold': 1.5, 'macro_agree': 2, 'sl_mult': 2.0, 'tp_mult': 4.0}
    else:
        best = grid_results[0]

    print(f"\n  Best params: z={best['z_threshold']}, agree={best['macro_agree']}, "
          f"sl={best['sl_mult']}, tp={best['tp_mult']}", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Phase 6: K-Fold 5-fold on D1 data
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 6: K-Fold 5-Fold Validation (D1)", flush=True)
    print("=" * 70, flush=True)

    kfold_results = []
    print(f"\n  {'Fold':<8s}  {'n':>5s}  {'Sharpe':>7s}  {'PnL':>10s}  {'WR':>6s}  {'MaxDD':>8s}", flush=True)
    print("  " + "-" * 50, flush=True)

    for fold_name, fold_start, fold_end in FOLDS:
        trades = backtest_cot_macro(
            d1, cot_daily, macro_signals,
            z_threshold=best['z_threshold'], macro_agree=best['macro_agree'],
            sl_mult=best['sl_mult'], tp_mult=best['tp_mult'],
            max_hold=max_hold_d1, start=fold_start, end=fold_end,
        )
        st = _compute_stats(trades)
        kfold_results.append({'fold': fold_name, **st})
        print(f"  {fold_name:<8s}  {st['n']:5d}  {st['sharpe']:7.3f}  ${st['pnl']:>9.0f}  "
              f"{st['wr']:5.1f}%  ${st['max_dd']:>7.0f}", flush=True)

    pos_folds = sum(1 for f in kfold_results if f['sharpe'] > 0)
    avg_sharpe = np.mean([f['sharpe'] for f in kfold_results])
    print(f"\n  Positive folds: {pos_folds}/{len(kfold_results)}", flush=True)
    print(f"  Average fold Sharpe: {avg_sharpe:.3f}", flush=True)

    all_results['phase6_kfold'] = kfold_results
    all_results['phase6_summary'] = {
        'positive_folds': pos_folds, 'total_folds': len(kfold_results),
        'avg_sharpe': round(avg_sharpe, 3),
    }

    # ═══════════════════════════════════════════════════════════════
    # Phase 7: Correlation with H1 strategies
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 7: Correlation with H1 Strategies (target < 0.1)", flush=True)
    print("=" * 70, flush=True)

    r136_full = backtest_cot_macro(
        d1, cot_daily, macro_signals,
        z_threshold=best['z_threshold'], macro_agree=best['macro_agree'],
        sl_mult=best['sl_mult'], tp_mult=best['tp_mult'],
        max_hold=max_hold_d1,
    )
    r136_daily = _trades_to_daily(r136_full)
    r136_stats = _compute_stats(r136_full)
    print(f"  R136 full: n={r136_stats['n']}, Sharpe={r136_stats['sharpe']:.3f}, "
          f"PnL=${r136_stats['pnl']:.0f}", flush=True)

    corr_results = {}
    try:
        bundle = DataBundle.load_default()
        for strat_name in ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']:
            try:
                result = run_variant(bundle, strat_name, verbose=False, **LIVE_PARITY_KWARGS)
                raw_trades = result.get('_trades', [])
                strat_trades = [{'pnl': t.pnl, 'exit_time': t.exit_time} for t in raw_trades]
                other_daily = _trades_to_daily(strat_trades)

                min_len = min(len(r136_daily), len(other_daily))
                if min_len < 20:
                    corr_results[strat_name] = {'error': 'too few overlapping days'}
                    continue
                c_val = np.corrcoef(r136_daily[:min_len], other_daily[:min_len])[0, 1]
                corr_results[strat_name] = round(c_val, 4)
                status = "GOOD" if abs(c_val) < 0.1 else ("OK" if abs(c_val) < 0.3 else "HIGH")
                print(f"  Corr(R136, {strat_name}): {c_val:.4f} [{status}]", flush=True)
            except Exception as e:
                print(f"    WARNING: {strat_name} failed: {e}", flush=True)
                corr_results[strat_name] = {'error': str(e)}
    except Exception as e:
        print(f"  WARNING: Could not load DataBundle: {e}", flush=True)
        corr_results['error'] = str(e)

    all_results['phase7_correlation'] = corr_results

    # ═══════════════════════════════════════════════════════════════
    # Phase 8: Walk-Forward validation
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 8: Walk-Forward Validation (4yr train / 2yr test)", flush=True)
    print("=" * 70, flush=True)

    wf_results = []
    print(f"\n  {'Window':<6s}  {'Train Sharpe':>12s}  {'Test Sharpe':>12s}  {'Test n':>6s}  "
          f"{'Test PnL':>10s}  {'Test WR':>7s}", flush=True)
    print("  " + "-" * 65, flush=True)

    for wf_name, tr_start, tr_end, te_start, te_end in WF_WINDOWS:
        train_grid = []
        for zt, ma in product([1.0, 1.5, 2.0], [1, 2, 3]):
            tr_trades = backtest_cot_macro(
                d1, cot_daily, macro_signals,
                z_threshold=zt, macro_agree=ma,
                sl_mult=best['sl_mult'], tp_mult=best['tp_mult'],
                max_hold=max_hold_d1, start=tr_start, end=tr_end,
            )
            st = _compute_stats(tr_trades)
            train_grid.append({'z_threshold': zt, 'macro_agree': ma, **st})

        train_grid.sort(key=lambda x: x['sharpe'], reverse=True)
        wf_best = train_grid[0] if train_grid else best

        te_trades = backtest_cot_macro(
            d1, cot_daily, macro_signals,
            z_threshold=wf_best['z_threshold'], macro_agree=wf_best['macro_agree'],
            sl_mult=best['sl_mult'], tp_mult=best['tp_mult'],
            max_hold=max_hold_d1, start=te_start, end=te_end,
        )
        te_st = _compute_stats(te_trades)

        wf_results.append({
            'window': wf_name,
            'train_sharpe': wf_best.get('sharpe', 0),
            'train_params': {'z_threshold': wf_best['z_threshold'],
                             'macro_agree': wf_best['macro_agree']},
            'test': te_st,
        })
        print(f"  {wf_name:<6s}  {wf_best.get('sharpe', 0):12.3f}  {te_st['sharpe']:12.3f}  "
              f"{te_st['n']:6d}  ${te_st['pnl']:>9.0f}  {te_st['wr']:6.1f}%", flush=True)

    wf_oos_sharpes = [w['test']['sharpe'] for w in wf_results]
    wf_pos = sum(1 for s in wf_oos_sharpes if s > 0)
    print(f"\n  OOS positive: {wf_pos}/{len(wf_results)}, "
          f"avg OOS Sharpe: {np.mean(wf_oos_sharpes):.3f}", flush=True)

    all_results['phase8_walk_forward'] = wf_results

    # ═══════════════════════════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    all_results['runtime_sec'] = round(elapsed, 1)
    all_results['best_params'] = {
        'z_threshold': best['z_threshold'], 'macro_agree': best['macro_agree'],
        'sl_mult': best['sl_mult'], 'tp_mult': best['tp_mult'],
    }
    out_file = OUTPUT_DIR / "r136_results.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*80}", flush=True)
    print(f"  R136 complete in {elapsed/60:.1f} min. Results → {out_file}", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == '__main__':
    main()
