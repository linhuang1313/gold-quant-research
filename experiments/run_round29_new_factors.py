"""
R29: New Factor Exploration
============================
A: ATR Ratio (fast/slow) as regime signal — IC analysis + trail/lot adjustment
B: Multi-period TSMOM as independent strategy
C: GVZ-style vol sizing (proxy: realized vol rank)
D: Volume anomaly as signal quality factor
E: Second-order momentum for trail adjustment
"""

import sys, os, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from backtest.runner import DataBundle, run_variant, run_kfold, LIVE_PARITY_KWARGS

OUT_DIR = Path("results/round29_results")
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


def get_l7_mh8():
    kw = {**LIVE_PARITY_KWARGS}
    kw['regime_config'] = {
        'low':    {'trail_act': 0.30, 'trail_dist': 0.06},
        'normal': {'trail_act': 0.20, 'trail_dist': 0.04},
        'high':   {'trail_act': 0.08, 'trail_dist': 0.01},
    }
    kw['time_adaptive_trail'] = True
    kw['time_adaptive_trail_start'] = 2
    kw['time_adaptive_trail_decay'] = 0.75
    kw['time_adaptive_trail_floor'] = 0.003
    kw['min_entry_gap_hours'] = 1.0
    kw['keltner_max_hold_m15'] = 8
    return kw


# ═══════════════════════════════════════════════════════════════
# Phase A: ATR Ratio Factor
# ═══════════════════════════════════════════════════════════════

def run_phase_A(h1_df):
    """ATR fast/slow ratio as regime signal."""
    print("\n" + "=" * 80)
    print("Phase A: ATR Ratio (Fast/Slow) Factor Analysis")
    print("=" * 80)

    df = h1_df.copy()
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)

    print("\n  --- A1: Factor Construction & IC Analysis ---")
    print(f"  Testing ATR(fast)/ATR(slow) ratio as predictor of future returns\n")

    for fast, slow in [(3, 14), (5, 14), (5, 20), (5, 50), (10, 30)]:
        atr_fast = tr.rolling(fast).mean()
        atr_slow = tr.rolling(slow).mean()
        ratio = (atr_fast / atr_slow).replace([np.inf, -np.inf], np.nan)

        for fwd in [1, 4, 8, 20]:
            fwd_ret = df['Close'].pct_change(fwd).shift(-fwd)
            valid = pd.DataFrame({'ratio': ratio, 'ret': fwd_ret}).dropna()
            if len(valid) < 500: continue
            ic = valid['ratio'].corr(valid['ret'])
            # Rank IC
            ric = valid['ratio'].corr(valid['ret'], method='spearman')
            print(f"  ATR({fast}/{slow}) vs ret_{fwd}h:  IC={ic:+.4f}  RankIC={ric:+.4f}  N={len(valid)}")
        print()

    print("  --- A2: ATR Ratio Regime Buckets ---")
    atr5 = tr.rolling(5).mean()
    atr20 = tr.rolling(20).mean()
    ratio = (atr5 / atr20).replace([np.inf, -np.inf], np.nan).dropna()

    df_a = df.loc[ratio.index].copy()
    df_a['atr_ratio'] = ratio

    # Future absolute return (proxy for trend strength)
    df_a['abs_ret_4'] = df_a['Close'].pct_change(4).shift(-4).abs()
    df_a['ret_4'] = df_a['Close'].pct_change(4).shift(-4)
    df_a = df_a.dropna(subset=['abs_ret_4', 'atr_ratio'])

    for q_label, lo, hi in [("Contracting (<0.8)", 0, 0.8),
                             ("Neutral (0.8-1.2)", 0.8, 1.2),
                             ("Expanding (>1.2)", 1.2, 10)]:
        mask = (df_a['atr_ratio'] >= lo) & (df_a['atr_ratio'] < hi)
        sub = df_a[mask]
        if len(sub) < 100: continue
        mean_abs = sub['abs_ret_4'].mean()
        mean_ret = sub['ret_4'].mean()
        sharpe_proxy = mean_ret / sub['ret_4'].std() * np.sqrt(252/4) if sub['ret_4'].std() > 0 else 0
        print(f"  {q_label}: N={len(sub)}, MeanAbsRet={mean_abs:.5f}, "
              f"MeanRet={mean_ret:+.5f}, SharpeProxy={sharpe_proxy:.2f}")

    # Quintile analysis
    print(f"\n  --- A3: ATR Ratio Quintile Analysis ---")
    df_a['ratio_q'] = pd.qcut(df_a['atr_ratio'], 5, labels=False, duplicates='drop')
    print(f"  {'Quintile':>8} {'N':>6} {'MeanAbsRet4':>12} {'MeanRet4':>10} {'SharpeProxy':>12}")
    for q in sorted(df_a['ratio_q'].unique()):
        sub = df_a[df_a['ratio_q'] == q]
        mr = sub['ret_4'].mean()
        sp = mr / sub['ret_4'].std() * np.sqrt(252/4) if sub['ret_4'].std() > 0 else 0
        lo_r = sub['atr_ratio'].min()
        hi_r = sub['atr_ratio'].max()
        print(f"  Q{int(q)}({lo_r:.2f}-{hi_r:.2f}) {len(sub):>6} {sub['abs_ret_4'].mean():>12.5f} "
              f"{mr:>+10.5f} {sp:>12.2f}")


# ═══════════════════════════════════════════════════════════════
# Phase B: TSMOM Independent Strategy
# ═══════════════════════════════════════════════════════════════

def run_phase_B(h1_df):
    """Multi-period time-series momentum as independent strategy."""
    print("\n" + "=" * 80)
    print("Phase B: Time-Series Momentum (TSMOM) Strategy")
    print("=" * 80)

    df = h1_df.copy()
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    print("\n  --- B1: Single-Period TSMOM Signal Scan ---")
    print(f"  {'Period':>8} {'N':>6} {'Sharpe':>8} {'PnL':>10} {'WR':>6} {'MaxDD':>8}")

    for lookback in [24, 48, 120, 240, 480, 720]:  # hours
        label = f"TSMOM_{lookback}h"
        r = _run_tsmom(df, label, lookback=lookback, sl_atr=3.5, tp_atr=8.0,
                       trail_act=0.28, trail_dist=0.06, max_hold=30)
        print(f"  {lookback:>6}h {r['n']:>6} {r['sharpe']:>8.2f} ${r['total_pnl']:>9.0f} "
              f"{r['win_rate']:>5.1f}% ${r['max_dd']:>7.0f}")

    print(f"\n  --- B2: Multi-Period Weighted TSMOM ---")
    print(f"  {'Combo':>20} {'N':>6} {'Sharpe':>8} {'PnL':>10} {'WR':>6} {'MaxDD':>8}")

    combos = [
        ("5d+20d", [(120, 0.5), (480, 0.5)]),
        ("5d+20d+60d", [(120, 0.33), (480, 0.33), (1440, 0.34)]),
        ("1d+5d+20d", [(24, 0.33), (120, 0.33), (480, 0.34)]),
        ("20d+60d", [(480, 0.5), (1440, 0.5)]),
    ]
    best_combo = None; best_sharpe = 0
    for name, weights in combos:
        r = _run_tsmom_multi(df, f"TSMOM_{name}", weights,
                             sl_atr=3.5, tp_atr=8.0, trail_act=0.28, trail_dist=0.06, max_hold=30)
        print(f"  {name:>20} {r['n']:>6} {r['sharpe']:>8.2f} ${r['total_pnl']:>9.0f} "
              f"{r['win_rate']:>5.1f}% ${r['max_dd']:>7.0f}")
        if r['sharpe'] > best_sharpe:
            best_sharpe = r['sharpe']; best_combo = name

    if best_sharpe > 1.0:
        print(f"\n  >>> Best TSMOM: {best_combo}, Sharpe={best_sharpe:.2f}")
        print(f"  Running parameter sweep on best combo...")
        _tsmom_param_sweep(df, best_combo, combos)
    else:
        print(f"\n  >>> No TSMOM combo reaches Sharpe > 1.0")

    # B3: TSMOM K-Fold if promising
    if best_sharpe > 2.0:
        print(f"\n  --- B3: TSMOM K-Fold ---")
        _tsmom_kfold(h1_df, best_combo, combos)


def _run_tsmom(df, label, lookback=120, sl_atr=3.5, tp_atr=8.0,
               trail_act=0.28, trail_dist=0.06, max_hold=30, spread=0.30,
               lot=0.03, return_trades=False):
    """Single-period TSMOM: go long if close > close[lookback], short if below."""
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    atr = df['ATR'].values if 'ATR' in df.columns else np.full(len(df), 1.0)
    times = df.index; n = len(close)
    trades = []; pos = None; last_exit = -999

    mom = np.full(n, np.nan)
    for i in range(lookback, n):
        mom[i] = close[i] / close[i - lookback] - 1.0

    for i in range(lookback + 1, n):
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (high[i] - pos['entry'] - spread) * lot * 100
                pnl_l = (low[i] - pos['entry'] - spread) * lot * 100
                pnl_c = (close[i] - pos['entry'] - spread) * lot * 100
            else:
                pnl_h = (pos['entry'] - low[i] - spread) * lot * 100
                pnl_l = (pos['entry'] - high[i] - spread) * lot * 100
                pnl_c = (pos['entry'] - close[i] - spread) * lot * 100
            tp_val = tp_atr * pos['atr'] * lot * 100
            sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                _rec(trades, pos, times[i], "TP", i, tp_val); exited = True
            elif pnl_l <= -sl_val:
                _rec(trades, pos, times[i], "SL", i, -sl_val); exited = True
            else:
                ad = trail_act * pos['atr']; td = trail_dist * pos['atr']
                if pos['dir'] == 'BUY' and high[i] - pos['entry'] >= ad:
                    ts = high[i] - td
                    if low[i] <= ts:
                        _rec(trades, pos, times[i], "Trail", i,
                             (ts - pos['entry'] - spread) * lot * 100); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - low[i] >= ad:
                    ts = low[i] + td
                    if high[i] >= ts:
                        _rec(trades, pos, times[i], "Trail", i,
                             (pos['entry'] - ts - spread) * lot * 100); exited = True
                if not exited and held >= max_hold:
                    _rec(trades, pos, times[i], "Timeout", i, pnl_c); exited = True
            # Signal reversal exit
            if not exited and not np.isnan(mom[i]):
                if pos['dir'] == 'BUY' and mom[i] < 0:
                    _rec(trades, pos, times[i], "Reversal", i, pnl_c); exited = True
                elif pos['dir'] == 'SELL' and mom[i] > 0:
                    _rec(trades, pos, times[i], "Reversal", i, pnl_c); exited = True
            if exited: pos = None; last_exit = i; continue

        if pos is not None or i - last_exit < 2: continue
        if np.isnan(mom[i]) or np.isnan(atr[i]) or atr[i] < 0.1: continue

        if mom[i] > 0 and mom[i - 1] <= 0:
            pos = {'dir': 'BUY', 'entry': close[i] + spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif mom[i] < 0 and mom[i - 1] >= 0:
            pos = {'dir': 'SELL', 'entry': close[i] - spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}

    return _stats(trades, label, return_trades)


def _run_tsmom_multi(df, label, weights, **kw):
    """Multi-period weighted TSMOM: enter when weighted sum crosses zero."""
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    atr = df['ATR'].values if 'ATR' in df.columns else np.full(len(df), 1.0)
    times = df.index; n = len(close)
    max_lb = max(lb for lb, _ in weights)
    trades = []; pos = None; last_exit = -999
    spread = kw.get('spread', 0.30); lot = kw.get('lot', 0.03)
    sl_atr = kw['sl_atr']; tp_atr = kw['tp_atr']
    trail_act = kw['trail_act']; trail_dist = kw['trail_dist']
    max_hold = kw['max_hold']

    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0
        for lb, w in weights:
            if i >= lb:
                s += w * np.sign(close[i] / close[i - lb] - 1.0)
        score[i] = s

    for i in range(max_lb + 1, n):
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (high[i] - pos['entry'] - spread) * lot * 100
                pnl_l = (low[i] - pos['entry'] - spread) * lot * 100
                pnl_c = (close[i] - pos['entry'] - spread) * lot * 100
            else:
                pnl_h = (pos['entry'] - low[i] - spread) * lot * 100
                pnl_l = (pos['entry'] - high[i] - spread) * lot * 100
                pnl_c = (pos['entry'] - close[i] - spread) * lot * 100
            tp_val = tp_atr * pos['atr'] * lot * 100
            sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                _rec(trades, pos, times[i], "TP", i, tp_val); exited = True
            elif pnl_l <= -sl_val:
                _rec(trades, pos, times[i], "SL", i, -sl_val); exited = True
            else:
                ad = trail_act * pos['atr']; td = trail_dist * pos['atr']
                if pos['dir'] == 'BUY' and high[i] - pos['entry'] >= ad:
                    ts = high[i] - td
                    if low[i] <= ts:
                        _rec(trades, pos, times[i], "Trail", i,
                             (ts - pos['entry'] - spread) * lot * 100); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - low[i] >= ad:
                    ts = low[i] + td
                    if high[i] >= ts:
                        _rec(trades, pos, times[i], "Trail", i,
                             (pos['entry'] - ts - spread) * lot * 100); exited = True
                if not exited and held >= max_hold:
                    _rec(trades, pos, times[i], "Timeout", i, pnl_c); exited = True
            if not exited and not np.isnan(score[i]):
                if pos['dir'] == 'BUY' and score[i] < 0:
                    _rec(trades, pos, times[i], "Reversal", i, pnl_c); exited = True
                elif pos['dir'] == 'SELL' and score[i] > 0:
                    _rec(trades, pos, times[i], "Reversal", i, pnl_c); exited = True
            if exited: pos = None; last_exit = i; continue

        if pos is not None or i - last_exit < 2: continue
        if np.isnan(score[i]) or np.isnan(atr[i]) or atr[i] < 0.1: continue

        if score[i] > 0 and score[i - 1] <= 0:
            pos = {'dir': 'BUY', 'entry': close[i] + spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i - 1] >= 0:
            pos = {'dir': 'SELL', 'entry': close[i] - spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}

    return _stats(trades, label, False)


def _tsmom_param_sweep(df, best_name, combos):
    """Sweep SL/TP/Trail/MaxHold on best TSMOM combo."""
    best_weights = None
    for name, w in combos:
        if name == best_name: best_weights = w; break
    if not best_weights: return

    print(f"\n  {'Config':<30} {'N':>5} {'Sharpe':>7} {'PnL':>9} {'WR':>5}")
    for sl in [2.0, 3.0, 3.5, 5.0]:
        for tp in [4.0, 6.0, 8.0, 12.0]:
            for mh in [15, 30, 50]:
                lbl = f"SL{sl}_TP{tp}_MH{mh}"
                r = _run_tsmom_multi(df, lbl, best_weights,
                                     sl_atr=sl, tp_atr=tp, trail_act=0.28,
                                     trail_dist=0.06, max_hold=mh)
                if r['n'] > 50:
                    print(f"  {lbl:<30} {r['n']:>5} {r['sharpe']:>7.2f} ${r['total_pnl']:>8.0f} {r['win_rate']:>4.1f}%")


def _tsmom_kfold(h1_df, best_name, combos):
    best_weights = None
    for name, w in combos:
        if name == best_name: best_weights = w; break
    if not best_weights: return

    folds = [("2015-01-01","2017-01-01"),("2017-01-01","2019-01-01"),
             ("2019-01-01","2021-01-01"),("2021-01-01","2023-01-01"),
             ("2023-01-01","2025-01-01"),("2025-01-01","2026-04-01")]
    sharpes = []
    for i, (s, e) in enumerate(folds):
        ts = pd.Timestamp(s, tz='UTC'); te = pd.Timestamp(e, tz='UTC')
        h1s = h1_df[(h1_df.index >= ts) & (h1_df.index < te)]
        if len(h1s) < 500: continue
        tr = pd.DataFrame({
            'hl': h1s['High'] - h1s['Low'],
            'hc': (h1s['High'] - h1s['Close'].shift(1)).abs(),
            'lc': (h1s['Low'] - h1s['Close'].shift(1)).abs(),
        }).max(axis=1)
        h1s = h1s.copy()
        h1s['ATR'] = tr.rolling(14).mean()
        r = _run_tsmom_multi(h1s, f"TSMOM_F{i+1}", best_weights,
                             sl_atr=3.5, tp_atr=8.0, trail_act=0.28, trail_dist=0.06, max_hold=30)
        print(f"    F{i+1}: N={r['n']}, Sharpe={r['sharpe']:.2f}, PnL=${r['total_pnl']:.0f}")
        sharpes.append(r['sharpe'])
    if sharpes:
        pos = sum(1 for s in sharpes if s > 0)
        print(f"  K-Fold: {pos}/{len(sharpes)} positive, mean={np.mean(sharpes):.2f}")


# ═══════════════════════════════════════════════════════════════
# Phase C: Realized Volatility Rank as Sizing Signal
# ═══════════════════════════════════════════════════════════════

def run_phase_C(data, h1_df):
    """Use rolling realized vol rank as lot sizing signal (GVZ proxy)."""
    print("\n" + "=" * 80)
    print("Phase C: Realized Volatility Rank Sizing (GVZ Proxy)")
    print("=" * 80)

    l7 = run_variant(data, "L7MH8_volsz", verbose=False, **get_l7_mh8())
    trades = l7['_trades']
    if not trades:
        print("  No trades!"); return

    df = h1_df.copy()
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['RealVol20'] = df['Close'].pct_change().rolling(20*24).std() * np.sqrt(252*24)
    df['VolRank'] = df['RealVol20'].rolling(252*24, min_periods=100).rank(pct=True)

    print(f"\n  --- C1: Trade Performance by Vol Rank at Entry ---")
    pnls_by_q = {0: [], 1: [], 2: [], 3: [], 4: []}
    for t in trades:
        entry_time = pd.Timestamp(t.entry_time)
        idx = df.index.searchsorted(entry_time)
        if idx >= len(df): continue
        vr = df['VolRank'].iloc[min(idx, len(df)-1)]
        if np.isnan(vr): continue
        q = min(int(vr * 5), 4)
        pnls_by_q[q].append(t.pnl)

    print(f"  {'VolQuintile':>12} {'N':>6} {'MeanPnL':>10} {'WR':>6} {'TotalPnL':>10}")
    for q in range(5):
        pnls = pnls_by_q[q]
        if not pnls: continue
        wins = sum(1 for p in pnls if p > 0)
        print(f"  Q{q} (p{q*20}-{(q+1)*20}) {len(pnls):>6} ${np.mean(pnls):>9.2f} "
              f"{wins/len(pnls)*100:>5.1f}% ${sum(pnls):>9.0f}")

    print(f"\n  --- C2: Simulated Vol-Rank Sizing ---")
    print(f"  {'Scheme':>30} {'Sharpe':>8} {'PnL':>10} {'MaxDD':>8}")

    base_daily = {}
    for t in trades:
        d = pd.Timestamp(t.exit_time).date()
        base_daily.setdefault(d, 0); base_daily[d] += t.pnl

    def _sh(daily):
        da = np.array(list(daily.values()))
        return da.mean() / da.std() * np.sqrt(252) if len(da) > 5 and da.std() > 0 else 0

    base_sh = _sh(base_daily)
    base_pnl = sum(base_daily.values())
    base_eq = np.cumsum(list(base_daily.values()))
    base_dd = (np.maximum.accumulate(base_eq) - base_eq).max()
    print(f"  {'Baseline (1.0x all)':>30} {base_sh:>8.2f} ${base_pnl:>9.0f} ${base_dd:>7.0f}")

    for scheme_name, q_mults in [
        ("HighVol 0.5x", {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.75, 4: 0.5}),
        ("HighVol 0.3x", {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.6, 4: 0.3}),
        ("LowVol 1.5x", {0: 1.5, 1: 1.2, 2: 1.0, 3: 1.0, 4: 1.0}),
        ("Inverse Vol", {0: 1.5, 1: 1.2, 2: 1.0, 3: 0.8, 4: 0.5}),
        ("Extreme Only", {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.3}),
    ]:
        daily = {}
        for t in trades:
            entry_time = pd.Timestamp(t.entry_time)
            idx = df.index.searchsorted(entry_time)
            if idx >= len(df): continue
            vr = df['VolRank'].iloc[min(idx, len(df)-1)]
            if np.isnan(vr): mult = 1.0
            else: mult = q_mults.get(min(int(vr * 5), 4), 1.0)
            d = pd.Timestamp(t.exit_time).date()
            daily.setdefault(d, 0); daily[d] += t.pnl * mult
        sh = _sh(daily)
        pnl = sum(daily.values())
        eq = np.cumsum(list(daily.values()))
        dd = (np.maximum.accumulate(eq) - eq).max() if len(eq) > 0 else 0
        delta = sh - base_sh
        print(f"  {scheme_name:>30} {sh:>8.2f} ${pnl:>9.0f} ${dd:>7.0f}  ({delta:+.2f})")


# ═══════════════════════════════════════════════════════════════
# Phase D: Volume Anomaly Factor
# ═══════════════════════════════════════════════════════════════

def run_phase_D(data, h1_df):
    """Test tick volume as signal quality factor for lot adjustment."""
    print("\n" + "=" * 80)
    print("Phase D: Volume Anomaly as Signal Quality Factor")
    print("=" * 80)

    l7 = run_variant(data, "L7MH8_vol", verbose=False, **get_l7_mh8())
    trades = l7['_trades']
    if not trades:
        print("  No trades!"); return

    df = h1_df.copy()
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']

    print(f"\n  --- D1: IC Analysis: Volume Ratio vs Trade PnL ---")
    vol_pnl = []
    for t in trades:
        entry_time = pd.Timestamp(t.entry_time)
        idx = df.index.searchsorted(entry_time)
        if idx >= len(df) or idx < 1: continue
        vr = df['Vol_Ratio'].iloc[idx - 1]  # bar before entry
        if np.isnan(vr): continue
        vol_pnl.append((vr, t.pnl))

    if len(vol_pnl) > 100:
        vp = pd.DataFrame(vol_pnl, columns=['vol_ratio', 'pnl'])
        ic = vp['vol_ratio'].corr(vp['pnl'])
        ric = vp['vol_ratio'].corr(vp['pnl'], method='spearman')
        print(f"  IC(VolRatio, PnL) = {ic:+.4f}, RankIC = {ric:+.4f}, N={len(vp)}")

        print(f"\n  --- D2: Trade Performance by Volume Quintile ---")
        vp['vq'] = pd.qcut(vp['vol_ratio'], 5, labels=False, duplicates='drop')
        print(f"  {'VolQ':>8} {'N':>6} {'MeanPnL':>10} {'WR':>6} {'TotalPnL':>10}")
        for q in sorted(vp['vq'].unique()):
            sub = vp[vp['vq'] == q]
            wins = sum(1 for p in sub['pnl'] if p > 0)
            lo_v = sub['vol_ratio'].min(); hi_v = sub['vol_ratio'].max()
            print(f"  Q{int(q)}({lo_v:.1f}-{hi_v:.1f}) {len(sub):>6} ${sub['pnl'].mean():>9.2f} "
                  f"{wins/len(sub)*100:>5.1f}% ${sub['pnl'].sum():>9.0f}")

        print(f"\n  --- D3: Simulated Volume-Based Lot Adjustment ---")
        base_daily = {}; vol_daily = {}
        for t in trades:
            d = pd.Timestamp(t.exit_time).date()
            base_daily.setdefault(d, 0); base_daily[d] += t.pnl
        def _sh(daily):
            da = np.array(list(daily.values()))
            return da.mean() / da.std() * np.sqrt(252) if len(da) > 5 and da.std() > 0 else 0
        base_sh = _sh(base_daily)

        for scheme, mults in [
            ("HighVol 1.3x / LowVol 0.7x", {0: 0.7, 1: 0.85, 2: 1.0, 3: 1.15, 4: 1.3}),
            ("HighVol 1.5x / LowVol 0.5x", {0: 0.5, 1: 0.75, 2: 1.0, 3: 1.25, 4: 1.5}),
            ("HighVol Only 1.5x", {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.5}),
            ("LowVol Cut 0.5x", {0: 0.5, 1: 0.75, 2: 1.0, 3: 1.0, 4: 1.0}),
        ]:
            daily = {}
            for t in trades:
                entry_time = pd.Timestamp(t.entry_time)
                idx = df.index.searchsorted(entry_time)
                if idx >= len(df) or idx < 1: continue
                vr = df['Vol_Ratio'].iloc[idx - 1]
                if np.isnan(vr): mult = 1.0
                else:
                    vq = min(int(pd.Series([vr]).rank(pct=True).iloc[0] * 5), 4)
                    # Use global quintiles instead
                    vq_global = 2  # default middle
                    for qq in range(5):
                        if vr <= vp['vol_ratio'].quantile((qq + 1) / 5):
                            vq_global = qq; break
                    mult = mults.get(vq_global, 1.0)
                d = pd.Timestamp(t.exit_time).date()
                daily.setdefault(d, 0); daily[d] += t.pnl * mult
            sh = _sh(daily)
            print(f"  {scheme:<35} Sharpe={sh:.2f} ({sh - base_sh:+.2f})")


# ═══════════════════════════════════════════════════════════════
# Phase E: Second-Order Momentum
# ═══════════════════════════════════════════════════════════════

def run_phase_E(h1_df):
    """Test momentum acceleration as trail adjustment signal."""
    print("\n" + "=" * 80)
    print("Phase E: Second-Order Momentum (Acceleration)")
    print("=" * 80)

    df = h1_df.copy()
    df['ROC_20'] = df['Close'].pct_change(20)
    df['ROC_accel'] = df['ROC_20'] - df['ROC_20'].shift(5)
    df['ret_4'] = df['Close'].pct_change(4).shift(-4)

    valid = df[['ROC_accel', 'ret_4']].dropna()
    if len(valid) < 500:
        print("  Not enough data"); return

    ic = valid['ROC_accel'].corr(valid['ret_4'])
    ric = valid['ROC_accel'].corr(valid['ret_4'], method='spearman')
    print(f"\n  IC(MomAccel, ret_4h) = {ic:+.4f}, RankIC = {ric:+.4f}, N={len(valid)}")

    print(f"\n  --- E1: Acceleration Quintile Analysis ---")
    valid = valid.copy()
    valid['aq'] = pd.qcut(valid['ROC_accel'], 5, labels=False, duplicates='drop')
    print(f"  {'Quintile':>10} {'N':>6} {'MeanRet4':>10} {'StdRet4':>10} {'SharpeProxy':>12}")
    for q in sorted(valid['aq'].unique()):
        sub = valid[valid['aq'] == q]
        mr = sub['ret_4'].mean()
        sr = sub['ret_4'].std()
        sp = mr / sr * np.sqrt(252/4) if sr > 0 else 0
        print(f"  Q{int(q):>8} {len(sub):>6} {mr:>+10.5f} {sr:>10.5f} {sp:>12.2f}")

    # Test different ROC periods
    print(f"\n  --- E2: Acceleration IC Scan (various periods) ---")
    print(f"  {'ROC_period':>10} {'Accel_lag':>10} {'IC_ret4':>10} {'IC_ret8':>10} {'IC_ret20':>10}")
    for roc_p in [5, 10, 20, 40]:
        for lag in [3, 5, 10]:
            roc = df['Close'].pct_change(roc_p)
            accel = roc - roc.shift(lag)
            for fwd, fwd_label in [(4, 'ret4'), (8, 'ret8'), (20, 'ret20')]:
                fret = df['Close'].pct_change(fwd).shift(-fwd)
                v = pd.DataFrame({'a': accel, 'r': fret}).dropna()
                if len(v) < 500: continue
                ic_val = v['a'].corr(v['r'])
                if fwd == 4:
                    ic4 = ic_val
                elif fwd == 8:
                    ic8 = ic_val
                else:
                    ic20 = ic_val
            print(f"  ROC({roc_p:>3}) lag={lag:<3} {ic4:>+10.4f} {ic8:>+10.4f} {ic20:>+10.4f}")


# ── Shared helpers ──

def _rec(trades, pos, exit_time, reason, bar_idx, pnl):
    trades.append({'dir': pos['dir'], 'entry': pos['entry'],
                   'entry_time': pos['time'], 'exit_time': exit_time,
                   'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']})


def _stats(trades, label, return_trades=False):
    if not trades:
        r = {'label': label, 'n': 0, 'sharpe': 0, 'total_pnl': 0, 'win_rate': 0, 'max_dd': 0}
        if return_trades: r['_trades'] = []; r['_daily_pnl'] = {}
        return r
    pnls = [t['pnl'] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    eq = np.cumsum(pnls); dd = (np.maximum.accumulate(eq + 2000) - (eq + 2000)).max()
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily.setdefault(d, 0); daily[d] += t['pnl']
    da = np.array(list(daily.values()))
    sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0
    r = {'label': label, 'n': len(trades), 'sharpe': sh, 'total_pnl': sum(pnls),
         'win_rate': wins/len(trades)*100, 'max_dd': dd}
    if return_trades: r['_trades'] = trades; r['_daily_pnl'] = daily
    return r


def main():
    t0 = time.time()
    out_path = OUT_DIR / "R29_output.txt"
    out = open(out_path, 'w', encoding='utf-8', buffering=1)
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, out)

    print(f"# R29: New Factor Exploration")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()

    # Add ATR to h1_df for TSMOM
    tr = pd.DataFrame({
        'hl': h1_df['High'] - h1_df['Low'],
        'hc': (h1_df['High'] - h1_df['Close'].shift(1)).abs(),
        'lc': (h1_df['Low'] - h1_df['Close'].shift(1)).abs(),
    }).max(axis=1)
    h1_df['ATR'] = tr.rolling(14).mean()

    phases = [
        ("A", run_phase_A, (h1_df,)),
        ("B", run_phase_B, (h1_df,)),
        ("C", run_phase_C, (data, h1_df)),
        ("D", run_phase_D, (data, h1_df)),
        ("E", run_phase_E, (h1_df,)),
    ]

    for name, fn, args in phases:
        try:
            fn(*args)
            out.flush()
            print(f"\n# Phase {name} completed at {datetime.now().strftime('%H:%M:%S')}")
            out.flush()
        except Exception as e:
            print(f"\n# Phase {name} FAILED: {e}")
            import traceback; traceback.print_exc()
            out.flush()

    elapsed = time.time() - t0
    print(f"\n# Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Elapsed: {elapsed/60:.1f} minutes")

    sys.stdout = old_stdout
    out.close()
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
