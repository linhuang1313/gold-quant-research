#!/usr/bin/env python3
"""
R199 - Keltner Trail Parameter Mega Sweep (M15 + H1 Dual Resolution)
=====================================================================
MOTIVATION: R196c/R196d showed trail_act=0.02/trail_dist=0.005 is optimal
in H1 backtests, but live trading reveals "hair-trigger" exits - the H1 bar
resolution hides intra-bar noise that prematurely triggers tight trails.

This experiment uses BOTH M15 and H1 data to find parameters that:
  1. Still perform well in H1 (strategic view)
  2. Survive M15 noise (tactical reality check)

PHASES:
  1. H1 Full Grid Sweep (baseline, already known from R196c)
  2. M15 Full Grid Sweep (NEW - higher resolution reveals noise sensitivity)
  3. H1 vs M15 Divergence Analysis (find params that diverge = noise-sensitive)
  4. "Noise Robustness Score" = M15_Sharpe / H1_Sharpe ratio ranking
  5. K-Fold 6-Fold Cross-Validation (on M15 data for top candidates)
  6. Walk-Forward 19 windows (on M15 data)
  7. Era Segmented Analysis (4 eras on M15)
  8. Exit Reason Distribution (M15 vs H1 for each candidate)
  9. Monte Carlo Parameter Perturbation (+-20%, 500 trials on M15)
  10. Spread Robustness (0.20 ~ 1.00 on M15)
  11. Recent Period Focus (last 6 months M15, simulates current market)
  12. Final Verdict & Recommendation

TRAIL GRID:
  activate: 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12
  distance: 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03
  (filtered: distance < activate)
  = ~50 valid combinations
"""
import sys, os, time, json, warnings, copy
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r199_trail_mega")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
BASE_CONFIG = {
    'lot': 0.04, 'cap': 70, 'sl': 6.0, 'tp': 8.0,
    'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 2
}

TA_GRID = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12]
TD_GRID = [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03]

import glob as _glob
t0 = time.time()


def phase_done(name):
    return (OUTPUT_DIR / f"{name}.json").exists()


def save_phase(name, data):
    with open(OUTPUT_DIR / f"{name}.json", 'w') as f:
        json.dump(data, f, indent=2, default=str)


# ========================= INDICATORS =========================
def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs()
    }).max(axis=1)
    return tr.rolling(period).mean()


def compute_adx(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    pdm = h.diff(); mdm = -l.diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
    mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
    tr = pd.DataFrame({'hl': h - l, 'hc': (h - c.shift(1)).abs(), 'lc': (l - c.shift(1)).abs()}).max(axis=1)
    atr_s = tr.rolling(period).mean()
    pdi = 100 * (pdm.rolling(period).mean() / atr_s)
    mdi = 100 * (mdm.rolling(period).mean() / atr_s)
    dx = 100 * ((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan))
    return dx.rolling(period).mean()


def compute_atr_pctl(atr_series, lb=300):
    n = len(atr_series); p = np.full(n, np.nan); v = atr_series.values
    for i in range(lb, n):
        w = v[i - lb:i]; valid = w[~np.isnan(w)]
        if len(valid) >= 30:
            p[i] = np.sum(valid <= v[i]) / len(valid) * 100
    return pd.Series(p, index=atr_series.index)


# ========================= BACKTEST ENGINE =========================
def _mk(pos, ep, et, reason, bi, pnl):
    return {
        'dir': pos['dir'], 'entry': pos['entry'], 'exit': ep,
        'entry_time': pos['time'], 'exit_time': et,
        'pnl': pnl, 'reason': reason, 'bars': bi - pos['bar'],
        'atr': pos['atr']
    }


def _run_exit_m15(pos, i, h, lo, c, spread, lot, pv, times,
                   sl_atr, tp_atr, ta, td, mh_bars, cap):
    """
    M15-resolution exit: checks TP/SL/Cap first, then trail within bar.
    For M15, max_hold is in M15 bars (mh * 4 for H1 equivalent).
    Trail logic: within each M15 bar, price can trigger trail.
    """
    if pos['dir'] == 'BUY':
        pnl_c = (c - pos['entry'] - spread) * lot * pv
        pnl_h = (h - pos['entry'] - spread) * lot * pv
        pnl_l = (lo - pos['entry'] - spread) * lot * pv
    else:
        pnl_c = (pos['entry'] - c - spread) * lot * pv
        pnl_h = (pos['entry'] - lo - spread) * lot * pv
        pnl_l = (pos['entry'] - h - spread) * lot * pv

    tp_v = tp_atr * pos['atr'] * lot * pv
    sl_v = sl_atr * pos['atr'] * lot * pv

    if pnl_h >= tp_v:
        return _mk(pos, c, times[i], "TP", i, tp_v)
    if pnl_l <= -sl_v:
        return _mk(pos, c, times[i], "SL", i, -sl_v)
    if cap > 0 and pnl_c < -cap:
        return _mk(pos, c, times[i], "Cap", i, -cap)

    ad = ta * pos['atr']; tdd = td * pos['atr']
    if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
        ts = h - tdd
        if lo <= ts:
            trail_pnl = (ts - pos['entry'] - spread) * lot * pv
            return _mk(pos, c, times[i], "Trail", i, trail_pnl)
    elif pos['dir'] == 'SELL' and pos['entry'] - lo >= ad:
        ts = lo + tdd
        if h >= ts:
            trail_pnl = (pos['entry'] - ts - spread) * lot * pv
            return _mk(pos, c, times[i], "Trail", i, trail_pnl)

    held = i - pos['bar']
    if held >= mh_bars:
        return _mk(pos, c, times[i], "Timeout", i, pnl_c)
    return None


def bt_keltner_h1(h1, cfg, pctl_v, pctl_f=30, spread=0.30, skip_hours=None):
    """H1 backtest (same as R196d)."""
    df = h1.copy()
    df['ATR'] = compute_atr(df)
    df['ADX'] = compute_adx(df)
    df['EMA_T'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['KC_mid'] = df['Close'].ewm(span=25, adjust=False).mean()
    df['KC_upper'] = df['KC_mid'] + 1.2 * df['ATR']
    df['KC_lower'] = df['KC_mid'] - 1.2 * df['ATR']
    df = df.dropna(subset=['ATR', 'ADX', 'EMA_T', 'KC_upper'])

    pv_a = pctl_v.reindex(df.index).values if pctl_v is not None else None
    c, h, lo = df['Close'].values, df['High'].values, df['Low'].values
    atr, adx, ema = df['ATR'].values, df['ADX'].values, df['EMA_T'].values
    ku, kl = df['KC_upper'].values, df['KC_lower'].values
    times = df.index; n = len(df); hrs = df.index.hour

    lot = cfg['lot']; sl = cfg['sl']; tp = cfg['tp']
    ta = cfg['trail_act']; td = cfg['trail_dist']
    mh = cfg['max_hold']; cap = cfg['cap']

    trades = []; pos = None; le = -999
    for i in range(1, n):
        if pos:
            r = _run_exit_m15(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                              sl, tp, ta, td, mh, cap)
            if r:
                trades.append(r); pos = None; le = i; continue
            continue
        if i - le < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pv_a is not None and (np.isnan(pv_a[i]) or pv_a[i] < pctl_f): continue
        if np.isnan(adx[i]) or adx[i] < 14: continue
        if skip_hours and hrs[i] in skip_hours: continue
        if c[i] > ku[i] and c[i] > ema[i]:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < kl[i] and c[i] < ema[i]:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_keltner_m15(h1, m15, cfg, pctl_v, pctl_f=30, spread=0.30, skip_hours=None):
    """
    Hybrid: H1 entry signals (same as H1 backtest), M15 exit resolution.
    Entry: on H1 bar close (KC breakout + ADX + EMA + pctl filter).
    Exit: simulate M15-by-M15 within max_hold H1 bars.
    """
    df = h1.copy()
    df['ATR'] = compute_atr(df)
    df['ADX'] = compute_adx(df)
    df['EMA_T'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['KC_mid'] = df['Close'].ewm(span=25, adjust=False).mean()
    df['KC_upper'] = df['KC_mid'] + 1.2 * df['ATR']
    df['KC_lower'] = df['KC_mid'] - 1.2 * df['ATR']
    df = df.dropna(subset=['ATR', 'ADX', 'EMA_T', 'KC_upper'])

    pv_a = pctl_v.reindex(df.index).values if pctl_v is not None else None
    c_h1, h_h1, lo_h1 = df['Close'].values, df['High'].values, df['Low'].values
    atr_arr, adx_arr, ema_arr = df['ATR'].values, df['ADX'].values, df['EMA_T'].values
    ku_arr, kl_arr = df['KC_upper'].values, df['KC_lower'].values
    h1_times = df.index; n_h1 = len(df); hrs = df.index.hour

    lot = cfg['lot']; sl = cfg['sl']; tp = cfg['tp']
    ta = cfg['trail_act']; td = cfg['trail_dist']
    mh_h1 = cfg['max_hold']; cap = cfg['cap']
    mh_m15 = mh_h1 * 4

    m15_c = m15['Close'].values; m15_h = m15['High'].values; m15_l = m15['Low'].values
    m15_times = m15.index; n_m15 = len(m15)

    trades = []; le = -999

    for i in range(1, n_h1):
        if i - le < 2: continue
        if np.isnan(atr_arr[i]) or atr_arr[i] < 0.1: continue
        if pv_a is not None and (np.isnan(pv_a[i]) or pv_a[i] < pctl_f): continue
        if np.isnan(adx_arr[i]) or adx_arr[i] < 14: continue
        if skip_hours and hrs[i] in skip_hours: continue

        entry_dir = None
        if c_h1[i] > ku_arr[i] and c_h1[i] > ema_arr[i]:
            entry_dir = 'BUY'
        elif c_h1[i] < kl_arr[i] and c_h1[i] < ema_arr[i]:
            entry_dir = 'SELL'

        if entry_dir is None:
            continue

        entry_price = c_h1[i] + spread / 2 if entry_dir == 'BUY' else c_h1[i] - spread / 2
        entry_atr = atr_arr[i]
        entry_time = h1_times[i]

        m15_start_idx = m15_times.searchsorted(entry_time)
        if m15_start_idx >= n_m15 - 1:
            continue

        pos = {'dir': entry_dir, 'entry': entry_price, 'bar': m15_start_idx,
               'time': entry_time, 'atr': entry_atr}
        closed = False

        for j in range(m15_start_idx + 1, min(m15_start_idx + mh_m15 + 1, n_m15)):
            r = _run_exit_m15(pos, j, m15_h[j], m15_l[j], m15_c[j],
                              spread, lot, PV, m15_times, sl, tp, ta, td,
                              mh_m15, cap)
            if r:
                trades.append(r); le = i; closed = True; break

        if not closed:
            end_j = min(m15_start_idx + mh_m15, n_m15 - 1)
            pnl_close = (m15_c[end_j] - entry_price - spread) * lot * PV if entry_dir == 'BUY' \
                else (entry_price - m15_c[end_j] - spread) * lot * PV
            trades.append({
                'dir': entry_dir, 'entry': entry_price, 'exit': m15_c[end_j],
                'entry_time': entry_time, 'exit_time': m15_times[end_j],
                'pnl': pnl_close, 'reason': 'Timeout',
                'bars': end_j - m15_start_idx, 'atr': entry_atr
            })
            le = i

    return trades


# ========================= STATS HELPERS =========================
def _daily(trades):
    if not trades: return pd.Series(dtype=float)
    d = {}
    for t in trades:
        k = pd.Timestamp(t['exit_time']).normalize()
        d[k] = d.get(k, 0) + t['pnl']
    return pd.Series(d).sort_index()


def _sharpe(daily):
    if len(daily) < 10 or daily.std() == 0: return 0.0
    return float(daily.mean() / daily.std() * np.sqrt(252))


def _stats(trades):
    if not trades: return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0, 'avg_pnl': 0}
    daily = _daily(trades)
    pnls = [t['pnl'] for t in trades]; n = len(trades)
    wins = [p for p in pnls if p > 0]
    eq = daily.cumsum()
    dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 1 else 0
    return {
        'n': n, 'sharpe': round(_sharpe(daily), 3),
        'pnl': round(sum(pnls), 2), 'wr': round(len(wins) / n * 100, 1),
        'max_dd': round(dd, 2), 'avg_pnl': round(sum(pnls) / n, 2)
    }


def _exit_dist(trades):
    if not trades: return {}
    reasons = Counter(t['reason'] for t in trades)
    total = len(trades)
    result = {}
    for r in ['TP', 'SL', 'Cap', 'Trail', 'Timeout']:
        cnt = reasons.get(r, 0)
        avg = np.mean([t['pnl'] for t in trades if t['reason'] == r]) if cnt > 0 else 0
        result[r] = {'n': cnt, 'pct': round(cnt / total * 100, 1), 'avg_pnl': round(avg, 2)}
    return result


# ========================= DATA LOADING =========================
def load_h1():
    candidates = sorted(_glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    if not candidates: raise FileNotFoundError("No H1 data")
    df = pd.read_csv(candidates[-1])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    print(f"  H1: {len(df)} bars ({df.index[0]} ~ {df.index[-1]})", flush=True)
    return df


def load_m15():
    candidates = sorted(_glob.glob("data/download/xauusd-m15-bid-2015-*.csv"))
    if not candidates: raise FileNotFoundError("No M15 data")
    df = pd.read_csv(candidates[-1])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    print(f"  M15: {len(df)} bars ({df.index[0]} ~ {df.index[-1]})", flush=True)
    return df


# ========================= PHASE 1: H1 Full Grid =========================
def phase_1_h1_grid(h1, pctl):
    if phase_done("phase_1_h1_grid"):
        print("  Phase 1 cached", flush=True); return
    print(f"\n{'='*80}\n  PHASE 1: H1 FULL TRAIL GRID SWEEP\n{'='*80}", flush=True)

    results = {}
    total = sum(1 for ta in TA_GRID for td in TD_GRID if td < ta)
    done = 0

    for ta in TA_GRID:
        for td in TD_GRID:
            if td >= ta: continue
            cfg = copy.deepcopy(BASE_CONFIG)
            cfg['trail_act'] = ta; cfg['trail_dist'] = td
            trades = bt_keltner_h1(h1, cfg, pctl)
            s = _stats(trades)
            s['exit_dist'] = _exit_dist(trades)
            key = f"ta{ta}_td{td}"
            results[key] = s
            done += 1
            if done % 10 == 0:
                print(f"    H1 grid: {done}/{total} done ({time.time()-t0:.0f}s)", flush=True)

    best = max(results.items(), key=lambda x: x[1]['sharpe'])
    print(f"\n  H1 Grid complete: {len(results)} configs tested", flush=True)
    print(f"  H1 Best: {best[0]} -> Sharpe={best[1]['sharpe']:.3f}, PnL=${best[1]['pnl']:.0f}", flush=True)
    print(f"  H1 Current (0.06/0.01): Sharpe={results.get('ta0.06_td0.01',{}).get('sharpe',0):.3f}", flush=True)

    save_phase("phase_1_h1_grid", results)
    print(f"  Phase 1 done ({(time.time()-t0)/60:.1f}m)", flush=True)


# ========================= PHASE 2: M15 Full Grid =========================
def phase_2_m15_grid(h1, m15, pctl):
    if phase_done("phase_2_m15_grid"):
        print("  Phase 2 cached", flush=True); return
    print(f"\n{'='*80}\n  PHASE 2: M15 FULL TRAIL GRID SWEEP (HIGH RESOLUTION)\n{'='*80}", flush=True)

    results = {}
    total = sum(1 for ta in TA_GRID for td in TD_GRID if td < ta)
    done = 0

    for ta in TA_GRID:
        for td in TD_GRID:
            if td >= ta: continue
            cfg = copy.deepcopy(BASE_CONFIG)
            cfg['trail_act'] = ta; cfg['trail_dist'] = td
            trades = bt_keltner_m15(h1, m15, cfg, pctl)
            s = _stats(trades)
            s['exit_dist'] = _exit_dist(trades)
            key = f"ta{ta}_td{td}"
            results[key] = s
            done += 1
            if done % 5 == 0:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) / 60
                print(f"    M15 grid: {done}/{total} done ({elapsed:.0f}s, ETA {eta:.1f}m)", flush=True)

    best = max(results.items(), key=lambda x: x[1]['sharpe'])
    print(f"\n  M15 Grid complete: {len(results)} configs tested", flush=True)
    print(f"  M15 Best: {best[0]} -> Sharpe={best[1]['sharpe']:.3f}, PnL=${best[1]['pnl']:.0f}", flush=True)
    print(f"  M15 Current (0.06/0.01): Sharpe={results.get('ta0.06_td0.01',{}).get('sharpe',0):.3f}", flush=True)

    save_phase("phase_2_m15_grid", results)
    print(f"  Phase 2 done ({(time.time()-t0)/60:.1f}m)", flush=True)


# ========================= PHASE 3: Divergence Analysis =========================
def phase_3_divergence():
    if phase_done("phase_3_divergence"):
        print("  Phase 3 cached", flush=True); return
    print(f"\n{'='*80}\n  PHASE 3: H1 vs M15 DIVERGENCE ANALYSIS\n{'='*80}", flush=True)

    with open(OUTPUT_DIR / "phase_1_h1_grid.json") as f: h1_data = json.load(f)
    with open(OUTPUT_DIR / "phase_2_m15_grid.json") as f: m15_data = json.load(f)

    results = {}
    print(f"  {'Config':<20} {'H1_Sh':>7} {'M15_Sh':>7} {'Ratio':>7} {'H1_PnL':>10} {'M15_PnL':>10} {'H1_WR':>7} {'M15_WR':>7}", flush=True)
    print(f"  {'-'*80}", flush=True)

    for key in sorted(h1_data.keys()):
        if key not in m15_data: continue
        h1_s = h1_data[key]; m15_s = m15_data[key]
        ratio = m15_s['sharpe'] / h1_s['sharpe'] if h1_s['sharpe'] > 0 else 0
        results[key] = {
            'h1_sharpe': h1_s['sharpe'], 'm15_sharpe': m15_s['sharpe'],
            'ratio': round(ratio, 3),
            'h1_pnl': h1_s['pnl'], 'm15_pnl': m15_s['pnl'],
            'h1_wr': h1_s['wr'], 'm15_wr': m15_s['wr'],
            'h1_avg_pnl': h1_s.get('avg_pnl', 0), 'm15_avg_pnl': m15_s.get('avg_pnl', 0),
        }
        print(f"  {key:<20} {h1_s['sharpe']:>7.3f} {m15_s['sharpe']:>7.3f} {ratio:>7.3f} "
              f"{h1_s['pnl']:>10.0f} {m15_s['pnl']:>10.0f} {h1_s['wr']:>6.1f}% {m15_s['wr']:>6.1f}%", flush=True)

    sorted_by_ratio = sorted(results.items(), key=lambda x: x[1]['ratio'], reverse=True)
    print(f"\n  === TOP 10 by Noise Robustness (M15/H1 Sharpe Ratio) ===", flush=True)
    for i, (k, v) in enumerate(sorted_by_ratio[:10]):
        print(f"  #{i+1} {k}: ratio={v['ratio']:.3f}, M15_Sh={v['m15_sharpe']:.3f}, H1_Sh={v['h1_sharpe']:.3f}", flush=True)

    sorted_by_m15 = sorted(results.items(), key=lambda x: x[1]['m15_sharpe'], reverse=True)
    print(f"\n  === TOP 10 by M15 Sharpe (absolute performance) ===", flush=True)
    for i, (k, v) in enumerate(sorted_by_m15[:10]):
        print(f"  #{i+1} {k}: M15_Sh={v['m15_sharpe']:.3f}, H1_Sh={v['h1_sharpe']:.3f}, ratio={v['ratio']:.3f}", flush=True)

    # Most noise-sensitive (biggest drop from H1 to M15)
    sorted_by_drop = sorted(results.items(), key=lambda x: x[1]['h1_sharpe'] - x[1]['m15_sharpe'], reverse=True)
    print(f"\n  === MOST NOISE-SENSITIVE (biggest H1->M15 drop) ===", flush=True)
    for i, (k, v) in enumerate(sorted_by_drop[:5]):
        drop = v['h1_sharpe'] - v['m15_sharpe']
        print(f"  #{i+1} {k}: drop={drop:.3f}, H1={v['h1_sharpe']:.3f}, M15={v['m15_sharpe']:.3f}", flush=True)

    save_phase("phase_3_divergence", results)
    print(f"  Phase 3 done ({(time.time()-t0)/60:.1f}m)", flush=True)


# ========================= PHASE 4: Select Top Candidates =========================
def phase_4_select_candidates():
    if phase_done("phase_4_candidates"):
        print("  Phase 4 cached", flush=True); return
    print(f"\n{'='*80}\n  PHASE 4: SELECT TOP CANDIDATES FOR DEEP VALIDATION\n{'='*80}", flush=True)

    with open(OUTPUT_DIR / "phase_3_divergence.json") as f: div_data = json.load(f)

    for k, v in div_data.items():
        v['composite'] = round(v['m15_sharpe'] * 0.6 + v['h1_sharpe'] * 0.2 + v['ratio'] * v['m15_sharpe'] * 0.2, 3)

    sorted_candidates = sorted(div_data.items(), key=lambda x: x[1]['composite'], reverse=True)

    candidates = []
    always_include = ['ta0.06_td0.01', 'ta0.02_td0.005']
    for key in always_include:
        if key in div_data:
            candidates.append(key)

    for k, v in sorted_candidates:
        if k not in candidates:
            candidates.append(k)
        if len(candidates) >= 8: break

    print(f"  Selected {len(candidates)} candidates for deep validation:", flush=True)
    for i, k in enumerate(candidates):
        v = div_data[k]
        label = ""
        if k == 'ta0.06_td0.01': label = " [OLD BASELINE]"
        elif k == 'ta0.02_td0.005': label = " [CURRENT LIVE]"
        print(f"  #{i+1} {k}{label}: M15={v['m15_sharpe']:.3f}, H1={v['h1_sharpe']:.3f}, "
              f"ratio={v['ratio']:.3f}, composite={v['composite']:.3f}", flush=True)

    save_phase("phase_4_candidates", {'candidates': candidates, 'details': {k: div_data[k] for k in candidates}})
    print(f"  Phase 4 done ({(time.time()-t0)/60:.1f}m)", flush=True)
    return candidates


# ========================= PHASE 5: K-Fold on M15 =========================
def phase_5_kfold_m15(h1, m15, pctl):
    if phase_done("phase_5_kfold_m15"):
        print("  Phase 5 cached", flush=True); return
    print(f"\n{'='*80}\n  PHASE 5: 6-FOLD CROSS-VALIDATION ON M15\n{'='*80}", flush=True)

    with open(OUTPUT_DIR / "phase_4_candidates.json") as f: cand_data = json.load(f)
    candidates = cand_data['candidates']

    n = len(h1)
    fold_size = n // 6
    results = {}

    for cand in candidates:
        parts = cand.replace('ta', '').replace('td', '').split('_')
        ta_val = float(parts[0]); td_val = float(parts[1])

        cfg_cand = copy.deepcopy(BASE_CONFIG)
        cfg_cand['trail_act'] = ta_val; cfg_cand['trail_dist'] = td_val

        cfg_base = copy.deepcopy(BASE_CONFIG)

        wins = 0
        fold_results = []
        for fold in range(6):
            test_start = fold * fold_size
            test_end = min(test_start + fold_size, n)
            h1_test = h1.iloc[test_start:test_end]
            if len(h1_test) < 200: continue

            m15_start = m15.index.searchsorted(h1_test.index[0])
            m15_end = m15.index.searchsorted(h1_test.index[-1])
            m15_test = m15.iloc[m15_start:m15_end + 1]
            if len(m15_test) < 800: continue

            pctl_fold = compute_atr_pctl(compute_atr(h1_test), lb=min(300, len(h1_test) // 3))

            s_base = _stats(bt_keltner_m15(h1_test, m15_test, cfg_base, pctl_fold))
            s_cand = _stats(bt_keltner_m15(h1_test, m15_test, cfg_cand, pctl_fold))

            won = s_cand['sharpe'] >= s_base['sharpe']
            if won: wins += 1
            fold_results.append({
                'fold': fold, 'base_sharpe': s_base['sharpe'], 'cand_sharpe': s_cand['sharpe'],
                'won': won
            })

        total_folds = len(fold_results)
        passed = wins >= 4
        results[cand] = {
            'wins': wins, 'total': total_folds, 'passed': passed,
            'folds': fold_results
        }
        status = "PASS" if passed else "FAIL"
        print(f"  {cand}: K-Fold {wins}/{total_folds} -> {status}", flush=True)

    save_phase("phase_5_kfold_m15", results)
    print(f"  Phase 5 done ({(time.time()-t0)/60:.1f}m)", flush=True)


# ========================= PHASE 6: Walk-Forward on M15 =========================
def phase_6_walkforward_m15(h1, m15, pctl):
    if phase_done("phase_6_wf_m15"):
        print("  Phase 6 cached", flush=True); return
    print(f"\n{'='*80}\n  PHASE 6: WALK-FORWARD 19 WINDOWS ON M15\n{'='*80}", flush=True)

    with open(OUTPUT_DIR / "phase_4_candidates.json") as f: cand_data = json.load(f)
    candidates = cand_data['candidates']

    n = len(h1)
    n_windows = 19
    oos_size = int(n * 0.4 / n_windows)
    oos_start_base = int(n * 0.6)

    results = {}
    for cand in candidates:
        parts = cand.replace('ta', '').replace('td', '').split('_')
        ta_val = float(parts[0]); td_val = float(parts[1])

        cfg_cand = copy.deepcopy(BASE_CONFIG)
        cfg_cand['trail_act'] = ta_val; cfg_cand['trail_dist'] = td_val
        cfg_base = copy.deepcopy(BASE_CONFIG)

        wins = 0; total = 0
        for w in range(n_windows):
            start_idx = oos_start_base + w * oos_size
            end_idx = min(start_idx + oos_size, n)
            if end_idx <= start_idx: continue

            h1_w = h1.iloc[start_idx:end_idx]
            if len(h1_w) < 200: continue

            m15_s = m15.index.searchsorted(h1_w.index[0])
            m15_e = m15.index.searchsorted(h1_w.index[-1])
            m15_w = m15.iloc[m15_s:m15_e + 1]
            if len(m15_w) < 800: continue

            pctl_w = compute_atr_pctl(compute_atr(h1_w), lb=min(300, len(h1_w) // 3))
            sb = _stats(bt_keltner_m15(h1_w, m15_w, cfg_base, pctl_w))['sharpe']
            sc = _stats(bt_keltner_m15(h1_w, m15_w, cfg_cand, pctl_w))['sharpe']
            if sc >= sb: wins += 1
            total += 1

        passed = wins >= 10
        results[cand] = {'wins': wins, 'total': total, 'passed': passed}
        status = "PASS" if passed else "FAIL"
        print(f"  {cand}: WF {wins}/{total} -> {status}", flush=True)

    save_phase("phase_6_wf_m15", results)
    print(f"  Phase 6 done ({(time.time()-t0)/60:.1f}m)", flush=True)


# ========================= PHASE 7: Era Segmented =========================
def phase_7_era_m15(h1, m15, pctl):
    if phase_done("phase_7_era_m15"):
        print("  Phase 7 cached", flush=True); return
    print(f"\n{'='*80}\n  PHASE 7: ERA SEGMENTED ANALYSIS ON M15\n{'='*80}", flush=True)

    with open(OUTPUT_DIR / "phase_4_candidates.json") as f: cand_data = json.load(f)
    candidates = cand_data['candidates']

    eras = [
        ("2015-2017 (Low Vol)", "2015-01-01", "2017-12-31"),
        ("2018-2019 (Range)", "2018-01-01", "2019-12-31"),
        ("2020-2022 (COVID+Post)", "2020-01-01", "2022-12-31"),
        ("2023-2026 (Recent)", "2023-01-01", "2026-12-31"),
    ]

    results = {}
    for cand in candidates:
        parts = cand.replace('ta', '').replace('td', '').split('_')
        ta_val = float(parts[0]); td_val = float(parts[1])

        cfg_cand = copy.deepcopy(BASE_CONFIG)
        cfg_cand['trail_act'] = ta_val; cfg_cand['trail_dist'] = td_val

        cand_results = {}
        all_positive = True
        print(f"\n  {cand}:", flush=True)

        for era_name, start, end in eras:
            h1_era = h1[start:end]
            if len(h1_era) < 500:
                print(f"    {era_name}: SKIP (too few bars)", flush=True)
                continue

            m15_s = m15.index.searchsorted(h1_era.index[0])
            m15_e = m15.index.searchsorted(h1_era.index[-1])
            m15_era = m15.iloc[m15_s:m15_e + 1]

            pctl_era = compute_atr_pctl(compute_atr(h1_era), lb=min(300, len(h1_era) // 3))
            s = _stats(bt_keltner_m15(h1_era, m15_era, cfg_cand, pctl_era))
            cand_results[era_name] = s
            if s['sharpe'] <= 0: all_positive = False
            print(f"    {era_name}: Sharpe={s['sharpe']:.3f}, PnL=${s['pnl']:.0f}, WR={s['wr']:.1f}%", flush=True)

        cand_results['all_positive'] = all_positive
        results[cand] = cand_results
        print(f"    All eras positive: {'YES' if all_positive else 'NO'}", flush=True)

    save_phase("phase_7_era_m15", results)
    print(f"  Phase 7 done ({(time.time()-t0)/60:.1f}m)", flush=True)


# ========================= PHASE 8: Exit Distribution Comparison =========================
def phase_8_exit_compare(h1, m15, pctl):
    if phase_done("phase_8_exit_compare"):
        print("  Phase 8 cached", flush=True); return
    print(f"\n{'='*80}\n  PHASE 8: EXIT REASON DISTRIBUTION (H1 vs M15)\n{'='*80}", flush=True)

    with open(OUTPUT_DIR / "phase_4_candidates.json") as f: cand_data = json.load(f)
    candidates = cand_data['candidates']

    results = {}
    for cand in candidates:
        parts = cand.replace('ta', '').replace('td', '').split('_')
        ta_val = float(parts[0]); td_val = float(parts[1])

        cfg = copy.deepcopy(BASE_CONFIG)
        cfg['trail_act'] = ta_val; cfg['trail_dist'] = td_val

        t_h1 = bt_keltner_h1(h1, cfg, pctl)
        t_m15 = bt_keltner_m15(h1, m15, cfg, pctl)

        h1_dist = _exit_dist(t_h1)
        m15_dist = _exit_dist(t_m15)

        h1_avg = round(np.mean([t['pnl'] for t in t_h1]), 2) if t_h1 else 0
        m15_avg = round(np.mean([t['pnl'] for t in t_m15]), 2) if t_m15 else 0

        results[cand] = {
            'h1': {'exit_dist': h1_dist, 'n': len(t_h1), 'avg_pnl': h1_avg},
            'm15': {'exit_dist': m15_dist, 'n': len(t_m15), 'avg_pnl': m15_avg}
        }

        trail_h1_pct = h1_dist.get('Trail', {}).get('pct', 0)
        trail_m15_pct = m15_dist.get('Trail', {}).get('pct', 0)
        trail_h1_avg = h1_dist.get('Trail', {}).get('avg_pnl', 0)
        trail_m15_avg = m15_dist.get('Trail', {}).get('avg_pnl', 0)

        print(f"  {cand}:", flush=True)
        print(f"    H1:  N={len(t_h1)}, avg=${h1_avg:.2f}, Trail={trail_h1_pct:.1f}% (avg ${trail_h1_avg:.2f})", flush=True)
        print(f"    M15: N={len(t_m15)}, avg=${m15_avg:.2f}, Trail={trail_m15_pct:.1f}% (avg ${trail_m15_avg:.2f})", flush=True)

    save_phase("phase_8_exit_compare", results)
    print(f"  Phase 8 done ({(time.time()-t0)/60:.1f}m)", flush=True)


# ========================= PHASE 9: Monte Carlo on M15 =========================
def phase_9_monte_carlo_m15(h1, m15, pctl):
    if phase_done("phase_9_mc_m15"):
        print("  Phase 9 cached", flush=True); return
    print(f"\n{'='*80}\n  PHASE 9: MONTE CARLO PARAMETER PERTURBATION ON M15 (500 trials)\n{'='*80}", flush=True)

    with open(OUTPUT_DIR / "phase_4_candidates.json") as f: cand_data = json.load(f)
    candidates = cand_data['candidates']

    cfg_base = copy.deepcopy(BASE_CONFIG)
    base_trades = bt_keltner_m15(h1, m15, cfg_base, pctl)
    base_sh = _stats(base_trades)['sharpe']
    print(f"  Baseline M15 Sharpe: {base_sh:.3f}", flush=True)

    np.random.seed(42)
    results = {}

    for cand in candidates:
        parts = cand.replace('ta', '').replace('td', '').split('_')
        ta_center = float(parts[0]); td_center = float(parts[1])

        sharpes = []
        for trial in range(500):
            ta_p = ta_center * (1 + np.random.uniform(-0.25, 0.25))
            td_p = td_center * (1 + np.random.uniform(-0.25, 0.25))
            if td_p >= ta_p: td_p = ta_p * 0.7

            cfg = copy.deepcopy(BASE_CONFIG)
            cfg['trail_act'] = ta_p; cfg['trail_dist'] = td_p
            t = bt_keltner_m15(h1, m15, cfg, pctl)
            sharpes.append(_stats(t)['sharpe'])

        sharpes = np.array(sharpes)
        pct_better = float(np.sum(sharpes > base_sh) / 500 * 100)

        results[cand] = {
            'mean': round(float(sharpes.mean()), 3),
            'std': round(float(sharpes.std()), 3),
            'min': round(float(sharpes.min()), 3),
            'p5': round(float(np.percentile(sharpes, 5)), 3),
            'p50': round(float(np.median(sharpes)), 3),
            'p95': round(float(np.percentile(sharpes, 95)), 3),
            'max': round(float(sharpes.max()), 3),
            'pct_better_than_base': round(pct_better, 1)
        }
        print(f"  {cand}: MC mean={sharpes.mean():.3f}, std={sharpes.std():.3f}, "
              f"p5={np.percentile(sharpes,5):.3f}, >base={pct_better:.0f}%", flush=True)

    save_phase("phase_9_mc_m15", results)
    print(f"  Phase 9 done ({(time.time()-t0)/60:.1f}m)", flush=True)


# ========================= PHASE 10: Spread Robustness on M15 =========================
def phase_10_spread_m15(h1, m15, pctl):
    if phase_done("phase_10_spread_m15"):
        print("  Phase 10 cached", flush=True); return
    print(f"\n{'='*80}\n  PHASE 10: SPREAD ROBUSTNESS ON M15\n{'='*80}", flush=True)

    with open(OUTPUT_DIR / "phase_4_candidates.json") as f: cand_data = json.load(f)
    candidates = cand_data['candidates']

    spreads = [0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
    results = {}

    for cand in candidates:
        parts = cand.replace('ta', '').replace('td', '').split('_')
        ta_val = float(parts[0]); td_val = float(parts[1])

        cand_results = {}
        print(f"\n  {cand}:", flush=True)
        for sp in spreads:
            cfg = copy.deepcopy(BASE_CONFIG)
            cfg['trail_act'] = ta_val; cfg['trail_dist'] = td_val
            t = bt_keltner_m15(h1, m15, cfg, pctl, spread=sp)
            s = _stats(t)
            cand_results[f"sp_{sp}"] = s
            print(f"    spread={sp:.2f}: Sharpe={s['sharpe']:.3f}, PnL=${s['pnl']:.0f}", flush=True)

        results[cand] = cand_results

    save_phase("phase_10_spread_m15", results)
    print(f"  Phase 10 done ({(time.time()-t0)/60:.1f}m)", flush=True)


# ========================= PHASE 11: Recent Period Focus =========================
def phase_11_recent_m15(h1, m15, pctl):
    if phase_done("phase_11_recent_m15"):
        print("  Phase 11 cached", flush=True); return
    print(f"\n{'='*80}\n  PHASE 11: RECENT 6 MONTHS FOCUS (M15)\n{'='*80}", flush=True)

    with open(OUTPUT_DIR / "phase_4_candidates.json") as f: cand_data = json.load(f)
    candidates = cand_data['candidates']

    cutoff = h1.index[-1] - pd.Timedelta(days=180)
    h1_recent = h1[h1.index >= cutoff]
    m15_recent = m15[m15.index >= cutoff]
    pctl_recent = compute_atr_pctl(compute_atr(h1_recent), lb=min(300, len(h1_recent) // 3))

    print(f"  Recent period: {h1_recent.index[0]} ~ {h1_recent.index[-1]}", flush=True)
    print(f"  H1 bars: {len(h1_recent)}, M15 bars: {len(m15_recent)}", flush=True)

    results = {}
    print(f"\n  {'Config':<20} {'M15_Sh':>7} {'PnL':>10} {'WR':>7} {'N':>6} {'AvgPnL':>8} {'MaxDD':>8} {'Trail%':>8}", flush=True)
    print(f"  {'-'*80}", flush=True)

    for cand in candidates:
        parts = cand.replace('ta', '').replace('td', '').split('_')
        ta_val = float(parts[0]); td_val = float(parts[1])

        cfg = copy.deepcopy(BASE_CONFIG)
        cfg['trail_act'] = ta_val; cfg['trail_dist'] = td_val
        trades = bt_keltner_m15(h1_recent, m15_recent, cfg, pctl_recent)
        s = _stats(trades)
        ed = _exit_dist(trades)
        trail_pct = ed.get('Trail', {}).get('pct', 0)

        results[cand] = {**s, 'exit_dist': ed}
        print(f"  {cand:<20} {s['sharpe']:>7.3f} {s['pnl']:>10.0f} {s['wr']:>6.1f}% {s['n']:>6} "
              f"{s['avg_pnl']:>8.2f} {s['max_dd']:>8.0f} {trail_pct:>7.1f}%", flush=True)

    save_phase("phase_11_recent_m15", results)
    print(f"  Phase 11 done ({(time.time()-t0)/60:.1f}m)", flush=True)


# ========================= PHASE 12: Final Verdict =========================
def phase_12_verdict():
    print(f"\n{'='*80}\n  PHASE 12: FINAL VERDICT & RECOMMENDATION\n{'='*80}", flush=True)

    with open(OUTPUT_DIR / "phase_4_candidates.json") as f: cand_data = json.load(f)
    candidates = cand_data['candidates']

    # Load all phase results
    phases = {}
    for p in ['phase_3_divergence', 'phase_5_kfold_m15', 'phase_6_wf_m15',
              'phase_7_era_m15', 'phase_9_mc_m15', 'phase_11_recent_m15']:
        try:
            with open(OUTPUT_DIR / f"{p}.json") as f: phases[p] = json.load(f)
        except: pass

    scorecard = {}
    print(f"\n  {'Config':<20} {'M15/H1':>7} {'KFold':>8} {'WF':>8} {'Era':>5} {'MC>base':>8} {'Recent':>8} {'TOTAL':>7}", flush=True)
    print(f"  {'-'*80}", flush=True)

    for cand in candidates:
        score = 0; details = {}

        # Noise robustness ratio
        div = phases.get('phase_3_divergence', {}).get(cand, {})
        ratio = div.get('ratio', 0)
        details['ratio'] = ratio

        # K-Fold
        kf = phases.get('phase_5_kfold_m15', {}).get(cand, {})
        kf_pass = kf.get('passed', False)
        kf_wins = kf.get('wins', 0)
        kf_total = kf.get('total', 0)
        if kf_pass: score += 2
        details['kfold'] = f"{kf_wins}/{kf_total}"

        # Walk-Forward
        wf = phases.get('phase_6_wf_m15', {}).get(cand, {})
        wf_pass = wf.get('passed', False)
        wf_wins = wf.get('wins', 0)
        wf_total = wf.get('total', 0)
        if wf_pass: score += 2
        details['wf'] = f"{wf_wins}/{wf_total}"

        # Era
        era = phases.get('phase_7_era_m15', {}).get(cand, {})
        era_ok = era.get('all_positive', False)
        if era_ok: score += 2
        details['era'] = "OK" if era_ok else "NO"

        # Monte Carlo
        mc = phases.get('phase_9_mc_m15', {}).get(cand, {})
        mc_pct = mc.get('pct_better_than_base', 0)
        if mc_pct >= 60: score += 1
        if mc_pct >= 80: score += 1
        details['mc'] = f"{mc_pct:.0f}%"

        # Recent performance
        rec = phases.get('phase_11_recent_m15', {}).get(cand, {})
        rec_sh = rec.get('sharpe', 0)
        if rec_sh > 0: score += 1
        details['recent_sh'] = rec_sh

        scorecard[cand] = {'score': score, 'max_score': 10, **details}

        label = ""
        if cand == 'ta0.06_td0.01': label = " [OLD]"
        elif cand == 'ta0.02_td0.005': label = " [LIVE]"

        print(f"  {cand:<20} {ratio:>7.3f} {details['kfold']:>8} {details['wf']:>8} "
              f"{details['era']:>5} {details['mc']:>8} {rec_sh:>8.3f} {score:>5}/10{label}", flush=True)

    best = max(scorecard.items(), key=lambda x: x[1]['score'])
    print(f"\n  === RECOMMENDATION ===", flush=True)
    print(f"  Best candidate: {best[0]} (score {best[1]['score']}/10)", flush=True)

    if best[0] == 'ta0.02_td0.005':
        print(f"  -> Current live params ARE optimal even in M15", flush=True)
    elif best[0] == 'ta0.06_td0.01':
        print(f"  -> Should REVERT to old baseline params", flush=True)
    else:
        print(f"  -> Should SWITCH to {best[0]}", flush=True)

    save_phase("phase_12_verdict", scorecard)

    total_m = (time.time() - t0) / 60
    print(f"\n  Total time: {total_m:.1f} minutes")
    print(f"{'='*80}")


# ========================= MAIN =========================
if __name__ == '__main__':
    print(f"{'='*80}")
    print(f"  R199 - Keltner Trail Parameter Mega Sweep (M15 + H1)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Grid: {len(TA_GRID)} activate x {len(TD_GRID)} distance")
    print(f"{'='*80}\n")

    h1 = load_h1()
    m15 = load_m15()

    df_temp = h1.copy(); df_temp['ATR'] = compute_atr(df_temp)
    pctl = compute_atr_pctl(df_temp['ATR'], lb=300)
    print(f"  ATR pctl: {pctl.notna().sum()} valid\n", flush=True)

    phase_1_h1_grid(h1, pctl)
    phase_2_m15_grid(h1, m15, pctl)
    phase_3_divergence()
    candidates = phase_4_select_candidates()
    phase_5_kfold_m15(h1, m15, pctl)
    phase_6_walkforward_m15(h1, m15, pctl)
    phase_7_era_m15(h1, m15, pctl)
    phase_8_exit_compare(h1, m15, pctl)
    phase_9_monte_carlo_m15(h1, m15, pctl)
    phase_10_spread_m15(h1, m15, pctl)
    phase_11_recent_m15(h1, m15, pctl)
    phase_12_verdict()
