#!/usr/bin/env python3
"""
R178b — Annual Stability Test for Session-ADX Configuration
=============================================================
Validates the R178 recommendation (Asia=OFF, London=10, NY=16, Evening=OFF)
by breaking down Sharpe/PnL/WR per calendar year.

Key question: is London ADX=10 stable across all years, or does it leak
low-quality signals in sideways years (2016-2018)?

Tests: London ADX in {10, 12, 14} × {Asia OFF, Evening OFF fixed}
Output: per-year Sharpe table + recommendation.
"""
import sys, os, time, json, warnings, glob
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r178b_annual_stability")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01

SESSION_GROUPS = {
    'asia':    set(range(22, 24)) | set(range(0, 7)),
    'london':  set(range(7, 12)),
    'ny':      set(range(12, 17)),
    'evening': set(range(17, 22)),
}

YEARS = list(range(2015, 2027))

CONFIGS_TO_TEST = [
    {'label': 'Baseline_ADX14',       'asia': 14,  'london': 14, 'ny': 14,  'evening': 14},
    {'label': 'R178_London10_NY16',   'asia': 999, 'london': 10, 'ny': 16,  'evening': 999},
    {'label': 'Alt_London12_NY16',    'asia': 999, 'london': 12, 'ny': 16,  'evening': 999},
    {'label': 'Alt_London12_NY14',    'asia': 999, 'london': 12, 'ny': 14,  'evening': 999},
    {'label': 'Alt_London10_NY14',    'asia': 999, 'london': 10, 'ny': 14,  'evening': 999},
    {'label': 'Alt_London14_NY14',    'asia': 999, 'london': 14, 'ny': 14,  'evening': 999},
]


def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


def compute_adx(df, period=14):
    high = df['High']; low = df['Low']; close = df['Close']
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.DataFrame({'hl': high - low, 'hc': (high - close.shift(1)).abs(),
                        'lc': (low - close.shift(1)).abs()}).max(axis=1)
    atr_s = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_s)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_s)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(period).mean()


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


def _run_exit(pos, i, h, lo_v, c, spread, lot, pv, times,
              sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap=35):
    held = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_h = (h - pos['entry'] - spread) * lot * pv
        pnl_l = (lo_v - pos['entry'] - spread) * lot * pv
        pnl_c = (c - pos['entry'] - spread) * lot * pv
    else:
        pnl_h = (pos['entry'] - lo_v - spread) * lot * pv
        pnl_l = (pos['entry'] - h - spread) * lot * pv
        pnl_c = (pos['entry'] - c - spread) * lot * pv
    tp_val = tp_atr * pos['atr'] * lot * pv
    sl_val = sl_atr * pos['atr'] * lot * pv
    if pnl_h >= tp_val:
        return _mk(pos, c, times[i], "TP", i, tp_val)
    if pnl_l <= -sl_val:
        return _mk(pos, c, times[i], "SL", i, -sl_val)
    if maxloss_cap > 0 and pnl_c < -maxloss_cap:
        return _mk(pos, c, times[i], "MaxLossCap", i, -maxloss_cap)
    ad = trail_act * pos['atr']; td = trail_dist * pos['atr']
    if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
        ts_p = h - td
        if lo_v <= ts_p:
            return _mk(pos, c, times[i], "Trail", i, (ts_p - pos['entry'] - spread) * lot * pv)
    elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
        ts_p = lo_v + td
        if h >= ts_p:
            return _mk(pos, c, times[i], "Trail", i, (pos['entry'] - ts_p - spread) * lot * pv)
    if held >= max_hold:
        return _mk(pos, c, times[i], "Timeout", i, pnl_c)
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
    if len(arr) == 0:
        return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def _compute_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0.0, 'pnl': 0.0, 'wr': 0.0, 'max_dd': 0.0,
                'avg_pnl': 0.0, 'sl_pct': 0.0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    sl_hits = sum(1 for t in trades if t.get('reason') == 'SL')
    return {
        'n': n,
        'sharpe': round(_sharpe(daily), 3),
        'pnl': round(sum(pnls), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'max_dd': round(_max_dd(daily), 2),
        'avg_pnl': round(float(np.mean(pnls)), 4),
        'sl_pct': round(sl_hits / n * 100, 1),
    }


def load_h1():
    candidates = sorted(glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    csv_path = candidates[-1]
    print(f"  Loading: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.set_index('time')[['open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return df


def prepare_kc(df, ema_period=25, kc_mult=1.2):
    out = df.copy()
    out['ATR'] = compute_atr(out)
    out['ADX'] = compute_adx(out)
    out['EMA100'] = out['Close'].ewm(span=100, adjust=False).mean()
    ema_kc = out['Close'].ewm(span=ema_period, adjust=False).mean()
    out['KC_upper'] = ema_kc + kc_mult * out['ATR']
    out['KC_lower'] = ema_kc - kc_mult * out['ATR']
    return out


def build_hour_adx_map(asia, london, ny, evening):
    m = {}
    for hr in SESSION_GROUPS['asia']:
        m[hr] = asia
    for hr in SESSION_GROUPS['london']:
        m[hr] = london
    for hr in SESSION_GROUPS['ny']:
        m[hr] = ny
    for hr in SESSION_GROUPS['evening']:
        m[hr] = evening
    return m


def bt_keltner_session(df_prepared, spread, lot, hour_adx_map,
                       sl_atr=3.5, tp_atr=8.0, trail_act=0.14,
                       trail_dist=0.025, max_hold=20, maxloss_cap=35):
    c = df_prepared['Close'].values; h = df_prepared['High'].values
    lo = df_prepared['Low'].values
    atr = df_prepared['ATR'].values; adx = df_prepared['ADX'].values
    kc_up = df_prepared['KC_upper'].values; kc_lo = df_prepared['KC_lower'].values
    ema100 = df_prepared['EMA100'].values
    times = df_prepared.index; hours = df_prepared.index.hour
    n = len(df_prepared)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1 or np.isnan(adx[i]): continue
        hr = hours[i]
        adx_th = hour_adx_map.get(hr, 14)
        if adx[i] < adx_th: continue
        if c[i] > kc_up[i] and c[i] > ema100[i]:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < kc_lo[i] and c[i] < ema100[i]:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def main():
    t0 = time.time()
    print("=" * 100, flush=True)
    print("  R178b — Annual Stability Test for Session-ADX", flush=True)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 100, flush=True)

    h1_df = load_h1()
    print(f"  {len(h1_df)} bars, {h1_df.index[0]} ~ {h1_df.index[-1]}", flush=True)

    df_kc = prepare_kc(h1_df)

    # Per-year ATR context
    print(f"\n  Per-Year ATR Context:", flush=True)
    for yr in YEARS:
        yr_data = df_kc[(df_kc.index >= f'{yr}-01-01') & (df_kc.index < f'{yr+1}-01-01')]
        if len(yr_data) < 100:
            continue
        yr_atr = yr_data['ATR'].dropna().mean()
        yr_adx = yr_data['ADX'].dropna().mean()
        print(f"    {yr}: mean_ATR=${yr_atr:.2f}  mean_ADX={yr_adx:.1f}  bars={len(yr_data)}", flush=True)

    results_all = {}

    # ══════════════════════════════════════════════════════════
    # Phase 1: Annual breakdown per config
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}", flush=True)
    print("  Phase 1: Annual Sharpe Breakdown per Config", flush=True)
    print(f"{'=' * 100}", flush=True)

    for cfg in CONFIGS_TO_TEST:
        label = cfg['label']
        hour_map = build_hour_adx_map(cfg['asia'], cfg['london'], cfg['ny'], cfg['evening'])

        # Full backtest to get all trades
        all_trades = bt_keltner_session(df_kc, SPREAD, UNIT_LOT, hour_map)
        full_stats = _compute_stats(all_trades)

        print(f"\n  {'─' * 95}", flush=True)
        print(f"  {label}  (A={cfg['asia'] if cfg['asia'] < 900 else 'OFF'}, "
              f"L={cfg['london']}, N={cfg['ny']}, "
              f"E={cfg['evening'] if cfg['evening'] < 900 else 'OFF'})", flush=True)
        print(f"  Full: n={full_stats['n']}  Sharpe={full_stats['sharpe']:.3f}  "
              f"PnL=${full_stats['pnl']:,.0f}  WR={full_stats['wr']:.1f}%", flush=True)

        print(f"\n  {'Year':>6} {'N':>5} {'Sharpe':>8} {'PnL':>10} {'WR%':>6} "
              f"{'MaxDD':>8} {'AvgPnL':>8} {'SL%':>5} {'ATR':>6} {'ADX':>5}", flush=True)
        print(f"  {'-'*6} {'-'*5} {'-'*8} {'-'*10} {'-'*6} "
              f"{'-'*8} {'-'*8} {'-'*5} {'-'*6} {'-'*5}", flush=True)

        year_data = {}
        for yr in YEARS:
            yr_trades = [t for t in all_trades
                         if pd.Timestamp(t['entry_time']).year == yr]
            s = _compute_stats(yr_trades)

            yr_slice = df_kc[(df_kc.index >= f'{yr}-01-01') & (df_kc.index < f'{yr+1}-01-01')]
            yr_atr = yr_slice['ATR'].dropna().mean() if len(yr_slice) > 0 else 0
            yr_adx = yr_slice['ADX'].dropna().mean() if len(yr_slice) > 0 else 0

            if s['n'] > 0:
                marker = ""
                if s['sharpe'] < 0:
                    marker = " *** NEGATIVE ***"
                elif s['sharpe'] < 1.0:
                    marker = " * weak"
                print(f"  {yr:>6} {s['n']:>5} {s['sharpe']:>8.3f} ${s['pnl']:>9,.0f} "
                      f"{s['wr']:>5.1f}% ${s['max_dd']:>7,.0f} ${s['avg_pnl']:>7.3f} "
                      f"{s['sl_pct']:>4.1f}% ${yr_atr:>5.1f} {yr_adx:>4.1f}{marker}", flush=True)
            else:
                print(f"  {yr:>6}     0        -          -      -        -        -"
                      f"     - ${yr_atr:>5.1f} {yr_adx:>4.1f}", flush=True)

            year_data[str(yr)] = {**s, 'atr_mean': round(yr_atr, 2), 'adx_mean': round(yr_adx, 1)}

        # Stability metrics
        year_sharpes = [year_data[str(yr)]['sharpe'] for yr in YEARS if year_data[str(yr)]['n'] > 5]
        if year_sharpes:
            sh_mean = float(np.mean(year_sharpes))
            sh_std = float(np.std(year_sharpes, ddof=1))
            sh_min = float(min(year_sharpes))
            sh_max = float(max(year_sharpes))
            neg_years = sum(1 for s in year_sharpes if s < 0)
            weak_years = sum(1 for s in year_sharpes if 0 < s < 1.0)
            cv = sh_std / abs(sh_mean) if abs(sh_mean) > 0.01 else float('inf')
            print(f"\n  Stability: mean={sh_mean:.3f}  std={sh_std:.3f}  "
                  f"CV={cv:.2f}  min={sh_min:.3f}  max={sh_max:.3f}", flush=True)
            print(f"  Negative years: {neg_years}  Weak years (<1.0): {weak_years}  "
                  f"Total: {len(year_sharpes)}", flush=True)

            year_data['_stability'] = {
                'mean': round(sh_mean, 3), 'std': round(sh_std, 3),
                'cv': round(cv, 3), 'min': round(sh_min, 3), 'max': round(sh_max, 3),
                'negative_years': neg_years, 'weak_years': weak_years,
            }

        results_all[label] = {
            'config': cfg,
            'full_stats': full_stats,
            'per_year': year_data,
        }

    # ══════════════════════════════════════════════════════════
    # Phase 2: London-hour drill-down (ADX=10 vs 12 vs 14)
    # ══════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 100}", flush=True)
    print("  Phase 2: London Session Drill-Down (Hour 7-11, ADX=10 vs 12 vs 14)", flush=True)
    print(f"{'=' * 100}", flush=True)

    london_hours = sorted(SESSION_GROUPS['london'])

    for adx_th in [10, 12, 14]:
        hour_map = build_hour_adx_map(999, adx_th, 16, 999)
        all_trades = bt_keltner_session(df_kc, SPREAD, UNIT_LOT, hour_map)
        london_trades = [t for t in all_trades
                         if pd.Timestamp(t['entry_time']).hour in london_hours]

        print(f"\n  London ADX={adx_th}:", flush=True)
        print(f"  {'Year':>6} {'N':>5} {'Sharpe':>8} {'PnL':>10} {'WR%':>6} {'LondonN':>7}", flush=True)
        print(f"  {'-'*6} {'-'*5} {'-'*8} {'-'*10} {'-'*6} {'-'*7}", flush=True)

        for yr in YEARS:
            yr_trades = [t for t in london_trades if pd.Timestamp(t['entry_time']).year == yr]
            s = _compute_stats(yr_trades)
            all_yr = [t for t in all_trades if pd.Timestamp(t['entry_time']).year == yr]
            full_s = _compute_stats(all_yr)
            if s['n'] > 0:
                marker = " ***" if s['sharpe'] < 0 else ""
                print(f"  {yr:>6} {full_s['n']:>5} {full_s['sharpe']:>8.3f} ${full_s['pnl']:>9,.0f} "
                      f"{full_s['wr']:>5.1f}% {s['n']:>7}{marker}", flush=True)
            else:
                print(f"  {yr:>6}     0        -          -      -       0", flush=True)

    # ══════════════════════════════════════════════════════════
    # Phase 3: Signal quality (low-ADX trades)
    # ══════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 100}", flush=True)
    print("  Phase 3: Low-ADX Signal Quality Analysis", flush=True)
    print(f"{'=' * 100}", flush=True)

    print(f"  Trades that fire at ADX in [10,14) during London hours:", flush=True)
    print(f"  These are the EXTRA trades ADX=10 adds vs ADX=14.", flush=True)

    flat14_map = {hr: 14 for hr in range(24)}
    flat14_map.update({hr: 999 for hr in SESSION_GROUPS['asia']})
    flat14_map.update({hr: 999 for hr in SESSION_GROUPS['evening']})

    flat10_map = dict(flat14_map)
    for hr in SESSION_GROUPS['london']:
        flat10_map[hr] = 10
    for hr in SESSION_GROUPS['ny']:
        flat10_map[hr] = 16

    trades_adx14 = set()
    for t in bt_keltner_session(df_kc, SPREAD, UNIT_LOT, flat14_map):
        trades_adx14.add(str(t['entry_time']))

    trades_adx10 = bt_keltner_session(df_kc, SPREAD, UNIT_LOT, flat10_map)
    extra_trades = [t for t in trades_adx10 if str(t['entry_time']) not in trades_adx14]

    if extra_trades:
        extra_stats = _compute_stats(extra_trades)
        print(f"\n  Extra trades (ADX 10-14 window in London): {extra_stats['n']}", flush=True)
        print(f"  WR={extra_stats['wr']:.1f}%  Sharpe={extra_stats['sharpe']:.3f}  "
              f"PnL=${extra_stats['pnl']:,.0f}  AvgPnL=${extra_stats['avg_pnl']:.4f}", flush=True)

        print(f"\n  Per-year breakdown of extra trades:", flush=True)
        print(f"  {'Year':>6} {'N':>4} {'WR%':>6} {'PnL':>9} {'AvgPnL':>9}", flush=True)
        print(f"  {'-'*6} {'-'*4} {'-'*6} {'-'*9} {'-'*9}", flush=True)
        extra_by_year = {}
        for yr in YEARS:
            yr_extra = [t for t in extra_trades if pd.Timestamp(t['entry_time']).year == yr]
            s = _compute_stats(yr_extra)
            extra_by_year[str(yr)] = s
            if s['n'] > 0:
                marker = " *** DRAG" if s['avg_pnl'] < 0 else ""
                print(f"  {yr:>6} {s['n']:>4} {s['wr']:>5.1f}% ${s['pnl']:>8,.2f} "
                      f"${s['avg_pnl']:>8.4f}{marker}", flush=True)

        results_all['extra_low_adx_trades'] = {
            'total': extra_stats,
            'per_year': extra_by_year,
        }
    else:
        print(f"\n  No extra trades found (unexpected).", flush=True)

    # ══════════════════════════════════════════════════════════
    # Recommendation
    # ══════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 100}", flush=True)
    print("  RECOMMENDATION", flush=True)
    print(f"{'=' * 100}", flush=True)

    best_label = None
    best_cv = float('inf')
    for label, data in results_all.items():
        if isinstance(data, dict) and 'per_year' in data:
            stab = data['per_year'].get('_stability', {})
            cv = stab.get('cv', float('inf'))
            neg = stab.get('negative_years', 99)
            if neg == 0 and cv < best_cv:
                best_cv = cv
                best_label = label

    if best_label is None:
        for label, data in results_all.items():
            if isinstance(data, dict) and 'per_year' in data:
                stab = data['per_year'].get('_stability', {})
                cv = stab.get('cv', float('inf'))
                if cv < best_cv:
                    best_cv = cv
                    best_label = label

    print(f"\n  Config comparison (annual stability):", flush=True)
    print(f"  {'Config':<25} {'Full_Sh':>8} {'Yr_Mean':>8} {'Yr_Std':>7} {'CV':>6} "
          f"{'Neg':>4} {'Weak':>5} {'Verdict':>10}", flush=True)
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*4} {'-'*5} {'-'*10}", flush=True)

    for label, data in results_all.items():
        if not isinstance(data, dict) or 'per_year' not in data:
            continue
        stab = data['per_year'].get('_stability', {})
        if not stab:
            continue
        full_sh = data['full_stats']['sharpe']
        verdict = "BEST" if label == best_label else ("OK" if stab.get('negative_years', 99) == 0 else "RISKY")
        print(f"  {label:<25} {full_sh:>8.3f} {stab['mean']:>8.3f} {stab['std']:>7.3f} "
              f"{stab['cv']:>6.2f} {stab.get('negative_years', '?'):>4} "
              f"{stab.get('weak_years', '?'):>5} {verdict:>10}", flush=True)

    if best_label:
        best_data = results_all[best_label]
        cfg = best_data['config']
        print(f"\n  RECOMMENDED: {best_label}", flush=True)
        print(f"    Config: Asia={'OFF' if cfg['asia'] >= 900 else cfg['asia']}, "
              f"London={cfg['london']}, NY={cfg['ny']}, "
              f"Evening={'OFF' if cfg['evening'] >= 900 else cfg['evening']}", flush=True)
        print(f"    Full Sharpe: {best_data['full_stats']['sharpe']:.3f}", flush=True)
        print(f"    CV: {best_cv:.3f} (lower = more stable across years)", flush=True)

    # ══════════════════════════════════════════════════════════
    # Phase 4: Era-Segmented Testing (4 dimensions)
    # ══════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 100}", flush=True)
    print("  Phase 4: Era-Segmented Testing (Full / Hike / Cut / Recent 3Y)", flush=True)
    print(f"{'=' * 100}", flush=True)

    ERA_SEGMENTS = {
        'Full':      None,
        'Hike':      [("2015-12-01", "2019-01-01"), ("2022-03-01", "2023-08-01")],
        'Cut':       [("2019-07-01", "2022-03-01"), ("2024-09-01", "2026-06-01")],
        'Recent_3Y': [("2023-06-01", "2026-06-01")],
    }

    def filter_trades_by_era(trades, era_name):
        if era_name == 'Full' or ERA_SEGMENTS[era_name] is None:
            return trades
        periods = ERA_SEGMENTS[era_name]
        filtered = []
        for t in trades:
            entry = pd.Timestamp(t['entry_time'])
            for start, end in periods:
                if pd.Timestamp(start) <= entry < pd.Timestamp(end):
                    filtered.append(t)
                    break
        return filtered

    era_results = {}
    for cfg in CONFIGS_TO_TEST:
        label = cfg['label']
        hour_map = build_hour_adx_map(cfg['asia'], cfg['london'], cfg['ny'], cfg['evening'])
        all_trades = bt_keltner_session(df_kc, SPREAD, UNIT_LOT, hour_map)

        print(f"\n  {label}:", flush=True)
        print(f"  {'Era':<12} {'N':>5} {'Sharpe':>8} {'PnL':>10} {'WR%':>6} {'MaxDD':>8}", flush=True)
        print(f"  {'-'*12} {'-'*5} {'-'*8} {'-'*10} {'-'*6} {'-'*8}", flush=True)

        cfg_era = {}
        for era_name in ['Full', 'Hike', 'Cut', 'Recent_3Y']:
            era_trades = filter_trades_by_era(all_trades, era_name)
            s = _compute_stats(era_trades)
            cfg_era[era_name] = s
            print(f"  {era_name:<12} {s['n']:>5} {s['sharpe']:>8.3f} ${s['pnl']:>9,.0f} "
                  f"{s['wr']:>5.1f}% ${s['max_dd']:>7,.0f}", flush=True)

        era_results[label] = cfg_era

    # Era comparison across configs
    print(f"\n\n  Era Cross-Config Comparison (Sharpe):", flush=True)
    labels = [c['label'] for c in CONFIGS_TO_TEST]
    print(f"  {'Config':<25} {'Full':>8} {'Hike':>8} {'Cut':>8} {'Rec3Y':>8} {'Hike/Full':>10}", flush=True)
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}", flush=True)
    for label in labels:
        d = era_results[label]
        full_sh = d['Full']['sharpe']
        hike_sh = d['Hike']['sharpe']
        cut_sh = d['Cut']['sharpe']
        rec_sh = d['Recent_3Y']['sharpe']
        ratio = hike_sh / full_sh if abs(full_sh) > 0.01 else 0
        flag = " *** WEAK HIKE" if ratio < 0.5 else ""
        print(f"  {label:<25} {full_sh:>8.3f} {hike_sh:>8.3f} {cut_sh:>8.3f} "
              f"{rec_sh:>8.3f} {ratio:>9.1%}{flag}", flush=True)

    results_all['era_segmented'] = era_results

    elapsed = time.time() - t0
    print(f"\n  R178b complete in {elapsed:.0f}s ({elapsed / 60:.1f}min)", flush=True)
    print(f"{'=' * 100}", flush=True)

    out_path = OUTPUT_DIR / "r178b_results.json"
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=2, default=str)
    print(f"  Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
