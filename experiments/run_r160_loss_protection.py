#!/usr/bin/env python3
"""
R160 — Loss Protection Enhancements
=====================================
Three targeted improvements to address specific loss patterns:

Part A: ADX Low-Volatility Filter
  - When ADX < threshold, skip Keltner entries (prevents 03-25 chop losses)
  - Test thresholds: 12, 14, 16, 18, 20, 22, 25

Part B: Rule B Sigma Sensitivity
  - Compare sigma 2.0, 2.5, 3.0, 3.5 (lower = more aggressive filtering)
  - Goal: catch 04-03~04-08 cluster (Z was 1.3~2.7)

Part C: Adaptive SL / Fixed Dollar Cap
  - Cap SL at fixed dollar amount regardless of ATR
  - Test caps: $30, $40, $50, $60 per trade (at 0.02 lot)
  - Also test ATR-adaptive lot sizing: reduce lot when ATR > 2x average

Each part runs independently on the L8_MAX (Keltner) strategy with MaxHold=8.
K-Fold validation on the best settings from each part.
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS

OUTPUT_DIR = Path("results/r160_loss_protection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01
CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]

t0 = time.time()


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


def compute_adx(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    mask = plus_dm < minus_dm
    plus_dm[mask] = 0
    mask2 = minus_dm < plus_dm
    minus_dm[mask2] = 0
    tr = pd.DataFrame({
        'hl': h - l, 'hc': (h - c.shift()).abs(), 'lc': (l - c.shift()).abs()
    }).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).clip(lower=1e-6) * 100
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx


def _normalize_ts(ts):
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    return t


def trades_to_daily(trades):
    if not trades:
        return np.array([0.0])
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).normalize()
        if hasattr(d, 'tz') and d.tzinfo is not None:
            d = d.tz_localize(None)
        daily[d] = daily.get(d, 0) + t['pnl']
    if not daily:
        return np.array([0.0])
    return np.array([daily[k] for k in sorted(daily.keys())])


def sharpe(arr):
    if len(arr) < 10 or np.std(arr, ddof=1) == 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))


def max_dd(arr):
    if len(arr) < 2:
        return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def compute_stats(trades, label=""):
    n = len(trades)
    if n == 0:
        return {'label': label, 'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0,
                'max_dd': 0, 'avg_pnl': 0, 'worst_trade': 0}
    pnls = np.array([t['pnl'] for t in trades])
    daily = trades_to_daily(trades)
    return {
        'label': label, 'n': n,
        'sharpe': round(sharpe(daily), 3),
        'pnl': round(float(pnls.sum()), 2),
        'wr': round(float((pnls > 0).sum()) / n * 100, 1),
        'max_dd': round(max_dd(daily), 2),
        'avg_pnl': round(float(pnls.mean()), 4),
        'worst_trade': round(float(pnls.min()), 2),
    }


# ═══════════════════════════════════════════════════════════════
# Part A: ADX Low-Volatility Filter (Keltner only)
# ═══════════════════════════════════════════════════════════════

def run_l8_with_adx_filter(bundle, adx_threshold=14):
    """Run L8_MAX with varying ADX threshold."""
    kw = dict(LIVE_PARITY_KWARGS)
    kw['maxloss_cap'] = CAPS['L8_MAX']
    kw['spread_cost'] = SPREAD
    kw['initial_capital'] = 2000
    kw['keltner_max_hold_m15'] = 8
    kw['min_lot_size'] = UNIT_LOT
    kw['max_lot_size'] = UNIT_LOT
    kw['keltner_adx_threshold'] = adx_threshold
    result = run_variant(bundle, "L8_MAX", verbose=False, **kw)
    raw = result.get('_trades', [])
    return [
        {'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
         'entry_time': t.entry_time, 'exit_time': t.exit_time,
         'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0}
        for t in raw
    ]


def part_a_adx_filter(bundle):
    print("\n" + "=" * 70)
    print("  PART A: ADX Low-Volatility Filter")
    print("  Current production: adx_threshold=14")
    print("=" * 70, flush=True)

    thresholds = [0, 10, 12, 14, 16, 18, 20, 22, 25, 30]
    results = {}

    print(f"\n  {'ADX_thresh':>10} {'N':>6} {'Sharpe':>7} {'PnL':>10} "
          f"{'MaxDD':>8} {'WR':>6} {'Worst':>8}")
    print("  " + "-" * 65)

    for adx_t in thresholds:
        trades = run_l8_with_adx_filter(bundle, adx_t)
        st = compute_stats(trades, f"ADX>={adx_t}")
        results[adx_t] = st
        print(f"  {adx_t:>10} {st['n']:>6} {st['sharpe']:>7.3f} {st['pnl']:>10.1f} "
              f"{st['max_dd']:>8.1f} {st['wr']:>5.1f}% {st['worst_trade']:>8.2f}",
              flush=True)

    best_adx = max(results.items(), key=lambda x: x[1]['sharpe'])
    print(f"\n  Best ADX threshold: {best_adx[0]} (Sharpe={best_adx[1]['sharpe']:.3f})",
          flush=True)
    return results, best_adx[0]


# ═══════════════════════════════════════════════════════════════
# Part B: Rule B Sigma Sensitivity
# ═══════════════════════════════════════════════════════════════

def build_extreme_mask(h1_df, cusum_sigma=3.0, extreme_window=12):
    n = len(h1_df)
    extreme = np.zeros(n, dtype=bool)
    atr = compute_atr(h1_df).values
    atr_clean = np.nan_to_num(atr, nan=0.0)
    atr_mean = pd.Series(atr_clean).rolling(60, min_periods=20).mean().values
    atr_std = pd.Series(atr_clean).rolling(60, min_periods=20).std().values
    atr_std = np.maximum(atr_std, 1e-6)
    cusum_trigger = atr_clean > (atr_mean + cusum_sigma * atr_std)
    for i in range(n):
        if cusum_trigger[i]:
            end_i = min(i + extreme_window, n)
            extreme[i:end_i] = True
    return extreme


def apply_rule_b(trades, h1_df, extreme_mask, skip_bars=8):
    if not trades:
        return [], 0
    times_idx = h1_df.index
    n_bars = len(times_idx)
    idx_is_tz_aware = times_idx.tz is not None

    def _find_bar(ts):
        ts = pd.Timestamp(ts)
        if idx_is_tz_aware and ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        elif not idx_is_tz_aware and ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        idx = times_idx.searchsorted(ts)
        return min(idx, n_bars - 1) if idx < n_bars else n_bars - 1

    protected = []; skipped = 0
    for t in trades:
        entry_bar = _find_bar(t['entry_time'])
        skip_end = entry_bar
        for j in range(max(0, entry_bar - skip_bars), entry_bar):
            if j < n_bars and extreme_mask[j]:
                skip_end = j + skip_bars
                break
        if entry_bar < skip_end:
            skipped += 1
            continue
        protected.append(t)
    return protected, skipped


def add_psar(df, af_start=0.01, af_max=0.05):
    df = df.copy(); n = len(df)
    psar = np.zeros(n); direction = np.ones(n)
    af = af_start; ep = df['High'].iloc[0]; psar[0] = df['Low'].iloc[0]
    for i in range(1, n):
        prev = psar[i-1]
        if direction[i-1] == 1:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], df['Low'].iloc[i-1], df['Low'].iloc[max(0, i-2)])
            if df['Low'].iloc[i] < psar[i]:
                direction[i] = -1; psar[i] = ep; ep = df['Low'].iloc[i]; af = af_start
            else:
                direction[i] = 1
                if df['High'].iloc[i] > ep:
                    ep = df['High'].iloc[i]; af = min(af+af_start, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], df['High'].iloc[i-1], df['High'].iloc[max(0, i-2)])
            if df['High'].iloc[i] > psar[i]:
                direction[i] = 1; psar[i] = ep; ep = df['High'].iloc[i]; af = af_start
            else:
                direction[i] = -1
                if df['Low'].iloc[i] < ep:
                    ep = df['Low'].iloc[i]; af = min(af+af_start, af_max)
    df['PSAR_dir'] = direction; df['ATR'] = compute_atr(df)
    return df


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


def _run_exit_with_cap(pos, i, h, lo_v, c, spread, lot, pv, times,
                       sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold,
                       maxloss_cap=0):
    held = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_c = (c - pos['entry'] - spread) * lot * pv
    else:
        pnl_c = (pos['entry'] - c - spread) * lot * pv
    tp_val = tp_atr * pos['atr'] * lot * pv
    sl_val = sl_atr * pos['atr'] * lot * pv
    if pos['dir'] == 'BUY':
        pnl_h = (h - pos['entry'] - spread) * lot * pv
        pnl_l = (lo_v - pos['entry'] - spread) * lot * pv
    else:
        pnl_h = (pos['entry'] - lo_v - spread) * lot * pv
        pnl_l = (pos['entry'] - h - spread) * lot * pv
    if pnl_h >= tp_val:
        return _mk(pos, c, times[i], "TP", i, tp_val)
    if pnl_l <= -sl_val:
        return _mk(pos, c, times[i], "SL", i, -sl_val)
    if maxloss_cap > 0 and pnl_c < -maxloss_cap:
        return _mk(pos, c, times[i], "MaxLossCap", i, -maxloss_cap)
    ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
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


def bt_psar(h1_df, spread, lot, maxloss_cap=0):
    sl_atr=4.5; tp_atr=16.0; trail_act=0.20; trail_dist=0.04; max_hold=20
    df = add_psar(h1_df).dropna(subset=['PSAR_dir', 'ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    pdir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pdir[i-1] == -1 and pdir[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif pdir[i-1] == 1 and pdir[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom(h1_df, spread, lot, maxloss_cap=0):
    fast=480; slow=720; sl_atr=4.5; tp_atr=6.0
    trail_act=0.14; trail_dist=0.025; max_hold=20
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    max_lb = max(fast, slow)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        if c[i-fast] > 0: s += 0.5 * np.sign(c[i]/c[i-fast] - 1.0)
        if c[i-slow] > 0: s += 0.5 * np.sign(c[i]/c[i-slow] - 1.0)
        score[i] = s
    trades = []; pos = None; last_exit = -999
    for i in range(max_lb+1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                pnl = (c[i]-pos['entry']-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                pnl = (pos['entry']-c[i]-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(score[i]) or np.isnan(score[i-1]): continue
        if score[i] > 0 and score[i-1] <= 0:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_sess_bo(h1_df, spread, lot, maxloss_cap=0):
    session_hour=12; lookback=4; sl_atr=4.5; tp_atr=4.0
    trail_act=0.14; trail_dist=0.025; max_hold=20
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(lookback, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if hours[i] != session_hour: continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_l8_max(data_bundle, spread, lot, maxloss_cap=35):
    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': maxloss_cap,
          'spread_cost': spread, 'initial_capital': 2000,
          'keltner_max_hold_m15': 8,
          'min_lot_size': lot, 'max_lot_size': lot}
    result = run_variant(data_bundle, "L8_MAX", verbose=False, **kw)
    raw = result.get('_trades', [])
    return [
        {'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
         'entry_time': t.entry_time, 'exit_time': t.exit_time,
         'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0}
        for t in raw
    ]


def merge_portfolio_trades(strat_trades_dict, lot_scale=None):
    if lot_scale is None:
        lot_scale = R89_LOTS
    all_trades = []
    for strat_name, trades in strat_trades_dict.items():
        mult = lot_scale.get(strat_name, UNIT_LOT) / UNIT_LOT
        for t in trades:
            all_trades.append({
                'strategy': strat_name,
                'dir': t['dir'], 'entry': t['entry'], 'exit': t['exit'],
                'entry_time': _normalize_ts(t['entry_time']),
                'exit_time': _normalize_ts(t['exit_time']),
                'pnl': t['pnl'] * mult,
                'pnl_unit': t['pnl'],
                'reason': t['reason'],
            })
    all_trades.sort(key=lambda x: x['exit_time'])
    return all_trades


def part_b_sigma_sensitivity(h1_df, portfolio_all):
    print("\n" + "=" * 70)
    print("  PART B: Rule B Sigma Sensitivity")
    print("  Testing different trigger thresholds")
    print("=" * 70, flush=True)

    sigmas = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    skip_bars_list = [6, 8, 10]
    results = {}

    print(f"\n  {'Sigma':>6} {'SkipBars':>9} {'N':>6} {'Sharpe':>7} {'PnL':>10} "
          f"{'MaxDD':>8} {'Skipped':>8} {'Extreme%':>9}")
    print("  " + "-" * 75)

    for sigma in sigmas:
        mask = build_extreme_mask(h1_df, cusum_sigma=sigma, extreme_window=12)
        pct_extreme = mask.sum() / len(mask) * 100
        for sb in skip_bars_list:
            filtered, skipped = apply_rule_b(portfolio_all, h1_df, mask, sb)
            st = compute_stats(filtered, f"s{sigma}_sb{sb}")
            key = f"sigma{sigma}_skip{sb}"
            results[key] = {**st, 'sigma': sigma, 'skip_bars': sb,
                            'skipped': skipped, 'extreme_pct': round(pct_extreme, 2)}
            print(f"  {sigma:>6.1f} {sb:>9} {st['n']:>6} {st['sharpe']:>7.3f} "
                  f"{st['pnl']:>10.1f} {st['max_dd']:>8.1f} {skipped:>8} "
                  f"{pct_extreme:>8.1f}%", flush=True)

    best_key = max(results.items(), key=lambda x: x[1]['sharpe'])
    print(f"\n  Best: {best_key[0]} (Sharpe={best_key[1]['sharpe']:.3f}, "
          f"skipped={best_key[1]['skipped']})", flush=True)
    return results, best_key


# ═══════════════════════════════════════════════════════════════
# Part C: Adaptive SL / Fixed Dollar Cap
# ═══════════════════════════════════════════════════════════════

def run_l8_with_dollar_cap(bundle, dollar_cap=50):
    """Run L8_MAX with a fixed dollar cap per trade (at 0.02 lot equivalent).
    Dollar cap is specified for 0.02 lot; scale to unit lot internally."""
    unit_cap = dollar_cap * (UNIT_LOT / 0.02)
    kw = dict(LIVE_PARITY_KWARGS)
    kw['maxloss_cap'] = unit_cap
    kw['spread_cost'] = SPREAD
    kw['initial_capital'] = 2000
    kw['keltner_max_hold_m15'] = 8
    kw['min_lot_size'] = UNIT_LOT
    kw['max_lot_size'] = UNIT_LOT
    result = run_variant(bundle, "L8_MAX", verbose=False, **kw)
    raw = result.get('_trades', [])
    return [
        {'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
         'entry_time': t.entry_time, 'exit_time': t.exit_time,
         'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0}
        for t in raw
    ]


def run_l8_adaptive_lot(bundle, h1_df, atr_mult_threshold=2.0, reduced_lot_ratio=0.5):
    """Run L8 normally, then post-filter: trades entered when ATR > threshold
    get their PnL scaled by reduced_lot_ratio (simulating smaller lot)."""
    trades = bt_l8_max(bundle, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
    atr_series = compute_atr(h1_df)
    atr_mean = atr_series.rolling(60*24, min_periods=100).mean()
    times_idx = h1_df.index
    idx_is_tz_aware = times_idx.tz is not None

    adjusted = []
    n_reduced = 0
    for t in trades:
        ts = pd.Timestamp(t['entry_time'])
        if idx_is_tz_aware and ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        elif not idx_is_tz_aware and ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        idx = min(times_idx.searchsorted(ts), len(times_idx)-1)
        current_atr = atr_series.iloc[idx] if idx < len(atr_series) else 0
        mean_atr = atr_mean.iloc[idx] if idx < len(atr_mean) else 0
        if pd.notna(mean_atr) and mean_atr > 0 and current_atr > atr_mult_threshold * mean_atr:
            adj_t = dict(t)
            adj_t['pnl'] = t['pnl'] * reduced_lot_ratio
            adjusted.append(adj_t)
            n_reduced += 1
        else:
            adjusted.append(t)
    return adjusted, n_reduced


def part_c_adaptive_sl(bundle, h1_df):
    print("\n" + "=" * 70)
    print("  PART C: Adaptive SL / Fixed Dollar Cap")
    print("  Current: MaxLoss cap = $35 (at 0.02 lot = $17.50 at 0.01)")
    print("=" * 70, flush=True)

    # C1: Fixed dollar caps (specified at 0.02 lot level)
    print("\n  --- C1: Fixed Dollar Cap per Trade (at 0.02 lot) ---")
    dollar_caps = [20, 25, 30, 35, 40, 50, 60, 80, 0]
    results_c1 = {}

    print(f"\n  {'Cap($)':>7} {'N':>6} {'Sharpe':>7} {'PnL':>10} "
          f"{'MaxDD':>8} {'WR':>6} {'Worst':>8}")
    print("  " + "-" * 60)

    for cap in dollar_caps:
        if cap == 0:
            trades = run_l8_with_dollar_cap(bundle, dollar_cap=9999)
            label = "No cap"
        else:
            trades = run_l8_with_dollar_cap(bundle, dollar_cap=cap)
            label = f"${cap}"
        st = compute_stats(trades, label)
        results_c1[cap] = st
        worst_at_02 = st['worst_trade'] * (0.02 / UNIT_LOT)
        print(f"  {label:>7} {st['n']:>6} {st['sharpe']:>7.3f} {st['pnl']:>10.1f} "
              f"{st['max_dd']:>8.1f} {st['wr']:>5.1f}% {worst_at_02:>8.2f}", flush=True)

    # C2: Adaptive lot sizing (reduce when ATR high)
    print("\n  --- C2: Adaptive Lot (reduce to 50% when ATR > X*mean) ---")
    atr_thresholds = [1.5, 2.0, 2.5, 3.0]
    results_c2 = {}

    print(f"\n  {'ATR_mult':>8} {'N':>6} {'Sharpe':>7} {'PnL':>10} "
          f"{'MaxDD':>8} {'N_reduced':>10}")
    print("  " + "-" * 55)

    for mult in atr_thresholds:
        trades, n_red = run_l8_adaptive_lot(bundle, h1_df, mult, 0.5)
        st = compute_stats(trades, f"ATR>{mult}x")
        results_c2[mult] = {**st, 'n_reduced': n_red}
        print(f"  {mult:>8.1f} {st['n']:>6} {st['sharpe']:>7.3f} {st['pnl']:>10.1f} "
              f"{st['max_dd']:>8.1f} {n_red:>10}", flush=True)

    return results_c1, results_c2


# ═══════════════════════════════════════════════════════════════
# K-Fold on best from each part
# ═══════════════════════════════════════════════════════════════

def kfold_adx(bundle, best_adx):
    print(f"\n  K-Fold: ADX threshold={best_adx}", flush=True)
    fold_sharpes = []
    for fold_name, start, end in FOLDS:
        try:
            fold_b = bundle.slice(start, end)
            trades = run_l8_with_adx_filter(fold_b, best_adx)
        except Exception:
            fold_sharpes.append(0.0)
            continue
        daily = trades_to_daily(trades)
        fold_sharpes.append(sharpe(daily))
    pos = sum(1 for s in fold_sharpes if s > 0)
    print(f"    Folds: {[round(s,2) for s in fold_sharpes]}, {pos}/6 positive",
          flush=True)
    return {'fold_sharpes': [round(s, 3) for s in fold_sharpes],
            'positive_folds': pos, 'mean_sharpe': round(np.mean(fold_sharpes), 3)}


def kfold_sigma(bundle, h1_df, best_sigma_key, best_sigma_info):
    sigma = best_sigma_info['sigma']
    sb = best_sigma_info['skip_bars']
    print(f"\n  K-Fold: Rule B sigma={sigma}, skip_bars={sb}", flush=True)

    fold_sharpes = []
    for fold_name, start, end in FOLDS:
        fs = pd.Timestamp(start); fe = pd.Timestamp(end)
        h1_fold = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
        if len(h1_fold) < 100:
            fold_sharpes.append(0.0)
            continue

        fold_strat = {}
        fold_strat['PSAR'] = bt_psar(h1_fold, SPREAD, UNIT_LOT, CAPS['PSAR'])
        fold_strat['TSMOM'] = bt_tsmom(h1_fold, SPREAD, UNIT_LOT, CAPS['TSMOM'])
        fold_strat['SESS_BO'] = bt_sess_bo(h1_fold, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
        try:
            l8_fold = bundle.slice(start, end)
            fold_strat['L8_MAX'] = bt_l8_max(l8_fold, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
        except Exception:
            fold_sharpes.append(0.0)
            continue

        fold_portfolio = merge_portfolio_trades(fold_strat)
        if not fold_portfolio:
            fold_sharpes.append(0.0)
            continue

        fold_mask = build_extreme_mask(h1_fold, cusum_sigma=sigma, extreme_window=12)
        filtered, _ = apply_rule_b(fold_portfolio, h1_fold, fold_mask, sb)
        daily = trades_to_daily(filtered)
        fold_sharpes.append(sharpe(daily))

    pos = sum(1 for s in fold_sharpes if s > 0)
    print(f"    Folds: {[round(s,2) for s in fold_sharpes]}, {pos}/6 positive",
          flush=True)
    return {'fold_sharpes': [round(s, 3) for s in fold_sharpes],
            'positive_folds': pos, 'mean_sharpe': round(np.mean(fold_sharpes), 3)}


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH

    print("=" * 80)
    print("  R160 — Loss Protection Enhancements")
    print("  A: ADX Filter | B: Rule B Sigma | C: Adaptive SL")
    print("=" * 80, flush=True)

    print("\n  Loading data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    l8_bundle = DataBundle.load_custom()
    print(f"  H1: {len(h1_df)} bars, M15: {len(m15_raw)} bars", flush=True)

    # Part A
    part_a_results, best_adx = part_a_adx_filter(l8_bundle)

    # Build portfolio for Part B
    print("\n  Building portfolio for Part B...", flush=True)
    strat_trades = {}
    strat_trades['L8_MAX'] = bt_l8_max(l8_bundle, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
    strat_trades['PSAR'] = bt_psar(h1_df, SPREAD, UNIT_LOT, CAPS['PSAR'])
    strat_trades['TSMOM'] = bt_tsmom(h1_df, SPREAD, UNIT_LOT, CAPS['TSMOM'])
    strat_trades['SESS_BO'] = bt_sess_bo(h1_df, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
    portfolio_all = merge_portfolio_trades(strat_trades)
    print(f"  Portfolio: {len(portfolio_all)} trades", flush=True)

    # Part B
    part_b_results, (best_sigma_key, best_sigma_info) = part_b_sigma_sensitivity(
        h1_df, portfolio_all)

    # Part C
    results_c1, results_c2 = part_c_adaptive_sl(l8_bundle, h1_df)

    # K-Fold validation
    print("\n" + "=" * 70)
    print("  K-Fold Validation on Best Settings")
    print("=" * 70, flush=True)

    kfold_a = kfold_adx(l8_bundle, best_adx)
    kfold_b = kfold_sigma(l8_bundle, h1_df, best_sigma_key, best_sigma_info)

    # Also K-Fold baseline (ADX=14, current production)
    print(f"\n  K-Fold: Baseline (ADX=14)", flush=True)
    kfold_baseline = kfold_adx(l8_bundle, 14)

    # Save
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  R160 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}", flush=True)

    output = {
        'experiment': 'R160 Loss Protection Enhancements',
        'part_a_adx': {
            'results': {str(k): v for k, v in part_a_results.items()},
            'best_threshold': best_adx,
            'kfold': kfold_a,
            'kfold_baseline': kfold_baseline,
        },
        'part_b_sigma': {
            'results': part_b_results,
            'best_config': best_sigma_key,
            'best_info': best_sigma_info,
            'kfold': kfold_b,
        },
        'part_c_sl': {
            'fixed_dollar_cap': {str(k): v for k, v in results_c1.items()},
            'adaptive_lot': {str(k): v for k, v in results_c2.items()},
        },
        'elapsed_s': round(elapsed, 1),
    }

    with open(OUTPUT_DIR / "r160_results.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r160_results.json", flush=True)


if __name__ == "__main__":
    main()
