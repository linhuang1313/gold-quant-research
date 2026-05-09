#!/usr/bin/env python3
"""
R93 — Portfolio Re-Optimization with ML-Filtered Keltner (L8_MAX)
==================================================================
Based on R89 framework, but replaces raw L8_MAX with ML-filtered version.

R92-B showed L8_MAX ML Exit passes 5/5 robustness tests:
  - Walk-Forward AUC: 0.71-0.75
  - Sharpe: 4.40 → 5.77 (+31%)

This experiment re-runs the lot grid search with the ML-filtered L8_MAX
daily PnL to determine if Keltner deserves a larger lot allocation.

Phase 1: Backtest all strategies at unit lot (with ML filter for L8_MAX)
Phase 2: Grid search 15^4 = 50,625 lot combos
Phase 3: K-Fold validation on top 5 combos
Phase 4: Sensitivity analysis
Phase 5: Compare R89 vs R93 allocation
"""
import sys, os, io, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import product
from copy import deepcopy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r93_portfolio_ml_keltner")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
CAPITAL = 5000
MAX_DD_LIMIT = 1000
SPREAD = 0.30
UNIT_LOT = 0.01

CAPS = {
    'L8_MAX': 35,
    'PSAR':    5,
    'TSMOM':   0,
    'SESS_BO': 35,
}

LOT_GRID = [round(x * 0.01, 2) for x in range(1, 16)]

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]

STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']

ML_FEATURES = [
    'atr_14', 'adx_14', 'rsi_14', 'rsi_2',
    'kc_breakout_strength', 'volume_ratio', 'atr_percentile',
    'ema9_ema21_cross', 'close_ema100_dist',
    'hour_of_day', 'day_of_week', 'direction',
]


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


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


# ═══════════════════════════════════════════════════════════════
# Strategy backtests
# ═══════════════════════════════════════════════════════════════

def bt_psar(h1_df, spread, lot, maxloss_cap=0,
            sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04, max_hold=20):
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


def bt_tsmom(h1_df, spread, lot, maxloss_cap=0,
             fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
             trail_act=0.14, trail_dist=0.025, max_hold=20):
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


def bt_sess_bo(h1_df, spread, lot, maxloss_cap=0,
               session_hour=12, lookback=4, sl_atr=4.5, tp_atr=4.0,
               trail_act=0.14, trail_dist=0.025, max_hold=20):
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
    from backtest.runner import run_variant, LIVE_PARITY_KWARGS
    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': maxloss_cap,
           'spread_cost': spread, 'initial_capital': 2000,
           'min_lot_size': lot, 'max_lot_size': lot}
    result = run_variant(data_bundle, "L8_MAX", verbose=False, **kw)
    raw_trades = result.get('_trades', [])
    trades = []
    for t in raw_trades:
        trades.append({
            'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0,
        })
    return trades


# ═══════════════════════════════════════════════════════════════
# ML Filter for L8_MAX
# ═══════════════════════════════════════════════════════════════

def compute_h1_indicators(h1_df):
    df = h1_df.copy()
    df['ATR_14'] = compute_atr(df, 14)
    df['ADX_14'] = compute_adx(df, 14)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    rs_14 = gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.nan)
    df['RSI_14'] = 100 - 100 / (1 + rs_14)
    rs_2 = gain.rolling(2).mean() / loss.rolling(2).mean().replace(0, np.nan)
    df['RSI_2'] = 100 - 100 / (1 + rs_2)
    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    atr14 = df['ATR_14']
    kc_width = 2 * 1.5 * atr14
    df['KC_breakout_strength'] = (df['Close'] - ema20) / kc_width.replace(0, np.nan)
    range_vol = df['High'] - df['Low']
    df['Volume_ratio'] = range_vol / range_vol.rolling(20).mean().replace(0, np.nan)
    df['ATR_percentile'] = df['ATR_14'].rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    ema9 = df['Close'].ewm(span=9, adjust=False).mean()
    ema21 = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA9_EMA21_cross'] = (ema9 - ema21) / atr14.replace(0, np.nan)
    ema100 = df['Close'].ewm(span=100, adjust=False).mean()
    df['Close_EMA100_dist'] = (df['Close'] - ema100) / atr14.replace(0, np.nan)
    return df


def build_trade_features(trades, h1_indicators):
    """Build feature matrix for trades using H1 indicator data (tech + time only)."""
    h1_idx = h1_indicators.index
    if h1_idx.tz is not None:
        h1_indicators = h1_indicators.copy()
        h1_indicators.index = h1_idx.tz_localize(None)
    records = []; labels = []; valid_indices = []
    for idx, t in enumerate(trades):
        entry_time = pd.Timestamp(t['entry_time'])
        if entry_time.tzinfo is not None:
            entry_time = entry_time.tz_localize(None)
        if entry_time in h1_indicators.index:
            h1_row = h1_indicators.loc[entry_time]
        else:
            loc = h1_indicators.index.get_indexer([entry_time], method='ffill')
            if loc[0] < 0: continue
            h1_row = h1_indicators.iloc[loc[0]]
        feat = {
            'atr_14': h1_row.get('ATR_14', np.nan),
            'adx_14': h1_row.get('ADX_14', np.nan),
            'rsi_14': h1_row.get('RSI_14', np.nan),
            'rsi_2': h1_row.get('RSI_2', np.nan),
            'kc_breakout_strength': h1_row.get('KC_breakout_strength', np.nan),
            'volume_ratio': h1_row.get('Volume_ratio', np.nan),
            'atr_percentile': h1_row.get('ATR_percentile', np.nan),
            'ema9_ema21_cross': h1_row.get('EMA9_EMA21_cross', np.nan),
            'close_ema100_dist': h1_row.get('Close_EMA100_dist', np.nan),
            'hour_of_day': entry_time.hour,
            'day_of_week': entry_time.dayofweek,
            'direction': 1 if t['dir'] == 'BUY' else -1,
        }
        records.append(feat)
        labels.append(1 if t['pnl'] > 0 else 0)
        valid_indices.append(idx)
    if not records:
        return pd.DataFrame(), np.array([]), []
    return pd.DataFrame(records), np.array(labels), valid_indices


def get_xgb_model():
    try:
        import xgboost as xgb
        try:
            m = xgb.XGBClassifier(n_estimators=300, max_depth=5,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', tree_method='hist', device='cuda',
                random_state=42, verbosity=0)
            m.fit(np.random.randn(10, 5), np.random.randint(0, 2, 10))
            return xgb.XGBClassifier(n_estimators=300, max_depth=5,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', tree_method='hist', device='cuda',
                random_state=42, verbosity=0)
        except Exception:
            return xgb.XGBClassifier(n_estimators=300, max_depth=5,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', tree_method='hist', random_state=42, verbosity=0)
    except ImportError:
        return None


def ml_filter_l8_trades(trades, h1_indicators, threshold=0.50):
    """Apply walk-forward ML filter to L8_MAX trades.
    Returns (filtered_trades, oos_preds, valid_indices) so that
    precomputed predictions can be reused in K-Fold validation."""
    X, y, valid_indices = build_trade_features(trades, h1_indicators)
    if len(X) < 100:
        print(f"    [ML] Too few samples ({len(X)}), returning all trades")
        return trades, np.array([]), []

    entry_times = pd.Series([pd.Timestamp(trades[vi]['entry_time']) for vi in valid_indices])
    if entry_times.dt.tz is not None:
        entry_times = entry_times.dt.tz_localize(None)

    model = get_xgb_model()
    if model is None:
        print("    [ML] XGBoost not available, returning all trades")
        return trades, np.array([]), []

    X_use = X[ML_FEATURES].copy()
    oos_preds = np.full(len(y), np.nan)

    for fold_name, fold_start, fold_end in FOLDS:
        fs = pd.Timestamp(fold_start); fe = pd.Timestamp(fold_end)
        train_mask = entry_times < fs
        test_mask = (entry_times >= fs) & (entry_times < fe)
        if train_mask.sum() < 50 or test_mask.sum() < 20:
            continue
        Xtr = X_use[train_mask].copy(); ytr = y[train_mask.values]
        Xte = X_use[test_mask].copy()
        med = Xtr.median(); Xtr = Xtr.fillna(med); Xte = Xte.fillna(med)
        const = [c for c in Xtr.columns if Xtr[c].nunique() <= 1]
        if const:
            Xtr = Xtr.drop(columns=const); Xte = Xte.drop(columns=const)
        try:
            m = deepcopy(model); m.fit(Xtr, ytr)
            probs = m.predict_proba(Xte)[:, 1]
            oos_preds[np.where(test_mask)[0]] = probs
        except Exception as e:
            print(f"    [ML] Fold {fold_name} error: {e}")

    kept_indices = set()
    skipped = 0
    for i, vi in enumerate(valid_indices):
        if np.isnan(oos_preds[i]):
            kept_indices.add(vi)
        elif oos_preds[i] >= threshold:
            kept_indices.add(vi)
        else:
            skipped += 1

    for i in range(len(trades)):
        if i not in valid_indices:
            kept_indices.add(i)

    filtered = [trades[i] for i in sorted(kept_indices)]
    print(f"    [ML] L8_MAX: {len(trades)} -> {len(filtered)} trades "
          f"(skipped {skipped}, threshold={threshold})")
    return filtered, oos_preds, valid_indices


def filter_trades_by_precomputed(fold_trades, all_trades, oos_preds, valid_indices,
                                  threshold=0.50):
    """Filter fold trades using precomputed walk-forward predictions from full sample.

    Matches fold trades to all_trades by entry_time, looks up the OOS prediction
    probability, and skips trades below threshold.
    """
    if len(oos_preds) == 0 or len(valid_indices) == 0:
        return fold_trades

    pred_lookup = {}
    for i, vi in enumerate(valid_indices):
        t = all_trades[vi]
        et = pd.Timestamp(t['entry_time'])
        if et.tzinfo is not None:
            et = et.tz_localize(None)
        key = (et, t['dir'])
        pred_lookup[key] = oos_preds[i]

    kept = []
    skipped = 0
    no_match = 0
    for t in fold_trades:
        et = pd.Timestamp(t['entry_time'])
        if et.tzinfo is not None:
            et = et.tz_localize(None)
        key = (et, t['dir'])
        prob = pred_lookup.get(key)
        if prob is None or np.isnan(prob):
            kept.append(t)
            no_match += (prob is None)
        elif prob >= threshold:
            kept.append(t)
        else:
            skipped += 1

    print(f"    [ML-precomputed] {len(fold_trades)} -> {len(kept)} trades "
          f"(skipped {skipped}, no_match {no_match})")
    return kept


# ═══════════════════════════════════════════════════════════════
# Daily PnL helpers
# ═══════════════════════════════════════════════════════════════

def trades_to_daily_series(trades):
    if not trades:
        return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


def sharpe(arr):
    if len(arr) < 10:
        return 0.0
    s = np.std(arr, ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(arr) / s * np.sqrt(252))


def max_dd(arr):
    if len(arr) == 0:
        return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def cvar99(arr):
    if len(arr) < 20:
        return 0.0
    v = np.percentile(arr, 1)
    tail = arr[arr <= v]
    return float(tail.mean()) if len(tail) > 0 else float(v)


def build_portfolio_daily(unit_dailies, lots):
    all_dates = set()
    for ds in unit_dailies.values():
        all_dates.update(ds.index)
    all_dates = sorted(all_dates)
    idx = pd.DatetimeIndex(all_dates)
    portfolio = np.zeros(len(idx))
    for name in STRAT_ORDER:
        if name not in unit_dailies:
            continue
        ds = unit_dailies[name]
        multiplier = lots[name] / UNIT_LOT
        aligned = ds.reindex(idx, fill_value=0.0).values * multiplier
        portfolio += aligned
    return portfolio


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80)
    print("  R93 — Portfolio Re-Optimization with ML-Filtered Keltner")
    print(f"  Capital: ${CAPITAL:,}  |  MaxDD limit: ${MAX_DD_LIMIT:,}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80, flush=True)

    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH, DataBundle

    # ── Phase 1: Per-Strategy Backtest ──
    print(f"\n{'='*80}")
    print(f"  Phase 1: Per-Strategy Backtest (ML-filtered L8_MAX)")
    print(f"{'='*80}\n", flush=True)

    print("  Loading data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})", flush=True)

    print("  Computing H1 indicators for ML filter...", flush=True)
    h1_indicators = compute_h1_indicators(h1_df)
    print("  H1 indicators ready.", flush=True)

    print("  Preparing L8_MAX DataBundle...", flush=True)
    l8_bundle = DataBundle.load_custom()
    print("  L8 bundle ready.\n", flush=True)

    unit_trades = {}
    unit_dailies = {}
    unit_stats = {}

    # -- Run both raw and ML-filtered L8_MAX --
    print("  Running L8_MAX (raw)...", flush=True)
    l8_raw_trades = bt_l8_max(l8_bundle, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])
    l8_raw_daily = trades_to_daily_series(l8_raw_trades)
    pnls_raw = [t['pnl'] for t in l8_raw_trades]
    raw_arr = l8_raw_daily.values
    l8_raw_stats = {
        'n_trades': len(l8_raw_trades), 'pnl': round(sum(pnls_raw), 2),
        'sharpe': round(sharpe(raw_arr), 2), 'max_dd': round(max_dd(raw_arr), 2),
        'wr': round(sum(1 for p in pnls_raw if p > 0) / max(len(pnls_raw), 1) * 100, 1),
    }
    print(f"    L8_MAX raw: {l8_raw_stats['n_trades']} trades, "
          f"Sharpe={l8_raw_stats['sharpe']:.2f}", flush=True)

    print("  Applying ML Exit filter to L8_MAX...", flush=True)
    l8_filtered_trades, l8_oos_preds, l8_valid_indices = ml_filter_l8_trades(
        l8_raw_trades, h1_indicators, threshold=0.50)
    unit_trades['L8_MAX'] = l8_filtered_trades
    unit_dailies['L8_MAX'] = trades_to_daily_series(l8_filtered_trades)
    pnls = [t['pnl'] for t in l8_filtered_trades]
    daily_arr = unit_dailies['L8_MAX'].values
    unit_stats['L8_MAX'] = {
        'n_trades': len(l8_filtered_trades), 'pnl': round(sum(pnls), 2),
        'sharpe': round(sharpe(daily_arr), 2), 'max_dd': round(max_dd(daily_arr), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / max(len(pnls), 1) * 100, 1),
        'cap': CAPS['L8_MAX'], 'ml_filtered': True,
    }
    print(f"    L8_MAX ML:  {unit_stats['L8_MAX']['n_trades']} trades, "
          f"Sharpe={unit_stats['L8_MAX']['sharpe']:.2f} "
          f"(raw {l8_raw_stats['sharpe']:.2f} -> ML {unit_stats['L8_MAX']['sharpe']:.2f})", flush=True)

    # -- H1 strategies (unchanged) --
    h1_strats = {
        'PSAR':    (bt_psar, {}),
        'TSMOM':   (bt_tsmom, {}),
        'SESS_BO': (bt_sess_bo, {}),
    }
    for name, (fn, kw) in h1_strats.items():
        cap = CAPS[name]
        trades = fn(h1_df, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=cap, **kw)
        unit_trades[name] = trades
        unit_dailies[name] = trades_to_daily_series(trades)
        pnls = [t['pnl'] for t in trades]
        daily_arr = unit_dailies[name].values
        unit_stats[name] = {
            'n_trades': len(trades), 'pnl': round(sum(pnls), 2),
            'sharpe': round(sharpe(daily_arr), 2), 'max_dd': round(max_dd(daily_arr), 2),
            'wr': round(sum(1 for p in pnls if p > 0) / max(len(pnls), 1) * 100, 1),
            'cap': cap, 'ml_filtered': False,
        }
        print(f"    {name:>8}: {len(trades)} trades, Sharpe={unit_stats[name]['sharpe']:.2f}, "
              f"PnL={fmt(unit_stats[name]['pnl'])}", flush=True)

    print(f"\n  Phase 1 complete.", flush=True)

    # ── Phase 2: Grid Search ──
    print(f"\n{'='*80}")
    print(f"  Phase 2: Lot Grid Search ({len(LOT_GRID)**4:,} combos)")
    print(f"{'='*80}\n", flush=True)

    results = []
    total = len(LOT_GRID) ** 4
    checked = 0; feasible = 0

    for l8_lot, psar_lot, tsmom_lot, sess_lot in product(LOT_GRID, repeat=4):
        checked += 1
        if checked % 10000 == 0:
            print(f"    Progress: {checked:,}/{total:,}, feasible={feasible}...", flush=True)

        lots = {'L8_MAX': l8_lot, 'PSAR': psar_lot, 'TSMOM': tsmom_lot, 'SESS_BO': sess_lot}
        port_daily = build_portfolio_daily(unit_dailies, lots)

        dd = max_dd(port_daily)
        if dd > MAX_DD_LIMIT:
            continue
        feasible += 1
        sh = sharpe(port_daily)
        pnl = float(np.sum(port_daily))
        cv = cvar99(port_daily)

        results.append({
            'lots': lots,
            'sharpe': round(sh, 3),
            'pnl': round(pnl, 2),
            'max_dd': round(dd, 2),
            'cvar99': round(cv, 2),
            'dd_pct': round(dd / CAPITAL * 100, 1),
            'annual_return_pct': round(pnl / (2754/252) / CAPITAL * 100, 1),
        })

    print(f"\n  Grid complete: {checked:,} tested, {feasible:,} feasible", flush=True)
    results.sort(key=lambda x: x['sharpe'], reverse=True)

    print(f"\n  Top 20:")
    print(f"  {'Rank':>4} {'L8_MAX':>7} {'PSAR':>6} {'TSMOM':>6} {'SESSBO':>7} "
          f"{'Sharpe':>7} {'PnL':>12} {'MaxDD':>10} {'DD%':>6}")
    print(f"  {'-'*4} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*12} {'-'*10} {'-'*6}")
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:>4} {r['lots']['L8_MAX']:>7.2f} {r['lots']['PSAR']:>6.2f} "
              f"{r['lots']['TSMOM']:>6.2f} {r['lots']['SESS_BO']:>7.2f} "
              f"{r['sharpe']:>7.3f} {fmt(r['pnl']):>12} {fmt(r['max_dd']):>10} "
              f"{r['dd_pct']:>5.1f}%", flush=True)

    # ── Phase 3: K-Fold Validation ──
    print(f"\n{'='*80}")
    print(f"  Phase 3: K-Fold Validation (top 5)")
    print(f"{'='*80}\n", flush=True)

    fold_unit_dailies = {}
    for fold_name, start, end in FOLDS:
        fold_unit_dailies[fold_name] = {}
        h1_fold = h1_df[start:end]
        for name, (fn, kw) in h1_strats.items():
            cap = CAPS[name]
            trades = fn(h1_fold, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=cap, **kw)
            fold_unit_dailies[fold_name][name] = trades_to_daily_series(trades)

    # L8_MAX ML-filtered folds (use precomputed OOS predictions from Phase 1)
    for fold_name, start, end in FOLDS:
        try:
            l8_fold = l8_bundle.slice(start, end)
            fold_trades = bt_l8_max(l8_fold, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])
            fold_filtered = filter_trades_by_precomputed(
                fold_trades, l8_raw_trades, l8_oos_preds, l8_valid_indices, threshold=0.50)
            fold_unit_dailies[fold_name]['L8_MAX'] = trades_to_daily_series(fold_filtered)
        except Exception as e:
            print(f"    [WARN] L8_MAX fold {fold_name}: {e}", flush=True)
            fold_unit_dailies[fold_name]['L8_MAX'] = pd.Series(dtype=float)

    kfold_results = {}
    for rank, combo in enumerate(results[:5]):
        lots = combo['lots']
        label = f"L8={lots['L8_MAX']:.2f}_P={lots['PSAR']:.2f}_T={lots['TSMOM']:.2f}_S={lots['SESS_BO']:.2f}"
        fold_sharpes = []
        for fold_name, start, end in FOLDS:
            port_daily = build_portfolio_daily(fold_unit_dailies[fold_name], lots)
            fold_sharpes.append(sharpe(port_daily))
        positive = sum(1 for s in fold_sharpes if s > 0)
        mean_sh = float(np.mean(fold_sharpes))
        passed = positive >= 4
        kfold_results[label] = {
            'rank': rank + 1, 'lots': lots,
            'fold_sharpes': [round(s, 2) for s in fold_sharpes],
            'positive_folds': positive, 'mean_sharpe': round(mean_sh, 2),
            'pass_4of6': passed,
        }
        status = "PASS" if passed else "FAIL"
        print(f"  #{rank+1} {label}: {positive}/6 positive, mean={mean_sh:.2f} [{status}]", flush=True)
        print(f"       folds={[round(s, 1) for s in fold_sharpes]}", flush=True)

    # ── Phase 4: Sensitivity ──
    print(f"\n{'='*80}")
    print(f"  Phase 4: Sensitivity Analysis")
    print(f"{'='*80}\n", flush=True)

    winner = None
    for combo in results[:10]:
        lots = combo['lots']
        label = f"L8={lots['L8_MAX']:.2f}_P={lots['PSAR']:.2f}_T={lots['TSMOM']:.2f}_S={lots['SESS_BO']:.2f}"
        if label in kfold_results and kfold_results[label]['pass_4of6']:
            winner = combo; break
    if winner is None and results:
        winner = results[0]

    sensitivity = {}
    if winner:
        base_lots = winner['lots']
        print(f"  Winner: L8={base_lots['L8_MAX']:.2f}  PSAR={base_lots['PSAR']:.2f}  "
              f"TSMOM={base_lots['TSMOM']:.2f}  SESS_BO={base_lots['SESS_BO']:.2f}")
        print(f"  Sharpe={winner['sharpe']:.3f}  PnL={fmt(winner['pnl'])}  MaxDD={fmt(winner['max_dd'])}\n")

        print(f"  {'Strategy':<10} {'Lot-0.01':>10} {'BaseLot':>10} {'Lot+0.01':>10}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for name in STRAT_ORDER:
            row = {}
            for delta_label, delta in [('-0.01', -0.01), ('base', 0), ('+0.01', 0.01)]:
                test_lots = dict(base_lots)
                new_lot = round(test_lots[name] + delta, 2)
                if new_lot < 0.01 or new_lot > 0.20:
                    row[delta_label] = None; continue
                test_lots[name] = new_lot
                port_daily = build_portfolio_daily(unit_dailies, test_lots)
                row[delta_label] = round(sharpe(port_daily), 3)
            def sv(v): return f"{v:.3f}" if v is not None else "N/A"
            sensitivity[name] = row
            print(f"  {name:<10} {sv(row.get('-0.01')):>10} {sv(row.get('base')):>10} "
                  f"{sv(row.get('+0.01')):>10}", flush=True)

        print(f"\n  Per-strategy contribution:")
        print(f"  {'Strategy':<10} {'Lot':>6} {'PnL':>12} {'MaxDD':>10} {'Sharpe':>8} {'%ofPnL':>8}")
        print(f"  {'-'*10} {'-'*6} {'-'*12} {'-'*10} {'-'*8} {'-'*8}")
        total_pnl = 0; strat_pnls = {}
        for name in STRAT_ORDER:
            lot = base_lots[name]; mult = lot / UNIT_LOT
            scaled = unit_dailies[name].values * mult
            strat_pnls[name] = float(np.sum(scaled))
            total_pnl += strat_pnls[name]
        for name in STRAT_ORDER:
            lot = base_lots[name]; mult = lot / UNIT_LOT
            scaled = unit_dailies[name].values * mult
            pnl_v = strat_pnls[name]
            sh_v = sharpe(scaled); dd_v = max_dd(scaled)
            pct = pnl_v / total_pnl * 100 if total_pnl > 0 else 0
            print(f"  {name:<10} {lot:>6.2f} {fmt(pnl_v):>12} {fmt(dd_v):>10} "
                  f"{sh_v:>8.2f} {pct:>7.1f}%", flush=True)

    # ── Phase 5: R89 vs R93 comparison ──
    print(f"\n{'='*80}")
    print(f"  Phase 5: R89 vs R93 Comparison")
    print(f"{'='*80}\n", flush=True)

    r89_lots = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.08, 'SESS_BO': 0.08}
    r89_daily = build_portfolio_daily(unit_dailies, r89_lots)
    r89_sh = sharpe(r89_daily)
    r89_dd = max_dd(r89_daily)
    r89_pnl = float(np.sum(r89_daily))

    print(f"  {'':>12} {'R89 (old)':>15} {'R93 (new)':>15} {'Change':>10}")
    print(f"  {'-'*12} {'-'*15} {'-'*15} {'-'*10}")
    if winner:
        r93_lots = winner['lots']
        r93_sh = winner['sharpe']
        r93_dd = winner['max_dd']
        r93_pnl = winner['pnl']
        sh_chg = (r93_sh - r89_sh) / r89_sh * 100
        dd_chg = (r93_dd - r89_dd) / r89_dd * 100
        print(f"  {'Sharpe':>12} {r89_sh:>15.3f} {r93_sh:>15.3f} {sh_chg:>+9.1f}%")
        print(f"  {'MaxDD':>12} {fmt(r89_dd):>15} {fmt(r93_dd):>15} {dd_chg:>+9.1f}%")
        print(f"  {'PnL':>12} {fmt(r89_pnl):>15} {fmt(r93_pnl):>15}")
        print(f"\n  Lot changes:")
        for name in STRAT_ORDER:
            old = r89_lots[name]; new = r93_lots[name]
            arrow = "↑" if new > old else ("↓" if new < old else "=")
            print(f"    {name:<10}: {old:.2f} -> {new:.2f} {arrow}")

    # ── Summary ──
    print(f"\n{'='*80}")
    print(f"  FINAL RECOMMENDATION (R93)")
    print(f"{'='*80}\n")
    if winner:
        lots = winner['lots']
        print(f"  Capital: ${CAPITAL:,}")
        print(f"  MaxDD Limit: ${MAX_DD_LIMIT:,}")
        print(f"  ML Exit: Applied to L8_MAX only (R92-B validated, 5/5 PASS)")
        print(f"  ")
        print(f"  Recommended lot sizes:")
        for name in STRAT_ORDER:
            lot = lots[name]; cap = CAPS[name]
            cap_str = f"Cap${cap}" if cap > 0 else "NoCap"
            ml_str = " [ML-filtered]" if name == 'L8_MAX' else ""
            print(f"    {name:<10}  {lot:.2f} lot  ({cap_str}){ml_str}")
        print(f"  ")
        print(f"  Portfolio metrics:")
        print(f"    Sharpe:        {winner['sharpe']:.3f}")
        print(f"    Total PnL:     {fmt(winner['pnl'])}")
        print(f"    MaxDD:         {fmt(winner['max_dd'])}  ({winner['dd_pct']:.1f}%)")
        print(f"    CVaR99:        {fmt(winner['cvar99'])}")
        print(f"    Annual Return: {winner['annual_return_pct']:.1f}%")

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  R93 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}", flush=True)

    output = {
        'config': {
            'capital': CAPITAL, 'max_dd_limit': MAX_DD_LIMIT,
            'spread': SPREAD, 'caps': CAPS,
            'lot_grid': LOT_GRID, 'unit_lot': UNIT_LOT,
            'ml_filter': {'strategy': 'L8_MAX', 'threshold': 0.50,
                          'features': ML_FEATURES, 'model': 'XGBoost walk-forward'},
        },
        'unit_stats': unit_stats,
        'l8_raw_stats': l8_raw_stats,
        'top_20': results[:20],
        'kfold': kfold_results,
        'sensitivity': sensitivity,
        'winner': winner,
        'r89_comparison': {
            'r89_lots': r89_lots, 'r89_sharpe': round(r89_sh, 3),
            'r89_max_dd': round(r89_dd, 2), 'r89_pnl': round(r89_pnl, 2),
        },
        'total_feasible': feasible,
        'total_tested': total,
        'elapsed_s': round(elapsed, 1),
    }
    with open(OUTPUT_DIR / "r93_results.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r93_results.json", flush=True)


if __name__ == "__main__":
    main()
