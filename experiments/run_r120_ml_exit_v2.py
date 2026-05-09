#!/usr/bin/env python3
"""
R120 — ML Exit Optimization v2 (All Strategies)
=================================================
Extend R62's L8_MAX exit filter to PSAR, TSMOM, SESS_BO.
Uses intra-trade features to predict whether to exit early.

Phase 1: Generate trades for all 4 strategies
Phase 2: Build intra-trade feature matrix
Phase 3: Walk-forward XGBoost per strategy
Phase 4: Exit filter evaluation (threshold sweep)
Phase 5: K-Fold validation (5 folds)
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'xgboost', 'scikit-learn'])
    import xgboost as xgb

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

OUTPUT_DIR = Path("results/r120_ml_exit_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']

THRESHOLDS = [0.50, 0.55, 0.60, 0.65]

FOLDS = [
    ("Fold1", "2015-01-01", "2017-03-01"),
    ("Fold2", "2017-03-01", "2019-05-01"),
    ("Fold3", "2019-05-01", "2021-07-01"),
    ("Fold4", "2021-07-01", "2023-09-01"),
    ("Fold5", "2023-09-01", "2026-05-01"),
]

INTRA_FEATURE_NAMES = [
    'bars_held', 'unrealized_pnl_norm', 'atr_ratio',
    'distance_from_extreme', 'close_vs_sma20', 'close_vs_sma50',
    'hour_of_day', 'day_of_week',
]


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_psar(df, af_step=0.01, af_max=0.05):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    n = len(df)
    psar = np.empty(n); psar[:] = np.nan
    af = af_step; rising = True; ep = h[0]; psar[0] = l[0]
    for i in range(1, n):
        prev = psar[i-1]
        if rising:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], l[i-1], l[max(0, i-2)])
            if l[i] < psar[i]:
                rising = False; psar[i] = ep; ep = l[i]; af = af_step
            else:
                if h[i] > ep: ep = h[i]; af = min(af + af_step, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], h[i-1], h[max(0, i-2)])
            if h[i] > psar[i]:
                rising = True; psar[i] = ep; ep = h[i]; af = af_step
            else:
                if l[i] < ep: ep = l[i]; af = min(af + af_step, af_max)
    df['PSAR'] = psar


def _mk(pos, exit_px, exit_time, reason, bar_i, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_px,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_i - pos['bar']}


def _run_exit_with_cap(pos, i, hi, lo, cl, spread, lot, pv, times,
                       sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap):
    atr = pos['atr']
    sl = atr * sl_atr; tp = atr * tp_atr; bars = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_now = (cl - pos['entry'] - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if lo <= pos['entry'] - sl:
            return _mk(pos, pos['entry'] - sl, times[i], "SL", i, -sl * lot * pv)
        if hi >= pos['entry'] + tp:
            return _mk(pos, pos['entry'] + tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry']); extreme = max(extreme, hi); pos['extreme'] = extreme
        if extreme - pos['entry'] >= atr * trail_act:
            trail_price = extreme - atr * trail_dist
            if cl <= trail_price:
                return _mk(pos, trail_price, times[i], "Trail", i, (trail_price - pos['entry'] - spread) * lot * pv)
        if bars >= max_hold:
            return _mk(pos, cl, times[i], "TimeExit", i, pnl_now)
    else:
        pnl_now = (pos['entry'] - cl - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if hi >= pos['entry'] + sl:
            return _mk(pos, pos['entry'] + sl, times[i], "SL", i, -sl * lot * pv)
        if lo <= pos['entry'] - tp:
            return _mk(pos, pos['entry'] - tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry']); extreme = min(extreme, lo); pos['extreme'] = extreme
        if pos['entry'] - extreme >= atr * trail_act:
            trail_price = extreme + atr * trail_dist
            if cl >= trail_price:
                return _mk(pos, trail_price, times[i], "Trail", i, (pos['entry'] - trail_price - spread) * lot * pv)
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


def _compute_stats(trades):
    if not trades:
        return {'n_trades': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    return {
        'n_trades': n,
        'sharpe': round(_sharpe(daily), 3),
        'pnl': round(sum(pnls), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════
# Strategy backtests WITH bar-by-bar snapshots
# ═══════════════════════════════════════════════════════════════

def _collect_snapshots(pos, i, cl, hi, lo, atr_val, times, spread, lot, pv):
    """Record a snapshot for the current bar while position is open."""
    bars_held = i - pos['bar']
    if pos['dir'] == 'BUY':
        unrealized = (cl - pos['entry'] - spread) * lot * pv
        extreme = pos.get('extreme', pos['entry'])
        extreme = max(extreme, hi)
        extreme_pnl = (extreme - pos['entry'] - spread) * lot * pv
    else:
        unrealized = (pos['entry'] - cl - spread) * lot * pv
        extreme = pos.get('extreme', pos['entry'])
        extreme = min(extreme, lo)
        extreme_pnl = (pos['entry'] - extreme - spread) * lot * pv

    close_vs_entry_pct = (cl - pos['entry']) / pos['entry'] * 100 if pos['entry'] > 0 else 0.0
    ts = times[i]
    return {
        'bar_idx': i,
        'bars_held': bars_held,
        'unrealized_pnl': unrealized,
        'current_atr': atr_val,
        'entry_atr': pos['atr'],
        'close_vs_entry_pct': close_vs_entry_pct,
        'extreme_pnl': extreme_pnl,
        'hour': ts.hour if hasattr(ts, 'hour') else pd.Timestamp(ts).hour,
        'dow': ts.dayofweek if hasattr(ts, 'dayofweek') else pd.Timestamp(ts).dayofweek,
    }


def bt_psar_snap(h1_df, spread, lot, maxloss_cap=0,
                 sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04, max_hold=20):
    df = h1_df.copy(); add_psar(df); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR', 'PSAR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; psar = df['PSAR'].values; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999; snapshots_buf = []
    for i in range(1, n):
        if pos is not None:
            snapshots_buf.append(_collect_snapshots(pos, i, c[i], h[i], lo[i], atr[i], times, spread, lot, PV))
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                result['snapshots'] = snapshots_buf
                trades.append(result); pos = None; last_exit = i; snapshots_buf = []; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if psar[i-1] > c[i-1] and psar[i] < c[i]:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif psar[i-1] < c[i-1] and psar[i] > c[i]:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom_snap(h1_df, spread, lot, maxloss_cap=0,
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
    trades = []; pos = None; last_exit = -999; snapshots_buf = []
    for i in range(max_lb+1, n):
        if pos is not None:
            snapshots_buf.append(_collect_snapshots(pos, i, c[i], h[i], lo[i], atr[i], times, spread, lot, PV))
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                result['snapshots'] = snapshots_buf
                trades.append(result); pos = None; last_exit = i; snapshots_buf = []; continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                pnl = (c[i]-pos['entry']-spread)*lot*PV
                r = _mk(pos, c[i], times[i], "Reversal", i, pnl)
                r['snapshots'] = snapshots_buf
                trades.append(r); pos = None; last_exit = i; snapshots_buf = []; continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                pnl = (pos['entry']-c[i]-spread)*lot*PV
                r = _mk(pos, c[i], times[i], "Reversal", i, pnl)
                r['snapshots'] = snapshots_buf
                trades.append(r); pos = None; last_exit = i; snapshots_buf = []; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(score[i]) or np.isnan(score[i-1]): continue
        if score[i] > 0 and score[i-1] <= 0:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_sess_bo_snap(h1_df, spread, lot, maxloss_cap=0,
                    session_hour=12, lookback=4, sl_atr=4.5, tp_atr=4.0,
                    trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999; snapshots_buf = []
    for i in range(lookback, n):
        if pos is not None:
            snapshots_buf.append(_collect_snapshots(pos, i, c[i], h[i], lo[i], atr[i], times, spread, lot, PV))
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                result['snapshots'] = snapshots_buf
                trades.append(result); pos = None; last_exit = i; snapshots_buf = []; continue
            continue
        if hours[i] != session_hour: continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_l8_max_snap(data_bundle, h1_df, spread, lot, maxloss_cap=35):
    """Run L8_MAX via runner, then reconstruct bar-by-bar snapshots from H1."""
    from backtest.runner import run_variant, LIVE_PARITY_KWARGS
    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': maxloss_cap,
          'spread_cost': spread, 'initial_capital': 2000,
          'min_lot_size': lot, 'max_lot_size': lot}
    result = run_variant(data_bundle, "L8_MAX", verbose=False, **kw)
    raw_trades = result.get('_trades', [])

    h1_tz = h1_df.copy()
    if h1_tz.index.tz is not None:
        h1_tz.index = h1_tz.index.tz_localize(None)
    atr_series = compute_atr(h1_tz)
    c_arr = h1_tz['Close'].values
    h_arr = h1_tz['High'].values
    l_arr = h1_tz['Low'].values
    atr_arr = atr_series.values
    h1_idx = h1_tz.index

    trades = []
    for t in raw_trades:
        entry_ts = pd.Timestamp(t.entry_time)
        exit_ts = pd.Timestamp(t.exit_time)
        if hasattr(entry_ts, 'tz') and entry_ts.tz is not None:
            entry_ts = entry_ts.tz_localize(None)
        if hasattr(exit_ts, 'tz') and exit_ts.tz is not None:
            exit_ts = exit_ts.tz_localize(None)

        entry_bar = h1_idx.searchsorted(entry_ts)
        exit_bar = h1_idx.searchsorted(exit_ts)
        entry_bar = min(entry_bar, len(h1_idx) - 1)
        exit_bar = min(exit_bar, len(h1_idx) - 1)

        entry_price = t.entry_price
        direction = t.direction
        entry_atr = atr_arr[entry_bar] if not np.isnan(atr_arr[entry_bar]) else 1.0

        snapshots = []
        extreme_price = entry_price
        for bi in range(entry_bar + 1, exit_bar + 1):
            if bi >= len(c_arr):
                break
            cl = c_arr[bi]; hi = h_arr[bi]; lo_v = l_arr[bi]
            bars_held = bi - entry_bar
            if direction in ('BUY', 'LONG', 1):
                unrealized = (cl - entry_price - spread) * lot * PV
                extreme_price = max(extreme_price, hi)
                extreme_pnl = (extreme_price - entry_price - spread) * lot * PV
            else:
                unrealized = (entry_price - cl - spread) * lot * PV
                extreme_price = min(extreme_price, lo_v)
                extreme_pnl = (entry_price - extreme_price - spread) * lot * PV

            cur_atr = atr_arr[bi] if not np.isnan(atr_arr[bi]) else entry_atr
            snapshots.append({
                'bar_idx': bi,
                'bars_held': bars_held,
                'unrealized_pnl': unrealized,
                'current_atr': cur_atr,
                'entry_atr': entry_atr,
                'close_vs_entry_pct': (cl - entry_price) / entry_price * 100 if entry_price > 0 else 0,
                'extreme_pnl': extreme_pnl,
                'hour': h1_idx[bi].hour,
                'dow': h1_idx[bi].dayofweek,
            })

        trades.append({
            'dir': direction, 'entry': entry_price, 'exit': t.exit_price,
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'pnl': t.pnl, 'reason': t.exit_reason,
            'bars': exit_bar - entry_bar,
            'snapshots': snapshots,
        })
    return trades


# ═══════════════════════════════════════════════════════════════
# Phase 2: Build intra-trade feature matrix
# ═══════════════════════════════════════════════════════════════

def build_intra_trade_features(all_trades, h1_df):
    """Build feature matrix with one row per bar-snapshot per trade.

    Label: 1 if exiting NOW would be better than holding to actual exit
    (i.e., the remaining PnL from this bar to trade end is negative).
    """
    h1_tz = h1_df.copy()
    if h1_tz.index.tz is not None:
        h1_tz.index = h1_tz.index.tz_localize(None)
    sma20 = h1_tz['Close'].rolling(20).mean()
    sma50 = h1_tz['Close'].rolling(50).mean()
    h1_close = h1_tz['Close']

    rows = []
    for strat_name, trades in all_trades.items():
        for trade_idx, t in enumerate(trades):
            snaps = t.get('snapshots', [])
            if not snaps:
                continue
            final_pnl = t['pnl']

            for snap in snaps:
                bi = snap['bar_idx']
                unrealized = snap['unrealized_pnl']
                entry_atr = snap['entry_atr']
                if entry_atr <= 0:
                    entry_atr = 1.0

                remaining_pnl = final_pnl - unrealized
                label = 1 if remaining_pnl < 0 else 0

                extreme_pnl = snap['extreme_pnl']
                distance_from_extreme = (extreme_pnl - unrealized) / (entry_atr * UNIT_LOT * PV) if entry_atr > 0 else 0

                sma20_val = sma20.iloc[bi] if bi < len(sma20) else np.nan
                sma50_val = sma50.iloc[bi] if bi < len(sma50) else np.nan
                cl = h1_close.iloc[bi] if bi < len(h1_close) else np.nan

                close_vs_sma20 = (cl - sma20_val) / sma20_val if (not np.isnan(sma20_val) and sma20_val > 0) else 0
                close_vs_sma50 = (cl - sma50_val) / sma50_val if (not np.isnan(sma50_val) and sma50_val > 0) else 0

                rows.append({
                    'strategy': strat_name,
                    'trade_idx': trade_idx,
                    'entry_time': str(t['entry_time']),
                    'bars_held': snap['bars_held'],
                    'unrealized_pnl_norm': unrealized / (entry_atr * UNIT_LOT * PV) if entry_atr > 0 else 0,
                    'atr_ratio': snap['current_atr'] / entry_atr if entry_atr > 0 else 1.0,
                    'distance_from_extreme': distance_from_extreme,
                    'close_vs_sma20': close_vs_sma20,
                    'close_vs_sma50': close_vs_sma50,
                    'hour_of_day': snap['hour'],
                    'day_of_week': snap['dow'],
                    'label': label,
                    'unrealized_pnl_raw': unrealized,
                    'final_pnl': final_pnl,
                })

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in INTRA_FEATURE_NAMES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


# ═══════════════════════════════════════════════════════════════
# Phase 3: Walk-forward XGBoost per strategy
# ═══════════════════════════════════════════════════════════════

def train_exit_model(feat_df, strategy_name, n_splits=5):
    """Walk-forward XGBoost for a single strategy's intra-trade data."""
    sdf = feat_df[feat_df['strategy'] == strategy_name].copy()
    if len(sdf) < 200:
        return None, {'strategy': strategy_name, 'skip': True, 'reason': f'too_few_samples_{len(sdf)}'}

    sdf = sdf.sort_values('entry_time').reset_index(drop=True)
    X = sdf[INTRA_FEATURE_NAMES].values
    y = sdf['label'].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_aucs = []
    oos_probs = np.full(len(sdf), np.nan)

    for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            fold_aucs.append(np.nan)
            continue

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=42, verbosity=0,
        )
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        oos_probs[test_idx] = probs

        auc = roc_auc_score(y_te, probs)
        fold_aucs.append(round(auc, 4))

    valid_aucs = [a for a in fold_aucs if not np.isnan(a)]
    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    final_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, verbosity=0,
    )
    final_model.fit(X, y)

    importance = {}
    for fname, fval in zip(INTRA_FEATURE_NAMES, final_model.feature_importances_):
        importance[fname] = round(float(fval), 4)

    result = {
        'strategy': strategy_name,
        'n_samples': len(sdf),
        'class_balance': round(float(y.mean()), 4),
        'fold_aucs': fold_aucs,
        'mean_auc': round(mean_auc, 4),
        'feature_importance': importance,
    }
    return final_model, result


# ═══════════════════════════════════════════════════════════════
# Phase 4: Exit filter simulation
# ═══════════════════════════════════════════════════════════════

def simulate_exit_filter(all_trades, models, h1_df, threshold):
    """Simulate: at each bar during trade, if model says 'exit' with P > threshold, close early."""
    h1_tz = h1_df.copy()
    if h1_tz.index.tz is not None:
        h1_tz.index = h1_tz.index.tz_localize(None)
    sma20 = h1_tz['Close'].rolling(20).mean()
    sma50 = h1_tz['Close'].rolling(50).mean()
    h1_close = h1_tz['Close']

    filtered_trades = {}
    for strat_name, trades in all_trades.items():
        model = models.get(strat_name)
        new_trades = []
        for t in trades:
            snaps = t.get('snapshots', [])
            if model is None or not snaps:
                new_trades.append(t)
                continue

            entry_atr = snaps[0]['entry_atr'] if snaps else 1.0
            if entry_atr <= 0:
                entry_atr = 1.0

            early_exit = False
            for snap in snaps:
                bi = snap['bar_idx']
                sma20_val = sma20.iloc[bi] if bi < len(sma20) else np.nan
                sma50_val = sma50.iloc[bi] if bi < len(sma50) else np.nan
                cl = h1_close.iloc[bi] if bi < len(h1_close) else np.nan

                extreme_pnl = snap['extreme_pnl']
                unrealized = snap['unrealized_pnl']
                distance_from_extreme = (extreme_pnl - unrealized) / (entry_atr * UNIT_LOT * PV) if entry_atr > 0 else 0

                feats = np.array([[
                    snap['bars_held'],
                    unrealized / (entry_atr * UNIT_LOT * PV) if entry_atr > 0 else 0,
                    snap['current_atr'] / entry_atr if entry_atr > 0 else 1.0,
                    distance_from_extreme,
                    (cl - sma20_val) / sma20_val if (not np.isnan(sma20_val) and sma20_val > 0) else 0,
                    (cl - sma50_val) / sma50_val if (not np.isnan(sma50_val) and sma50_val > 0) else 0,
                    snap['hour'],
                    snap['dow'],
                ]])
                feats = np.nan_to_num(feats, nan=0.0)
                prob = model.predict_proba(feats)[0, 1]
                if prob > threshold:
                    new_t = dict(t)
                    new_t['pnl'] = unrealized
                    new_t['reason'] = 'ML_Exit'
                    new_t['bars'] = snap['bars_held']
                    new_trades.append(new_t)
                    early_exit = True
                    break

            if not early_exit:
                new_trades.append(t)
        filtered_trades[strat_name] = new_trades
    return filtered_trades


# ═══════════════════════════════════════════════════════════════
# Phase 5: K-Fold validation
# ═══════════════════════════════════════════════════════════════

def kfold_validation(h1_df, bundle, best_threshold, n_folds=5):
    """Time-based K-fold: for each fold, train ML on out-of-fold data, apply exit filter, measure Sharpe."""
    results = []

    for fold_i, (fold_name, fold_start, fold_end) in enumerate(FOLDS[:n_folds]):
        print(f"    {fold_name} ({fold_start} ~ {fold_end})...", flush=True)

        h1_train_parts = []
        for j, (_, fs, fe) in enumerate(FOLDS[:n_folds]):
            if j == fold_i:
                continue
            part = h1_df[(h1_df.index >= fs) & (h1_df.index < fe)]
            if len(part) > 0:
                h1_train_parts.append(part)

        if not h1_train_parts:
            results.append({'fold': fold_name, 'skip': True})
            continue

        h1_train = pd.concat(h1_train_parts).sort_index()
        h1_test = h1_df[(h1_df.index >= fold_start) & (h1_df.index < fold_end)]
        if len(h1_test) < 100:
            results.append({'fold': fold_name, 'skip': True})
            continue

        train_trades = {}
        train_trades['PSAR'] = bt_psar_snap(h1_train, SPREAD, UNIT_LOT, CAPS['PSAR'])
        train_trades['TSMOM'] = bt_tsmom_snap(h1_train, SPREAD, UNIT_LOT, CAPS['TSMOM'])
        train_trades['SESS_BO'] = bt_sess_bo_snap(h1_train, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
        train_trades['L8_MAX'] = bt_l8_max_snap(bundle, h1_train, SPREAD, UNIT_LOT, CAPS['L8_MAX'])

        train_feat_df = build_intra_trade_features(train_trades, h1_train)

        fold_models = {}
        for sn in STRAT_ORDER:
            sdf = train_feat_df[train_feat_df['strategy'] == sn]
            if len(sdf) < 100:
                continue
            X = sdf[INTRA_FEATURE_NAMES].values
            y_arr = sdf['label'].values
            if len(np.unique(y_arr)) < 2:
                continue
            m = xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', random_state=42, verbosity=0,
            )
            m.fit(X, y_arr)
            fold_models[sn] = m

        test_trades = {}
        test_trades['PSAR'] = bt_psar_snap(h1_test, SPREAD, UNIT_LOT, CAPS['PSAR'])
        test_trades['TSMOM'] = bt_tsmom_snap(h1_test, SPREAD, UNIT_LOT, CAPS['TSMOM'])
        test_trades['SESS_BO'] = bt_sess_bo_snap(h1_test, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
        test_trades['L8_MAX'] = bt_l8_max_snap(bundle, h1_test, SPREAD, UNIT_LOT, CAPS['L8_MAX'])

        base_all = []
        for sn in STRAT_ORDER:
            base_all.extend(test_trades.get(sn, []))
        base_stats = _compute_stats(base_all)

        filtered = simulate_exit_filter(test_trades, fold_models, h1_test, best_threshold)
        filt_all = []
        for sn in STRAT_ORDER:
            filt_all.extend(filtered.get(sn, []))
        filt_stats = _compute_stats(filt_all)

        per_strat_base = {}
        per_strat_filt = {}
        for sn in STRAT_ORDER:
            per_strat_base[sn] = _compute_stats(test_trades.get(sn, []))
            per_strat_filt[sn] = _compute_stats(filtered.get(sn, []))

        results.append({
            'fold': fold_name,
            'base_sharpe': base_stats['sharpe'],
            'filt_sharpe': filt_stats['sharpe'],
            'base_pnl': base_stats['pnl'],
            'filt_pnl': filt_stats['pnl'],
            'per_strategy_base': per_strat_base,
            'per_strategy_filt': per_strat_filt,
        })

        print(f"      Base Sharpe={base_stats['sharpe']:.3f}, Filtered Sharpe={filt_stats['sharpe']:.3f}", flush=True)

    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("  R120 — ML Exit Optimization v2 (All Strategies)", flush=True)
    print("=" * 80, flush=True)

    # ── Load data ──
    print("\n  Loading data...", flush=True)
    from backtest.runner import DataBundle, load_m15, load_h1_aligned, H1_CSV_PATH

    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    bundle = DataBundle.load_custom()
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0].date()} ~ {h1_df.index[-1].date()})", flush=True)

    results = {'experiment': 'R120 ML Exit Optimization v2 (All Strategies)'}

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Generate trades with bar-by-bar snapshots
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 1: Generate trades with bar-by-bar snapshots", flush=True)
    print("=" * 70, flush=True)

    all_trades = {}

    print("    PSAR...", end=" ", flush=True)
    all_trades['PSAR'] = bt_psar_snap(h1_df, SPREAD, UNIT_LOT, CAPS['PSAR'])
    print(f"{len(all_trades['PSAR'])} trades, "
          f"{sum(len(t.get('snapshots', [])) for t in all_trades['PSAR'])} snapshots", flush=True)

    print("    TSMOM...", end=" ", flush=True)
    all_trades['TSMOM'] = bt_tsmom_snap(h1_df, SPREAD, UNIT_LOT, CAPS['TSMOM'])
    print(f"{len(all_trades['TSMOM'])} trades, "
          f"{sum(len(t.get('snapshots', [])) for t in all_trades['TSMOM'])} snapshots", flush=True)

    print("    SESS_BO...", end=" ", flush=True)
    all_trades['SESS_BO'] = bt_sess_bo_snap(h1_df, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
    print(f"{len(all_trades['SESS_BO'])} trades, "
          f"{sum(len(t.get('snapshots', [])) for t in all_trades['SESS_BO'])} snapshots", flush=True)

    print("    L8_MAX...", end=" ", flush=True)
    all_trades['L8_MAX'] = bt_l8_max_snap(bundle, h1_df, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
    print(f"{len(all_trades['L8_MAX'])} trades, "
          f"{sum(len(t.get('snapshots', [])) for t in all_trades['L8_MAX'])} snapshots", flush=True)

    phase1 = {}
    for sn in STRAT_ORDER:
        trades = all_trades[sn]
        stats = _compute_stats(trades)
        n_snaps = sum(len(t.get('snapshots', [])) for t in trades)
        phase1[sn] = {**stats, 'total_snapshots': n_snaps}
        print(f"    {sn:10s}: {stats['n_trades']:>5} trades, Sharpe={stats['sharpe']:.3f}, "
              f"WR={stats['wr']:.1f}%, PnL=${stats['pnl']:,.2f}, snaps={n_snaps}", flush=True)
    results['phase1'] = phase1

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Build intra-trade feature matrix
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: Build intra-trade feature matrix", flush=True)
    print("=" * 70, flush=True)

    feat_df = build_intra_trade_features(all_trades, h1_df)
    print(f"    Total samples (bar-snapshots): {len(feat_df)}", flush=True)
    print(f"    Features: {INTRA_FEATURE_NAMES}", flush=True)

    for sn in STRAT_ORDER:
        sdf = feat_df[feat_df['strategy'] == sn]
        print(f"    {sn:10s}: {len(sdf):>7} samples, label_mean={sdf['label'].mean():.3f}", flush=True)

    results['phase2'] = {
        'total_samples': len(feat_df),
        'features': INTRA_FEATURE_NAMES,
        'per_strategy': {sn: {'n_samples': int((feat_df['strategy'] == sn).sum()),
                              'label_mean': round(float(feat_df[feat_df['strategy'] == sn]['label'].mean()), 4)}
                         for sn in STRAT_ORDER},
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Walk-forward XGBoost per strategy
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3: Walk-forward XGBoost per strategy", flush=True)
    print("=" * 70, flush=True)

    models = {}
    phase3 = {}
    for sn in STRAT_ORDER:
        print(f"\n    {sn}:", flush=True)
        model, res = train_exit_model(feat_df, sn, n_splits=5)
        phase3[sn] = res
        if model is not None:
            models[sn] = model
            print(f"      Samples: {res['n_samples']}, Class balance: {res['class_balance']:.3f}", flush=True)
            print(f"      Fold AUCs: {res['fold_aucs']}", flush=True)
            print(f"      Mean AUC:  {res['mean_auc']:.4f}", flush=True)
            sorted_imp = sorted(res['feature_importance'].items(), key=lambda x: -x[1])
            print(f"      Top features: {', '.join(f'{k}={v:.3f}' for k, v in sorted_imp[:4])}", flush=True)
        else:
            print(f"      SKIPPED: {res.get('reason', 'unknown')}", flush=True)

    results['phase3'] = phase3

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Exit filter evaluation (threshold sweep)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 4: Exit filter evaluation (threshold sweep)", flush=True)
    print("=" * 70, flush=True)

    baseline_all = []
    for sn in STRAT_ORDER:
        baseline_all.extend(all_trades[sn])
    baseline_stats = _compute_stats(baseline_all)
    print(f"\n    Baseline (no filter): {baseline_stats['n_trades']} trades, "
          f"Sharpe={baseline_stats['sharpe']:.3f}, PnL=${baseline_stats['pnl']:,.2f}", flush=True)

    print(f"\n    {'Threshold':>10} {'Sharpe':>8} {'PnL':>12} {'Trades':>8} {'ML_Exits':>10}", flush=True)
    print(f"    {'-'*10} {'-'*8} {'-'*12} {'-'*8} {'-'*10}", flush=True)

    threshold_results = {}
    best_threshold = THRESHOLDS[0]
    best_sharpe = -999

    for thr in THRESHOLDS:
        filtered = simulate_exit_filter(all_trades, models, h1_df, thr)
        filt_all = []
        ml_exits = 0
        for sn in STRAT_ORDER:
            for t in filtered.get(sn, []):
                filt_all.append(t)
                if t.get('reason') == 'ML_Exit':
                    ml_exits += 1

        filt_stats = _compute_stats(filt_all)
        threshold_results[str(thr)] = {
            **filt_stats,
            'ml_exits': ml_exits,
            'delta_sharpe': round(filt_stats['sharpe'] - baseline_stats['sharpe'], 3),
        }

        print(f"    {thr:>10.2f} {filt_stats['sharpe']:>8.3f} ${filt_stats['pnl']:>11,.2f} "
              f"{filt_stats['n_trades']:>8} {ml_exits:>10}", flush=True)

        if filt_stats['sharpe'] > best_sharpe:
            best_sharpe = filt_stats['sharpe']
            best_threshold = thr

    # Per-strategy breakdown at best threshold
    print(f"\n    Best threshold: {best_threshold:.2f} (Sharpe={best_sharpe:.3f}, "
          f"Δ={best_sharpe - baseline_stats['sharpe']:+.3f})", flush=True)

    filtered_best = simulate_exit_filter(all_trades, models, h1_df, best_threshold)
    print(f"\n    Per-strategy at threshold={best_threshold:.2f}:", flush=True)
    print(f"    {'Strategy':<10} {'Base Shp':>9} {'Filt Shp':>9} {'Delta':>7} {'Base PnL':>12} {'Filt PnL':>12}", flush=True)
    print(f"    {'-'*10} {'-'*9} {'-'*9} {'-'*7} {'-'*12} {'-'*12}", flush=True)

    per_strat_phase4 = {}
    for sn in STRAT_ORDER:
        base_s = _compute_stats(all_trades[sn])
        filt_s = _compute_stats(filtered_best.get(sn, []))
        delta = round(filt_s['sharpe'] - base_s['sharpe'], 3)
        per_strat_phase4[sn] = {
            'base': base_s, 'filtered': filt_s, 'delta_sharpe': delta,
        }
        print(f"    {sn:<10} {base_s['sharpe']:>9.3f} {filt_s['sharpe']:>9.3f} {delta:>+7.3f} "
              f"${base_s['pnl']:>11,.2f} ${filt_s['pnl']:>11,.2f}", flush=True)

    results['phase4'] = {
        'baseline': baseline_stats,
        'threshold_sweep': threshold_results,
        'best_threshold': best_threshold,
        'best_sharpe': round(best_sharpe, 3),
        'per_strategy': per_strat_phase4,
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 5: K-Fold validation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 5: K-Fold validation (5 folds)", flush=True)
    print("=" * 70, flush=True)

    kfold_results = kfold_validation(h1_df, bundle, best_threshold, n_folds=5)

    valid_folds = [f for f in kfold_results if not f.get('skip')]
    base_sharpes = [f['base_sharpe'] for f in valid_folds]
    filt_sharpes = [f['filt_sharpe'] for f in valid_folds]

    print(f"\n    {'Fold':<8} {'Base Shp':>9} {'Filt Shp':>9} {'Delta':>7}", flush=True)
    print(f"    {'-'*8} {'-'*9} {'-'*9} {'-'*7}", flush=True)
    for f in kfold_results:
        if f.get('skip'):
            print(f"    {f['fold']:<8} SKIPPED", flush=True)
            continue
        delta = round(f['filt_sharpe'] - f['base_sharpe'], 3)
        print(f"    {f['fold']:<8} {f['base_sharpe']:>9.3f} {f['filt_sharpe']:>9.3f} {delta:>+7.3f}", flush=True)

    filt_better = sum(1 for b, f in zip(base_sharpes, filt_sharpes) if f > b)
    mean_base = float(np.mean(base_sharpes)) if base_sharpes else 0
    mean_filt = float(np.mean(filt_sharpes)) if filt_sharpes else 0

    print(f"\n    Mean Base Sharpe:     {mean_base:.3f}", flush=True)
    print(f"    Mean Filtered Sharpe: {mean_filt:.3f}", flush=True)
    print(f"    Filter better in {filt_better}/{len(valid_folds)} folds", flush=True)

    results['phase5'] = {
        'folds': kfold_results,
        'mean_base_sharpe': round(mean_base, 3),
        'mean_filt_sharpe': round(mean_filt, 3),
        'filt_better_count': filt_better,
        'total_valid_folds': len(valid_folds),
    }

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    print(f"\n{'='*80}", flush=True)
    print(f"  R120 SUMMARY — ML Exit Optimization v2", flush=True)
    print(f"{'='*80}", flush=True)

    print(f"\n  Per-strategy AUC:", flush=True)
    for sn in STRAT_ORDER:
        p3 = phase3.get(sn, {})
        auc = p3.get('mean_auc', 'N/A')
        print(f"    {sn:10s}: AUC={auc}", flush=True)

    print(f"\n  Best exit threshold: {best_threshold:.2f}", flush=True)
    print(f"  Baseline Sharpe:     {baseline_stats['sharpe']:.3f}", flush=True)
    print(f"  Filtered Sharpe:     {best_sharpe:.3f} (Δ={best_sharpe - baseline_stats['sharpe']:+.3f})", flush=True)
    print(f"  K-Fold:              filter better in {filt_better}/{len(valid_folds)} folds", flush=True)
    print(f"  Mean fold Sharpe:    {mean_base:.3f} → {mean_filt:.3f}", flush=True)

    if best_sharpe > baseline_stats['sharpe'] and filt_better >= 3:
        verdict = (f"ML exit filter v2 improves Sharpe {baseline_stats['sharpe']:.3f} → {best_sharpe:.3f} "
                   f"at threshold={best_threshold:.2f}, validated in {filt_better}/{len(valid_folds)} folds")
    elif best_sharpe > baseline_stats['sharpe']:
        verdict = "ML exit filter shows promise but inconsistent across folds — needs more tuning"
    else:
        verdict = "ML exit filter v2 does not improve over baseline — intra-trade signals insufficient"

    print(f"\n  VERDICT: {verdict}", flush=True)
    results['verdict'] = verdict
    results['elapsed_s'] = round(elapsed, 1)

    out_file = OUTPUT_DIR / "r120_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"  Saved: {out_file}", flush=True)
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
