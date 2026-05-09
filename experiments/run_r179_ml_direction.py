#!/usr/bin/env python3
"""
R179 — ML Direction Predictor: 2-class up/down entry signal for XAUUSD H1
"""

import sys
import os
import json
import warnings
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]

XGB_PARAMS = {
    'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 200,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 10,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42,
    'eval_metric': 'logloss', 'verbosity': 0, 'use_label_encoder': False,
}

THRESHOLDS = [0.55, 0.60, 0.65, 0.70]


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


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'entry_bar': pos['bar'],
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
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    return {
        'n': n,
        'sharpe': round(_sharpe(daily), 3),
        'pnl': round(sum(pnls), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1) if n > 0 else 0,
        'max_dd': round(_max_dd(daily), 2),
    }


def load_h1():
    candidates = sorted(glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    if not candidates:
        raise FileNotFoundError("No xauusd H1 CSV found")
    csv_path = candidates[-1]
    print(f"  Loading H1 from: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.set_index('time')[['open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[df['Close'] > 100]
    return df


def load_macro():
    macro_path = Path("data/external/aligned_daily.csv")
    if not macro_path.exists():
        print("  [WARN] Macro data not found, macro features disabled", flush=True)
        return None
    df = pd.read_csv(macro_path, parse_dates=['Date'])
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.set_index('Date')
    col_map = {
        'REAL_YIELD_Change5': 'real_yield_change5',
        'REAL_YIELD_Change20': 'real_yield_change20',
        'VIX_Zscore': 'vix_zscore',
        'VIX_Close': 'vix_close',
        'DXY_Mom5': 'dxy_mom5',
        'DXY_Mom20': 'dxy_mom20',
        'CREDIT_STRESS': 'credit_stress',
        'YIELD_CURVE_10Y2Y': 'yield_curve_10y2y',
        'COPPER_GOLD_RATIO': 'copper_gold_ratio',
        'CRUDE_Mom5': 'crude_mom5',
        'USDCNH_Mom5': 'usdcnh_mom5',
        'COT_MM_Net_Zscore': 'cot_mm_net_zscore',
        'FED_FUNDS_DFF': 'fed_funds_dff',
        'RISK_APPETITE_Z': 'risk_appetite_z',
    }
    rename = {}
    for orig, target in col_map.items():
        if orig in df.columns:
            rename[orig] = target
    df = df.rename(columns=rename)
    print(f"  Macro data: {len(df)} rows ({df.index[0].date()} -> {df.index[-1].date()})", flush=True)
    return df


# ═══════════════════════════════════════════════════════════════
# Phase 1: Feature Engineering
# ═══════════════════════════════════════════════════════════════

def build_features(h1_df, macro_df=None):
    df = h1_df.copy()

    df['ATR'] = compute_atr(df, 14)
    df['ADX'] = compute_adx(df, 14)
    df['RSI_14'] = compute_rsi(df['Close'], 14)
    df['RSI_2'] = compute_rsi(df['Close'], 2)

    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

    kc_mid = df['Close'].ewm(span=25, adjust=False).mean()
    kc_upper = kc_mid + 1.2 * df['ATR']
    kc_lower = kc_mid - 1.2 * df['ATR']
    kc_range = kc_upper - kc_lower
    kc_range = kc_range.replace(0, np.nan)

    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    feat = pd.DataFrame(index=df.index)

    feat['atr_14'] = df['ATR']
    feat['adx_14'] = df['ADX']
    feat['rsi_14'] = df['RSI_14']
    feat['rsi_2'] = df['RSI_2']

    feat['kc_breakout_strength'] = (df['Close'] - kc_mid) / kc_range
    feat['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, np.nan)
    feat['atr_percentile'] = df['ATR'].rolling(500, min_periods=100).rank(pct=True)
    feat['ema9_ema21_cross'] = (df['EMA9'] - df['EMA21']) / df['ATR'].replace(0, np.nan)
    feat['close_ema100_dist'] = (df['Close'] - df['EMA100']) / df['ATR'].replace(0, np.nan)

    macd_line = df['EMA12'] - df['EMA26']
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    feat['macd_hist'] = macd_line - macd_signal

    feat['squeeze'] = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype(int)
    feat['bb_width'] = (bb_upper - bb_lower) / df['Close'].replace(0, np.nan)

    feat['returns_1h'] = np.log(df['Close'] / df['Close'].shift(1))
    feat['returns_4h'] = np.log(df['Close'] / df['Close'].shift(4))
    feat['returns_12h'] = np.log(df['Close'] / df['Close'].shift(12))
    feat['returns_24h'] = np.log(df['Close'] / df['Close'].shift(24))

    h4_range = df['High'].rolling(4).max() - df['Low'].rolling(4).min()
    feat['high_low_range_4h'] = h4_range / df['ATR'].replace(0, np.nan)

    total_range = (df['High'] - df['Low']).replace(0, np.nan)
    feat['upper_wick_ratio'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / total_range
    feat['lower_wick_ratio'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / total_range
    feat['body_ratio'] = (df['Close'] - df['Open']).abs() / total_range

    close_arr = df['Close'].values
    slope_20 = np.full(len(close_arr), np.nan)
    xx = np.arange(20, dtype=float)
    xx_mean = xx.mean()
    xx_var = ((xx - xx_mean) ** 2).sum()
    for i in range(19, len(close_arr)):
        yy = close_arr[i-19:i+1]
        if np.any(np.isnan(yy)):
            continue
        slope_20[i] = ((xx - xx_mean) * (yy - yy.mean())).sum() / xx_var
    feat['trend_strength_20'] = slope_20 / df['ATR'].replace(0, np.nan).values

    feat['hour_of_day'] = df.index.hour
    feat['day_of_week'] = df.index.dayofweek

    if macro_df is not None:
        macro_cols = [c for c in macro_df.columns if c in [
            'real_yield_change5', 'real_yield_change20', 'vix_zscore', 'vix_close',
            'dxy_mom5', 'dxy_mom20', 'credit_stress', 'yield_curve_10y2y',
            'copper_gold_ratio', 'crude_mom5', 'usdcnh_mom5', 'cot_mm_net_zscore',
            'fed_funds_dff', 'risk_appetite_z',
        ]]
        if macro_cols:
            macro_daily = macro_df[macro_cols].copy()
            macro_daily = macro_daily.reindex(
                pd.date_range(macro_daily.index.min(), df.index.max(), freq='h')
            ).ffill()
            for col in macro_cols:
                if col in macro_daily.columns:
                    feat[col] = macro_daily[col].reindex(df.index).ffill()

    return feat


# ═══════════════════════════════════════════════════════════════
# Phase 2: Label Construction
# ═══════════════════════════════════════════════════════════════

def build_labels(h1_df, atr_series, horizon=4, threshold_atr=0.5):
    future_close = h1_df['Close'].shift(-horizon)
    future_return = future_close - h1_df['Close']
    threshold = threshold_atr * atr_series

    label = pd.Series(np.nan, index=h1_df.index)
    label[future_return > threshold] = 1
    label[future_return < -threshold] = 0

    return label


# ═══════════════════════════════════════════════════════════════
# Phase 3: Walk-Forward XGBoost Training
# ═══════════════════════════════════════════════════════════════

def walk_forward_train(features, labels, folds, xgb_params):
    results = []
    all_preds = pd.DataFrame(index=features.index, columns=['proba', 'fold'])

    for fold_name, fold_start, fold_end in folds:
        train_mask = features.index < fold_start
        test_mask = (features.index >= fold_start) & (features.index < fold_end)

        valid_train = train_mask & labels.notna()
        valid_test = test_mask & labels.notna()

        X_train = features.loc[valid_train].values
        y_train = labels.loc[valid_train].values
        X_test = features.loc[valid_test].values
        y_test = labels.loc[valid_test].values

        if len(X_train) < 200 or len(X_test) < 50:
            print(f"  {fold_name}: SKIP (train={len(X_train)}, test={len(X_test)})", flush=True)
            results.append({'fold': fold_name, 'auc': None, 'acc': None, 'n_train': len(X_train),
                            'n_test': len(X_test), 'importance': {}})
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_s = np.nan_to_num(X_test_s, nan=0.0, posinf=0.0, neginf=0.0)

        model = XGBClassifier(**xgb_params)
        model.fit(X_train_s, y_train)

        proba = model.predict_proba(X_test_s)[:, 1]
        pred = (proba > 0.5).astype(int)

        auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, pred)

        imp = dict(zip(features.columns, model.feature_importances_))
        imp = dict(sorted(imp.items(), key=lambda x: -x[1]))

        print(f"  {fold_name}: AUC={auc:.4f}  Acc={acc:.4f}  "
              f"train={len(X_train)}  test={len(X_test)}  "
              f"class_bal={y_test.mean():.3f}", flush=True)

        all_preds.loc[valid_test, 'proba'] = proba
        all_preds.loc[valid_test, 'fold'] = fold_name

        results.append({
            'fold': fold_name, 'auc': round(auc, 4), 'acc': round(acc, 4),
            'n_train': int(len(X_train)), 'n_test': int(len(X_test)),
            'class_balance': round(float(y_test.mean()), 3),
            'importance': {k: round(float(v), 5) for k, v in list(imp.items())[:15]},
        })

    return results, all_preds


# ═══════════════════════════════════════════════════════════════
# Phase 4: Signal-to-Trade Backtest
# ═══════════════════════════════════════════════════════════════

def backtest_ml_direction(h1_df, predictions, threshold,
                          sl_atr=3.5, tp_atr=8.0,
                          trail_act=0.14, trail_dist=0.025,
                          max_hold=20, maxloss_cap=35):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df, 14)
    df = df.dropna(subset=['ATR'])

    proba = predictions['proba'].reindex(df.index).astype(float)
    valid_mask = proba.notna()

    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    proba_vals = proba.values

    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], SPREAD, UNIT_LOT, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i
            continue

        if i - last_exit < 2:
            continue
        if not valid_mask.iloc[i]:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue

        p = proba_vals[i]
        if np.isnan(p):
            continue

        if p > threshold:
            pos = {'dir': 'BUY', 'entry': c[i] + SPREAD / 2,
                   'bar': i, 'time': times[i], 'atr': atr[i]}
        elif (1 - p) > threshold:
            pos = {'dir': 'SELL', 'entry': c[i] - SPREAD / 2,
                   'bar': i, 'time': times[i], 'atr': atr[i]}

    return trades


def bt_l8_max(h1_df, spread, lot, maxloss_cap=35,
              adx_th=14, ema_period=25, kc_mult=1.2,
              sl_atr=3.5, tp_atr=8.0,
              trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df['ADX'] = compute_adx(df)
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['KC_mid'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    df['KC_upper'] = df['KC_mid'] + kc_mult * df['ATR']
    df['KC_lower'] = df['KC_mid'] - kc_mult * df['ATR']
    df = df.dropna(subset=['ATR', 'ADX', 'EMA100', 'KC_upper'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; adx = df['ADX'].values
    ema100 = df['EMA100'].values
    kc_u = df['KC_upper'].values; kc_l = df['KC_lower'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1 or np.isnan(adx[i]):
            continue
        if adx[i] < adx_th:
            continue
        if c[i] > kc_u[i] and c[i] > ema100[i]:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < kc_l[i] and c[i] < ema100[i]:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def backtest_random_entry(h1_df, n_trades, spread, lot, maxloss_cap=35,
                          sl_atr=3.5, tp_atr=8.0,
                          trail_act=0.14, trail_dist=0.025, max_hold=20, seed=42):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df, 14)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)

    rng = np.random.RandomState(seed)
    entry_bars = sorted(rng.choice(range(500, n - max_hold - 5), size=min(n_trades, n - 600), replace=False))
    directions = rng.choice(['BUY', 'SELL'], size=len(entry_bars))

    trades = []
    for idx, bar_i in enumerate(entry_bars):
        d = directions[idx]
        entry_price = c[bar_i] + (SPREAD / 2 if d == 'BUY' else -SPREAD / 2)
        pos = {'dir': d, 'entry': entry_price, 'bar': bar_i, 'time': times[bar_i], 'atr': atr[bar_i]}
        for j in range(bar_i + 1, min(bar_i + max_hold + 1, n)):
            result = _run_exit_with_cap(pos, j, h[j], lo[j], c[j], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); break
    return trades


# ═══════════════════════════════════════════════════════════════
# Phase 5: Robustness Checks
# ═══════════════════════════════════════════════════════════════

def robustness_holdout(features, labels, xgb_params):
    train_mask = features.index < '2023-01-01'
    test_mask = features.index >= '2023-01-01'
    valid_train = train_mask & labels.notna()
    valid_test = test_mask & labels.notna()

    X_train = features.loc[valid_train].values
    y_train = labels.loc[valid_train].values
    X_test = features.loc[valid_test].values
    y_test = labels.loc[valid_test].values

    if len(X_test) < 50:
        return {'holdout_auc': None, 'holdout_acc': None}

    scaler = StandardScaler()
    X_train_s = np.nan_to_num(scaler.fit_transform(X_train), nan=0.0)
    X_test_s = np.nan_to_num(scaler.transform(X_test), nan=0.0)

    model = XGBClassifier(**xgb_params)
    model.fit(X_train_s, y_train)
    proba = model.predict_proba(X_test_s)[:, 1]

    return {
        'holdout_auc': round(float(roc_auc_score(y_test, proba)), 4),
        'holdout_acc': round(float(accuracy_score(y_test, (proba > 0.5).astype(int))), 4),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
    }


def robustness_param_perturbation(features, labels, base_params, n_variants=5):
    train_mask = features.index < '2023-01-01'
    test_mask = features.index >= '2023-01-01'
    valid_train = train_mask & labels.notna()
    valid_test = test_mask & labels.notna()

    X_train = features.loc[valid_train].values
    y_train = labels.loc[valid_train].values
    X_test = features.loc[valid_test].values
    y_test = labels.loc[valid_test].values

    if len(X_test) < 50:
        return {'aucs': [], 'std': None, 'pass': False}

    scaler = StandardScaler()
    X_train_s = np.nan_to_num(scaler.fit_transform(X_train), nan=0.0)
    X_test_s = np.nan_to_num(scaler.transform(X_test), nan=0.0)

    rng = np.random.RandomState(123)
    aucs = []
    for v in range(n_variants):
        params = base_params.copy()
        params['max_depth'] = rng.choice([3, 4, 5, 6])
        params['learning_rate'] = rng.choice([0.03, 0.05, 0.08, 0.1])
        params['n_estimators'] = rng.choice([150, 200, 250, 300])
        params['subsample'] = rng.choice([0.7, 0.8, 0.9])
        params['min_child_weight'] = rng.choice([5, 10, 15, 20])
        params['random_state'] = 42 + v

        model = XGBClassifier(**params)
        model.fit(X_train_s, y_train)
        proba = model.predict_proba(X_test_s)[:, 1]
        aucs.append(float(roc_auc_score(y_test, proba)))

    std = float(np.std(aucs))
    return {
        'aucs': [round(a, 4) for a in aucs],
        'mean_auc': round(float(np.mean(aucs)), 4),
        'std': round(std, 4),
        'pass': std < 0.05,
    }


def robustness_shuffle_test(features, labels, xgb_params, n_perms=10):
    train_mask = features.index < '2023-01-01'
    test_mask = features.index >= '2023-01-01'
    valid_train = train_mask & labels.notna()
    valid_test = test_mask & labels.notna()

    X_train = features.loc[valid_train].values
    y_train = labels.loc[valid_train].values
    X_test = features.loc[valid_test].values
    y_test = labels.loc[valid_test].values

    if len(X_test) < 50:
        return {'z_score': None, 'pass': False}

    scaler = StandardScaler()
    X_train_s = np.nan_to_num(scaler.fit_transform(X_train), nan=0.0)
    X_test_s = np.nan_to_num(scaler.transform(X_test), nan=0.0)

    model = XGBClassifier(**xgb_params)
    model.fit(X_train_s, y_train)
    real_auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])

    shuffled_aucs = []
    for perm in range(n_perms):
        y_shuf = y_train.copy()
        np.random.RandomState(perm).shuffle(y_shuf)
        m = XGBClassifier(**{**xgb_params, 'random_state': perm})
        m.fit(X_train_s, y_shuf)
        proba = m.predict_proba(X_test_s)[:, 1]
        try:
            shuffled_aucs.append(float(roc_auc_score(y_test, proba)))
        except ValueError:
            shuffled_aucs.append(0.5)

    shuf_mean = np.mean(shuffled_aucs)
    shuf_std = np.std(shuffled_aucs) if np.std(shuffled_aucs) > 0 else 0.001
    z = (real_auc - shuf_mean) / shuf_std

    return {
        'real_auc': round(float(real_auc), 4),
        'shuffled_mean': round(float(shuf_mean), 4),
        'shuffled_std': round(float(shuf_std), 4),
        'z_score': round(float(z), 2),
        'pass': z > 3,
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    out_dir = Path("results/r179_ml_direction")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    print("=" * 70, flush=True)
    print("R179 — ML Direction Predictor (2-class Up/Down)", flush=True)
    print("=" * 70, flush=True)

    print("\n[1/6] Loading data...", flush=True)
    h1 = load_h1()
    macro = load_macro()
    print(f"  H1: {len(h1)} bars ({h1.index[0]} -> {h1.index[-1]})", flush=True)

    print("\n[2/6] Building features...", flush=True)
    features = build_features(h1, macro)
    feat_cols = list(features.columns)
    print(f"  {len(feat_cols)} features: {feat_cols[:10]}...", flush=True)

    atr = compute_atr(h1, 14)
    labels = build_labels(h1, atr, horizon=4, threshold_atr=0.5)
    n_up = (labels == 1).sum()
    n_down = (labels == 0).sum()
    n_skip = labels.isna().sum()
    print(f"  Labels: UP={n_up}  DOWN={n_down}  SKIP={n_skip}  "
          f"balance={n_up/(n_up+n_down):.3f}", flush=True)
    results['label_stats'] = {
        'n_up': int(n_up), 'n_down': int(n_down), 'n_skip': int(n_skip),
        'balance': round(float(n_up / (n_up + n_down)), 3),
    }

    valid = labels.notna()
    features = features.loc[valid]
    labels = labels.loc[valid]
    for col in features.columns:
        features[col] = features[col].fillna(features[col].median())
    features = features.fillna(0)
    print(f"  Valid samples after NaN removal: {len(features)}", flush=True)

    print("\n[3/6] Walk-forward XGBoost training...", flush=True)
    print("-" * 60, flush=True)
    fold_results, all_preds = walk_forward_train(features, labels, FOLDS, XGB_PARAMS)
    results['walk_forward'] = fold_results

    valid_folds = [f for f in fold_results if f['auc'] is not None]
    if valid_folds:
        avg_auc = np.mean([f['auc'] for f in valid_folds])
        avg_acc = np.mean([f['acc'] for f in valid_folds])
        print(f"\n  Avg AUC={avg_auc:.4f}  Avg Acc={avg_acc:.4f} across {len(valid_folds)} folds", flush=True)
        results['avg_metrics'] = {'avg_auc': round(avg_auc, 4), 'avg_acc': round(avg_acc, 4)}

    agg_imp = {}
    for f in valid_folds:
        for feat, score in f['importance'].items():
            agg_imp[feat] = agg_imp.get(feat, 0) + score
    if agg_imp:
        n_f = len(valid_folds)
        agg_imp = {k: round(v / n_f, 5) for k, v in sorted(agg_imp.items(), key=lambda x: -x[1])}
        results['feature_importance'] = dict(list(agg_imp.items())[:20])
        print("\n  Top-10 features (avg importance):", flush=True)
        for rank, (feat, score) in enumerate(list(agg_imp.items())[:10], 1):
            print(f"    {rank:2d}. {feat:30s} {score:.5f}", flush=True)

    print("\n[4/6] Signal-to-trade backtest...", flush=True)
    print("-" * 60, flush=True)
    bt_results = {}
    for thr in THRESHOLDS:
        trades = backtest_ml_direction(h1, all_preds, thr)
        stats = _compute_stats(trades)
        n_buy = sum(1 for t in trades if t['dir'] == 'BUY')
        n_sell = sum(1 for t in trades if t['dir'] == 'SELL')
        stats['n_buy'] = n_buy
        stats['n_sell'] = n_sell
        stats['threshold'] = thr
        bt_results[f"thr_{thr}"] = stats
        print(f"  Threshold {thr:.2f}: n={stats['n']:4d} (B={n_buy}/S={n_sell})  "
              f"PnL=${stats['pnl']:8.2f}  Sharpe={stats['sharpe']:.3f}  "
              f"WR={stats['wr']:.1f}%  MaxDD=${stats['max_dd']:.2f}", flush=True)
    results['backtest'] = bt_results

    best_thr = max(bt_results.keys(), key=lambda k: bt_results[k]['sharpe'])
    best_stats = bt_results[best_thr]
    print(f"\n  Best threshold: {best_thr} (Sharpe={best_stats['sharpe']:.3f})", flush=True)

    print("\n[5/6] Comparison vs baselines...", flush=True)
    print("-" * 60, flush=True)

    keltner_trades = bt_l8_max(h1, SPREAD, UNIT_LOT, maxloss_cap=35)
    keltner_stats = _compute_stats(keltner_trades)
    print(f"  Keltner L8_MAX: n={keltner_stats['n']:4d}  PnL=${keltner_stats['pnl']:8.2f}  "
          f"Sharpe={keltner_stats['sharpe']:.3f}  WR={keltner_stats['wr']:.1f}%  "
          f"MaxDD=${keltner_stats['max_dd']:.2f}", flush=True)
    results['keltner_baseline'] = keltner_stats

    n_ml_trades = best_stats['n'] if best_stats['n'] > 0 else 200
    random_trades = backtest_random_entry(h1, n_ml_trades, SPREAD, UNIT_LOT, maxloss_cap=35)
    random_stats = _compute_stats(random_trades)
    print(f"  Random entry:   n={random_stats['n']:4d}  PnL=${random_stats['pnl']:8.2f}  "
          f"Sharpe={random_stats['sharpe']:.3f}  WR={random_stats['wr']:.1f}%  "
          f"MaxDD=${random_stats['max_dd']:.2f}", flush=True)
    results['random_baseline'] = random_stats

    print(f"\n  ML-Direction vs Keltner:  Sharpe delta = {best_stats['sharpe'] - keltner_stats['sharpe']:+.3f}", flush=True)
    print(f"  ML-Direction vs Random:  Sharpe delta = {best_stats['sharpe'] - random_stats['sharpe']:+.3f}", flush=True)
    results['comparison'] = {
        'ml_vs_keltner_sharpe_delta': round(best_stats['sharpe'] - keltner_stats['sharpe'], 3),
        'ml_vs_random_sharpe_delta': round(best_stats['sharpe'] - random_stats['sharpe'], 3),
        'best_ml_threshold': best_thr,
    }

    print("\n[6/6] Robustness checks...", flush=True)
    print("-" * 60, flush=True)

    print("  (a) Holdout AUC...", flush=True)
    holdout = robustness_holdout(features, labels, XGB_PARAMS)
    print(f"      Holdout AUC={holdout.get('holdout_auc')}  Acc={holdout.get('holdout_acc')}", flush=True)

    print("  (b) Param perturbation (5 variants)...", flush=True)
    perturb = robustness_param_perturbation(features, labels, XGB_PARAMS, n_variants=5)
    print(f"      AUCs={perturb['aucs']}  std={perturb['std']}  pass={perturb['pass']}", flush=True)

    print("  (c) Shuffle test (10 permutations)...", flush=True)
    shuffle = robustness_shuffle_test(features, labels, XGB_PARAMS, n_perms=10)
    print(f"      Real AUC={shuffle.get('real_auc')}  Shuffled mean={shuffle.get('shuffled_mean')}  "
          f"z={shuffle.get('z_score')}  pass={shuffle.get('pass')}", flush=True)

    results['robustness'] = {
        'holdout': holdout,
        'param_perturbation': perturb,
        'shuffle_test': shuffle,
    }

    all_pass = True
    if holdout.get('holdout_auc') and holdout['holdout_auc'] < 0.52:
        all_pass = False
    if not perturb.get('pass', False):
        all_pass = False
    if not shuffle.get('pass', False):
        all_pass = False

    if best_stats['sharpe'] > 0.5 and best_stats['n'] > 50 and all_pass:
        recommendation = "PROMISING — ML direction shows edge, consider adding to portfolio"
    elif best_stats['sharpe'] > 0 and all_pass:
        recommendation = "MARGINAL — Some edge but weak Sharpe, needs further tuning"
    else:
        recommendation = "NO EDGE — ML direction does not improve over baselines"

    results['recommendation'] = recommendation
    results['runtime_seconds'] = round((datetime.now() - t0).total_seconds(), 1)

    print("\n" + "=" * 70, flush=True)
    print(f"RECOMMENDATION: {recommendation}", flush=True)
    print(f"Runtime: {results['runtime_seconds']:.1f}s", flush=True)
    print("=" * 70, flush=True)

    out_path = out_dir / "r179_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
