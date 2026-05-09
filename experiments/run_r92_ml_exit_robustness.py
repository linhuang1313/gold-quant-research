#!/usr/bin/env python3
"""
R92 — ML Exit Robustness Verification Suite
==============================================
5 independent tests to verify the R90-D ML Exit filter (+49% Sharpe on TSMOM)
is NOT overfitted before live deployment.

Tests:
  1. Independent Holdout (train 2015-2022, test 2023-2026)
  2. Parameter Perturbation (10 XGBoost hyperparameter variants)
  3. Random Label Shuffle (20 permutations, null hypothesis test)
  4. Feature Ablation (tech-only / macro-only / time-only / full)
  5. Per-Fold AUC Stability (coefficient of variation analysis)

Pass criteria per test defined in each section.
Expected runtime: ~10-15 minutes on GPU server.
"""
import sys, os, io, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r92_robustness")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
LOTS = {'L8_MAX': 0.05, 'TSMOM': 0.04, 'SESS_BO': 0.02, 'PSAR': 0.01}

STRATEGIES_TO_TEST = ['TSMOM']
HOLDOUT_SPLIT = "2023-01-01"
N_SHUFFLE = 20

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]

PARAM_GRID = [
    {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300},
    {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 300},
    {'max_depth': 5, 'learning_rate': 0.03, 'n_estimators': 300},
    {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 100},
    {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 300},
    {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 500},
    {'max_depth': 5, 'learning_rate': 0.08, 'n_estimators': 300},
    {'max_depth': 5, 'learning_rate': 0.10, 'n_estimators': 300},
    {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 300},
    {'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 300},
]

TECH_FEATURES = [
    'atr_14', 'adx_14', 'rsi_14', 'rsi_2', 'kc_breakout_strength',
    'volume_ratio', 'atr_percentile', 'ema9_ema21_cross', 'close_ema100_dist',
]
MACRO_FEATURES = [
    'real_yield_change5', 'real_yield_change20', 'vix_zscore', 'vix_close',
    'dxy_mom5', 'dxy_mom20', 'credit_stress', 'yield_curve_10y2y',
    'copper_gold_ratio', 'crude_mom5', 'usdcnh_mom5', 'cot_mm_net_zscore',
    'fed_funds_dff', 'risk_appetite_z',
]
TIME_FEATURES = ['hour_of_day', 'day_of_week', 'direction']


# ═══════════════════════════════════════════════════════════════
# Shared helpers (from R90-D)
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
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'max_dd': round(_max_dd(daily), 2),
    }


# ═══════════════════════════════════════════════════════════════
# Strategy Backtest
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# Feature Engineering (from R90-D)
# ═══════════════════════════════════════════════════════════════

def compute_h1_indicators(h1_df):
    df = h1_df.copy()
    df['ATR_14'] = compute_atr(df, 14)
    df['ADX_14'] = compute_adx(df, 14)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain_14 = gain.rolling(14).mean()
    avg_loss_14 = loss.rolling(14).mean()
    rs_14 = avg_gain_14 / avg_loss_14.replace(0, np.nan)
    df['RSI_14'] = 100 - 100 / (1 + rs_14)
    avg_gain_2 = gain.rolling(2).mean()
    avg_loss_2 = loss.rolling(2).mean()
    rs_2 = avg_gain_2 / avg_loss_2.replace(0, np.nan)
    df['RSI_2'] = 100 - 100 / (1 + rs_2)
    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    atr14 = df['ATR_14']
    kc_upper = ema20 + 1.5 * atr14
    kc_lower = ema20 - 1.5 * atr14
    kc_width = kc_upper - kc_lower
    df['KC_breakout_strength'] = (df['Close'] - ema20) / kc_width.replace(0, np.nan)
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        df['Volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, np.nan)
    else:
        range_vol = df['High'] - df['Low']
        df['Volume_ratio'] = range_vol / range_vol.rolling(20).mean().replace(0, np.nan)
    df['ATR_percentile'] = df['ATR_14'].rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA9_EMA21_cross'] = (df['EMA9'] - df['EMA21']) / atr14.replace(0, np.nan)
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['Close_EMA100_dist'] = (df['Close'] - df['EMA100']) / atr14.replace(0, np.nan)
    return df


def build_features_for_trades(trades, h1_indicators, external_daily):
    records = []
    labels = []
    valid_indices = []
    for idx, t in enumerate(trades):
        entry_time = pd.Timestamp(t['entry_time'])
        entry_date = entry_time.normalize()
        if entry_date.tzinfo is not None:
            entry_date = entry_date.tz_localize(None)
        if entry_time in h1_indicators.index:
            h1_row = h1_indicators.loc[entry_time]
        else:
            loc = h1_indicators.index.get_indexer([entry_time], method='ffill')
            if loc[0] < 0:
                continue
            h1_row = h1_indicators.iloc[loc[0]]
        feat = {}
        feat['atr_14'] = h1_row.get('ATR_14', np.nan)
        feat['adx_14'] = h1_row.get('ADX_14', np.nan)
        feat['rsi_14'] = h1_row.get('RSI_14', np.nan)
        feat['rsi_2'] = h1_row.get('RSI_2', np.nan)
        feat['kc_breakout_strength'] = h1_row.get('KC_breakout_strength', np.nan)
        feat['volume_ratio'] = h1_row.get('Volume_ratio', np.nan)
        feat['atr_percentile'] = h1_row.get('ATR_percentile', np.nan)
        feat['ema9_ema21_cross'] = h1_row.get('EMA9_EMA21_cross', np.nan)
        feat['close_ema100_dist'] = h1_row.get('Close_EMA100_dist', np.nan)
        feat['hour_of_day'] = entry_time.hour
        feat['day_of_week'] = entry_time.dayofweek
        feat['direction'] = 1 if t['dir'] == 'BUY' else -1
        ext_row = None
        if external_daily is not None and len(external_daily) > 0:
            if entry_date in external_daily.index:
                ext_row = external_daily.loc[entry_date]
            else:
                loc_ext = external_daily.index.get_indexer([entry_date], method='ffill')
                if loc_ext[0] >= 0:
                    ext_row = external_daily.iloc[loc_ext[0]]
        if ext_row is not None:
            for col_name, col_key in [
                ('real_yield_change5', 'REAL_YIELD_Change5'),
                ('real_yield_change20', 'REAL_YIELD_Change20'),
                ('vix_zscore', 'VIX_Zscore'),
                ('vix_close', 'VIX_Close'),
                ('dxy_mom5', 'DXY_Mom5'),
                ('dxy_mom20', 'DXY_Mom20'),
                ('credit_stress', 'CREDIT_STRESS'),
                ('yield_curve_10y2y', 'YIELD_CURVE_10Y2Y'),
                ('copper_gold_ratio', 'COPPER_GOLD_RATIO'),
                ('crude_mom5', 'CRUDE_Mom5'),
                ('usdcnh_mom5', 'USDCNH_Mom5'),
                ('cot_mm_net_zscore', 'COT_MM_Net_Zscore'),
                ('fed_funds_dff', 'FED_FUNDS_DFF'),
                ('risk_appetite_z', 'RISK_APPETITE_Z'),
            ]:
                feat[col_name] = ext_row.get(col_key, np.nan)
        else:
            for col_name in MACRO_FEATURES:
                feat[col_name] = np.nan
        feat['regime_label'] = np.nan
        feat['ml_direction_prob'] = np.nan
        records.append(feat)
        labels.append(1 if t['pnl'] > 0 else 0)
        valid_indices.append(idx)
    if not records:
        return pd.DataFrame(), np.array([]), []
    X = pd.DataFrame(records)
    y = np.array(labels)
    return X, y, valid_indices


# ═══════════════════════════════════════════════════════════════
# ML Training Utilities
# ═══════════════════════════════════════════════════════════════

def get_xgb_model(max_depth=5, learning_rate=0.05, n_estimators=300):
    try:
        import xgboost as xgb
        try:
            model = xgb.XGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                tree_method='hist', device='cuda', random_state=42, verbosity=0
            )
            model.fit(np.random.randn(10, 5), np.random.randint(0, 2, 10))
            return xgb.XGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                tree_method='hist', device='cuda', random_state=42, verbosity=0
            )
        except Exception:
            return xgb.XGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                tree_method='hist', random_state=42, verbosity=0
            )
    except ImportError:
        return None


def train_and_evaluate(X, y, entry_times, model, feature_cols=None):
    """Train with walk-forward and return per-fold AUC + overall AUC."""
    from sklearn.metrics import roc_auc_score

    if feature_cols is not None:
        available = [c for c in feature_cols if c in X.columns]
        X_use = X[available].copy()
    else:
        X_use = X.copy()

    if hasattr(entry_times, 'dt') and entry_times.dt.tz is not None:
        entry_times = entry_times.dt.tz_localize(None)
    elif hasattr(entry_times, 'tz') and entry_times.tz is not None:
        entry_times = entry_times.tz_localize(None)

    oos_preds = np.full(len(y), np.nan)
    fold_aucs = []

    for fold_name, fold_start, fold_end in FOLDS:
        fold_start_ts = pd.Timestamp(fold_start)
        fold_end_ts = pd.Timestamp(fold_end)
        train_mask = entry_times < fold_start_ts
        test_mask = (entry_times >= fold_start_ts) & (entry_times < fold_end_ts)
        n_train = train_mask.sum()
        n_test = test_mask.sum()
        if n_train < 30 or n_test < 5:
            continue

        X_train = X_use[train_mask].copy()
        y_train = y[train_mask]
        X_test = X_use[test_mask].copy()
        y_test = y[test_mask]

        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)
        constant_cols = [c for c in X_train.columns if X_train[c].nunique() <= 1]
        if constant_cols:
            X_train = X_train.drop(columns=constant_cols)
            X_test = X_test.drop(columns=constant_cols)
        if len(X_train.columns) == 0:
            continue

        try:
            m = deepcopy(model)
            m.fit(X_train, y_train)
            probs = m.predict_proba(X_test)[:, 1]
            test_indices = np.where(test_mask)[0]
            oos_preds[test_indices] = probs
            if len(np.unique(y_test)) > 1:
                fold_aucs.append(roc_auc_score(y_test, probs))
        except Exception:
            continue

    valid_oos = ~np.isnan(oos_preds)
    overall_auc = None
    if valid_oos.sum() > 10:
        y_valid = y[valid_oos]
        p_valid = oos_preds[valid_oos]
        if len(np.unique(y_valid)) > 1:
            overall_auc = float(roc_auc_score(y_valid, p_valid))

    return {
        'overall_auc': overall_auc,
        'fold_aucs': fold_aucs,
        'oos_preds': oos_preds,
    }


def holdout_train_evaluate(X, y, entry_times, model, split_date, feature_cols=None):
    """Single holdout split: train before split_date, test after."""
    from sklearn.metrics import roc_auc_score

    if feature_cols is not None:
        available = [c for c in feature_cols if c in X.columns]
        X_use = X[available].copy()
    else:
        X_use = X.copy()

    if hasattr(entry_times, 'dt') and entry_times.dt.tz is not None:
        entry_times = entry_times.dt.tz_localize(None)
    elif hasattr(entry_times, 'tz') and entry_times.tz is not None:
        entry_times = entry_times.tz_localize(None)

    split_ts = pd.Timestamp(split_date)
    train_mask = entry_times < split_ts
    test_mask = entry_times >= split_ts

    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())

    if n_train < 30 or n_test < 10:
        return {'auc': None, 'n_train': n_train, 'n_test': n_test, 'status': 'insufficient_data'}

    X_train = X_use[train_mask].copy()
    y_train = y[train_mask]
    X_test = X_use[test_mask].copy()
    y_test = y[test_mask]

    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    constant_cols = [c for c in X_train.columns if X_train[c].nunique() <= 1]
    if constant_cols:
        X_train = X_train.drop(columns=constant_cols)
        X_test = X_test.drop(columns=constant_cols)

    try:
        m = deepcopy(model)
        m.fit(X_train, y_train)
        probs = m.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else None
        return {
            'auc': auc,
            'n_train': n_train,
            'n_test': n_test,
            'probs': probs,
            'y_test': y_test,
            'test_mask': test_mask,
            'status': 'ok'
        }
    except Exception as e:
        return {'auc': None, 'n_train': n_train, 'n_test': n_test, 'status': f'error: {e}'}


# ═══════════════════════════════════════════════════════════════
# TEST 1: Independent Holdout
# ═══════════════════════════════════════════════════════════════

def test_1_independent_holdout(X, y, entry_times, trades, valid_indices):
    print("\n" + "=" * 70)
    print("  TEST 1: Independent Holdout (Train 2015-2022, Test 2023-2026)")
    print("=" * 70, flush=True)

    model = get_xgb_model()
    if model is None:
        return {'passed': False, 'reason': 'xgboost not available'}

    result = holdout_train_evaluate(X, y, entry_times, model, HOLDOUT_SPLIT)
    print(f"    Train: {result.get('n_train', 0)} samples (2015-2022)")
    print(f"    Test:  {result.get('n_test', 0)} samples (2023-2026)")

    if result['auc'] is None:
        print(f"    Status: {result.get('status', 'unknown')}")
        return {'passed': False, 'reason': result.get('status', 'no_auc'), 'details': {}}

    print(f"    Holdout AUC: {result['auc']:.4f}")

    # Compute filtered Sharpe on holdout period
    probs = result.get('probs', np.array([]))
    test_mask = result.get('test_mask', pd.Series(dtype=bool))

    test_trade_indices = [valid_indices[i] for i in range(len(valid_indices)) if test_mask.iloc[i]]
    holdout_trades = [trades[ti] for ti in test_trade_indices]
    baseline_stats = _compute_stats(holdout_trades)

    filtered_trades = []
    for i, prob in enumerate(probs):
        if prob >= 0.5:
            filtered_trades.append(holdout_trades[i])
    filtered_stats = _compute_stats(filtered_trades)

    print(f"    Baseline Sharpe (holdout): {baseline_stats['sharpe']:.3f} ({baseline_stats['n']} trades)")
    print(f"    Filtered Sharpe (holdout): {filtered_stats['sharpe']:.3f} ({filtered_stats['n']} trades)")

    sharpe_improvement = filtered_stats['sharpe'] - baseline_stats['sharpe']
    pct_improvement = (sharpe_improvement / baseline_stats['sharpe'] * 100) if baseline_stats['sharpe'] != 0 else 0

    print(f"    Sharpe improvement: {'+' if sharpe_improvement >= 0 else ''}{sharpe_improvement:.3f} ({pct_improvement:+.1f}%)")

    passed = result['auc'] > 0.65 and filtered_stats['sharpe'] > baseline_stats['sharpe']
    verdict = "PASS" if passed else "FAIL"
    print(f"\n    >>> TEST 1 VERDICT: {verdict}")
    print(f"        Criteria: AUC > 0.65 AND filtered_sharpe > baseline_sharpe")

    return {
        'passed': passed,
        'holdout_auc': round(result['auc'], 4),
        'baseline_sharpe': baseline_stats['sharpe'],
        'filtered_sharpe': filtered_stats['sharpe'],
        'sharpe_pct_improvement': round(pct_improvement, 1),
        'n_train': result['n_train'],
        'n_test': result['n_test'],
    }


# ═══════════════════════════════════════════════════════════════
# TEST 2: Parameter Perturbation
# ═══════════════════════════════════════════════════════════════

def test_2_param_perturbation(X, y, entry_times):
    print("\n" + "=" * 70)
    print("  TEST 2: Parameter Perturbation (10 XGBoost hyperparameter variants)")
    print("=" * 70, flush=True)

    aucs = []
    for i, params in enumerate(PARAM_GRID):
        model = get_xgb_model(**params)
        if model is None:
            continue
        result = train_and_evaluate(X, y, entry_times, model)
        auc = result['overall_auc']
        auc_str = f"{auc:.4f}" if auc else "N/A"
        print(f"    [{i+1:>2}/{len(PARAM_GRID)}] depth={params['max_depth']}, "
              f"lr={params['learning_rate']}, n_est={params['n_estimators']}  "
              f"-> AUC={auc_str}")
        if auc is not None:
            aucs.append(auc)

    if len(aucs) < 3:
        print(f"    Too few valid results ({len(aucs)})")
        return {'passed': False, 'reason': 'insufficient_results', 'aucs': aucs}

    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    min_auc = float(np.min(aucs))
    max_auc = float(np.max(aucs))

    print(f"\n    Results: mean={mean_auc:.4f}, std={std_auc:.4f}, "
          f"range=[{min_auc:.4f}, {max_auc:.4f}]")

    passed = std_auc < 0.05 and min_auc > 0.68
    verdict = "PASS" if passed else "FAIL"
    print(f"\n    >>> TEST 2 VERDICT: {verdict}")
    print(f"        Criteria: std < 0.05 AND all variants AUC > 0.68")

    return {
        'passed': passed,
        'mean_auc': round(mean_auc, 4),
        'std_auc': round(std_auc, 4),
        'min_auc': round(min_auc, 4),
        'max_auc': round(max_auc, 4),
        'n_variants': len(aucs),
        'all_aucs': [round(a, 4) for a in aucs],
    }


# ═══════════════════════════════════════════════════════════════
# TEST 3: Random Label Shuffle (Null Hypothesis)
# ═══════════════════════════════════════════════════════════════

def test_3_random_shuffle(X, y, entry_times):
    print("\n" + "=" * 70)
    print(f"  TEST 3: Random Label Shuffle ({N_SHUFFLE} permutations)")
    print("=" * 70, flush=True)

    model = get_xgb_model()
    if model is None:
        return {'passed': False, 'reason': 'xgboost not available'}

    real_result = train_and_evaluate(X, y, entry_times, model)
    real_auc = real_result['overall_auc']
    print(f"    Real labels AUC: {real_auc:.4f}" if real_auc else "    Real labels AUC: N/A")

    if real_auc is None:
        return {'passed': False, 'reason': 'real_auc_unavailable'}

    shuffle_aucs = []
    for i in range(N_SHUFFLE):
        y_shuffled = np.random.permutation(y)
        model_s = get_xgb_model()
        result_s = train_and_evaluate(X, y_shuffled, entry_times, model_s)
        auc_s = result_s['overall_auc']
        if auc_s is not None:
            shuffle_aucs.append(auc_s)
        if (i + 1) % 5 == 0:
            print(f"    Shuffle [{i+1:>2}/{N_SHUFFLE}] done...", flush=True)

    if len(shuffle_aucs) < 5:
        return {'passed': False, 'reason': 'insufficient_shuffle_results'}

    shuffle_mean = float(np.mean(shuffle_aucs))
    shuffle_std = float(np.std(shuffle_aucs))
    z_score = (real_auc - shuffle_mean) / shuffle_std if shuffle_std > 0 else 0
    p_value = sum(1 for s in shuffle_aucs if s >= real_auc) / len(shuffle_aucs)

    print(f"\n    Shuffle distribution: mean={shuffle_mean:.4f}, std={shuffle_std:.4f}")
    print(f"    Real AUC: {real_auc:.4f}")
    print(f"    Z-score: {z_score:.2f}")
    print(f"    Empirical p-value: {p_value:.4f} ({sum(1 for s in shuffle_aucs if s >= real_auc)}/{len(shuffle_aucs)} >= real)")

    passed = z_score > 3.0 and p_value < 0.05
    verdict = "PASS" if passed else "FAIL"
    print(f"\n    >>> TEST 3 VERDICT: {verdict}")
    print(f"        Criteria: z-score > 3.0 AND p-value < 0.05")

    return {
        'passed': passed,
        'real_auc': round(real_auc, 4),
        'shuffle_mean': round(shuffle_mean, 4),
        'shuffle_std': round(shuffle_std, 4),
        'z_score': round(z_score, 2),
        'p_value': round(p_value, 4),
        'n_shuffles': len(shuffle_aucs),
    }


# ═══════════════════════════════════════════════════════════════
# TEST 4: Feature Ablation
# ═══════════════════════════════════════════════════════════════

def test_4_feature_ablation(X, y, entry_times):
    print("\n" + "=" * 70)
    print("  TEST 4: Feature Ablation (tech / macro / time / full)")
    print("=" * 70, flush=True)

    groups = {
        'tech_only': TECH_FEATURES,
        'macro_only': MACRO_FEATURES,
        'time_only': TIME_FEATURES,
        'full': None,
    }

    results = {}
    for group_name, feature_cols in groups.items():
        model_g = get_xgb_model()
        if model_g is None:
            results[group_name] = None
            continue
        result_g = train_and_evaluate(X, y, entry_times, model_g, feature_cols=feature_cols)
        auc_g = result_g['overall_auc']
        n_feats = len([c for c in (feature_cols or X.columns) if c in X.columns])
        auc_str = f"{auc_g:.4f}" if auc_g else "N/A"
        print(f"    {group_name:>12}: AUC={auc_str} ({n_feats} features)")
        results[group_name] = auc_g

    full_auc = results.get('full')
    tech_auc = results.get('tech_only')
    macro_auc = results.get('macro_only')
    time_auc = results.get('time_only')

    external_adds_value = (full_auc is not None and tech_auc is not None and full_auc > tech_auc)
    time_not_predictive = (time_auc is not None and time_auc < 0.60)

    passed = external_adds_value and time_not_predictive
    verdict = "PASS" if passed else "FAIL"
    print(f"\n    >>> TEST 4 VERDICT: {verdict}")
    print(f"        Criteria: full > tech_only (external adds value) AND time_only AUC < 0.60")
    if full_auc and tech_auc:
        print(f"        full({full_auc:.4f}) > tech_only({tech_auc:.4f}): {'YES' if external_adds_value else 'NO'}")
    if time_auc:
        print(f"        time_only({time_auc:.4f}) < 0.60: {'YES' if time_not_predictive else 'NO'}")

    return {
        'passed': passed,
        'full_auc': round(full_auc, 4) if full_auc else None,
        'tech_only_auc': round(tech_auc, 4) if tech_auc else None,
        'macro_only_auc': round(macro_auc, 4) if macro_auc else None,
        'time_only_auc': round(time_auc, 4) if time_auc else None,
        'external_adds_value': external_adds_value,
        'time_not_predictive': time_not_predictive,
    }


# ═══════════════════════════════════════════════════════════════
# TEST 5: Per-Fold AUC Stability
# ═══════════════════════════════════════════════════════════════

def test_5_fold_stability(X, y, entry_times):
    print("\n" + "=" * 70)
    print("  TEST 5: Per-Fold AUC Stability")
    print("=" * 70, flush=True)

    model = get_xgb_model()
    if model is None:
        return {'passed': False, 'reason': 'xgboost not available'}

    result = train_and_evaluate(X, y, entry_times, model)
    fold_aucs = result['fold_aucs']

    if len(fold_aucs) < 3:
        print(f"    Only {len(fold_aucs)} folds with valid AUC")
        return {'passed': False, 'reason': f'only_{len(fold_aucs)}_folds'}

    for i, auc in enumerate(fold_aucs):
        print(f"    Fold {i+1}: AUC = {auc:.4f}")

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    cv = std_auc / mean_auc if mean_auc > 0 else 999
    min_auc = float(np.min(fold_aucs))
    max_auc = float(np.max(fold_aucs))

    print(f"\n    Mean: {mean_auc:.4f}, Std: {std_auc:.4f}, CV: {cv:.4f}")
    print(f"    Range: [{min_auc:.4f}, {max_auc:.4f}]")

    passed = min_auc > 0.58 and cv < 0.20
    verdict = "PASS" if passed else "FAIL"
    print(f"\n    >>> TEST 5 VERDICT: {verdict}")
    print(f"        Criteria: min_fold_AUC > 0.58 AND CV < 0.20")

    return {
        'passed': passed,
        'fold_aucs': [round(a, 4) for a in fold_aucs],
        'mean_auc': round(mean_auc, 4),
        'std_auc': round(std_auc, 4),
        'cv': round(cv, 4),
        'min_auc': round(min_auc, 4),
        'max_auc': round(max_auc, 4),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("  R92: ML Exit Robustness Verification Suite")
    print("  5 tests to verify ML Exit filter is NOT overfitted")
    print("=" * 70, flush=True)

    # Load data
    print("\n  Loading data...", flush=True)
    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0].date()} ~ {h1_df.index[-1].date()})")

    ext_path = Path("data/external/aligned_daily.csv")
    if ext_path.exists():
        external_daily = pd.read_csv(ext_path, parse_dates=['Date'], index_col='Date')
        external_daily.index = external_daily.index.normalize()
        if external_daily.index.tz is not None:
            external_daily.index = external_daily.index.tz_localize(None)
        print(f"    External daily: {len(external_daily)} rows")
    else:
        external_daily = None
        print("    [WARN] No external daily data, macro features will be NaN")

    # Compute indicators
    print("  Computing H1 indicators...", flush=True)
    h1_indicators = compute_h1_indicators(h1_df)

    # Run TSMOM backtest
    print("  Running TSMOM backtest...", flush=True)
    cap = CAPS['TSMOM']
    lot = LOTS['TSMOM']
    trades = bt_tsmom(h1_df, spread=SPREAD, lot=lot, maxloss_cap=cap)
    stats = _compute_stats(trades)
    print(f"    TSMOM: {stats['n']} trades, WR={stats['wr']:.1f}%, "
          f"Sharpe={stats['sharpe']:.3f}, PnL=${stats['pnl']:,.0f}")

    # Build features
    print("  Building feature matrix...", flush=True)
    X, y, valid_indices = build_features_for_trades(trades, h1_indicators, external_daily)
    print(f"    Features: {X.shape[1]} cols, {len(X)} samples (win_rate={y.mean()*100:.1f}%)")

    entry_times = pd.Series([pd.Timestamp(trades[vi]['entry_time']) for vi in valid_indices])
    if entry_times.dt.tz is not None:
        entry_times = entry_times.dt.tz_localize(None)

    # Run 5 Tests
    all_results = {}
    all_results['test_1_holdout'] = test_1_independent_holdout(X, y, entry_times, trades, valid_indices)
    all_results['test_2_param_perturb'] = test_2_param_perturbation(X, y, entry_times)
    all_results['test_3_shuffle'] = test_3_random_shuffle(X, y, entry_times)
    all_results['test_4_ablation'] = test_4_feature_ablation(X, y, entry_times)
    all_results['test_5_fold_stability'] = test_5_fold_stability(X, y, entry_times)

    # Final Summary
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  {'Test':<30} | {'Result':<8} | {'Key Metric':<40}")
    print(f"  {'-'*30}-+-{'-'*8}-+-{'-'*40}")

    n_passed = 0
    test_names = {
        'test_1_holdout': '1. Independent Holdout',
        'test_2_param_perturb': '2. Param Perturbation',
        'test_3_shuffle': '3. Random Label Shuffle',
        'test_4_ablation': '4. Feature Ablation',
        'test_5_fold_stability': '5. Per-Fold Stability',
    }
    for key, label in test_names.items():
        res = all_results.get(key, {})
        passed = res.get('passed', False)
        if passed:
            n_passed += 1
        verdict = "PASS" if passed else "FAIL"

        if key == 'test_1_holdout':
            metric = f"AUC={res.get('holdout_auc', 'N/A')}, Sharpe {res.get('sharpe_pct_improvement', 'N/A'):+}%"
        elif key == 'test_2_param_perturb':
            metric = f"range [{res.get('min_auc', 'N/A')}, {res.get('max_auc', 'N/A')}], std={res.get('std_auc', 'N/A')}"
        elif key == 'test_3_shuffle':
            metric = f"z={res.get('z_score', 'N/A')}, p={res.get('p_value', 'N/A')}"
        elif key == 'test_4_ablation':
            metric = f"full={res.get('full_auc', 'N/A')} > tech={res.get('tech_only_auc', 'N/A')}"
        elif key == 'test_5_fold_stability':
            metric = f"min={res.get('min_auc', 'N/A')}, CV={res.get('cv', 'N/A')}"
        else:
            metric = ""

        print(f"  {label:<30} | {verdict:<8} | {metric:<40}")

    print(f"  {'-'*30}-+-{'-'*8}-+-{'-'*40}")
    overall = "ROBUST - proceed with deployment" if n_passed >= 4 else "NOT ROBUST - do NOT deploy"
    print(f"\n  OVERALL: {n_passed}/5 PASSED --> {overall}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("=" * 70, flush=True)

    # Save results
    output = {
        'strategy': 'TSMOM',
        'n_trades': stats['n'],
        'baseline_sharpe': stats['sharpe'],
        'tests': all_results,
        'n_passed': n_passed,
        'overall_verdict': 'ROBUST' if n_passed >= 4 else 'NOT_ROBUST',
        'elapsed_s': round(elapsed, 1),
    }
    with open(OUTPUT_DIR / "r92_results.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_DIR}/r92_results.json", flush=True)


if __name__ == "__main__":
    main()
