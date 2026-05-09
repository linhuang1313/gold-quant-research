#!/usr/bin/env python3
"""
R92-B — ML Exit Robustness for PSAR / SESS_BO / L8_MAX
=========================================================
Re-run the 5-test robustness suite on strategies with sufficient sample size:
  - PSAR:    ~3,155 trades (24/month) — Primary candidate
  - SESS_BO: ~2,000 trades (15/month)
  - L8_MAX:  ~22,000 trades (167/month)

These have 20-180x more samples than TSMOM (121), making ML validation meaningful.

Expected runtime: ~20-30 minutes (L8_MAX dominates due to full engine backtest).
"""
import sys, os, io, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r92b_multi_strategy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'SESS_BO': 35}
LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'SESS_BO': 0.08}

STRATEGIES_TO_TEST = ['PSAR', 'SESS_BO', 'L8_MAX']
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
    {'max_depth': 5, 'learning_rate': 0.03, 'n_estimators': 300},
    {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 300},
    {'max_depth': 5, 'learning_rate': 0.10, 'n_estimators': 300},
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


def _compute_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    return {
        'n': n,
        'sharpe': round(_sharpe(daily), 3),
        'pnl': round(sum(pnls), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════
# Strategy Backtests
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
            'entry_bar': 0,
            'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0,
        })
    return trades


# ═══════════════════════════════════════════════════════════════
# Feature Engineering
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


def build_features_for_trades(trades, h1_indicators, external_daily):
    h1_idx = h1_indicators.index
    if h1_idx.tz is not None:
        h1_indicators = h1_indicators.copy()
        h1_indicators.index = h1_idx.tz_localize(None)
    records = []; labels = []; valid_indices = []
    for idx, t in enumerate(trades):
        entry_time = pd.Timestamp(t['entry_time'])
        entry_date = entry_time.normalize()
        if entry_date.tzinfo is not None:
            entry_date = entry_date.tz_localize(None)
        if entry_time.tzinfo is not None:
            entry_time_naive = entry_time.tz_localize(None)
        else:
            entry_time_naive = entry_time
        if entry_time_naive in h1_indicators.index:
            h1_row = h1_indicators.loc[entry_time_naive]
        else:
            loc = h1_indicators.index.get_indexer([entry_time_naive], method='ffill')
            if loc[0] < 0: continue
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
                ('real_yield_change5', 'REAL_YIELD_Change5'), ('real_yield_change20', 'REAL_YIELD_Change20'),
                ('vix_zscore', 'VIX_Zscore'), ('vix_close', 'VIX_Close'),
                ('dxy_mom5', 'DXY_Mom5'), ('dxy_mom20', 'DXY_Mom20'),
                ('credit_stress', 'CREDIT_STRESS'), ('yield_curve_10y2y', 'YIELD_CURVE_10Y2Y'),
                ('copper_gold_ratio', 'COPPER_GOLD_RATIO'), ('crude_mom5', 'CRUDE_Mom5'),
                ('usdcnh_mom5', 'USDCNH_Mom5'), ('cot_mm_net_zscore', 'COT_MM_Net_Zscore'),
                ('fed_funds_dff', 'FED_FUNDS_DFF'), ('risk_appetite_z', 'RISK_APPETITE_Z'),
            ]:
                feat[col_name] = ext_row.get(col_key, np.nan)
        else:
            for col_name in MACRO_FEATURES:
                feat[col_name] = np.nan
        records.append(feat)
        labels.append(1 if t['pnl'] > 0 else 0)
        valid_indices.append(idx)
    if not records:
        return pd.DataFrame(), np.array([]), []
    return pd.DataFrame(records), np.array(labels), valid_indices


# ═══════════════════════════════════════════════════════════════
# ML Utilities
# ═══════════════════════════════════════════════════════════════

def get_xgb_model(max_depth=5, learning_rate=0.05, n_estimators=300):
    try:
        import xgboost as xgb
        try:
            m = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', tree_method='hist', device='cuda',
                random_state=42, verbosity=0)
            m.fit(np.random.randn(10, 5), np.random.randint(0, 2, 10))
            return xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', tree_method='hist', device='cuda',
                random_state=42, verbosity=0)
        except Exception:
            return xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', tree_method='hist', random_state=42, verbosity=0)
    except ImportError:
        return None


def walk_forward_eval(X, y, entry_times, model, feature_cols=None):
    from sklearn.metrics import roc_auc_score
    X_use = X[[c for c in (feature_cols or X.columns) if c in X.columns]].copy() if feature_cols else X.copy()
    et = entry_times.copy()
    if hasattr(et, 'dt') and et.dt.tz is not None: et = et.dt.tz_localize(None)
    elif hasattr(et, 'tz') and et.tz is not None: et = et.tz_localize(None)
    oos_preds = np.full(len(y), np.nan); fold_aucs = []
    for _, fold_start, fold_end in FOLDS:
        fs = pd.Timestamp(fold_start); fe = pd.Timestamp(fold_end)
        train_mask = et < fs; test_mask = (et >= fs) & (et < fe)
        if train_mask.sum() < 50 or test_mask.sum() < 20: continue
        Xtr = X_use[train_mask].copy(); ytr = y[train_mask]
        Xte = X_use[test_mask].copy(); yte = y[test_mask]
        med = Xtr.median(); Xtr = Xtr.fillna(med); Xte = Xte.fillna(med)
        const = [c for c in Xtr.columns if Xtr[c].nunique() <= 1]
        if const: Xtr = Xtr.drop(columns=const); Xte = Xte.drop(columns=const)
        if len(Xtr.columns) == 0: continue
        try:
            m = deepcopy(model); m.fit(Xtr, ytr)
            probs = m.predict_proba(Xte)[:, 1]
            oos_preds[np.where(test_mask)[0]] = probs
            if len(np.unique(yte)) > 1: fold_aucs.append(roc_auc_score(yte, probs))
        except Exception: continue
    valid = ~np.isnan(oos_preds); overall_auc = None
    if valid.sum() > 20:
        yv = y[valid]; pv = oos_preds[valid]
        if len(np.unique(yv)) > 1: overall_auc = float(roc_auc_score(yv, pv))
    return {'overall_auc': overall_auc, 'fold_aucs': fold_aucs, 'oos_preds': oos_preds}


# ═══════════════════════════════════════════════════════════════
# 5 Tests (condensed)
# ═══════════════════════════════════════════════════════════════

def run_5_tests(X, y, entry_times, trades, valid_indices, strat_name):
    from sklearn.metrics import roc_auc_score
    results = {}
    et = entry_times.copy()
    if hasattr(et, 'dt') and et.dt.tz is not None: et = et.dt.tz_localize(None)
    elif hasattr(et, 'tz') and et.tz is not None: et = et.tz_localize(None)

    # TEST 1: Independent Holdout
    print(f"\n    [Test 1] Independent Holdout...", flush=True)
    model = get_xgb_model()
    split_ts = pd.Timestamp(HOLDOUT_SPLIT)
    train_mask = et < split_ts; test_mask = et >= split_ts
    n_train = int(train_mask.sum()); n_test = int(test_mask.sum())
    if n_train >= 50 and n_test >= 20 and model is not None:
        Xtr = X[train_mask].copy(); ytr = y[train_mask]
        Xte = X[test_mask].copy(); yte = y[test_mask]
        med = Xtr.median(); Xtr = Xtr.fillna(med); Xte = Xte.fillna(med)
        const = [c for c in Xtr.columns if Xtr[c].nunique() <= 1]
        if const: Xtr = Xtr.drop(columns=const); Xte = Xte.drop(columns=const)
        m = deepcopy(model); m.fit(Xtr, ytr)
        probs = m.predict_proba(Xte)[:, 1]
        auc = float(roc_auc_score(yte, probs)) if len(np.unique(yte)) > 1 else None
        # Filtered Sharpe
        test_ti = [valid_indices[i] for i in range(len(valid_indices)) if test_mask.iloc[i]]
        holdout_trades = [trades[ti] for ti in test_ti]
        filtered_trades = [holdout_trades[i] for i, p in enumerate(probs) if p >= 0.5]
        base_sh = _compute_stats(holdout_trades)['sharpe']
        filt_sh = _compute_stats(filtered_trades)['sharpe']
        pct = ((filt_sh - base_sh) / base_sh * 100) if base_sh != 0 else 0
        passed = auc is not None and auc > 0.55 and filt_sh > base_sh
        print(f"      Train={n_train}, Test={n_test}, AUC={auc:.4f}, "
              f"Sharpe: {base_sh:.3f} -> {filt_sh:.3f} ({pct:+.1f}%) {'PASS' if passed else 'FAIL'}")
        results['test_1'] = {'passed': passed, 'auc': round(auc, 4) if auc else None,
                             'base_sharpe': base_sh, 'filt_sharpe': filt_sh, 'pct': round(pct, 1)}
    else:
        print(f"      SKIP (train={n_train}, test={n_test})")
        results['test_1'] = {'passed': False, 'reason': 'insufficient_data'}

    # TEST 2: Parameter Perturbation
    print(f"    [Test 2] Param Perturbation...", flush=True)
    aucs = []
    for params in PARAM_GRID:
        m = get_xgb_model(**params)
        if m is None: continue
        r = walk_forward_eval(X, y, entry_times, m)
        if r['overall_auc'] is not None: aucs.append(r['overall_auc'])
    if len(aucs) >= 3:
        std_a = float(np.std(aucs)); min_a = float(np.min(aucs))
        passed = std_a < 0.05 and min_a > 0.52
        print(f"      AUCs: {[round(a,4) for a in aucs]}")
        print(f"      mean={np.mean(aucs):.4f}, std={std_a:.4f}, min={min_a:.4f} {'PASS' if passed else 'FAIL'}")
        results['test_2'] = {'passed': passed, 'mean': round(float(np.mean(aucs)), 4),
                             'std': round(std_a, 4), 'min': round(min_a, 4), 'aucs': [round(a,4) for a in aucs]}
    else:
        results['test_2'] = {'passed': False, 'reason': 'insufficient_results'}

    # TEST 3: Random Label Shuffle
    print(f"    [Test 3] Random Shuffle ({N_SHUFFLE}x)...", flush=True)
    model = get_xgb_model()
    real_r = walk_forward_eval(X, y, entry_times, model)
    real_auc = real_r['overall_auc']
    if real_auc is not None and model is not None:
        shuffle_aucs = []
        for i in range(N_SHUFFLE):
            ys = np.random.permutation(y)
            ms = get_xgb_model()
            rs = walk_forward_eval(X, ys, entry_times, ms)
            if rs['overall_auc'] is not None: shuffle_aucs.append(rs['overall_auc'])
        if len(shuffle_aucs) >= 5:
            sm = float(np.mean(shuffle_aucs)); ss = float(np.std(shuffle_aucs))
            z = (real_auc - sm) / ss if ss > 0 else 0
            p = sum(1 for s in shuffle_aucs if s >= real_auc) / len(shuffle_aucs)
            passed = z > 2.0 and p < 0.10
            print(f"      Real AUC={real_auc:.4f}, Shuffle mean={sm:.4f}+/-{ss:.4f}, z={z:.2f}, p={p:.3f} {'PASS' if passed else 'FAIL'}")
            results['test_3'] = {'passed': passed, 'real_auc': round(real_auc, 4),
                                 'shuffle_mean': round(sm, 4), 'z': round(z, 2), 'p': round(p, 3)}
        else:
            results['test_3'] = {'passed': False, 'reason': 'insufficient_shuffles'}
    else:
        results['test_3'] = {'passed': False, 'reason': 'no_real_auc'}

    # TEST 4: Feature Ablation
    print(f"    [Test 4] Feature Ablation...", flush=True)
    ablation = {}
    for gname, fcols in [('tech', TECH_FEATURES), ('macro', MACRO_FEATURES),
                          ('time', TIME_FEATURES), ('full', None)]:
        m = get_xgb_model()
        if m is None: continue
        r = walk_forward_eval(X, y, entry_times, m, feature_cols=fcols)
        ablation[gname] = r['overall_auc']
        print(f"      {gname:>6}: AUC={r['overall_auc']:.4f}" if r['overall_auc'] else f"      {gname:>6}: AUC=N/A")
    full_a = ablation.get('full'); tech_a = ablation.get('tech'); time_a = ablation.get('time')
    ext_adds = full_a is not None and tech_a is not None and full_a > tech_a
    time_low = time_a is not None and time_a < 0.58
    passed = (full_a is not None and full_a > 0.55) and time_low
    print(f"      full>tech: {ext_adds}, time<0.58: {time_low} {'PASS' if passed else 'FAIL'}")
    results['test_4'] = {'passed': passed, 'full': round(full_a, 4) if full_a else None,
                         'tech': round(tech_a, 4) if tech_a else None,
                         'time': round(time_a, 4) if time_a else None,
                         'macro': round(ablation.get('macro', 0) or 0, 4)}

    # TEST 5: Per-Fold Stability
    print(f"    [Test 5] Fold Stability...", flush=True)
    model = get_xgb_model()
    r = walk_forward_eval(X, y, entry_times, model)
    fa = r['fold_aucs']
    if len(fa) >= 3:
        mn = float(np.mean(fa)); st = float(np.std(fa)); cv = st/mn if mn > 0 else 999
        mi = float(np.min(fa))
        passed = mi > 0.52 and cv < 0.25
        print(f"      Folds: {[round(a,4) for a in fa]}, min={mi:.4f}, CV={cv:.4f} {'PASS' if passed else 'FAIL'}")
        results['test_5'] = {'passed': passed, 'folds': [round(a,4) for a in fa],
                             'min': round(mi, 4), 'cv': round(cv, 4)}
    else:
        results['test_5'] = {'passed': False, 'reason': f'only_{len(fa)}_folds'}

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("  R92-B: ML Exit Robustness — PSAR / SESS_BO / L8_MAX")
    print("  Strategies with 2,000-22,000 trades (vs TSMOM's 121)")
    print("=" * 70, flush=True)

    print("\n  Loading data...", flush=True)
    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH, DataBundle
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0].date()} ~ {h1_df.index[-1].date()})")

    ext_path = Path("data/external/aligned_daily.csv")
    external_daily = None
    if ext_path.exists():
        external_daily = pd.read_csv(ext_path, parse_dates=['Date'], index_col='Date')
        external_daily.index = external_daily.index.normalize()
        if external_daily.index.tz is not None:
            external_daily.index = external_daily.index.tz_localize(None)
        print(f"    External daily: {len(external_daily)} rows")

    print("  Computing H1 indicators...", flush=True)
    h1_indicators = compute_h1_indicators(h1_df)

    # Load L8 bundle (needed for L8_MAX full engine backtest)
    print("  Loading L8_MAX DataBundle...", flush=True)
    try:
        l8_bundle = DataBundle.load_custom()
        print("    Done.")
    except Exception as e:
        print(f"    WARNING: Cannot load DataBundle: {e}")
        print("    L8_MAX will be skipped.")
        l8_bundle = None

    all_strategy_results = {}

    for strat_name in STRATEGIES_TO_TEST:
        print(f"\n{'='*70}")
        print(f"  STRATEGY: {strat_name}")
        print(f"{'='*70}", flush=True)

        # Run backtest
        cap = CAPS[strat_name]; lot = LOTS[strat_name]
        if strat_name == 'PSAR':
            trades = bt_psar(h1_df, SPREAD, lot, cap)
        elif strat_name == 'SESS_BO':
            trades = bt_sess_bo(h1_df, SPREAD, lot, cap)
        elif strat_name == 'L8_MAX':
            if l8_bundle is None:
                print("    SKIP: DataBundle not available")
                all_strategy_results[strat_name] = {'n_passed': 0, 'reason': 'data_unavailable'}
                continue
            trades = bt_l8_max(l8_bundle, SPREAD, lot, cap)

        stats = _compute_stats(trades)
        n_loss = sum(1 for t in trades if t['pnl'] <= 0)
        print(f"    Trades: {stats['n']}, WR={stats['wr']:.1f}%, Sharpe={stats['sharpe']:.3f}")
        print(f"    Losses: {n_loss} ({n_loss/max(stats['n'],1)*100:.1f}%)")

        if stats['n'] < 100:
            print(f"    SKIP: Too few trades ({stats['n']} < 100)")
            all_strategy_results[strat_name] = {'n_passed': 0, 'reason': f'too_few_trades_{stats["n"]}'}
            continue

        # Build features
        X, y, valid_indices = build_features_for_trades(trades, h1_indicators, external_daily)
        print(f"    Features: {X.shape[1]} cols, {len(X)} samples (win_rate={y.mean()*100:.1f}%)")

        if len(X) < 100:
            print(f"    SKIP: Too few feature samples ({len(X)} < 100)")
            all_strategy_results[strat_name] = {'n_passed': 0, 'reason': f'too_few_features_{len(X)}'}
            continue

        entry_times = pd.Series([pd.Timestamp(trades[vi]['entry_time']) for vi in valid_indices])

        # Run 5 tests
        test_results = run_5_tests(X, y, entry_times, trades, valid_indices, strat_name)
        n_passed = sum(1 for v in test_results.values() if v.get('passed', False))
        test_results['n_passed'] = n_passed
        test_results['n_trades'] = stats['n']
        test_results['n_losses'] = n_loss
        test_results['baseline_sharpe'] = stats['sharpe']
        all_strategy_results[strat_name] = test_results

        verdict = "ROBUST" if n_passed >= 4 else ("MARGINAL" if n_passed >= 3 else "NOT ROBUST")
        print(f"\n    >>> {strat_name}: {n_passed}/5 PASSED --> {verdict}")

    # Final Summary
    elapsed = time.time() - t0
    print(f"\n\n{'='*70}")
    print("  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"\n  {'Strategy':<12} {'Trades':>7} {'Losses':>7} {'T1':>5} {'T2':>5} {'T3':>5} {'T4':>5} {'T5':>5} {'Score':>6} {'Verdict':<12}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*12}")

    for strat in STRATEGIES_TO_TEST:
        r = all_strategy_results[strat]
        nt = r['n_trades']; nl = r['n_losses']; np_ = r['n_passed']
        t1 = 'P' if r.get('test_1', {}).get('passed') else 'F'
        t2 = 'P' if r.get('test_2', {}).get('passed') else 'F'
        t3 = 'P' if r.get('test_3', {}).get('passed') else 'F'
        t4 = 'P' if r.get('test_4', {}).get('passed') else 'F'
        t5 = 'P' if r.get('test_5', {}).get('passed') else 'F'
        verdict = "ROBUST" if np_ >= 4 else ("MARGINAL" if np_ >= 3 else "FAIL")
        print(f"  {strat:<12} {nt:>7} {nl:>7} {t1:>5} {t2:>5} {t3:>5} {t4:>5} {t5:>5} {np_:>4}/5 {verdict:<12}")

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}", flush=True)

    output = {'strategies': all_strategy_results, 'elapsed_s': round(elapsed, 1)}
    with open(OUTPUT_DIR / "r92b_results.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_DIR}/r92b_results.json", flush=True)


if __name__ == "__main__":
    main()
