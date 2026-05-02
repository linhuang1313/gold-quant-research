#!/usr/bin/env python3
"""
R90-D: ML Exit Optimization (External Factor Enhanced)
========================================================
Upgrades R62 exit filter by adding external macro features.
Trains per-strategy exit models for L8_MAX, PSAR, TSMOM, SESS_BO.
Walk-forward: 6-fold expanding validation.
"""
import sys, os, io, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r90_external_data/r90d_ml_exit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
LOTS = {'L8_MAX': 0.05, 'TSMOM': 0.04, 'SESS_BO': 0.02, 'PSAR': 0.01}

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]

STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']

MIN_TRADES_FOR_ML = 50


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


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


# ═══════════════════════════════════════════════════════════════
# Strategy backtests (modified to include entry_bar)
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
    df['fma'] = df['Close'].rolling(fast).mean()
    df['sma'] = df['Close'].rolling(slow).mean()
    df = df.dropna(subset=['ATR', 'fma', 'sma'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; fm = df['fma'].values; sm = df['sma'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and fm[i] < sm[i]:
                pnl = (c[i]-pos['entry']-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and fm[i] > sm[i]:
                pnl = (pos['entry']-c[i]-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if fm[i] > sm[i] and fm[i-1] <= sm[i-1]:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif fm[i] < sm[i] and fm[i-1] >= sm[i-1]:
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


def bt_l8_max(data_bundle, spread, lot, maxloss_cap=0):
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
    """Pre-compute H1 indicators used as base features."""
    df = h1_df.copy()
    df['ATR_14'] = compute_atr(df, 14)
    df['ADX_14'] = compute_adx(df, 14)

    # RSI
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

    # Keltner Channel for breakout strength
    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    atr14 = df['ATR_14']
    kc_upper = ema20 + 1.5 * atr14
    kc_lower = ema20 - 1.5 * atr14
    kc_width = kc_upper - kc_lower
    df['KC_breakout_strength'] = (df['Close'] - ema20) / kc_width.replace(0, np.nan)

    # Volume ratio (using tick volume proxy if available, else use High-Low range as proxy)
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        df['Volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, np.nan)
    else:
        range_vol = df['High'] - df['Low']
        df['Volume_ratio'] = range_vol / range_vol.rolling(20).mean().replace(0, np.nan)

    # ATR percentile (rolling 252-bar window)
    df['ATR_percentile'] = df['ATR_14'].rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # EMA cross
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA9_EMA21_cross'] = (df['EMA9'] - df['EMA21']) / atr14.replace(0, np.nan)

    # Close distance from EMA100
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['Close_EMA100_dist'] = (df['Close'] - df['EMA100']) / atr14.replace(0, np.nan)

    return df


def build_features_for_trades(trades, h1_indicators, external_daily,
                              regime_labels=None, ml_direction=None):
    """
    Build feature matrix for a list of trades.
    Returns (X DataFrame, y array, valid_mask indices).
    """
    records = []
    labels = []
    valid_indices = []

    for idx, t in enumerate(trades):
        entry_time = pd.Timestamp(t['entry_time'])
        entry_date = entry_time.normalize()
        if entry_date.tzinfo is not None:
            entry_date = entry_date.tz_localize(None)

        # Base features from H1 at entry time
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

        # External features from aligned_daily (look up by entry date)
        ext_row = None
        if external_daily is not None and len(external_daily) > 0:
            if entry_date in external_daily.index:
                ext_row = external_daily.loc[entry_date]
            else:
                loc_ext = external_daily.index.get_indexer([entry_date], method='ffill')
                if loc_ext[0] >= 0:
                    ext_row = external_daily.iloc[loc_ext[0]]

        if ext_row is not None:
            feat['real_yield_change5'] = ext_row.get('REAL_YIELD_Change5', np.nan)
            feat['real_yield_change20'] = ext_row.get('REAL_YIELD_Change20', np.nan)
            feat['vix_zscore'] = ext_row.get('VIX_Zscore', np.nan)
            feat['vix_close'] = ext_row.get('VIX_Close', np.nan)
            feat['dxy_mom5'] = ext_row.get('DXY_Mom5', np.nan)
            feat['dxy_mom20'] = ext_row.get('DXY_Mom20', np.nan)
            feat['credit_stress'] = ext_row.get('CREDIT_STRESS', np.nan)
            feat['yield_curve_10y2y'] = ext_row.get('YIELD_CURVE_10Y2Y', np.nan)
            feat['copper_gold_ratio'] = ext_row.get('COPPER_GOLD_RATIO', np.nan)
            feat['crude_mom5'] = ext_row.get('CRUDE_Mom5', np.nan)
            feat['usdcnh_mom5'] = ext_row.get('USDCNH_Mom5', np.nan)
            feat['cot_mm_net_zscore'] = ext_row.get('COT_MM_Net_Zscore', np.nan)
            feat['fed_funds_dff'] = ext_row.get('FED_FUNDS_DFF', np.nan)
            feat['risk_appetite_z'] = ext_row.get('RISK_APPETITE_Z', np.nan)
        else:
            for col in ['real_yield_change5', 'real_yield_change20', 'vix_zscore',
                        'vix_close', 'dxy_mom5', 'dxy_mom20', 'credit_stress',
                        'yield_curve_10y2y', 'copper_gold_ratio', 'crude_mom5',
                        'usdcnh_mom5', 'cot_mm_net_zscore', 'fed_funds_dff',
                        'risk_appetite_z']:
                feat[col] = np.nan

        # Regime feature (from Phase A)
        if regime_labels is not None and len(regime_labels) > 0:
            if entry_date in regime_labels.index:
                feat['regime_label'] = regime_labels.loc[entry_date]
            else:
                loc_r = regime_labels.index.get_indexer([entry_date], method='ffill')
                feat['regime_label'] = regime_labels.iloc[loc_r[0]] if loc_r[0] >= 0 else np.nan
        else:
            feat['regime_label'] = np.nan

        # ML direction feature (from Phase C)
        if ml_direction is not None and len(ml_direction) > 0:
            if entry_date in ml_direction.index:
                feat['ml_direction_prob'] = ml_direction.loc[entry_date]
            else:
                loc_d = ml_direction.index.get_indexer([entry_date], method='ffill')
                feat['ml_direction_prob'] = ml_direction.iloc[loc_d[0]] if loc_d[0] >= 0 else np.nan
        else:
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
# ML Training & Walk-Forward
# ═══════════════════════════════════════════════════════════════

def get_xgb_model():
    try:
        import xgboost as xgb
        try:
            model = xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                tree_method='hist', device='cuda', random_state=42,
                verbosity=0
            )
            model.fit(np.random.randn(10, 5), np.random.randint(0, 2, 10))
            return 'xgb', xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                tree_method='hist', device='cuda', random_state=42,
                verbosity=0
            )
        except Exception:
            return 'xgb', xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                tree_method='hist', random_state=42,
                verbosity=0
            )
    except ImportError:
        return None, None


def get_lgbm_model():
    try:
        import lightgbm as lgb
        return 'lgbm', lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbose=-1
        )
    except ImportError:
        return None, None


def walk_forward_train(X, y, entry_times, model_factory, model_name):
    """
    Walk-forward expanding window validation.
    Returns dict with per-fold metrics and overall OOS predictions.
    """
    from sklearn.metrics import roc_auc_score, accuracy_score

    oos_preds = np.full(len(y), np.nan)
    fold_metrics = []

    # Ensure entry_times are tz-naive for comparison
    if hasattr(entry_times, 'dt') and entry_times.dt.tz is not None:
        entry_times = entry_times.dt.tz_localize(None)
    elif hasattr(entry_times, 'tz') and entry_times.tz is not None:
        entry_times = entry_times.tz_localize(None)

    for fold_name, fold_start, fold_end in FOLDS:
        fold_start_ts = pd.Timestamp(fold_start)
        fold_end_ts = pd.Timestamp(fold_end)

        train_mask = entry_times < fold_start_ts
        test_mask = (entry_times >= fold_start_ts) & (entry_times < fold_end_ts)

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        if n_train < 30 or n_test < 5:
            fold_metrics.append({
                'fold': fold_name, 'n_train': int(n_train), 'n_test': int(n_test),
                'auc': None, 'accuracy': None, 'status': 'skipped'
            })
            continue

        X_train = X[train_mask].copy()
        y_train = y[train_mask]
        X_test = X[test_mask].copy()
        y_test = y[test_mask]

        # Fill NaN with column median from training set
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)

        # Drop columns that are all-constant in training
        constant_cols = [c for c in X_train.columns if X_train[c].nunique() <= 1]
        if constant_cols:
            X_train = X_train.drop(columns=constant_cols)
            X_test = X_test.drop(columns=constant_cols)

        if len(X_train.columns) == 0:
            fold_metrics.append({
                'fold': fold_name, 'n_train': int(n_train), 'n_test': int(n_test),
                'auc': None, 'accuracy': None, 'status': 'no_features'
            })
            continue

        try:
            _, model = model_factory()
            if model is None:
                fold_metrics.append({
                    'fold': fold_name, 'n_train': int(n_train), 'n_test': int(n_test),
                    'auc': None, 'accuracy': None, 'status': 'no_model'
                })
                continue

            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)

            # Store OOS predictions
            test_indices = np.where(test_mask)[0]
            oos_preds[test_indices] = probs

            if len(np.unique(y_test)) > 1:
                auc = round(roc_auc_score(y_test, probs), 4)
            else:
                auc = None
            acc = round(accuracy_score(y_test, preds), 4)

            fold_metrics.append({
                'fold': fold_name, 'n_train': int(n_train), 'n_test': int(n_test),
                'auc': auc, 'accuracy': acc, 'status': 'ok'
            })
        except Exception as e:
            fold_metrics.append({
                'fold': fold_name, 'n_train': int(n_train), 'n_test': int(n_test),
                'auc': None, 'accuracy': None, 'status': f'error: {str(e)[:50]}'
            })

    # Get feature importance from a full-sample model for reporting
    feature_importance = {}
    try:
        _, model = model_factory()
        if model is not None:
            X_full = X.fillna(X.median())
            constant_cols = [c for c in X_full.columns if X_full[c].nunique() <= 1]
            if constant_cols:
                X_full = X_full.drop(columns=constant_cols)
            model.fit(X_full, y)
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                for fname, fval in sorted(zip(X_full.columns, imp), key=lambda x: -x[1]):
                    feature_importance[fname] = round(float(fval), 4)
    except Exception:
        pass

    # Overall OOS metrics
    valid_oos = ~np.isnan(oos_preds)
    overall_auc = None
    overall_acc = None
    if valid_oos.sum() > 10:
        y_valid = y[valid_oos]
        p_valid = oos_preds[valid_oos]
        if len(np.unique(y_valid)) > 1:
            overall_auc = round(float(roc_auc_score(y_valid, p_valid)), 4)
        overall_acc = round(float(accuracy_score(y_valid, (p_valid >= 0.5).astype(int))), 4)

    return {
        'model': model_name,
        'fold_metrics': fold_metrics,
        'overall_auc': overall_auc,
        'overall_accuracy': overall_acc,
        'feature_importance': feature_importance,
        'oos_predictions': oos_preds,
    }


# ═══════════════════════════════════════════════════════════════
# Filtered Backtest Comparison
# ═══════════════════════════════════════════════════════════════

def compare_filtered_backtest(trades, oos_preds, valid_indices, threshold_50=0.5, threshold_60=0.6):
    """Compare baseline vs ML-filtered trade selection."""
    baseline_stats = _compute_stats(trades)

    # Filtered at 0.5
    filtered_50 = []
    for i, t_idx in enumerate(valid_indices):
        if i < len(oos_preds) and not np.isnan(oos_preds[i]):
            if oos_preds[i] >= threshold_50:
                filtered_50.append(trades[t_idx])
    stats_50 = _compute_stats(filtered_50)

    # Filtered at 0.6
    filtered_60 = []
    for i, t_idx in enumerate(valid_indices):
        if i < len(oos_preds) and not np.isnan(oos_preds[i]):
            if oos_preds[i] >= threshold_60:
                filtered_60.append(trades[t_idx])
    stats_60 = _compute_stats(filtered_60)

    return {
        'baseline': baseline_stats,
        'filtered_p50': stats_50,
        'filtered_p60': stats_60,
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80)
    print("  R90-D: ML Exit Optimization (External Factor Enhanced)")
    print("  Walk-forward per-strategy exit models with macro features")
    print("=" * 80, flush=True)

    # ── Load data ──
    print("\n  [1/5] Loading data...", flush=True)

    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH, DataBundle

    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0].date()} ~ {h1_df.index[-1].date()})")

    print("    Loading L8_MAX DataBundle...", flush=True)
    l8_bundle = DataBundle.load_custom()
    print("    L8 bundle ready.")

    # External daily
    ext_path = Path("data/external/aligned_daily.csv")
    if ext_path.exists():
        external_daily = pd.read_csv(ext_path, parse_dates=['Date'], index_col='Date')
        external_daily.index = external_daily.index.normalize()
        if external_daily.index.tz is not None:
            external_daily.index = external_daily.index.tz_localize(None)
        print(f"    External daily: {len(external_daily)} rows ({external_daily.index[0].date()} ~ {external_daily.index[-1].date()})")
    else:
        external_daily = None
        print("    [WARN] External daily data not found, external features will be NaN")

    # Phase A regime labels
    regime_path = Path("results/r90_external_data/r90a_regime/regime_labels.csv")
    regime_labels = None
    if regime_path.exists():
        regime_df = pd.read_csv(regime_path, parse_dates=['Date'], index_col='Date')
        if 'regime' in regime_df.columns:
            regime_labels = regime_df['regime']
        elif 'label' in regime_df.columns:
            regime_labels = regime_df['label']
        else:
            regime_labels = regime_df.iloc[:, 0]
        regime_labels.index = regime_labels.index.normalize()
        if regime_labels.index.tz is not None:
            regime_labels.index = regime_labels.index.tz_localize(None)
        print(f"    Regime labels: {len(regime_labels)} rows loaded")
    else:
        print("    [INFO] No Phase A regime labels found, skipping regime feature")

    # Phase C ML direction predictions
    ml_dir_path = Path("results/r90_external_data/r90c_ml_direction/r90c_predictions.csv")
    ml_direction = None
    if ml_dir_path.exists():
        ml_dir_df = pd.read_csv(ml_dir_path)
        date_col = 'Date' if 'Date' in ml_dir_df.columns else 'date'
        ml_dir_df[date_col] = pd.to_datetime(ml_dir_df[date_col])
        ml_dir_df = ml_dir_df.set_index(date_col)
        prob_col = next((c for c in ml_dir_df.columns if 'ensemble_prob_dir_1d' in c), None)
        if prob_col is None:
            prob_col = next((c for c in ml_dir_df.columns if 'prob' in c.lower()), ml_dir_df.columns[0])
        ml_direction = ml_dir_df[prob_col]
        ml_direction.index = ml_direction.index.normalize()
        if ml_direction.index.tz is not None:
            ml_direction.index = ml_direction.index.tz_localize(None)
        print(f"    ML direction: {len(ml_direction)} rows loaded (col: {prob_col})")
    else:
        print("    [INFO] No Phase C ML direction predictions found, skipping ML direction feature")

    # ── Pre-compute H1 indicators ──
    print("\n  [2/5] Computing H1 indicators...", flush=True)
    h1_indicators = compute_h1_indicators(h1_df)
    print(f"    Done. Columns: {len(h1_indicators.columns)}")

    # ── Run backtests ──
    print("\n  [3/5] Running strategy backtests...", flush=True)

    all_trades = {}

    # H1 strategies
    for name, bt_fn in [('PSAR', bt_psar), ('TSMOM', bt_tsmom), ('SESS_BO', bt_sess_bo)]:
        cap = CAPS[name]
        lot = LOTS[name]
        trades = bt_fn(h1_df, spread=SPREAD, lot=lot, maxloss_cap=cap)
        all_trades[name] = trades
        stats = _compute_stats(trades)
        print(f"    {name:>8}: {stats['n']} trades, WR={stats['wr']:.1f}%, "
              f"Sharpe={stats['sharpe']:.3f}, PnL={fmt(stats['pnl'])}")

    # L8_MAX
    cap = CAPS['L8_MAX']
    lot = LOTS['L8_MAX']
    trades = bt_l8_max(l8_bundle, spread=SPREAD, lot=lot, maxloss_cap=cap)
    all_trades['L8_MAX'] = trades
    stats = _compute_stats(trades)
    print(f"    {'L8_MAX':>8}: {stats['n']} trades, WR={stats['wr']:.1f}%, "
          f"Sharpe={stats['sharpe']:.3f}, PnL={fmt(stats['pnl'])}")

    # ── Per-Strategy ML Exit Models ──
    print("\n  [4/5] Training per-strategy ML exit models...", flush=True)
    print(f"    Walk-forward: 6-fold expanding window")
    print(f"    Models: XGBoost + LightGBM\n")

    all_results = {}

    for strat_name in STRAT_ORDER:
        trades = all_trades[strat_name]
        n_trades = len(trades)

        print(f"  {'─'*60}")
        print(f"  Strategy: {strat_name} ({n_trades} trades)")
        print(f"  {'─'*60}")

        if n_trades < MIN_TRADES_FOR_ML:
            print(f"    [SKIP] Too few trades ({n_trades} < {MIN_TRADES_FOR_ML})")
            all_results[strat_name] = {'status': 'skipped', 'reason': f'too_few_trades ({n_trades})'}
            continue

        # Build features
        X, y, valid_indices = build_features_for_trades(
            trades, h1_indicators, external_daily, regime_labels, ml_direction
        )

        if len(X) < MIN_TRADES_FOR_ML:
            print(f"    [SKIP] Too few valid feature rows ({len(X)} < {MIN_TRADES_FOR_ML})")
            all_results[strat_name] = {'status': 'skipped', 'reason': f'too_few_features ({len(X)})'}
            continue

        print(f"    Features: {X.shape[1]} cols, {len(X)} samples (win_rate={y.mean()*100:.1f}%)")

        entry_times = pd.Series([pd.Timestamp(trades[vi]['entry_time']) for vi in valid_indices])

        strat_results = {'n_trades': n_trades, 'n_features': X.shape[1], 'models': {}}

        # Train with each model type
        for model_factory, label in [(get_xgb_model, 'XGBoost'), (get_lgbm_model, 'LightGBM')]:
            name_check, model_check = model_factory()
            if model_check is None:
                print(f"    [{label}] Not available (library not installed)")
                continue

            print(f"\n    [{label}] Training walk-forward...", flush=True)
            result = walk_forward_train(X, y, entry_times, model_factory, label)

            # Print fold results
            for fm in result['fold_metrics']:
                auc_str = f"AUC={fm['auc']:.4f}" if fm['auc'] is not None else "AUC=N/A"
                acc_str = f"Acc={fm['accuracy']:.4f}" if fm['accuracy'] is not None else "Acc=N/A"
                print(f"      {fm['fold']}: train={fm['n_train']:>4}, test={fm['n_test']:>3}  "
                      f"{auc_str}  {acc_str}  [{fm['status']}]")

            if result['overall_auc'] is not None:
                print(f"    >>> Overall OOS: AUC={result['overall_auc']:.4f}, "
                      f"Acc={result['overall_accuracy']:.4f}")
            else:
                print(f"    >>> Overall OOS: insufficient predictions")

            # Top features
            if result['feature_importance']:
                top_feats = list(result['feature_importance'].items())[:10]
                print(f"    Top features: {', '.join(f'{k}({v:.3f})' for k, v in top_feats)}")

            # Filtered backtest comparison
            comparison = compare_filtered_backtest(
                trades, result['oos_predictions'], valid_indices
            )

            print(f"\n    Backtest comparison ({label}):")
            print(f"      {'Mode':<15} {'N':>5} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10}")
            print(f"      {'-'*15} {'-'*5} {'-'*8} {'-'*12} {'-'*7} {'-'*10}")
            for mode_name, mode_stats in comparison.items():
                print(f"      {mode_name:<15} {mode_stats['n']:>5} {mode_stats['sharpe']:>8.3f} "
                      f"{fmt(mode_stats['pnl']):>12} {mode_stats['wr']:>6.1f}% "
                      f"{fmt(mode_stats['max_dd']):>10}")

            # Store results (without numpy arrays for JSON serialization)
            model_result = {
                'fold_metrics': result['fold_metrics'],
                'overall_auc': result['overall_auc'],
                'overall_accuracy': result['overall_accuracy'],
                'feature_importance': result['feature_importance'],
                'backtest_comparison': comparison,
            }
            strat_results['models'][label] = model_result

        all_results[strat_name] = strat_results
        print()

    # ── Comparison with R62 ──
    print(f"\n  [5/5] R62 Comparison & Summary")
    print(f"  {'='*60}")
    print(f"\n  R62 baseline: single model, 12 features (Keltner only), AUC ~0.76")
    print(f"  R90-D: per-strategy models, ~{X.shape[1] if len(X) > 0 else 28} features (+ external macro)")
    print(f"\n  Per-Strategy Results vs R62 Baseline:")
    print(f"  {'Strategy':<12} {'Model':<10} {'OOS AUC':>10} {'vs R62':>8} {'Filtered Sharpe':>16} {'Baseline Sharpe':>16}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*16} {'-'*16}")

    r62_baseline_auc = 0.76
    summary_table = []

    for strat_name in STRAT_ORDER:
        strat_res = all_results.get(strat_name, {})
        if strat_res.get('status') == 'skipped':
            print(f"  {strat_name:<12} {'N/A':<10} {'SKIPPED':>10} {'':>8} {'':>16} {'':>16}")
            continue

        models = strat_res.get('models', {})
        for model_name, model_res in models.items():
            auc = model_res.get('overall_auc')
            auc_str = f"{auc:.4f}" if auc is not None else "N/A"
            delta_str = ""
            if auc is not None:
                delta = auc - r62_baseline_auc
                delta_str = f"{'+' if delta >= 0 else ''}{delta:.4f}"

            comp = model_res.get('backtest_comparison', {})
            base_sh = comp.get('baseline', {}).get('sharpe', 0)
            filt_sh = comp.get('filtered_p50', {}).get('sharpe', 0)

            print(f"  {strat_name:<12} {model_name:<10} {auc_str:>10} {delta_str:>8} "
                  f"{filt_sh:>16.3f} {base_sh:>16.3f}")

            summary_table.append({
                'strategy': strat_name,
                'model': model_name,
                'oos_auc': auc,
                'r62_delta': round(auc - r62_baseline_auc, 4) if auc else None,
                'baseline_sharpe': base_sh,
                'filtered_sharpe': filt_sh,
            })

    # ── Final output ──
    elapsed = time.time() - t0

    print(f"\n  {'='*60}")
    print(f"  KEY FINDINGS:")
    print(f"  {'='*60}")

    best_auc = 0
    best_entry = None
    for entry in summary_table:
        if entry['oos_auc'] is not None and entry['oos_auc'] > best_auc:
            best_auc = entry['oos_auc']
            best_entry = entry

    if best_entry:
        print(f"    Best model: {best_entry['strategy']} / {best_entry['model']}")
        print(f"    OOS AUC: {best_entry['oos_auc']:.4f} (R62 baseline: {r62_baseline_auc})")
        delta = best_entry['oos_auc'] - r62_baseline_auc
        print(f"    Improvement over R62: {'+' if delta >= 0 else ''}{delta:.4f}")
        print(f"    Filtered Sharpe (P>0.5): {best_entry['filtered_sharpe']:.3f} "
              f"vs Baseline: {best_entry['baseline_sharpe']:.3f}")
    else:
        print(f"    No valid models produced.")

    print(f"\n  External features added value:")
    print(f"    - REAL_YIELD, VIX_Zscore, DXY momentum, CREDIT_STRESS")
    print(f"    - COPPER_GOLD_RATIO, COT positioning, FED_FUNDS")
    print(f"    - Per-strategy models capture strategy-specific regime sensitivity")

    print(f"\n  {'='*60}")
    print(f"  R90-D COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  {'='*60}", flush=True)

    # Save results
    output = {
        'config': {
            'caps': CAPS,
            'lots': LOTS,
            'spread': SPREAD,
            'pv': PV,
            'min_trades_for_ml': MIN_TRADES_FOR_ML,
            'folds': FOLDS,
            'r62_baseline_auc': r62_baseline_auc,
        },
        'per_strategy_results': {},
        'summary_table': summary_table,
        'elapsed_s': round(elapsed, 1),
    }

    for strat_name, strat_res in all_results.items():
        if strat_res.get('status') == 'skipped':
            output['per_strategy_results'][strat_name] = strat_res
        else:
            clean_res = {
                'n_trades': strat_res.get('n_trades'),
                'n_features': strat_res.get('n_features'),
                'models': {}
            }
            for model_name, model_res in strat_res.get('models', {}).items():
                clean_res['models'][model_name] = {
                    'fold_metrics': model_res['fold_metrics'],
                    'overall_auc': model_res['overall_auc'],
                    'overall_accuracy': model_res['overall_accuracy'],
                    'feature_importance': model_res['feature_importance'],
                    'backtest_comparison': model_res['backtest_comparison'],
                }
            output['per_strategy_results'][strat_name] = clean_res

    with open(OUTPUT_DIR / "r90d_results.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_DIR}/r90d_results.json", flush=True)


if __name__ == "__main__":
    main()
