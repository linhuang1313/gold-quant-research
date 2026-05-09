#!/usr/bin/env python3
"""
R107 — ML Entry Direction Prediction
=======================================
XGBoost/LightGBM models to predict gold price direction using macro + technical features.
Walk-forward validation with expanding window.

  Phase 1: Feature engineering (macro + technical)
  Phase 2: Walk-forward XGBoost training (12 folds)
  Phase 3: Walk-forward LightGBM training (12 folds)
  Phase 4: Strategy integration — ML as direction filter
  Phase 5: Robustness tests (shuffle, ablation, perturbation)
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import DataBundle, load_m15, load_h1_aligned, H1_CSV_PATH
from backtest.runner import run_variant, LIVE_PARITY_KWARGS

OUTPUT_DIR = Path("results/r107_ml_entry")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']

ALIGNED_CSV = Path("data/external/aligned_daily.csv")

t0 = time.time()


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

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


# ═══════════════════════════════════════════════════════════════
# Strategy backtests
# ═══════════════════════════════════════════════════════════════

def add_psar(df, af_step=0.01, af_max=0.05):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    n = len(df)
    psar = np.empty(n); psar[:] = np.nan
    af = af_step; rising = True; ep = h[0]; psar[0] = l[0]
    for i in range(1, n):
        prev = psar[i-1]
        if rising:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], l[i-1], l[max(0,i-2)])
            if l[i] < psar[i]:
                rising = False; psar[i] = ep; ep = l[i]; af = af_step
            else:
                if h[i] > ep: ep = h[i]; af = min(af + af_step, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], h[i-1], h[max(0,i-2)])
            if h[i] > psar[i]:
                rising = True; psar[i] = ep; ep = h[i]; af = af_step
            else:
                if l[i] < ep: ep = l[i]; af = min(af + af_step, af_max)
    df['PSAR'] = psar

def bt_psar(h1_df, spread, lot, maxloss_cap=0,
            sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04, max_hold=20):
    df = h1_df.copy(); add_psar(df); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR', 'PSAR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; psar = df['PSAR'].values; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if psar[i-1] > c[i-1] and psar[i] < c[i]:
            pos = {'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif psar[i-1] < c[i-1] and psar[i] > c[i]:
            pos = {'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
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
            pos = {'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
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
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif c[i] < ll:
            pos = {'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
    return trades

def bt_l8_max(data_bundle, spread, lot, maxloss_cap=35):
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
# Metric helpers
# ═══════════════════════════════════════════════════════════════

def trades_to_daily_series(trades):
    if not trades: return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))

def sharpe(arr):
    if len(arr) < 10: return 0.0
    s = np.std(arr, ddof=1)
    return float(np.mean(arr) / s * np.sqrt(252)) if s > 0 else 0.0

def max_dd(arr):
    if len(arr) == 0: return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())

def build_portfolio_daily(unit_dailies, lots):
    all_dates = set()
    for ds in unit_dailies.values():
        all_dates.update(ds.index)
    all_dates = sorted(all_dates)
    idx = pd.DatetimeIndex(all_dates)
    portfolio = np.zeros(len(idx))
    for name in STRAT_ORDER:
        if name not in unit_dailies: continue
        ds = unit_dailies[name]
        multiplier = lots[name] / UNIT_LOT
        aligned = ds.reindex(idx, fill_value=0.0).values * multiplier
        portfolio += aligned
    return portfolio

def portfolio_metrics(daily_arr):
    return {
        'sharpe': round(sharpe(daily_arr), 3),
        'pnl': round(float(np.sum(daily_arr)), 2),
        'max_dd': round(max_dd(daily_arr), 2),
    }


# ═══════════════════════════════════════════════════════════════
# Feature engineering
# ═══════════════════════════════════════════════════════════════

def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _bollinger_pctb(close, period=20, num_std=2):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    pctb = (close - lower) / (upper - lower).replace(0, np.nan)
    return pctb


def build_features(h1_df):
    """Build daily feature matrix from H1 gold data + optional macro CSV."""
    daily = pd.DataFrame()
    daily['gold_close'] = h1_df['Close'].resample('D').last()
    daily['gold_high'] = h1_df['High'].resample('D').max()
    daily['gold_low'] = h1_df['Low'].resample('D').min()
    daily['gold_open'] = h1_df['Open'].resample('D').first()
    daily = daily.dropna(subset=['gold_close'])
    if daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)

    hourly_ret = h1_df['Close'].pct_change()
    vol_daily = hourly_ret.resample('D').std()
    if vol_daily.index.tz is not None:
        vol_daily.index = vol_daily.index.tz_localize(None)
    daily['hourly_vol_20d'] = vol_daily.reindex(daily.index).rolling(20).mean()

    close = daily['gold_close']
    high = daily['gold_high']
    low = daily['gold_low']

    daily['rsi_14'] = _rsi(close, 14)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    daily['atr_14_rank'] = atr_14.rolling(200, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    daily['bb_pctb'] = _bollinger_pctb(close, 20, 2)

    daily['ret_1d'] = close.pct_change(1)
    daily['ret_5d'] = close.pct_change(5)
    daily['ret_10d'] = close.pct_change(10)
    daily['ret_20d'] = close.pct_change(20)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    daily['sma20_dist'] = (close - sma20) / sma20
    daily['sma50_dist'] = (close - sma50) / sma50

    daily['close_high_ratio'] = close / high.replace(0, np.nan)
    daily['close_low_ratio'] = close / low.replace(0, np.nan)

    macro_cols_used = []
    if ALIGNED_CSV.exists():
        print("    Loading aligned_daily.csv for macro features...")
        macro = pd.read_csv(ALIGNED_CSV, parse_dates=['Date'], index_col='Date')
        macro = macro.sort_index()
        if macro.index.tz is not None:
            macro.index = macro.index.tz_localize(None)

        if 'DXY' in macro.columns:
            dxy = macro['DXY'].reindex(daily.index, method='ffill')
            daily['dxy_ret_5d'] = dxy.pct_change(5)
            daily['dxy_ret_20d'] = dxy.pct_change(20)
            macro_cols_used.extend(['dxy_ret_5d', 'dxy_ret_20d'])

            gold_ret = close.pct_change()
            dxy_ret = dxy.pct_change()
            daily['gold_dxy_corr_20d'] = gold_ret.rolling(20).corr(dxy_ret)
            macro_cols_used.append('gold_dxy_corr_20d')

        if 'VIX' in macro.columns:
            vix = macro['VIX'].reindex(daily.index, method='ffill')
            daily['vix_level'] = vix
            daily['vix_5d_change'] = vix.diff(5)
            macro_cols_used.extend(['vix_level', 'vix_5d_change'])
        elif 'VIX_Close' in macro.columns:
            vix = macro['VIX_Close'].reindex(daily.index, method='ffill')
            daily['vix_level'] = vix
            daily['vix_5d_change'] = vix.diff(5)
            macro_cols_used.extend(['vix_level', 'vix_5d_change'])

        if 'US10Y' in macro.columns and 'US2Y' in macro.columns:
            us10 = macro['US10Y'].reindex(daily.index, method='ffill')
            us2 = macro['US2Y'].reindex(daily.index, method='ffill')
            daily['yield_curve'] = us10 - us2
            macro_cols_used.append('yield_curve')

        if 'SPX' in macro.columns:
            spx = macro['SPX'].reindex(daily.index, method='ffill')
            daily['spx_ret_5d'] = spx.pct_change(5)
            macro_cols_used.append('spx_ret_5d')

            gold_ret = close.pct_change()
            spx_ret = spx.pct_change()
            daily['gold_spx_corr_20d'] = gold_ret.rolling(20).corr(spx_ret)
            macro_cols_used.append('gold_spx_corr_20d')

        if 'crude_wti' in macro.columns:
            crude = macro['crude_wti'].reindex(daily.index, method='ffill')
            daily['crude_ret_5d'] = crude.pct_change(5)
            macro_cols_used.append('crude_ret_5d')
        elif 'WTI' in macro.columns:
            crude = macro['WTI'].reindex(daily.index, method='ffill')
            daily['crude_ret_5d'] = crude.pct_change(5)
            macro_cols_used.append('crude_ret_5d')

    # Targets: next-day and next-5-day return sign
    daily['target_1d'] = (close.shift(-1) > close).astype(int)
    daily['target_5d'] = (close.shift(-5) > close).astype(int)

    tech_features = [
        'rsi_14', 'atr_14_rank', 'bb_pctb',
        'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d',
        'sma20_dist', 'sma50_dist',
        'hourly_vol_20d', 'close_high_ratio', 'close_low_ratio',
    ]
    feature_cols = tech_features + macro_cols_used
    feature_cols = [c for c in feature_cols if c in daily.columns]

    daily = daily.dropna(subset=feature_cols + ['target_1d'])

    print(f"    Features: {len(feature_cols)} cols  "
          f"({len(tech_features)} technical + {len(macro_cols_used)} macro)")
    print(f"    Date range: {daily.index[0]} ~ {daily.index[-1]}  "
          f"({len(daily)} rows)")

    return daily, feature_cols


# ═══════════════════════════════════════════════════════════════
# Walk-forward ML helpers
# ═══════════════════════════════════════════════════════════════

def _split_walk_forward(n_rows, n_folds=12):
    """Divide data into n_folds+1 chunks; for fold k train on 0..k, test on k+1."""
    chunk_size = n_rows // (n_folds + 1)
    folds = []
    for k in range(n_folds):
        train_end = (k + 1) * chunk_size
        test_start = train_end
        test_end = min(test_start + chunk_size, n_rows)
        if test_end <= test_start:
            break
        folds.append((0, train_end, test_start, test_end))
    return folds


def _train_xgb(X_train, y_train, X_test, y_test, params=None):
    """Train XGBoost and return predictions + metrics. Import inside function."""
    try:
        import xgboost as xgb
    except ImportError:
        print("    [WARN] xgboost not installed — skipping")
        return None

    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

    default_params = {
        'max_depth': 4,
        'n_estimators': 200,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'logloss',
        'random_state': 42,
        'verbosity': 0,
    }
    if params:
        default_params.update(params)

    try:
        model = xgb.XGBClassifier(tree_method='hist', device='cuda', **default_params)
        model.fit(X_train, y_train, verbose=False)
    except Exception:
        model = xgb.XGBClassifier(tree_method='hist', **default_params)
        model.fit(X_train, y_train, verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    importances = dict(zip(
        [f"f{i}" for i in range(len(model.feature_importances_))],
        model.feature_importances_.tolist()
    ))
    if hasattr(X_train, 'columns'):
        importances = dict(zip(X_train.columns, model.feature_importances_.tolist()))

    return {
        'model': model,
        'y_prob': y_prob,
        'y_pred': y_pred,
        'auc': auc,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'importances': importances,
    }


def _train_lgb(X_train, y_train, X_test, y_test, params=None):
    """Train LightGBM and return predictions + metrics. Import inside function."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("    [WARN] lightgbm not installed — skipping")
        return None

    from sklearn.metrics import roc_auc_score, accuracy_score

    default_params = {
        'max_depth': 4,
        'n_estimators': 200,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'verbose': -1,
        'random_state': 42,
    }
    if params:
        default_params.update(params)

    try:
        model = lgb.LGBMClassifier(device='gpu', **default_params)
        model.fit(X_train, y_train)
    except Exception:
        model = lgb.LGBMClassifier(**default_params)
        model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
    acc = accuracy_score(y_test, y_pred)

    importances = dict(zip(
        [f"f{i}" for i in range(len(model.feature_importances_))],
        model.feature_importances_.tolist()
    ))
    if hasattr(X_train, 'columns'):
        importances = dict(zip(X_train.columns, model.feature_importances_.tolist()))

    return {
        'model': model,
        'y_prob': y_prob,
        'y_pred': y_pred,
        'auc': auc,
        'accuracy': acc,
        'importances': importances,
    }


# ═══════════════════════════════════════════════════════════════
# Phase 4: Strategy integration helpers
# ═══════════════════════════════════════════════════════════════

def _ml_standalone_pnl(daily_df, feature_cols, model, spread=SPREAD, lot=0.01):
    """ML as standalone strategy: long when P(up)>0.55, short when P(up)<0.45, hold 1 day."""
    X = daily_df[feature_cols].values
    probs = model.predict_proba(X)[:, 1]
    closes = daily_df['gold_close'].values
    pnls = []
    for i in range(len(probs) - 1):
        if probs[i] > 0.55:
            pnl = (closes[i + 1] - closes[i] - spread) * lot * PV
            pnls.append({'date': daily_df.index[i], 'pnl': pnl, 'dir': 'BUY'})
        elif probs[i] < 0.45:
            pnl = (closes[i] - closes[i + 1] - spread) * lot * PV
            pnls.append({'date': daily_df.index[i], 'pnl': pnl, 'dir': 'SELL'})
    return pnls


def _filter_trades_by_ml(trades, daily_df, feature_cols, model):
    """Only allow strategy trades when ML agrees with trade direction."""
    if not trades:
        return []
    X = daily_df[feature_cols].values
    probs = model.predict_proba(X)[:, 1]
    prob_series = pd.Series(probs, index=daily_df.index)

    filtered = []
    for t in trades:
        entry_date = pd.Timestamp(t['entry_time']).normalize()
        if hasattr(entry_date, 'tz') and entry_date.tz is not None:
            entry_date = entry_date.tz_localize(None)
        if prob_series.index.tz is not None:
            prob_series.index = prob_series.index.tz_localize(None)
        idx = prob_series.index.searchsorted(entry_date)
        idx = min(max(idx, 0), len(prob_series) - 1)
        p = prob_series.iloc[idx]
        if t['dir'] == 'BUY' and p > 0.50:
            filtered.append(t)
        elif t['dir'] == 'SELL' and p < 0.50:
            filtered.append(t)
    return filtered


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  R107 — ML Entry Direction Prediction")
    print("=" * 80)

    print("\n  Loading data...")
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    bundle = DataBundle.load_custom()
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0].date()} ~ {h1_df.index[-1].date()})")

    results = {'experiment': 'R107 ML Entry Direction Prediction'}

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Feature Engineering
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 1: Feature Engineering")
    print("=" * 60)

    daily_df, feature_cols = build_features(h1_df)

    X_all = daily_df[feature_cols]
    y_1d = daily_df['target_1d']
    y_5d = daily_df['target_5d'].dropna()

    print(f"    Target class balance (1d): "
          f"up={y_1d.mean():.3f}, down={1 - y_1d.mean():.3f}")

    results['phase1'] = {
        'n_features': len(feature_cols),
        'n_rows': len(daily_df),
        'feature_names': feature_cols,
        'date_start': str(daily_df.index[0].date()),
        'date_end': str(daily_df.index[-1].date()),
        'target_1d_up_pct': round(float(y_1d.mean()), 4),
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Walk-Forward XGBoost (12 folds)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 2: Walk-Forward XGBoost (12 folds)")
    print("=" * 60)

    n_folds = 12
    folds = _split_walk_forward(len(daily_df), n_folds)
    print(f"    {len(folds)} folds, chunk ~{len(daily_df) // (n_folds + 1)} rows each")

    xgb_fold_results = []
    xgb_all_probs = np.full(len(daily_df), np.nan)
    xgb_importances_agg = {}
    xgb_available = True

    for fold_i, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        X_train = X_all.iloc[tr_start:tr_end]
        y_train = y_1d.iloc[tr_start:tr_end]
        X_test = X_all.iloc[te_start:te_end]
        y_test = y_1d.iloc[te_start:te_end]

        res = _train_xgb(X_train, y_train, X_test, y_test)
        if res is None:
            xgb_available = False
            print("    XGBoost not available — skipping Phase 2")
            break

        xgb_all_probs[te_start:te_end] = res['y_prob']

        for feat, imp in res['importances'].items():
            xgb_importances_agg[feat] = xgb_importances_agg.get(feat, 0) + imp

        fold_result = {
            'fold': fold_i + 1,
            'train_size': tr_end - tr_start,
            'test_size': te_end - te_start,
            'auc': round(res['auc'], 4),
            'accuracy': round(res['accuracy'], 4),
            'precision': round(res['precision'], 4),
            'recall': round(res['recall'], 4),
        }
        xgb_fold_results.append(fold_result)
        print(f"    Fold {fold_i+1:2d}: AUC={res['auc']:.4f}  "
              f"Acc={res['accuracy']:.4f}  "
              f"Prec={res['precision']:.4f}  Rec={res['recall']:.4f}  "
              f"(train={tr_end - tr_start}, test={te_end - te_start})")

    xgb_mean_auc = 0.0
    xgb_last_model = None
    if xgb_available and xgb_fold_results:
        xgb_mean_auc = np.mean([f['auc'] for f in xgb_fold_results])
        print(f"\n    XGBoost Mean AUC: {xgb_mean_auc:.4f}")

        for k in xgb_importances_agg:
            xgb_importances_agg[k] /= len(folds)
        top_feats = sorted(xgb_importances_agg.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"    Top features: {', '.join(f'{k}={v:.3f}' for k, v in top_feats)}")

        # Retrain final model on all data for Phase 4
        res_final = _train_xgb(X_all, y_1d, X_all, y_1d)
        if res_final:
            xgb_last_model = res_final['model']

    results['phase2_xgboost'] = {
        'available': xgb_available,
        'folds': xgb_fold_results,
        'mean_auc': round(xgb_mean_auc, 4),
        'importances': {k: round(v, 4) for k, v in xgb_importances_agg.items()},
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Walk-Forward LightGBM (12 folds)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 3: Walk-Forward LightGBM (12 folds)")
    print("=" * 60)

    lgb_fold_results = []
    lgb_all_probs = np.full(len(daily_df), np.nan)
    lgb_importances_agg = {}
    lgb_available = True

    for fold_i, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        X_train = X_all.iloc[tr_start:tr_end]
        y_train = y_1d.iloc[tr_start:tr_end]
        X_test = X_all.iloc[te_start:te_end]
        y_test = y_1d.iloc[te_start:te_end]

        res = _train_lgb(X_train, y_train, X_test, y_test)
        if res is None:
            lgb_available = False
            print("    LightGBM not available — skipping Phase 3")
            break

        lgb_all_probs[te_start:te_end] = res['y_prob']

        for feat, imp in res['importances'].items():
            lgb_importances_agg[feat] = lgb_importances_agg.get(feat, 0) + imp

        fold_result = {
            'fold': fold_i + 1,
            'train_size': tr_end - tr_start,
            'test_size': te_end - te_start,
            'auc': round(res['auc'], 4),
            'accuracy': round(res['accuracy'], 4),
        }
        lgb_fold_results.append(fold_result)
        print(f"    Fold {fold_i+1:2d}: AUC={res['auc']:.4f}  "
              f"Acc={res['accuracy']:.4f}  "
              f"(train={tr_end - tr_start}, test={te_end - te_start})")

    lgb_mean_auc = 0.0
    lgb_last_model = None
    if lgb_available and lgb_fold_results:
        lgb_mean_auc = np.mean([f['auc'] for f in lgb_fold_results])
        print(f"\n    LightGBM Mean AUC: {lgb_mean_auc:.4f}")

        for k in lgb_importances_agg:
            lgb_importances_agg[k] /= len(folds)
        top_feats = sorted(lgb_importances_agg.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"    Top features: {', '.join(f'{k}={v:.3f}' for k, v in top_feats)}")

        res_final = _train_lgb(X_all, y_1d, X_all, y_1d)
        if res_final:
            lgb_last_model = res_final['model']

    results['phase3_lightgbm'] = {
        'available': lgb_available,
        'folds': lgb_fold_results,
        'mean_auc': round(lgb_mean_auc, 4),
        'importances': {k: round(v, 4) for k, v in lgb_importances_agg.items()},
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Strategy Integration
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 4: Strategy Integration — ML as Direction Filter")
    print("=" * 60)

    best_ml = 'xgboost' if xgb_mean_auc >= lgb_mean_auc else 'lightgbm'
    best_model = xgb_last_model if best_ml == 'xgboost' else lgb_last_model
    best_auc = max(xgb_mean_auc, lgb_mean_auc)
    print(f"    Best model: {best_ml} (mean AUC={best_auc:.4f})")

    phase4 = {'best_model': best_ml, 'best_auc': round(best_auc, 4)}

    if best_model is None:
        print("    No ML model available — skipping Phase 4")
        phase4['skipped'] = True
        results['phase4_integration'] = phase4
    else:
        # (a) ML as standalone strategy
        print("\n    (a) ML as standalone strategy:")
        standalone_pnls = _ml_standalone_pnl(daily_df, feature_cols, best_model,
                                             SPREAD, UNIT_LOT)
        n_standalone = len(standalone_pnls)
        if n_standalone > 0:
            standalone_daily = {}
            for p in standalone_pnls:
                d = p['date'].date() if hasattr(p['date'], 'date') else p['date']
                standalone_daily[d] = standalone_daily.get(d, 0) + p['pnl']
            s_arr = np.array(list(standalone_daily.values()))
            s_sharpe = sharpe(s_arr)
            s_pnl = float(np.sum(s_arr))
            s_maxdd = max_dd(s_arr)
            n_long = sum(1 for p in standalone_pnls if p['dir'] == 'BUY')
            n_short = n_standalone - n_long
            print(f"        Trades: {n_standalone} (long={n_long}, short={n_short})")
            print(f"        Sharpe={s_sharpe:.3f}, PnL=${s_pnl:,.2f}, MaxDD=${s_maxdd:,.2f}")
            phase4['standalone'] = {
                'n_trades': n_standalone,
                'n_long': n_long,
                'n_short': n_short,
                'sharpe': round(s_sharpe, 3),
                'pnl': round(s_pnl, 2),
                'max_dd': round(s_maxdd, 2),
            }
        else:
            print("        No standalone trades generated.")
            phase4['standalone'] = {'n_trades': 0}

        # (b) ML as direction filter for existing strategies
        print("\n    (b) ML as direction filter for 4 strategies:")

        print("        Running unfiltered strategies...")
        base_trades = {}
        base_trades['PSAR'] = bt_psar(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['PSAR'])
        base_trades['TSMOM'] = bt_tsmom(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['TSMOM'])
        base_trades['SESS_BO'] = bt_sess_bo(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['SESS_BO'])
        base_trades['L8_MAX'] = bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])

        unit_dailies_unfiltered = {}
        for name in STRAT_ORDER:
            unit_dailies_unfiltered[name] = trades_to_daily_series(base_trades[name])

        unfiltered_daily = build_portfolio_daily(unit_dailies_unfiltered, R89_LOTS)
        unfiltered_m = portfolio_metrics(unfiltered_daily)

        print("        Running ML-filtered strategies...")
        filtered_trades = {}
        for name in STRAT_ORDER:
            filtered_trades[name] = _filter_trades_by_ml(
                base_trades[name], daily_df, feature_cols, best_model
            )

        unit_dailies_filtered = {}
        for name in STRAT_ORDER:
            unit_dailies_filtered[name] = trades_to_daily_series(filtered_trades[name])

        filtered_daily = build_portfolio_daily(unit_dailies_filtered, R89_LOTS)
        filtered_m = portfolio_metrics(filtered_daily)

        print(f"\n        {'Strategy':<12} {'Unfiltered':>12} {'Filtered':>12} {'Kept%':>8}")
        print(f"        {'-'*48}")
        filter_details = {}
        for name in STRAT_ORDER:
            n_orig = len(base_trades[name])
            n_filt = len(filtered_trades[name])
            kept_pct = 100 * n_filt / n_orig if n_orig > 0 else 0
            print(f"        {name:<12} {n_orig:>12d} {n_filt:>12d} {kept_pct:>7.1f}%")
            filter_details[name] = {
                'unfiltered': n_orig,
                'filtered': n_filt,
                'kept_pct': round(kept_pct, 1),
            }

        sharpe_delta = filtered_m['sharpe'] - unfiltered_m['sharpe']
        dd_delta = unfiltered_m['max_dd'] - filtered_m['max_dd']
        print(f"\n        Portfolio (unfiltered): Sharpe={unfiltered_m['sharpe']:.3f}, "
              f"PnL=${unfiltered_m['pnl']:,.2f}, MaxDD=${unfiltered_m['max_dd']:,.2f}")
        print(f"        Portfolio (filtered):   Sharpe={filtered_m['sharpe']:.3f}, "
              f"PnL=${filtered_m['pnl']:,.2f}, MaxDD=${filtered_m['max_dd']:,.2f}")
        print(f"        Sharpe improvement: {sharpe_delta:+.3f}")
        print(f"        MaxDD improvement:  ${dd_delta:+,.2f}")

        phase4['direction_filter'] = {
            'unfiltered': unfiltered_m,
            'filtered': filtered_m,
            'sharpe_delta': round(sharpe_delta, 3),
            'maxdd_improvement': round(dd_delta, 2),
            'per_strategy': filter_details,
        }

        results['phase4_integration'] = phase4

    # ═══════════════════════════════════════════════════════════
    # Phase 5: Robustness Tests
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 5: Robustness Tests")
    print("=" * 60)

    phase5 = {}

    # --- 5a: Random label shuffle ---
    print("\n    (a) Random Label Shuffle:")
    if xgb_available or lgb_available:
        train_fn = _train_xgb if best_ml == 'xgboost' else _train_lgb
        shuffle_aucs = []
        n_shuffle_trials = 5
        rng = np.random.RandomState(42)
        for trial in range(n_shuffle_trials):
            y_shuffled = y_1d.values.copy()
            rng.shuffle(y_shuffled)

            mid = len(daily_df) * 2 // 3
            X_tr = X_all.iloc[:mid]
            y_tr = pd.Series(y_shuffled[:mid], index=X_all.index[:mid])
            X_te = X_all.iloc[mid:]
            y_te = pd.Series(y_shuffled[mid:], index=X_all.index[mid:])

            res = train_fn(X_tr, y_tr, X_te, y_te)
            if res:
                shuffle_aucs.append(res['auc'])
                print(f"        Trial {trial+1}: AUC={res['auc']:.4f}")

        if shuffle_aucs:
            mean_shuffle = np.mean(shuffle_aucs)
            print(f"        Mean shuffled AUC: {mean_shuffle:.4f} "
                  f"(expected ~0.50, real={best_auc:.4f})")
            phase5['label_shuffle'] = {
                'trial_aucs': [round(a, 4) for a in shuffle_aucs],
                'mean_auc': round(mean_shuffle, 4),
                'real_auc': round(best_auc, 4),
                'auc_drop': round(best_auc - mean_shuffle, 4),
                'pass': mean_shuffle < best_auc,
            }
        else:
            phase5['label_shuffle'] = {'skipped': True}
    else:
        print("        Skipped — no ML model available")
        phase5['label_shuffle'] = {'skipped': True}

    # --- 5b: Feature ablation ---
    print("\n    (b) Feature Ablation (drop top 3 features):")
    if (xgb_available or lgb_available) and best_model is not None:
        importances = xgb_importances_agg if best_ml == 'xgboost' else lgb_importances_agg
        top3 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_names = [f[0] for f in top3]
        print(f"        Dropping: {top3_names}")

        ablated_cols = [c for c in feature_cols if c not in top3_names]
        if len(ablated_cols) < 2:
            print("        Not enough features left — skipping ablation")
            phase5['feature_ablation'] = {'skipped': True}
        else:
            ablation_aucs = []
            X_abl = daily_df[ablated_cols]
            for fold_i, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
                X_tr = X_abl.iloc[tr_start:tr_end]
                y_tr = y_1d.iloc[tr_start:tr_end]
                X_te = X_abl.iloc[te_start:te_end]
                y_te = y_1d.iloc[te_start:te_end]

                res = train_fn(X_tr, y_tr, X_te, y_te)
                if res:
                    ablation_aucs.append(res['auc'])

            if ablation_aucs:
                mean_abl = np.mean(ablation_aucs)
                auc_drop = best_auc - mean_abl
                print(f"        Ablated mean AUC: {mean_abl:.4f} "
                      f"(drop={auc_drop:+.4f} from {best_auc:.4f})")
                phase5['feature_ablation'] = {
                    'dropped_features': top3_names,
                    'mean_auc': round(mean_abl, 4),
                    'auc_drop': round(auc_drop, 4),
                    'n_remaining_features': len(ablated_cols),
                }
            else:
                phase5['feature_ablation'] = {'skipped': True}
    else:
        print("        Skipped — no ML model available")
        phase5['feature_ablation'] = {'skipped': True}

    # --- 5c: Parameter perturbation ---
    print("\n    (c) Parameter Perturbation:")
    if xgb_available or lgb_available:
        depth_grid = [3, 4, 5, 6]
        nest_grid = [100, 200, 300]
        perturb_results = []

        mid = len(daily_df) * 2 // 3
        X_tr = X_all.iloc[:mid]
        y_tr = y_1d.iloc[:mid]
        X_te = X_all.iloc[mid:]
        y_te = y_1d.iloc[mid:]

        for depth in depth_grid:
            for nest in nest_grid:
                params = {'max_depth': depth, 'n_estimators': nest}
                res = train_fn(X_tr, y_tr, X_te, y_te, params=params)
                if res:
                    label = f"depth={depth},n={nest}"
                    perturb_results.append({
                        'max_depth': depth,
                        'n_estimators': nest,
                        'auc': round(res['auc'], 4),
                        'accuracy': round(res['accuracy'], 4),
                    })
                    print(f"        {label:22s}  AUC={res['auc']:.4f}  Acc={res['accuracy']:.4f}")

        if perturb_results:
            all_aucs = [r['auc'] for r in perturb_results]
            auc_std = np.std(all_aucs)
            auc_range = max(all_aucs) - min(all_aucs)
            print(f"        AUC std={auc_std:.4f}, range={auc_range:.4f}")
            phase5['param_perturbation'] = {
                'results': perturb_results,
                'auc_std': round(auc_std, 4),
                'auc_range': round(auc_range, 4),
                'auc_min': round(min(all_aucs), 4),
                'auc_max': round(max(all_aucs), 4),
                'stable': auc_std < 0.02,
            }
        else:
            phase5['param_perturbation'] = {'skipped': True}
    else:
        print("        Skipped — no ML model available")
        phase5['param_perturbation'] = {'skipped': True}

    results['phase5_robustness'] = phase5

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    print("\n" + "=" * 80)
    print("  R107 SUMMARY — ML Entry Direction Prediction")
    print("=" * 80)

    print(f"\n  Features: {len(feature_cols)} "
          f"({results['phase1']['date_start']} ~ {results['phase1']['date_end']})")

    if xgb_available:
        print(f"  XGBoost  walk-forward AUC: {xgb_mean_auc:.4f} ({len(xgb_fold_results)} folds)")
    if lgb_available:
        print(f"  LightGBM walk-forward AUC: {lgb_mean_auc:.4f} ({len(lgb_fold_results)} folds)")

    if 'phase4_integration' in results and not results['phase4_integration'].get('skipped'):
        p4 = results['phase4_integration']
        if 'standalone' in p4 and p4['standalone'].get('n_trades', 0) > 0:
            s = p4['standalone']
            print(f"\n  ML Standalone: {s['n_trades']} trades, "
                  f"Sharpe={s['sharpe']:.3f}, PnL=${s['pnl']:,.2f}")
        if 'direction_filter' in p4:
            df_ = p4['direction_filter']
            print(f"\n  Direction Filter:")
            print(f"    Unfiltered: Sharpe={df_['unfiltered']['sharpe']:.3f}, "
                  f"MaxDD=${df_['unfiltered']['max_dd']:,.2f}")
            print(f"    Filtered:   Sharpe={df_['filtered']['sharpe']:.3f}, "
                  f"MaxDD=${df_['filtered']['max_dd']:,.2f}")
            print(f"    Delta Sharpe: {df_['sharpe_delta']:+.3f}, "
                  f"Delta MaxDD: ${df_['maxdd_improvement']:+,.2f}")

    p5 = results.get('phase5_robustness', {})
    print(f"\n  Robustness:")
    if 'label_shuffle' in p5 and not p5['label_shuffle'].get('skipped'):
        ls = p5['label_shuffle']
        status = "PASS" if ls['pass'] else "FAIL"
        print(f"    Label shuffle:   {status} (shuffled AUC={ls['mean_auc']:.4f} "
              f"vs real={ls['real_auc']:.4f})")
    if 'feature_ablation' in p5 and not p5['feature_ablation'].get('skipped'):
        fa = p5['feature_ablation']
        print(f"    Feature ablation: AUC drop={fa['auc_drop']:+.4f} "
              f"(dropped {fa['dropped_features']})")
    if 'param_perturbation' in p5 and not p5['param_perturbation'].get('skipped'):
        pp = p5['param_perturbation']
        status = "STABLE" if pp['stable'] else "UNSTABLE"
        print(f"    Param perturb:   {status} (AUC std={pp['auc_std']:.4f}, "
              f"range={pp['auc_range']:.4f})")

    verdict = "PROMISING" if best_auc > 0.52 else "INCONCLUSIVE"
    if best_auc <= 0.50:
        verdict = "NO SIGNAL"
    print(f"\n  Verdict: {verdict} (best AUC={best_auc:.4f})")

    results['elapsed_s'] = round(elapsed, 1)
    results['verdict'] = verdict

    out_file = OUTPUT_DIR / "r107_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Saved: {out_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
