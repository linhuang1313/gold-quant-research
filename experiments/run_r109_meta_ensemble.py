#!/usr/bin/env python3
"""
R109 — Meta-Ensemble Model (Strategy of Strategies)
======================================================
XGBoost meta-model that learns when to trust each strategy's signals.
Uses strategy signals, recent performance, and market state as features.

  Phase 1: Build meta-features (strategy signals + context)
  Phase 2: Walk-forward XGBoost meta-model (12 folds)
  Phase 3: Decision layer — meta-model filters strategy signals
  Phase 4: Robustness tests
  Phase 5: Comparison vs other filtering methods
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

OUTPUT_DIR = Path("results/r109_meta_ensemble")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']

ALIGNED_CSV = Path("data/external/aligned_daily.csv")


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


# ═══════════════════════════════════════════════════════════════
# XGBoost helpers
# ═══════════════════════════════════════════════════════════════

def get_xgb_model(n_estimators=200, max_depth=4, learning_rate=0.05):
    try:
        import xgboost as xgb
        try:
            m = xgb.XGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, subsample=0.8,
                colsample_bytree=0.8, eval_metric='logloss',
                tree_method='hist', device='cuda',
                random_state=42, verbosity=0)
            m.fit(np.random.randn(10, 5), np.random.randint(0, 2, 10))
            return xgb.XGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, subsample=0.8,
                colsample_bytree=0.8, eval_metric='logloss',
                tree_method='hist', device='cuda',
                random_state=42, verbosity=0)
        except Exception:
            return xgb.XGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, subsample=0.8,
                colsample_bytree=0.8, eval_metric='logloss',
                tree_method='hist', random_state=42, verbosity=0)
    except ImportError:
        print("  ERROR: xgboost not installed. pip install xgboost", flush=True)
        return None


# ═══════════════════════════════════════════════════════════════
# Meta-feature builder
# ═══════════════════════════════════════════════════════════════

def build_meta_features(all_trades_by_strat, h1_df, external_daily=None):
    """Build meta-feature matrix from all strategy trades.

    Returns (X DataFrame, y array, trade_list with strategy labels).
    """
    h1_tz_free = h1_df.copy()
    if h1_tz_free.index.tz is not None:
        h1_tz_free.index = h1_tz_free.index.tz_localize(None)
    atr_series = compute_atr(h1_tz_free)
    close_series = h1_tz_free['Close']
    sma50 = close_series.rolling(50).mean()
    sma200 = close_series.rolling(200).mean()
    ret_24 = close_series.pct_change(24)
    ret_120 = close_series.pct_change(120)

    atr_pctrank = atr_series.rolling(500, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    has_macro = False
    vix_series = dxy_ret5 = yield_spread = None
    if external_daily is not None and ALIGNED_CSV.exists():
        try:
            ext = external_daily.copy()
            if 'VIX_Close' in ext.columns:
                vix_series = ext['VIX_Close']
            if 'DXY_Close' in ext.columns:
                dxy_ret5 = ext['DXY_Close'].pct_change(5)
            y10_col = next((c for c in ext.columns if '10Y' in c.upper() or 'US10Y' in c.upper()), None)
            y2_col = next((c for c in ext.columns if '2Y' in c.upper() or 'US2Y' in c.upper()), None)
            if y10_col and y2_col:
                yield_spread = ext[y10_col] - ext[y2_col]
            elif 'yield_curve_10y2y' in ext.columns:
                yield_spread = ext['yield_curve_10y2y']
            has_macro = vix_series is not None
        except Exception:
            pass

    combined_trades = []
    for strat_name in STRAT_ORDER:
        for t in all_trades_by_strat.get(strat_name, []):
            combined_trades.append({**t, '_strat': strat_name})

    combined_trades.sort(key=lambda t: pd.Timestamp(t['entry_time']))

    trailing_pnl = {s: [] for s in STRAT_ORDER}
    trailing_equity = {s: 0.0 for s in STRAT_ORDER}
    trailing_peak = {s: 0.0 for s in STRAT_ORDER}
    last_trade_time = {s: None for s in STRAT_ORDER}

    rows = []
    targets = []
    meta_trades = []

    for t in combined_trades:
        strat = t['_strat']
        entry_ts = pd.Timestamp(t['entry_time'])
        if hasattr(entry_ts, 'tz') and entry_ts.tz is not None:
            entry_ts = entry_ts.tz_localize(None)

        idx_pos = atr_series.index.searchsorted(entry_ts)
        idx_pos = min(max(idx_pos - 1, 0), len(atr_series) - 1)

        atr_val = atr_series.iloc[idx_pos]
        if np.isnan(atr_val):
            continue

        feat = {}

        feat['is_psar'] = 1 if strat == 'PSAR' else 0
        feat['is_tsmom'] = 1 if strat == 'TSMOM' else 0
        feat['is_sessbo'] = 1 if strat == 'SESS_BO' else 0
        feat['is_l8'] = 1 if strat == 'L8_MAX' else 0

        feat['direction'] = 1 if t['dir'] == 'BUY' else 0
        feat['hour_of_day'] = entry_ts.hour
        feat['day_of_week'] = entry_ts.dayofweek

        feat['atr_14'] = float(atr_val)
        atr_pct_val = atr_pctrank.iloc[idx_pos] if idx_pos < len(atr_pctrank) else np.nan
        feat['atr_percentile'] = float(atr_pct_val) if not np.isnan(atr_pct_val) else 0.5

        sma50_val = sma50.iloc[idx_pos] if idx_pos < len(sma50) else np.nan
        sma200_val = sma200.iloc[idx_pos] if idx_pos < len(sma200) else np.nan
        cl = close_series.iloc[idx_pos]

        feat['close_vs_sma50'] = float((cl - sma50_val) / sma50_val) if (not np.isnan(sma50_val) and sma50_val > 0) else 0.0
        feat['close_vs_sma200'] = float((cl - sma200_val) / sma200_val) if (not np.isnan(sma200_val) and sma200_val > 0) else 0.0

        r24 = ret_24.iloc[idx_pos] if idx_pos < len(ret_24) else np.nan
        r120 = ret_120.iloc[idx_pos] if idx_pos < len(ret_120) else np.nan
        feat['momentum_24'] = float(r24) if not np.isnan(r24) else 0.0
        feat['momentum_120'] = float(r120) if not np.isnan(r120) else 0.0

        trail = trailing_pnl[strat]
        feat['trail_10_pnl'] = float(sum(trail[-10:])) if len(trail) > 0 else 0.0
        feat['trail_10_wr'] = float(sum(1 for p in trail[-10:] if p > 0) / max(len(trail[-10:]), 1))
        feat['strat_rolling_dd'] = float(trailing_peak[strat] - trailing_equity[strat]) if trailing_peak[strat] > trailing_equity[strat] else 0.0

        if last_trade_time[strat] is not None:
            delta = (entry_ts - last_trade_time[strat]).total_seconds() / 86400.0
            feat['days_since_last'] = float(delta)
        else:
            feat['days_since_last'] = 30.0

        if has_macro:
            entry_date = entry_ts.normalize()
            if hasattr(entry_date, 'tz') and entry_date.tz is not None:
                entry_date = entry_date.tz_localize(None)
            if vix_series is not None:
                if vix_series.index.tz is not None:
                    vix_series.index = vix_series.index.tz_localize(None)
                vix_idx = vix_series.index.searchsorted(entry_date)
                vix_idx = min(max(vix_idx - 1, 0), len(vix_series) - 1)
                feat['vix_level'] = float(vix_series.iloc[vix_idx]) if not np.isnan(vix_series.iloc[vix_idx]) else 20.0
            if dxy_ret5 is not None:
                if dxy_ret5.index.tz is not None:
                    dxy_ret5.index = dxy_ret5.index.tz_localize(None)
                dxy_idx = dxy_ret5.index.searchsorted(entry_date)
                dxy_idx = min(max(dxy_idx - 1, 0), len(dxy_ret5) - 1)
                feat['dxy_5d_return'] = float(dxy_ret5.iloc[dxy_idx]) if not np.isnan(dxy_ret5.iloc[dxy_idx]) else 0.0
            if yield_spread is not None:
                if yield_spread.index.tz is not None:
                    yield_spread.index = yield_spread.index.tz_localize(None)
                ys_idx = yield_spread.index.searchsorted(entry_date)
                ys_idx = min(max(ys_idx - 1, 0), len(yield_spread) - 1)
                feat['yield_spread_10y2y'] = float(yield_spread.iloc[ys_idx]) if not np.isnan(yield_spread.iloc[ys_idx]) else 0.0

        rows.append(feat)
        targets.append(1 if t['pnl'] > 0 else 0)
        meta_trades.append(t)

        trailing_pnl[strat].append(t['pnl'])
        trailing_equity[strat] += t['pnl']
        trailing_peak[strat] = max(trailing_peak[strat], trailing_equity[strat])
        last_trade_time[strat] = entry_ts

    X = pd.DataFrame(rows)
    y = np.array(targets)
    return X, y, meta_trades


# ═══════════════════════════════════════════════════════════════
# Walk-forward meta-model
# ═══════════════════════════════════════════════════════════════

def walk_forward_meta(X, y, meta_trades, n_folds=12, model_kwargs=None):
    """Walk-forward: divide into n_folds chunks, for fold k train on 1..k, test on k+1.
    Returns oos_probs array (NaN for untested trades), per-fold results."""
    from sklearn.metrics import roc_auc_score, accuracy_score

    entry_times = pd.Series([pd.Timestamp(t['entry_time']) for t in meta_trades])
    if entry_times.dt.tz is not None:
        entry_times = entry_times.dt.tz_localize(None)

    n = len(X)
    chunk_size = n // n_folds
    oos_probs = np.full(n, np.nan)
    fold_results = []

    for k in range(n_folds - 1):
        train_end = (k + 1) * chunk_size
        test_start = train_end
        test_end = min(test_start + chunk_size, n) if k < n_folds - 2 else n

        if train_end < 50 or test_end - test_start < 10:
            fold_results.append({'fold': k + 1, 'auc': np.nan, 'acc': np.nan, 'n_test': 0})
            continue

        Xtr = X.iloc[:train_end].copy()
        ytr = y[:train_end]
        Xte = X.iloc[test_start:test_end].copy()
        yte = y[test_start:test_end]

        med = Xtr.median()
        Xtr = Xtr.fillna(med)
        Xte = Xte.fillna(med)

        const_cols = [c for c in Xtr.columns if Xtr[c].nunique() <= 1]
        if const_cols:
            Xtr = Xtr.drop(columns=const_cols)
            Xte = Xte.drop(columns=const_cols)
        if len(Xtr.columns) == 0:
            fold_results.append({'fold': k + 1, 'auc': np.nan, 'acc': np.nan, 'n_test': 0})
            continue

        try:
            model = get_xgb_model(**(model_kwargs or {}))
            if model is None:
                fold_results.append({'fold': k + 1, 'auc': np.nan, 'acc': np.nan, 'n_test': 0})
                continue
            model.fit(Xtr, ytr)
            probs = model.predict_proba(Xte)[:, 1]
            oos_probs[test_start:test_end] = probs

            auc = roc_auc_score(yte, probs) if len(np.unique(yte)) > 1 else np.nan
            acc = accuracy_score(yte, (probs >= 0.5).astype(int))

            fold_results.append({
                'fold': k + 1, 'auc': round(float(auc), 4) if not np.isnan(auc) else np.nan,
                'acc': round(float(acc), 4), 'n_test': int(test_end - test_start),
            })
        except Exception as e:
            fold_results.append({'fold': k + 1, 'auc': np.nan, 'acc': np.nan,
                                 'n_test': 0, 'error': str(e)})

    last_model = model if 'model' in dir() else None
    feature_names = list(X.columns)
    importance = {}
    if last_model is not None:
        try:
            imp = last_model.feature_importances_
            used_cols = [c for c in feature_names if c not in const_cols] if 'const_cols' in dir() else feature_names
            for fname, fval in zip(used_cols, imp):
                importance[fname] = round(float(fval), 4)
        except Exception:
            pass

    return oos_probs, fold_results, importance


# ═══════════════════════════════════════════════════════════════
# Portfolio from filtered trades
# ═══════════════════════════════════════════════════════════════

def build_filtered_portfolio(meta_trades, oos_probs, threshold, lots):
    """Keep only trades where P(profitable) > threshold, build portfolio daily PnL."""
    kept_by_strat = {s: [] for s in STRAT_ORDER}
    total_valid = 0
    total_kept = 0

    for i, t in enumerate(meta_trades):
        if np.isnan(oos_probs[i]):
            continue
        total_valid += 1
        if oos_probs[i] > threshold:
            total_kept += 1
            strat = t['_strat']
            kept_by_strat[strat].append(t)

    unit_dailies = {}
    for s in STRAT_ORDER:
        unit_dailies[s] = trades_to_daily_series(kept_by_strat[s])

    port_daily = build_portfolio_daily(unit_dailies, lots)
    pct_filtered = round((1 - total_kept / max(total_valid, 1)) * 100, 1)

    return port_daily, total_kept, total_valid, pct_filtered, kept_by_strat


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("  R109 — Meta-Ensemble Model (Strategy of Strategies)", flush=True)
    print("=" * 80, flush=True)

    # ── Load data ──
    print("\n  Loading data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    bundle = DataBundle.load_custom()
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0].date()} ~ {h1_df.index[-1].date()})")

    external_daily = None
    if ALIGNED_CSV.exists():
        external_daily = pd.read_csv(ALIGNED_CSV, parse_dates=['Date'], index_col='Date')
        external_daily.index = external_daily.index.normalize()
        if external_daily.index.tz is not None:
            external_daily.index = external_daily.index.tz_localize(None)
        print(f"    External daily: {len(external_daily)} rows")
    else:
        print("    No aligned_daily.csv — macro features skipped")

    # ── Run all 4 strategies at unit lot ──
    print("\n  Running 4 strategies at unit lot (0.01)...", flush=True)
    base_trades = {}
    base_trades['PSAR'] = bt_psar(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['PSAR'])
    base_trades['TSMOM'] = bt_tsmom(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['TSMOM'])
    base_trades['SESS_BO'] = bt_sess_bo(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['SESS_BO'])
    base_trades['L8_MAX'] = bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])

    unit_dailies = {}
    for name in STRAT_ORDER:
        unit_dailies[name] = trades_to_daily_series(base_trades[name])
        n_t = len(base_trades[name])
        pnl = sum(t['pnl'] for t in base_trades[name])
        print(f"    {name:10s}: {n_t:5d} trades, unit PnL=${pnl:,.2f}")

    # R89 baseline
    baseline_daily = build_portfolio_daily(unit_dailies, R89_LOTS)
    baseline_sharpe = round(sharpe(baseline_daily), 3)
    baseline_pnl = round(float(np.sum(baseline_daily)), 2)
    baseline_maxdd = round(max_dd(baseline_daily), 2)
    print(f"\n  R89 Baseline: Sharpe={baseline_sharpe}, PnL=${baseline_pnl:,.2f}, "
          f"MaxDD=${baseline_maxdd:,.2f}")

    results = {'experiment': 'R109 Meta-Ensemble Model',
               'baseline': {'sharpe': baseline_sharpe, 'pnl': baseline_pnl,
                             'max_dd': baseline_maxdd}}

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Build Meta-Features
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 1: Build Meta-Features", flush=True)
    print("=" * 70, flush=True)

    X, y, meta_trades = build_meta_features(base_trades, h1_df, external_daily)
    n_samples = len(X)
    n_features = X.shape[1]
    class_balance = float(y.mean())

    print(f"    Total samples: {n_samples}")
    print(f"    Feature count: {n_features}")
    print(f"    Features: {list(X.columns)}")
    print(f"    Class balance: {class_balance*100:.1f}% profitable")

    per_strat_counts = {}
    for t in meta_trades:
        s = t['_strat']
        per_strat_counts[s] = per_strat_counts.get(s, 0) + 1
    for s in STRAT_ORDER:
        print(f"      {s}: {per_strat_counts.get(s, 0)} trades")

    results['phase1'] = {
        'n_samples': n_samples, 'n_features': n_features,
        'class_balance': round(class_balance, 4),
        'per_strategy': per_strat_counts,
        'feature_names': list(X.columns),
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Walk-Forward XGBoost Meta-Model (12 folds)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: Walk-Forward XGBoost Meta-Model (12 folds)", flush=True)
    print("=" * 70, flush=True)

    oos_probs, fold_results, importance = walk_forward_meta(X, y, meta_trades, n_folds=12)

    valid_oos = ~np.isnan(oos_probs)
    n_valid = int(valid_oos.sum())
    print(f"\n    OOS predictions: {n_valid}/{n_samples} trades")

    print(f"\n    {'Fold':>6} {'AUC':>8} {'Acc':>8} {'N_test':>8}")
    print(f"    {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    aucs = []
    for fr in fold_results:
        auc_str = f"{fr['auc']:.4f}" if fr['auc'] is not None and not np.isnan(fr.get('auc', np.nan)) else "  N/A"
        acc_str = f"{fr['acc']:.4f}" if fr['acc'] is not None and not np.isnan(fr.get('acc', np.nan)) else "  N/A"
        print(f"    {fr['fold']:>6} {auc_str:>8} {acc_str:>8} {fr['n_test']:>8}")
        if fr['auc'] is not None and not np.isnan(fr.get('auc', np.nan)):
            aucs.append(fr['auc'])

    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    print(f"\n    Mean AUC: {mean_auc:.4f} (across {len(aucs)} folds)")

    if importance:
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n    Top 10 Feature Importance:")
        for rank, (fname, fval) in enumerate(sorted_imp, 1):
            bar = "█" * int(fval * 100)
            print(f"      #{rank:2d}: {fname:<25s} {fval:.4f} {bar}")

    results['phase2'] = {
        'n_valid_oos': n_valid, 'mean_auc': round(mean_auc, 4),
        'fold_results': fold_results,
        'top_features': dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]),
    }

    if n_valid < 100:
        print("\n  ERROR: Too few OOS predictions to proceed.", flush=True)
        results['error'] = 'Too few OOS predictions'
        with open(OUTPUT_DIR / "r109_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        return

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Decision Layer
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3: Decision Layer — Threshold Sweep", flush=True)
    print("=" * 70, flush=True)

    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60]

    print(f"\n    {'Thresh':>8} {'Sharpe':>8} {'MaxDD':>10} {'PnL':>12} "
          f"{'Kept':>6} {'Filter%':>8}", flush=True)
    print(f"    {'-'*8} {'-'*8} {'-'*10} {'-'*12} {'-'*6} {'-'*8}")

    threshold_results = {}
    for thr in thresholds:
        port_daily, kept, valid_total, pct_filt, _ = build_filtered_portfolio(
            meta_trades, oos_probs, thr, R89_LOTS)

        sh = round(sharpe(port_daily), 3)
        md = round(max_dd(port_daily), 2)
        pnl = round(float(np.sum(port_daily)), 2)

        threshold_results[str(thr)] = {
            'sharpe': sh, 'max_dd': md, 'pnl': pnl,
            'n_kept': kept, 'n_valid': valid_total,
            'pct_filtered': pct_filt,
        }

        print(f"    {thr:>8.2f} {sh:>8.3f} ${md:>9.2f} ${pnl:>11.2f} "
              f"{kept:>6} {pct_filt:>7.1f}%", flush=True)

    best_thr = max(threshold_results.items(), key=lambda x: x[1]['sharpe'])
    best_threshold = float(best_thr[0])
    best_thr_sharpe = best_thr[1]['sharpe']
    delta_vs_base = round(best_thr_sharpe - baseline_sharpe, 3)

    print(f"\n    Optimal threshold: {best_threshold:.2f} "
          f"(Sharpe={best_thr_sharpe:.3f}, Δ vs baseline={delta_vs_base:+.3f})")

    results['phase3'] = {
        'threshold_sweep': threshold_results,
        'optimal_threshold': best_threshold,
        'optimal_sharpe': best_thr_sharpe,
        'delta_vs_baseline': delta_vs_base,
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Robustness Tests
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 4: Robustness Tests", flush=True)
    print("=" * 70, flush=True)

    from sklearn.metrics import roc_auc_score
    robustness = {}

    # ── Test 4a: Random label shuffle ──
    print("\n  4a) Random Label Shuffle (null hypothesis):", flush=True)
    shuffle_aucs = []
    n_shuffle = 10
    for trial in range(n_shuffle):
        y_shuffled = y.copy()
        np.random.seed(trial + 100)
        np.random.shuffle(y_shuffled)
        _, shuf_folds, _ = walk_forward_meta(X, y_shuffled, meta_trades, n_folds=12)
        trial_aucs = [fr['auc'] for fr in shuf_folds
                      if fr['auc'] is not None and not np.isnan(fr.get('auc', np.nan))]
        if trial_aucs:
            shuffle_aucs.append(float(np.mean(trial_aucs)))
        if (trial + 1) % 5 == 0:
            print(f"      Shuffle trial {trial+1}/{n_shuffle}...", flush=True)

    mean_shuffle_auc = float(np.mean(shuffle_aucs)) if shuffle_aucs else 0.5
    print(f"      Shuffled mean AUC: {mean_shuffle_auc:.4f} (expected ~0.50)")
    print(f"      Real mean AUC:     {mean_auc:.4f}")
    shuffle_pass = mean_auc > mean_shuffle_auc + 0.02
    print(f"      Null hypothesis test: {'PASS' if shuffle_pass else 'FAIL'} "
          f"(real > shuffle+0.02)")

    robustness['shuffle_test'] = {
        'real_auc': round(mean_auc, 4),
        'shuffled_auc': round(mean_shuffle_auc, 4),
        'pass': shuffle_pass,
    }

    # ── Test 4b: Feature ablation ──
    print("\n  4b) Feature Ablation:", flush=True)

    strat_identity_cols = ['is_psar', 'is_tsmom', 'is_sessbo', 'is_l8']
    trail_cols = ['trail_10_pnl', 'trail_10_wr', 'strat_rolling_dd', 'days_since_last']

    ablation_results = {}
    for label, drop_cols in [("no_strategy_id", strat_identity_cols),
                              ("no_trailing_perf", trail_cols)]:
        existing_drops = [c for c in drop_cols if c in X.columns]
        if not existing_drops:
            ablation_results[label] = {'auc': mean_auc, 'delta': 0.0, 'dropped': []}
            continue

        X_abl = X.drop(columns=existing_drops)
        _, abl_folds, _ = walk_forward_meta(X_abl, y, meta_trades, n_folds=12)
        abl_aucs = [fr['auc'] for fr in abl_folds
                    if fr['auc'] is not None and not np.isnan(fr.get('auc', np.nan))]
        abl_mean = float(np.mean(abl_aucs)) if abl_aucs else 0.5
        delta_auc = round(abl_mean - mean_auc, 4)
        ablation_results[label] = {
            'auc': round(abl_mean, 4), 'delta': delta_auc,
            'dropped': existing_drops,
        }
        print(f"      {label:25s}: AUC={abl_mean:.4f} (Δ={delta_auc:+.4f})")

    robustness['feature_ablation'] = ablation_results

    # ── Test 4c: Temporal stability ──
    print("\n  4c) Temporal Stability (AUC across folds):", flush=True)
    if len(aucs) >= 3:
        auc_std = float(np.std(aucs))
        auc_cv = auc_std / mean_auc if mean_auc > 0 else 999
        first_half_auc = float(np.mean(aucs[:len(aucs)//2]))
        second_half_auc = float(np.mean(aucs[len(aucs)//2:]))
        drift = second_half_auc - first_half_auc

        print(f"      AUC std:  {auc_std:.4f}")
        print(f"      AUC CV:   {auc_cv:.4f}")
        print(f"      1st half: {first_half_auc:.4f}")
        print(f"      2nd half: {second_half_auc:.4f}")
        print(f"      Drift:    {drift:+.4f}")

        temporal_stable = auc_cv < 0.20 and abs(drift) < 0.10
        print(f"      Temporal stability: {'PASS' if temporal_stable else 'WARN'}")

        robustness['temporal_stability'] = {
            'auc_std': round(auc_std, 4), 'auc_cv': round(auc_cv, 4),
            'first_half_auc': round(first_half_auc, 4),
            'second_half_auc': round(second_half_auc, 4),
            'drift': round(drift, 4), 'stable': temporal_stable,
        }
    else:
        print("      Not enough folds for temporal analysis.")
        robustness['temporal_stability'] = {'stable': False, 'reason': 'insufficient folds'}

    # ── Robustness summary ──
    n_pass = sum([
        robustness.get('shuffle_test', {}).get('pass', False),
        robustness.get('temporal_stability', {}).get('stable', False),
    ])
    print(f"\n  Robustness Summary: {n_pass}/2 core tests passed")
    robustness['n_pass'] = n_pass
    results['phase4'] = robustness

    # ═══════════════════════════════════════════════════════════
    # Phase 5: Comparison
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 5: Comparison vs Other Filtering Methods", flush=True)
    print("=" * 70, flush=True)

    # a) Raw unfiltered baseline
    baseline_n_trades = sum(len(base_trades[s]) for s in STRAT_ORDER)

    # b) Meta-ensemble filtered (best threshold)
    meta_daily, meta_kept, meta_valid, meta_filt, _ = build_filtered_portfolio(
        meta_trades, oos_probs, best_threshold, R89_LOTS)
    meta_sh = round(sharpe(meta_daily), 3)
    meta_md = round(max_dd(meta_daily), 2)
    meta_pnl = round(float(np.sum(meta_daily)), 2)

    # c) Simple R98-style filter: threshold 0.65 on L8_MAX only
    simple_daily_parts = {}
    for s in STRAT_ORDER:
        if s == 'L8_MAX':
            l8_trades_in_meta = [(i, t) for i, t in enumerate(meta_trades) if t['_strat'] == 'L8_MAX']
            kept_l8 = []
            for idx, t in l8_trades_in_meta:
                if not np.isnan(oos_probs[idx]) and oos_probs[idx] > 0.65:
                    kept_l8.append(t)
                elif np.isnan(oos_probs[idx]):
                    kept_l8.append(t)
            simple_daily_parts[s] = trades_to_daily_series(kept_l8)
        else:
            simple_daily_parts[s] = unit_dailies[s]
    simple_daily = build_portfolio_daily(simple_daily_parts, R89_LOTS)
    simple_sh = round(sharpe(simple_daily), 3)
    simple_md = round(max_dd(simple_daily), 2)
    simple_pnl = round(float(np.sum(simple_daily)), 2)
    simple_n = sum(len(trades_to_daily_series(base_trades[s])) for s in STRAT_ORDER if s != 'L8_MAX')
    simple_n += len([t for i, t in enumerate(meta_trades)
                     if t['_strat'] == 'L8_MAX' and
                     (np.isnan(oos_probs[i]) or oos_probs[i] > 0.65)])

    print(f"\n    {'Method':<30} {'Sharpe':>8} {'MaxDD':>10} {'PnL':>12} {'Trades':>8}")
    print(f"    {'-'*30} {'-'*8} {'-'*10} {'-'*12} {'-'*8}")
    print(f"    {'a) R89 Unfiltered':<30} {baseline_sharpe:>8.3f} ${baseline_maxdd:>9.2f} "
          f"${baseline_pnl:>11.2f} {baseline_n_trades:>8}")
    print(f"    {'b) Meta-Ensemble (thr=' + f'{best_threshold:.2f})':<30} {meta_sh:>8.3f} ${meta_md:>9.2f} "
          f"${meta_pnl:>11.2f} {meta_kept:>8}")
    print(f"    {'c) Simple L8 filter (0.65)':<30} {simple_sh:>8.3f} ${simple_md:>9.2f} "
          f"${simple_pnl:>11.2f} {simple_n:>8}")

    winner = 'meta_ensemble' if meta_sh >= max(baseline_sharpe, simple_sh) else (
        'simple_filter' if simple_sh >= baseline_sharpe else 'baseline')

    results['phase5'] = {
        'baseline': {
            'sharpe': baseline_sharpe, 'max_dd': baseline_maxdd,
            'pnl': baseline_pnl, 'n_trades': baseline_n_trades,
        },
        'meta_ensemble': {
            'sharpe': meta_sh, 'max_dd': meta_md, 'pnl': meta_pnl,
            'n_trades': meta_kept, 'threshold': best_threshold,
        },
        'simple_l8_filter': {
            'sharpe': simple_sh, 'max_dd': simple_md, 'pnl': simple_pnl,
            'n_trades': simple_n, 'threshold': 0.65,
        },
        'winner': winner,
    }

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    print(f"\n{'='*80}", flush=True)
    print(f"  R109 SUMMARY — Meta-Ensemble Model", flush=True)
    print(f"{'='*80}", flush=True)

    print(f"\n  Meta-model mean AUC:    {mean_auc:.4f}")
    print(f"  Optimal threshold:      {best_threshold:.2f}")
    print(f"  Baseline Sharpe:        {baseline_sharpe:.3f}")
    print(f"  Meta-Ensemble Sharpe:   {meta_sh:.3f} (Δ={meta_sh - baseline_sharpe:+.3f})")
    print(f"  Trades filtered:        {meta_filt:.1f}%")
    print(f"  Robustness:             {n_pass}/2 tests passed")

    if meta_sh > baseline_sharpe and n_pass >= 1:
        verdict = (f"Meta-ensemble adds value at threshold={best_threshold:.2f} "
                   f"(Sharpe {baseline_sharpe:.3f} -> {meta_sh:.3f})")
    elif meta_sh > baseline_sharpe:
        verdict = "Meta-ensemble improves Sharpe but robustness is marginal — use cautiously"
    else:
        verdict = "Meta-ensemble does not improve over R89 baseline — keep fixed portfolio"

    print(f"\n  VERDICT: {verdict}")
    results['verdict'] = verdict
    results['elapsed_s'] = round(elapsed, 1)

    out_file = OUTPUT_DIR / "r109_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Saved: {out_file}")
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
