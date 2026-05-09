#!/usr/bin/env python3
"""
R121 — Multi-Factor Regime Detection
======================================
Build systematic macro regime detector to adjust portfolio behavior.
Uses VIX, DXY, yields, credit, gold vol from aligned_daily.csv.

Phase 1: Load macro data + run baseline portfolio
Phase 2: Regime detection (HMM, KMeans, Rule-based)
Phase 3: Per-regime strategy performance analysis
Phase 4: Regime-conditional lot multipliers
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

from backtest.runner import run_variant, LIVE_PARITY_KWARGS, DataBundle

OUTPUT_DIR = Path("results/r121_macro_regime")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01

FOLDS = [
    ("Fold1", "2015-01-01", "2017-03-01"),
    ("Fold2", "2017-03-01", "2019-06-01"),
    ("Fold3", "2019-06-01", "2021-09-01"),
    ("Fold4", "2021-09-01", "2023-12-01"),
    ("Fold5", "2023-12-01", "2027-01-01"),
]

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
    sl = atr * sl_atr
    tp = atr * tp_atr
    bars = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_now = (cl - pos['entry'] - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if lo <= pos['entry'] - sl:
            return _mk(pos, pos['entry'] - sl, times[i], "SL", i, -sl * lot * pv)
        if hi >= pos['entry'] + tp:
            return _mk(pos, pos['entry'] + tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry'])
        extreme = max(extreme, hi)
        pos['extreme'] = extreme
        act_dist = atr * trail_act
        if extreme - pos['entry'] >= act_dist:
            trail_price = extreme - atr * trail_dist
            if cl <= trail_price:
                pnl = (trail_price - pos['entry'] - spread) * lot * pv
                return _mk(pos, trail_price, times[i], "Trail", i, pnl)
        if bars >= max_hold:
            pnl = (cl - pos['entry'] - spread) * lot * pv
            return _mk(pos, cl, times[i], "TimeExit", i, pnl)
    else:
        pnl_now = (pos['entry'] - cl - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if hi >= pos['entry'] + sl:
            return _mk(pos, pos['entry'] + sl, times[i], "SL", i, -sl * lot * pv)
        if lo <= pos['entry'] - tp:
            return _mk(pos, pos['entry'] - tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry'])
        extreme = min(extreme, lo)
        pos['extreme'] = extreme
        act_dist = atr * trail_act
        if pos['entry'] - extreme >= act_dist:
            trail_price = extreme + atr * trail_dist
            if cl >= trail_price:
                pnl = (pos['entry'] - trail_price - spread) * lot * pv
                return _mk(pos, trail_price, times[i], "Trail", i, pnl)
        if bars >= max_hold:
            pnl = (pos['entry'] - cl - spread) * lot * pv
            return _mk(pos, cl, times[i], "TimeExit", i, pnl)
    return None


# ═══════════════════════════════════════════════════════════════
# Strategy backtests
# ═══════════════════════════════════════════════════════════════

def add_psar(df, af_step=0.01, af_max=0.05):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    n = len(df)
    psar = np.empty(n); psar[:] = np.nan
    af = af_step; rising = True
    ep = h[0]; psar[0] = l[0]
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


def bt_psar(h1_df, spread, lot, maxloss_cap=5,
            sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04,
            max_hold=20, af_step=0.01, af_max=0.05, min_atr=0.1):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    add_psar(df, af_step, af_max)
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
        if np.isnan(atr[i]) or atr[i] < min_atr: continue
        if psar[i-1] > c[i-1] and psar[i] < c[i]:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif psar[i-1] < c[i-1] and psar[i] > c[i]:
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


def bt_sess_bo(h1_df, spread, lot, maxloss_cap=35,
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
        ll_val = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll_val:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
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
# Metrics and portfolio helpers
# ═══════════════════════════════════════════════════════════════

def sharpe(arr, ann=252):
    if len(arr) < 10: return 0.0
    s = np.std(arr, ddof=1)
    return float(np.mean(arr) / s * np.sqrt(ann)) if s > 0 else 0.0


def max_dd(arr):
    if len(arr) == 0: return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def trades_to_daily_series(trades):
    if not trades: return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


def metrics(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'max_dd': 0, 'wr': 0, 'avg_pnl': 0}
    daily = trades_to_daily_series(trades)
    wins = [t for t in trades if t['pnl'] > 0]
    return {
        'n': len(trades),
        'sharpe': round(sharpe(daily.values), 3) if len(daily) > 0 else 0,
        'pnl': round(sum(t['pnl'] for t in trades), 2),
        'max_dd': round(max_dd(daily.values), 2) if len(daily) > 0 else 0,
        'wr': round(len(wins) / len(trades) * 100, 1),
        'avg_pnl': round(sum(t['pnl'] for t in trades) / len(trades), 3),
    }


def build_portfolio(trade_dict, lots):
    all_daily = {}
    for name, trades in trade_dict.items():
        lot = lots.get(name, UNIT_LOT)
        scale = lot / UNIT_LOT
        for t in trades:
            d = pd.Timestamp(t['exit_time']).date()
            all_daily[d] = all_daily.get(d, 0) + t['pnl'] * scale
    dates = sorted(all_daily.keys())
    return pd.Series([all_daily[d] for d in dates], index=pd.DatetimeIndex(dates))


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_h1():
    csv_candidates = sorted(DATA_DIR.glob("download/xauusd-h1-bid-*.csv"))
    if not csv_candidates:
        raise FileNotFoundError("No Dukascopy H1 CSV found in data/download/")
    csv_path = csv_candidates[-1]
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.set_index('time')[['open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[df['Close'] > 100]
    return df


def load_macro():
    macro_path = DATA_DIR / "external" / "aligned_daily.csv"
    if not macro_path.exists():
        print(f"  WARNING: {macro_path} not found, macro features unavailable", flush=True)
        return None
    df = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


# ═══════════════════════════════════════════════════════════════
# Regime detection methods
# ═══════════════════════════════════════════════════════════════

def build_regime_features(macro, h1_daily):
    """Build rolling 60-day regime features from macro data."""
    feat = pd.DataFrame(index=macro.index)

    if 'VIX_Close' in macro.columns:
        vix = macro['VIX_Close'].ffill()
        vix_mean = vix.rolling(60).mean()
        vix_std = vix.rolling(60).std()
        feat['vix_z'] = (vix - vix_mean) / vix_std.replace(0, np.nan)
        feat['vix_level'] = vix
    else:
        feat['vix_z'] = 0.0
        feat['vix_level'] = 20.0

    if 'DXY_Close' in macro.columns:
        dxy = macro['DXY_Close'].ffill()
        feat['dxy_mom'] = dxy.pct_change(20)
    else:
        feat['dxy_mom'] = 0.0

    if 'US10Y_Close' in macro.columns and 'US2Y_Close' in macro.columns:
        feat['yield_curve'] = macro['US10Y_Close'].ffill() - macro['US2Y_Close'].ffill()
    elif 'YIELD_CURVE_10Y2Y' in macro.columns:
        feat['yield_curve'] = macro['YIELD_CURVE_10Y2Y'].ffill()
    else:
        feat['yield_curve'] = 0.0

    if 'GVZ_Close' in macro.columns:
        feat['gvz_level'] = macro['GVZ_Close'].ffill()
    else:
        feat['gvz_level'] = 18.0

    if 'CREDIT_STRESS' in macro.columns:
        feat['credit_stress'] = macro['CREDIT_STRESS'].ffill()
    else:
        feat['credit_stress'] = 0.0

    if 'GLD_Close' in macro.columns:
        gold_price = macro['GLD_Close'].ffill()
    elif h1_daily is not None:
        gold_price = h1_daily
    else:
        gold_price = pd.Series(1800, index=macro.index)

    feat['gold_sma200'] = gold_price.rolling(200).mean()
    feat['gold_price'] = gold_price
    feat['gold_above_sma200'] = (gold_price > feat['gold_sma200']).astype(int)

    return feat.dropna(how='all')


def regime_kmeans(features, n_clusters=3):
    """KMeans on normalized rolling features -> cluster labels."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    cols = [c for c in ['vix_z', 'dxy_mom', 'yield_curve', 'gvz_level', 'credit_stress']
            if c in features.columns]
    feat_df = features[cols].dropna()
    if len(feat_df) < 60:
        return pd.Series(1, index=features.index, name='regime')

    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df.values)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    regime = pd.Series(labels, index=feat_df.index, name='regime')

    cluster_vix = {}
    for c_id in range(n_clusters):
        mask = regime == c_id
        if mask.sum() > 0:
            cluster_vix[c_id] = feat_df.loc[mask, 'vix_z'].mean() if 'vix_z' in feat_df.columns else 0
        else:
            cluster_vix[c_id] = 0

    sorted_clusters = sorted(cluster_vix.items(), key=lambda x: x[1])
    label_map = {}
    label_map[sorted_clusters[0][0]] = 0   # bull (low VIX z)
    label_map[sorted_clusters[-1][0]] = 2   # bear (high VIX z)
    for c_id in range(n_clusters):
        if c_id not in label_map:
            label_map[c_id] = 1             # neutral

    regime = regime.map(label_map)
    return regime.reindex(features.index, method='ffill')


def regime_rules(features):
    """Rule-based thresholds: Bull/Neutral/Bear."""
    vix = features.get('vix_level', pd.Series(20.0, index=features.index))
    gold_above = features.get('gold_above_sma200', pd.Series(1, index=features.index))

    regime = pd.Series(1, index=features.index, name='regime')  # neutral default

    bull_mask = (vix < 20) & (gold_above == 1)
    bear_mask = (vix > 25) | (gold_above == 0)

    regime[bull_mask] = 0
    regime[bear_mask] = 2

    regime[bull_mask & bear_mask] = 1

    return regime


def regime_hmm(features, n_states=3):
    """HMM on returns + volatility, fallback to KMeans."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("    hmmlearn not installed, falling back to KMeans for HMM", flush=True)
        return regime_kmeans(features, n_clusters=n_states)

    gold_price = features.get('gold_price', None)
    if gold_price is None or gold_price.dropna().empty:
        return regime_kmeans(features, n_clusters=n_states)

    ret = gold_price.pct_change().dropna()
    vol = ret.rolling(20).std().dropna()

    common = ret.index.intersection(vol.index)
    if len(common) < 100:
        return regime_kmeans(features, n_clusters=n_states)

    X = np.column_stack([ret.loc[common].values, vol.loc[common].values])
    valid_mask = np.all(np.isfinite(X), axis=1)
    X = X[valid_mask]
    valid_idx = common[valid_mask]

    if len(X) < 100:
        return regime_kmeans(features, n_clusters=n_states)

    try:
        model = GaussianHMM(n_components=n_states, covariance_type='diag',
                            n_iter=200, random_state=42)
        model.fit(X)
        labels = model.predict(X)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"    HMM fit failed ({e}), falling back to KMeans", flush=True)
        return regime_kmeans(features, n_clusters=n_states)

    regime = pd.Series(labels, index=valid_idx, name='regime')

    cluster_vol = {}
    for s in range(n_states):
        mask = regime == s
        if mask.sum() > 0:
            cluster_vol[s] = X[labels == s, 1].mean()
        else:
            cluster_vol[s] = 0
    sorted_states = sorted(cluster_vol.items(), key=lambda x: x[1])
    label_map = {}
    label_map[sorted_states[0][0]] = 0   # bull (low vol)
    label_map[sorted_states[-1][0]] = 2   # bear (high vol)
    for s in range(n_states):
        if s not in label_map:
            label_map[s] = 1

    regime = regime.map(label_map)
    return regime.reindex(features.index, method='ffill')


REGIME_NAMES = {0: 'Bull', 1: 'Neutral', 2: 'Bear'}

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80, flush=True)
    print("  R121 — Multi-Factor Regime Detection", flush=True)
    print("=" * 80, flush=True)

    # ════════════════════════════════════════════════════════════════
    # Phase 1: Load macro data + run baseline portfolio
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 1: Load Data + Baseline Portfolio", flush=True)
    print("=" * 70, flush=True)

    h1 = load_h1()
    print(f"  H1 data: {len(h1)} bars ({h1.index[0]} ~ {h1.index[-1]})", flush=True)

    macro = load_macro()
    if macro is not None:
        print(f"  Macro data: {len(macro)} rows ({macro.index[0].date()} ~ {macro.index[-1].date()})", flush=True)
        print(f"  Macro columns: {list(macro.columns[:10])}... ({len(macro.columns)} total)", flush=True)
    else:
        print("  No macro data — will use gold-only features for regime detection", flush=True)

    h1_daily_close = h1['Close'].resample('D').last().dropna()

    print("\n  Running 4 baseline strategies...", flush=True)
    bundle = DataBundle.load_custom()
    psar_trades = bt_psar(h1, SPREAD, UNIT_LOT, maxloss_cap=5)
    tsmom_trades = bt_tsmom(h1, SPREAD, UNIT_LOT, maxloss_cap=0)
    sess_trades = bt_sess_bo(h1, SPREAD, UNIT_LOT, maxloss_cap=35)
    l8_trades = bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=35)

    strat_trades = {
        'PSAR': psar_trades, 'TSMOM': tsmom_trades,
        'SESS_BO': sess_trades, 'L8_MAX': l8_trades,
    }

    BASE_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}
    baseline_port = build_portfolio(strat_trades, BASE_LOTS)
    baseline_sharpe = sharpe(baseline_port.values)
    baseline_pnl = baseline_port.sum()
    baseline_dd = max_dd(baseline_port.values)

    print(f"\n  Baseline portfolio (fixed lots):", flush=True)
    print(f"    Sharpe={baseline_sharpe:.3f}, PnL=${baseline_pnl:.0f}, MaxDD=${baseline_dd:.0f}", flush=True)
    for name, trades in strat_trades.items():
        m = metrics(trades)
        print(f"    {name:10s}: {m['n']:5d} trades, Sharpe={m['sharpe']:6.3f}, "
              f"PnL=${m['pnl']:>8.0f}, WR={m['wr']:.1f}%", flush=True)

    all_results = {
        'experiment': 'R121 Multi-Factor Regime Detection',
        'baseline': {
            'sharpe': round(baseline_sharpe, 3),
            'pnl': round(baseline_pnl, 2),
            'max_dd': round(baseline_dd, 2),
            'strategies': {name: metrics(trades) for name, trades in strat_trades.items()},
        },
    }

    # ════════════════════════════════════════════════════════════════
    # Phase 2: Regime Detection
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: Regime Detection (KMeans, Rule-based, HMM)", flush=True)
    print("=" * 70, flush=True)

    if macro is not None:
        features = build_regime_features(macro, h1_daily_close)
    else:
        features = pd.DataFrame(index=h1_daily_close.index)
        features['gold_price'] = h1_daily_close
        features['gold_sma200'] = h1_daily_close.rolling(200).mean()
        features['gold_above_sma200'] = (h1_daily_close > features['gold_sma200']).astype(int)
        features['vix_z'] = 0.0
        features['vix_level'] = 20.0
        features['dxy_mom'] = 0.0
        features['yield_curve'] = 0.0
        features['gvz_level'] = 18.0
        features['credit_stress'] = 0.0

    print(f"  Feature matrix: {len(features)} rows", flush=True)

    regimes = {}

    try:
        from sklearn.cluster import KMeans as _KM
        reg_km = regime_kmeans(features, n_clusters=3)
        regimes['KMeans'] = reg_km
        print(f"\n  KMeans regime distribution:", flush=True)
        for r_id in [0, 1, 2]:
            cnt = (reg_km == r_id).sum()
            pct = cnt / len(reg_km) * 100
            print(f"    {REGIME_NAMES[r_id]:8s}: {cnt:5d} days ({pct:.1f}%)", flush=True)
    except ImportError:
        print("  sklearn not available, skipping KMeans", flush=True)

    reg_rules = regime_rules(features)
    regimes['Rules'] = reg_rules
    print(f"\n  Rule-based regime distribution:", flush=True)
    for r_id in [0, 1, 2]:
        cnt = (reg_rules == r_id).sum()
        pct = cnt / len(reg_rules) * 100
        print(f"    {REGIME_NAMES[r_id]:8s}: {cnt:5d} days ({pct:.1f}%)", flush=True)

    reg_hmm = regime_hmm(features, n_states=3)
    regimes['HMM'] = reg_hmm
    print(f"\n  HMM regime distribution:", flush=True)
    for r_id in [0, 1, 2]:
        cnt = (reg_hmm == r_id).sum()
        pct = cnt / len(reg_hmm) * 100 if len(reg_hmm) > 0 else 0
        print(f"    {REGIME_NAMES[r_id]:8s}: {cnt:5d} days ({pct:.1f}%)", flush=True)

    regime_info = {}
    for method, reg in regimes.items():
        dist = {}
        for r_id in [0, 1, 2]:
            cnt = int((reg == r_id).sum())
            dist[REGIME_NAMES[r_id]] = cnt
        regime_info[method] = dist
    all_results['phase2_regimes'] = regime_info

    # ════════════════════════════════════════════════════════════════
    # Phase 3: Per-regime strategy performance
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3: Per-Regime Strategy Performance", flush=True)
    print("=" * 70, flush=True)

    phase3 = {}
    for method, reg in regimes.items():
        print(f"\n  === {method} ===", flush=True)
        method_results = {}
        for strat_name, trades in strat_trades.items():
            strat_regime_perf = {}
            for r_id in [0, 1, 2]:
                regime_dates = set(reg[reg == r_id].index.date)
                regime_trades = [t for t in trades
                                 if pd.Timestamp(t['exit_time']).normalize().date() in regime_dates]
                m = metrics(regime_trades)
                strat_regime_perf[REGIME_NAMES[r_id]] = m
            method_results[strat_name] = strat_regime_perf

            print(f"    {strat_name:10s}: "
                  f"Bull(n={strat_regime_perf['Bull']['n']:4d}, S={strat_regime_perf['Bull']['sharpe']:6.3f})  "
                  f"Neut(n={strat_regime_perf['Neutral']['n']:4d}, S={strat_regime_perf['Neutral']['sharpe']:6.3f})  "
                  f"Bear(n={strat_regime_perf['Bear']['n']:4d}, S={strat_regime_perf['Bear']['sharpe']:6.3f})",
                  flush=True)

        phase3[method] = method_results

    all_results['phase3_per_regime'] = phase3

    # ════════════════════════════════════════════════════════════════
    # Phase 4: Regime-conditional lot multipliers
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 4: Regime-Conditional Lot Multipliers", flush=True)
    print("=" * 70, flush=True)

    MULT_GRID = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]

    phase4 = {}
    for method, reg in regimes.items():
        print(f"\n  === {method} ===", flush=True)
        regime_dates_map = {}
        for r_id in [0, 1, 2]:
            regime_dates_map[r_id] = set(reg[reg == r_id].index.date)

        best_combo = None
        best_combo_sharpe = -999

        for m_bull in MULT_GRID:
            for m_neut in MULT_GRID:
                for m_bear in MULT_GRID:
                    if m_bull == 1.0 and m_neut == 1.0 and m_bear == 1.0:
                        continue
                    all_daily = {}
                    for strat_name, trades in strat_trades.items():
                        lot = BASE_LOTS.get(strat_name, UNIT_LOT)
                        for t in trades:
                            d = pd.Timestamp(t['exit_time']).normalize().date()
                            if d in regime_dates_map[0]:
                                mult = m_bull
                            elif d in regime_dates_map[2]:
                                mult = m_bear
                            else:
                                mult = m_neut
                            scale = (lot / UNIT_LOT) * mult
                            all_daily[d] = all_daily.get(d, 0) + t['pnl'] * scale
                    dates = sorted(all_daily.keys())
                    if len(dates) < 30:
                        continue
                    arr = np.array([all_daily[d] for d in dates])
                    sh = sharpe(arr)
                    if sh > best_combo_sharpe:
                        best_combo_sharpe = sh
                        best_combo = (m_bull, m_neut, m_bear)

        if best_combo is None:
            best_combo = (1.0, 1.0, 1.0)
            best_combo_sharpe = baseline_sharpe

        all_daily_adj = {}
        for strat_name, trades in strat_trades.items():
            lot = BASE_LOTS.get(strat_name, UNIT_LOT)
            for t in trades:
                d = pd.Timestamp(t['exit_time']).normalize().date()
                if d in regime_dates_map[0]:
                    mult = best_combo[0]
                elif d in regime_dates_map[2]:
                    mult = best_combo[2]
                else:
                    mult = best_combo[1]
                scale = (lot / UNIT_LOT) * mult
                all_daily_adj[d] = all_daily_adj.get(d, 0) + t['pnl'] * scale
        dates = sorted(all_daily_adj.keys())
        arr_adj = np.array([all_daily_adj[d] for d in dates])
        adj_sharpe = sharpe(arr_adj)
        adj_pnl = arr_adj.sum()
        adj_dd = max_dd(arr_adj)

        print(f"    Best multipliers: Bull={best_combo[0]}, Neutral={best_combo[1]}, Bear={best_combo[2]}", flush=True)
        print(f"    Adjusted: Sharpe={adj_sharpe:.3f}, PnL=${adj_pnl:.0f}, MaxDD=${adj_dd:.0f}", flush=True)
        print(f"    Baseline: Sharpe={baseline_sharpe:.3f}, PnL=${baseline_pnl:.0f}, MaxDD=${baseline_dd:.0f}", flush=True)
        delta = adj_sharpe - baseline_sharpe
        print(f"    Delta Sharpe: {delta:+.3f} ({'IMPROVED' if delta > 0 else 'NO IMPROVEMENT'})", flush=True)

        phase4[method] = {
            'best_mult': {'bull': best_combo[0], 'neutral': best_combo[1], 'bear': best_combo[2]},
            'adjusted_sharpe': round(adj_sharpe, 3),
            'adjusted_pnl': round(adj_pnl, 2),
            'adjusted_max_dd': round(adj_dd, 2),
            'baseline_sharpe': round(baseline_sharpe, 3),
            'delta_sharpe': round(delta, 3),
        }

    all_results['phase4_multipliers'] = phase4

    # ════════════════════════════════════════════════════════════════
    # Phase 5: K-Fold Validation (5 folds)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 5: K-Fold Validation (5 folds)", flush=True)
    print("=" * 70, flush=True)

    phase5 = {}
    for method in regimes.keys():
        print(f"\n  === {method} ===", flush=True)
        best_mult = phase4[method]['best_mult']
        m_bull = best_mult['bull']
        m_neut = best_mult['neutral']
        m_bear = best_mult['bear']

        fold_sharpes_adj = []
        fold_sharpes_fix = []

        for fname, start, end in FOLDS:
            fold_h1 = h1[(h1.index >= start) & (h1.index < end)]
            if len(fold_h1) < 200:
                fold_sharpes_adj.append(0.0)
                fold_sharpes_fix.append(0.0)
                continue

            fold_h1_daily = fold_h1['Close'].resample('D').last().dropna()

            if macro is not None:
                fold_macro = macro[(macro.index >= start) & (macro.index < end)]
                if len(fold_macro) < 30:
                    fold_features = pd.DataFrame(index=fold_h1_daily.index)
                    fold_features['gold_price'] = fold_h1_daily
                    fold_features['gold_sma200'] = fold_h1_daily.rolling(200).mean()
                    fold_features['gold_above_sma200'] = (fold_h1_daily > fold_features['gold_sma200']).astype(int)
                    fold_features['vix_z'] = 0; fold_features['vix_level'] = 20
                    fold_features['dxy_mom'] = 0; fold_features['yield_curve'] = 0
                    fold_features['gvz_level'] = 18; fold_features['credit_stress'] = 0
                else:
                    fold_features = build_regime_features(fold_macro, fold_h1_daily)
            else:
                fold_features = pd.DataFrame(index=fold_h1_daily.index)
                fold_features['gold_price'] = fold_h1_daily
                fold_features['gold_sma200'] = fold_h1_daily.rolling(200).mean()
                fold_features['gold_above_sma200'] = (fold_h1_daily > fold_features['gold_sma200']).astype(int)
                fold_features['vix_z'] = 0; fold_features['vix_level'] = 20
                fold_features['dxy_mom'] = 0; fold_features['yield_curve'] = 0
                fold_features['gvz_level'] = 18; fold_features['credit_stress'] = 0

            if method == 'KMeans':
                try:
                    fold_reg = regime_kmeans(fold_features, n_clusters=3)
                except Exception:
                    fold_reg = pd.Series(1, index=fold_features.index)
            elif method == 'Rules':
                fold_reg = regime_rules(fold_features)
            else:
                fold_reg = regime_hmm(fold_features, n_states=3)

            regime_dates_map = {}
            for r_id in [0, 1, 2]:
                regime_dates_map[r_id] = set(fold_reg[fold_reg == r_id].index.date)

            fold_strats = {
                'PSAR': bt_psar(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=5),
                'TSMOM': bt_tsmom(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=0),
                'SESS_BO': bt_sess_bo(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=35),
                'L8_MAX': bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=35),
            }

            fix_port = build_portfolio(fold_strats, BASE_LOTS)
            fold_sharpes_fix.append(round(sharpe(fix_port.values), 3))

            adj_daily = {}
            for sname, trades in fold_strats.items():
                lot = BASE_LOTS.get(sname, UNIT_LOT)
                for t in trades:
                    d = pd.Timestamp(t['exit_time']).normalize().date()
                    if d in regime_dates_map[0]:
                        mult = m_bull
                    elif d in regime_dates_map[2]:
                        mult = m_bear
                    else:
                        mult = m_neut
                    scale = (lot / UNIT_LOT) * mult
                    adj_daily[d] = adj_daily.get(d, 0) + t['pnl'] * scale

            if adj_daily:
                dates = sorted(adj_daily.keys())
                arr = np.array([adj_daily[d] for d in dates])
                fold_sharpes_adj.append(round(sharpe(arr), 3))
            else:
                fold_sharpes_adj.append(0.0)

        pos_adj = sum(1 for s in fold_sharpes_adj if s > 0)
        pos_fix = sum(1 for s in fold_sharpes_fix if s > 0)
        wins = sum(1 for a, f in zip(fold_sharpes_adj, fold_sharpes_fix) if a > f)
        status = "PASS" if pos_adj >= 3 else "FAIL"

        print(f"    Fixed:    {fold_sharpes_fix}  mean={np.mean(fold_sharpes_fix):.3f}", flush=True)
        print(f"    Adjusted: {fold_sharpes_adj}  mean={np.mean(fold_sharpes_adj):.3f}", flush=True)
        print(f"    Adjusted wins: {wins}/5 folds, positive: {pos_adj}/5 [{status}]", flush=True)

        phase5[method] = {
            'fold_sharpes_fixed': fold_sharpes_fix,
            'fold_sharpes_adjusted': fold_sharpes_adj,
            'mean_fixed': round(np.mean(fold_sharpes_fix), 3),
            'mean_adjusted': round(np.mean(fold_sharpes_adj), 3),
            'adjusted_wins': wins,
            'positive_folds': pos_adj,
            'pass': pos_adj >= 3,
        }

    all_results['phase5_kfold'] = phase5

    # ════════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    all_results['elapsed_s'] = round(elapsed, 1)

    print("\n" + "=" * 80, flush=True)
    print("  R121 FINAL SUMMARY", flush=True)
    print("=" * 80, flush=True)

    print(f"\n  Baseline portfolio: Sharpe={baseline_sharpe:.3f}, PnL=${baseline_pnl:.0f}", flush=True)

    for method in regimes.keys():
        p4 = phase4[method]
        p5 = phase5[method]
        status = "PASS" if p5['pass'] else "FAIL"
        print(f"\n  {method}:", flush=True)
        print(f"    Best multipliers: Bull={p4['best_mult']['bull']}, "
              f"Neutral={p4['best_mult']['neutral']}, Bear={p4['best_mult']['bear']}", flush=True)
        print(f"    Full-sample: Sharpe={p4['adjusted_sharpe']:.3f} (delta={p4['delta_sharpe']:+.3f})", flush=True)
        print(f"    K-Fold: mean_adj={p5['mean_adjusted']:.3f} vs mean_fix={p5['mean_fixed']:.3f}, "
              f"wins={p5['adjusted_wins']}/5, positive={p5['positive_folds']}/5 [{status}]", flush=True)

    best_method = max(phase5.keys(), key=lambda m: phase5[m]['mean_adjusted'])
    best_p5 = phase5[best_method]
    print(f"\n  Best method: {best_method} (mean_adj_Sharpe={best_p5['mean_adjusted']:.3f})", flush=True)

    all_results['recommendation'] = {
        'best_method': best_method,
        'mean_adjusted_sharpe': best_p5['mean_adjusted'],
        'kfold_pass': best_p5['pass'],
        'multipliers': phase4[best_method]['best_mult'],
    }

    out_file = OUTPUT_DIR / "r121_results.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Saved: {out_file}", flush=True)
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)", flush=True)


if __name__ == '__main__':
    main()
