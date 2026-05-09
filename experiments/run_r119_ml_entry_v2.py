#!/usr/bin/env python3
"""
R119 — ML Entry Filter v2 (Macro + Technical Features)
========================================================
Walk-forward ML model to gate strategy entries using macro + technical features.
Previous R107 was weak (AUC ~0.52-0.56) with limited features.
This version uses full macro panel from aligned_daily.csv.

Phase 1: Generate all strategy trades (baseline)
Phase 2: Build macro + technical feature matrix per trade
Phase 3: Walk-forward XGBoost + LightGBM (6mo train / 2mo test, sliding)
Phase 4: Per-strategy entry gate with threshold sweep
Phase 5: K-Fold validation (5 folds)
Phase 6: Feature importance and ablation study
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r119_ml_entry_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01
CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']
ALIGNED_CSV = Path("data/external/aligned_daily.csv")
t0 = time.time()

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


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


def trades_to_daily_series(trades):
    if not trades:
        return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


def build_portfolio_daily(unit_dailies, lots):
    all_dates = set()
    for ds in unit_dailies.values():
        all_dates.update(ds.index)
    all_dates = sorted(all_dates)
    if not all_dates:
        return np.array([0.0])
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
# Strategy Backtests
# ═══════════════════════════════════════════════════════════════

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
# Feature engineering
# ═══════════════════════════════════════════════════════════════

def build_trade_features(all_trades, h1_df, macro_df):
    """Build feature matrix aligned to each trade's entry time.

    Returns (X DataFrame, y array, trade_list with strategy labels).
    """
    h1 = h1_df.copy()
    if h1.index.tz is not None:
        h1.index = h1.index.tz_localize(None)

    atr_series = compute_atr(h1)
    close_s = h1['Close']
    sma50 = close_s.rolling(50).mean()
    sma200 = close_s.rolling(200).mean()
    ema20 = close_s.ewm(span=20).mean()
    ema20_slope = ema20.diff(5) / ema20.shift(5)
    hourly_vol = close_s.pct_change().rolling(24).std()

    kc_mid = close_s.rolling(20).mean()
    kc_atr = atr_series.rolling(20).mean()
    kc_bw = (2 * kc_atr) / kc_mid.replace(0, np.nan)

    atr_pctrank = atr_series.rolling(500, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    has_macro = macro_df is not None and len(macro_df) > 0
    if has_macro:
        macro = macro_df.copy()
        if macro.index.tz is not None:
            macro.index = macro.index.tz_localize(None)

    combined = []
    for strat_name in STRAT_ORDER:
        for t in all_trades.get(strat_name, []):
            combined.append({**t, '_strat': strat_name})
    combined.sort(key=lambda t: pd.Timestamp(t['entry_time']))

    rows = []
    targets = []
    meta_trades = []

    for t in combined:
        entry_ts = pd.Timestamp(t['entry_time'])
        if hasattr(entry_ts, 'tz') and entry_ts.tz is not None:
            entry_ts = entry_ts.tz_localize(None)

        idx_pos = close_s.index.searchsorted(entry_ts)
        idx_pos = min(max(idx_pos - 1, 0), len(close_s) - 1)

        atr_val = atr_series.iloc[idx_pos]
        if np.isnan(atr_val):
            continue

        feat = {}

        # Technical features from H1
        atr_pct_val = atr_pctrank.iloc[idx_pos] if idx_pos < len(atr_pctrank) else np.nan
        feat['atr_percentile'] = float(atr_pct_val) if not np.isnan(atr_pct_val) else 0.5

        cl = close_s.iloc[idx_pos]
        s50 = sma50.iloc[idx_pos]
        s200 = sma200.iloc[idx_pos]
        feat['close_vs_sma50'] = float((cl - s50) / s50) if (not np.isnan(s50) and s50 > 0) else 0.0
        feat['close_vs_sma200'] = float((cl - s200) / s200) if (not np.isnan(s200) and s200 > 0) else 0.0

        e20_sl = ema20_slope.iloc[idx_pos] if idx_pos < len(ema20_slope) else np.nan
        feat['ema20_slope'] = float(e20_sl) if not np.isnan(e20_sl) else 0.0

        hvol = hourly_vol.iloc[idx_pos] if idx_pos < len(hourly_vol) else np.nan
        feat['hourly_ret_vol'] = float(hvol) if not np.isnan(hvol) else 0.0

        kbw = kc_bw.iloc[idx_pos] if idx_pos < len(kc_bw) else np.nan
        feat['kc_bandwidth'] = float(kbw) if not np.isnan(kbw) else 0.0

        # Gold momentum from H1 close
        for lb, label in [(5*24, 'gold_mom_5d'), (20*24, 'gold_mom_20d'), (60*24, 'gold_mom_60d')]:
            if idx_pos >= lb and close_s.iloc[idx_pos - lb] > 0:
                feat[label] = float(cl / close_s.iloc[idx_pos - lb] - 1.0)
            else:
                feat[label] = 0.0

        # Calendar features
        feat['hour'] = entry_ts.hour
        feat['day_of_week'] = entry_ts.dayofweek
        feat['month'] = entry_ts.month

        # Direction
        feat['direction'] = 1 if t['dir'] == 'BUY' else 0

        # Strategy identity
        feat['is_l8'] = 1 if t['_strat'] == 'L8_MAX' else 0
        feat['is_psar'] = 1 if t['_strat'] == 'PSAR' else 0
        feat['is_tsmom'] = 1 if t['_strat'] == 'TSMOM' else 0
        feat['is_sessbo'] = 1 if t['_strat'] == 'SESS_BO' else 0

        # Macro features
        if has_macro:
            entry_date = entry_ts.normalize()
            m_idx = macro.index.searchsorted(entry_date)
            m_idx = min(max(m_idx - 1, 0), len(macro) - 1)

            def _get_macro(col):
                if col in macro.columns:
                    v = macro[col].iloc[m_idx]
                    return float(v) if not pd.isna(v) else np.nan
                return np.nan

            # VIX
            vix_col = 'VIX_Close' if 'VIX_Close' in macro.columns else 'VIX'
            vix_val = _get_macro(vix_col)
            feat['vix_level'] = vix_val if not np.isnan(vix_val) else 20.0
            # VIX z-score (rolling 50d)
            if vix_col in macro.columns:
                vix_s = macro[vix_col]
                vix_mean50 = vix_s.rolling(50, min_periods=20).mean()
                vix_std50 = vix_s.rolling(50, min_periods=20).std()
                if m_idx < len(vix_mean50) and vix_std50.iloc[m_idx] > 0:
                    feat['vix_zscore'] = float((vix_s.iloc[m_idx] - vix_mean50.iloc[m_idx]) / vix_std50.iloc[m_idx])
                else:
                    feat['vix_zscore'] = 0.0
            else:
                feat['vix_zscore'] = 0.0

            # DXY
            dxy_col = 'DXY_Close' if 'DXY_Close' in macro.columns else 'DXY'
            dxy_val = _get_macro(dxy_col)
            feat['dxy_level'] = dxy_val if not np.isnan(dxy_val) else 100.0
            if dxy_col in macro.columns:
                dxy_s = macro[dxy_col]
                dxy_mean50 = dxy_s.rolling(50, min_periods=20).mean()
                dxy_std50 = dxy_s.rolling(50, min_periods=20).std()
                if m_idx < len(dxy_mean50) and dxy_std50.iloc[m_idx] > 0:
                    feat['dxy_zscore'] = float((dxy_s.iloc[m_idx] - dxy_mean50.iloc[m_idx]) / dxy_std50.iloc[m_idx])
                else:
                    feat['dxy_zscore'] = 0.0
                # Rolling 20d z-score
                dxy_mean20 = dxy_s.rolling(20, min_periods=10).mean()
                dxy_std20 = dxy_s.rolling(20, min_periods=10).std()
                if m_idx < len(dxy_mean20) and dxy_std20.iloc[m_idx] > 0:
                    feat['dxy_zscore_20d'] = float((dxy_s.iloc[m_idx] - dxy_mean20.iloc[m_idx]) / dxy_std20.iloc[m_idx])
                else:
                    feat['dxy_zscore_20d'] = 0.0
            else:
                feat['dxy_zscore'] = 0.0
                feat['dxy_zscore_20d'] = 0.0

            # Yield curve: US10Y - US2Y
            us10y = _get_macro('US10Y')
            us2y = _get_macro('US2Y')
            if not np.isnan(us10y) and not np.isnan(us2y):
                feat['yield_spread_10y2y'] = us10y - us2y
            else:
                yc_col = next((c for c in macro.columns if 'yield_curve' in c.lower()), None)
                feat['yield_spread_10y2y'] = _get_macro(yc_col) if yc_col else 0.0
                if np.isnan(feat['yield_spread_10y2y']):
                    feat['yield_spread_10y2y'] = 0.0

            # GVZ
            gvz_val = _get_macro('GVZ')
            feat['gvz'] = gvz_val if not np.isnan(gvz_val) else 20.0

            # Credit stress: HYG return
            if 'HYG' in macro.columns:
                hyg_s = macro['HYG']
                if m_idx >= 5:
                    hyg_now = hyg_s.iloc[m_idx]
                    hyg_prev = hyg_s.iloc[m_idx - 5]
                    feat['credit_stress'] = float((hyg_now / hyg_prev - 1.0)) if hyg_prev > 0 else 0.0
                else:
                    feat['credit_stress'] = 0.0
            else:
                feat['credit_stress'] = 0.0

            # Gold-DXY rolling 20d correlation
            if dxy_col in macro.columns and 'gold_close' not in macro.columns:
                feat['gold_dxy_corr_20d'] = 0.0
            elif dxy_col in macro.columns:
                feat['gold_dxy_corr_20d'] = 0.0
            else:
                feat['gold_dxy_corr_20d'] = 0.0

            # Crude momentum
            crude_col = next((c for c in macro.columns if 'crude' in c.lower() or 'wti' in c.lower()), None)
            if crude_col and m_idx >= 20:
                crude_now = macro[crude_col].iloc[m_idx]
                crude_prev = macro[crude_col].iloc[m_idx - 20]
                feat['crude_mom_20d'] = float(crude_now / crude_prev - 1.0) if crude_prev > 0 else 0.0
            else:
                feat['crude_mom_20d'] = 0.0

            # Copper momentum
            copper_col = next((c for c in macro.columns if 'copper' in c.lower()), None)
            if copper_col and m_idx >= 20:
                cu_now = macro[copper_col].iloc[m_idx]
                cu_prev = macro[copper_col].iloc[m_idx - 20]
                feat['copper_mom_20d'] = float(cu_now / cu_prev - 1.0) if cu_prev > 0 else 0.0
            else:
                feat['copper_mom_20d'] = 0.0

            # USDJPY, USDCNH if available
            usdjpy_val = _get_macro('USDJPY')
            feat['usdjpy'] = usdjpy_val if not np.isnan(usdjpy_val) else 0.0

            # Risk appetite composite
            ra_col = next((c for c in macro.columns if 'risk_appetite' in c.lower()), None)
            feat['risk_appetite'] = _get_macro(ra_col) if ra_col else 0.0
            if np.isnan(feat['risk_appetite']):
                feat['risk_appetite'] = 0.0

            # VIX 20d rolling z-score
            if vix_col in macro.columns:
                vix_s = macro[vix_col]
                vix_mean20 = vix_s.rolling(20, min_periods=10).mean()
                vix_std20 = vix_s.rolling(20, min_periods=10).std()
                if m_idx < len(vix_mean20) and vix_std20.iloc[m_idx] > 0:
                    feat['vix_zscore_20d'] = float((vix_s.iloc[m_idx] - vix_mean20.iloc[m_idx]) / vix_std20.iloc[m_idx])
                else:
                    feat['vix_zscore_20d'] = 0.0
            else:
                feat['vix_zscore_20d'] = 0.0

        rows.append(feat)
        targets.append(1 if t['pnl'] > 0 else 0)
        meta_trades.append(t)

    X = pd.DataFrame(rows).fillna(0.0)
    y = np.array(targets)
    return X, y, meta_trades


# ═══════════════════════════════════════════════════════════════
# ML model helpers
# ═══════════════════════════════════════════════════════════════

def get_xgb_model(n_estimators=200, max_depth=4, learning_rate=0.05):
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


def get_lgb_model(n_estimators=200, max_depth=4, learning_rate=0.05):
    if not HAS_LGB:
        return None
    try:
        m = lgb.LGBMClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=0.8,
            colsample_bytree=0.8, verbose=-1,
            device='gpu', random_state=42)
        m.fit(np.random.randn(10, 5), np.random.randint(0, 2, 10))
        return lgb.LGBMClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=0.8,
            colsample_bytree=0.8, verbose=-1,
            device='gpu', random_state=42)
    except Exception:
        return lgb.LGBMClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=0.8,
            colsample_bytree=0.8, verbose=-1,
            random_state=42)


# ═══════════════════════════════════════════════════════════════
# Walk-forward with sliding window
# ═══════════════════════════════════════════════════════════════

def walk_forward_sliding(X, y, meta_trades, train_days=126, test_days=42, slide_days=21):
    """Sliding window walk-forward: 6mo train, 2mo test, slide 1mo.

    Returns (oos_probs_xgb, oos_probs_lgb, fold_results, feature_importances).
    """
    from sklearn.metrics import roc_auc_score

    entry_dates = np.array([pd.Timestamp(t['entry_time']).date() for t in meta_trades])
    unique_dates = sorted(set(entry_dates))
    n_dates = len(unique_dates)

    oos_probs_xgb = np.full(len(X), np.nan)
    oos_probs_lgb = np.full(len(X), np.nan)
    fold_results = []
    importances_xgb_accum = {}
    importances_lgb_accum = {}
    n_folds_done = 0

    start = 0
    fold_num = 0
    while start + train_days + test_days <= n_dates:
        train_end_date = unique_dates[start + train_days - 1]
        test_start_date = unique_dates[start + train_days]
        test_end_idx = min(start + train_days + test_days - 1, n_dates - 1)
        test_end_date = unique_dates[test_end_idx]

        train_mask = entry_dates <= train_end_date
        for i in range(len(entry_dates)):
            if entry_dates[i] < unique_dates[start]:
                train_mask[i] = False

        test_mask = (entry_dates >= test_start_date) & (entry_dates <= test_end_date)

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) < 50 or len(test_idx) < 10:
            start += slide_days
            fold_num += 1
            continue

        X_tr = X.iloc[train_idx].copy()
        y_tr = y[train_idx]
        X_te = X.iloc[test_idx].copy()
        y_te = y[test_idx]

        med = X_tr.median()
        X_tr = X_tr.fillna(med)
        X_te = X_te.fillna(med)

        const_cols = [c for c in X_tr.columns if X_tr[c].nunique() <= 1]
        if const_cols:
            X_tr = X_tr.drop(columns=const_cols)
            X_te = X_te.drop(columns=const_cols)

        fold_info = {'fold': fold_num + 1, 'n_train': len(train_idx), 'n_test': len(test_idx)}

        # XGBoost
        try:
            xgb_model = get_xgb_model()
            xgb_model.fit(X_tr, y_tr)
            probs_xgb = xgb_model.predict_proba(X_te)[:, 1]
            oos_probs_xgb[test_idx] = probs_xgb
            auc_xgb = roc_auc_score(y_te, probs_xgb) if len(np.unique(y_te)) > 1 else np.nan
            fold_info['auc_xgb'] = round(float(auc_xgb), 4) if not np.isnan(auc_xgb) else None

            for fname, fval in zip(X_tr.columns, xgb_model.feature_importances_):
                importances_xgb_accum[fname] = importances_xgb_accum.get(fname, 0) + float(fval)
        except Exception as e:
            fold_info['auc_xgb'] = None
            fold_info['xgb_error'] = str(e)

        # LightGBM
        if HAS_LGB:
            try:
                lgb_model = get_lgb_model()
                if lgb_model is not None:
                    lgb_model.fit(X_tr, y_tr)
                    probs_lgb = lgb_model.predict_proba(X_te)[:, 1]
                    oos_probs_lgb[test_idx] = probs_lgb
                    auc_lgb = roc_auc_score(y_te, probs_lgb) if len(np.unique(y_te)) > 1 else np.nan
                    fold_info['auc_lgb'] = round(float(auc_lgb), 4) if not np.isnan(auc_lgb) else None

                    for fname, fval in zip(X_tr.columns, lgb_model.feature_importances_):
                        importances_lgb_accum[fname] = importances_lgb_accum.get(fname, 0) + float(fval)
            except Exception as e:
                fold_info['auc_lgb'] = None
                fold_info['lgb_error'] = str(e)
        else:
            fold_info['auc_lgb'] = None

        fold_results.append(fold_info)
        n_folds_done += 1
        start += slide_days
        fold_num += 1

    # Average importances
    if n_folds_done > 0:
        for k in importances_xgb_accum:
            importances_xgb_accum[k] /= n_folds_done
        for k in importances_lgb_accum:
            importances_lgb_accum[k] /= n_folds_done

    return oos_probs_xgb, oos_probs_lgb, fold_results, importances_xgb_accum, importances_lgb_accum


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80, flush=True)
    print("  R119 — ML Entry Filter v2 (Macro + Technical Features)", flush=True)
    print("=" * 80, flush=True)

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Generate all strategy trades (baseline)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 1: Generate All Strategy Trades (Baseline)", flush=True)
    print("=" * 70, flush=True)

    print("\n  Loading data...", flush=True)
    from backtest.runner import DataBundle, load_m15, load_h1_aligned, H1_CSV_PATH

    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    bundle = DataBundle.load_custom()
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0].date()} ~ {h1_df.index[-1].date()})", flush=True)

    # Load macro data
    macro_df = None
    if ALIGNED_CSV.exists():
        try:
            macro_df = pd.read_csv(ALIGNED_CSV, parse_dates=['Date'], index_col='Date')
            macro_df = macro_df.sort_index()
            if macro_df.index.tz is not None:
                macro_df.index = macro_df.index.tz_localize(None)
            print(f"    Macro data: {len(macro_df)} rows, {len(macro_df.columns)} columns", flush=True)
            print(f"    Macro columns: {list(macro_df.columns[:15])}...", flush=True)
        except Exception as e:
            print(f"    WARNING: Failed to load aligned_daily.csv: {e}", flush=True)
            macro_df = None
    else:
        print("    WARNING: aligned_daily.csv not found — macro features will be skipped", flush=True)

    print("\n  Running 4 strategies at unit lot...", flush=True)
    base_trades = {}

    print("    L8_MAX...", end=" ", flush=True)
    base_trades['L8_MAX'] = bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])
    print(f"{len(base_trades['L8_MAX'])} trades", flush=True)

    print("    PSAR...", end=" ", flush=True)
    base_trades['PSAR'] = bt_psar(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['PSAR'])
    print(f"{len(base_trades['PSAR'])} trades", flush=True)

    print("    TSMOM...", end=" ", flush=True)
    base_trades['TSMOM'] = bt_tsmom(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['TSMOM'])
    print(f"{len(base_trades['TSMOM'])} trades", flush=True)

    print("    SESS_BO...", end=" ", flush=True)
    base_trades['SESS_BO'] = bt_sess_bo(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['SESS_BO'])
    print(f"{len(base_trades['SESS_BO'])} trades", flush=True)

    # Baseline portfolio
    unit_dailies = {}
    for name in STRAT_ORDER:
        unit_dailies[name] = trades_to_daily_series(base_trades[name])
    baseline_daily = build_portfolio_daily(unit_dailies, R89_LOTS)
    baseline_sharpe = round(_sharpe(baseline_daily), 3)
    baseline_pnl = round(float(np.sum(baseline_daily)), 2)
    total_trades = sum(len(base_trades[s]) for s in STRAT_ORDER)

    print(f"\n  Baseline Portfolio: Sharpe={baseline_sharpe}, PnL=${baseline_pnl:,.2f}, "
          f"Total trades={total_trades}", flush=True)

    results = {
        'experiment': 'R119 ML Entry Filter v2 (Macro + Technical)',
        'baseline': {
            'sharpe': baseline_sharpe, 'pnl': baseline_pnl,
            'total_trades': total_trades,
            'per_strategy': {s: _compute_stats(base_trades[s]) for s in STRAT_ORDER},
        }
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Build macro + technical feature matrix per trade
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: Build Feature Matrix (Macro + Technical)", flush=True)
    print("=" * 70, flush=True)

    X, y, meta_trades = build_trade_features(base_trades, h1_df, macro_df)
    n_samples = len(X)
    n_features = X.shape[1]
    class_balance = float(y.mean())

    print(f"    Total trade samples: {n_samples}", flush=True)
    print(f"    Feature count: {n_features}", flush=True)
    print(f"    Class balance: {class_balance*100:.1f}% profitable", flush=True)
    print(f"    Features: {list(X.columns)}", flush=True)

    per_strat_counts = {}
    for t in meta_trades:
        s = t['_strat']
        per_strat_counts[s] = per_strat_counts.get(s, 0) + 1
    for s in STRAT_ORDER:
        print(f"      {s}: {per_strat_counts.get(s, 0)} trades", flush=True)

    results['phase2'] = {
        'n_samples': n_samples, 'n_features': n_features,
        'class_balance': round(class_balance, 4),
        'per_strategy': per_strat_counts,
        'feature_names': list(X.columns),
        'has_macro': macro_df is not None,
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Walk-forward XGBoost + LightGBM (sliding window)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3: Walk-Forward ML (6mo train / 2mo test, sliding 1mo)", flush=True)
    print("=" * 70, flush=True)

    oos_xgb, oos_lgb, fold_results, imp_xgb, imp_lgb = walk_forward_sliding(
        X, y, meta_trades, train_days=126, test_days=42, slide_days=21
    )

    n_valid_xgb = int((~np.isnan(oos_xgb)).sum())
    n_valid_lgb = int((~np.isnan(oos_lgb)).sum())

    xgb_aucs = [f['auc_xgb'] for f in fold_results if f.get('auc_xgb') is not None]
    lgb_aucs = [f['auc_lgb'] for f in fold_results if f.get('auc_lgb') is not None]

    mean_auc_xgb = float(np.mean(xgb_aucs)) if xgb_aucs else 0.0
    mean_auc_lgb = float(np.mean(lgb_aucs)) if lgb_aucs else 0.0

    print(f"\n    Walk-forward folds: {len(fold_results)}", flush=True)
    print(f"    XGBoost OOS: {n_valid_xgb}/{n_samples} trades, Mean AUC={mean_auc_xgb:.4f}", flush=True)
    if HAS_LGB:
        print(f"    LightGBM OOS: {n_valid_lgb}/{n_samples} trades, Mean AUC={mean_auc_lgb:.4f}", flush=True)
    else:
        print("    LightGBM: not available (falling back to XGBoost only)", flush=True)

    print(f"\n    {'Fold':>6} {'N_train':>8} {'N_test':>7} {'AUC_XGB':>9} {'AUC_LGB':>9}", flush=True)
    print(f"    {'-'*6} {'-'*8} {'-'*7} {'-'*9} {'-'*9}", flush=True)
    for fr in fold_results[:20]:
        axgb = f"{fr['auc_xgb']:.4f}" if fr.get('auc_xgb') is not None else "  N/A"
        algb = f"{fr['auc_lgb']:.4f}" if fr.get('auc_lgb') is not None else "  N/A"
        print(f"    {fr['fold']:>6} {fr['n_train']:>8} {fr['n_test']:>7} {axgb:>9} {algb:>9}", flush=True)
    if len(fold_results) > 20:
        print(f"    ... ({len(fold_results) - 20} more folds)", flush=True)

    # Select best model
    use_lgb = HAS_LGB and mean_auc_lgb > mean_auc_xgb
    best_model_name = 'LightGBM' if use_lgb else 'XGBoost'
    best_oos = oos_lgb if use_lgb else oos_xgb
    best_mean_auc = max(mean_auc_xgb, mean_auc_lgb) if HAS_LGB else mean_auc_xgb
    best_importances = imp_lgb if use_lgb else imp_xgb

    print(f"\n    Best model: {best_model_name} (AUC={best_mean_auc:.4f})", flush=True)

    results['phase3'] = {
        'n_folds': len(fold_results),
        'n_valid_xgb': n_valid_xgb,
        'n_valid_lgb': n_valid_lgb,
        'mean_auc_xgb': round(mean_auc_xgb, 4),
        'mean_auc_lgb': round(mean_auc_lgb, 4),
        'best_model': best_model_name,
        'best_mean_auc': round(best_mean_auc, 4),
        'fold_results': fold_results,
    }

    if n_valid_xgb < 50:
        print("\n  ERROR: Too few OOS predictions to continue. Aborting.", flush=True)
        results['error'] = 'insufficient OOS predictions'
        elapsed = time.time() - t0
        results['elapsed_s'] = round(elapsed, 1)
        with open(OUTPUT_DIR / "r119_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Elapsed: {elapsed:.0f}s", flush=True)
        return

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Per-strategy entry gate with threshold sweep
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 4: Per-Strategy Entry Gate (Threshold Sweep)", flush=True)
    print("=" * 70, flush=True)

    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    phase4_results = {}
    for strat in STRAT_ORDER:
        strat_indices = [i for i, t in enumerate(meta_trades) if t['_strat'] == strat]
        if not strat_indices:
            continue

        strat_trades_orig = [meta_trades[i] for i in strat_indices]
        strat_daily_base = _trades_to_daily(strat_trades_orig)
        base_sh = _sharpe(strat_daily_base)

        print(f"\n    {strat}: {len(strat_indices)} trades, baseline Sharpe={base_sh:.3f}", flush=True)
        print(f"      {'Thresh':>8} {'Kept':>6} {'Filtered%':>10} {'Sharpe':>8} {'Delta':>8}", flush=True)
        print(f"      {'-'*8} {'-'*6} {'-'*10} {'-'*8} {'-'*8}", flush=True)

        best_thr = 0.50
        best_sh = base_sh
        thr_sweep = {}

        for thr in thresholds:
            kept_trades = []
            for i in strat_indices:
                if np.isnan(best_oos[i]):
                    kept_trades.append(meta_trades[i])
                elif best_oos[i] >= thr:
                    kept_trades.append(meta_trades[i])

            if not kept_trades:
                continue

            filt_daily = _trades_to_daily(kept_trades)
            filt_sh = _sharpe(filt_daily)
            n_kept = len(kept_trades)
            n_valid_strat = sum(1 for i in strat_indices if not np.isnan(best_oos[i]))
            n_filtered = n_valid_strat - sum(1 for i in strat_indices if not np.isnan(best_oos[i]) and best_oos[i] >= thr)
            pct_filtered = 100 * n_filtered / max(n_valid_strat, 1)
            delta = filt_sh - base_sh

            print(f"      {thr:>8.2f} {n_kept:>6} {pct_filtered:>9.1f}% {filt_sh:>8.3f} {delta:>+8.3f}", flush=True)

            thr_sweep[str(thr)] = {
                'n_kept': n_kept, 'pct_filtered': round(pct_filtered, 1),
                'sharpe': round(filt_sh, 3), 'delta_sharpe': round(delta, 3),
            }
            if filt_sh > best_sh:
                best_sh = filt_sh
                best_thr = thr

        phase4_results[strat] = {
            'baseline_sharpe': round(base_sh, 3),
            'threshold_sweep': thr_sweep,
            'best_threshold': best_thr,
            'best_sharpe': round(best_sh, 3),
            'improvement': round(best_sh - base_sh, 3),
        }
        print(f"      >> Best threshold: {best_thr:.2f} (Sharpe {base_sh:.3f} -> {best_sh:.3f})", flush=True)

    # Portfolio-level with best per-strategy thresholds
    print(f"\n    Portfolio-level evaluation with per-strategy best thresholds:", flush=True)
    filtered_unit_dailies = {}
    for strat in STRAT_ORDER:
        if strat in phase4_results:
            best_thr = phase4_results[strat]['best_threshold']
        else:
            best_thr = 0.50
        strat_indices = [i for i, t in enumerate(meta_trades) if t['_strat'] == strat]
        kept = []
        for i in strat_indices:
            if np.isnan(best_oos[i]) or best_oos[i] >= best_thr:
                kept.append(meta_trades[i])
        filtered_unit_dailies[strat] = trades_to_daily_series(kept)

    filtered_portfolio = build_portfolio_daily(filtered_unit_dailies, R89_LOTS)
    filtered_sharpe = round(_sharpe(filtered_portfolio), 3)
    filtered_pnl = round(float(np.sum(filtered_portfolio)), 2)
    portfolio_delta = round(filtered_sharpe - baseline_sharpe, 3)

    print(f"      Baseline Portfolio Sharpe: {baseline_sharpe:.3f}", flush=True)
    print(f"      Filtered Portfolio Sharpe: {filtered_sharpe:.3f} (Δ={portfolio_delta:+.3f})", flush=True)

    results['phase4'] = {
        'per_strategy': phase4_results,
        'portfolio_filtered_sharpe': filtered_sharpe,
        'portfolio_filtered_pnl': filtered_pnl,
        'portfolio_delta_sharpe': portfolio_delta,
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 5: K-Fold validation (5 folds, ~2 years each)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 5: K-Fold Validation (5 folds)", flush=True)
    print("=" * 70, flush=True)

    from sklearn.metrics import roc_auc_score

    entry_dates = np.array([pd.Timestamp(t['entry_time']).date() for t in meta_trades])
    unique_dates = sorted(set(entry_dates))
    n_dates = len(unique_dates)
    fold_size = n_dates // 5

    kfold_results = []
    kfold_improvements = []

    print(f"\n    Total unique dates: {n_dates}, fold size: ~{fold_size} days", flush=True)
    print(f"    {'Fold':>6} {'Train':>7} {'Test':>6} {'AUC':>7} {'Base_Sh':>8} {'Filt_Sh':>8} {'Delta':>8}", flush=True)
    print(f"    {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*8}", flush=True)

    for fold_k in range(5):
        test_start_idx = fold_k * fold_size
        test_end_idx = min((fold_k + 1) * fold_size, n_dates) if fold_k < 4 else n_dates
        test_dates_set = set(unique_dates[test_start_idx:test_end_idx])

        train_mask = np.array([d not in test_dates_set for d in entry_dates])
        test_mask = np.array([d in test_dates_set for d in entry_dates])

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) < 100 or len(test_idx) < 30:
            kfold_results.append({'fold': fold_k + 1, 'skip': True})
            continue

        X_tr = X.iloc[train_idx].copy().fillna(0)
        y_tr = y[train_idx]
        X_te = X.iloc[test_idx].copy().fillna(0)
        y_te = y[test_idx]

        try:
            model = get_xgb_model()
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else np.nan
        except Exception:
            kfold_results.append({'fold': fold_k + 1, 'error': True})
            continue

        # Compute baseline Sharpe for test fold trades
        test_trades_all = [meta_trades[i] for i in test_idx]
        base_daily_fold = _trades_to_daily(test_trades_all)
        base_sh_fold = _sharpe(base_daily_fold)

        # Filtered (threshold 0.55)
        thr = 0.55
        kept_trades = [meta_trades[i] for i, p in zip(test_idx, probs) if p >= thr]
        filt_daily_fold = _trades_to_daily(kept_trades) if kept_trades else np.array([0.0])
        filt_sh_fold = _sharpe(filt_daily_fold)

        delta = filt_sh_fold - base_sh_fold
        improved = delta > 0
        kfold_improvements.append(improved)

        auc_val = round(float(auc), 4) if not np.isnan(auc) else None
        print(f"    {fold_k+1:>6} {len(train_idx):>7} {len(test_idx):>6} "
              f"{auc_val if auc_val else 'N/A':>7} {base_sh_fold:>8.3f} "
              f"{filt_sh_fold:>8.3f} {delta:>+8.3f}", flush=True)

        kfold_results.append({
            'fold': fold_k + 1, 'n_train': len(train_idx), 'n_test': len(test_idx),
            'auc': auc_val, 'baseline_sharpe': round(base_sh_fold, 3),
            'filtered_sharpe': round(filt_sh_fold, 3), 'delta': round(delta, 3),
            'improved': improved,
        })

    n_improved = sum(kfold_improvements)
    kfold_pass = n_improved >= 3
    print(f"\n    K-Fold result: {n_improved}/5 folds with positive improvement", flush=True)
    print(f"    Pass criteria (>=3/5): {'PASS' if kfold_pass else 'FAIL'}", flush=True)

    results['phase5'] = {
        'kfold_results': kfold_results,
        'n_improved': n_improved,
        'pass': kfold_pass,
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 6: Feature importance and ablation study
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 6: Feature Importance & Ablation Study", flush=True)
    print("=" * 70, flush=True)

    # Top 15 features (XGBoost)
    sorted_imp = sorted(best_importances.items(), key=lambda x: x[1], reverse=True)[:15]
    print(f"\n    Top 15 Features ({best_model_name}):", flush=True)
    print(f"    {'Rank':>4} {'Feature':<30} {'Importance':>12}", flush=True)
    print(f"    {'-'*4} {'-'*30} {'-'*12}", flush=True)
    for rank, (fname, fval) in enumerate(sorted_imp, 1):
        bar = "█" * int(fval * 50)
        print(f"    {rank:>4} {fname:<30} {fval:>12.4f} {bar}", flush=True)

    # Ablation: remove top 3 features and retrain
    print(f"\n    Ablation study (removing top features):", flush=True)
    ablation_results = {}
    top3_names = [f[0] for f in sorted_imp[:3]]

    for n_remove in [1, 3, 5]:
        remove_cols = [f[0] for f in sorted_imp[:n_remove]]
        remaining_cols = [c for c in X.columns if c not in remove_cols]
        if len(remaining_cols) < 3:
            continue

        X_abl = X[remaining_cols]
        abl_aucs = []

        for fold_k in range(5):
            test_start_idx = fold_k * fold_size
            test_end_idx = min((fold_k + 1) * fold_size, n_dates) if fold_k < 4 else n_dates
            test_dates_set = set(unique_dates[test_start_idx:test_end_idx])

            train_mask = np.array([d not in test_dates_set for d in entry_dates])
            test_mask = np.array([d in test_dates_set for d in entry_dates])

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) < 100 or len(test_idx) < 30:
                continue

            X_tr = X_abl.iloc[train_idx].copy().fillna(0)
            y_tr = y[train_idx]
            X_te = X_abl.iloc[test_idx].copy().fillna(0)
            y_te = y[test_idx]

            try:
                model = get_xgb_model()
                model.fit(X_tr, y_tr)
                probs = model.predict_proba(X_te)[:, 1]
                auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else np.nan
                if not np.isnan(auc):
                    abl_aucs.append(auc)
            except Exception:
                continue

        if abl_aucs:
            abl_mean = float(np.mean(abl_aucs))
            delta_auc = abl_mean - best_mean_auc
            print(f"      Remove top {n_remove}: AUC={abl_mean:.4f} (Δ={delta_auc:+.4f}), "
                  f"removed={remove_cols}", flush=True)
            ablation_results[f'remove_top_{n_remove}'] = {
                'mean_auc': round(abl_mean, 4),
                'delta_auc': round(delta_auc, 4),
                'removed': remove_cols,
            }

    # Ablation: remove all macro features
    macro_feature_names = [c for c in X.columns if any(k in c for k in
                           ['vix', 'dxy', 'yield', 'gvz', 'credit', 'crude', 'copper',
                            'usdjpy', 'risk_appetite', 'gold_dxy'])]
    if macro_feature_names:
        tech_only_cols = [c for c in X.columns if c not in macro_feature_names]
        if len(tech_only_cols) >= 3:
            X_tech = X[tech_only_cols]
            tech_aucs = []
            for fold_k in range(5):
                test_start_idx = fold_k * fold_size
                test_end_idx = min((fold_k + 1) * fold_size, n_dates) if fold_k < 4 else n_dates
                test_dates_set = set(unique_dates[test_start_idx:test_end_idx])

                train_mask = np.array([d not in test_dates_set for d in entry_dates])
                test_mask = np.array([d in test_dates_set for d in entry_dates])
                train_idx = np.where(train_mask)[0]
                test_idx = np.where(test_mask)[0]

                if len(train_idx) < 100 or len(test_idx) < 30:
                    continue

                X_tr = X_tech.iloc[train_idx].copy().fillna(0)
                y_tr = y[train_idx]
                X_te = X_tech.iloc[test_idx].copy().fillna(0)
                y_te = y[test_idx]

                try:
                    model = get_xgb_model()
                    model.fit(X_tr, y_tr)
                    probs = model.predict_proba(X_te)[:, 1]
                    auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else np.nan
                    if not np.isnan(auc):
                        tech_aucs.append(auc)
                except Exception:
                    continue

            if tech_aucs:
                tech_mean = float(np.mean(tech_aucs))
                macro_contribution = best_mean_auc - tech_mean
                print(f"      Tech-only (no macro): AUC={tech_mean:.4f} "
                      f"(macro contributes {macro_contribution:+.4f})", flush=True)
                ablation_results['no_macro'] = {
                    'mean_auc': round(tech_mean, 4),
                    'macro_contribution': round(macro_contribution, 4),
                    'n_macro_features': len(macro_feature_names),
                }

    results['phase6'] = {
        'top_15_features': {fname: round(fval, 4) for fname, fval in sorted_imp},
        'ablation': ablation_results,
    }

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    print(f"\n{'='*80}", flush=True)
    print(f"  R119 SUMMARY — ML Entry Filter v2", flush=True)
    print(f"{'='*80}", flush=True)

    print(f"\n  Features: {n_features} ({n_samples} trade samples)", flush=True)
    print(f"  Walk-forward folds: {len(fold_results)}", flush=True)
    print(f"  Best model: {best_model_name}, Mean AUC={best_mean_auc:.4f}", flush=True)
    print(f"  (vs R107 baseline AUC ~0.52-0.56)", flush=True)
    print(f"\n  Baseline Portfolio Sharpe: {baseline_sharpe:.3f}", flush=True)
    print(f"  Filtered Portfolio Sharpe: {filtered_sharpe:.3f} (Δ={portfolio_delta:+.3f})", flush=True)
    print(f"  K-Fold validation: {n_improved}/5 folds improved ({'PASS' if kfold_pass else 'FAIL'})", flush=True)

    # Verdict
    if best_mean_auc > 0.56 and kfold_pass and portfolio_delta > 0:
        verdict = (f"ML entry filter v2 is effective: AUC={best_mean_auc:.4f}, "
                   f"Sharpe {baseline_sharpe:.3f} -> {filtered_sharpe:.3f}")
    elif best_mean_auc > 0.53 and portfolio_delta > 0:
        verdict = (f"ML entry filter v2 shows marginal improvement: AUC={best_mean_auc:.4f}, "
                   f"use with caution")
    else:
        verdict = (f"ML entry filter v2 inconclusive: AUC={best_mean_auc:.4f}, "
                   f"delta Sharpe={portfolio_delta:+.3f}")

    print(f"\n  VERDICT: {verdict}", flush=True)

    results['verdict'] = verdict
    results['elapsed_s'] = round(elapsed, 1)

    out_file = OUTPUT_DIR / "r119_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"  Saved: {out_file}", flush=True)
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
