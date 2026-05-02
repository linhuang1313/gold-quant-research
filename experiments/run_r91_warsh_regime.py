#!/usr/bin/env python3
"""
R91 — Warsh Regime Analysis & Politicization Risk Score
==========================================================
Based on Kevin Warsh's policy framework, test gold strategy performance
under three macro scenarios:

  Regime A: "Successful Soft Landing" — orderly rate cuts + firm QT + strong USD
  Regime B: "Hard Landing" — QT too aggressive + recession risk + credit stress
  Regime C: "Politically Compromised" — forced rate cuts + weak discipline + USD credit damage

Additionally constructs a "Politicization Risk Score" (0-100) from:
  P: Public political pressure proxy
  D: Policy-rule deviation (Taylor Rule gap)
  M: Market perception of independence loss

Phases:
  1. Build Warsh 3-Scenario Regime classifier from aligned_daily
  2. Compute Taylor Rule deviation (simple + market-based)
  3. Build Politicization Risk Score time series
  4. Backtest all 4 strategies under each Warsh Regime
  5. Regime-conditional portfolio optimization
  6. Analyze Score-Gold return relationship
"""
import sys, os, io, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r91_warsh_regime")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01
CAPITAL = 5000
MAX_DD_LIMIT = 1000
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']
CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.08, 'SESS_BO': 0.08}

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]


# ═══════════════════════════════════════════════════════════════
# Shared backtest functions (from R88/R89)
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs()
    }).max(axis=1)
    return tr.rolling(period).mean()

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

def _mk(pos, exit_price, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_price,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}

def _run_exit_with_cap(pos, i, h, lo_v, c, spread, lot, pv, times,
                       sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold, maxloss_cap=0):
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
    if pnl_h >= tp_val: return _mk(pos, c, times[i], "TP", i, tp_val)
    if pnl_l <= -sl_val: return _mk(pos, c, times[i], "SL", i, -sl_val)
    if maxloss_cap > 0 and pnl_c < -maxloss_cap:
        return _mk(pos, c, times[i], "MaxLossCap", i, -maxloss_cap)
    ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
    if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
        ts_p = h - td
        if lo_v <= ts_p: return _mk(pos, c, times[i], "Trail", i, (ts_p - pos['entry'] - spread) * lot * pv)
    elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
        ts_p = lo_v + td
        if h >= ts_p: return _mk(pos, c, times[i], "Trail", i, (pos['entry'] - ts_p - spread) * lot * pv)
    if held >= max_hold: return _mk(pos, c, times[i], "Timeout", i, pnl_c)
    return None

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
            'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0,
        })
    return trades


# ═══════════════════════════════════════════════════════════════
# Stats helpers
# ═══════════════════════════════════════════════════════════════

def trades_to_daily_series(trades):
    if not trades:
        return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).tz_localize(None).normalize() if hasattr(pd.Timestamp(t['exit_time']), 'tz') else pd.Timestamp(t['exit_time']).normalize()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))

def sharpe(daily_pnl):
    if len(daily_pnl) < 10: return 0.0
    arr = np.array(daily_pnl) if not isinstance(daily_pnl, np.ndarray) else daily_pnl
    if arr.std() == 0: return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(252))

def max_dd(daily_pnl):
    arr = np.array(daily_pnl) if not isinstance(daily_pnl, np.ndarray) else daily_pnl
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(dd.max()) if len(dd) > 0 else 0.0

def cvar99(daily_pnl):
    arr = np.array(daily_pnl)
    if len(arr) < 10: return 0.0
    threshold = np.percentile(arr, 1)
    tail = arr[arr <= threshold]
    return float(tail.mean()) if len(tail) > 0 else float(threshold)


# ═══════════════════════════════════════════════════════════════
# Phase 1: Warsh 3-Scenario Regime Classifier
# ═══════════════════════════════════════════════════════════════

def build_warsh_regimes(ext_df):
    """
    Classify each day into one of three Warsh scenarios:
    A: Successful Soft Landing (discipline holds, economy OK)
    B: Hard Landing Risk (over-tightening, recession signals)
    C: Politically Compromised (forced easing, USD credit concern)
    
    Uses a scoring system based on observable market variables.
    """
    print("\n  Building Warsh 3-Scenario Regime classifier...", flush=True)
    
    df = ext_df.copy()
    
    # Compute additional derived signals
    # Real yield direction (rising = hawkish discipline)
    df['ry_rising'] = df['REAL_YIELD_DFII10'].diff(20) > 0
    # DXY strength
    df['dxy_strong'] = df['DXY_Mom20'] > 0
    # VIX calm
    df['vix_calm'] = df['VIX_Zscore'] < 0.5
    # Credit tight (low stress = good)
    df['credit_tight'] = df['CREDIT_STRESS'] < df['CREDIT_STRESS'].rolling(252).median()
    # Yield curve: inverted = recession risk
    df['curve_inverted'] = df['YIELD_CURVE_10Y2Y'] < 0
    # Long rate rising despite rate cuts (policy credibility loss)
    df['long_rate_rising'] = df['US10Y_Change20'] > 0
    # Fed funds declining (rate cut mode)
    df['fed_cutting'] = df['FED_FUNDS_DFF'].diff(60) < -0.25
    # Inflation expectations up (from breakeven proxy: crude + copper + M2)
    df['inflation_up'] = (df['CRUDE_Mom20'] > 0.05) | (df['M2_YoY'] > df['M2_YoY'].rolling(252).quantile(0.7))
    # Risk appetite high
    df['risk_on'] = df['RISK_APPETITE_Z'] > 0.5
    # Credit stress rising
    df['credit_widening'] = df['CREDIT_STRESS'] > df['CREDIT_STRESS'].rolling(60).mean()
    
    # Score for each regime (higher = more likely that regime)
    score_a = pd.Series(0.0, index=df.index)  # Soft Landing
    score_b = pd.Series(0.0, index=df.index)  # Hard Landing
    score_c = pd.Series(0.0, index=df.index)  # Politically Compromised
    
    # --- Regime A: Successful Soft Landing ---
    # Real yield stable/rising + DXY strong + VIX calm + credit tight + no inversion
    score_a += df['ry_rising'].astype(float) * 2
    score_a += df['dxy_strong'].astype(float) * 2
    score_a += df['vix_calm'].astype(float) * 1.5
    score_a += df['credit_tight'].astype(float) * 1.5
    score_a += (~df['curve_inverted']).astype(float) * 1
    score_a += df['risk_on'].astype(float) * 1
    
    # --- Regime B: Hard Landing ---
    # Real yield rising sharply + curve inverted + credit widening + VIX elevated
    score_b += (df['REAL_YIELD_Change20'] > 0.3).astype(float) * 2
    score_b += df['curve_inverted'].astype(float) * 2.5
    score_b += df['credit_widening'].astype(float) * 2
    score_b += (df['VIX_Zscore'] > 1.0).astype(float) * 2
    score_b += (~df['risk_on']).astype(float) * 1
    score_b += (df['SPX_Mom5'] < -0.03).astype(float) * 1.5
    
    # --- Regime C: Politically Compromised ---
    # Fed cutting + long rate NOT falling (credibility loss) + DXY weak + inflation up
    score_c += df['fed_cutting'].astype(float) * 2
    score_c += (df['fed_cutting'] & df['long_rate_rising']).astype(float) * 3  # Key signal
    score_c += (~df['dxy_strong']).astype(float) * 1.5
    score_c += df['inflation_up'].astype(float) * 2
    score_c += (~df['ry_rising']).astype(float) * 1
    score_c += (df['USDCNH_Mom20'] > 0.01).astype(float) * 1  # CNH weakening = USD anxiety
    
    # Assign regime by highest score
    scores = pd.DataFrame({'A': score_a, 'B': score_b, 'C': score_c}, index=df.index)
    regime = scores.idxmax(axis=1)
    
    # Handle NaN (early period)
    regime = regime.fillna('A')
    
    # Smooth: require at least 5 consecutive days to switch regime
    smoothed = regime.copy()
    for i in range(5, len(smoothed)):
        window = regime.iloc[i-4:i+1]
        if window.nunique() == 1:
            smoothed.iloc[i] = window.iloc[0]
        else:
            smoothed.iloc[i] = smoothed.iloc[i-1]
    
    # Stats
    counts = smoothed.value_counts()
    print(f"    Regime distribution:", flush=True)
    for r in ['A', 'B', 'C']:
        n = counts.get(r, 0)
        pct = n / len(smoothed) * 100
        print(f"      {r}: {n} days ({pct:.1f}%)", flush=True)
    
    return smoothed, scores


# ═══════════════════════════════════════════════════════════════
# Phase 2: Taylor Rule Deviation
# ═══════════════════════════════════════════════════════════════

def compute_taylor_rule_deviation(ext_df):
    """
    Compute two versions of Taylor Rule deviation:
    1. Simple Taylor Rule: i* = r* + pi + 0.5*(pi-2%) + 0.5*output_gap
       Proxy: r*=1%, pi from crude/M2 proxy, output_gap from SPX deviation
    2. Market-based: 2Y yield vs Fed Funds gap (market's implied deviation)
    """
    print("\n  Computing Taylor Rule deviations...", flush=True)
    df = ext_df.copy()
    
    # --- Method 1: Simple Taylor Rule ---
    # Proxy for inflation: use crude momentum + M2 growth as inflation expectation proxy
    # (We don't have CPI directly, so use market-implied proxies)
    r_star = 1.0  # neutral real rate assumption
    pi_target = 2.0  # Fed target
    
    # Inflation proxy: normalize crude mom + M2 growth to ~2-5% range
    crude_inflation = df['CRUDE_Mom20'].clip(-0.3, 0.3) * 10  # scale to ~0-3%
    m2_inflation = df['M2_YoY'].fillna(0) * 100  # already in pct
    pi_proxy = (crude_inflation.fillna(0) * 0.3 + m2_inflation.fillna(0) * 0.3 + 2.0).clip(0, 8)
    
    # Output gap proxy: SPX deviation from 200d SMA (positive = above trend)
    spx_gap = (df['SPX_Close'] / df['SPX_Close'].rolling(200).mean() - 1).fillna(0) * 100
    output_gap = spx_gap.clip(-10, 10)
    
    # Taylor Rule rate
    taylor_simple = r_star + pi_proxy + 0.5 * (pi_proxy - pi_target) + 0.5 * output_gap
    
    # Deviation: actual Fed Funds - Taylor Rule (negative = "too loose")
    deviation_simple = df['FED_FUNDS_DFF'] - taylor_simple
    
    # --- Method 2: Market-based ---
    # US2Y yield reflects market's expectation of where rates should be
    # Gap between US2Y and Fed Funds: if US2Y > FF, market says rates should be higher
    deviation_market = df['US2Y_Close'] - df['FED_FUNDS_DFF']
    
    # Normalize both to 0-100 scale (for the politicization score)
    # More negative = more "politically compressed" = higher risk
    dev_simple_norm = (-deviation_simple).clip(0, None)
    dev_simple_score = (dev_simple_norm / dev_simple_norm.quantile(0.95).clip(0.1, None) * 100).clip(0, 100)
    
    dev_market_norm = (-deviation_market).clip(0, None)
    dev_market_score = (dev_market_norm / dev_market_norm.quantile(0.95).clip(0.1, None) * 100).clip(0, 100)
    
    print(f"    Simple Taylor deviation: mean={deviation_simple.mean():.2f}, "
          f"min={deviation_simple.min():.2f}, max={deviation_simple.max():.2f}", flush=True)
    print(f"    Market-based deviation: mean={deviation_market.mean():.2f}, "
          f"min={deviation_market.min():.2f}, max={deviation_market.max():.2f}", flush=True)
    
    return {
        'taylor_simple': taylor_simple,
        'deviation_simple': deviation_simple,
        'deviation_market': deviation_market,
        'D_score_simple': dev_simple_score,
        'D_score_market': dev_market_score,
    }


# ═══════════════════════════════════════════════════════════════
# Phase 3: Politicization Risk Score
# ═══════════════════════════════════════════════════════════════

def build_politicization_score(ext_df, taylor_data):
    """
    Build composite Politicization Risk Score (0-100) from three dimensions:
    
    P (Public Pressure): Proxy from DXY reaction to rate decisions
       - We don't have NLP data, so use: abrupt DXY moves after Fed dates as proxy
       - Also: VIX spikes around FOMC = "uncertainty about independence"
       
    D (Policy-Rule Deviation): Taylor Rule gap (from Phase 2)
    
    M (Market Perception): 
       - M1: Breakeven inflation high while rates low (inflation_expectation - rates gap)
       - M2: Term premium rising (10Y-2Y spread behavior when cutting)
       - M3: Gold-DXY decorrelation (both rising = systemic distrust)
    """
    print("\n  Building Politicization Risk Score...", flush=True)
    df = ext_df.copy()
    
    # --- Dimension P: Political Pressure Proxy (0-100) ---
    # Use VIX elevation around Fed actions + sudden policy moves as proxy
    # Higher GVZ (gold vol) relative to VIX = market pricing "policy uncertainty"
    gvz_vix_ratio = (df['GVZ_Close'] / df['VIX_Close'].clip(1, None)).fillna(1)
    p_score = (gvz_vix_ratio / gvz_vix_ratio.quantile(0.9) * 50).clip(0, 100)
    
    # Add: rapid Fed Funds changes (proxy for political pressure to act)
    ff_volatility = df['FED_FUNDS_DFF'].diff().abs().rolling(60).mean()
    ff_vol_score = (ff_volatility / ff_volatility.quantile(0.9) * 50).clip(0, 100)
    p_score = (p_score * 0.6 + ff_vol_score * 0.4).fillna(0)
    
    # --- Dimension D: Policy-Rule Deviation (0-100) ---
    # Average of simple and market-based Taylor deviations
    d_score = (taylor_data['D_score_simple'].fillna(0) * 0.5 + 
               taylor_data['D_score_market'].fillna(0) * 0.5)
    
    # --- Dimension M: Market Perception (0-100) ---
    # M1: Inflation expectations vs rate path (breakeven proxy)
    # High crude + weak DXY + low rates = market sees inflation not being addressed
    inflation_gap = (df['CRUDE_Mom20'].fillna(0) * 30 - 
                     df['FED_FUNDS_DFF'].diff(20).fillna(0) * 10)
    m1 = (inflation_gap / inflation_gap.quantile(0.9).clip(0.1, None) * 100).clip(0, 100).fillna(0)
    
    # M2: Term premium signal (10Y rising while Fed cutting = credibility question)
    cutting_but_long_rising = (df['FED_FUNDS_DFF'].diff(60) < -0.25) & (df['US10Y_Change20'] > 0.1)
    m2 = cutting_but_long_rising.astype(float).rolling(60).mean() * 100
    m2 = m2.clip(0, 100).fillna(0)
    
    # M3: Gold-DXY decorrelation (normally negative; when positive = systemic distrust)
    gold_ret = df['GLD_Close'].pct_change(5)
    dxy_ret = df['DXY_Close'].pct_change(5)
    roll_corr = gold_ret.rolling(60).corr(dxy_ret)
    m3 = ((roll_corr + 0.5) / 1.0 * 100).clip(0, 100).fillna(50)  # normally around 25, high = both up
    
    m_score = (m1 * 0.4 + m2 * 0.3 + m3 * 0.3).fillna(0)
    
    # --- Composite Score ---
    # S = 0.3*P + 0.4*D + 0.3*M
    composite = 0.3 * p_score + 0.4 * d_score + 0.3 * m_score
    composite = composite.clip(0, 100)
    
    print(f"    P (pressure): mean={p_score.mean():.1f}, p75={p_score.quantile(0.75):.1f}", flush=True)
    print(f"    D (deviation): mean={d_score.mean():.1f}, p75={d_score.quantile(0.75):.1f}", flush=True)
    print(f"    M (market perception): mean={m_score.mean():.1f}, p75={m_score.quantile(0.75):.1f}", flush=True)
    print(f"    COMPOSITE: mean={composite.mean():.1f}, p75={composite.quantile(0.75):.1f}, "
          f"max={composite.max():.1f}", flush=True)
    
    return {
        'P_score': p_score,
        'D_score': d_score,
        'M_score': m_score,
        'composite': composite,
    }


# ═══════════════════════════════════════════════════════════════
# Phase 4-6: Strategy backtests + analysis
# ═══════════════════════════════════════════════════════════════

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
    return portfolio, idx


def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("  R91: Warsh Regime Analysis & Politicization Risk Score", flush=True)
    print("=" * 80, flush=True)

    # ── Load Data ──
    print("\n  Loading data...", flush=True)
    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH, DataBundle

    ext_df = pd.read_csv("data/external/aligned_daily.csv", parse_dates=['Date'], index_col='Date')
    ext_df = ext_df.sort_index()
    if ext_df.index.tz is not None:
        ext_df.index = ext_df.index.tz_localize(None)
    print(f"    External: {len(ext_df)} days ({ext_df.index[0].date()} ~ {ext_df.index[-1].date()})", flush=True)

    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"    H1: {len(h1_df)} bars", flush=True)

    l8_bundle = DataBundle.load_custom()
    print(f"    L8 bundle ready.", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Build Warsh Regime
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print("  Phase 1: Warsh 3-Scenario Regime Classification", flush=True)
    print(f"{'='*80}", flush=True)

    warsh_regime, regime_scores = build_warsh_regimes(ext_df)

    # ══════════════════════════════════════════════════════════════
    # Phase 2: Taylor Rule Deviations
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print("  Phase 2: Taylor Rule Deviation (Simple + Market-Based)", flush=True)
    print(f"{'='*80}", flush=True)

    taylor_data = compute_taylor_rule_deviation(ext_df)

    # Correlation with gold
    gold_ret_20d = ext_df['GLD_Close'].pct_change(20).shift(-20)  # forward 20d return
    for key in ['deviation_simple', 'deviation_market']:
        corr = taylor_data[key].corr(gold_ret_20d)
        print(f"    {key} vs gold 20d fwd return: r={corr:.4f}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Phase 3: Politicization Risk Score
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print("  Phase 3: Politicization Risk Score", flush=True)
    print(f"{'='*80}", flush=True)

    pol_score = build_politicization_score(ext_df, taylor_data)

    # Score vs gold forward return relationship
    composite = pol_score['composite']
    for horizon in [5, 20, 60]:
        gold_fwd = ext_df['GLD_Close'].pct_change(horizon).shift(-horizon)
        corr = composite.corr(gold_fwd)
        print(f"    Score vs gold {horizon}d fwd return: r={corr:.4f}", flush=True)

    # Quintile analysis
    print(f"\n    Quintile analysis (Score → Gold 20d fwd return):", flush=True)
    valid = pd.DataFrame({'score': composite, 'fwd': gold_ret_20d}).dropna()
    if len(valid) > 100:
        valid['q'] = pd.qcut(valid['score'], 5, labels=[1,2,3,4,5], duplicates='drop')
        q_stats = valid.groupby('q')['fwd'].agg(['mean', 'std', 'count'])
        for q_label, row in q_stats.iterrows():
            print(f"      Q{q_label}: mean_ret={row['mean']*100:.2f}%, std={row['std']*100:.2f}%, n={int(row['count'])}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Phase 4: Strategy backtests at unit lot
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print("  Phase 4: Strategy Backtests at Unit Lot", flush=True)
    print(f"{'='*80}", flush=True)

    unit_dailies = {}
    h1_strats = {
        'PSAR': (bt_psar, {}),
        'TSMOM': (bt_tsmom, {}),
        'SESS_BO': (bt_sess_bo, {}),
    }
    for name, (fn, kw) in h1_strats.items():
        trades = fn(h1_df, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=CAPS[name], **kw)
        unit_dailies[name] = trades_to_daily_series(trades)
        print(f"    {name}: {len(trades)} trades", flush=True)

    trades_l8 = bt_l8_max(l8_bundle, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])
    unit_dailies['L8_MAX'] = trades_to_daily_series(trades_l8)
    print(f"    L8_MAX: {len(trades_l8)} trades", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Phase 5: Per-Regime Performance Analysis
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print("  Phase 5: Per-Regime Strategy Performance", flush=True)
    print(f"{'='*80}", flush=True)

    # Map regime to dates
    regime_map = warsh_regime.to_dict()

    print(f"\n  {'Regime':<8} {'Strategy':<10} {'Days':>6} {'Sharpe':>8} {'MeanPnL':>10} "
          f"{'MaxDD':>8} {'WinRate':>8}", flush=True)
    print(f"  {'-'*8} {'-'*10} {'-'*6} {'-'*8} {'-'*10} {'-'*8} {'-'*8}", flush=True)

    regime_perf = {}
    for regime in ['A', 'B', 'C']:
        regime_dates = set()
        for dt, r in regime_map.items():
            if r == regime:
                regime_dates.add(pd.Timestamp(dt).normalize())

        regime_perf[regime] = {}
        for name in STRAT_ORDER:
            ds = unit_dailies[name]
            mask = ds.index.normalize().isin(regime_dates)
            regime_pnl = ds[mask].values

            n_days = len(regime_pnl)
            if n_days < 5:
                regime_perf[regime][name] = {'n_days': 0, 'sharpe': 0, 'mean': 0, 'dd': 0, 'wr': 0}
                continue

            sh = sharpe(regime_pnl)
            mean_pnl = float(regime_pnl.mean())
            dd = max_dd(regime_pnl)
            wr = float((regime_pnl > 0).sum() / max(1, (regime_pnl != 0).sum()) * 100)

            regime_perf[regime][name] = {
                'n_days': n_days, 'sharpe': round(sh, 3),
                'mean': round(mean_pnl, 4), 'dd': round(dd, 2), 'wr': round(wr, 1)
            }
            print(f"  {regime:<8} {name:<10} {n_days:>6} {sh:>8.3f} {mean_pnl:>10.4f} "
                  f"{dd:>8.2f} {wr:>7.1f}%", flush=True)

    # Gold daily return per regime
    print(f"\n  Gold daily return by Warsh Regime:", flush=True)
    gold_daily_ret = ext_df['GLD_Close'].pct_change()
    for regime in ['A', 'B', 'C']:
        mask = warsh_regime == regime
        regime_gold = gold_daily_ret[mask].dropna()
        if len(regime_gold) > 10:
            ann_ret = regime_gold.mean() * 252 * 100
            ann_vol = regime_gold.std() * np.sqrt(252) * 100
            sh = regime_gold.mean() / regime_gold.std() * np.sqrt(252) if regime_gold.std() > 0 else 0
            print(f"    Regime {regime}: ann_return={ann_ret:.1f}%, ann_vol={ann_vol:.1f}%, "
                  f"sharpe={sh:.2f}, n={len(regime_gold)}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Phase 6: Score-Conditional Analysis
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print("  Phase 6: Politicization Score → Strategy Performance", flush=True)
    print(f"{'='*80}", flush=True)

    # Divide composite score into bands: Low (0-30), Mid (30-60), High (60-100)
    score_bands = pd.cut(composite, bins=[0, 30, 60, 100], labels=['Low', 'Mid', 'High'])

    print(f"\n  Score Band → Portfolio Sharpe:", flush=True)
    for band in ['Low', 'Mid', 'High']:
        band_dates = set(composite[score_bands == band].index)
        port_daily, port_idx = build_portfolio_daily(unit_dailies, R89_LOTS)
        band_mask = pd.DatetimeIndex(port_idx).normalize().isin(band_dates)
        band_pnl = port_daily[band_mask]
        if len(band_pnl) > 10:
            sh = sharpe(band_pnl)
            mean_d = band_pnl.mean()
            n = len(band_pnl)
            print(f"    {band:>5}: Sharpe={sh:.3f}, mean_daily=${mean_d:.2f}, n={n}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Save results
    # ══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{'='*80}", flush=True)
    print(f"  R91 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"{'='*80}", flush=True)

    results = {
        'experiment': 'R91 Warsh Regime Analysis',
        'elapsed_s': round(elapsed, 1),
        'regime_distribution': warsh_regime.value_counts().to_dict(),
        'taylor_rule': {
            'simple_deviation_mean': round(float(taylor_data['deviation_simple'].mean()), 3),
            'market_deviation_mean': round(float(taylor_data['deviation_market'].mean()), 3),
            'corr_simple_vs_gold20d': round(float(taylor_data['deviation_simple'].corr(gold_ret_20d)), 4),
            'corr_market_vs_gold20d': round(float(taylor_data['deviation_market'].corr(gold_ret_20d)), 4),
        },
        'politicization_score': {
            'mean': round(float(composite.mean()), 1),
            'p75': round(float(composite.quantile(0.75)), 1),
            'max': round(float(composite.max()), 1),
        },
        'regime_performance': regime_perf,
        'gold_by_regime': {},
    }

    for regime in ['A', 'B', 'C']:
        mask = warsh_regime == regime
        regime_gold = gold_daily_ret[mask].dropna()
        if len(regime_gold) > 10:
            results['gold_by_regime'][regime] = {
                'ann_return_pct': round(float(regime_gold.mean() * 252 * 100), 2),
                'ann_vol_pct': round(float(regime_gold.std() * np.sqrt(252) * 100), 2),
                'sharpe': round(float(regime_gold.mean() / regime_gold.std() * np.sqrt(252)), 3),
                'n_days': int(len(regime_gold)),
            }

    # Save regime labels
    regime_df = pd.DataFrame({
        'warsh_regime': warsh_regime,
        'score_A': regime_scores['A'],
        'score_B': regime_scores['B'],
        'score_C': regime_scores['C'],
        'pol_score': composite,
        'taylor_dev_simple': taylor_data['deviation_simple'],
        'taylor_dev_market': taylor_data['deviation_market'],
    })
    regime_df.to_csv(OUTPUT_DIR / "warsh_regime_labels.csv")
    print(f"  Saved: {OUTPUT_DIR}/warsh_regime_labels.csv", flush=True)

    with open(OUTPUT_DIR / "r91_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r91_results.json", flush=True)


if __name__ == "__main__":
    main()
