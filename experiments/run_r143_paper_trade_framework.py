#!/usr/bin/env python3
"""
R143 — Paper Trade Automated Verification Framework
=====================================================
Creates an automated comparison framework for paper trading results
vs backtest expectations.

  Phase 1: Define strategy variants (V1-V5) and compute backtest baselines
  Phase 2: Compute monthly expected metrics → store as JSON
  Phase 3: Build comparison framework (load paper trades, compare, t-test)
  Phase 4: Decay detection (rolling 20-trade Sharpe, alert thresholds)
  Phase 5: Auto-report generation (markdown, chart data)
  Phase 6: Demo mode — simulate 6mo paper trades, run framework, verify alerts
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats
from collections import defaultdict

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS
from backtest.runner import load_csv, load_m15, load_h1_aligned, H1_CSV_PATH
from indicators import calc_chandelier

OUTPUT_DIR = Path("results/r143_paper_trade_framework")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.08, 'SESS_BO': 0.08}
CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']

PSAR_DEFAULT = {'sl_atr': 4.5, 'tp_atr': 16.0, 'trail_act': 0.20, 'trail_dist': 0.04, 'max_hold': 20}
PSAR_OPTIMIZED = {'sl_atr': 4.0, 'tp_atr': 6.0, 'trail_act': 0.08, 'trail_dist': 0.015, 'max_hold': 15}

DECAY_ALERT_RATIO = 0.70
CONSECUTIVE_MONTHS_REVIEW = 3
DEMO_MONTHS = 6
DEMO_NOISE_PNL = 0.10
DEMO_MISS_RATE = 0.05

t0 = time.time()


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


def _run_exit(pos, i, hi, lo, cl, spread, lot, pv, times,
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
        return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


def _sharpe(arr):
    if len(arr) < 10:
        return 0.0
    s = np.std(arr, ddof=1)
    return float(np.mean(arr) / s * np.sqrt(252)) if s > 0 else 0.0


def _max_dd(arr):
    if len(arr) == 0:
        return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def _compute_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0.0, 'pnl': 0.0, 'max_dd': 0.0, 'wr': 0.0}
    pnls = [t['pnl'] for t in trades]
    daily = _trades_to_daily(trades)
    daily_arr = daily.values if len(daily) > 0 else np.array([])
    wins = sum(1 for p in pnls if p > 0)
    return {
        'n': len(trades),
        'sharpe': round(_sharpe(daily_arr), 3),
        'pnl': round(sum(pnls), 2),
        'max_dd': round(_max_dd(daily_arr), 2),
        'wr': round(wins / len(trades) * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════
# Strategy backtests
# ═══════════════════════════════════════════════════════════════

def bt_psar(h1_df, spread, lot, maxloss_cap=0, params=None):
    if params is None:
        params = PSAR_DEFAULT
    df = h1_df.copy(); add_psar(df); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR', 'PSAR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; psar = df['PSAR'].values; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               params['sl_atr'], params['tp_atr'], params['trail_act'],
                               params['trail_dist'], params['max_hold'], maxloss_cap)
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
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                pnl = (c[i]-pos['entry']-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl))
                pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                pnl = (pos['entry']-c[i]-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl))
                pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(score[i]) or np.isnan(score[i-1]): continue
        if score[i] > 0 and score[i-1] <= 0:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom_d1_filter(h1_df, spread, lot, maxloss_cap=0,
                       fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
                       trail_act=0.14, trail_dist=0.025, max_hold=20):
    """TSMOM with D1 EMA20 trend filter: only trade in direction of daily trend."""
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df['EMA20_D1'] = df['Close'].ewm(span=20*24).mean()
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    ema_d1 = df['EMA20_D1'].values
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
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                pnl = (c[i]-pos['entry']-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl))
                pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                pnl = (pos['entry']-c[i]-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl))
                pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(score[i]) or np.isnan(score[i-1]): continue
        if np.isnan(ema_d1[i]): continue
        if score[i] > 0 and score[i-1] <= 0 and c[i] > ema_d1[i]:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i-1] >= 0 and c[i] < ema_d1[i]:
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
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
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


def bt_chandelier(h1_df, spread, lot, maxloss_cap=0,
                  period=22, mult=3.0, sl_atr=4.5, tp_atr=6.0,
                  trail_act=0.14, trail_dist=0.025, max_hold=20):
    """S4 Chandelier Exit Flip strategy."""
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    ch = calc_chandelier(df, period, mult)
    df['Chand_long'] = ch['Chand_long']
    df['Chand_short'] = ch['Chand_short']
    df = df.dropna(subset=['ATR', 'Chand_long', 'Chand_short'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    chand_long = df['Chand_long'].values
    chand_short = df['Chand_short'].values
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        prev_above = c[i-1] > chand_long[i-1]
        curr_above = c[i] > chand_long[i]
        prev_below = c[i-1] < chand_short[i-1]
        curr_below = c[i] < chand_short[i]
        if curr_above and not prev_above:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif curr_below and not prev_below:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_l8_max(data_bundle, spread, lot, maxloss_cap=35):
    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': maxloss_cap,
          'spread_cost': spread, 'initial_capital': 2000,
          'min_lot_size': lot, 'max_lot_size': lot}
    result = run_variant(data_bundle, "L8_MAX", verbose=False, **kw)
    trades = []
    for t in result.get('_trades', []):
        trades.append({
            'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0,
        })
    return trades


# ═══════════════════════════════════════════════════════════════
# Portfolio helpers
# ═══════════════════════════════════════════════════════════════

def build_portfolio_daily(strat_trades, lots):
    all_daily = {}
    for sn, trades in strat_trades.items():
        lot = lots.get(sn, UNIT_LOT); multiplier = lot / UNIT_LOT
        for t in trades:
            d = pd.Timestamp(t['exit_time']).date()
            all_daily[d] = all_daily.get(d, 0) + t['pnl'] * multiplier
    dates = sorted(all_daily.keys())
    return np.array([all_daily[d] for d in dates]), dates


def compute_monthly_metrics(trades):
    """Group trades by month and compute per-month stats."""
    if not trades:
        return []
    months = defaultdict(list)
    for t in trades:
        m = pd.Timestamp(t['exit_time']).to_period('M')
        months[m].append(t)

    monthly = []
    for m in sorted(months.keys()):
        tlist = months[m]
        pnls = [t['pnl'] for t in tlist]
        daily = _trades_to_daily(tlist)
        daily_arr = daily.values if len(daily) > 0 else np.array([0.0])
        wins = sum(1 for p in pnls if p > 0)
        monthly.append({
            'month': str(m),
            'n_trades': len(tlist),
            'pnl': round(sum(pnls), 2),
            'wr': round(wins / len(tlist) * 100, 1) if tlist else 0,
            'sharpe': round(_sharpe(daily_arr), 3),
        })
    return monthly


# ═══════════════════════════════════════════════════════════════
# Comparison framework
# ═══════════════════════════════════════════════════════════════

def load_paper_trades(variant_name):
    """Load paper trades from CSV. Returns list of trade dicts or None."""
    csv_path = Path(f"results/{variant_name}/paper_trades.csv")
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    required = ['entry_time', 'exit_time', 'pnl']
    if not all(col in df.columns for col in required):
        return None
    trades = []
    for _, row in df.iterrows():
        trades.append({
            'dir': row.get('dir', 'BUY'),
            'entry': row.get('entry', 0),
            'exit': row.get('exit', 0),
            'entry_time': row['entry_time'],
            'exit_time': row['exit_time'],
            'pnl': float(row['pnl']),
            'reason': row.get('reason', ''),
            'bars': int(row.get('bars', 0)),
        })
    return trades


def compare_vs_baseline(paper_trades, baseline):
    """Compare paper trade results against backtest baseline.

    Returns a report dict with comparison metrics and statistical test.
    """
    if not paper_trades or not baseline:
        return {'status': 'insufficient_data'}

    paper_monthly = compute_monthly_metrics(paper_trades)
    if len(paper_monthly) < 2:
        return {'status': 'too_few_months', 'months': len(paper_monthly)}

    paper_pnls_m = [m['pnl'] for m in paper_monthly]
    base_pnls_m = [m['pnl'] for m in baseline.get('monthly', [])]

    paper_stats = _compute_stats(paper_trades)
    base_stats = baseline.get('overall', {})

    report = {
        'status': 'ok',
        'paper': {
            'n_trades': paper_stats['n'],
            'sharpe': paper_stats['sharpe'],
            'pnl': paper_stats['pnl'],
            'wr': paper_stats['wr'],
            'monthly_avg_pnl': round(np.mean(paper_pnls_m), 2) if paper_pnls_m else 0,
            'monthly_avg_trades': round(np.mean([m['n_trades'] for m in paper_monthly]), 1),
        },
        'baseline': {
            'sharpe': base_stats.get('sharpe', 0),
            'pnl': base_stats.get('pnl', 0),
            'wr': base_stats.get('wr', 0),
            'monthly_avg_pnl': base_stats.get('monthly_avg_pnl', 0),
            'monthly_avg_trades': base_stats.get('monthly_avg_trades', 0),
        },
        'deltas': {
            'sharpe': round(paper_stats['sharpe'] - base_stats.get('sharpe', 0), 3),
            'pnl_pct': round((paper_stats['pnl'] - base_stats.get('pnl', 0)) /
                             max(abs(base_stats.get('pnl', 1)), 1) * 100, 1),
            'wr': round(paper_stats['wr'] - base_stats.get('wr', 0), 1),
        },
    }

    if base_pnls_m and len(paper_pnls_m) >= 3:
        t_stat, p_val = sp_stats.ttest_ind(paper_pnls_m, base_pnls_m, equal_var=False)
        report['t_test'] = {
            't_stat': round(float(t_stat), 3),
            'p_value': round(float(p_val), 4),
            'significant': p_val < 0.05,
        }
    else:
        report['t_test'] = {'status': 'insufficient_data'}

    return report


def detect_decay(trades, baseline_sharpe, window=20):
    """Rolling 20-trade Sharpe decay detection.

    Returns list of alerts and whether review is triggered.
    """
    if len(trades) < window:
        return {'alerts': [], 'review_triggered': False, 'status': 'too_few_trades'}

    alerts = []
    rolling_sharpes = []
    threshold = baseline_sharpe * DECAY_ALERT_RATIO

    for i in range(window, len(trades) + 1):
        chunk = trades[i-window:i]
        daily = _trades_to_daily(chunk)
        sh = _sharpe(daily.values) if len(daily) > 0 else 0
        rolling_sharpes.append({'trade_idx': i, 'sharpe': round(sh, 3)})
        if sh < threshold:
            alerts.append({
                'trade_idx': i,
                'rolling_sharpe': round(sh, 3),
                'threshold': round(threshold, 3),
                'severity': 'ALERT',
            })

    monthly_below = 0
    review_triggered = False
    monthly = compute_monthly_metrics(trades)
    for m in monthly:
        if m['sharpe'] < threshold:
            monthly_below += 1
            if monthly_below >= CONSECUTIVE_MONTHS_REVIEW:
                review_triggered = True
        else:
            monthly_below = 0

    return {
        'alerts': alerts,
        'n_alerts': len(alerts),
        'alert_rate': round(len(alerts) / max(len(rolling_sharpes), 1) * 100, 1),
        'review_triggered': review_triggered,
        'consecutive_months_below': monthly_below,
        'rolling_sharpes': rolling_sharpes,
    }


def generate_report_md(variant_name, comparison, decay_info, baseline):
    """Generate markdown report comparing paper vs backtest."""
    lines = [
        f"# Paper Trade Report: {variant_name}",
        f"",
        f"## Summary",
        f"",
    ]

    if comparison.get('status') != 'ok':
        lines.append(f"**Status:** {comparison.get('status', 'unknown')}")
        return '\n'.join(lines)

    p = comparison['paper']
    b = comparison['baseline']
    d = comparison['deltas']

    lines.extend([
        f"| Metric | Paper | Backtest | Delta |",
        f"|--------|-------|----------|-------|",
        f"| Sharpe | {p['sharpe']:.3f} | {b['sharpe']:.3f} | {d['sharpe']:+.3f} |",
        f"| PnL | ${p['pnl']:,.2f} | ${b['pnl']:,.2f} | {d['pnl_pct']:+.1f}% |",
        f"| Win Rate | {p['wr']:.1f}% | {b['wr']:.1f}% | {d['wr']:+.1f}% |",
        f"| Monthly Avg PnL | ${p['monthly_avg_pnl']:,.2f} | ${b['monthly_avg_pnl']:,.2f} | |",
        f"| Monthly Avg Trades | {p['monthly_avg_trades']:.1f} | {b['monthly_avg_trades']:.1f} | |",
        f"",
    ])

    t_test = comparison.get('t_test', {})
    if t_test.get('status') != 'insufficient_data':
        sig = "YES" if t_test.get('significant') else "NO"
        lines.extend([
            f"## Statistical Test",
            f"- t-statistic: {t_test.get('t_stat', 'N/A')}",
            f"- p-value: {t_test.get('p_value', 'N/A')}",
            f"- Significant (p<0.05): {sig}",
            f"",
        ])

    lines.extend([
        f"## Decay Detection",
        f"- Alerts triggered: {decay_info.get('n_alerts', 0)}",
        f"- Alert rate: {decay_info.get('alert_rate', 0):.1f}%",
        f"- Review triggered: {'YES' if decay_info.get('review_triggered') else 'NO'}",
        f"",
    ])

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
# Demo: simulate paper trades from backtest distribution
# ═══════════════════════════════════════════════════════════════

def simulate_paper_trades(trades, n_months=DEMO_MONTHS, noise_pnl=DEMO_NOISE_PNL,
                          miss_rate=DEMO_MISS_RATE, seed=42):
    """Sample from backtest trade distribution to simulate paper trading.

    Adds noise (+/-10% PnL per trade) and drops 5% of trades.
    """
    rng = np.random.RandomState(seed)
    if not trades:
        return []

    monthly = defaultdict(list)
    for t in trades:
        m = pd.Timestamp(t['exit_time']).to_period('M')
        monthly[m].append(t)

    all_months = sorted(monthly.keys())
    if len(all_months) < n_months:
        sample_months = all_months
    else:
        start_idx = rng.randint(0, len(all_months) - n_months)
        sample_months = all_months[start_idx:start_idx + n_months]

    simulated = []
    for m in sample_months:
        for t in monthly[m]:
            if rng.rand() < miss_rate:
                continue
            t2 = dict(t)
            noise = rng.uniform(-noise_pnl, noise_pnl)
            t2['pnl'] = t2['pnl'] * (1.0 + noise)
            simulated.append(t2)

    return simulated


def load_h1():
    candidates = [
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-27.csv"),
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv"),
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-03-25.csv"),
    ]
    h1_path = next((p for p in candidates if p.exists()), candidates[-1])
    return load_csv(str(h1_path))


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    results = {'experiment': 'R143 Paper Trade Automated Verification Framework'}

    print("=" * 80, flush=True)
    print("  R143 — Paper Trade Automated Verification Framework", flush=True)
    print("=" * 80, flush=True)

    # ─── Load data ────────────────────────────────────────────
    print("\n  Loading data...", flush=True)
    h1_df = load_h1()
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0].date()} ~ {h1_df.index[-1].date()})", flush=True)

    print("  Loading DataBundle for L8_MAX...", flush=True)
    try:
        bundle = DataBundle.load_custom()
        have_l8 = True
    except Exception as e:
        print(f"  WARN: DataBundle load failed: {e}", flush=True)
        bundle = None; have_l8 = False

    # ═════════════════════════════════════════════════════════════
    # Phase 1: Define strategy variants
    # ═════════════════════════════════════════════════════════════
    print("\n" + "=" * 60, flush=True)
    print("  Phase 1: Define Strategy Variants & Run Backtests", flush=True)
    print("=" * 60, flush=True)

    variants = {
        'V1_BASE': {
            'description': 'Current production (PSAR default, TSMOM unfiltered, no S4)',
            'strategies': ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO'],
            'lots': R89_LOTS,
        },
        'V2_PSAR_OPT': {
            'description': 'PSAR with optimized params (sl=4.0, tp=6.0, trail_act=0.08)',
            'strategies': ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO'],
            'lots': R89_LOTS,
            'psar_params': PSAR_OPTIMIZED,
        },
        'V3_TSMOM_D1': {
            'description': 'TSMOM with D1 EMA20 filter',
            'strategies': ['L8_MAX', 'PSAR', 'TSMOM_D1', 'SESS_BO'],
            'lots': {**R89_LOTS, 'TSMOM_D1': R89_LOTS['TSMOM']},
        },
        'V4_S4': {
            'description': 'Chandelier Exit Flip standalone',
            'strategies': ['CHANDELIER'],
            'lots': {'CHANDELIER': 0.05},
        },
        'V5_FULL': {
            'description': 'V2 + V3 + V4 combined portfolio',
            'strategies': ['L8_MAX', 'PSAR', 'TSMOM_D1', 'SESS_BO', 'CHANDELIER'],
            'lots': {**R89_LOTS, 'TSMOM_D1': R89_LOTS['TSMOM'], 'CHANDELIER': 0.05},
            'psar_params': PSAR_OPTIMIZED,
        },
    }

    variant_trades = {}
    variant_stats = {}

    for vname, vconfig in variants.items():
        print(f"\n  Running {vname}: {vconfig['description']}", flush=True)
        strat_trades = {}
        psar_params = vconfig.get('psar_params', PSAR_DEFAULT)

        for sn in vconfig['strategies']:
            if sn == 'L8_MAX':
                if have_l8:
                    strat_trades[sn] = bt_l8_max(bundle, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
                else:
                    strat_trades[sn] = []
            elif sn == 'PSAR':
                strat_trades[sn] = bt_psar(h1_df, SPREAD, UNIT_LOT, CAPS['PSAR'], psar_params)
            elif sn == 'TSMOM':
                strat_trades[sn] = bt_tsmom(h1_df, SPREAD, UNIT_LOT, CAPS['TSMOM'])
            elif sn == 'TSMOM_D1':
                strat_trades[sn] = bt_tsmom_d1_filter(h1_df, SPREAD, UNIT_LOT, CAPS['TSMOM'])
            elif sn == 'SESS_BO':
                strat_trades[sn] = bt_sess_bo(h1_df, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
            elif sn == 'CHANDELIER':
                strat_trades[sn] = bt_chandelier(h1_df, SPREAD, UNIT_LOT)
            print(f"    {sn}: {len(strat_trades[sn])} trades", flush=True)

        all_trades = []
        for sn in vconfig['strategies']:
            lot = vconfig['lots'].get(sn, UNIT_LOT)
            for t in strat_trades[sn]:
                t2 = dict(t)
                t2['pnl'] = t['pnl'] * (lot / UNIT_LOT)
                all_trades.append(t2)

        stats = _compute_stats(all_trades)
        variant_trades[vname] = all_trades
        variant_stats[vname] = stats
        print(f"    Portfolio: {stats['n']} trades, Sharpe={stats['sharpe']:.3f}, "
              f"PnL=${stats['pnl']:,.2f}, WR={stats['wr']:.1f}%", flush=True)

    results['phase1_variants'] = {v: variant_stats[v] for v in variants}

    # ═════════════════════════════════════════════════════════════
    # Phase 2: Compute monthly baselines
    # ═════════════════════════════════════════════════════════════
    print("\n" + "=" * 60, flush=True)
    print("  Phase 2: Monthly Expected Metrics (Baselines)", flush=True)
    print("=" * 60, flush=True)

    baselines = {}
    print(f"\n  {'Variant':<15s} {'Months':>7} {'AvgTrades':>10} {'AvgPnL':>10} "
          f"{'AvgSharpe':>10} {'ExpWR':>7}", flush=True)
    print(f"  {'─'*62}", flush=True)

    for vname in variants:
        trades = variant_trades[vname]
        monthly = compute_monthly_metrics(trades)
        overall = variant_stats[vname]

        if monthly:
            avg_trades = np.mean([m['n_trades'] for m in monthly])
            avg_pnl = np.mean([m['pnl'] for m in monthly])
            avg_sharpe = np.mean([m['sharpe'] for m in monthly])
        else:
            avg_trades = avg_pnl = avg_sharpe = 0

        baselines[vname] = {
            'overall': {
                **overall,
                'monthly_avg_pnl': round(avg_pnl, 2),
                'monthly_avg_trades': round(avg_trades, 1),
                'monthly_avg_sharpe': round(avg_sharpe, 3),
            },
            'monthly': monthly,
        }

        print(f"  {vname:<15s} {len(monthly):>7d} {avg_trades:>10.1f} ${avg_pnl:>9.2f} "
              f"{avg_sharpe:>10.3f} {overall['wr']:>6.1f}%", flush=True)

    baseline_path = OUTPUT_DIR / "baselines.json"
    with open(baseline_path, 'w') as f:
        json.dump(baselines, f, indent=2, default=str)
    print(f"\n  Baselines saved: {baseline_path}", flush=True)

    results['phase2_baselines'] = {v: baselines[v]['overall'] for v in variants}

    # ═════════════════════════════════════════════════════════════
    # Phase 3: Comparison framework (with real paper trades if available)
    # ═════════════════════════════════════════════════════════════
    print("\n" + "=" * 60, flush=True)
    print("  Phase 3: Comparison Framework", flush=True)
    print("=" * 60, flush=True)

    paper_comparisons = {}
    for vname in variants:
        paper_trades = load_paper_trades(vname)
        if paper_trades is not None:
            comparison = compare_vs_baseline(paper_trades, baselines[vname])
            paper_comparisons[vname] = comparison
            print(f"\n  {vname}: Paper trades found ({len(paper_trades)} trades)", flush=True)
            if comparison.get('status') == 'ok':
                d = comparison['deltas']
                print(f"    dSharpe={d['sharpe']:+.3f}, dPnL={d['pnl_pct']:+.1f}%, dWR={d['wr']:+.1f}%", flush=True)
        else:
            paper_comparisons[vname] = {'status': 'no_paper_trades'}
            print(f"  {vname}: No paper trades found (will test in demo mode)", flush=True)

    results['phase3_comparisons'] = paper_comparisons

    # ═════════════════════════════════════════════════════════════
    # Phase 4: Decay detection (on backtest data as a dry run)
    # ═════════════════════════════════════════════════════════════
    print("\n" + "=" * 60, flush=True)
    print("  Phase 4: Decay Detection Framework", flush=True)
    print("=" * 60, flush=True)

    decay_results = {}
    for vname in variants:
        trades = variant_trades[vname]
        base_sharpe = variant_stats[vname]['sharpe']
        decay = detect_decay(trades, base_sharpe)
        decay_results[vname] = {
            'n_alerts': decay['n_alerts'],
            'alert_rate': decay['alert_rate'],
            'review_triggered': decay['review_triggered'],
        }
        status = "REVIEW" if decay['review_triggered'] else (
            f"{decay['n_alerts']} alerts" if decay['n_alerts'] > 0 else "CLEAN")
        print(f"  {vname:<15s}: {status} (alert_rate={decay['alert_rate']:.1f}%)", flush=True)

    results['phase4_decay'] = decay_results

    # ═════════════════════════════════════════════════════════════
    # Phase 5: Auto-report generation
    # ═════════════════════════════════════════════════════════════
    print("\n" + "=" * 60, flush=True)
    print("  Phase 5: Auto-Report Generation", flush=True)
    print("=" * 60, flush=True)

    chart_data = {}
    for vname in variants:
        trades = variant_trades[vname]
        monthly = baselines[vname]['monthly']

        cumulative_pnl = []
        running = 0.0
        for m in monthly:
            running += m['pnl']
            cumulative_pnl.append({'month': m['month'], 'cumulative_pnl': round(running, 2)})

        chart_data[vname] = {
            'monthly_pnl': [{'month': m['month'], 'pnl': m['pnl']} for m in monthly],
            'cumulative_pnl': cumulative_pnl,
            'monthly_sharpe': [{'month': m['month'], 'sharpe': m['sharpe']} for m in monthly],
        }

    chart_path = OUTPUT_DIR / "chart_data.json"
    with open(chart_path, 'w') as f:
        json.dump(chart_data, f, indent=2)
    print(f"  Chart data saved: {chart_path}", flush=True)

    results['phase5_chart_data'] = 'chart_data.json'

    # ═════════════════════════════════════════════════════════════
    # Phase 6: Demo mode — simulate 6 months of paper trades
    # ═════════════════════════════════════════════════════════════
    print("\n" + "=" * 60, flush=True)
    print(f"  Phase 6: Demo Mode ({DEMO_MONTHS}-month Paper Trade Simulation)", flush=True)
    print("=" * 60, flush=True)

    demo_results = {}

    for vname in variants:
        trades = variant_trades[vname]
        base_sharpe = variant_stats[vname]['sharpe']

        sim_trades = simulate_paper_trades(trades, n_months=DEMO_MONTHS)
        if not sim_trades:
            demo_results[vname] = {'status': 'no_trades_to_simulate'}
            print(f"\n  {vname}: No trades to simulate", flush=True)
            continue

        comparison = compare_vs_baseline(sim_trades, baselines[vname])
        decay = detect_decay(sim_trades, base_sharpe)

        report_md = generate_report_md(vname, comparison, decay, baselines[vname])
        report_path = OUTPUT_DIR / f"demo_report_{vname}.md"
        with open(report_path, 'w') as f:
            f.write(report_md)

        demo_results[vname] = {
            'n_simulated_trades': len(sim_trades),
            'comparison': comparison,
            'decay': {
                'n_alerts': decay['n_alerts'],
                'alert_rate': decay['alert_rate'],
                'review_triggered': decay['review_triggered'],
            },
            'report_path': str(report_path),
        }

        print(f"\n  {vname} ({len(sim_trades)} simulated trades):", flush=True)
        if comparison.get('status') == 'ok':
            p = comparison['paper']
            d = comparison['deltas']
            print(f"    Sharpe={p['sharpe']:.3f} (delta={d['sharpe']:+.3f}), "
                  f"PnL=${p['pnl']:,.2f} ({d['pnl_pct']:+.1f}%), "
                  f"WR={p['wr']:.1f}% ({d['wr']:+.1f}%)", flush=True)
            t_test = comparison.get('t_test', {})
            if t_test.get('status') != 'insufficient_data':
                sig = "SIGNIFICANT" if t_test.get('significant') else "not significant"
                print(f"    t-test: t={t_test.get('t_stat', 'N/A')}, "
                      f"p={t_test.get('p_value', 'N/A')} ({sig})", flush=True)
        else:
            print(f"    Comparison status: {comparison.get('status', 'unknown')}", flush=True)

        decay_status = "REVIEW" if decay['review_triggered'] else (
            f"{decay['n_alerts']} alerts ({decay['alert_rate']:.1f}%)" if decay['n_alerts'] > 0 else "CLEAN")
        print(f"    Decay: {decay_status}", flush=True)
        print(f"    Report: {report_path}", flush=True)

    results['phase6_demo'] = demo_results

    # Verify alerts trigger correctly on degraded scenario
    print(f"\n  ─── Alert verification (deliberately degraded scenario) ───", flush=True)
    for vname in ['V1_BASE']:
        trades = variant_trades[vname]
        if not trades:
            continue
        base_sharpe = variant_stats[vname]['sharpe']

        degraded = []
        rng = np.random.RandomState(999)
        for t in trades:
            t2 = dict(t)
            t2['pnl'] = t2['pnl'] * 0.3 + rng.normal(0, 0.5)
            degraded.append(t2)

        deg_decay = detect_decay(degraded, base_sharpe)
        print(f"  {vname} (degraded 70%):", flush=True)
        print(f"    Alerts: {deg_decay['n_alerts']}, rate={deg_decay['alert_rate']:.1f}%", flush=True)
        print(f"    Review triggered: {'YES' if deg_decay['review_triggered'] else 'NO'}", flush=True)

        alerts_triggered = deg_decay['n_alerts'] > 0
        review_triggered = deg_decay['review_triggered']
        print(f"    Alert verification: {'PASS' if alerts_triggered else 'FAIL'} "
              f"(expected alerts to trigger)", flush=True)
        print(f"    Review verification: {'PASS' if review_triggered else 'FAIL'} "
              f"(expected review to trigger)", flush=True)

        results['phase6_verification'] = {
            'degraded_alerts': deg_decay['n_alerts'],
            'degraded_review': deg_decay['review_triggered'],
            'alert_test_pass': alerts_triggered,
            'review_test_pass': review_triggered,
        }

    # ─── Final summary ────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("  Final Summary", flush=True)
    print("=" * 60, flush=True)

    print(f"\n  {'Variant':<15s} {'Sharpe':>7} {'PnL':>12} {'WR':>6} {'Trades':>7}", flush=True)
    print(f"  {'─'*50}", flush=True)
    for vname in variants:
        s = variant_stats[vname]
        print(f"  {vname:<15s} {s['sharpe']:>7.3f} ${s['pnl']:>11,.2f} {s['wr']:>5.1f}% "
              f"{s['n']:>7d}", flush=True)

    print(f"\n  Framework capabilities:", flush=True)
    print(f"    - load_paper_trades(variant) -> reads results/<variant>/paper_trades.csv", flush=True)
    print(f"    - compare_vs_baseline(paper_trades, baseline) -> comparison report", flush=True)
    print(f"    - detect_decay(trades, baseline_sharpe) -> alerts + review trigger", flush=True)
    print(f"    - generate_report_md(variant, comparison, decay, baseline) -> markdown", flush=True)
    print(f"    - simulate_paper_trades(trades, n_months) -> simulated paper trades", flush=True)

    # ─── Save results ─────────────────────────────────────────
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)
    print(f"\n  Total elapsed: {elapsed:.0f}s", flush=True)

    out_path = OUTPUT_DIR / "r143_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {out_path}", flush=True)


if __name__ == '__main__':
    main()
