#!/usr/bin/env python3
"""
R148 — Live vs Backtest Audit
==============================
Compares live trade log (_live/data/gold_trade_log.json) with backtest results
over the same period (2026-03-25 ~ 2026-05-05) to validate signal alignment
and quantify execution gaps.

Phases:
  1. Same-period backtest for all 6 strategies
  2. Signal alignment: match live trades to backtest trades
  3. PnL comparison per strategy and overall
  4. Summary report
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import load_csv, DataBundle, run_variant, LIVE_PARITY_KWARGS

OUTPUT_DIR = Path("results/r148_live_vs_backtest")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30

LIVE_PERIOD_START = "2026-03-25"
LIVE_PERIOD_END = "2026-05-06"

H1_CANDIDATES = [
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-05-05.csv"),
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-27.csv"),
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv"),
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-03-25.csv"),
]

# ═══════════════════════════════════════════════════════════════
# Helpers from R146 (inline to avoid import issues)
# ═══════════════════════════════════════════════════════════════

def calc_dual_thrust_range(df, n_bars=6):
    hh = df['High'].rolling(n_bars).max()
    lc = df['Close'].rolling(n_bars).min()
    hc = df['Close'].rolling(n_bars).max()
    ll = df['Low'].rolling(n_bars).min()
    return pd.concat([hh - lc, hc - ll], axis=1).max(axis=1)

def calc_chandelier(df, period=22, mult=3.0):
    atr = (df['High'] - df['Low']).rolling(14).mean()
    hh = df['High'].rolling(period).max()
    ll = df['Low'].rolling(period).min()
    out = pd.DataFrame(index=df.index)
    out['Chand_long'] = hh - mult * atr
    out['Chand_short'] = ll + mult * atr
    return out

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


# ═══════════════════════════════════════════════════════════════
# Strategy backtest functions
# ═══════════════════════════════════════════════════════════════

def bt_s3(h1_df, spread, lot, start=None, end=None, cap=35):
    df = h1_df.copy()
    df['ATR14'] = compute_atr(df, 14)
    dt_range = calc_dual_thrust_range(df, 6)
    daily_open = df.groupby(df.index.date)['Open'].transform('first')
    sig = pd.Series(0, index=df.index)
    sig[df['Close'] > daily_open + 0.5 * dt_range] = 1
    sig[df['Close'] < daily_open - 0.5 * dt_range] = -1

    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr_arr = df['ATR14'].values; times = df.index; n = len(df)
    sig_arr = sig.values; dates = df.index.date

    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if start and str(dates[i]) < start: continue
        if end and str(dates[i]) > end: break
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               4.5, 8.0, 0.14, 0.025, 20, cap)
            if result: trades.append(result); pos = None; last_exit = i
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr_arr[i]) or atr_arr[i] < 0.1: continue
        if sig_arr[i] == 1 and sig_arr[i-1] != 1:
            pos = {'dir': 'BUY', 'entry': c[i] + spread/2, 'bar': i, 'time': times[i], 'atr': atr_arr[i]}
        elif sig_arr[i] == -1 and sig_arr[i-1] != -1:
            pos = {'dir': 'SELL', 'entry': c[i] - spread/2, 'bar': i, 'time': times[i], 'atr': atr_arr[i]}
    return trades


def bt_s4(h1_df, spread, lot, start=None, end=None, cap=35):
    df = h1_df.copy()
    df['ATR14'] = compute_atr(df, 14)
    ch = calc_chandelier(df, 22, 3.0)
    ema100 = df['Close'].ewm(span=100).mean()
    above_long = df['Close'] > ch['Chand_long']
    flip_bull = above_long & (~above_long.shift(1).fillna(False))
    below_short = df['Close'] < ch['Chand_short']
    flip_bear = below_short & (~below_short.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index)
    sig[flip_bull & (df['Close'] > ema100)] = 1
    sig[flip_bear & (df['Close'] < ema100)] = -1

    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr_arr = df['ATR14'].values; times = df.index; n = len(df)
    sig_arr = sig.values; dates = df.index.date

    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if start and str(dates[i]) < start: continue
        if end and str(dates[i]) > end: break
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               4.5, 8.0, 0.14, 0.025, 20, cap)
            if result: trades.append(result); pos = None; last_exit = i
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr_arr[i]) or atr_arr[i] < 0.1: continue
        if sig_arr[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i] + spread/2, 'bar': i, 'time': times[i], 'atr': atr_arr[i]}
        elif sig_arr[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i] - spread/2, 'bar': i, 'time': times[i], 'atr': atr_arr[i]}
    return trades


def bt_psar(h1_df, spread, lot, start=None, end=None, cap=5):
    """PSAR with R127 optimized params."""
    df = h1_df.copy(); add_psar(df); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR', 'PSAR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; psar = df['PSAR'].values; times = df.index; n = len(df)
    dates = df.index.date
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if start and str(dates[i]) < start: continue
        if end and str(dates[i]) > end: break
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               4.0, 6.0, 0.08, 0.015, 15, cap)
            if result: trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if psar[i-1] > c[i-1] and psar[i] < c[i]:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif psar[i-1] < c[i-1] and psar[i] > c[i]:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom(h1_df, spread, lot, start=None, end=None, cap=0):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    dates = df.index.date
    fast, slow = 480, 720
    max_lb = max(fast, slow)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        if c[i-fast] > 0: s += 0.5 * np.sign(c[i]/c[i-fast] - 1.0)
        if c[i-slow] > 0: s += 0.5 * np.sign(c[i]/c[i-slow] - 1.0)
        score[i] = s
    trades = []; pos = None; last_exit = -999
    for i in range(max_lb+1, n):
        if start and str(dates[i]) < start: continue
        if end and str(dates[i]) > end: break
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               4.5, 6.0, 0.14, 0.025, 20, cap)
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


def bt_sess_bo(h1_df, spread, lot, start=None, end=None, cap=35):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour; times = df.index; n = len(df)
    dates = df.index.date
    session_hour, lookback = 12, 4
    trades = []; pos = None; last_exit = -999
    for i in range(lookback+1, n):
        if start and str(dates[i]) < start: continue
        if end and str(dates[i]) > end: break
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               4.5, 4.0, 0.14, 0.025, 20, cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if hours[i] != session_hour: continue
        sess_h = max(h[i-lookback:i+1])
        sess_l = min(lo[i-lookback:i+1])
        if c[i] > sess_h - 0.001:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < sess_l + 0.001:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


# ═══════════════════════════════════════════════════════════════
# Keltner via engine (uses M15 + H1)
# ═══════════════════════════════════════════════════════════════

def bt_keltner_engine(start, end):
    """Run keltner via the full engine with LIVE_PARITY_KWARGS."""
    data = DataBundle.load_default()
    data = data.slice(start, end)
    kw = dict(LIVE_PARITY_KWARGS)
    kw['max_loss_cap'] = 35
    stats = run_variant(data, "Keltner_LiveParity", verbose=False, **kw)
    return stats.get('_trades', [])


# ═══════════════════════════════════════════════════════════════
# Statistics helpers
# ═══════════════════════════════════════════════════════════════

def compute_stats(trades):
    if not trades:
        return {'n': 0, 'pnl': 0, 'wr': 0, 'avg_pnl': 0, 'max_dd': 0}
    pnls = [t['pnl'] if isinstance(t, dict) and 'pnl' in t else t.pnl for t in trades]
    n = len(pnls); total = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    daily = {}
    for t in trades:
        if isinstance(t, dict):
            d = pd.Timestamp(t.get('exit_time', t.get('entry_time', ''))).date()
            p = t['pnl']
        else:
            d = pd.Timestamp(t.exit_time).date()
            p = t.pnl
        daily[d] = daily.get(d, 0) + p
    arr = np.array([daily[k] for k in sorted(daily.keys())])
    eq = np.cumsum(arr)
    dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 1 else 0
    return {
        'n': n, 'pnl': round(total, 2), 'wr': round(wins/n*100, 1) if n else 0,
        'avg_pnl': round(total/n, 2) if n else 0, 'max_dd': round(dd, 2),
    }


# ═══════════════════════════════════════════════════════════════
# Signal alignment
# ═══════════════════════════════════════════════════════════════

def match_trades(live_trades, bt_trades, time_window_hours=3):
    """Match live trades to backtest trades by time and direction."""
    matched = []
    bt_unmatched = list(range(len(bt_trades)))
    live_unmatched = []

    for li, lt in enumerate(live_trades):
        live_time = pd.Timestamp(lt['time'])
        live_dir = lt.get('direction', '').upper()
        live_price = lt.get('price', lt.get('open_price', 0))

        best_match = None
        best_dt = timedelta(hours=999)

        for bi in bt_unmatched:
            bt = bt_trades[bi]
            bt_time = pd.Timestamp(bt['entry_time'])
            if bt_time.tzinfo:
                bt_time = bt_time.tz_localize(None)
            bt_dir = bt['dir'].upper()

            dt = abs(live_time - bt_time)
            if dt < timedelta(hours=time_window_hours) and live_dir[:3] == bt_dir[:3]:
                if dt < best_dt:
                    best_dt = dt
                    best_match = bi

        if best_match is not None:
            bt = bt_trades[best_match]
            bt_time = pd.Timestamp(bt['entry_time'])
            if bt_time.tzinfo:
                bt_time = bt_time.tz_localize(None)
            slippage = live_price - bt['entry']
            matched.append({
                'live_idx': li,
                'bt_idx': best_match,
                'live_time': str(live_time),
                'bt_time': str(bt_time),
                'time_diff_min': round(best_dt.total_seconds()/60, 1),
                'direction': live_dir,
                'live_price': live_price,
                'bt_price': round(bt['entry'], 2),
                'slippage': round(slippage, 2),
                'live_pnl': lt.get('profit', 0),
                'bt_pnl': round(bt['pnl'], 2),
            })
            bt_unmatched.remove(best_match)
        else:
            live_unmatched.append(li)

    return matched, live_unmatched, bt_unmatched


def main():
    t0 = time.time()

    # Load live trade log
    live_log = json.load(open('_live/data/gold_trade_log.json'))
    equity_curve = json.load(open('_live/data/equity_curve.json'))

    live_opens = [t for t in live_log if t.get('action') == 'OPEN']
    live_closes = [t for t in live_log if t.get('action') in ('CLOSE', 'CLOSE_DETECTED', 'CLOSE_DETECTED_DUPLICATE')]

    print("=" * 80)
    print("  R148: Live vs Backtest Audit")
    print("=" * 80)
    print(f"\n  Live period: {LIVE_PERIOD_START} -> {LIVE_PERIOD_END}")
    print(f"  Live trades: {len(live_opens)} opens, {len(live_closes)} closes")
    print(f"  Equity curve: {len(equity_curve)} days")
    total_live_pnl = sum(t.get('profit', 0) for t in live_closes)
    print(f"  Total live PnL: ${total_live_pnl:.2f}")

    # Group live trades by strategy
    live_by_strat = {}
    for t in live_opens:
        s = t.get('strategy', 'unknown')
        live_by_strat.setdefault(s, []).append(t)

    live_close_by_strat = {}
    for t in live_closes:
        s = t.get('strategy', 'unknown')
        live_close_by_strat.setdefault(s, []).append(t)

    print("\n  Live trades by strategy:")
    for s in sorted(live_by_strat.keys()):
        opens = len(live_by_strat[s])
        closes = len(live_close_by_strat.get(s, []))
        pnl = sum(t.get('profit', 0) for t in live_close_by_strat.get(s, []))
        print(f"    {s:>15s}: {opens:>3d} opens, {closes:>3d} closes, PnL=${pnl:>8.2f}")

    # ══════════════════════════════════════════════════════════
    # Phase 1: Same-period backtest
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 1: Same-Period Backtest (2026-03-25 ~ 2026-05-05)")
    print("=" * 80)

    csv_path = next((p for p in H1_CANDIDATES if p.exists()), H1_CANDIDATES[0])
    print(f"\n  Loading H1: {csv_path}")
    h1 = load_csv(str(csv_path))
    print(f"  {len(h1)} bars: {h1.index[0]} -> {h1.index[-1]}")

    bt_results = {}

    # Keltner via full engine
    print("\n  Running Keltner (full engine)...", flush=True)
    try:
        keltner_trades_raw = bt_keltner_engine(LIVE_PERIOD_START, LIVE_PERIOD_END)
        keltner_trades = []
        for t in keltner_trades_raw:
            keltner_trades.append({
                'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
                'entry_time': t.entry_time, 'exit_time': t.exit_time,
                'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0,
            })
        bt_results['keltner'] = keltner_trades
        st = compute_stats(keltner_trades)
        print(f"    Keltner: {st['n']} trades, PnL=${st['pnl']:.2f}, WR={st['wr']}%")
    except Exception as e:
        print(f"    Keltner engine error: {e}")
        bt_results['keltner'] = []

    # S3 Dual Thrust
    print("  Running S3 Dual Thrust...", flush=True)
    s3_trades = bt_s3(h1, SPREAD, 0.04, start=LIVE_PERIOD_START, end=LIVE_PERIOD_END)
    bt_results['dual_thrust'] = s3_trades
    st = compute_stats(s3_trades)
    print(f"    S3: {st['n']} trades, PnL=${st['pnl']:.2f}, WR={st['wr']}%")

    # S4 Chandelier
    print("  Running S4 Chandelier...", flush=True)
    s4_trades = bt_s4(h1, SPREAD, 0.08, start=LIVE_PERIOD_START, end=LIVE_PERIOD_END)
    bt_results['chandelier'] = s4_trades
    st = compute_stats(s4_trades)
    print(f"    S4: {st['n']} trades, PnL=${st['pnl']:.2f}, WR={st['wr']}%")

    # PSAR
    print("  Running PSAR (R127 params)...", flush=True)
    psar_trades = bt_psar(h1, SPREAD, 0.09, start=LIVE_PERIOD_START, end=LIVE_PERIOD_END)
    bt_results['psar'] = psar_trades
    st = compute_stats(psar_trades)
    print(f"    PSAR: {st['n']} trades, PnL=${st['pnl']:.2f}, WR={st['wr']}%")

    # TSMOM
    print("  Running TSMOM...", flush=True)
    tsmom_trades = bt_tsmom(h1, SPREAD, 0.15, start=LIVE_PERIOD_START, end=LIVE_PERIOD_END)
    bt_results['tsmom'] = tsmom_trades
    st = compute_stats(tsmom_trades)
    print(f"    TSMOM: {st['n']} trades, PnL=${st['pnl']:.2f}, WR={st['wr']}%")

    # SESS_BO
    print("  Running SESS_BO...", flush=True)
    sessbo_trades = bt_sess_bo(h1, SPREAD, 0.13, start=LIVE_PERIOD_START, end=LIVE_PERIOD_END)
    bt_results['sess_bo'] = sessbo_trades
    st = compute_stats(sessbo_trades)
    print(f"    SESS_BO: {st['n']} trades, PnL=${st['pnl']:.2f}, WR={st['wr']}%")

    # ══════════════════════════════════════════════════════════
    # Phase 2: Signal Alignment
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 2: Signal Alignment")
    print("=" * 80)

    alignment_results = {}

    # Only compare strategies that have live records
    strategies_with_live = ['keltner', 'dual_thrust']

    for strat in strategies_with_live:
        live_strat_opens = live_by_strat.get(strat, [])
        bt_strat_trades = bt_results.get(strat, [])

        if not live_strat_opens and not bt_strat_trades:
            continue

        matched, live_only, bt_only = match_trades(live_strat_opens, bt_strat_trades)

        n_live = len(live_strat_opens)
        n_bt = len(bt_strat_trades)
        n_matched = len(matched)

        print(f"\n  {strat.upper()}:")
        print(f"    Live opens: {n_live}, Backtest trades: {n_bt}")
        print(f"    Matched: {n_matched}")
        if n_live > 0:
            print(f"    Live match rate: {n_matched/n_live*100:.0f}%")
        if n_bt > 0:
            bt_match_pct = n_matched/n_bt*100
            print(f"    Backtest match rate: {bt_match_pct:.0f}%")
        print(f"    Live-only (extra): {len(live_only)}")
        print(f"    Backtest-only (missed): {len(bt_only)}")

        if matched:
            slippages = [m['slippage'] for m in matched]
            time_diffs = [m['time_diff_min'] for m in matched]
            print(f"    Avg slippage: ${np.mean(slippages):.2f} (std=${np.std(slippages):.2f})")
            print(f"    Avg time diff: {np.mean(time_diffs):.1f} min")

            print(f"\n    {'Live Time':>22s}  {'BT Time':>22s}  {'Diff':>6s}  {'Dir':>4s}  "
                  f"{'LivePx':>9s}  {'BTPx':>9s}  {'Slip':>6s}  {'LivePnL':>8s}  {'BTPnL':>8s}")
            print(f"    {'-'*100}")
            for m in matched[:15]:
                print(f"    {m['live_time'][:19]:>22s}  {m['bt_time'][:19]:>22s}  "
                      f"{m['time_diff_min']:>5.0f}m  {m['direction']:>4s}  "
                      f"${m['live_price']:>8.2f}  ${m['bt_price']:>8.2f}  "
                      f"${m['slippage']:>5.2f}  ${m['live_pnl']:>7.2f}  ${m['bt_pnl']:>7.2f}")

        alignment_results[strat] = {
            'n_live': n_live, 'n_bt': n_bt, 'n_matched': n_matched,
            'live_only': len(live_only), 'bt_only': len(bt_only),
            'matched_details': matched[:20],
            'avg_slippage': round(float(np.mean([m['slippage'] for m in matched])), 2) if matched else 0,
            'avg_time_diff_min': round(float(np.mean([m['time_diff_min'] for m in matched])), 1) if matched else 0,
        }

    # ══════════════════════════════════════════════════════════
    # Phase 3: PnL Comparison
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 3: PnL Comparison")
    print("=" * 80)

    comparison = {}
    all_strategies = set(list(live_close_by_strat.keys()) + list(bt_results.keys()))

    print(f"\n  {'Strategy':>15s}  {'Live N':>7s}  {'Live PnL':>10s}  {'BT N':>5s}  {'BT PnL':>10s}  {'Gap':>10s}  {'Source':>10s}")
    print(f"  {'-'*75}")

    total_bt_pnl = 0
    for strat in sorted(all_strategies):
        live_c = live_close_by_strat.get(strat, [])
        live_pnl = sum(t.get('profit', 0) for t in live_c)
        live_n = len(live_c)

        bt_t = bt_results.get(strat, [])
        bt_st = compute_stats(bt_t)
        bt_pnl = bt_st['pnl']
        bt_n = bt_st['n']
        total_bt_pnl += bt_pnl

        gap = live_pnl - bt_pnl if bt_n > 0 else 0
        source = "Python" if strat in ('keltner', 'm15_rsi', 'orb', 'dual_thrust') else "MT4 EA"

        print(f"  {strat:>15s}  {live_n:>7d}  ${live_pnl:>9.2f}  {bt_n:>5d}  ${bt_pnl:>9.2f}  ${gap:>9.2f}  {source:>10s}")

        comparison[strat] = {
            'live_n': live_n, 'live_pnl': round(live_pnl, 2),
            'bt_n': bt_n, 'bt_pnl': round(bt_pnl, 2),
            'gap': round(gap, 2), 'source': source,
        }

    print(f"  {'-'*75}")
    print(f"  {'TOTAL':>15s}  {len(live_closes):>7d}  ${total_live_pnl:>9.2f}  "
          f"{'':>5s}  ${total_bt_pnl:>9.2f}  ${total_live_pnl - total_bt_pnl:>9.2f}")

    # Backtest-only strategies (no live record in Python log)
    print("\n  Strategies with backtest data but no Python live log:")
    for strat in ['psar', 'tsmom', 'sess_bo', 'chandelier']:
        bt_t = bt_results.get(strat, [])
        st = compute_stats(bt_t)
        if st['n'] > 0:
            print(f"    {strat:>15s}: {st['n']} trades, PnL=${st['pnl']:.2f} (via MT4 EA, not in Python log)")

    # ══════════════════════════════════════════════════════════
    # Phase 4: Equity Curve Comparison
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 4: Daily Equity Comparison")
    print("=" * 80)

    # Build backtest daily PnL from all strategies
    bt_daily = {}
    for strat, trades in bt_results.items():
        for t in trades:
            if isinstance(t, dict):
                d = str(pd.Timestamp(t.get('exit_time', '')).date())
                p = t['pnl']
            else:
                d = str(pd.Timestamp(t.exit_time).date())
                p = t.pnl
            bt_daily[d] = bt_daily.get(d, 0) + p

    # Compare with live equity curve
    print(f"\n  {'Date':>12s}  {'Live PnL':>10s}  {'BT PnL':>10s}  {'Gap':>10s}  {'Live Cum':>10s}  {'BT Cum':>10s}")
    print(f"  {'-'*65}")

    live_cum = 0; bt_cum = 0
    daily_comparison = []
    for entry in equity_curve:
        d = entry['date']
        live_d = entry.get('daily_pnl', 0)
        bt_d = bt_daily.get(d, 0)
        gap = live_d - bt_d
        live_cum += live_d
        bt_cum += bt_d
        daily_comparison.append({
            'date': d, 'live_pnl': round(live_d, 2), 'bt_pnl': round(bt_d, 2),
            'gap': round(gap, 2), 'live_cum': round(live_cum, 2), 'bt_cum': round(bt_cum, 2),
        })
        print(f"  {d:>12s}  ${live_d:>9.2f}  ${bt_d:>9.2f}  ${gap:>9.2f}  ${live_cum:>9.2f}  ${bt_cum:>9.2f}")

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)

    print(f"\n  Live period: {LIVE_PERIOD_START} ~ {LIVE_PERIOD_END} ({len(equity_curve)} trading days)")
    print(f"  Live total PnL: ${total_live_pnl:.2f}")
    print(f"  Backtest total PnL (all 6 strategies): ${total_bt_pnl:.2f}")

    # For strategies with live Python log
    python_strats_bt_pnl = sum(compute_stats(bt_results.get(s, []))['pnl'] for s in ['keltner', 'dual_thrust'])
    python_strats_live_pnl = sum(sum(t.get('profit', 0) for t in live_close_by_strat.get(s, [])) for s in ['keltner', 'dual_thrust'])
    print(f"\n  Python-managed strategies (keltner + dual_thrust):")
    print(f"    Live PnL: ${python_strats_live_pnl:.2f}")
    print(f"    Backtest PnL: ${python_strats_bt_pnl:.2f}")
    if python_strats_bt_pnl != 0:
        print(f"    Execution efficiency: {python_strats_live_pnl/python_strats_bt_pnl*100:.0f}%")

    ea_strats_bt_pnl = sum(compute_stats(bt_results.get(s, []))['pnl'] for s in ['psar', 'tsmom', 'sess_bo', 'chandelier'])
    print(f"\n  MT4 EA strategies (PSAR + TSMOM + SESS_BO + Chandelier):")
    print(f"    Backtest expected PnL: ${ea_strats_bt_pnl:.2f}")
    print(f"    (No Python log available - check MT4 reports for actual)")

    elapsed = time.time() - t0
    print(f"\n  Runtime: {elapsed:.1f}s")

    # Save results
    results = {
        'experiment': 'R148 Live vs Backtest Audit',
        'period': f'{LIVE_PERIOD_START} ~ {LIVE_PERIOD_END}',
        'live_summary': {
            'total_opens': len(live_opens),
            'total_closes': len(live_closes),
            'total_pnl': round(total_live_pnl, 2),
            'by_strategy': {s: {'opens': len(live_by_strat.get(s, [])),
                                'closes': len(live_close_by_strat.get(s, [])),
                                'pnl': round(sum(t.get('profit', 0) for t in live_close_by_strat.get(s, [])), 2)}
                            for s in sorted(all_strategies)},
        },
        'backtest_summary': {s: compute_stats(bt_results.get(s, [])) for s in sorted(bt_results.keys())},
        'alignment': alignment_results,
        'comparison': comparison,
        'daily_comparison': daily_comparison,
        'runtime_sec': round(elapsed, 1),
    }
    with open(OUTPUT_DIR / "r148_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_DIR / 'r148_results.json'}")


if __name__ == '__main__':
    main()
