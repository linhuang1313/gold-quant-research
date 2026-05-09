#!/usr/bin/env python3
"""
R153 — Rolling Correlation Monitor (6 Strategies)
===================================================
Analyzes rolling correlation between all 6 trading strategies over time.

Phases:
  1. Run all 6 strategies at unit lot (0.01), get trade lists
  2. Convert to daily PnL series for each strategy
  3. Full-sample 6x6 correlation matrix
  4. Rolling 90-day window correlation analysis
  5. Yearly correlation breakdown
"""
import sys, os, time, json, warnings, glob
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r153_correlation_monitor")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01

STRAT_NAMES = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO', 'DUAL_THRUST', 'CHANDELIER']

t0 = time.time()


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_h1():
    candidates = sorted(glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    if not candidates:
        raise FileNotFoundError("No xauusd H1 CSV found")
    csv_path = candidates[-1]
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.set_index('time')[['open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[df['Close'] > 100]
    return df


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


def _run_exit_with_cap(pos, i, hi, lo, cl, spread, lot, times,
                       sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap):
    atr = pos['atr']
    sl = atr * sl_atr; tp = atr * tp_atr; bars = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_now = (cl - pos['entry'] - spread) * lot * PV
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if lo <= pos['entry'] - sl:
            return _mk(pos, pos['entry'] - sl, times[i], "SL", i, -sl * lot * PV)
        if hi >= pos['entry'] + tp:
            return _mk(pos, pos['entry'] + tp, times[i], "TP", i, tp * lot * PV)
        extreme = pos.get('extreme', pos['entry']); extreme = max(extreme, hi); pos['extreme'] = extreme
        if extreme - pos['entry'] >= atr * trail_act:
            trail_price = extreme - atr * trail_dist
            if cl <= trail_price:
                return _mk(pos, trail_price, times[i], "Trail", i, (trail_price - pos['entry'] - spread) * lot * PV)
        if bars >= max_hold:
            return _mk(pos, cl, times[i], "TimeExit", i, pnl_now)
    else:
        pnl_now = (pos['entry'] - cl - spread) * lot * PV
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if hi >= pos['entry'] + sl:
            return _mk(pos, pos['entry'] + sl, times[i], "SL", i, -sl * lot * PV)
        if lo <= pos['entry'] - tp:
            return _mk(pos, pos['entry'] - tp, times[i], "TP", i, tp * lot * PV)
        extreme = pos.get('extreme', pos['entry']); extreme = min(extreme, lo); pos['extreme'] = extreme
        if pos['entry'] - extreme >= atr * trail_act:
            trail_price = extreme + atr * trail_dist
            if cl >= trail_price:
                return _mk(pos, trail_price, times[i], "Trail", i, (pos['entry'] - trail_price - spread) * lot * PV)
        if bars >= max_hold:
            return _mk(pos, cl, times[i], "TimeExit", i, pnl_now)
    return None


# ═══════════════════════════════════════════════════════════════
# Strategy backtests
# ═══════════════════════════════════════════════════════════════

def bt_l8_max(h1_df, spread, lot):
    """L8_MAX (Keltner) via DataBundle + run_variant."""
    from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    kw = dict(LIVE_PARITY_KWARGS)
    kw['maxloss_cap'] = 35
    stats = run_variant(data, "L8_MAX_corr", verbose=False, spread_cost=spread,
                        initial_capital=2000, min_lot_size=lot, max_lot_size=lot, **kw)
    trades_raw = stats.get('_trades', [])
    trades = []
    for t in trades_raw:
        if isinstance(t, dict):
            trades.append(t)
        else:
            trades.append({
                'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
                'entry_time': t.entry_time, 'exit_time': t.exit_time,
                'pnl': t.pnl, 'reason': t.exit_reason, 'bars': t.bars_held,
            })
    return trades


def bt_psar(h1_df, spread, lot, cap=5):
    df = h1_df.copy(); add_psar(df); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR', 'PSAR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; psar = df['PSAR'].values; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, times,
                                        4.5, 16.0, 0.20, 0.04, 20, cap)
            if result: trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if psar[i-1] > c[i-1] and psar[i] < c[i]:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif psar[i-1] < c[i-1] and psar[i] > c[i]:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom(h1_df, spread, lot, cap=0):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
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
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, times,
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


def bt_sess_bo(h1_df, spread, lot, cap=35):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour; times = df.index; n = len(df)
    session_hour, lookback = 12, 4
    trades = []; pos = None; last_exit = -999
    for i in range(lookback+1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, times,
                                        4.5, 4.0, 0.14, 0.025, 20, cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if hours[i] != session_hour: continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_dual_thrust(h1_df, spread, lot, cap=35):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    n_bars, k = 6, 0.5
    hh = df['High'].rolling(n_bars).max()
    lc = df['Close'].rolling(n_bars).min()
    hc = df['Close'].rolling(n_bars).max()
    ll = df['Low'].rolling(n_bars).min()
    dt_range = pd.concat([hh - lc, hc - ll], axis=1).max(axis=1)
    daily_open = df.groupby(df.index.date)['Open'].transform('first')
    sig = pd.Series(0, index=df.index)
    sig[df['Close'] > daily_open + k * dt_range] = 1
    sig[df['Close'] < daily_open - k * dt_range] = -1

    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr_arr = df['ATR'].values; times = df.index; n = len(df)
    sig_arr = sig.values

    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, times,
                                        4.5, 8.0, 0.14, 0.025, 20, cap)
            if result: trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr_arr[i]) or atr_arr[i] < 0.1: continue
        if sig_arr[i] == 1 and sig_arr[i-1] != 1:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr_arr[i]}
        elif sig_arr[i] == -1 and sig_arr[i-1] != -1:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr_arr[i]}
    return trades


def bt_chandelier(h1_df, spread, lot, cap=35):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    period, mult, ema_period = 22, 3.0, 100
    atr14 = (df['High'] - df['Low']).rolling(14).mean()
    hh = df['High'].rolling(period).max()
    ll_roll = df['Low'].rolling(period).min()
    chand_long = hh - mult * atr14
    chand_short = ll_roll + mult * atr14
    ema100 = df['Close'].ewm(span=ema_period).mean()

    above_long = df['Close'] > chand_long
    flip_bull = above_long & (~above_long.shift(1).fillna(False))
    below_short = df['Close'] < chand_short
    flip_bear = below_short & (~below_short.shift(1).fillna(False))

    sig = pd.Series(0, index=df.index)
    sig[flip_bull & (df['Close'] > ema100)] = 1
    sig[flip_bear & (df['Close'] < ema100)] = -1

    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr_arr = df['ATR'].values; times = df.index; n = len(df)
    sig_arr = sig.values

    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, times,
                                        4.5, 8.0, 0.14, 0.025, 20, cap)
            if result: trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr_arr[i]) or atr_arr[i] < 0.1: continue
        if sig_arr[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr_arr[i]}
        elif sig_arr[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr_arr[i]}
    return trades


# ═══════════════════════════════════════════════════════════════
# Analysis helpers
# ═══════════════════════════════════════════════════════════════

def trades_to_daily_pnl(trades, all_dates):
    """Convert trade list to daily PnL series aligned to all_dates."""
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    return pd.Series([daily.get(d, 0.0) for d in all_dates], index=all_dates)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80, flush=True)
    print("  R153 — Rolling Correlation Monitor (6 Strategies)", flush=True)
    print("=" * 80, flush=True)

    h1_df = load_h1()
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})", flush=True)

    results = {}

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Run all 6 strategies
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 1: Running all 6 strategies at unit lot", flush=True)
    print("=" * 70, flush=True)

    all_trades = {}

    print("  Running L8_MAX (Keltner via engine)...", flush=True)
    all_trades['L8_MAX'] = bt_l8_max(h1_df, SPREAD, UNIT_LOT)
    print(f"    L8_MAX: {len(all_trades['L8_MAX'])} trades", flush=True)

    print("  Running PSAR...", flush=True)
    all_trades['PSAR'] = bt_psar(h1_df, SPREAD, UNIT_LOT, cap=5)
    print(f"    PSAR: {len(all_trades['PSAR'])} trades", flush=True)

    print("  Running TSMOM...", flush=True)
    all_trades['TSMOM'] = bt_tsmom(h1_df, SPREAD, UNIT_LOT, cap=0)
    print(f"    TSMOM: {len(all_trades['TSMOM'])} trades", flush=True)

    print("  Running SESS_BO...", flush=True)
    all_trades['SESS_BO'] = bt_sess_bo(h1_df, SPREAD, UNIT_LOT, cap=35)
    print(f"    SESS_BO: {len(all_trades['SESS_BO'])} trades", flush=True)

    print("  Running DUAL_THRUST...", flush=True)
    all_trades['DUAL_THRUST'] = bt_dual_thrust(h1_df, SPREAD, UNIT_LOT, cap=35)
    print(f"    DUAL_THRUST: {len(all_trades['DUAL_THRUST'])} trades", flush=True)

    print("  Running CHANDELIER...", flush=True)
    all_trades['CHANDELIER'] = bt_chandelier(h1_df, SPREAD, UNIT_LOT, cap=35)
    print(f"    CHANDELIER: {len(all_trades['CHANDELIER'])} trades", flush=True)

    results['phase1_trade_counts'] = {k: len(v) for k, v in all_trades.items()}

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Convert to daily PnL series
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: Converting to daily PnL series", flush=True)
    print("=" * 70, flush=True)

    all_exit_dates = set()
    for trades in all_trades.values():
        for t in trades:
            all_exit_dates.add(pd.Timestamp(t['exit_time']).date())
    all_dates = sorted(all_exit_dates)
    print(f"  Date range: {all_dates[0]} ~ {all_dates[-1]} ({len(all_dates)} trading days)", flush=True)

    daily_pnl = {}
    for sname in STRAT_NAMES:
        daily_pnl[sname] = trades_to_daily_pnl(all_trades[sname], all_dates)
        total = daily_pnl[sname].sum()
        active_days = (daily_pnl[sname] != 0).sum()
        print(f"    {sname:<14}: total PnL=${total:8.2f}  active days={active_days}", flush=True)

    results['phase2_daily_stats'] = {
        sname: {
            'total_pnl': round(float(daily_pnl[sname].sum()), 2),
            'active_days': int((daily_pnl[sname] != 0).sum()),
            'total_days': len(all_dates),
        }
        for sname in STRAT_NAMES
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Full-sample 6x6 correlation matrix
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3: Full-sample 6x6 correlation matrix", flush=True)
    print("=" * 70, flush=True)

    pnl_df = pd.DataFrame(daily_pnl)
    corr_matrix = pnl_df.corr()

    print(f"\n  {'':>14}", end="", flush=True)
    for s in STRAT_NAMES:
        print(f" {s[:8]:>8}", end="", flush=True)
    print("", flush=True)

    corr_dict = {}
    for s1 in STRAT_NAMES:
        row_str = f"  {s1:<14}"
        for s2 in STRAT_NAMES:
            val = corr_matrix.loc[s1, s2]
            row_str += f" {val:8.3f}"
        print(row_str, flush=True)
        corr_dict[s1] = {s2: round(float(corr_matrix.loc[s1, s2]), 4) for s2 in STRAT_NAMES}

    results['phase3_correlation_matrix'] = corr_dict

    high_corr_pairs = []
    for i, s1 in enumerate(STRAT_NAMES):
        for j, s2 in enumerate(STRAT_NAMES):
            if j <= i: continue
            val = corr_matrix.loc[s1, s2]
            if abs(val) > 0.30:
                high_corr_pairs.append({'pair': f"{s1}/{s2}", 'correlation': round(float(val), 4)})

    if high_corr_pairs:
        print(f"\n  WARNING: {len(high_corr_pairs)} pair(s) exceed |0.30| correlation:", flush=True)
        for p in high_corr_pairs:
            print(f"    {p['pair']}: {p['correlation']:.4f}", flush=True)
    else:
        print("\n  All pairs below |0.30| correlation threshold.", flush=True)

    results['phase3_high_corr_pairs'] = high_corr_pairs

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Rolling 90-day correlation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 4: Rolling 90-day window correlation analysis", flush=True)
    print("=" * 70, flush=True)

    window = 90
    rolling_stats = {}
    flagged_pairs = []

    print(f"\n  {'Pair':<22} {'Avg':>7} {'Max':>7} {'Min':>7} {'Flag':>5}", flush=True)
    print(f"  {'='*50}", flush=True)

    for i, s1 in enumerate(STRAT_NAMES):
        for j, s2 in enumerate(STRAT_NAMES):
            if j <= i: continue
            pair_name = f"{s1}/{s2}"
            rolling_corr = pnl_df[s1].rolling(window).corr(pnl_df[s2])
            rolling_corr = rolling_corr.dropna()

            if len(rolling_corr) == 0:
                continue

            avg_corr = float(rolling_corr.mean())
            max_corr = float(rolling_corr.max())
            min_corr = float(rolling_corr.min())
            exceeds = bool(max_corr > 0.30 or min_corr < -0.30)
            flag_str = " ***" if exceeds else ""

            print(f"  {pair_name:<22} {avg_corr:7.3f} {max_corr:7.3f} {min_corr:7.3f}{flag_str}", flush=True)

            pair_stats = {
                'avg': round(avg_corr, 4),
                'max': round(max_corr, 4),
                'min': round(min_corr, 4),
                'exceeds_030': exceeds,
            }
            rolling_stats[pair_name] = pair_stats

            if exceeds:
                flagged_pairs.append({'pair': pair_name, 'max': round(max_corr, 4), 'min': round(min_corr, 4)})

    print(f"\n  Flagged pairs (ever exceed |0.30|): {len(flagged_pairs)}/{len(rolling_stats)}", flush=True)
    for fp in flagged_pairs:
        print(f"    {fp['pair']}: max={fp['max']:.4f}, min={fp['min']:.4f}", flush=True)

    results['phase4_rolling_90d'] = rolling_stats
    results['phase4_flagged_pairs'] = flagged_pairs

    # ═══════════════════════════════════════════════════════════
    # Phase 5: Yearly correlation breakdown
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 5: Yearly correlation breakdown", flush=True)
    print("=" * 70, flush=True)

    pnl_df_indexed = pnl_df.copy()
    pnl_df_indexed.index = pd.to_datetime(all_dates)
    years = sorted(pnl_df_indexed.index.year.unique())

    yearly_corr = {}
    for year in years:
        year_data = pnl_df_indexed[pnl_df_indexed.index.year == year]
        if len(year_data) < 30:
            continue
        year_corr = year_data.corr()

        print(f"\n  --- {year} ({len(year_data)} days) ---", flush=True)
        print(f"  {'':>14}", end="", flush=True)
        for s in STRAT_NAMES:
            print(f" {s[:8]:>8}", end="", flush=True)
        print("", flush=True)

        year_dict = {}
        for s1 in STRAT_NAMES:
            row_str = f"  {s1:<14}"
            for s2 in STRAT_NAMES:
                val = year_corr.loc[s1, s2] if s1 in year_corr.index and s2 in year_corr.columns else 0
                row_str += f" {val:8.3f}"
            print(row_str, flush=True)
            year_dict[s1] = {s2: round(float(year_corr.loc[s1, s2]), 4)
                             if s1 in year_corr.index and s2 in year_corr.columns else 0
                             for s2 in STRAT_NAMES}

        yearly_corr[str(year)] = year_dict

    results['phase5_yearly_correlation'] = yearly_corr

    # Trend analysis: how correlations evolve
    print(f"\n  Correlation evolution (selected pairs):", flush=True)
    print(f"  {'Pair':<22}", end="", flush=True)
    for y in years:
        if str(y) in yearly_corr:
            print(f" {y:>6}", end="", flush=True)
    print("", flush=True)

    evolution = {}
    for i, s1 in enumerate(STRAT_NAMES):
        for j, s2 in enumerate(STRAT_NAMES):
            if j <= i: continue
            pair_name = f"{s1}/{s2}"
            row_str = f"  {pair_name:<22}"
            pair_vals = []
            for y in years:
                ystr = str(y)
                if ystr in yearly_corr and s1 in yearly_corr[ystr] and s2 in yearly_corr[ystr][s1]:
                    val = yearly_corr[ystr][s1][s2]
                    row_str += f" {val:6.3f}"
                    pair_vals.append(val)
                else:
                    row_str += f"    N/A"
            print(row_str, flush=True)
            evolution[pair_name] = pair_vals

    results['phase5_evolution'] = {k: [round(v, 4) for v in vals] for k, vals in evolution.items()}

    # ═══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    print("\n" + "=" * 80, flush=True)
    print("  R153 FINAL SUMMARY", flush=True)
    print("=" * 80, flush=True)

    avg_full_corr = []
    for i, s1 in enumerate(STRAT_NAMES):
        for j, s2 in enumerate(STRAT_NAMES):
            if j <= i: continue
            avg_full_corr.append(corr_matrix.loc[s1, s2])

    print(f"\n  Portfolio diversification metrics:", flush=True)
    print(f"    Average pairwise correlation: {np.mean(avg_full_corr):.4f}", flush=True)
    print(f"    Max pairwise correlation:     {np.max(avg_full_corr):.4f}", flush=True)
    print(f"    Min pairwise correlation:     {np.min(avg_full_corr):.4f}", flush=True)
    print(f"    Pairs exceeding |0.30| (full sample): {len(high_corr_pairs)}/15", flush=True)
    print(f"    Pairs ever exceeding |0.30| (90d rolling): {len(flagged_pairs)}/15", flush=True)

    results['summary'] = {
        'avg_pairwise_corr': round(float(np.mean(avg_full_corr)), 4),
        'max_pairwise_corr': round(float(np.max(avg_full_corr)), 4),
        'min_pairwise_corr': round(float(np.min(avg_full_corr)), 4),
        'full_sample_high_corr_pairs': len(high_corr_pairs),
        'rolling_flagged_pairs': len(flagged_pairs),
        'total_pairs': 15,
    }

    results['elapsed_s'] = round(elapsed, 1)
    out_file = OUTPUT_DIR / "r153_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"  Saved: {out_file}", flush=True)
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
