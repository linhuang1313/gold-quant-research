#!/usr/bin/env python3
"""
R65 — Spread & Slippage Stress Test (Anti-Overfitting)
======================================================
Calibrated with REAL slippage data from live trading:
  - Avg slippage: 0.38 pts (29 trades)
  - Max adverse: 3.19 pts
  - 69% positive (unfavorable), 31% negative (favorable)

Test 1: Spread sensitivity (0.30 to 5.00)
Test 2: Slippage additive on top of spread=0.50
Test 3: Worst-case combined (spread=1.50 + slippage=0.50)
Test 4: Realistic scenario using actual slippage distribution
"""
import sys, os, io, time, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r65_spread_stress")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_LOT = 0.03
PORTFOLIO = {'l8': 0.01, 'psar': 0.02, 'ts': 0.02, 'sb': 0.02}
STRAT_NAMES = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']
STRAT_KEYS  = ['l8', 'psar', 'ts', 'sb']

SPREAD_LEVELS = [0.30, 0.50, 0.68, 0.88, 1.00, 1.30, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00]
SLIPPAGE_LEVELS = [0, 0.10, 0.20, 0.38, 0.50, 0.70, 1.00, 1.50, 2.00, 3.19]

def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"

# ═══════════════════════════════════════════════════════════════
# Backtest helpers (from R61)
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()

def add_psar(df, af_start=0.02, af_max=0.20):
    df = df.copy()
    n = len(df); psar = np.zeros(n); direction = np.ones(n)
    af = af_start; ep = df['High'].iloc[0]; psar[0] = df['Low'].iloc[0]
    for i in range(1, n):
        prev_psar = psar[i-1]
        if direction[i-1] == 1:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = min(psar[i], df['Low'].iloc[i-1], df['Low'].iloc[max(0, i-2)])
            if df['Low'].iloc[i] < psar[i]:
                direction[i] = -1; psar[i] = ep; ep = df['Low'].iloc[i]; af = af_start
            else:
                direction[i] = 1
                if df['High'].iloc[i] > ep: ep = df['High'].iloc[i]; af = min(af + af_start, af_max)
        else:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = max(psar[i], df['High'].iloc[i-1], df['High'].iloc[max(0, i-2)])
            if df['High'].iloc[i] > psar[i]:
                direction[i] = 1; psar[i] = ep; ep = df['High'].iloc[i]; af = af_start
            else:
                direction[i] = -1
                if df['Low'].iloc[i] < ep: ep = df['Low'].iloc[i]; af = min(af + af_start, af_max)
    df['PSAR'] = psar; df['PSAR_dir'] = direction
    tr = pd.DataFrame({'hl': df['High']-df['Low'], 'hc': (df['High']-df['Close'].shift(1)).abs(),
                        'lc': (df['Low']-df['Close'].shift(1)).abs()}).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    return df

def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}

def backtest_psar_trades(df_prepared, sl_atr=2.0, tp_atr=16.0,
                         trail_act_atr=0.20, trail_dist_atr=0.04, max_hold=80,
                         spread=0.30, lot=BASE_LOT):
    df = df_prepared.dropna(subset=['PSAR_dir', 'ATR'])
    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    psar_dir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df); last_exit = -999
    for i in range(1, n):
        c = close[i]; h = high[i]; lo_v = low[i]; cur_atr = atr[i]
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (h - pos['entry'] - spread) * lot * 100
                pnl_l = (lo_v - pos['entry'] - spread) * lot * 100
                pnl_c = (c - pos['entry'] - spread) * lot * 100
            else:
                pnl_h = (pos['entry'] - lo_v - spread) * lot * 100
                pnl_l = (pos['entry'] - h - spread) * lot * 100
                pnl_c = (pos['entry'] - c - spread) * lot * 100
            tp_val = tp_atr * pos['atr'] * lot * 100; sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); exited = True
            else:
                ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
                    ts_p = h - td
                    if lo_v <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos['entry'] - spread) * lot * 100)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
                    ts_p = lo_v + td
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * 100)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if exited: pos = None; last_exit = i; continue
        if pos is not None: continue
        if i - last_exit < 2: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        prev_d = psar_dir[i-1]; cur_d = psar_dir[i]
        if prev_d == -1 and cur_d == 1:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif prev_d == 1 and cur_d == -1:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades

def backtest_tsmom_trades(h1_df, fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
                          trail_act_atr=0.14, trail_dist_atr=0.025, max_hold=20,
                          spread=0.30, lot=BASE_LOT):
    df = h1_df.copy()
    if 'ATR' not in df.columns: df['ATR'] = compute_atr(df)
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(close)
    max_lb = max(fast, slow)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        for lb, w in [(fast, 0.5), (slow, 0.5)]:
            if i >= lb: s += w * np.sign(close[i] / close[i - lb] - 1.0)
        score[i] = s
    trades = []; pos = None; last_exit = -999
    for i in range(max_lb + 1, n):
        c = close[i]; h = high[i]; lo_v = low[i]; cur_atr = atr[i]
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (h - pos['entry'] - spread) * lot * 100
                pnl_l = (lo_v - pos['entry'] - spread) * lot * 100
                pnl_c = (c - pos['entry'] - spread) * lot * 100
            else:
                pnl_h = (pos['entry'] - lo_v - spread) * lot * 100
                pnl_l = (pos['entry'] - h - spread) * lot * 100
                pnl_c = (pos['entry'] - c - spread) * lot * 100
            tp_val = tp_atr * pos['atr'] * lot * 100; sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); exited = True
            else:
                ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
                    ts_p = h - td
                    if lo_v <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos['entry'] - spread) * lot * 100)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
                    ts_p = lo_v + td
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * 100)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if not exited and not np.isnan(score[i]):
                if pos['dir'] == 'BUY' and score[i] < 0:
                    trades.append(_mk(pos, c, times[i], "Reversal", i, pnl_c)); exited = True
                elif pos['dir'] == 'SELL' and score[i] > 0:
                    trades.append(_mk(pos, c, times[i], "Reversal", i, pnl_c)); exited = True
            if exited: pos = None; last_exit = i; continue
        if pos is not None or i - last_exit < 2: continue
        if np.isnan(score[i]) or np.isnan(cur_atr) or cur_atr < 0.1: continue
        if score[i] > 0 and score[i - 1] <= 0:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif score[i] < 0 and score[i - 1] >= 0:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades

def backtest_session_trades(h1_df, session="peak_12_14", lookback_bars=3,
                            sl_atr=3.0, tp_atr=6.0, trail_act_atr=0.14,
                            trail_dist_atr=0.025, max_hold=20, spread=0.30, lot=BASE_LOT):
    SESSION_DEFS = {"peak_12_14": (12, 14)}
    df = h1_df.copy(); df['ATR'] = compute_atr(df); df = df.dropna(subset=['ATR'])
    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour
    times = df.index; n = len(df); last_exit = -999
    sess_start = SESSION_DEFS[session][0]
    for i in range(lookback_bars, n):
        c = close[i]; h = high[i]; lo_v = low[i]; cur_atr = atr[i]; cur_hour = hours[i]
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (h - pos['entry'] - spread) * lot * 100
                pnl_l = (lo_v - pos['entry'] - spread) * lot * 100
                pnl_c = (c - pos['entry'] - spread) * lot * 100
            else:
                pnl_h = (pos['entry'] - lo_v - spread) * lot * 100
                pnl_l = (pos['entry'] - h - spread) * lot * 100
                pnl_c = (pos['entry'] - c - spread) * lot * 100
            tp_val = tp_atr * pos['atr'] * lot * 100; sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); exited = True
            else:
                ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
                    ts_p = h - td
                    if lo_v <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos['entry'] - spread) * lot * 100)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
                    ts_p = lo_v + td
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * 100)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if exited: pos = None; last_exit = i; continue
        if pos is not None: continue
        if i - last_exit < 2: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        if cur_hour != sess_start: continue
        if i > 0 and hours[i-1] == sess_start: continue
        rh = max(high[i-lookback_bars:i]); rl = min(low[i-lookback_bars:i])
        if c > rh:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif c < rl:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades

def trades_to_daily_pnl(trades):
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    if not daily: return pd.Series(dtype=float)
    return pd.Series(daily).sort_index()

def _run_l8_max(m15_df, h1_df, lot=BASE_LOT, maxloss_cap=37, spread=0.30):
    from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS
    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': maxloss_cap, 'spread_cost': spread,
          'initial_capital': 2000, 'min_lot_size': lot, 'max_lot_size': lot}
    data = DataBundle(m15_df, h1_df)
    result = run_variant(data, "L8_MAX", verbose=False, **kw)
    raw = result.get('_trades', [])
    return [{'pnl': (t.pnl if hasattr(t,'pnl') else t.get('pnl',0)),
             'exit_time': (t.exit_time if hasattr(t,'exit_time') else t.get('exit_time',''))}
            for t in raw]

def run_at_spread(h1_df, m15_df, spread):
    daily = {}
    daily['L8_MAX'] = trades_to_daily_pnl(_run_l8_max(m15_df, h1_df, spread=spread))
    h1_psar = add_psar(h1_df.copy(), 0.01, 0.05)
    daily['PSAR'] = trades_to_daily_pnl(
        backtest_psar_trades(h1_psar, sl_atr=2.0, tp_atr=16.0, trail_act_atr=0.20,
                             trail_dist_atr=0.04, max_hold=80, spread=spread))
    daily['TSMOM'] = trades_to_daily_pnl(
        backtest_tsmom_trades(h1_df, fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
                              trail_act_atr=0.14, trail_dist_atr=0.025, max_hold=20, spread=spread))
    daily['SESS_BO'] = trades_to_daily_pnl(
        backtest_session_trades(h1_df, session="peak_12_14", lookback_bars=3,
                                 sl_atr=3.0, tp_atr=6.0, trail_act_atr=0.14,
                                 trail_dist_atr=0.025, max_hold=20, spread=spread))
    all_dates = set()
    for s in daily.values(): all_dates.update(s.index.tolist())
    if not all_dates: return {'sharpe': 0, 'total_pnl': 0, 'max_dd': 0}, {}
    all_dates = sorted(all_dates); idx = pd.Index(all_dates)
    combined = np.zeros(len(idx))
    strat_stats = {}
    for name, key in zip(STRAT_NAMES, STRAT_KEYS):
        lot_val = PORTFOLIO.get(key, 0)
        if lot_val > 0:
            arr = daily[name].reindex(idx, fill_value=0.0).values * (lot_val / BASE_LOT)
            combined += arr
            eq_s = np.cumsum(arr)
            dd_s = float((np.maximum.accumulate(eq_s) - eq_s).max()) if len(eq_s) > 0 else 0
            std_s = arr.std()
            sh_s = float(arr.mean() / std_s * np.sqrt(252)) if std_s > 0 else 0
            strat_stats[name] = {'sharpe': round(sh_s, 3), 'pnl': round(float(arr.sum()), 2),
                                 'max_dd': round(dd_s, 2)}
    eq = np.cumsum(combined)
    dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 0 else 0
    std = combined.std()
    sharpe = float(combined.mean() / std * np.sqrt(252)) if std > 0 else 0
    n_trades = 0
    for name in STRAT_NAMES:
        n_trades += len(daily.get(name, []))
    return {'sharpe': round(sharpe, 4), 'total_pnl': round(float(combined.sum()), 2),
            'max_dd': round(dd, 2), 'n_days': len(combined)}, strat_stats


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80)
    print("  R65: Spread & Slippage Stress Test")
    print("  Calibrated with real slippage: avg=0.38 pts, max=3.19 pts")
    print("=" * 80)

    from backtest.runner import DataBundle
    data = DataBundle.load_default()
    h1_df = data.h1_df.copy(); m15_df = data.m15_df.copy()
    print(f"  H1: {len(h1_df)} bars | M15: {len(m15_df)} bars", flush=True)

    # Pre-compute PSAR once
    h1_psar_base = add_psar(h1_df.copy(), 0.01, 0.05)

    # ═══════════════════════════════════════════════════════
    # Test 1: Spread Sensitivity
    # ═══════════════════════════════════════════════════════
    print(f"\n  [Test 1] Spread Sensitivity ({len(SPREAD_LEVELS)} levels)...", flush=True)
    t1_results = []
    break_even_spread = None
    for sp in SPREAD_LEVELS:
        print(f"    Spread={sp:.2f}...", end="", flush=True)
        stats, strat_s = run_at_spread(h1_df, m15_df, sp)
        t1_results.append({'spread': sp, **stats, 'per_strategy': strat_s})
        print(f" Sharpe={stats['sharpe']:.2f} PnL={fmt(stats['total_pnl'])}", flush=True)
        if break_even_spread is None and stats['sharpe'] < 1.0:
            break_even_spread = sp

    # Test 2: Slippage additive (base spread=0.50)
    print(f"\n  [Test 2] Slippage Sensitivity (base spread=0.50, {len(SLIPPAGE_LEVELS)} levels)...", flush=True)
    t2_results = []
    for slip in SLIPPAGE_LEVELS:
        eff_spread = 0.50 + slip
        print(f"    Slippage={slip:.2f} (eff={eff_spread:.2f})...", end="", flush=True)
        stats, _ = run_at_spread(h1_df, m15_df, eff_spread)
        t2_results.append({'slippage': slip, 'effective_spread': eff_spread, **stats})
        print(f" Sharpe={stats['sharpe']:.2f}", flush=True)

    # Test 3: Worst-case
    print(f"\n  [Test 3] Worst-case (spread=1.50 + slippage=0.50 = 2.00)...", flush=True)
    wc_stats, wc_strat = run_at_spread(h1_df, m15_df, 2.00)
    print(f"    Sharpe={wc_stats['sharpe']:.2f} PnL={fmt(wc_stats['total_pnl'])} DD={fmt(wc_stats['max_dd'])}", flush=True)

    # Test 4: Realistic scenario (spread=0.50 + avg slippage=0.38)
    print(f"\n  [Test 4] Realistic (spread=0.50 + avg_slip=0.38 = 0.88)...", flush=True)
    real_stats, real_strat = run_at_spread(h1_df, m15_df, 0.88)
    print(f"    Sharpe={real_stats['sharpe']:.2f} PnL={fmt(real_stats['total_pnl'])} DD={fmt(real_stats['max_dd'])}", flush=True)

    elapsed = time.time() - t0

    # Summary
    lines = [
        "R65 Spread & Slippage Stress Test — Summary",
        "=" * 80,
        f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)",
        f"Portfolio: L8=0.01, PSAR=0.02, TSMOM=0.02, SESS_BO=0.02 (total=0.07)\n",
        "--- Test 1: Spread Sensitivity ---",
        f"{'Spread':>8} {'Sharpe':>8} {'PnL':>12} {'MaxDD':>10}",
        "-" * 42,
    ]
    for r in t1_results:
        marker = " <<<" if abs(r['spread'] - 0.30) < 0.01 else ""
        marker = " <<< BASELINE" if abs(r['spread'] - 0.30) < 0.01 else marker
        marker = " <<< REALISTIC" if abs(r['spread'] - 0.88) < 0.01 else marker
        lines.append(f"{r['spread']:>8.2f} {r['sharpe']:>8.2f} {fmt(r['total_pnl']):>12} {fmt(r['max_dd']):>10}{marker}")

    if break_even_spread:
        lines.append(f"\n  Break-even spread (Sharpe < 1.0): ~{break_even_spread:.2f}")
    else:
        lines.append(f"\n  Break-even spread: > {SPREAD_LEVELS[-1]:.2f} (strategy survives all levels!)")

    lines.extend([
        "",
        "--- Test 2: Slippage Sensitivity (base spread=0.50) ---",
        f"{'Slippage':>10} {'Eff Spread':>12} {'Sharpe':>8} {'PnL':>12} {'MaxDD':>10}",
        "-" * 56,
    ])
    for r in t2_results:
        marker = " <<< YOUR AVG" if abs(r['slippage'] - 0.38) < 0.01 else ""
        marker = " <<< YOUR MAX" if abs(r['slippage'] - 3.19) < 0.01 else marker
        lines.append(f"{r['slippage']:>10.2f} {r['effective_spread']:>12.2f} "
                     f"{r['sharpe']:>8.2f} {fmt(r['total_pnl']):>12} {fmt(r['max_dd']):>10}{marker}")

    lines.extend([
        "",
        "--- Test 3: Worst-Case (spread=1.50 + slippage=0.50) ---",
        f"  Portfolio Sharpe: {wc_stats['sharpe']:.2f}",
        f"  Portfolio PnL:    {fmt(wc_stats['total_pnl'])}",
        f"  Portfolio MaxDD:  {fmt(wc_stats['max_dd'])}",
    ])
    for name, st in wc_strat.items():
        lines.append(f"    {name:>10}: Sharpe={st['sharpe']:.2f} PnL={fmt(st['pnl'])} DD={fmt(st['max_dd'])}")

    lines.extend([
        "",
        "--- Test 4: Realistic (spread=0.50 + avg_slippage=0.38 = 0.88) ---",
        f"  Portfolio Sharpe: {real_stats['sharpe']:.2f}",
        f"  Portfolio PnL:    {fmt(real_stats['total_pnl'])}",
        f"  Portfolio MaxDD:  {fmt(real_stats['max_dd'])}",
    ])
    for name, st in real_strat.items():
        lines.append(f"    {name:>10}: Sharpe={st['sharpe']:.2f} PnL={fmt(st['pnl'])} DD={fmt(st['max_dd'])}")

    summary = "\n".join(lines)
    print(f"\n{summary}", flush=True)
    with open(OUTPUT_DIR / "r65_summary.txt", 'w', encoding='utf-8') as f: f.write(summary)
    with open(OUTPUT_DIR / "r65_results.json", 'w', encoding='utf-8') as f:
        json.dump({'test1_spread': t1_results, 'test2_slippage': t2_results,
                   'test3_worst_case': wc_stats, 'test3_per_strategy': {k: v for k,v in wc_strat.items()},
                   'test4_realistic': real_stats, 'test4_per_strategy': {k: v for k,v in real_strat.items()},
                   'break_even_spread': break_even_spread,
                   'elapsed_s': round(elapsed, 1)}, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  R65 Complete — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
