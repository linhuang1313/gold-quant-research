#!/usr/bin/env python3
"""
R66 — Era Bias Diagnostic
==========================
Core question: Does the strategy only work because gold went up?

Tests:
  1. Per-year Sharpe for each strategy and the portfolio
  2. Buy-and-Hold gold benchmark for the same periods
  3. BUY-only vs SELL-only PnL split (= R69 combined here)
  4. Rolling 2-year Sharpe stability
"""
import sys, os, io, time, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r66_era_bias")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.30
BASE_LOT = 0.03
PORTFOLIO = {'l8': 0.01, 'psar': 0.02, 'ts': 0.02, 'sb': 0.02}
STRAT_NAMES = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']
STRAT_KEYS  = ['l8', 'psar', 'ts', 'sb']

def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"

# ═══════════════════════════════════════════════════════════════
# Backtest helpers (self-contained, from R63/R64)
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

def backtest_psar_trades(df_prepared, spread=SPREAD, lot=BASE_LOT):
    df = df_prepared.dropna(subset=['PSAR_dir', 'ATR'])
    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    psar_dir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df); last_exit = -999
    sl_atr=2.0; tp_atr=16.0; trail_act_atr=0.20; trail_dist_atr=0.04; max_hold=80
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
        if pos is None and i - last_exit >= 2 and not np.isnan(cur_atr) and cur_atr >= 0.1:
            prev_d = psar_dir[i-1]; cur_d = psar_dir[i]
            if prev_d == -1 and cur_d == 1:
                pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
            elif prev_d == 1 and cur_d == -1:
                pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades

def backtest_tsmom_trades(h1_df, spread=SPREAD, lot=BASE_LOT):
    df = h1_df.copy()
    if 'ATR' not in df.columns: df['ATR'] = compute_atr(df)
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(close)
    fast=480; slow=720; sl_atr=4.5; tp_atr=6.0; trail_act_atr=0.14; trail_dist_atr=0.025; max_hold=20
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

def backtest_session_trades(h1_df, spread=SPREAD, lot=BASE_LOT):
    df = h1_df.copy(); df['ATR'] = compute_atr(df); df = df.dropna(subset=['ATR'])
    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour
    times = df.index; n = len(df); last_exit = -999
    lookback_bars=3; sl_atr=3.0; tp_atr=6.0; trail_act_atr=0.14; trail_dist_atr=0.025; max_hold=20
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
        if pos is None and i - last_exit >= 2 and not np.isnan(cur_atr) and cur_atr >= 0.1:
            if cur_hour != 12: continue
            if i > 0 and hours[i-1] == 12: continue
            rh = max(high[i-lookback_bars:i]); rl = min(low[i-lookback_bars:i])
            if c > rh:
                pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
            elif c < rl:
                pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades

def _run_l8_max(m15_df, h1_df, lot=BASE_LOT, spread=SPREAD):
    from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS
    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': 37, 'spread_cost': spread,
          'initial_capital': 2000, 'min_lot_size': lot, 'max_lot_size': lot}
    data = DataBundle(m15_df, h1_df)
    result = run_variant(data, "L8_MAX", verbose=False, **kw)
    raw = result.get('_trades', [])
    trades = []
    for t in raw:
        pnl = t.pnl if hasattr(t, 'pnl') else t.get('pnl', 0)
        ext = t.exit_time if hasattr(t, 'exit_time') else t.get('exit_time', '')
        ent = t.entry_time if hasattr(t, 'entry_time') else t.get('entry_time', '')
        d = t.direction if hasattr(t, 'direction') else t.get('direction', '')
        trades.append({'pnl': pnl, 'exit_time': ext, 'entry_time': ent, 'dir': d})
    return trades

def sharpe_from_daily(daily_arr):
    if len(daily_arr) < 10 or daily_arr.std() == 0: return 0.0
    return float(daily_arr.mean() / daily_arr.std() * np.sqrt(252))


def main():
    t0 = time.time()
    print("=" * 80)
    print("  R66: Era Bias Diagnostic + Direction Bias")
    print("=" * 80)

    from backtest.runner import DataBundle
    data = DataBundle.load_default()
    h1_df = data.h1_df.copy(); m15_df = data.m15_df.copy()
    print(f"  H1: {len(h1_df)} bars | M15: {len(m15_df)} bars", flush=True)

    # Run all strategies on full data and collect individual trades
    print("\n  Running all strategies...", flush=True)
    l8_trades = _run_l8_max(m15_df, h1_df)
    h1_psar = add_psar(h1_df.copy(), 0.01, 0.05)
    psar_trades = backtest_psar_trades(h1_psar)
    tsmom_trades = backtest_tsmom_trades(h1_df)
    sess_trades = backtest_session_trades(h1_df)
    all_strat_trades = {'L8_MAX': l8_trades, 'PSAR': psar_trades,
                        'TSMOM': tsmom_trades, 'SESS_BO': sess_trades}
    for name, tr in all_strat_trades.items():
        print(f"    {name}: {len(tr)} trades", flush=True)

    # ═══════════════════════════════════════════════════════
    # Test 1: Per-year Sharpe
    # ═══════════════════════════════════════════════════════
    print("\n  [Test 1] Per-Year Sharpe...", flush=True)
    years = list(range(2015, 2027))
    yearly_results = []

    for year in years:
        yr_start = f"{year}-01-01"; yr_end = f"{year}-12-31"
        yr_data = {}
        for name, key in zip(STRAT_NAMES, STRAT_KEYS):
            trades = all_strat_trades[name]
            lot_val = PORTFOLIO.get(key, 0)
            if lot_val == 0: continue
            daily = {}
            for t in trades:
                d = pd.Timestamp(t['exit_time']).date()
                if d.year != year: continue
                daily[d] = daily.get(d, 0) + t['pnl'] * (lot_val / BASE_LOT)
            yr_data[name] = daily

        all_dates = set()
        for d in yr_data.values(): all_dates.update(d.keys())
        if not all_dates:
            yearly_results.append({'year': year, 'sharpe': 0, 'pnl': 0, 'n_days': 0})
            continue
        all_dates = sorted(all_dates)
        combined = np.zeros(len(all_dates))
        for idx, dt in enumerate(all_dates):
            for name in yr_data:
                combined[idx] += yr_data[name].get(dt, 0)
        sh = sharpe_from_daily(combined)
        yearly_results.append({'year': year, 'sharpe': round(sh, 2),
                               'pnl': round(float(combined.sum()), 2),
                               'n_days': len(combined)})

    # ═══════════════════════════════════════════════════════
    # Test 2: Buy & Hold Benchmark
    # ═══════════════════════════════════════════════════════
    print("  [Test 2] Buy & Hold Gold Benchmark...", flush=True)
    h1_close = h1_df['Close']
    bnh_yearly = []
    for year in years:
        mask = h1_close.index.year == year
        yr_close = h1_close[mask]
        if len(yr_close) < 20:
            bnh_yearly.append({'year': year, 'return_pct': 0, 'sharpe': 0}); continue
        daily_ret = yr_close.resample('D').last().dropna().pct_change().dropna()
        sh = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
        ret_pct = float((yr_close.iloc[-1] / yr_close.iloc[0] - 1) * 100)
        bnh_yearly.append({'year': year, 'return_pct': round(ret_pct, 1), 'sharpe': round(sh, 2)})

    # ═══════════════════════════════════════════════════════
    # Test 3: Direction Bias (BUY vs SELL)
    # ═══════════════════════════════════════════════════════
    print("  [Test 3] Direction Bias (BUY vs SELL)...", flush=True)
    dir_results = {}
    for name, key in zip(STRAT_NAMES, STRAT_KEYS):
        trades = all_strat_trades[name]
        lot_val = PORTFOLIO.get(key, 0)
        if lot_val == 0: continue
        buy_pnl = sum(t['pnl'] * (lot_val / BASE_LOT) for t in trades if t.get('dir', '') in ('BUY', 'buy'))
        sell_pnl = sum(t['pnl'] * (lot_val / BASE_LOT) for t in trades if t.get('dir', '') in ('SELL', 'sell'))
        n_buy = sum(1 for t in trades if t.get('dir', '') in ('BUY', 'buy'))
        n_sell = sum(1 for t in trades if t.get('dir', '') in ('SELL', 'sell'))

        buy_daily = {}; sell_daily = {}
        for t in trades:
            d = pd.Timestamp(t['exit_time']).date()
            pnl_scaled = t['pnl'] * (lot_val / BASE_LOT)
            if t.get('dir', '') in ('BUY', 'buy'):
                buy_daily[d] = buy_daily.get(d, 0) + pnl_scaled
            elif t.get('dir', '') in ('SELL', 'sell'):
                sell_daily[d] = sell_daily.get(d, 0) + pnl_scaled

        buy_arr = np.array(list(buy_daily.values())) if buy_daily else np.array([0])
        sell_arr = np.array(list(sell_daily.values())) if sell_daily else np.array([0])

        dir_results[name] = {
            'n_buy': n_buy, 'n_sell': n_sell,
            'buy_pnl': round(buy_pnl, 2), 'sell_pnl': round(sell_pnl, 2),
            'buy_sharpe': round(sharpe_from_daily(buy_arr), 2),
            'sell_sharpe': round(sharpe_from_daily(sell_arr), 2),
            'buy_pct': round(buy_pnl / (buy_pnl + sell_pnl) * 100, 1) if (buy_pnl + sell_pnl) != 0 else 0,
        }

    # ═══════════════════════════════════════════════════════
    # Test 4: Rolling 2-year Sharpe
    # ═══════════════════════════════════════════════════════
    print("  [Test 4] Rolling 2-Year Sharpe...", flush=True)
    rolling_results = []
    for start_year in range(2015, 2025):
        end_year = start_year + 1
        period = f"{start_year}-{end_year}"
        combined_daily = {}
        for name, key in zip(STRAT_NAMES, STRAT_KEYS):
            lot_val = PORTFOLIO.get(key, 0)
            if lot_val == 0: continue
            for t in all_strat_trades[name]:
                d = pd.Timestamp(t['exit_time']).date()
                if d.year < start_year or d.year > end_year: continue
                combined_daily[d] = combined_daily.get(d, 0) + t['pnl'] * (lot_val / BASE_LOT)
        if combined_daily:
            arr = np.array([combined_daily[k] for k in sorted(combined_daily.keys())])
            sh = sharpe_from_daily(arr)
        else:
            sh = 0
        rolling_results.append({'period': period, 'sharpe': round(sh, 2), 'n_days': len(combined_daily)})

    elapsed = time.time() - t0

    # Summary
    lines = [
        "R66 Era Bias Diagnostic — Summary",
        "=" * 80,
        f"Total time: {elapsed:.0f}s\n",
        "--- Test 1: Per-Year Portfolio Sharpe ---",
        f"{'Year':>6} {'Sharpe':>8} {'PnL':>12} {'Days':>6}  {'B&H Ret%':>10} {'B&H Sharpe':>10}",
        "-" * 60,
    ]
    for yr, bnh in zip(yearly_results, bnh_yearly):
        marker = ""
        if yr['sharpe'] < 1.0 and yr['n_days'] > 50: marker = " <<< WEAK"
        if yr['sharpe'] < 0: marker = " <<< NEGATIVE"
        lines.append(f"{yr['year']:>6} {yr['sharpe']:>8.2f} {fmt(yr['pnl']):>12} {yr['n_days']:>6}  "
                      f"{bnh['return_pct']:>9.1f}% {bnh['sharpe']:>10.2f}{marker}")

    early = [y for y in yearly_results if y['year'] <= 2020 and y['n_days'] > 50]
    late = [y for y in yearly_results if y['year'] >= 2021 and y['n_days'] > 50]
    avg_early = np.mean([y['sharpe'] for y in early]) if early else 0
    avg_late = np.mean([y['sharpe'] for y in late]) if late else 0
    lines.append(f"\n  Avg Sharpe 2015-2020: {avg_early:.2f}")
    lines.append(f"  Avg Sharpe 2021-2026: {avg_late:.2f}")
    if avg_late > avg_early * 2 and avg_early < 2:
        lines.append("  WARNING: Large era gap — strategy may depend on recent bull market")
    elif avg_early > 2:
        lines.append("  OK: Strategy shows strength in both eras")

    lines.extend([
        "",
        "--- Test 3: Direction Bias (BUY vs SELL) ---",
        f"{'Strategy':>10} {'N_Buy':>7} {'N_Sell':>7} {'Buy PnL':>12} {'Sell PnL':>12} {'Buy Sh':>8} {'Sell Sh':>8} {'Buy%':>6}",
        "-" * 80,
    ])
    for name in STRAT_NAMES:
        if name not in dir_results: continue
        d = dir_results[name]
        warn = " <<< BUY-HEAVY" if d['buy_pct'] > 80 else ""
        warn = " <<< SELL-HEAVY" if d['buy_pct'] < 20 else warn
        lines.append(f"{name:>10} {d['n_buy']:>7} {d['n_sell']:>7} {fmt(d['buy_pnl']):>12} "
                      f"{fmt(d['sell_pnl']):>12} {d['buy_sharpe']:>8.2f} {d['sell_sharpe']:>8.2f} "
                      f"{d['buy_pct']:>5.1f}%{warn}")

    lines.extend([
        "",
        "--- Test 4: Rolling 2-Year Sharpe ---",
        f"{'Period':>12} {'Sharpe':>8} {'Days':>6}",
        "-" * 30,
    ])
    for r in rolling_results:
        marker = " <<< WEAK" if r['sharpe'] < 1.0 and r['n_days'] > 100 else ""
        lines.append(f"{r['period']:>12} {r['sharpe']:>8.2f} {r['n_days']:>6}{marker}")

    summary = "\n".join(lines)
    print(f"\n{summary}", flush=True)
    with open(OUTPUT_DIR / "r66_summary.txt", 'w', encoding='utf-8') as f: f.write(summary)
    with open(OUTPUT_DIR / "r66_results.json", 'w', encoding='utf-8') as f:
        json.dump({'yearly': yearly_results, 'bnh': bnh_yearly,
                   'direction': dir_results, 'rolling_2yr': rolling_results,
                   'avg_sharpe_early': round(avg_early, 2),
                   'avg_sharpe_late': round(avg_late, 2),
                   'elapsed_s': round(elapsed, 1)}, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  R66 Complete — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
