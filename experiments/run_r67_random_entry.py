#!/usr/bin/env python3
"""
R67 — Random Entry Benchmark
==============================
Core question: Does the strategy's alpha come from ENTRY signals, or just
from exit management + market trend?

Method:
  - Keep identical exit logic (SL/TP/Trail/Timeout) for PSAR, TSMOM, SESS_BO
  - Replace entry signals with RANDOM (coin flip at same frequency)
  - Run 1000 iterations, compute Sharpe distribution
  - If random entries also produce high Sharpe -> alpha is NOT from entries

For L8_MAX (engine-based), we skip random entry test since the engine is
too tightly coupled. We test only the 3 self-contained strategies.
"""
import sys, os, io, time, json, random
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r67_random_entry")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.30
BASE_LOT = 0.03
N_TRIALS = 1000

def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"

def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()

def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}

def _run_exits(close, high, low, atr, times, n, pos_start_indices, pos_dirs, pos_entries, pos_atrs,
               sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold, spread, lot):
    """Shared exit logic: given entry positions, run the exit rules and return trades."""
    trades = []
    pos_q = list(zip(pos_start_indices, pos_dirs, pos_entries, pos_atrs))
    pos_idx = 0
    pos = None
    for i in range(1, n):
        c = close[i]; h = high[i]; lo_v = low[i]
        if pos is None and pos_idx < len(pos_q):
            si, sd, se, sa = pos_q[pos_idx]
            if i >= si:
                pos = {'dir': sd, 'entry': se, 'bar': si, 'time': times[si], 'atr': sa}
                pos_idx += 1
        if pos is None: continue
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
        if exited: pos = None
    return trades


def trades_to_daily_sharpe(trades):
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    if len(daily) < 10: return 0.0
    arr = np.array(list(daily.values()))
    if arr.std() == 0: return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(252))


def run_random_vs_real(h1_df, strat_name, real_entry_fn, exit_params, n_trials=N_TRIALS):
    """Run real strategy once, then n_trials random entry versions."""
    df = h1_df.copy()
    if 'ATR' not in df.columns: df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)

    # Real entries
    real_trades = real_entry_fn(df)
    real_sharpe = trades_to_daily_sharpe(real_trades)
    n_real_entries = len(real_trades)
    print(f"    {strat_name} real: {n_real_entries} trades, Sharpe={real_sharpe:.2f}", flush=True)

    # Random entries: same number of trades, random direction, random timing
    # We space entries at least max_hold+2 bars apart (same constraint)
    rng = random.Random(42)
    max_hold = exit_params['max_hold']
    min_gap = max_hold + 2
    random_sharpes = []

    valid_indices = [i for i in range(14, n - max_hold - 5) if not np.isnan(atr[i]) and atr[i] >= 0.1]

    for trial in range(n_trials):
        # Pick random entry bars
        rng_indices = list(valid_indices)
        rng.shuffle(rng_indices)
        entries = []
        last_bar = -min_gap
        for idx in rng_indices:
            if idx - last_bar >= min_gap:
                entries.append(idx)
                last_bar = idx
                if len(entries) >= n_real_entries:
                    break

        pos_starts = sorted(entries)
        pos_dirs = [rng.choice(['BUY', 'SELL']) for _ in pos_starts]
        pos_entries_p = []
        pos_atrs = []
        for idx, d in zip(pos_starts, pos_dirs):
            c = close[idx]
            if d == 'BUY':
                pos_entries_p.append(c + SPREAD / 2)
            else:
                pos_entries_p.append(c - SPREAD / 2)
            pos_atrs.append(atr[idx])

        rtrades = _run_exits(close, high, low, atr, times, n,
                             pos_starts, pos_dirs, pos_entries_p, pos_atrs,
                             exit_params['sl_atr'], exit_params['tp_atr'],
                             exit_params['trail_act_atr'], exit_params['trail_dist_atr'],
                             max_hold, SPREAD, BASE_LOT)
        random_sharpes.append(trades_to_daily_sharpe(rtrades))

        if (trial + 1) % 200 == 0:
            print(f"      {trial+1}/{n_trials}...", flush=True)

    rs = np.array(random_sharpes)
    pct_above_real = float((rs >= real_sharpe).mean() * 100)
    return {
        'strategy': strat_name,
        'real_sharpe': round(real_sharpe, 4),
        'real_trades': n_real_entries,
        'random_mean': round(float(rs.mean()), 4),
        'random_std': round(float(rs.std()), 4),
        'random_p5': round(float(np.percentile(rs, 5)), 4),
        'random_p50': round(float(np.percentile(rs, 50)), 4),
        'random_p95': round(float(np.percentile(rs, 95)), 4),
        'pct_random_above_real': round(pct_above_real, 2),
        'pct_random_above_zero': round(float((rs > 0).mean() * 100), 1),
    }


def _backtest_psar(df_prepared, spread=SPREAD, lot=BASE_LOT):
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

def _backtest_tsmom(h1_df, spread=SPREAD, lot=BASE_LOT):
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

def _backtest_sess(h1_df, spread=SPREAD, lot=BASE_LOT):
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


def main():
    t0 = time.time()
    print("=" * 80)
    print("  R67: Random Entry Benchmark (1000 trials)")
    print("=" * 80)

    from backtest.runner import DataBundle
    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()
    print(f"  H1: {len(h1_df)} bars", flush=True)

    # Strategy definitions with real entry functions and exit params
    def add_psar_to_df(df):
        df = df.copy(); n = len(df)
        psar = np.zeros(n); direction = np.ones(n)
        af_start = 0.01; af_max = 0.05
        af = af_start; ep = df['High'].iloc[0]; psar[0] = df['Low'].iloc[0]
        for i in range(1, n):
            prev_psar = psar[i-1]
            if direction[i-1] == 1:
                psar[i] = prev_psar + af * (ep - prev_psar)
                psar[i] = min(psar[i], df['Low'].iloc[i-1], df['Low'].iloc[max(0,i-2)])
                if df['Low'].iloc[i] < psar[i]:
                    direction[i] = -1; psar[i] = ep; ep = df['Low'].iloc[i]; af = af_start
                else:
                    direction[i] = 1
                    if df['High'].iloc[i] > ep: ep = df['High'].iloc[i]; af = min(af+af_start, af_max)
            else:
                psar[i] = prev_psar + af * (ep - prev_psar)
                psar[i] = max(psar[i], df['High'].iloc[i-1], df['High'].iloc[max(0,i-2)])
                if df['High'].iloc[i] > psar[i]:
                    direction[i] = 1; psar[i] = ep; ep = df['High'].iloc[i]; af = af_start
                else:
                    direction[i] = -1
                    if df['Low'].iloc[i] < ep: ep = df['Low'].iloc[i]; af = min(af+af_start, af_max)
        df['PSAR_dir'] = direction; df['ATR'] = compute_atr(df)
        return df

    h1_psar = add_psar_to_df(h1_df)

    def psar_real_entry(df):
        df2 = add_psar_to_df(df)
        return _backtest_psar(df2)

    def tsmom_real_entry(df):
        return _backtest_tsmom(df)

    def sess_real_entry(df):
        return _backtest_sess(df)

    strategies = [
        ('PSAR', psar_real_entry, {'sl_atr': 2.0, 'tp_atr': 16.0, 'trail_act_atr': 0.20,
                                    'trail_dist_atr': 0.04, 'max_hold': 80}),
        ('TSMOM', tsmom_real_entry, {'sl_atr': 4.5, 'tp_atr': 6.0, 'trail_act_atr': 0.14,
                                      'trail_dist_atr': 0.025, 'max_hold': 20}),
        ('SESS_BO', sess_real_entry, {'sl_atr': 3.0, 'tp_atr': 6.0, 'trail_act_atr': 0.14,
                                       'trail_dist_atr': 0.025, 'max_hold': 20}),
    ]

    results = []
    for name, real_fn, exit_params in strategies:
        print(f"\n  === {name} ===", flush=True)
        r = run_random_vs_real(h1_df, name, real_fn, exit_params)
        results.append(r)

    elapsed = time.time() - t0

    # Summary
    lines = [
        "R67 Random Entry Benchmark — Summary",
        "=" * 80,
        f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)",
        f"Trials per strategy: {N_TRIALS}\n",
        f"{'Strategy':>10} {'Real Sh':>8} {'Rand Mean':>10} {'Rand P50':>9} {'Rand P95':>9} "
        f"{'%Rand>Real':>11} {'%Rand>0':>8}   Verdict",
        "-" * 90,
    ]
    for r in results:
        if r['pct_random_above_real'] < 5:
            verdict = "ALPHA CONFIRMED (random < 5% beats real)"
        elif r['pct_random_above_real'] < 20:
            verdict = "MODERATE ALPHA (random < 20% beats real)"
        elif r['pct_random_above_real'] < 50:
            verdict = "WEAK ALPHA (random often close to real)"
        else:
            verdict = "NO ALPHA (random >= real; edge is exit/trend only)"
        lines.append(f"{r['strategy']:>10} {r['real_sharpe']:>8.2f} {r['random_mean']:>10.2f} "
                      f"{r['random_p50']:>9.2f} {r['random_p95']:>9.2f} "
                      f"{r['pct_random_above_real']:>10.1f}% {r['pct_random_above_zero']:>7.1f}%   {verdict}")

    lines.extend([
        "",
        "Interpretation:",
        "  - If random entries also produce Sharpe > 2, the market trend (gold bull) is doing the work",
        "  - If real >> random, the entry signal has genuine predictive power",
        "  - %Rand>Real: if < 5%, entry alpha is statistically significant at 95% confidence",
    ])

    summary = "\n".join(lines)
    print(f"\n{summary}", flush=True)
    with open(OUTPUT_DIR / "r67_summary.txt", 'w', encoding='utf-8') as f: f.write(summary)
    with open(OUTPUT_DIR / "r67_results.json", 'w', encoding='utf-8') as f:
        json.dump({'n_trials': N_TRIALS, 'results': results,
                   'elapsed_s': round(elapsed, 1)}, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  R67 Complete — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
