#!/usr/bin/env python3
"""
Round 55b — Weekly & Monthly Timeframe Brute-Force Parameter Search
====================================================================
Brute-force grid search for 3 strategy types on XAUUSD:
  B1) Weekly KC Breakout
  B2) Weekly TSMOM (momentum zero-crossing with Reversal exit)
  B3) Monthly KC + TSMOM

Weekly/monthly bars are few (~570 weeks, ~132 months), so no multiprocessing
— simple loops instead.

Each strategy: Layer 1 full grid → Top 50 K-Fold 6-Fold validation

USAGE (server)
--------------
    cd /root/gold-quant-research
    nohup python3 -u experiments/run_round55b_weekly_long.py \
        > results/round55b_results/stdout.txt 2>&1 &
"""
import sys, os, io, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

_script_dir = os.path.dirname(os.path.abspath(__file__))
for _candidate in [os.path.join(_script_dir, '..'), os.path.join(_script_dir, '..', '..'), os.getcwd()]:
    _candidate = os.path.abspath(_candidate)
    if os.path.isdir(os.path.join(_candidate, 'backtest')):
        sys.path.insert(0, _candidate)
        os.chdir(_candidate)
        break

OUTPUT_DIR = Path("results/round55b_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.50

KFOLD_FOLDS = [
    ("F1_2015-2016", "2015-01-01", "2016-12-31"),
    ("F2_2017-2018", "2017-01-01", "2018-12-31"),
    ("F3_2019-2020", "2019-01-01", "2020-12-31"),
    ("F4_2021-2022", "2021-01-01", "2022-12-31"),
    ("F5_2023-2024", "2023-01-01", "2024-12-31"),
    ("F6_2025-2026", "2025-01-01", "2026-12-31"),
]


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_adx(df, period=14):
    high = df['High']; low = df['Low']; close = df['Close']
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.DataFrame({'hl': high - low, 'hc': (high - close.shift(1)).abs(),
                        'lc': (low - close.shift(1)).abs()}).max(axis=1)
    atr_s = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_s)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_s)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(period).mean()


def add_kc(df, ema_period=20, atr_period=14, mult=1.5):
    df = df.copy()
    df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    df['ATR'] = tr.rolling(atr_period).mean()
    df['KC_upper'] = df['EMA'] + mult * df['ATR']
    df['KC_lower'] = df['EMA'] - mult * df['ATR']
    df['ADX'] = compute_adx(df, atr_period)
    return df


def calc_stats(trades, label=""):
    if not trades:
        return {'label': label, 'n': 0, 'sharpe': 0, 'total_pnl': 0,
                'win_rate': 0, 'max_dd': 0, 'avg_win': 0, 'avg_loss': 0}
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    eq = np.cumsum(pnls)
    dd = (np.maximum.accumulate(eq + 2000) - (eq + 2000)).max()
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    da = np.array(list(daily.values()))
    sh = float(da.mean() / da.std() * np.sqrt(252)) if len(da) > 1 and da.std() > 0 else 0.0
    return {
        'label': label, 'n': len(pnls), 'sharpe': round(sh, 2),
        'total_pnl': round(sum(pnls), 2), 'win_rate': round(len(wins)/len(pnls)*100, 1),
        'max_dd': round(dd, 2),
        'avg_win': round(np.mean(wins), 2) if wins else 0,
        'avg_loss': round(abs(np.mean(losses)), 2) if losses else 0,
    }


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


def save_checkpoint(data, filename):
    path = OUTPUT_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, default=str)
    print(f"  [Checkpoint] {path} ({len(data) if isinstance(data, list) else 'dict'})", flush=True)


def load_checkpoint(filename):
    path = OUTPUT_DIR / filename
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def fmt(x):
    return f"${x:,.0f}" if abs(x) >= 1 else f"${x:.2f}"


def write_ranking(results, filename, title):
    valid = [r for r in results if r.get('sharpe', 0) > 0 and r.get('n', 0) > 0]
    valid.sort(key=lambda x: x['sharpe'], reverse=True)
    lines = [title, "=" * 100, f"Total: {len(valid)} positive Sharpe / {len(results)} total", ""]
    lines.append(f"{'Rank':>4} {'Label':>55} {'Sharpe':>8} {'PnL':>12} {'N':>6} "
                 f"{'MaxDD':>10} {'WR':>6} {'AvgW':>8} {'AvgL':>8}")
    lines.append(f"{'':>4} {'-'*55} {'-'*8} {'-'*12} {'-'*6} {'-'*10} {'-'*6} {'-'*8} {'-'*8}")
    for i, r in enumerate(valid[:100], 1):
        lines.append(f"{i:>4} {r['label']:>55} {r['sharpe']:>8.2f} {fmt(r['total_pnl']):>12} "
                     f"{r['n']:>6} {fmt(r['max_dd']):>10} {r['win_rate']:>5.1f}% "
                     f"{r.get('avg_win',0):>8.2f} {r.get('avg_loss',0):>8.2f}")
    with open(OUTPUT_DIR / filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Ranking saved: {filename}", flush=True)


def resample_weekly(h1_df):
    return h1_df.resample('W').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna()


def resample_monthly(h1_df):
    return h1_df.resample('ME').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna()


def add_atr(df, period=14):
    """Add ATR column to a dataframe."""
    df = df.copy()
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    df['ATR'] = tr.rolling(period).mean()
    return df


# ═══════════════════════════════════════════════════════════════
# KC Breakout Backtest (reused for weekly and monthly)
# ═══════════════════════════════════════════════════════════════

def backtest_kc(df_prepared, label, adx_thresh=18,
                sl_atr=3.5, tp_atr=8.0, trail_act_atr=0.28, trail_dist_atr=0.06,
                max_hold=20, spread=SPREAD, lot=0.03):
    df = df_prepared
    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    kc_up = df['KC_upper'].values; kc_lo = df['KC_lower'].values
    atr = df['ATR'].values; adx_arr = df['ADX'].values
    times = df.index; n = len(df); last_exit = -999

    for i in range(1, n):
        c = close[i]; h = high[i]; lo_v = low[i]
        cur_atr = atr[i]; cur_adx = adx_arr[i]
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
            tp_val = tp_atr * pos['atr'] * lot * 100
            sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); exited = True
            else:
                act_dist = trail_act_atr * pos['atr']; trail_d = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= act_dist:
                    ts_p = h - trail_d
                    if lo_v <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos['entry'] - spread) * lot * 100)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= act_dist:
                    ts_p = lo_v + trail_d
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * 100)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if exited: pos = None; last_exit = i; continue
        if pos is not None: continue
        if i - last_exit < 2: continue
        if np.isnan(cur_adx) or cur_adx < adx_thresh: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        prev_c = close[i-1]
        if prev_c > kc_up[i-1]:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif prev_c < kc_lo[i-1]:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}

    return calc_stats(trades, label)


# ═══════════════════════════════════════════════════════════════
# TSMOM Backtest (with Reversal exit, from R53)
# ═══════════════════════════════════════════════════════════════

def compute_score(close, weights):
    """Compute multi-period TSMOM score: sum of w*sign(close/close[lb]-1)."""
    n = len(close)
    max_lb = max(lb for lb, _ in weights)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        for lb, w in weights:
            if i >= lb:
                s += w * np.sign(close[i] / close[i - lb] - 1.0)
        score[i] = s
    return score


def backtest_tsmom(score, df, label, sl_atr=3.5, tp_atr=12.0,
                   trail_act_atr=0.28, trail_dist_atr=0.06, max_hold=50,
                   spread=SPREAD, lot=0.03):
    """TSMOM backtest with Reversal exit on momentum flip."""
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    atr = df['ATR'].values
    times = df.index; n = len(close)
    trades = []; pos = None; last_exit = -999

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

            tp_val = tp_atr * pos['atr'] * lot * 100
            sl_val = sl_atr * pos['atr'] * lot * 100
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

    return calc_stats(trades, label)


# ═══════════════════════════════════════════════════════════════
# Grid definitions
# ═══════════════════════════════════════════════════════════════

B1_WEEKLY_KC_GRID = {
    'ema': [5, 10, 15, 20],
    'mult': [1.0, 1.5, 2.0, 2.5, 3.0],
    'adx': [10, 14, 18, 22],
    'sl': [2.0, 3.0, 4.0, 5.0],
    'tp': [4.0, 6.0, 8.0, 12.0],
    'mh': [4, 8, 12, 20],
    'trail': [(0.20, 0.05), (0.30, 0.08), (0.50, 0.15)],
}

B2_WEEKLY_TSMOM_GRID = {
    'fast': [2, 4, 8, 13, 26],
    'slow': [8, 13, 26, 52],
    'sl': [2.0, 3.0, 4.0, 5.0],
    'tp': [4.0, 6.0, 8.0, 12.0],
    'mh': [4, 8, 13, 26],
    'trail': [(0.20, 0.05), (0.30, 0.08), (0.50, 0.15)],
}

B3_MONTHLY_KC_GRID = {
    'ema': [3, 5, 10],
    'mult': [1.0, 1.5, 2.0, 2.5],
    'adx': [10, 14, 18],
    'sl': [2.0, 3.0, 4.0],
    'tp': [4.0, 6.0, 8.0],
    'mh': [3, 6, 12],
    'trail': [(0.30, 0.08), (0.50, 0.15)],
}

B3_MONTHLY_TSMOM_GRID = {
    'fast': [2, 3, 6],
    'slow': [6, 12],
    'sl': [2.0, 3.0, 4.0],
    'tp': [4.0, 6.0, 8.0],
    'mh': [3, 6, 12],
    'trail': [(0.30, 0.08), (0.50, 0.15)],
}


# ═══════════════════════════════════════════════════════════════
# B1: Weekly KC Breakout Grid
# ═══════════════════════════════════════════════════════════════

def run_b1_weekly_kc(h1_df, checkpoint_name="b1_weekly_kc_grid.json"):
    print(f"\n{'='*80}")
    print(f"  B1: Weekly KC Breakout — Full Parameter Grid")
    print(f"{'='*80}")

    existing = load_checkpoint(checkpoint_name)
    if existing:
        print(f"  [Resume] Found {len(existing)} results, skipping", flush=True)
        return existing

    wk_df = resample_weekly(h1_df)
    print(f"  Weekly bars: {len(wk_df)} ({wk_df.index[0]} -> {wk_df.index[-1]})")

    grid = B1_WEEKLY_KC_GRID
    combos = list(product(
        grid['ema'], grid['mult'], grid['adx'],
        grid['sl'], grid['tp'], grid['mh'], grid['trail']
    ))
    total = len(combos)
    print(f"  Total combos: {total:,}", flush=True)

    t0 = time.time()
    all_results = []
    batch_size = max(50, total // 20)

    for idx, (ema, mult, adx, sl, tp, mh, (trail_a, trail_d)) in enumerate(combos):
        label = f"WK_KC_E{ema}_M{mult}_ADX{adx}_SL{sl}_TP{tp}_MH{mh}_T{trail_a}/{trail_d}"
        df = add_kc(wk_df, ema, 14, mult)
        df = df.dropna()
        r = backtest_kc(df, label, adx_thresh=adx, sl_atr=sl, tp_atr=tp,
                        trail_act_atr=trail_a, trail_dist_atr=trail_d, max_hold=mh)
        all_results.append(r)

        if (idx + 1) % batch_size == 0 or (idx + 1) == total:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total - idx - 1) / rate if rate > 0 else 0
            print(f"    {idx+1}/{total} ({(idx+1)/total*100:.0f}%) | "
                  f"{elapsed/60:.1f}min | ETA {eta/60:.1f}min", flush=True)
            save_checkpoint(all_results, checkpoint_name)

    elapsed = time.time() - t0
    valid = [r for r in all_results if r.get('sharpe', 0) > 0]
    print(f"\n  Done: {len(valid)} positive / {len(all_results)} total in {elapsed/60:.1f}min")

    save_checkpoint(all_results, checkpoint_name)
    write_ranking(all_results, "b1_weekly_kc_ranking.txt", "R55b B1 Weekly KC Grid Results")
    return all_results


# ═══════════════════════════════════════════════════════════════
# B2: Weekly TSMOM Grid
# ═══════════════════════════════════════════════════════════════

def run_b2_weekly_tsmom(h1_df, checkpoint_name="b2_weekly_tsmom_grid.json"):
    print(f"\n{'='*80}")
    print(f"  B2: Weekly TSMOM — Full Parameter Grid")
    print(f"{'='*80}")

    existing = load_checkpoint(checkpoint_name)
    if existing:
        print(f"  [Resume] Found {len(existing)} results, skipping", flush=True)
        return existing

    wk_df = resample_weekly(h1_df)
    wk_df = add_atr(wk_df, 14)
    print(f"  Weekly bars: {len(wk_df)} ({wk_df.index[0]} -> {wk_df.index[-1]})")

    grid = B2_WEEKLY_TSMOM_GRID
    window_combos = [(f, s) for f in grid['fast'] for s in grid['slow'] if f < s]
    print(f"  Window combos (fast < slow): {len(window_combos)}", flush=True)

    close_arr = wk_df['Close'].values
    precomp_scores = {}
    for fast, slow in window_combos:
        key = f"W{fast}_{slow}"
        weights = [(fast, 0.5), (slow, 0.5)]
        precomp_scores[key] = compute_score(close_arr, weights)

    exit_combos = list(product(grid['sl'], grid['tp'], grid['mh'], grid['trail']))
    total = len(window_combos) * len(exit_combos)
    print(f"  Total combos: {total:,}", flush=True)

    t0 = time.time()
    all_results = []
    batch_size = max(50, total // 20)
    done = 0

    for fast, slow in window_combos:
        key = f"W{fast}_{slow}"
        score = precomp_scores[key]
        for sl, tp, mh, (trail_a, trail_d) in exit_combos:
            label = f"WK_TS_{key}_SL{sl}_TP{tp}_MH{mh}_T{trail_a}/{trail_d}"
            r = backtest_tsmom(score, wk_df, label, sl_atr=sl, tp_atr=tp,
                               trail_act_atr=trail_a, trail_dist_atr=trail_d, max_hold=mh)
            all_results.append(r)
            done += 1

            if done % batch_size == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"    {done}/{total} ({done/total*100:.0f}%) | "
                      f"{elapsed/60:.1f}min | ETA {eta/60:.1f}min", flush=True)
                save_checkpoint(all_results, checkpoint_name)

    elapsed = time.time() - t0
    valid = [r for r in all_results if r.get('sharpe', 0) > 0]
    print(f"\n  Done: {len(valid)} positive / {len(all_results)} total in {elapsed/60:.1f}min")

    save_checkpoint(all_results, checkpoint_name)
    write_ranking(all_results, "b2_weekly_tsmom_ranking.txt", "R55b B2 Weekly TSMOM Grid Results")
    return all_results


# ═══════════════════════════════════════════════════════════════
# B3: Monthly KC + TSMOM Grid
# ═══════════════════════════════════════════════════════════════

def run_b3_monthly_kc(h1_df, checkpoint_name="b3_monthly_kc_grid.json"):
    print(f"\n{'='*80}")
    print(f"  B3a: Monthly KC Breakout — Full Parameter Grid")
    print(f"{'='*80}")

    existing = load_checkpoint(checkpoint_name)
    if existing:
        print(f"  [Resume] Found {len(existing)} results, skipping", flush=True)
        return existing

    mo_df = resample_monthly(h1_df)
    print(f"  Monthly bars: {len(mo_df)} ({mo_df.index[0]} -> {mo_df.index[-1]})")

    grid = B3_MONTHLY_KC_GRID
    combos = list(product(
        grid['ema'], grid['mult'], grid['adx'],
        grid['sl'], grid['tp'], grid['mh'], grid['trail']
    ))
    total = len(combos)
    print(f"  Total combos: {total:,}", flush=True)

    t0 = time.time()
    all_results = []
    batch_size = max(20, total // 20)

    for idx, (ema, mult, adx, sl, tp, mh, (trail_a, trail_d)) in enumerate(combos):
        label = f"MO_KC_E{ema}_M{mult}_ADX{adx}_SL{sl}_TP{tp}_MH{mh}_T{trail_a}/{trail_d}"
        df = add_kc(mo_df, ema, 14, mult)
        df = df.dropna()
        r = backtest_kc(df, label, adx_thresh=adx, sl_atr=sl, tp_atr=tp,
                        trail_act_atr=trail_a, trail_dist_atr=trail_d, max_hold=mh)
        all_results.append(r)

        if (idx + 1) % batch_size == 0 or (idx + 1) == total:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total - idx - 1) / rate if rate > 0 else 0
            print(f"    {idx+1}/{total} ({(idx+1)/total*100:.0f}%) | "
                  f"{elapsed/60:.1f}min | ETA {eta/60:.1f}min", flush=True)
            save_checkpoint(all_results, checkpoint_name)

    elapsed = time.time() - t0
    valid = [r for r in all_results if r.get('sharpe', 0) > 0]
    print(f"\n  Done: {len(valid)} positive / {len(all_results)} total in {elapsed/60:.1f}min")

    save_checkpoint(all_results, checkpoint_name)
    write_ranking(all_results, "b3_monthly_kc_ranking.txt", "R55b B3a Monthly KC Grid Results")
    return all_results


def run_b3_monthly_tsmom(h1_df, checkpoint_name="b3_monthly_tsmom_grid.json"):
    print(f"\n{'='*80}")
    print(f"  B3b: Monthly TSMOM — Full Parameter Grid")
    print(f"{'='*80}")

    existing = load_checkpoint(checkpoint_name)
    if existing:
        print(f"  [Resume] Found {len(existing)} results, skipping", flush=True)
        return existing

    mo_df = resample_monthly(h1_df)
    mo_df = add_atr(mo_df, 14)
    print(f"  Monthly bars: {len(mo_df)} ({mo_df.index[0]} -> {mo_df.index[-1]})")

    grid = B3_MONTHLY_TSMOM_GRID
    window_combos = [(f, s) for f in grid['fast'] for s in grid['slow'] if f < s]
    print(f"  Window combos (fast < slow): {len(window_combos)}", flush=True)

    close_arr = mo_df['Close'].values
    precomp_scores = {}
    for fast, slow in window_combos:
        key = f"W{fast}_{slow}"
        weights = [(fast, 0.5), (slow, 0.5)]
        precomp_scores[key] = compute_score(close_arr, weights)

    exit_combos = list(product(grid['sl'], grid['tp'], grid['mh'], grid['trail']))
    total = len(window_combos) * len(exit_combos)
    print(f"  Total combos: {total:,}", flush=True)

    t0 = time.time()
    all_results = []
    batch_size = max(10, total // 20)
    done = 0

    for fast, slow in window_combos:
        key = f"W{fast}_{slow}"
        score = precomp_scores[key]
        for sl, tp, mh, (trail_a, trail_d) in exit_combos:
            label = f"MO_TS_{key}_SL{sl}_TP{tp}_MH{mh}_T{trail_a}/{trail_d}"
            r = backtest_tsmom(score, mo_df, label, sl_atr=sl, tp_atr=tp,
                               trail_act_atr=trail_a, trail_dist_atr=trail_d, max_hold=mh)
            all_results.append(r)
            done += 1

            if done % batch_size == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"    {done}/{total} ({done/total*100:.0f}%) | "
                      f"{elapsed/60:.1f}min | ETA {eta/60:.1f}min", flush=True)
                save_checkpoint(all_results, checkpoint_name)

    elapsed = time.time() - t0
    valid = [r for r in all_results if r.get('sharpe', 0) > 0]
    print(f"\n  Done: {len(valid)} positive / {len(all_results)} total in {elapsed/60:.1f}min")

    save_checkpoint(all_results, checkpoint_name)
    write_ranking(all_results, "b3_monthly_tsmom_ranking.txt", "R55b B3b Monthly TSMOM Grid Results")
    return all_results


# ═══════════════════════════════════════════════════════════════
# K-Fold validation
# ═══════════════════════════════════════════════════════════════

def _parse_kc_params(label):
    p = {}
    parts = label.split('_')
    for part in parts:
        if part.startswith('E') and not part.startswith('EMA'):
            try: p['ema'] = int(part[1:])
            except: pass
        elif part.startswith('M') and not part.startswith('MH') and not part.startswith('MO'):
            try: p['mult'] = float(part[1:])
            except: pass
        elif part.startswith('ADX'):
            try: p['adx'] = int(part[3:])
            except: pass
        elif part.startswith('SL'):
            try: p['sl'] = float(part[2:])
            except: pass
        elif part.startswith('TP'):
            try: p['tp'] = float(part[2:])
            except: pass
        elif part.startswith('MH'):
            try: p['mh'] = int(part[2:])
            except: pass
        elif '/' in part and part.startswith('T'):
            try:
                a, d = part[1:].split('/')
                p['trail_a'] = float(a); p['trail_d'] = float(d)
            except: pass
    p.setdefault('ema', 20); p.setdefault('mult', 1.5); p.setdefault('adx', 18)
    p.setdefault('trail_a', 0.28); p.setdefault('trail_d', 0.06)
    p.setdefault('mh', 20); p.setdefault('sl', 3.5); p.setdefault('tp', 8.0)
    return p


def _parse_tsmom_params(label):
    p = {}
    parts = label.split('_')
    for j, part in enumerate(parts):
        if part.startswith('W'):
            try: p['fast'] = int(part[1:])
            except: pass
        elif j >= 1 and 'fast' in p and 'slow' not in p:
            try: p['slow'] = int(part)
            except: pass
        elif part.startswith('SL'):
            try: p['sl'] = float(part[2:])
            except: pass
        elif part.startswith('TP'):
            try: p['tp'] = float(part[2:])
            except: pass
        elif part.startswith('MH'):
            try: p['mh'] = int(part[2:])
            except: pass
        elif '/' in part and part.startswith('T'):
            try:
                a, d = part[1:].split('/')
                p['trail_a'] = float(a); p['trail_d'] = float(d)
            except: pass
    p.setdefault('fast', 4); p.setdefault('slow', 26)
    p.setdefault('trail_a', 0.28); p.setdefault('trail_d', 0.06)
    p.setdefault('mh', 13); p.setdefault('sl', 3.5); p.setdefault('tp', 8.0)
    return p


def run_kfold_kc(h1_df, results, resample_fn, tf_label, ckpt_name, top_n=50):
    print(f"\n{'='*80}")
    print(f"  K-Fold 6-Fold: Top {top_n} {tf_label} KC")
    print(f"{'='*80}")

    existing = load_checkpoint(ckpt_name)
    if existing:
        print(f"  [Resume] Found checkpoint, skipping", flush=True)
        return existing

    valid = [r for r in results if r.get('sharpe', 0) > 0 and r.get('n', 0) > 0]
    valid.sort(key=lambda x: x['sharpe'], reverse=True)
    candidates = valid[:top_n]
    print(f"  Candidates: {len(candidates)}")

    kfold_results = []
    for ci, cand in enumerate(candidates):
        label = cand['label']
        params = _parse_kc_params(label)
        fold_sharpes = []

        for fname, start, end in KFOLD_FOLDS:
            fold_h1 = h1_df[start:end]
            if len(fold_h1) < 100:
                continue
            fold_df = resample_fn(fold_h1)
            if len(fold_df) < 5:
                continue
            fold_df = add_kc(fold_df, params['ema'], 14, params['mult'])
            fold_df = fold_df.dropna()
            if len(fold_df) < 3:
                continue
            r = backtest_kc(fold_df, f"{fname}_{label}", adx_thresh=params['adx'],
                            sl_atr=params['sl'], tp_atr=params['tp'],
                            trail_act_atr=params['trail_a'], trail_dist_atr=params['trail_d'],
                            max_hold=params['mh'])
            fold_sharpes.append(r['sharpe'])

        passed = (len(fold_sharpes) == 6
                  and all(s > 0 for s in fold_sharpes)
                  and np.mean(fold_sharpes) > 1.0)
        kfold_results.append({
            'label': label, 'full_sharpe': cand['sharpe'],
            'kfold_mean': round(np.mean(fold_sharpes), 2) if fold_sharpes else 0,
            'kfold_min': round(min(fold_sharpes), 2) if fold_sharpes else 0,
            'kfold_folds': [round(s, 2) for s in fold_sharpes],
            'passed': passed,
        })

        p_str = "PASS" if passed else "FAIL"
        print(f"  [{ci+1}/{len(candidates)}] {label[:55]:>55} "
              f"Full={cand['sharpe']:.2f} KF={np.mean(fold_sharpes):.2f} {p_str}", flush=True)

    n_passed = sum(1 for k in kfold_results if k['passed'])
    print(f"\n  Passed: {n_passed}/{len(kfold_results)}")

    save_checkpoint(kfold_results, ckpt_name)
    return kfold_results


def run_kfold_tsmom(h1_df, results, resample_fn, tf_label, ckpt_name, top_n=50):
    print(f"\n{'='*80}")
    print(f"  K-Fold 6-Fold: Top {top_n} {tf_label} TSMOM")
    print(f"{'='*80}")

    existing = load_checkpoint(ckpt_name)
    if existing:
        print(f"  [Resume] Found checkpoint, skipping", flush=True)
        return existing

    valid = [r for r in results if r.get('sharpe', 0) > 0 and r.get('n', 0) > 0]
    valid.sort(key=lambda x: x['sharpe'], reverse=True)
    candidates = valid[:top_n]
    print(f"  Candidates: {len(candidates)}")

    kfold_results = []
    for ci, cand in enumerate(candidates):
        label = cand['label']
        params = _parse_tsmom_params(label)
        weights = [(params['fast'], 0.5), (params['slow'], 0.5)]
        fold_sharpes = []

        for fname, start, end in KFOLD_FOLDS:
            fold_h1 = h1_df[start:end]
            if len(fold_h1) < 100:
                continue
            fold_df = resample_fn(fold_h1)
            fold_df = add_atr(fold_df, 14)
            if len(fold_df) < 5:
                continue
            score = compute_score(fold_df['Close'].values, weights)
            r = backtest_tsmom(score, fold_df, f"{fname}_{label}",
                               sl_atr=params['sl'], tp_atr=params['tp'],
                               trail_act_atr=params['trail_a'],
                               trail_dist_atr=params['trail_d'],
                               max_hold=params['mh'])
            fold_sharpes.append(r['sharpe'])

        passed = (len(fold_sharpes) == 6
                  and all(s > 0 for s in fold_sharpes)
                  and np.mean(fold_sharpes) > 1.0)
        kfold_results.append({
            'label': label, 'full_sharpe': cand['sharpe'],
            'kfold_mean': round(np.mean(fold_sharpes), 2) if fold_sharpes else 0,
            'kfold_min': round(min(fold_sharpes), 2) if fold_sharpes else 0,
            'kfold_folds': [round(s, 2) for s in fold_sharpes],
            'passed': passed,
        })

        p_str = "PASS" if passed else "FAIL"
        print(f"  [{ci+1}/{len(candidates)}] {label[:55]:>55} "
              f"Full={cand['sharpe']:.2f} KF={np.mean(fold_sharpes):.2f} {p_str}", flush=True)

    n_passed = sum(1 for k in kfold_results if k['passed'])
    print(f"\n  Passed: {n_passed}/{len(kfold_results)}")

    save_checkpoint(kfold_results, ckpt_name)
    return kfold_results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0_total = time.time()

    print("=" * 80)
    print("  R55b: Weekly & Monthly Timeframe Brute-Force Parameter Search")
    print(f"  Spread: ${SPREAD}")
    print("  No multiprocessing (few bars per timeframe)")
    print("=" * 80)

    print("\n  Loading H1 data...")
    from backtest.runner import DataBundle
    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} -> {h1_df.index[-1]})")

    wk_sample = resample_weekly(h1_df)
    mo_sample = resample_monthly(h1_df)
    print(f"  Weekly:  {len(wk_sample)} bars")
    print(f"  Monthly: {len(mo_sample)} bars")

    g1 = B1_WEEKLY_KC_GRID
    b1_combos = (len(g1['ema']) * len(g1['mult']) * len(g1['adx']) *
                 len(g1['sl']) * len(g1['tp']) * len(g1['mh']) * len(g1['trail']))

    g2 = B2_WEEKLY_TSMOM_GRID
    b2_windows = len([(f, s) for f in g2['fast'] for s in g2['slow'] if f < s])
    b2_exit = len(g2['sl']) * len(g2['tp']) * len(g2['mh']) * len(g2['trail'])
    b2_combos = b2_windows * b2_exit

    g3k = B3_MONTHLY_KC_GRID
    b3k_combos = (len(g3k['ema']) * len(g3k['mult']) * len(g3k['adx']) *
                  len(g3k['sl']) * len(g3k['tp']) * len(g3k['mh']) * len(g3k['trail']))

    g3t = B3_MONTHLY_TSMOM_GRID
    b3t_windows = len([(f, s) for f in g3t['fast'] for s in g3t['slow'] if f < s])
    b3t_exit = len(g3t['sl']) * len(g3t['tp']) * len(g3t['mh']) * len(g3t['trail'])
    b3t_combos = b3t_windows * b3t_exit

    grand_total = b1_combos + b2_combos + b3k_combos + b3t_combos
    print(f"\n  Grid sizes: B1 WK_KC={b1_combos:,} | B2 WK_TS={b2_combos:,} | "
          f"B3a MO_KC={b3k_combos:,} | B3b MO_TS={b3t_combos:,} | Total={grand_total:,}")

    # --- B1: Weekly KC ---
    b1_results = run_b1_weekly_kc(h1_df)
    b1_kfold = run_kfold_kc(h1_df, b1_results, resample_weekly,
                            "Weekly", "kfold_weekly_kc.json")

    # --- B2: Weekly TSMOM ---
    b2_results = run_b2_weekly_tsmom(h1_df)
    b2_kfold = run_kfold_tsmom(h1_df, b2_results, resample_weekly,
                               "Weekly", "kfold_weekly_tsmom.json")

    # --- B3a: Monthly KC ---
    b3k_results = run_b3_monthly_kc(h1_df)
    b3k_kfold = run_kfold_kc(h1_df, b3k_results, resample_monthly,
                             "Monthly", "kfold_monthly_kc.json")

    # --- B3b: Monthly TSMOM ---
    b3t_results = run_b3_monthly_tsmom(h1_df)
    b3t_kfold = run_kfold_tsmom(h1_df, b3t_results, resample_monthly,
                                "Monthly", "kfold_monthly_tsmom.json")

    # --- Final Summary ---
    elapsed_total = time.time() - t0_total
    print(f"\n{'='*80}")
    print(f"  R55b COMPLETE — {elapsed_total/60:.1f}min total")
    print(f"{'='*80}")

    for name, kf in [("B1 WK KC", b1_kfold), ("B2 WK TSMOM", b2_kfold),
                      ("B3a MO KC", b3k_kfold), ("B3b MO TSMOM", b3t_kfold)]:
        n_pass = sum(1 for k in kf if k.get('passed'))
        n_total = len(kf)
        best = max(kf, key=lambda x: x.get('kfold_mean', 0)) if kf else {}
        print(f"  {name:>14}: {n_pass}/{n_total} passed K-Fold | "
              f"Best: {best.get('label','N/A')[:45]} KF_mean={best.get('kfold_mean',0):.2f}")

    print(f"\n  Results in: {OUTPUT_DIR}")
