#!/usr/bin/env python3
"""
Round 53 — TSMOM Full Parameter Brute-Force Search
=====================================================
Exhaustive grid search over momentum lookback windows + exit parameters.
Then Top 50 → K-Fold 6-Fold validation.

USAGE (server)
--------------
    cd /root/gold-quant-research
    nohup python3 -u experiments/run_round53_tsmom_brute.py \
        > results/round53_results/stdout.txt 2>&1 &
"""
import sys, os, io, time, json, traceback
import multiprocessing as mp
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

OUTPUT_DIR = Path("results/round53_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = max(1, mp.cpu_count() - 1)
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
# Shared helpers (from R51)
# ═══════════════════════════════════════════════════════════════

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
            'reason': reason, 'bars': bar_idx - pos['bar'], 'pnl': pnl}


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


# ═══════════════════════════════════════════════════════════════
# TSMOM Backtest
# ═══════════════════════════════════════════════════════════════

def backtest_tsmom(score, df, label, sl_atr=3.5, tp_atr=12.0,
                   trail_act_atr=0.28, trail_dist_atr=0.06, max_hold=50,
                   spread=SPREAD, lot=0.03):
    """TSMOM backtest using pre-computed momentum score array."""
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

            # Reversal exit: momentum flipped
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


def compute_score(close, weights):
    """Compute multi-period TSMOM score."""
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


# ═══════════════════════════════════════════════════════════════
# Grid definition
# ═══════════════════════════════════════════════════════════════

# Lookback windows in H1 bars
FAST_WINDOWS = [24, 48, 120, 240, 480]       # 1d, 2d, 5d, 10d, 20d
SLOW_WINDOWS = [240, 480, 720, 1440, 2520]   # 10d, 20d, 30d, 60d, 105d(~6mo)

TSMOM_GRID = {
    'sl': [2.0, 3.0, 3.5, 4.0, 4.5],
    'tp': [6.0, 8.0, 12.0, 16.0],
    'mh': [10, 20, 30, 50, 80],
    'trail': [(0.14, 0.025), (0.20, 0.04), (0.28, 0.06), (0.40, 0.10)],
}


# ═══════════════════════════════════════════════════════════════
# Multiprocessing worker
# ═══════════════════════════════════════════════════════════════

def _worker_tsmom(args):
    df_path, score_path, label, sl, tp, mh, trail_a, trail_d = args
    df = pd.read_pickle(df_path)
    score = np.load(score_path)
    return backtest_tsmom(score, df, label, sl_atr=sl, tp_atr=tp,
                          trail_act_atr=trail_a, trail_dist_atr=trail_d, max_hold=mh)


# ═══════════════════════════════════════════════════════════════
# Main grid runner
# ═══════════════════════════════════════════════════════════════

def run_tsmom_grid(h1_df, checkpoint_name="tsmom_grid.json"):
    print(f"\n{'='*80}")
    print(f"  H1 TSMOM — Full Parameter Grid")
    print(f"{'='*80}")

    existing = load_checkpoint(checkpoint_name)
    if existing:
        print(f"  [Resume] Found {len(existing)} results, skipping", flush=True)
        return existing

    # Pre-compute ATR
    tr = pd.DataFrame({
        'hl': h1_df['High'] - h1_df['Low'],
        'hc': (h1_df['High'] - h1_df['Close'].shift(1)).abs(),
        'lc': (h1_df['Low'] - h1_df['Close'].shift(1)).abs(),
    }).max(axis=1)
    h1_df = h1_df.copy()
    h1_df['ATR'] = tr.rolling(14).mean()

    # Save base df for workers
    tmp_df_path = OUTPUT_DIR / "_tmp_h1_tsmom.pkl"
    h1_df.to_pickle(tmp_df_path)

    # Pre-compute score arrays for each (fast, slow) combo
    # Filter: fast < slow
    close = h1_df['Close'].values
    window_combos = [(f, s) for f in FAST_WINDOWS for s in SLOW_WINDOWS if f < s]
    print(f"  Window combos (fast < slow): {len(window_combos)}", flush=True)

    precomp_scores = {}
    for idx, (fast, slow) in enumerate(window_combos):
        key = f"W{fast}_{slow}"
        weights = [(fast, 0.5), (slow, 0.5)]
        score = compute_score(close, weights)
        score_path = OUTPUT_DIR / f"_tmp_score_{key}.npy"
        np.save(score_path, score)
        precomp_scores[key] = str(score_path)
        print(f"    [{idx+1}/{len(window_combos)}] {key} done", flush=True)

    print(f"  Score pre-computation done!", flush=True)

    # Build task list
    tasks = []
    for fast, slow in window_combos:
        key = f"W{fast}_{slow}"
        for sl, tp, mh in product(TSMOM_GRID['sl'], TSMOM_GRID['tp'], TSMOM_GRID['mh']):
            for trail_a, trail_d in TSMOM_GRID['trail']:
                label = f"TS_{key}_SL{sl}_TP{tp}_MH{mh}_T{trail_a}/{trail_d}"
                tasks.append((str(tmp_df_path), precomp_scores[key],
                              label, sl, tp, mh, trail_a, trail_d))

    total = len(tasks)
    print(f"  Total combos: {total:,}", flush=True)
    print(f"  Workers: {MAX_WORKERS}", flush=True)

    t0 = time.time()
    all_results = []
    batch_size = max(100, total // 20)

    with mp.Pool(MAX_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker_tsmom, tasks, chunksize=4)):
            all_results.append(result)
            if (i + 1) % batch_size == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"    {i+1}/{total} ({(i+1)/total*100:.0f}%) | "
                      f"{elapsed/60:.1f}min | ETA {eta/60:.1f}min", flush=True)
                save_checkpoint(all_results, checkpoint_name)

    # Cleanup temp files
    tmp_df_path.unlink(missing_ok=True)
    for p in precomp_scores.values():
        Path(p).unlink(missing_ok=True)

    elapsed = time.time() - t0
    valid = [r for r in all_results if r.get('sharpe', 0) > 0]
    print(f"\n  Done: {len(valid)} positive / {len(all_results)} total in {elapsed/60:.1f}min")

    save_checkpoint(all_results, checkpoint_name)
    write_ranking(all_results, "tsmom_grid_ranking.txt", "R53 H1 TSMOM Grid Results")
    return all_results


# ═══════════════════════════════════════════════════════════════
# K-Fold validation
# ═══════════════════════════════════════════════════════════════

def _parse_tsmom_params(label):
    """Parse label like TS_W480_1440_SL3.5_TP12.0_MH50_T0.28/0.06"""
    p = {}
    parts = label.split('_')
    for j, part in enumerate(parts):
        if part.startswith('W') and j == 1:
            try: p['fast'] = int(part[1:])
            except: pass
        elif j == 2 and 'fast' in p:
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
    p.setdefault('fast', 480); p.setdefault('slow', 1440)
    p.setdefault('trail_a', 0.28); p.setdefault('trail_d', 0.06)
    p.setdefault('mh', 50); p.setdefault('sl', 3.5); p.setdefault('tp', 12.0)
    return p


def run_kfold_top(h1_df, results, top_n=50):
    print(f"\n{'='*80}")
    print(f"  K-Fold 6-Fold: Top {top_n} TSMOM")
    print(f"{'='*80}")

    ckpt = "kfold_tsmom.json"
    existing = load_checkpoint(ckpt)
    if existing:
        print(f"  [Resume] Found checkpoint, skipping", flush=True)
        return existing

    # ATR for full df
    h1c = h1_df.copy()
    if 'ATR' not in h1c.columns:
        tr = pd.DataFrame({
            'hl': h1c['High'] - h1c['Low'],
            'hc': (h1c['High'] - h1c['Close'].shift(1)).abs(),
            'lc': (h1c['Low'] - h1c['Close'].shift(1)).abs(),
        }).max(axis=1)
        h1c['ATR'] = tr.rolling(14).mean()

    valid = [r for r in results if r.get('sharpe', 0) > 0 and r.get('n', 0) > 10]
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
            fold_h1 = h1c[start:end]
            if len(fold_h1) < 100:
                continue
            score = compute_score(fold_h1['Close'].values, weights)
            r = backtest_tsmom(score, fold_h1, f"{fname}_{label}",
                               sl_atr=params['sl'], tp_atr=params['tp'],
                               trail_act_atr=params['trail_a'],
                               trail_dist_atr=params['trail_d'],
                               max_hold=params['mh'])
            fold_sharpes.append(r['sharpe'])

        passed = (len(fold_sharpes) == 6
                  and all(s > 0 for s in fold_sharpes)
                  and np.mean(fold_sharpes) > 1.5)
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

    save_checkpoint(kfold_results, ckpt)
    return kfold_results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0_total = time.time()

    print("=" * 80)
    print("  R53: TSMOM Full Parameter Brute-Force Search")
    print(f"  Spread: ${SPREAD}")
    print(f"  Workers: {MAX_WORKERS}")
    print("=" * 80)

    print("\n  Loading H1 data...")
    from backtest.runner import DataBundle
    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} -> {h1_df.index[-1]})")

    n_windows = len([(f, s) for f in FAST_WINDOWS for s in SLOW_WINDOWS if f < s])
    n_exit = (len(TSMOM_GRID['sl']) * len(TSMOM_GRID['tp']) *
              len(TSMOM_GRID['mh']) * len(TSMOM_GRID['trail']))
    total = n_windows * n_exit
    print(f"\n  Window combos: {n_windows}")
    print(f"  Exit combos per window: {n_exit}")
    print(f"  Total combos: {total:,}")

    # --- Grid search ---
    results = run_tsmom_grid(h1_df)

    # --- K-Fold Top 50 ---
    kfold = run_kfold_top(h1_df, results)

    # --- Final summary ---
    elapsed_total = time.time() - t0_total
    print(f"\n{'='*80}")
    print(f"  R53 COMPLETE — {elapsed_total/60:.1f}min total")
    print(f"{'='*80}")

    n_pass = sum(1 for k in kfold if k.get('passed'))
    n_total = len(kfold)
    best = max(kfold, key=lambda x: x.get('kfold_mean', 0)) if kfold else {}
    print(f"  TSMOM: {n_pass}/{n_total} passed K-Fold | "
          f"Best: {best.get('label','N/A')[:55]} KF_mean={best.get('kfold_mean',0):.2f}")

    print(f"\n  Results in: {OUTPUT_DIR}")
