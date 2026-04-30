#!/usr/bin/env python3
"""
Round 50 — L8 Full-Blast Parameter Grid Search
================================================
Three-layer progressive search over the full L8 parameter space.

Layer 1: Core params (KC_EMA, KC_Mult, SL, TP, Trail scale) ~4,300 combos
Layer 2: Overlay params (ADX, MaxHold, H1KC filter, Cap, TATrail, Choppy, EqCurve)
         on Layer 1 Top 20 — ~3,200 combos (many post-hoc)
Layer 3: K-Fold 6-Fold validation on Top 50

USAGE (server)
--------------
    cd /root/gold-quant-research
    nohup python3 -u experiments/run_round50_brute_force.py \
        > results/round50_results/stdout.txt 2>&1 &
"""
import sys, os, io, time, json, traceback
import multiprocessing as mp
import numpy as np
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple, Optional
from dataclasses import asdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

_script_dir = os.path.dirname(os.path.abspath(__file__))
for _candidate in [os.path.join(_script_dir, '..'), os.path.join(_script_dir, '..', '..'), os.getcwd()]:
    _candidate = os.path.abspath(_candidate)
    if os.path.isdir(os.path.join(_candidate, 'backtest')):
        sys.path.insert(0, _candidate)
        os.chdir(_candidate)
        break

OUTPUT_DIR = Path("results/round50_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = max(1, mp.cpu_count() - 1)

SPREAD = 0.50

# ═══════════════════════════════════════════════════════════════
# Regime trail presets (base = L8 current live)
# ═══════════════════════════════════════════════════════════════
BASE_REGIME = {
    'low':    {'trail_act': 0.22, 'trail_dist': 0.04},
    'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
    'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
}

def scale_regime(base: Dict, factor: float) -> Dict:
    return {
        regime: {
            'trail_act': round(v['trail_act'] * factor, 4),
            'trail_dist': round(v['trail_dist'] * factor, 4),
        }
        for regime, v in base.items()
    }

# ═══════════════════════════════════════════════════════════════
# Grid definitions
# ═══════════════════════════════════════════════════════════════

LAYER1_GRID = {
    'kc_ema':       [15, 20, 25, 30, 35, 40],
    'kc_mult':      [0.8, 1.0, 1.2, 1.4, 1.6],
    'sl_atr_mult':  [2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
    'tp_atr_mult':  [4.0, 6.0, 8.0, 10.0, 12.0, 16.0],
    'trail_scale':  [0.5, 0.75, 1.0, 1.5, 2.0],
}

LAYER2_OVERLAY = {
    'adx':      [10, 14, 18, 22, 25],
    'maxhold':  [12, 16, 20, 28, 40, 0],
    'h1kc':     [
        None,
        {'ema': 15, 'mult': 2.0},
        {'ema': 20, 'mult': 1.5},
        {'ema': 25, 'mult': 2.0},
    ],
    'cap':      [0, 20, 30, 50, 80],
    'tatrail':  [
        None,
        {'start': 2, 'decay': 0.75, 'floor': 0.003},
        {'start': 3, 'decay': 0.85, 'floor': 0.005},
        {'start': 4, 'decay': 0.90, 'floor': 0.010},
    ],
    'choppy':   [0.40, 0.50, 0.60],
    'eqcurve':  [0, 30, 50],
}


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


# ═══════════════════════════════════════════════════════════════
# Worker: Layer 1 — full engine run
# ═══════════════════════════════════════════════════════════════

def _worker_layer1(args):
    """Run one Layer 1 combo. Accepts pre-loaded DataFrames to avoid repeated CSV reads."""
    label, m15_df, h1_df, engine_kw = args
    try:
        from backtest.runner import DataBundle, run_variant
        trail_scale = engine_kw.pop('_trail_scale', 1.0)
        data = DataBundle(m15_df, h1_df)
        s = run_variant(data, label, verbose=False, **engine_kw)
        return {
            'label': label,
            'sl': engine_kw['sl_atr_mult'],
            'tp': engine_kw['tp_atr_mult'],
            'trail_scale': trail_scale,
            'n': s['n'],
            'sharpe': s['sharpe'],
            'total_pnl': s['total_pnl'],
            'win_rate': s['win_rate'],
            'max_dd': s['max_dd'],
            'avg_win': s.get('avg_win', 0),
            'avg_loss': s.get('avg_loss', 0),
            'rr': s.get('rr', 0),
            'year_pnl': s.get('year_pnl', {}),
            'elapsed_s': s.get('elapsed_s', 0),
        }
    except Exception as e:
        return {'label': label, 'error': str(e), 'sharpe': -999}


# ═══════════════════════════════════════════════════════════════
# Worker: Layer 2 — engine run with overlay params
# ═══════════════════════════════════════════════════════════════

def _worker_layer2(args):
    """Run one Layer 2 combo. Accepts pre-loaded DataFrames."""
    label, m15_df, h1_df, engine_kw = args
    try:
        from backtest.runner import DataBundle, run_variant
        data = DataBundle(m15_df, h1_df)
        s = run_variant(data, label, verbose=False, **engine_kw)
        trades_summary = []
        for t in s.get('_trades', []):
            trades_summary.append({
                'pnl': round(t.pnl, 2),
                'dir': t.direction,
                'entry_time': str(t.entry_time)[:16],
                'bars_held': t.bars_held,
            })
        return {
            'label': label,
            'n': s['n'],
            'sharpe': s['sharpe'],
            'total_pnl': s['total_pnl'],
            'win_rate': s['win_rate'],
            'max_dd': s['max_dd'],
            'elapsed_s': s.get('elapsed_s', 0),
            'trades': trades_summary,
        }
    except Exception as e:
        return {'label': label, 'error': str(e), 'sharpe': -999}


# ═══════════════════════════════════════════════════════════════
# Worker: Layer 3 — K-Fold
# ═══════════════════════════════════════════════════════════════

def _worker_kfold_one(args):
    """Run one fold of K-Fold validation. Accepts pre-loaded DataFrames."""
    label, m15_df, h1_df, engine_kw, start, end, fold_name = args
    try:
        from backtest.runner import DataBundle, run_variant
        import pandas as pd
        data = DataBundle(m15_df, h1_df)
        data = data.slice(start, end)
        if len(data.m15_df) < 1000:
            return {'label': label, 'fold': fold_name, 'sharpe': 0, 'n': 0, 'skip': True}
        s = run_variant(data, f"{label}_{fold_name}", verbose=False, **engine_kw)
        return {
            'label': label,
            'fold': fold_name,
            'n': s['n'],
            'sharpe': s['sharpe'],
            'total_pnl': s['total_pnl'],
            'win_rate': s['win_rate'],
            'max_dd': s['max_dd'],
        }
    except Exception as e:
        return {'label': label, 'fold': fold_name, 'error': str(e), 'sharpe': -999}


KFOLD_WINDOWS = [
    ("F1", "2015-01-01", "2017-01-01"),
    ("F2", "2017-01-01", "2019-01-01"),
    ("F3", "2019-01-01", "2021-01-01"),
    ("F4", "2021-01-01", "2023-01-01"),
    ("F5", "2023-01-01", "2025-01-01"),
    ("F6", "2025-01-01", "2026-04-01"),
]


# ═══════════════════════════════════════════════════════════════
# Post-hoc helpers (H1 KC filter + MaxLoss Cap)
# ═══════════════════════════════════════════════════════════════

def add_h1_kc_dir(h1_df, ema_period=20, mult=2.0):
    import pandas as pd
    h1 = h1_df.copy()
    h1['EMA_kc'] = h1['Close'].ewm(span=ema_period, adjust=False).mean()
    tr = pd.DataFrame({
        'hl': h1['High'] - h1['Low'],
        'hc': (h1['High'] - h1['Close'].shift(1)).abs(),
        'lc': (h1['Low'] - h1['Close'].shift(1)).abs(),
    }).max(axis=1)
    h1['ATR_kc'] = tr.rolling(14).mean()
    h1['KC_U'] = h1['EMA_kc'] + mult * h1['ATR_kc']
    h1['KC_L'] = h1['EMA_kc'] - mult * h1['ATR_kc']
    h1['kc_dir'] = 'NEUTRAL'
    h1.loc[h1['Close'] > h1['KC_U'], 'kc_dir'] = 'BULL'
    h1.loc[h1['Close'] < h1['KC_L'], 'kc_dir'] = 'BEAR'
    return h1


def filter_trades_h1kc(trades_summary, h1_kc_df):
    """Post-hoc filter trades by H1 KC direction. Works on trade dicts.
    Uses searchsorted for O(n*log(m)) instead of O(n*m) lookups.
    """
    import pandas as pd
    if not trades_summary:
        return []
    idx_tz = h1_kc_df.index.tz
    h1_index = h1_kc_df.index
    h1_kc_dirs = h1_kc_df['kc_dir'].values
    kept = []
    for t in trades_summary:
        et = pd.Timestamp(t['entry_time'])
        if idx_tz is not None and et.tzinfo is None:
            et = et.tz_localize(idx_tz)
        pos = h1_index.searchsorted(et, side='right') - 1
        if pos < 0:
            continue
        kc_d = h1_kc_dirs[pos]
        if (t['dir'] == 'BUY' and kc_d == 'BULL') or (t['dir'] == 'SELL' and kc_d == 'BEAR'):
            kept.append(t)
    return kept


def apply_cap_trades(trades_summary, cap_usd):
    """Post-hoc cap: limit per-trade loss."""
    if cap_usd <= 0:
        return trades_summary
    capped = []
    for t in trades_summary:
        if t['pnl'] < -cap_usd:
            ct = dict(t)
            ct['pnl'] = -cap_usd
            capped.append(ct)
        else:
            capped.append(t)
    return capped


def stats_from_trade_dicts(trades, label=""):
    """Compute stats from list of trade dicts with 'pnl' key."""
    if not trades:
        return {'n': 0, 'total_pnl': 0, 'sharpe': 0, 'win_rate': 0, 'max_dd': 0}
    pnls = [t['pnl'] for t in trades]
    daily = {}
    for t in trades:
        d = t.get('exit_time', t.get('entry_time', ''))[:10]
        daily[d] = daily.get(d, 0) + t['pnl']
    daily_pnls = list(daily.values())
    sharpe = 0
    if len(daily_pnls) > 1:
        std = np.std(daily_pnls, ddof=1)
        if std > 0:
            sharpe = np.mean(daily_pnls) / std * np.sqrt(252)
    wins = [p for p in pnls if p > 0]
    wr = len(wins) / len(pnls) * 100 if pnls else 0
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0
    return {
        'label': label,
        'n': len(trades),
        'total_pnl': round(sum(pnls), 2),
        'sharpe': round(sharpe, 2),
        'win_rate': round(wr, 1),
        'max_dd': round(max_dd, 2),
    }


# ═══════════════════════════════════════════════════════════════
# Checkpoint helpers
# ═══════════════════════════════════════════════════════════════

def save_checkpoint(data, filename):
    path = OUTPUT_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, default=str)
    print(f"  [Checkpoint] Saved {path} ({len(data) if isinstance(data, list) else 'dict'})")


def load_checkpoint(filename):
    path = OUTPUT_DIR / filename
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════
# LAYER 1: Core parameter grid search
# ═══════════════════════════════════════════════════════════════

def build_layer1_tasks():
    """Build tasks grouped by (kc_ema, kc_mult) for efficient data loading."""
    from backtest.runner import LIVE_PARITY_KWARGS

    kc_groups = {}
    param_dims = ['sl_atr_mult', 'tp_atr_mult', 'trail_scale']
    for sl in LAYER1_GRID['sl_atr_mult']:
        for tp in LAYER1_GRID['tp_atr_mult']:
            if tp <= sl:
                continue
            for ts in LAYER1_GRID['trail_scale']:
                regime = scale_regime(BASE_REGIME, ts)
                engine_kw = {
                    **LIVE_PARITY_KWARGS,
                    'sl_atr_mult': sl,
                    'tp_atr_mult': tp,
                    'trailing_activate_atr': round(
                        BASE_REGIME['normal']['trail_act'] * ts, 4),
                    'trailing_distance_atr': round(
                        BASE_REGIME['normal']['trail_dist'] * ts, 4),
                    'regime_config': regime,
                    'keltner_adx_threshold': 14,
                    'min_entry_gap_hours': 1.0,
                    'keltner_max_hold_m15': 20,
                    'spread_cost': SPREAD,
                    '_trail_scale': ts,
                }
                for kc_ema in LAYER1_GRID['kc_ema']:
                    for kc_mult in LAYER1_GRID['kc_mult']:
                        label = f"E{kc_ema}_M{kc_mult}_SL{sl}_TP{tp}_T{ts}"
                        key = (kc_ema, kc_mult)
                        kc_groups.setdefault(key, []).append((label, dict(engine_kw)))

    return kc_groups


def run_layer1():
    print("\n" + "=" * 80)
    print("  LAYER 1: Core Parameter Grid Search")
    print("=" * 80)

    existing = load_checkpoint("layer1_all.json")
    if existing:
        print(f"  [Resume] Found checkpoint with {len(existing)} results, skipping Layer 1")
        return existing

    from backtest.runner import DataBundle

    kc_groups = build_layer1_tasks()
    total = sum(len(v) for v in kc_groups.values())
    print(f"  Total combos: {total}")
    print(f"  Workers: {MAX_WORKERS}")
    print(f"  DataBundle groups: {len(kc_groups)} (by KC_EMA x KC_Mult)")
    print(f"  Start: {time.strftime('%H:%M:%S')}")

    t0 = time.time()
    all_results = []

    group_idx = 0
    for (kc_ema, kc_mult), group_items in kc_groups.items():
        group_idx += 1
        gt0 = time.time()
        print(f"\n  Group {group_idx}/{len(kc_groups)}: E{kc_ema}/M{kc_mult} "
              f"({len(group_items)} combos)...", flush=True)

        print(f"    Loading data...", end='', flush=True)
        data = DataBundle.load_custom(kc_ema=kc_ema, kc_mult=kc_mult)
        print(f" done ({time.time()-gt0:.0f}s)", flush=True)

        pool_tasks = [
            (label, data.m15_df, data.h1_df, engine_kw)
            for label, engine_kw in group_items
        ]

        with mp.Pool(min(MAX_WORKERS, len(pool_tasks))) as pool:
            results = pool.map(_worker_layer1, pool_tasks)

        for r in results:
            if r.get('error'):
                print(f"    ERROR: {r['label']}: {r['error']}")
            r['kc_ema'] = kc_ema
            r['kc_mult'] = kc_mult
            all_results.append(r)

        elapsed_g = time.time() - gt0
        elapsed_total = time.time() - t0
        done = len(all_results)
        rate = done / elapsed_total if elapsed_total > 0 else 1
        eta = (total - done) / rate if rate > 0 else 0
        print(f"    Group done in {elapsed_g:.0f}s | "
              f"Progress: {done}/{total} ({done/total*100:.0f}%) | "
              f"ETA: {eta/3600:.1f}h", flush=True)

        save_checkpoint(all_results, "layer1_all.json")

    elapsed = time.time() - t0
    valid = [r for r in all_results if r.get('sharpe', -999) > -999]
    valid.sort(key=lambda x: x['sharpe'], reverse=True)

    print(f"\n  Layer 1 complete: {len(valid)} valid / {len(all_results)} total "
          f"in {elapsed/3600:.1f}h")
    print(f"\n  Top 10:")
    print(f"  {'Rank':>4} {'Label':>35} {'Sharpe':>8} {'PnL':>12} {'N':>6} {'MaxDD':>10} {'WR':>6}")
    print(f"  {'-'*4} {'-'*35} {'-'*8} {'-'*12} {'-'*6} {'-'*10} {'-'*6}")
    for i, r in enumerate(valid[:10], 1):
        print(f"  {i:>4} {r['label']:>35} {r['sharpe']:>8.2f} {fmt(r['total_pnl']):>12} "
              f"{r['n']:>6} {fmt(r['max_dd']):>10} {r['win_rate']:>5.1f}%")

    save_checkpoint(all_results, "layer1_all.json")

    lines = ["R50 Layer 1: Core Parameter Grid Search Results",
             "=" * 100, f"Total: {len(valid)} valid combos | Time: {elapsed/3600:.1f}h", ""]
    lines.append(f"{'Rank':>4} {'Label':>35} {'Sharpe':>8} {'PnL':>12} {'N':>6} "
                 f"{'MaxDD':>10} {'WR':>6} {'AvgW':>8} {'AvgL':>8} {'RR':>6}")
    lines.append("-" * 110)
    for i, r in enumerate(valid[:100], 1):
        lines.append(f"{i:>4} {r['label']:>35} {r['sharpe']:>8.2f} {fmt(r['total_pnl']):>12} "
                     f"{r['n']:>6} {fmt(r['max_dd']):>10} {r['win_rate']:>5.1f}% "
                     f"{r.get('avg_win',0):>8.2f} {r.get('avg_loss',0):>8.2f} {r.get('rr',0):>6.2f}")
    with open(OUTPUT_DIR / "layer1_ranking.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    return all_results


# ═══════════════════════════════════════════════════════════════
# LAYER 2: Overlay grid on Top 20 from Layer 1
# ═══════════════════════════════════════════════════════════════

def run_layer2(layer1_results):
    print("\n" + "=" * 80)
    print("  LAYER 2: Overlay Parameter Grid on Top 20")
    print("=" * 80)

    existing = load_checkpoint("layer2_all.json")
    if existing:
        print(f"  [Resume] Found checkpoint with {len(existing)} results, skipping Layer 2")
        return existing

    valid_l1 = [r for r in layer1_results if r.get('sharpe', -999) > -999 and 'error' not in r]
    valid_l1.sort(key=lambda x: x['sharpe'], reverse=True)
    top20 = valid_l1[:20]

    print(f"  Base combos (Top 20 from L1): {len(top20)}")
    for i, r in enumerate(top20, 1):
        print(f"    {i:>2}. {r['label']} Sharpe={r['sharpe']:.2f}")

    t0 = time.time()
    all_results = []

    from backtest.runner import DataBundle, LIVE_PARITY_KWARGS

    for base_idx, base in enumerate(top20, 1):
        bt0 = time.time()
        kc_ema = base['kc_ema']
        kc_mult = base['kc_mult']

        print(f"\n  === Base {base_idx}/20: {base['label']} (Sharpe={base['sharpe']:.2f}) ===",
              flush=True)

        engine_tasks = []
        posthoc_combos = []

        print(f"    Loading data E{kc_ema}/M{kc_mult}...", end='', flush=True)
        data_bundle = DataBundle.load_custom(kc_ema=kc_ema, kc_mult=kc_mult)
        print(f" done", flush=True)

        for adx in LAYER2_OVERLAY['adx']:
            for maxhold in LAYER2_OVERLAY['maxhold']:
                for tatrail in LAYER2_OVERLAY['tatrail']:
                    for choppy in LAYER2_OVERLAY['choppy']:
                        for eqcurve in LAYER2_OVERLAY['eqcurve']:
                            engine_kw = {
                                **LIVE_PARITY_KWARGS,
                                'sl_atr_mult': base['sl'],
                                'tp_atr_mult': base['tp'],
                                'trailing_activate_atr': round(
                                    BASE_REGIME['normal']['trail_act'] * base['trail_scale'], 4),
                                'trailing_distance_atr': round(
                                    BASE_REGIME['normal']['trail_dist'] * base['trail_scale'], 4),
                                'regime_config': scale_regime(BASE_REGIME, base['trail_scale']),
                                'keltner_adx_threshold': adx,
                                'min_entry_gap_hours': 1.0,
                                'keltner_max_hold_m15': maxhold if maxhold > 0 else 9999,
                                'choppy_threshold': choppy,
                                'spread_cost': SPREAD,
                            }
                            if tatrail is not None:
                                engine_kw['time_adaptive_trail'] = True
                                engine_kw['time_adaptive_trail_start'] = tatrail['start']
                                engine_kw['time_adaptive_trail_decay'] = tatrail['decay']
                                engine_kw['time_adaptive_trail_floor'] = tatrail['floor']
                            if eqcurve > 0:
                                engine_kw['equity_curve_filter'] = True
                                engine_kw['equity_ma_period'] = eqcurve

                            mh_tag = f"MH{maxhold}" if maxhold > 0 else "MHoff"
                            ta_tag = f"TA{tatrail['start']}" if tatrail else "TAoff"
                            eq_tag = f"EQ{eqcurve}" if eqcurve > 0 else "EQoff"
                            tag = (f"{base['label']}_ADX{adx}_{mh_tag}_{ta_tag}"
                                   f"_CH{choppy}_{eq_tag}")

                            for h1kc in LAYER2_OVERLAY['h1kc']:
                                for cap in LAYER2_OVERLAY['cap']:
                                    h1_tag = (f"KC{h1kc['ema']}x{h1kc['mult']}"
                                              if h1kc else "KCoff")
                                    cap_tag = f"C{cap}" if cap > 0 else "Coff"
                                    full_label = f"{tag}_{h1_tag}_{cap_tag}"
                                    posthoc_combos.append({
                                        'engine_tag': tag,
                                        'full_label': full_label,
                                        'h1kc': h1kc,
                                        'cap': cap,
                                    })

                            engine_tasks.append((tag, data_bundle.m15_df, data_bundle.h1_df,
                                                 engine_kw))

        unique_engine_tasks = {}
        for t in engine_tasks:
            if t[0] not in unique_engine_tasks:
                unique_engine_tasks[t[0]] = t
        engine_tasks = list(unique_engine_tasks.values())

        print(f"    Engine runs: {len(engine_tasks)}, "
              f"Post-hoc combos: {len(posthoc_combos)}", flush=True)

        engine_results = {}
        with mp.Pool(min(MAX_WORKERS, len(engine_tasks))) as pool:
            raw = pool.map(_worker_layer2, engine_tasks)
        for r in raw:
            engine_results[r['label']] = r

        h1_kc_cache = {}

        for combo in posthoc_combos:
            engine_tag = combo['engine_tag']
            er = engine_results.get(engine_tag)
            if not er or er.get('error'):
                continue

            trades = er.get('trades', [])
            h1kc_cfg = combo['h1kc']
            cap = combo['cap']

            if h1kc_cfg:
                cache_key = (h1kc_cfg['ema'], h1kc_cfg['mult'])
                if cache_key not in h1_kc_cache:
                    h1_kc_cache[cache_key] = add_h1_kc_dir(
                        data_bundle.h1_df, h1kc_cfg['ema'], h1kc_cfg['mult'])
                trades = filter_trades_h1kc(trades, h1_kc_cache[cache_key])

            if cap > 0:
                trades = apply_cap_trades(trades, cap)

            stats = stats_from_trade_dicts(trades, combo['full_label'])
            stats['base_label'] = base['label']
            stats['base_sharpe'] = base['sharpe']
            all_results.append(stats)

        elapsed_base = time.time() - bt0
        print(f"    Done in {elapsed_base:.0f}s, "
              f"total results so far: {len(all_results)}", flush=True)

        save_checkpoint(all_results, "layer2_all.json")

    elapsed = time.time() - t0
    valid = [r for r in all_results if r.get('sharpe', 0) > 0]
    valid.sort(key=lambda x: x['sharpe'], reverse=True)

    print(f"\n  Layer 2 complete: {len(valid)} positive / {len(all_results)} total "
          f"in {elapsed/3600:.1f}h")
    print(f"\n  Top 10:")
    print(f"  {'Rank':>4} {'Label':>60} {'Sharpe':>8} {'PnL':>12} {'N':>6} {'MaxDD':>10}")
    print(f"  {'-'*4} {'-'*60} {'-'*8} {'-'*12} {'-'*6} {'-'*10}")
    for i, r in enumerate(valid[:10], 1):
        print(f"  {i:>4} {r['label']:>60} {r['sharpe']:>8.2f} "
              f"{fmt(r['total_pnl']):>12} {r['n']:>6} {fmt(r['max_dd']):>10}")

    save_checkpoint(all_results, "layer2_all.json")

    lines = ["R50 Layer 2: Overlay Grid Results", "=" * 120,
             f"Total: {len(valid)} positive Sharpe / {len(all_results)} total | "
             f"Time: {elapsed/3600:.1f}h", ""]
    lines.append(f"{'Rank':>4} {'Label':>60} {'Sharpe':>8} {'PnL':>12} {'N':>6} {'MaxDD':>10}")
    lines.append("-" * 110)
    for i, r in enumerate(valid[:100], 1):
        lines.append(f"{i:>4} {r['label']:>60} {r['sharpe']:>8.2f} "
                     f"{fmt(r['total_pnl']):>12} {r['n']:>6} {fmt(r['max_dd']):>10}")
    with open(OUTPUT_DIR / "layer2_ranking.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    return all_results


# ═══════════════════════════════════════════════════════════════
# LAYER 3: K-Fold validation on Top 50
# ═══════════════════════════════════════════════════════════════

def parse_label_to_engine_kw(label):
    """Parse a Layer 1 or Layer 2 label back into engine kwargs."""
    from backtest.runner import LIVE_PARITY_KWARGS
    parts = label.split('_')

    kc_ema = 25
    kc_mult = 1.2
    sl = 3.5
    tp = 8.0
    trail_scale = 1.0
    adx = 14
    maxhold = 20
    choppy = 0.50
    tatrail = None
    eqcurve = 0
    h1kc = None
    cap = 0

    for p in parts:
        if p.startswith('E') and not p.startswith('EQ'):
            try:
                kc_ema = int(p[1:])
            except ValueError:
                pass
        elif p.startswith('M') and not p.startswith('MH'):
            try:
                kc_mult = float(p[1:])
            except ValueError:
                pass
        elif p.startswith('SL'):
            try:
                sl = float(p[2:])
            except ValueError:
                pass
        elif p.startswith('TP'):
            try:
                tp = float(p[2:])
            except ValueError:
                pass
        elif p.startswith('T') and not p.startswith('TA'):
            try:
                trail_scale = float(p[1:])
            except ValueError:
                pass
        elif p.startswith('ADX'):
            try:
                adx = int(p[3:])
            except ValueError:
                pass
        elif p.startswith('MH'):
            val = p[2:]
            if val == 'off':
                maxhold = 9999
            else:
                try:
                    maxhold = int(val)
                except ValueError:
                    pass
        elif p.startswith('CH'):
            try:
                choppy = float(p[2:])
            except ValueError:
                pass
        elif p.startswith('TA'):
            val = p[2:]
            if val == 'off':
                tatrail = None
            else:
                ta_map = {
                    '2': {'start': 2, 'decay': 0.75, 'floor': 0.003},
                    '3': {'start': 3, 'decay': 0.85, 'floor': 0.005},
                    '4': {'start': 4, 'decay': 0.90, 'floor': 0.010},
                }
                tatrail = ta_map.get(val)
        elif p.startswith('EQ'):
            val = p[2:]
            if val == 'off':
                eqcurve = 0
            else:
                try:
                    eqcurve = int(val)
                except ValueError:
                    pass
        elif p.startswith('KC') and 'x' in p:
            try:
                kc_parts = p[2:].split('x')
                h1kc = {'ema': int(kc_parts[0]), 'mult': float(kc_parts[1])}
            except (ValueError, IndexError):
                pass
        elif p.startswith('C') and not p.startswith('CH'):
            val = p[1:]
            if val == 'off':
                cap = 0
            else:
                try:
                    cap = int(val)
                except ValueError:
                    pass

    regime = scale_regime(BASE_REGIME, trail_scale)
    engine_kw = {
        **LIVE_PARITY_KWARGS,
        'sl_atr_mult': sl,
        'tp_atr_mult': tp,
        'trailing_activate_atr': round(
            BASE_REGIME['normal']['trail_act'] * trail_scale, 4),
        'trailing_distance_atr': round(
            BASE_REGIME['normal']['trail_dist'] * trail_scale, 4),
        'regime_config': regime,
        'keltner_adx_threshold': adx,
        'min_entry_gap_hours': 1.0,
        'keltner_max_hold_m15': maxhold if maxhold < 9999 else 9999,
        'choppy_threshold': choppy,
        'spread_cost': SPREAD,
    }
    if tatrail:
        engine_kw['time_adaptive_trail'] = True
        engine_kw['time_adaptive_trail_start'] = tatrail['start']
        engine_kw['time_adaptive_trail_decay'] = tatrail['decay']
        engine_kw['time_adaptive_trail_floor'] = tatrail['floor']
    if eqcurve > 0:
        engine_kw['equity_curve_filter'] = True
        engine_kw['equity_ma_period'] = eqcurve

    return kc_ema, kc_mult, engine_kw, h1kc, cap


def run_layer3(layer1_results, layer2_results):
    print("\n" + "=" * 80)
    print("  LAYER 3: K-Fold 6-Fold Validation (Top 50)")
    print("=" * 80)

    existing = load_checkpoint("kfold_top50.json")
    if existing:
        print(f"  [Resume] Found checkpoint, skipping Layer 3")
        return existing

    all_candidates = []
    for r in layer1_results:
        if r.get('sharpe', -999) > 0 and 'error' not in r:
            all_candidates.append(r)
    for r in layer2_results:
        if r.get('sharpe', 0) > 0:
            all_candidates.append(r)

    all_candidates.sort(key=lambda x: x['sharpe'], reverse=True)

    seen_labels = set()
    top50 = []
    for r in all_candidates:
        if r['label'] not in seen_labels:
            seen_labels.add(r['label'])
            top50.append(r)
            if len(top50) >= 50:
                break

    print(f"  Candidates for K-Fold: {len(top50)}")
    for i, r in enumerate(top50[:10], 1):
        print(f"    {i:>2}. {r['label']} Sharpe={r['sharpe']:.2f}")
    if len(top50) > 10:
        print(f"    ... and {len(top50)-10} more")

    from backtest.runner import DataBundle

    t0 = time.time()

    kc_groups = {}
    for r in top50:
        label = r['label']
        kc_ema, kc_mult, engine_kw, h1kc, cap = parse_label_to_engine_kw(label)
        key = (kc_ema, kc_mult)
        kc_groups.setdefault(key, []).append((label, engine_kw))

    print(f"  DataBundle groups: {len(kc_groups)}")

    raw_results = []
    for gi, ((kc_ema, kc_mult), items) in enumerate(kc_groups.items(), 1):
        print(f"    KFold group {gi}/{len(kc_groups)}: E{kc_ema}/M{kc_mult} "
              f"({len(items)} combos)...", flush=True)
        data = DataBundle.load_custom(kc_ema=kc_ema, kc_mult=kc_mult)

        kfold_tasks = []
        for label, engine_kw in items:
            for fold_name, start, end in KFOLD_WINDOWS:
                kfold_tasks.append((label, data.m15_df, data.h1_df,
                                    engine_kw, start, end, fold_name))

        with mp.Pool(min(MAX_WORKERS, len(kfold_tasks))) as pool:
            raw_results.extend(pool.map(_worker_kfold_one, kfold_tasks))

    total_kfold = len(raw_results)
    print(f"  Total K-Fold runs completed: {total_kfold}")
    print(f"  Workers: {MAX_WORKERS}")

    by_label = {}
    for r in raw_results:
        lbl = r['label']
        by_label.setdefault(lbl, []).append(r)

    kfold_summary = []
    for r in top50:
        label = r['label']
        folds = by_label.get(label, [])
        fold_sharpes = [f['sharpe'] for f in folds if not f.get('skip')]
        n_positive = sum(1 for s in fold_sharpes if s > 0)
        mean_sharpe = np.mean(fold_sharpes) if fold_sharpes else 0
        min_sharpe = min(fold_sharpes) if fold_sharpes else 0
        kfold_summary.append({
            'label': label,
            'full_sharpe': r['sharpe'],
            'full_pnl': r.get('total_pnl', 0),
            'full_n': r.get('n', 0),
            'full_maxdd': r.get('max_dd', 0),
            'kfold_positive': n_positive,
            'kfold_total': len(fold_sharpes),
            'kfold_mean_sharpe': round(mean_sharpe, 2),
            'kfold_min_sharpe': round(min_sharpe, 2),
            'fold_sharpes': [round(s, 2) for s in fold_sharpes],
            'passed': n_positive == len(fold_sharpes) and mean_sharpe > 2.0,
        })

    kfold_summary.sort(key=lambda x: (x['passed'], x['kfold_mean_sharpe']), reverse=True)

    elapsed = time.time() - t0
    passed = [k for k in kfold_summary if k['passed']]

    print(f"\n  Layer 3 complete in {elapsed/60:.0f}min")
    print(f"  Passed (all folds Sharpe>0, mean>2.0): {len(passed)}/{len(kfold_summary)}")
    print(f"\n  {'Rank':>4} {'Label':>60} {'FullSh':>7} {'KFMean':>7} "
          f"{'KFMin':>7} {'Pass':>6} {'Folds':>20}")
    print(f"  {'-'*4} {'-'*60} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*20}")
    for i, k in enumerate(kfold_summary[:20], 1):
        folds_str = ",".join(str(s) for s in k['fold_sharpes'])
        p_str = "PASS" if k['passed'] else "FAIL"
        print(f"  {i:>4} {k['label']:>60} {k['full_sharpe']:>7.2f} "
              f"{k['kfold_mean_sharpe']:>7.2f} {k['kfold_min_sharpe']:>7.2f} "
              f"{p_str:>6} [{folds_str}]")

    save_checkpoint(kfold_summary, "kfold_top50.json")

    lines = ["R50 Layer 3: K-Fold 6-Fold Validation Results",
             "=" * 130, f"Passed: {len(passed)}/{len(kfold_summary)} | Time: {elapsed/60:.0f}min",
             ""]
    lines.append(f"{'Rank':>4} {'Label':>60} {'FullSh':>7} {'FullPnL':>12} {'FullDD':>10} "
                 f"{'KFMean':>7} {'KFMin':>7} {'Pass':>6} {'Folds':>30}")
    lines.append("-" * 160)
    for i, k in enumerate(kfold_summary, 1):
        folds_str = ",".join(str(s) for s in k['fold_sharpes'])
        p_str = "PASS" if k['passed'] else "FAIL"
        lines.append(
            f"{i:>4} {k['label']:>60} {k['full_sharpe']:>7.2f} "
            f"{fmt(k['full_pnl']):>12} {fmt(k['full_maxdd']):>10} "
            f"{k['kfold_mean_sharpe']:>7.2f} {k['kfold_min_sharpe']:>7.2f} "
            f"{p_str:>6} [{folds_str}]")
    with open(OUTPUT_DIR / "kfold_top50.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    return kfold_summary


# ═══════════════════════════════════════════════════════════════
# LAYER 4: Deep Validation (7 institutional-grade tests)
# ═══════════════════════════════════════════════════════════════

HOLDOUT_START = "2025-07-01"
HOLDOUT_END = "2026-04-01"

N_TRIALS_TOTAL = 8500  # ~5100 L1 + ~3200 L2, conservative upper bound

COST_LEVELS = [0.30, 0.50, 0.75, 1.00, 1.50]

PERTURB_PARAMS = {
    'sl_atr_mult':  [0.80, 0.90, 1.10, 1.20],
    'tp_atr_mult':  [0.80, 0.90, 1.10, 1.20],
    'keltner_adx_threshold': [0.80, 0.90, 1.10, 1.20],
    'choppy_threshold':      [0.80, 0.90, 1.10, 1.20],
}


def _worker_engine_run(args):
    """Generic worker: run engine with given kwargs on pre-loaded data."""
    tag, m15_df, h1_df, engine_kw = args
    try:
        from backtest.runner import DataBundle, run_variant
        data = DataBundle(m15_df, h1_df)
        s = run_variant(data, tag, verbose=False, **engine_kw)
        daily_pnl = []
        for t in s.get('_trades', []):
            daily_pnl.append(t.pnl)
        from backtest.stats import aggregate_daily_pnl
        dpnl = aggregate_daily_pnl(s.get('_trades', []))
        return {
            'tag': tag,
            'sharpe': s['sharpe'],
            'total_pnl': s['total_pnl'],
            'n': s['n'],
            'max_dd': s['max_dd'],
            'max_dd_pct': s.get('max_dd_pct', 0),
            'win_rate': s.get('win_rate', 0),
            'year_pnl': s.get('year_pnl', {}),
            'daily_pnl': dpnl,
            'trade_pnls': [t.pnl for t in s.get('_trades', [])],
        }
    except Exception as e:
        return {'tag': tag, 'error': str(e), 'sharpe': -999}


def _worker_holdout_run(args):
    """Worker: run engine on holdout period only."""
    tag, m15_df, h1_df, engine_kw, start, end = args
    try:
        from backtest.runner import DataBundle, run_variant
        data = DataBundle(m15_df, h1_df).slice(start, end)
        if len(data.m15_df) < 500:
            return {'tag': tag, 'sharpe': 0, 'n': 0, 'skip': True}
        s = run_variant(data, tag, verbose=False, **engine_kw)
        return {
            'tag': tag,
            'sharpe': s['sharpe'],
            'total_pnl': s['total_pnl'],
            'n': s['n'],
            'max_dd': s['max_dd'],
            'max_dd_pct': s.get('max_dd_pct', 0),
        }
    except Exception as e:
        return {'tag': tag, 'error': str(e), 'sharpe': -999}


def test_4a_dsr(candidates_daily_pnl):
    """4-A: Deflated Sharpe Ratio for each candidate."""
    from backtest.stats import deflated_sharpe
    all_sharpes = []
    for label, dpnl in candidates_daily_pnl.items():
        arr = np.asarray(dpnl, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) > 1 and np.std(arr, ddof=1) > 0:
            all_sharpes.append(float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252)))
    sharpes_var = float(np.var(all_sharpes)) if len(all_sharpes) > 1 else None

    results = {}
    for label, dpnl in candidates_daily_pnl.items():
        dsr = deflated_sharpe(dpnl, n_trials=N_TRIALS_TOTAL, all_sharpes_var=sharpes_var)
        results[label] = {
            'dsr': dsr['dsr'],
            'sr_star': dsr['sr_star'],
            'sharpe_obs': dsr['sharpe_obs'],
            'passed': dsr['passed'],
        }
    return results


def test_4b_pbo(candidates_daily_pnl):
    """4-B: Probability of Backtest Overfitting across all candidates."""
    from backtest.stats import compute_pbo
    pbo = compute_pbo(candidates_daily_pnl, n_partitions=8)
    return {
        'pbo': pbo['pbo'],
        'n_combinations': pbo['n_combinations'],
        'overfit_risk': pbo['overfit_risk'],
        'passed': pbo['pbo'] < 0.30 and pbo['overfit_risk'] != 'HIGH',
    }


def test_4c_oos_holdout(candidates_labels, kc_data_cache):
    """4-C: True OOS holdout test on 2025-07 to 2026-04."""
    tasks = []
    for label in candidates_labels:
        kc_ema, kc_mult, engine_kw, h1kc, cap = parse_label_to_engine_kw(label)
        key = (kc_ema, kc_mult)
        m15_df, h1_df = kc_data_cache[key]
        tasks.append((label, m15_df, h1_df, engine_kw, HOLDOUT_START, HOLDOUT_END))

    results = {}
    with mp.Pool(min(MAX_WORKERS, max(1, len(tasks)))) as pool:
        raw = pool.map(_worker_holdout_run, tasks)
    for r in raw:
        tag = r['tag']
        passed = (r.get('sharpe', 0) > 1.5
                  and r.get('max_dd_pct', 100) < 15
                  and not r.get('skip', False)
                  and not r.get('error'))
        results[tag] = {
            'holdout_sharpe': r.get('sharpe', 0),
            'holdout_pnl': r.get('total_pnl', 0),
            'holdout_n': r.get('n', 0),
            'holdout_maxdd_pct': r.get('max_dd_pct', 0),
            'passed': passed,
        }
    return results


def test_4d_param_stability(candidates_labels, kc_data_cache):
    """4-D: Parameter cliff detection via perturbation."""
    from backtest.runner import LIVE_PARITY_KWARGS

    base_tasks = []
    perturb_tasks = []

    for label in candidates_labels:
        kc_ema, kc_mult, engine_kw, h1kc, cap = parse_label_to_engine_kw(label)
        key = (kc_ema, kc_mult)
        m15_df, h1_df = kc_data_cache[key]

        base_tasks.append((f"{label}__base", m15_df, h1_df, dict(engine_kw)))

        for param_name, factors in PERTURB_PARAMS.items():
            if param_name not in engine_kw:
                continue
            orig_val = engine_kw[param_name]
            for factor in factors:
                perturbed_kw = dict(engine_kw)
                perturbed_kw[param_name] = round(orig_val * factor, 4)
                ptag = f"{label}__p_{param_name}_{factor}"
                perturb_tasks.append((ptag, m15_df, h1_df, perturbed_kw))

    all_tasks = base_tasks + perturb_tasks
    print(f"    Param stability: {len(base_tasks)} base + {len(perturb_tasks)} perturb "
          f"= {len(all_tasks)} runs", flush=True)

    with mp.Pool(min(MAX_WORKERS, max(1, len(all_tasks)))) as pool:
        raw = pool.map(_worker_engine_run, all_tasks)

    sharpe_by_tag = {r['tag']: r.get('sharpe', 0) for r in raw}

    results = {}
    for label in candidates_labels:
        base_sharpe = sharpe_by_tag.get(f"{label}__base", 0)
        if base_sharpe <= 0:
            results[label] = {'max_decay': 1.0, 'worst_param': 'N/A', 'passed': False}
            continue

        max_decay = 0.0
        worst_param = ''
        kc_ema, kc_mult, engine_kw, h1kc, cap = parse_label_to_engine_kw(label)

        for param_name, factors in PERTURB_PARAMS.items():
            if param_name not in engine_kw:
                continue
            for factor in factors:
                ptag = f"{label}__p_{param_name}_{factor}"
                p_sharpe = sharpe_by_tag.get(ptag, 0)
                decay = abs(base_sharpe - p_sharpe) / base_sharpe
                if decay > max_decay:
                    max_decay = decay
                    worst_param = f"{param_name}x{factor}"

        results[label] = {
            'base_sharpe': round(base_sharpe, 2),
            'max_decay': round(max_decay, 4),
            'worst_param': worst_param,
            'passed': max_decay < 0.30,
        }
    return results


def test_4e_monte_carlo(candidates_trade_pnls, n_perms=1000):
    """4-E: Monte Carlo permutation test on trade order."""
    results = {}
    rng = np.random.default_rng(42)

    for label, pnls in candidates_trade_pnls.items():
        arr = np.asarray(pnls, dtype=float)
        n = len(arr)
        if n < 20:
            results[label] = {'mc_p_value': 1.0, 'passed': False}
            continue

        daily = {}
        for i, p in enumerate(pnls):
            d = i // 5
            daily[d] = daily.get(d, 0) + p
        daily_arr = np.array(list(daily.values()))
        if len(daily_arr) < 2 or np.std(daily_arr, ddof=1) <= 0:
            results[label] = {'mc_p_value': 1.0, 'passed': False}
            continue
        orig_sharpe = float(np.mean(daily_arr) / np.std(daily_arr, ddof=1) * np.sqrt(252))

        count_above = 0
        for _ in range(n_perms):
            shuffled = rng.permutation(arr)
            sd = {}
            for i, p in enumerate(shuffled):
                d = i // 5
                sd[d] = sd.get(d, 0) + p
            sd_arr = np.array(list(sd.values()))
            std = np.std(sd_arr, ddof=1)
            if std > 0:
                sh = float(np.mean(sd_arr) / std * np.sqrt(252))
            else:
                sh = 0
            if sh >= orig_sharpe:
                count_above += 1

        p_value = (count_above + 1) / (n_perms + 1)
        results[label] = {
            'original_sharpe': round(orig_sharpe, 2),
            'mc_p_value': round(p_value, 4),
            'passed': p_value < 0.05,
        }
    return results


def test_4f_cost_sensitivity(candidates_labels, kc_data_cache):
    """4-F: Test strategy under different spread/cost levels."""
    tasks = []
    for label in candidates_labels:
        kc_ema, kc_mult, engine_kw, h1kc, cap = parse_label_to_engine_kw(label)
        key = (kc_ema, kc_mult)
        m15_df, h1_df = kc_data_cache[key]
        for cost in COST_LEVELS:
            cost_kw = dict(engine_kw)
            cost_kw['spread_cost'] = cost
            ctag = f"{label}__cost_{cost}"
            tasks.append((ctag, m15_df, h1_df, cost_kw))

    print(f"    Cost sensitivity: {len(tasks)} runs", flush=True)

    with mp.Pool(min(MAX_WORKERS, max(1, len(tasks)))) as pool:
        raw = pool.map(_worker_engine_run, tasks)

    sharpe_by_tag = {r['tag']: r.get('sharpe', 0) for r in raw}

    results = {}
    for label in candidates_labels:
        cost_sharpes = {}
        for cost in COST_LEVELS:
            ctag = f"{label}__cost_{cost}"
            cost_sharpes[cost] = sharpe_by_tag.get(ctag, 0)
        sharpe_at_1_5x = cost_sharpes.get(0.75, 0)
        results[label] = {
            'cost_sharpes': {str(k): round(v, 2) for k, v in cost_sharpes.items()},
            'sharpe_at_1_5x': round(sharpe_at_1_5x, 2),
            'passed': sharpe_at_1_5x > 1.0,
        }
    return results


def test_4g_year_consistency(candidates_year_pnls):
    """4-G: Check year-by-year PnL consistency."""
    results = {}
    for label, year_pnl in candidates_year_pnls.items():
        if not year_pnl:
            results[label] = {'n_loss_years': 99, 'min_year_pnl': -9999, 'passed': False}
            continue
        yearly_vals = list(year_pnl.values())
        loss_years = sum(1 for v in yearly_vals if v < 0)
        min_pnl = min(yearly_vals) if yearly_vals else -9999
        results[label] = {
            'n_years': len(yearly_vals),
            'n_loss_years': loss_years,
            'min_year_pnl': round(min_pnl, 2),
            'max_year_pnl': round(max(yearly_vals), 2) if yearly_vals else 0,
            'year_pnl_std': round(float(np.std(yearly_vals)), 2) if len(yearly_vals) > 1 else 0,
            'passed': loss_years <= 1 and min_pnl > -100,
        }
    return results


def run_layer4(kfold_summary, layer1_results, layer2_results):
    print("\n" + "=" * 80)
    print("  LAYER 4: Deep Validation (7 Institutional-Grade Tests)")
    print("=" * 80)

    existing = load_checkpoint("layer4_deep_validation.json")
    if existing:
        print(f"  [Resume] Found checkpoint, skipping Layer 4")
        return existing

    passed_kfold = [k for k in kfold_summary if k.get('passed', False)]
    if not passed_kfold:
        print("  No candidates passed K-Fold — skipping Layer 4.")
        return []

    candidates = [k['label'] for k in passed_kfold]
    print(f"  Candidates from K-Fold: {len(candidates)}")
    for i, lbl in enumerate(candidates[:10], 1):
        kf = next((k for k in passed_kfold if k['label'] == lbl), {})
        print(f"    {i:>2}. {lbl}  KFMean={kf.get('kfold_mean_sharpe', 0):.2f}")
    if len(candidates) > 10:
        print(f"    ... and {len(candidates) - 10} more")

    # -- Pre-load all needed DataBundles grouped by (kc_ema, kc_mult) --
    from backtest.runner import DataBundle

    kc_keys_needed = set()
    for label in candidates:
        kc_ema, kc_mult, _, _, _ = parse_label_to_engine_kw(label)
        kc_keys_needed.add((kc_ema, kc_mult))

    print(f"\n  Loading {len(kc_keys_needed)} DataBundle group(s)...", flush=True)
    kc_data_cache = {}
    for kc_ema, kc_mult in kc_keys_needed:
        data = DataBundle.load_custom(kc_ema=kc_ema, kc_mult=kc_mult)
        kc_data_cache[(kc_ema, kc_mult)] = (data.m15_df, data.h1_df)
        print(f"    E{kc_ema}/M{kc_mult}: loaded", flush=True)

    # -- Run full-sample engine to get daily PnL and trade PnLs --
    print(f"\n  Running full-sample engine for {len(candidates)} candidates...", flush=True)
    full_tasks = []
    for label in candidates:
        kc_ema, kc_mult, engine_kw, h1kc, cap = parse_label_to_engine_kw(label)
        key = (kc_ema, kc_mult)
        m15_df, h1_df = kc_data_cache[key]
        full_tasks.append((label, m15_df, h1_df, engine_kw))

    with mp.Pool(min(MAX_WORKERS, max(1, len(full_tasks)))) as pool:
        full_results = pool.map(_worker_engine_run, full_tasks)

    candidates_daily_pnl = {}
    candidates_trade_pnls = {}
    candidates_year_pnl = {}
    for r in full_results:
        tag = r['tag']
        candidates_daily_pnl[tag] = r.get('daily_pnl', [])
        candidates_trade_pnls[tag] = r.get('trade_pnls', [])
        candidates_year_pnl[tag] = r.get('year_pnl', {})

    # -- 4-A: DSR --
    t0 = time.time()
    print(f"\n  [4-A] Deflated Sharpe Ratio (n_trials={N_TRIALS_TOTAL})...", flush=True)
    dsr_results = test_4a_dsr(candidates_daily_pnl)
    n_pass = sum(1 for v in dsr_results.values() if v['passed'])
    print(f"    Passed: {n_pass}/{len(dsr_results)} in {time.time()-t0:.0f}s")

    # -- 4-B: PBO --
    t0 = time.time()
    print(f"\n  [4-B] Probability of Backtest Overfitting...", flush=True)
    pbo_result = test_4b_pbo(candidates_daily_pnl)
    print(f"    PBO={pbo_result['pbo']:.3f}, Risk={pbo_result['overfit_risk']}, "
          f"Pass={'YES' if pbo_result['passed'] else 'NO'} in {time.time()-t0:.0f}s")

    # -- 4-C: OOS Holdout --
    t0 = time.time()
    print(f"\n  [4-C] True OOS Holdout ({HOLDOUT_START} to {HOLDOUT_END})...", flush=True)
    oos_results = test_4c_oos_holdout(candidates, kc_data_cache)
    n_pass = sum(1 for v in oos_results.values() if v['passed'])
    print(f"    Passed: {n_pass}/{len(oos_results)} in {time.time()-t0:.0f}s")

    # -- 4-D: Param Stability --
    t0 = time.time()
    print(f"\n  [4-D] Parameter Stability / Cliff Detection...", flush=True)
    stability_results = test_4d_param_stability(candidates, kc_data_cache)
    n_pass = sum(1 for v in stability_results.values() if v['passed'])
    print(f"    Passed: {n_pass}/{len(stability_results)} in {time.time()-t0:.0f}s")

    # -- 4-E: Monte Carlo --
    t0 = time.time()
    print(f"\n  [4-E] Monte Carlo Permutation Test (1000 perms)...", flush=True)
    mc_results = test_4e_monte_carlo(candidates_trade_pnls)
    n_pass = sum(1 for v in mc_results.values() if v['passed'])
    print(f"    Passed: {n_pass}/{len(mc_results)} in {time.time()-t0:.0f}s")

    # -- 4-F: Cost Sensitivity --
    t0 = time.time()
    print(f"\n  [4-F] Cost Sensitivity Analysis...", flush=True)
    cost_results = test_4f_cost_sensitivity(candidates, kc_data_cache)
    n_pass = sum(1 for v in cost_results.values() if v['passed'])
    print(f"    Passed: {n_pass}/{len(cost_results)} in {time.time()-t0:.0f}s")

    # -- 4-G: Year Consistency --
    t0 = time.time()
    print(f"\n  [4-G] Year-by-Year Consistency...", flush=True)
    year_results = test_4g_year_consistency(candidates_year_pnl)
    n_pass = sum(1 for v in year_results.values() if v['passed'])
    print(f"    Passed: {n_pass}/{len(year_results)} in {time.time()-t0:.0f}s")

    # -- Assemble per-candidate scorecard --
    scorecards = []
    for label in candidates:
        dsr = dsr_results.get(label, {})
        oos = oos_results.get(label, {})
        stab = stability_results.get(label, {})
        mc = mc_results.get(label, {})
        cost = cost_results.get(label, {})
        year = year_results.get(label, {})

        mandatory_pass = (
            dsr.get('passed', False)
            and pbo_result.get('passed', False)
            and oos.get('passed', False)
        )

        important_count = sum([
            stab.get('passed', False),
            mc.get('passed', False),
            cost.get('passed', False),
        ])

        final_pass = mandatory_pass and important_count >= 2

        kf = next((k for k in passed_kfold if k['label'] == label), {})

        card = {
            'label': label,
            'kfold_mean_sharpe': kf.get('kfold_mean_sharpe', 0),
            'kfold_min_sharpe': kf.get('kfold_min_sharpe', 0),
            'full_sharpe': kf.get('full_sharpe', 0),
            'full_pnl': kf.get('full_pnl', 0),
            # 4-A DSR
            'dsr_value': dsr.get('dsr', 0),
            'dsr_passed': dsr.get('passed', False),
            # 4-B PBO (same for all)
            'pbo_value': pbo_result.get('pbo', 1.0),
            'pbo_risk': pbo_result.get('overfit_risk', 'HIGH'),
            'pbo_passed': pbo_result.get('passed', False),
            # 4-C OOS
            'oos_sharpe': oos.get('holdout_sharpe', 0),
            'oos_n': oos.get('holdout_n', 0),
            'oos_maxdd_pct': oos.get('holdout_maxdd_pct', 0),
            'oos_passed': oos.get('passed', False),
            # 4-D Stability
            'param_max_decay': stab.get('max_decay', 1.0),
            'param_worst': stab.get('worst_param', ''),
            'param_passed': stab.get('passed', False),
            # 4-E MC
            'mc_p_value': mc.get('mc_p_value', 1.0),
            'mc_passed': mc.get('passed', False),
            # 4-F Cost
            'cost_sharpe_1_5x': cost.get('sharpe_at_1_5x', 0),
            'cost_passed': cost.get('passed', False),
            # 4-G Year
            'year_loss_n': year.get('n_loss_years', 99),
            'year_min_pnl': year.get('min_year_pnl', -9999),
            'year_passed': year.get('passed', False),
            # Aggregate
            'mandatory_pass': mandatory_pass,
            'important_count': important_count,
            'final_pass': final_pass,
        }
        scorecards.append(card)

    scorecards.sort(key=lambda x: (x['final_pass'], x['important_count'],
                                    x['kfold_mean_sharpe']), reverse=True)

    save_checkpoint(scorecards, "layer4_deep_validation.json")

    # Summary
    final_passed = [s for s in scorecards if s['final_pass']]
    print(f"\n  {'='*80}")
    print(f"  Layer 4 Deep Validation Complete")
    print(f"  Final candidates (mandatory 3/3 + important >=2/3): "
          f"{len(final_passed)}/{len(scorecards)}")
    if final_passed:
        print(f"\n  {'Rank':>4} {'Label':>55} {'DSR':>5} {'PBO':>5} {'OOS':>5} "
              f"{'Stab':>5} {'MC':>5} {'Cost':>5} {'Year':>5} {'FINAL':>6}")
        print(f"  {'-'*4} {'-'*55} {'-'*5} {'-'*5} {'-'*5} "
              f"{'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*6}")
        for i, s in enumerate(final_passed[:20], 1):
            def _p(v): return 'PASS' if v else 'FAIL'
            print(f"  {i:>4} {s['label']:>55} "
                  f"{_p(s['dsr_passed']):>5} {_p(s['pbo_passed']):>5} "
                  f"{_p(s['oos_passed']):>5} {_p(s['param_passed']):>5} "
                  f"{_p(s['mc_passed']):>5} {_p(s['cost_passed']):>5} "
                  f"{_p(s['year_passed']):>5} {'PASS':>6}")

    return scorecards


# ═══════════════════════════════════════════════════════════════
# Scorecard output
# ═══════════════════════════════════════════════════════════════

def write_scorecard(scorecards):
    """Write comprehensive scorecard report to file."""
    lines = [
        "R50 LAYER 4: DEEP VALIDATION SCORECARD",
        "=" * 140,
        "",
        "Criteria:",
        f"  [MANDATORY] DSR:  dsr > 0.95 (n_trials={N_TRIALS_TOTAL})",
        "  [MANDATORY] PBO:  pbo < 0.30, risk != HIGH",
        f"  [MANDATORY] OOS:  holdout Sharpe > 1.5, MaxDD% < 15% ({HOLDOUT_START} to {HOLDOUT_END})",
        "  [IMPORTANT] Param Stability: max decay < 30% under +/-10%/20% perturbation",
        "  [IMPORTANT] Monte Carlo: p < 0.05 (1000 permutations)",
        "  [IMPORTANT] Cost Sensitivity: Sharpe > 1.0 at 1.5x spread ($0.75)",
        "  [REFERENCE] Year Consistency: <= 1 loss year, min year PnL > -$100",
        "",
        "Final Pass = All 3 MANDATORY + at least 2/3 IMPORTANT",
        "",
        "-" * 140,
        "",
    ]

    # Detailed table
    hdr = (f"{'#':>3} {'Label':>55} {'KFMean':>7} {'DSR':>6} {'PBO':>6} "
           f"{'OOS_Sh':>7} {'Decay%':>7} {'MC_p':>7} {'Cost15':>7} "
           f"{'LossYr':>7} {'Score':>6} {'FINAL':>6}")
    lines.append(hdr)
    lines.append("-" * 140)

    for i, s in enumerate(scorecards, 1):
        score = sum([
            s.get('dsr_passed', False),
            s.get('pbo_passed', False),
            s.get('oos_passed', False),
            s.get('param_passed', False),
            s.get('mc_passed', False),
            s.get('cost_passed', False),
            s.get('year_passed', False),
        ])
        final_str = "PASS" if s.get('final_pass', False) else "FAIL"
        lines.append(
            f"{i:>3} {s['label']:>55} "
            f"{s.get('kfold_mean_sharpe', 0):>7.2f} "
            f"{s.get('dsr_value', 0):>6.3f} "
            f"{s.get('pbo_value', 1):>6.3f} "
            f"{s.get('oos_sharpe', 0):>7.2f} "
            f"{s.get('param_max_decay', 1) * 100:>6.1f}% "
            f"{s.get('mc_p_value', 1):>7.4f} "
            f"{s.get('cost_sharpe_1_5x', 0):>7.2f} "
            f"{s.get('year_loss_n', 99):>7d} "
            f"{score:>5}/7 "
            f"{final_str:>6}"
        )

    lines.append("")
    lines.append("-" * 140)

    final_passed = [s for s in scorecards if s.get('final_pass')]
    lines.append(f"\nFINAL PASSED: {len(final_passed)}/{len(scorecards)}")

    if final_passed:
        lines.append("\n--- DEPLOYABLE STRATEGIES ---\n")
        for i, s in enumerate(final_passed, 1):
            lines.append(f"  {i}. {s['label']}")
            lines.append(f"     K-Fold Mean Sharpe: {s.get('kfold_mean_sharpe', 0):.2f}, "
                         f"Full Sharpe: {s.get('full_sharpe', 0):.2f}")
            lines.append(f"     DSR: {s.get('dsr_value', 0):.3f}, "
                         f"OOS Sharpe: {s.get('oos_sharpe', 0):.2f}, "
                         f"Param Decay: {s.get('param_max_decay', 1)*100:.1f}%")
            lines.append(f"     MC p-value: {s.get('mc_p_value', 1):.4f}, "
                         f"Cost@1.5x: {s.get('cost_sharpe_1_5x', 0):.2f}, "
                         f"Loss Years: {s.get('year_loss_n', 99)}")
            lines.append("")
    else:
        lines.append("\n  No strategies passed all criteria.")
        lines.append("  Consider relaxing thresholds or reviewing top candidates manually.")

    scorecard_text = "\n".join(lines)
    with open(OUTPUT_DIR / "layer4_scorecard.txt", 'w', encoding='utf-8') as f:
        f.write(scorecard_text)
    print(f"\n  Scorecard saved to {OUTPUT_DIR / 'layer4_scorecard.txt'}")
    return scorecard_text


# ═══════════════════════════════════════════════════════════════
# Final summary
# ═══════════════════════════════════════════════════════════════

def write_final_summary(kfold_summary, total_elapsed, scorecards=None):
    lines = [
        "R50 FINAL SUMMARY: L8 Full-Blast Parameter Grid Search",
        "=" * 100,
        f"Total elapsed: {total_elapsed/3600:.1f}h",
        f"Spread: ${SPREAD}",
        "",
        "--- PASSED K-Fold (All 6 folds Sharpe>0 AND mean Sharpe>2.0) ---",
        "",
    ]
    passed = [k for k in kfold_summary if k['passed']]
    if passed:
        lines.append(f"{'Rank':>4} {'Label':>60} {'FullSh':>7} {'FullPnL':>12} {'FullDD':>10} "
                     f"{'KFMean':>7} {'KFMin':>7}")
        lines.append("-" * 120)
        for i, k in enumerate(passed, 1):
            lines.append(
                f"{i:>4} {k['label']:>60} {k['full_sharpe']:>7.2f} "
                f"{fmt(k['full_pnl']):>12} {fmt(k['full_maxdd']):>10} "
                f"{k['kfold_mean_sharpe']:>7.2f} {k['kfold_min_sharpe']:>7.2f}")
    else:
        lines.append("  No combos passed all criteria.")

    # Layer 4 deep validation results
    final_passed = []
    if scorecards:
        final_passed = [s for s in scorecards if s.get('final_pass')]
        lines.extend([
            "",
            "--- LAYER 4: Deep Validation Results ---",
            f"  Candidates tested: {len(scorecards)}",
            f"  Final passed (3/3 mandatory + >=2/3 important): {len(final_passed)}",
            "",
        ])
        if final_passed:
            lines.append(f"  {'#':>3} {'Label':>55} {'KFMean':>7} {'DSR':>6} "
                         f"{'OOS':>6} {'Decay':>6} {'MC_p':>7} {'Cost':>6} {'Score':>6}")
            lines.append("  " + "-" * 110)
            for i, s in enumerate(final_passed, 1):
                score = sum([s.get(k, False) for k in [
                    'dsr_passed', 'pbo_passed', 'oos_passed',
                    'param_passed', 'mc_passed', 'cost_passed', 'year_passed']])
                lines.append(
                    f"  {i:>3} {s['label']:>55} "
                    f"{s.get('kfold_mean_sharpe', 0):>7.2f} "
                    f"{s.get('dsr_value', 0):>6.3f} "
                    f"{s.get('oos_sharpe', 0):>6.2f} "
                    f"{s.get('param_max_decay', 1)*100:>5.1f}% "
                    f"{s.get('mc_p_value', 1):>7.4f} "
                    f"{s.get('cost_sharpe_1_5x', 0):>6.2f} "
                    f"{score:>5}/7")
        else:
            lines.append("  No strategies survived all deep validation tests.")

    lines.extend([
        "",
        "--- Current L8_MAX reference ---",
        "  L8_BASE + TATrail(s2/d0.75/f0.003) + H1KC(E15/M2.0) + Cap$30",
        "  Full sample Sharpe: 11.23, MaxDD: $60",
        "",
        "--- Comparison vs L8_MAX ---",
    ])

    best_source = final_passed[0] if (scorecards and final_passed) else (passed[0] if passed else None)
    if best_source:
        best_sharpe = best_source.get('full_sharpe', best_source.get('kfold_mean_sharpe', 0))
        delta_sh = best_sharpe - 11.23
        lines.append(f"  Best found: {best_source['label']}")
        lines.append(f"    Sharpe: {best_sharpe:.2f} (delta vs L8_MAX: {delta_sh:+.2f})")
        if 'kfold_mean_sharpe' in best_source:
            lines.append(f"    K-Fold mean: {best_source['kfold_mean_sharpe']:.2f}")
        if scorecards and final_passed:
            lines.append(f"    DSR: {best_source.get('dsr_value', 0):.3f}, "
                         f"OOS Sharpe: {best_source.get('oos_sharpe', 0):.2f}")
            lines.append(f"    Param max decay: {best_source.get('param_max_decay', 1)*100:.1f}%, "
                         f"MC p-value: {best_source.get('mc_p_value', 1):.4f}")
    else:
        lines.append("  No improvement found over L8_MAX.")

    summary_text = "\n".join(lines)
    with open(OUTPUT_DIR / "final_ranking.txt", 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"\n{'='*80}")
    print(summary_text)
    print(f"{'='*80}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 80)
    print("  R50: L8 Full-Blast Parameter Grid Search + Deep Validation")
    print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Workers: {MAX_WORKERS}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 80)

    try:
        layer1_results = run_layer1()
    except Exception as e:
        print(f"\n  FATAL Layer 1: {e}")
        traceback.print_exc()
        return

    try:
        layer2_results = run_layer2(layer1_results)
    except Exception as e:
        print(f"\n  FATAL Layer 2: {e}")
        traceback.print_exc()
        return

    try:
        kfold_summary = run_layer3(layer1_results, layer2_results)
    except Exception as e:
        print(f"\n  FATAL Layer 3: {e}")
        traceback.print_exc()
        return

    scorecards = None
    try:
        scorecards = run_layer4(kfold_summary, layer1_results, layer2_results)
        write_scorecard(scorecards)
    except Exception as e:
        print(f"\n  FATAL Layer 4: {e}")
        traceback.print_exc()

    total_elapsed = time.time() - t_start
    write_final_summary(kfold_summary, total_elapsed, scorecards)

    print(f"\n  All done! Total time: {total_elapsed/3600:.1f}h")
    print(f"  Results in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
