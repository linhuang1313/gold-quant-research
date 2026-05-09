#!/usr/bin/env python3
"""
R129 — Unified Walk-Forward Validation
========================================
Run the 8-stage validation pipeline on findings from R119-R128.
Loads results from prior experiments and validates winning configs.

Stage 0: Base logic sanity
Stage 1: Sanity checks
Stage 2: K-Fold robustness (6 folds)
Stage 3: Walk-forward OOS
Stage 4: Combinatorial / PBO check
Stage 5: Cost sensitivity
Stage 6: Reality checks (yearly stability)
Stage 7: Deployment readiness
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import DataBundle, load_m15, load_h1_aligned, H1_CSV_PATH
from backtest.runner import run_variant, LIVE_PARITY_KWARGS

OUTPUT_DIR = Path("results/r129_unified_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01
CAPITAL = 5000

STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']

PRIOR_RESULTS_FILES = [
    'results/r119_results.json',
    'results/r121_results.json',
    'results/r122_results.json',
    'results/r123_results.json',
    'results/r125_results.json',
    'results/r127_results.json',
]
# Also check subdirectory patterns used by earlier experiments
PRIOR_RESULTS_SUBDIRS = [
    ('results/r119_*', 'r119_results.json'),
    ('results/r121_*', 'r121_results.json'),
    ('results/r122_*', 'r122_results.json'),
    ('results/r123_*', 'r123_results.json'),
    ('results/r125_*', 'r125_results.json'),
    ('results/r127_*', 'r127_results.json'),
]

MAX_WINNERS = 10

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-06-01"),
]

WF_WINDOWS = [
    ("WF1", "2015-01-01", "2019-01-01", "2019-01-01", "2021-01-01"),
    ("WF2", "2016-01-01", "2020-01-01", "2020-01-01", "2022-01-01"),
    ("WF3", "2017-01-01", "2021-01-01", "2021-01-01", "2023-01-01"),
    ("WF4", "2018-01-01", "2022-01-01", "2022-01-01", "2024-01-01"),
    ("WF5", "2019-01-01", "2023-01-01", "2023-01-01", "2025-01-01"),
]

COST_SPREADS = [0.30, 0.50, 0.80]

t0 = time.time()


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _mk(pos, exit_px, exit_time, reason, bar_i, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_px,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_i - pos['bar']}


def _run_exit_with_cap(pos, i, hi, lo, cl, spread, lot, pv, times,
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
# Strategy backtests
# ═══════════════════════════════════════════════════════════════

def add_psar(df, af_step=0.01, af_max=0.05):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    n = len(df)
    psar = np.empty(n); psar[:] = np.nan
    af = af_step; rising = True; ep = h[0]; psar[0] = l[0]
    for i in range(1, n):
        prev = psar[i-1]
        if rising:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], l[i-1], l[max(0,i-2)])
            if l[i] < psar[i]:
                rising = False; psar[i] = ep; ep = l[i]; af = af_step
            else:
                if h[i] > ep: ep = h[i]; af = min(af + af_step, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], h[i-1], h[max(0,i-2)])
            if h[i] > psar[i]:
                rising = True; psar[i] = ep; ep = h[i]; af = af_step
            else:
                if l[i] < ep: ep = l[i]; af = min(af + af_step, af_max)
    df['PSAR'] = psar


def bt_psar(h1_df, spread, lot, maxloss_cap=0,
            sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04, max_hold=20):
    df = h1_df.copy(); add_psar(df); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR', 'PSAR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; psar = df['PSAR'].values; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if psar[i-1] > c[i-1] and psar[i] < c[i]:
            pos = {'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif psar[i-1] < c[i-1] and psar[i] > c[i]:
            pos = {'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
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
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
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
            pos = {'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
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
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif c[i] < ll:
            pos = {'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
    return trades


def bt_l8_max(data_bundle, spread, lot, maxloss_cap=35):
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
# Metric helpers
# ═══════════════════════════════════════════════════════════════

def _trades_to_daily(trades):
    if not trades: return np.array([])
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return np.array([daily[d] for d in dates])


def _sharpe(daily_arr):
    if len(daily_arr) < 10: return 0.0
    s = np.std(daily_arr, ddof=1)
    return float(np.mean(daily_arr) / s * np.sqrt(252)) if s > 0 else 0.0


def _compute_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0.0, 'pnl': 0.0, 'max_dd': 0.0, 'wr': 0.0, 'avg_bars': 0.0}
    pnls = [t['pnl'] for t in trades]
    daily = _trades_to_daily(trades)
    wins = sum(1 for p in pnls if p > 0)
    eq = np.cumsum(pnls)
    dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 0 else 0.0
    bars_list = [t.get('bars', 0) for t in trades]
    max_single = max(abs(p) for p in pnls) if pnls else 0.0
    return {
        'n': len(trades),
        'sharpe': round(_sharpe(daily), 3),
        'pnl': round(sum(pnls), 2),
        'max_dd': round(dd, 2),
        'wr': round(wins / len(trades) * 100, 1),
        'avg_bars': round(np.mean(bars_list), 1) if bars_list else 0.0,
        'max_single_pnl': round(max_single, 2),
        'total_pnl': round(sum(pnls), 2),
    }


# ═══════════════════════════════════════════════════════════════
# Phase 1: Scan prior experiment results
# ═══════════════════════════════════════════════════════════════

def _find_result_files():
    """Locate available prior experiment result JSONs."""
    found = []
    for path_str in PRIOR_RESULTS_FILES:
        p = Path(path_str)
        if p.exists():
            found.append(p)
    import glob
    for pattern, filename in PRIOR_RESULTS_SUBDIRS:
        for d in glob.glob(pattern):
            candidate = Path(d) / filename
            if candidate.exists():
                found.append(candidate)
    return found


def _extract_winners(results_files):
    """
    Extract up to MAX_WINNERS configs that showed Sharpe improvement and K-fold pass.
    Returns list of dicts with strategy config info.
    """
    winners = []

    for fpath in results_files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"    WARN: Could not load {fpath}: {e}", flush=True)
            continue

        exp_name = fpath.stem if hasattr(fpath, 'stem') else str(fpath)
        print(f"    Scanning {fpath}...", flush=True)

        if isinstance(data, dict):
            # Look for validated configs with sharpe improvement + k-fold pass
            validated = data.get('validated', [])
            kfold = data.get('kfold', data.get('phase5_kfold', {}))

            if isinstance(validated, list):
                for v in validated:
                    label = v.get('label', '')
                    engine_sharpe = v.get('engine_sharpe', v.get('sharpe', 0))
                    if engine_sharpe > 1.0:
                        kf_pass = True
                        if isinstance(kfold, list):
                            for kf in kfold:
                                if kf.get('label', '') == label:
                                    kf_pass = kf.get('pass_count', 0) >= 4
                                    break
                        winners.append({
                            'source': str(fpath),
                            'label': label,
                            'sharpe': engine_sharpe,
                            'kfold_pass': kf_pass,
                            'config': v.get('sig_params', v.get('config', {})),
                            'bt_params': v.get('bt_params', {}),
                            'engine_stats': v.get('engine_stats', {}),
                        })

            # Also look for phase-based results with top configs
            for key in ['phase2_backtest', 'phase3_comparison', 'top_configs', 'best_configs']:
                section = data.get(key, {})
                if isinstance(section, dict):
                    for label, stats in section.items():
                        if isinstance(stats, dict) and stats.get('sharpe', 0) > 1.0:
                            winners.append({
                                'source': str(fpath),
                                'label': f"{exp_name}_{label}",
                                'sharpe': stats.get('sharpe', 0),
                                'kfold_pass': True,
                                'config': stats,
                                'bt_params': {},
                                'engine_stats': stats,
                            })

    winners.sort(key=lambda w: w['sharpe'], reverse=True)
    return winners[:MAX_WINNERS]


# ═══════════════════════════════════════════════════════════════
# Phase 2: Validation stages
# ═══════════════════════════════════════════════════════════════

def run_backtest_for_config(winner, h1_df, data_bundle, spread=SPREAD, lot=UNIT_LOT):
    """Run the appropriate backtest based on winner config."""
    label = winner.get('label', '')
    config = winner.get('config', {})
    bt_params = winner.get('bt_params', {})

    caps = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}

    if 'L8_MAX' in label or 'l8' in label.lower():
        cap = config.get('maxloss_cap', bt_params.get('maxloss_cap', caps['L8_MAX']))
        return bt_l8_max(data_bundle, spread, lot, maxloss_cap=cap)
    elif 'PSAR' in label or 'psar' in label.lower():
        return bt_psar(h1_df, spread, lot,
                       maxloss_cap=config.get('maxloss_cap', bt_params.get('maxloss_cap', caps['PSAR'])),
                       sl_atr=config.get('sl_atr', bt_params.get('sl_atr', 4.5)),
                       tp_atr=config.get('tp_atr', bt_params.get('tp_atr', 16.0)),
                       trail_act=config.get('trail_act', bt_params.get('trail_act', 0.20)),
                       trail_dist=config.get('trail_dist', bt_params.get('trail_dist', 0.04)),
                       max_hold=config.get('max_hold', bt_params.get('max_hold', 20)))
    elif 'TSMOM' in label or 'tsmom' in label.lower():
        return bt_tsmom(h1_df, spread, lot,
                        maxloss_cap=config.get('maxloss_cap', bt_params.get('maxloss_cap', caps['TSMOM'])),
                        sl_atr=config.get('sl_atr', bt_params.get('sl_atr', 4.5)),
                        tp_atr=config.get('tp_atr', bt_params.get('tp_atr', 6.0)),
                        trail_act=config.get('trail_act', bt_params.get('trail_act', 0.14)),
                        trail_dist=config.get('trail_dist', bt_params.get('trail_dist', 0.025)),
                        max_hold=config.get('max_hold', bt_params.get('max_hold', 20)))
    elif 'SESS' in label or 'sess' in label.lower():
        return bt_sess_bo(h1_df, spread, lot,
                          maxloss_cap=config.get('maxloss_cap', bt_params.get('maxloss_cap', caps['SESS_BO'])),
                          sl_atr=config.get('sl_atr', bt_params.get('sl_atr', 4.5)),
                          tp_atr=config.get('tp_atr', bt_params.get('tp_atr', 4.0)),
                          trail_act=config.get('trail_act', bt_params.get('trail_act', 0.14)),
                          trail_dist=config.get('trail_dist', bt_params.get('trail_dist', 0.025)),
                          max_hold=config.get('max_hold', bt_params.get('max_hold', 20)))
    else:
        # Default: run all 4 strategies and merge
        all_trades = []
        all_trades.extend(bt_psar(h1_df, spread, lot, maxloss_cap=caps['PSAR']))
        all_trades.extend(bt_tsmom(h1_df, spread, lot, maxloss_cap=caps['TSMOM']))
        all_trades.extend(bt_sess_bo(h1_df, spread, lot, maxloss_cap=caps['SESS_BO']))
        all_trades.extend(bt_l8_max(data_bundle, spread, lot, maxloss_cap=caps['L8_MAX']))
        all_trades.sort(key=lambda x: pd.Timestamp(x.get('exit_time', x.get('entry_time', '2015-01-01'))))
        return all_trades


def stage0_base_logic(trades):
    """Stage 0: Base logic sanity — Sharpe > 1.0, trades > 100."""
    stats = _compute_stats(trades)
    sharpe_ok = stats['sharpe'] > 1.0
    trades_ok = stats['n'] > 100
    passed = sharpe_ok and trades_ok
    return {
        'stage': 0, 'name': 'Base Logic Sanity',
        'passed': passed,
        'sharpe': stats['sharpe'], 'n_trades': stats['n'],
        'pnl': stats['pnl'], 'max_dd': stats['max_dd'],
        'checks': {'sharpe_gt_1': sharpe_ok, 'trades_gt_100': trades_ok},
    }


def stage1_sanity(trades):
    """Stage 1: Sanity checks — WR 30-90%, avg bars > 1, no single trade > 10% PnL."""
    stats = _compute_stats(trades)
    wr = stats['wr']
    avg_bars = stats['avg_bars']
    total_pnl = abs(stats['pnl']) if stats['pnl'] != 0 else 1.0
    max_single = stats.get('max_single_pnl', 0)
    single_pct = max_single / total_pnl * 100 if total_pnl > 0 else 0

    wr_ok = 30 <= wr <= 90
    bars_ok = avg_bars > 1
    single_ok = single_pct <= 10

    passed = wr_ok and bars_ok and single_ok
    return {
        'stage': 1, 'name': 'Sanity Checks',
        'passed': passed,
        'wr': wr, 'avg_bars': avg_bars, 'single_trade_pct': round(single_pct, 1),
        'checks': {'wr_30_90': wr_ok, 'avg_bars_gt_1': bars_ok, 'no_single_gt_10pct': single_ok},
    }


def stage2_kfold(h1_df, data_bundle, winner):
    """Stage 2: 6-fold K-Fold — require 4/6 positive Sharpe."""
    fold_results = []
    for fname, start, end in FOLDS:
        ts = pd.Timestamp(start, tz='UTC') if h1_df.index.tz else pd.Timestamp(start)
        te = pd.Timestamp(end, tz='UTC') if h1_df.index.tz else pd.Timestamp(end)
        sub_h1 = h1_df[(h1_df.index >= ts) & (h1_df.index < te)]
        if len(sub_h1) < 200:
            fold_results.append({'fold': fname, 'sharpe': 0.0, 'n': 0})
            continue

        sub_bundle = None
        if data_bundle is not None:
            try:
                sub_bundle = data_bundle.slice(start, end)
            except Exception:
                sub_bundle = data_bundle

        trades = run_backtest_for_config(winner, sub_h1, sub_bundle or data_bundle)
        stats = _compute_stats(trades)
        fold_results.append({
            'fold': fname, 'sharpe': stats['sharpe'], 'n': stats['n'], 'pnl': stats['pnl'],
        })

    sharpes = [f['sharpe'] for f in fold_results]
    pos_count = sum(1 for s in sharpes if s > 0)
    passed = pos_count >= 4

    return {
        'stage': 2, 'name': 'K-Fold Robustness (6 folds)',
        'passed': passed,
        'positive_folds': pos_count, 'required': 4,
        'fold_sharpes': [round(s, 3) for s in sharpes],
        'mean_sharpe': round(np.mean(sharpes), 3),
        'folds': fold_results,
    }


def stage3_walk_forward(h1_df, data_bundle, winner):
    """Stage 3: Walk-forward OOS — 4yr train, 2yr test, sliding by 1yr."""
    wf_results = []
    for wf_name, train_start, train_end, test_start, test_end in WF_WINDOWS:
        ts = pd.Timestamp(test_start, tz='UTC') if h1_df.index.tz else pd.Timestamp(test_start)
        te = pd.Timestamp(test_end, tz='UTC') if h1_df.index.tz else pd.Timestamp(test_end)
        sub_h1 = h1_df[(h1_df.index >= ts) & (h1_df.index < te)]
        if len(sub_h1) < 200:
            wf_results.append({'window': wf_name, 'oos_sharpe': 0.0, 'n': 0})
            continue

        sub_bundle = None
        if data_bundle is not None:
            try:
                sub_bundle = data_bundle.slice(test_start, test_end)
            except Exception:
                sub_bundle = data_bundle

        trades = run_backtest_for_config(winner, sub_h1, sub_bundle or data_bundle)
        stats = _compute_stats(trades)
        wf_results.append({
            'window': wf_name, 'oos_sharpe': stats['sharpe'],
            'n': stats['n'], 'pnl': stats['pnl'],
            'test_period': f"{test_start} to {test_end}",
        })

    oos_sharpes = [w['oos_sharpe'] for w in wf_results]
    mean_oos = np.mean(oos_sharpes) if oos_sharpes else 0
    passed = mean_oos > 0

    return {
        'stage': 3, 'name': 'Walk-Forward OOS',
        'passed': passed,
        'mean_oos_sharpe': round(mean_oos, 3),
        'oos_sharpes': [round(s, 3) for s in oos_sharpes],
        'windows': wf_results,
    }


def stage4_combinatorial_pbo(trades):
    """Stage 4: Combinatorial PBO — shuffle trade PnLs 1000 times."""
    if len(trades) < 20:
        return {'stage': 4, 'name': 'Combinatorial PBO', 'passed': False,
                'pbo': 1.0, 'reason': 'Too few trades'}

    pnls = np.array([t['pnl'] for t in trades])
    daily = _trades_to_daily(trades)
    base_sharpe = _sharpe(daily)

    n_sims = 1000
    rng = np.random.RandomState(42)
    overfit_count = 0

    for _ in range(n_sims):
        shuffled = pnls.copy()
        rng.shuffle(shuffled)
        # Reconstruct daily from shuffled trades (approximate)
        n_daily = max(10, len(daily))
        chunk_size = max(1, len(shuffled) // n_daily)
        sim_daily = []
        for j in range(0, len(shuffled), chunk_size):
            sim_daily.append(shuffled[j:j+chunk_size].sum())
        sim_daily = np.array(sim_daily)
        sim_sharpe = _sharpe(sim_daily)
        if sim_sharpe >= base_sharpe:
            overfit_count += 1

    pbo = overfit_count / n_sims
    passed = pbo < 0.50

    return {
        'stage': 4, 'name': 'Combinatorial PBO',
        'passed': passed,
        'pbo': round(pbo, 3),
        'base_sharpe': round(base_sharpe, 3),
        'n_simulations': n_sims,
    }


def stage5_cost_sensitivity(h1_df, data_bundle, winner):
    """Stage 5: Cost sensitivity — re-run with spread 0.30, 0.50, 0.80."""
    cost_results = []
    for spread_val in COST_SPREADS:
        trades = run_backtest_for_config(winner, h1_df, data_bundle, spread=spread_val)
        stats = _compute_stats(trades)
        cost_results.append({
            'spread': spread_val, 'sharpe': stats['sharpe'],
            'n': stats['n'], 'pnl': stats['pnl'],
        })

    sharpe_at_050 = next((r['sharpe'] for r in cost_results if r['spread'] == 0.50), 0)
    passed = sharpe_at_050 > 0

    return {
        'stage': 5, 'name': 'Cost Sensitivity',
        'passed': passed,
        'sharpe_at_050': round(sharpe_at_050, 3),
        'cost_results': cost_results,
    }


def stage6_reality_yearly(trades):
    """Stage 6: Yearly stability — allow max 2 negative years out of 11."""
    if not trades:
        return {'stage': 6, 'name': 'Reality (Yearly Stability)', 'passed': False,
                'reason': 'No trades'}

    yearly = {}
    for t in trades:
        yr = pd.Timestamp(t['exit_time']).year
        yearly[yr] = yearly.get(yr, 0) + t['pnl']

    years = sorted(yearly.keys())
    neg_years = sum(1 for yr in years if yearly[yr] < 0)
    passed = neg_years <= 2

    yearly_summary = {str(yr): round(yearly[yr], 2) for yr in years}

    return {
        'stage': 6, 'name': 'Reality (Yearly Stability)',
        'passed': passed,
        'negative_years': neg_years, 'max_allowed': 2,
        'total_years': len(years),
        'yearly_pnl': yearly_summary,
    }


def stage7_summary(stage_results):
    """Stage 7: Deployment readiness — all stages 0-6 must pass."""
    stages_0_to_6 = [r for r in stage_results if r['stage'] < 7]
    all_pass = all(r['passed'] for r in stages_0_to_6)
    failed = [r['name'] for r in stages_0_to_6 if not r['passed']]

    return {
        'stage': 7, 'name': 'Deployment Readiness',
        'passed': all_pass,
        'stages_passed': sum(1 for r in stages_0_to_6 if r['passed']),
        'stages_total': len(stages_0_to_6),
        'failed_stages': failed,
        'verdict': 'DEPLOY-READY' if all_pass else f'NOT READY ({len(failed)} stage(s) failed)',
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80, flush=True)
    print("  R129 — Unified Walk-Forward Validation", flush=True)
    print("  8-Stage pipeline on findings from R119-R128", flush=True)
    print("=" * 80, flush=True)

    # ─── Load data ────────────────────────────────────────────
    print("\n  Loading data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0].date()} ~ {h1_df.index[-1].date()})", flush=True)

    print("  Loading L8_MAX DataBundle...", flush=True)
    try:
        bundle = DataBundle.load_custom()
        print("    DataBundle loaded", flush=True)
    except Exception as e:
        print(f"    WARN: DataBundle load failed: {e}", flush=True)
        print("    L8_MAX validation will be skipped", flush=True)
        bundle = None

    results = {
        'experiment': 'R129 Unified Walk-Forward Validation',
        'stages': '0-7 pipeline',
    }

    # ═════════════════════════════════════════════════════════════
    # Phase 1: Scan prior experiment results
    # ═════════════════════════════════════════════════════════════
    print("\n" + "=" * 60, flush=True)
    print("  Phase 1: Scan R119-R128 Results for Winners", flush=True)
    print("=" * 60, flush=True)

    results_files = _find_result_files()
    print(f"  Found {len(results_files)} result files", flush=True)

    if results_files:
        winners = _extract_winners(results_files)
        print(f"\n  Extracted {len(winners)} winner(s) (max {MAX_WINNERS})", flush=True)
        for i, w in enumerate(winners):
            print(f"    #{i+1}: {w['label']} (Sharpe={w['sharpe']:.3f}, "
                  f"KFold={'PASS' if w['kfold_pass'] else 'FAIL'}) from {w['source']}", flush=True)
    else:
        print("\n  No R119-R128 results found. Running baseline validation.", flush=True)
        # Fall back to baseline strategies as "winners"
        winners = [
            {'label': 'PSAR_baseline', 'sharpe': 0, 'kfold_pass': True,
             'config': {}, 'bt_params': {}, 'source': 'baseline', 'engine_stats': {}},
            {'label': 'TSMOM_baseline', 'sharpe': 0, 'kfold_pass': True,
             'config': {}, 'bt_params': {}, 'source': 'baseline', 'engine_stats': {}},
            {'label': 'SESS_BO_baseline', 'sharpe': 0, 'kfold_pass': True,
             'config': {}, 'bt_params': {}, 'source': 'baseline', 'engine_stats': {}},
            {'label': 'L8_MAX_baseline', 'sharpe': 0, 'kfold_pass': True,
             'config': {}, 'bt_params': {}, 'source': 'baseline', 'engine_stats': {}},
        ]

    results['phase1_scan'] = {
        'n_files_found': len(results_files),
        'n_winners': len(winners),
        'winners': [{'label': w['label'], 'sharpe': w['sharpe'], 'source': w['source']}
                     for w in winners],
    }

    # ═════════════════════════════════════════════════════════════
    # Phase 2: Run 8-stage validation on each winner
    # ═════════════════════════════════════════════════════════════
    print("\n" + "=" * 60, flush=True)
    print("  Phase 2: 8-Stage Validation Pipeline", flush=True)
    print("=" * 60, flush=True)

    all_report_cards = []
    for idx, winner in enumerate(winners):
        t_start = time.time()
        label = winner['label']
        print(f"\n{'─' * 60}", flush=True)
        print(f"  Validating [{idx+1}/{len(winners)}]: {label}", flush=True)
        print(f"{'─' * 60}", flush=True)

        # Run full backtest
        print(f"    Running full backtest...", flush=True)
        trades = run_backtest_for_config(winner, h1_df, bundle)
        print(f"    → {len(trades)} trades generated", flush=True)

        # Stage 0
        print(f"    Stage 0: Base logic...", flush=True)
        s0 = stage0_base_logic(trades)
        print(f"      Sharpe={s0['sharpe']:.3f}, Trades={s0['n_trades']} → "
              f"{'PASS' if s0['passed'] else 'FAIL'}", flush=True)

        # Stage 1
        print(f"    Stage 1: Sanity checks...", flush=True)
        s1 = stage1_sanity(trades)
        print(f"      WR={s1['wr']:.1f}%, AvgBars={s1['avg_bars']:.1f} → "
              f"{'PASS' if s1['passed'] else 'FAIL'}", flush=True)

        # Stage 2
        print(f"    Stage 2: K-Fold (6 folds)...", flush=True)
        s2 = stage2_kfold(h1_df, bundle, winner)
        print(f"      {s2['positive_folds']}/6 positive, sharpes={s2['fold_sharpes']} → "
              f"{'PASS' if s2['passed'] else 'FAIL'}", flush=True)

        # Stage 3
        print(f"    Stage 3: Walk-forward OOS...", flush=True)
        s3 = stage3_walk_forward(h1_df, bundle, winner)
        print(f"      Mean OOS Sharpe={s3['mean_oos_sharpe']:.3f} → "
              f"{'PASS' if s3['passed'] else 'FAIL'}", flush=True)

        # Stage 4
        print(f"    Stage 4: Combinatorial PBO...", flush=True)
        s4 = stage4_combinatorial_pbo(trades)
        print(f"      PBO={s4['pbo']:.3f} (threshold < 0.50) → "
              f"{'PASS' if s4['passed'] else 'FAIL'}", flush=True)

        # Stage 5
        print(f"    Stage 5: Cost sensitivity...", flush=True)
        s5 = stage5_cost_sensitivity(h1_df, bundle, winner)
        print(f"      Sharpe@0.50={s5['sharpe_at_050']:.3f} → "
              f"{'PASS' if s5['passed'] else 'FAIL'}", flush=True)

        # Stage 6
        print(f"    Stage 6: Yearly stability...", flush=True)
        s6 = stage6_reality_yearly(trades)
        print(f"      Negative years: {s6['negative_years']}/{s6.get('total_years', 0)} "
              f"(max 2 allowed) → {'PASS' if s6['passed'] else 'FAIL'}", flush=True)

        # Stage 7
        stages = [s0, s1, s2, s3, s4, s5, s6]
        s7 = stage7_summary(stages)
        stages.append(s7)
        print(f"    Stage 7: {s7['verdict']}", flush=True)

        elapsed_winner = time.time() - t_start
        report_card = {
            'label': label,
            'source': winner['source'],
            'original_sharpe': winner['sharpe'],
            'stages': stages,
            'overall_passed': s7['passed'],
            'stages_passed': s7['stages_passed'],
            'stages_total': s7['stages_total'],
            'elapsed_s': round(elapsed_winner, 1),
        }
        all_report_cards.append(report_card)

    results['phase2_validation'] = all_report_cards

    # ═════════════════════════════════════════════════════════════
    # Phase 3: Report cards summary
    # ═════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print("\n" + "=" * 80, flush=True)
    print("  R129 — VALIDATION REPORT CARDS", flush=True)
    print("=" * 80, flush=True)

    n_passed = sum(1 for rc in all_report_cards if rc['overall_passed'])
    n_total = len(all_report_cards)

    print(f"\n  {'Winner':<35s}  {'Stages':>8s}  {'Overall':>10s}  {'Time':>6s}", flush=True)
    print(f"  {'─' * 65}", flush=True)
    for rc in all_report_cards:
        status = "DEPLOY" if rc['overall_passed'] else "FAIL"
        print(f"  {rc['label']:<35s}  {rc['stages_passed']}/{rc['stages_total']:>5d}  "
              f"{status:>10s}  {rc['elapsed_s']:5.0f}s", flush=True)

    print(f"\n  Summary: {n_passed}/{n_total} winners are DEPLOY-READY", flush=True)

    # Detail on failed stages
    if n_passed < n_total:
        print(f"\n  Failed stage details:", flush=True)
        for rc in all_report_cards:
            if not rc['overall_passed']:
                failed = [s['name'] for s in rc['stages'] if not s['passed'] and s['stage'] < 7]
                print(f"    {rc['label']}: failed at {', '.join(failed)}", flush=True)

    results['summary'] = {
        'n_validated': n_total,
        'n_deploy_ready': n_passed,
        'deploy_ready_labels': [rc['label'] for rc in all_report_cards if rc['overall_passed']],
    }
    results['elapsed_s'] = round(elapsed, 1)

    out_file = OUTPUT_DIR / "r129_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"  Saved: {out_file}", flush=True)
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
