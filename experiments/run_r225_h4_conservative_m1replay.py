#!/usr/bin/env python3
"""R225: H4 Conservative Retest + M1-Replay Validation
========================================================
Two-part validation:

Part A — Conservative Parameters Retest:
  - SL = 3 ATR (fixed, no trailing)
  - TP = 6 ATR (let winners run)
  - No trailing stop
  - Max hold = 15 H4 bars (~60h)
  - Spread = $0.50 (pessimistic)
  - Run all 5 top candidates, full validation suite

Part B — M1-Resolution Replay:
  - Take H4-level signals and SL/TP
  - Replay exit logic at M1 resolution within each H4 bar
  - Compare H4-bar exit vs M1-intrabar exit
  - Measures: how often SL/TP would have been hit mid-bar,
    worst-case slippage, actual fill quality

This catches the "H4 bar paints over intrabar chaos" problem.
"""
from __future__ import annotations
import sys, json, time, os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.h4_engine import (
    H4BacktestEngine, prepare_h4_indicators, load_h4_with_indicators
)
from backtest.engine import TradeRecord

OUTPUT_DIR = Path("results/r225_h4_conservative")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERA_SEGMENTS = {
    "Pre-COVID (2015-2019)":      ("2015-01-01", "2020-01-01"),
    "COVID+Recovery (2020-2021)": ("2020-01-01", "2022-01-01"),
    "Tightening (2022-2023)":     ("2022-01-01", "2024-01-01"),
    "Recent (2024-2026)":         ("2024-01-01", "2026-06-01"),
}

CONSERVATIVE_PARAMS = {
    'sl_atr_mult': 3.0,
    'tp_atr_mult': 6.0,
    'trailing_activate_atr': 0.0,
    'trailing_distance_atr': 0.0,
    'max_hold': 15,
    'cooldown_bars': 3,
    'spread_cost': 0.50,
}


def save(name, data):
    p = OUTPUT_DIR / f'{name}.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    print(f'  -> saved {p}')


def calc_stats(trades):
    if not trades:
        return {'n': 0, 'pnl': 0, 'sharpe': 0, 'win_rate': 0, 'avg_pnl': 0, 'max_dd': 0}
    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    cum = np.cumsum(pnls)
    dd = np.maximum.accumulate(cum) - cum
    sharpe = float(pnls.mean() / max(pnls.std(ddof=1), 1e-9) * np.sqrt(252)) if n > 1 else 0
    return {
        'n': n, 'pnl': round(float(pnls.sum()), 2),
        'sharpe': round(sharpe, 3),
        'win_rate': round(100 * (pnls > 0).sum() / n, 2),
        'avg_pnl': round(float(pnls.mean()), 2),
        'max_dd': round(float(dd.max()), 2),
    }


def filter_period(trades, start, end):
    ts_s, ts_e = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    return [t for t in trades if ts_s <= pd.Timestamp(t.entry_time) < ts_e]


def kfold_6(trades):
    if len(trades) < 30:
        return {'skip': True, 'verdict': 'SKIP'}
    pnls = np.array([t.pnl for t in trades])
    fold_size = len(pnls) // 6
    folds, kf_pass = [], 0
    for fold in range(6):
        s_idx = fold * fold_size
        e_idx = s_idx + fold_size if fold < 5 else len(pnls)
        fp = pnls[s_idx:e_idx]
        if len(fp) < 5:
            continue
        sh = float(fp.mean() / max(fp.std(ddof=1), 1e-9) * np.sqrt(252))
        folds.append({'fold': fold + 1, 'n': len(fp), 'sharpe': round(sh, 3)})
        if sh > 0:
            kf_pass += 1
    rate = kf_pass / max(len(folds), 1)
    return {'folds': folds, 'pass_count': kf_pass, 'total_folds': len(folds),
            'pass_rate': round(rate, 3), 'verdict': 'PASS' if rate >= 0.67 else 'FAIL'}


# ═══════════════════════════════════════════════════════════════
# Signal functions (same as R220/R221)
# ═══════════════════════════════════════════════════════════════

def sig_kc_breakout(df):
    if len(df) < 30: return None
    row = df.iloc[-1]
    c, kc_u, kc_l = float(row['Close']), float(row.get('KC_upper', 0)), float(row.get('KC_lower', 0))
    atr = float(row.get('ATR', 0))
    if pd.isna(kc_u) or kc_u == 0 or atr <= 0: return None
    if c > kc_u: return {'strategy': 'h4_kc', 'signal': 'BUY'}
    if c < kc_l: return {'strategy': 'h4_kc', 'signal': 'SELL'}
    return None

def sig_ema_cross(df):
    if len(df) < 55: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    e20, e50 = float(curr['EMA20']), float(curr['EMA50'])
    e20p, e50p = float(prev['EMA20']), float(prev['EMA50'])
    if pd.isna(e20) or pd.isna(e50) or float(curr.get('ATR', 0)) <= 0: return None
    if e20 > e50 and e20p <= e50p: return {'strategy': 'h4_ema_cross', 'signal': 'BUY'}
    if e20 < e50 and e20p >= e50p: return {'strategy': 'h4_ema_cross', 'signal': 'SELL'}
    return None

def sig_macd_cross(df):
    if len(df) < 30: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    macd, sig = float(curr['MACD']), float(curr['MACD_signal'])
    macd_p, sig_p = float(prev['MACD']), float(prev['MACD_signal'])
    if pd.isna(macd) or pd.isna(sig) or float(curr.get('ATR', 0)) <= 0: return None
    if macd > sig and macd_p <= sig_p: return {'strategy': 'h4_macd', 'signal': 'BUY'}
    if macd < sig and macd_p >= sig_p: return {'strategy': 'h4_macd', 'signal': 'SELL'}
    return None

def sig_cci_momentum(df):
    if len(df) < 25: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    cci, cci_p = float(curr.get('CCI', 0)), float(prev.get('CCI', 0))
    slope = float(curr.get('EMA50_slope', 0))
    if pd.isna(cci) or pd.isna(cci_p) or float(curr.get('ATR', 0)) <= 0: return None
    if cci > 0 and cci_p <= 0 and slope > 0: return {'strategy': 'h4_cci', 'signal': 'BUY'}
    if cci < 0 and cci_p >= 0 and slope < 0: return {'strategy': 'h4_cci', 'signal': 'SELL'}
    return None

def sig_bb_squeeze(df):
    if len(df) < 15: return None
    row = df.iloc[-1]
    bb_u, bb_l = float(row.get('BB_upper', 0)), float(row.get('BB_lower', 0))
    kc_u, kc_l = float(row.get('KC_upper', 0)), float(row.get('KC_lower', 0))
    c, atr = float(row['Close']), float(row.get('ATR', 0))
    if pd.isna(bb_u) or pd.isna(kc_u) or kc_u == 0 or atr <= 0: return None
    if (bb_u < kc_u) and (bb_l > kc_l): return None
    squeeze_count = 0
    for j in range(max(0, len(df) - 11), len(df) - 1):
        r = df.iloc[j]
        if (float(r.get('BB_upper', 0)) < float(r.get('KC_upper', 0))
            and float(r.get('BB_lower', 0)) > float(r.get('KC_lower', 0))):
            squeeze_count += 1
        else:
            squeeze_count = 0
    if squeeze_count < 5: return None
    kc_mid = float(row.get('KC_mid', 0))
    if c > kc_mid: return {'strategy': 'h4_squeeze', 'signal': 'BUY'}
    else: return {'strategy': 'h4_squeeze', 'signal': 'SELL'}

STRATEGY_MAP = {
    'h4_kc': sig_kc_breakout,
    'h4_ema_cross': sig_ema_cross,
    'h4_macd': sig_macd_cross,
    'h4_cci': sig_cci_momentum,
    'h4_squeeze': sig_bb_squeeze,
}
TOP_CANDIDATES = ['h4_kc', 'h4_macd', 'h4_cci', 'h4_ema_cross', 'h4_squeeze']


# ═══════════════════════════════════════════════════════════════
# M1 data loading
# ═══════════════════════════════════════════════════════════════

_M1_CANDIDATES = [
    Path("data/download/xauusd-m1-bid-2015-01-01-2026-05-13.csv"),
    Path("data/download/xauusd-m1-bid-2015-01-01-2026-04-10.csv"),
]

def load_m1_chunked():
    """Load M1 data in memory-efficient way, return dict of date -> DataFrame."""
    csv_path = next((p for p in _M1_CANDIDATES if p.exists()), None)
    if csv_path is None:
        print("  WARNING: M1 data not found, skipping M1 replay")
        return None

    print(f"  Loading M1 data: {csv_path.name}...")
    chunks = {}
    chunk_size = 500_000
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        chunk['datetime'] = pd.to_datetime(chunk['timestamp'], unit='ms', utc=True)
        chunk.set_index('datetime', inplace=True)
        chunk.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                              'close': 'Close', 'volume': 'Volume'}, inplace=True)
        for date, grp in chunk.groupby(chunk.index.date):
            if date in chunks:
                chunks[date] = pd.concat([chunks[date], grp])
            else:
                chunks[date] = grp
    print(f"    Loaded {sum(len(v) for v in chunks.values())} M1 bars across {len(chunks)} days")
    return chunks


def m1_replay_trade(m1_chunks, entry_time, direction, entry_price, sl_price, tp_price, max_hold_hours, lot_size=0.02, spread=0.50):
    """Replay a single trade at M1 resolution.

    Returns dict with actual exit details vs H4 exit.
    """
    entry_ts = pd.Timestamp(entry_time)
    max_exit_ts = entry_ts + pd.Timedelta(hours=max_hold_hours)

    m1_bars = []
    current_date = entry_ts.date()
    end_date = max_exit_ts.date() + pd.Timedelta(days=1)

    while current_date <= end_date.date() if hasattr(end_date, 'date') else current_date <= end_date:
        if current_date in m1_chunks:
            day_data = m1_chunks[current_date]
            relevant = day_data[(day_data.index >= entry_ts) & (day_data.index <= max_exit_ts)]
            if len(relevant) > 0:
                m1_bars.append(relevant)
        current_date += pd.Timedelta(days=1)

    if not m1_bars:
        return {'m1_available': False}

    m1_df = pd.concat(m1_bars).sort_index()
    if len(m1_df) == 0:
        return {'m1_available': False}

    # Walk through M1 bars
    for i in range(len(m1_df)):
        bar = m1_df.iloc[i]
        h, l, c = float(bar['High']), float(bar['Low']), float(bar['Close'])
        bar_time = m1_df.index[i]

        if direction == 'BUY':
            if l <= sl_price:
                pnl = (sl_price - entry_price) * lot_size * 100 - spread
                return {
                    'm1_available': True, 'exit_reason': 'SL',
                    'exit_price': sl_price, 'exit_time': str(bar_time),
                    'bars_to_exit': i + 1, 'pnl': round(pnl, 2),
                }
            if h >= tp_price:
                pnl = (tp_price - entry_price) * lot_size * 100 - spread
                return {
                    'm1_available': True, 'exit_reason': 'TP',
                    'exit_price': tp_price, 'exit_time': str(bar_time),
                    'bars_to_exit': i + 1, 'pnl': round(pnl, 2),
                }
        else:  # SELL
            if h >= sl_price:
                pnl = (entry_price - sl_price) * lot_size * 100 - spread
                return {
                    'm1_available': True, 'exit_reason': 'SL',
                    'exit_price': sl_price, 'exit_time': str(bar_time),
                    'bars_to_exit': i + 1, 'pnl': round(pnl, 2),
                }
            if l <= tp_price:
                pnl = (entry_price - tp_price) * lot_size * 100 - spread
                return {
                    'm1_available': True, 'exit_reason': 'TP',
                    'exit_price': tp_price, 'exit_time': str(bar_time),
                    'bars_to_exit': i + 1, 'pnl': round(pnl, 2),
                }

    # Timeout at last M1 bar
    final_c = float(m1_df.iloc[-1]['Close'])
    if direction == 'BUY':
        pnl = (final_c - entry_price) * lot_size * 100 - spread
    else:
        pnl = (entry_price - final_c) * lot_size * 100 - spread
    return {
        'm1_available': True, 'exit_reason': 'Timeout',
        'exit_price': final_c, 'exit_time': str(m1_df.index[-1]),
        'bars_to_exit': len(m1_df), 'pnl': round(pnl, 2),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print('=' * 80)
    print('R225: H4 Conservative Retest + M1-Replay Validation')
    print('=' * 80)

    h4_df = load_h4_with_indicators()

    # ─── Part A: Conservative Parameters Retest ─────────────────
    print('\n' + '=' * 80)
    print('PART A: Conservative Parameters Retest')
    print(f'  SL={CONSERVATIVE_PARAMS["sl_atr_mult"]} ATR, TP={CONSERVATIVE_PARAMS["tp_atr_mult"]} ATR')
    print(f'  No trailing, max_hold={CONSERVATIVE_PARAMS["max_hold"]} bars')
    print(f'  Spread=${CONSERVATIVE_PARAMS["spread_cost"]} (pessimistic)')
    print('=' * 80)

    part_a = {}
    for strat_name in TOP_CANDIDATES:
        print(f'\n  --- {strat_name} (conservative) ---')
        sig_func = STRATEGY_MAP[strat_name]
        engine = H4BacktestEngine(
            h4_df,
            signal_funcs=[(strat_name, sig_func)],
            **CONSERVATIVE_PARAMS,
        )
        trades = engine.run()
        st = [t for t in trades if t.strategy == strat_name]
        s = calc_stats(st)
        print(f'    n={s["n"]:>5}  Sharpe={s["sharpe"]:.3f}  PnL=${s["pnl"]:.0f}  '
              f'WR={s["win_rate"]:.1f}%  MaxDD=${s["max_dd"]:.0f}  AvgPnL=${s["avg_pnl"]:.2f}')

        # Era breakdown
        eras = {}
        for era_name, (es, ee) in ERA_SEGMENTS.items():
            era_t = filter_period(st, es, ee)
            era_s = calc_stats(era_t)
            eras[era_name] = era_s
            print(f'      {era_name:<30} n={era_s["n"]:>4}  Sharpe={era_s["sharpe"]:.3f}')

        # K-Fold
        kf = kfold_6(st)
        print(f'    K-Fold: {kf.get("verdict", "SKIP")}')
        if kf.get('folds'):
            for f in kf['folds']:
                print(f'      Fold {f["fold"]}: n={f["n"]:>4}  Sharpe={f["sharpe"]:.3f}')

        # Exit reason distribution
        exit_dist = {}
        for t in st:
            r = t.exit_reason
            exit_dist[r] = exit_dist.get(r, 0) + 1
        print(f'    Exit distribution: {exit_dist}')

        # Monte Carlo
        if len(st) >= 20:
            pnls = np.array([t.pnl for t in st])
            rng = np.random.default_rng(42)
            boot_sharpes = []
            for _ in range(1000):
                sample = rng.choice(pnls, size=len(pnls), replace=True)
                sh = float(sample.mean() / max(sample.std(ddof=1), 1e-9) * np.sqrt(252))
                boot_sharpes.append(sh)
            p_val = (np.array(boot_sharpes) <= 0).sum() / 1000
            ci_5 = np.percentile(boot_sharpes, 5)
            mc_verdict = 'PASS' if p_val < 0.05 else 'FAIL'
            print(f'    MC p-value={p_val:.4f} CI5%={ci_5:.3f} -> {mc_verdict}')
        else:
            p_val, ci_5, mc_verdict = 1.0, 0, 'SKIP'

        part_a[strat_name] = {
            'stats': s, 'eras': eras,
            'kfold_verdict': kf.get('verdict', 'SKIP'),
            'kfold': kf,
            'exit_distribution': exit_dist,
            'mc_p_value': round(float(p_val), 4),
            'mc_ci5': round(float(ci_5), 3),
            'mc_verdict': mc_verdict,
        }

    save('part_a_conservative', part_a)

    # Sanity check: compare conservative vs optimistic
    print('\n  --- Conservative vs R221 (optimistic) comparison ---')
    r221_sharpes = {'h4_kc': 4.527, 'h4_macd': 3.469, 'h4_cci': 4.557,
                    'h4_ema_cross': 4.574, 'h4_squeeze': 4.653}
    for strat_name in TOP_CANDIDATES:
        cons_sh = part_a[strat_name]['stats']['sharpe']
        opt_sh = r221_sharpes.get(strat_name, 0)
        drop = opt_sh - cons_sh
        pct = (drop / max(abs(opt_sh), 1e-9)) * 100
        print(f'    {strat_name:<15} Optimistic={opt_sh:.3f} -> Conservative={cons_sh:.3f}  '
              f'Drop={drop:+.3f} ({pct:+.1f}%)')

    # ─── Part B: M1 Resolution Replay ───────────────────────────
    print('\n' + '=' * 80)
    print('PART B: M1-Resolution Replay Validation')
    print('=' * 80)

    m1_chunks = load_m1_chunked()

    if m1_chunks is None:
        print("  SKIPPED: No M1 data available")
        save('part_b_m1_replay', {'skipped': True, 'reason': 'No M1 data'})
    else:
        part_b = {}
        for strat_name in TOP_CANDIDATES:
            print(f'\n  --- {strat_name} M1 Replay ---')
            sig_func = STRATEGY_MAP[strat_name]
            engine = H4BacktestEngine(
                h4_df,
                signal_funcs=[(strat_name, sig_func)],
                **CONSERVATIVE_PARAMS,
            )
            h4_trades = engine.run()
            st = [t for t in h4_trades if t.strategy == strat_name]

            # Replay each trade at M1 resolution
            m1_results = []
            h4_pnl_total = 0
            m1_pnl_total = 0
            match_count = 0
            worse_at_m1 = 0
            better_at_m1 = 0
            sl_hit_earlier = 0
            tp_hit_earlier = 0

            sample_size = min(len(st), 500)
            sample_trades = st[:sample_size]

            for ti, trade in enumerate(sample_trades):
                if ti % 50 == 0:
                    print(f'    Replaying {ti}/{sample_size}...', flush=True)

                h4_pnl_total += trade.pnl

                # Reconstruct SL/TP from trade
                atr = trade.entry_price * 0.005  # rough ATR estimate
                h4_idx = h4_df.index.get_indexer([pd.Timestamp(trade.entry_time)], method='nearest')[0]
                if 0 <= h4_idx < len(h4_df):
                    atr = float(h4_df.iloc[h4_idx].get('ATR', atr))

                sl_dist = atr * CONSERVATIVE_PARAMS['sl_atr_mult']
                tp_dist = atr * CONSERVATIVE_PARAMS['tp_atr_mult']

                if trade.direction == 'BUY':
                    sl_price = trade.entry_price - sl_dist
                    tp_price = trade.entry_price + tp_dist
                else:
                    sl_price = trade.entry_price + sl_dist
                    tp_price = trade.entry_price - tp_dist

                m1_result = m1_replay_trade(
                    m1_chunks, trade.entry_time, trade.direction,
                    trade.entry_price, sl_price, tp_price,
                    max_hold_hours=CONSERVATIVE_PARAMS['max_hold'] * 4,
                    spread=CONSERVATIVE_PARAMS['spread_cost'],
                )

                if not m1_result.get('m1_available'):
                    continue

                m1_pnl_total += m1_result['pnl']
                m1_results.append(m1_result)

                # Compare outcomes
                if abs(m1_result['pnl'] - trade.pnl) < 0.05:
                    match_count += 1
                elif m1_result['pnl'] < trade.pnl:
                    worse_at_m1 += 1
                else:
                    better_at_m1 += 1

                if m1_result['exit_reason'] == 'SL' and trade.exit_reason != 'SL':
                    sl_hit_earlier += 1
                if m1_result['exit_reason'] == 'TP' and trade.exit_reason != 'TP':
                    tp_hit_earlier += 1

            n_replayed = len(m1_results)
            if n_replayed == 0:
                print(f'    No M1 data overlap for {strat_name}')
                part_b[strat_name] = {'n_replayed': 0}
                continue

            h4_avg = h4_pnl_total / sample_size
            m1_avg = m1_pnl_total / n_replayed
            pnl_gap = m1_avg - h4_avg
            pnl_gap_pct = (pnl_gap / max(abs(h4_avg), 1e-9)) * 100

            m1_exit_dist = {}
            for r in m1_results:
                reason = r['exit_reason']
                m1_exit_dist[reason] = m1_exit_dist.get(reason, 0) + 1

            print(f'    Replayed: {n_replayed}/{sample_size} trades')
            print(f'    H4 avg PnL: ${h4_avg:.2f}  M1 avg PnL: ${m1_avg:.2f}  Gap: ${pnl_gap:.2f} ({pnl_gap_pct:+.1f}%)')
            print(f'    Exact match: {match_count}  Worse@M1: {worse_at_m1}  Better@M1: {better_at_m1}')
            print(f'    SL hit earlier at M1: {sl_hit_earlier}  TP hit earlier: {tp_hit_earlier}')
            print(f'    M1 exit distribution: {m1_exit_dist}')

            discrepancy = 'ACCEPTABLE' if abs(pnl_gap_pct) < 20 else ('CAUTION' if abs(pnl_gap_pct) < 50 else 'REJECT')
            print(f'    Discrepancy: {discrepancy}')

            part_b[strat_name] = {
                'n_replayed': n_replayed,
                'n_sample': sample_size,
                'h4_avg_pnl': round(h4_avg, 2),
                'm1_avg_pnl': round(m1_avg, 2),
                'pnl_gap': round(pnl_gap, 2),
                'pnl_gap_pct': round(pnl_gap_pct, 1),
                'exact_match': match_count,
                'worse_at_m1': worse_at_m1,
                'better_at_m1': better_at_m1,
                'sl_hit_earlier': sl_hit_earlier,
                'tp_hit_earlier': tp_hit_earlier,
                'm1_exit_distribution': m1_exit_dist,
                'discrepancy_verdict': discrepancy,
            }

        save('part_b_m1_replay', part_b)

    # ─── Final Summary ──────────────────────────────────────────
    print('\n' + '=' * 80)
    print('FINAL VERDICT (Conservative + M1 Replay)')
    print('=' * 80)

    final = {}
    for strat_name in TOP_CANDIDATES:
        a = part_a[strat_name]
        b = part_b.get(strat_name, {}) if m1_chunks else {}

        cons_ok = a['stats']['sharpe'] > 0.5 and a['kfold_verdict'] == 'PASS' and a['mc_verdict'] == 'PASS'
        m1_ok = b.get('discrepancy_verdict', 'N/A') in ('ACCEPTABLE', 'N/A')

        if cons_ok and m1_ok:
            verdict = 'VIABLE_FOR_PAPER_TRADE'
        elif cons_ok:
            verdict = 'NEEDS_M1_REVIEW'
        else:
            verdict = 'REJECT'

        final[strat_name] = {
            'conservative_sharpe': a['stats']['sharpe'],
            'conservative_wr': a['stats']['win_rate'],
            'conservative_n': a['stats']['n'],
            'kfold': a['kfold_verdict'],
            'mc': a['mc_verdict'],
            'm1_discrepancy': b.get('discrepancy_verdict', 'N/A'),
            'm1_pnl_gap_pct': b.get('pnl_gap_pct', 'N/A'),
            'final_verdict': verdict,
        }
        print(f'  {strat_name:<15} Sh={a["stats"]["sharpe"]:.3f}  WR={a["stats"]["win_rate"]:.1f}%  '
              f'KF={a["kfold_verdict"]}  MC={a["mc_verdict"]}  '
              f'M1={b.get("discrepancy_verdict", "N/A")}  -> {verdict}')

    save('R225_final_verdict', final)

    elapsed = time.time() - t_start
    print(f'\n  Total runtime: {elapsed:.0f}s')


if __name__ == '__main__':
    main()
