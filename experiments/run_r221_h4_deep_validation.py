#!/usr/bin/env python3
"""R221: H4 Top Candidates — Deep Validation
================================================
Engine: H4BacktestEngine
Data: Dukascopy H4 XAUUSD 2015-2026

Performs rigorous validation on R220's top candidates:
  - h4_kc (Keltner breakout) — Sharpe 3.0, 1050 trades
  - h4_macd (MACD crossover) — Sharpe 2.2, 981 trades
  - h4_cci (CCI momentum)    — Sharpe 2.3, 545 trades
  - h4_ema_cross (EMA cross)  — Sharpe 2.4, 296 trades
  - h4_squeeze (BB squeeze)   — Sharpe 1.8, 364 trades

Validation phases:
  Phase 1: Walk-Forward Optimization (expanding window, 6 periods)
  Phase 2: Era Stability (4-era breakdown, require 3/4 positive)
  Phase 3: Parameter Sensitivity (±20% perturbation, check Sharpe drop)
  Phase 4: Monte Carlo Bootstrap (1000 resamples, p-value for Sharpe > 0)
  Phase 5: Drawdown Stress Test (worst streak, max DD recovery)
  Phase 6: Strategy Correlation (pairwise daily PnL correlation)
  Phase 7: Final 3-Gate Verdict (K-Fold + WF + Era)
"""
from __future__ import annotations
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.h4_engine import (
    H4BacktestEngine, prepare_h4_indicators, load_h4_with_indicators
)
from backtest.engine import TradeRecord

OUTPUT_DIR = Path("results/r221_h4_deep_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERA_SEGMENTS = {
    "Pre-COVID (2015-2019)":      ("2015-01-01", "2020-01-01"),
    "COVID+Recovery (2020-2021)": ("2020-01-01", "2022-01-01"),
    "Tightening (2022-2023)":     ("2022-01-01", "2024-01-01"),
    "Recent (2024-2026)":         ("2024-01-01", "2026-06-01"),
}

SPREAD = 0.30

BEST_PARAMS = {
    'h4_kc':        {'sl': 5.0, 'tp': 6.0, 'trail_act': 0.3, 'trail_dist': 0.08},
    'h4_ema_cross': {'sl': 3.0, 'tp': 4.0, 'trail_act': 0.3, 'trail_dist': 0.08},
    'h4_macd':      {'sl': 2.0, 'tp': 6.0, 'trail_act': 0.3, 'trail_dist': 0.08},
    'h4_cci':       {'sl': 4.0, 'tp': 6.0, 'trail_act': 0.3, 'trail_dist': 0.08},
    'h4_squeeze':   {'sl': 4.0, 'tp': 4.0, 'trail_act': 0.3, 'trail_dist': 0.08},
}


# ═══════════════════════════════════════════════════════════════
# Signal functions (same as R220)
# ═══════════════════════════════════════════════════════════════

def sig_kc_breakout(df: pd.DataFrame) -> Optional[Dict]:
    if len(df) < 30:
        return None
    row = df.iloc[-1]
    c, kc_u, kc_l = float(row['Close']), float(row.get('KC_upper', 0)), float(row.get('KC_lower', 0))
    atr = float(row.get('ATR', 0))
    if pd.isna(kc_u) or kc_u == 0 or atr <= 0:
        return None
    if c > kc_u:
        return {'strategy': 'h4_kc', 'signal': 'BUY'}
    if c < kc_l:
        return {'strategy': 'h4_kc', 'signal': 'SELL'}
    return None


def sig_ema_cross(df: pd.DataFrame) -> Optional[Dict]:
    if len(df) < 55:
        return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    ema20, ema50 = float(curr['EMA20']), float(curr['EMA50'])
    ema20_p, ema50_p = float(prev['EMA20']), float(prev['EMA50'])
    atr = float(curr.get('ATR', 0))
    if pd.isna(ema20) or pd.isna(ema50) or atr <= 0:
        return None
    if ema20 > ema50 and ema20_p <= ema50_p:
        return {'strategy': 'h4_ema_cross', 'signal': 'BUY'}
    if ema20 < ema50 and ema20_p >= ema50_p:
        return {'strategy': 'h4_ema_cross', 'signal': 'SELL'}
    return None


def sig_macd_cross(df: pd.DataFrame) -> Optional[Dict]:
    if len(df) < 30:
        return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    macd, sig_line = float(curr['MACD']), float(curr['MACD_signal'])
    macd_p, sig_p = float(prev['MACD']), float(prev['MACD_signal'])
    atr = float(curr.get('ATR', 0))
    if pd.isna(macd) or pd.isna(sig_line) or atr <= 0:
        return None
    if macd > sig_line and macd_p <= sig_p:
        return {'strategy': 'h4_macd', 'signal': 'BUY'}
    if macd < sig_line and macd_p >= sig_p:
        return {'strategy': 'h4_macd', 'signal': 'SELL'}
    return None


def sig_cci_momentum(df: pd.DataFrame) -> Optional[Dict]:
    if len(df) < 25:
        return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    cci, cci_p = float(curr.get('CCI', 0)), float(prev.get('CCI', 0))
    atr = float(curr.get('ATR', 0))
    ema50_slope = float(curr.get('EMA50_slope', 0))
    if pd.isna(cci) or pd.isna(cci_p) or atr <= 0:
        return None
    if cci > 0 and cci_p <= 0 and ema50_slope > 0:
        return {'strategy': 'h4_cci', 'signal': 'BUY'}
    if cci < 0 and cci_p >= 0 and ema50_slope < 0:
        return {'strategy': 'h4_cci', 'signal': 'SELL'}
    return None


def sig_bb_squeeze(df: pd.DataFrame) -> Optional[Dict]:
    if len(df) < 15:
        return None
    row = df.iloc[-1]
    bb_u, bb_l = float(row.get('BB_upper', 0)), float(row.get('BB_lower', 0))
    kc_u, kc_l = float(row.get('KC_upper', 0)), float(row.get('KC_lower', 0))
    c, atr = float(row['Close']), float(row.get('ATR', 0))
    if pd.isna(bb_u) or pd.isna(kc_u) or kc_u == 0 or atr <= 0:
        return None
    is_squeeze = (bb_u < kc_u) and (bb_l > kc_l)
    if is_squeeze:
        return None
    squeeze_count = 0
    for j in range(max(0, len(df) - 11), len(df) - 1):
        r = df.iloc[j]
        if (float(r.get('BB_upper', 0)) < float(r.get('KC_upper', 0))
            and float(r.get('BB_lower', 0)) > float(r.get('KC_lower', 0))):
            squeeze_count += 1
        else:
            squeeze_count = 0
    if squeeze_count < 5:
        return None
    kc_mid = float(row.get('KC_mid', 0))
    if c > kc_mid:
        return {'strategy': 'h4_squeeze', 'signal': 'BUY'}
    else:
        return {'strategy': 'h4_squeeze', 'signal': 'SELL'}


STRATEGY_MAP = {
    'h4_kc': sig_kc_breakout,
    'h4_ema_cross': sig_ema_cross,
    'h4_macd': sig_macd_cross,
    'h4_cci': sig_cci_momentum,
    'h4_squeeze': sig_bb_squeeze,
}

TOP_CANDIDATES = ['h4_kc', 'h4_macd', 'h4_cci', 'h4_ema_cross', 'h4_squeeze']


# ═══════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════

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
        'n': n,
        'pnl': round(float(pnls.sum()), 2),
        'sharpe': round(sharpe, 3),
        'win_rate': round(100 * (pnls > 0).sum() / n, 2),
        'avg_pnl': round(float(pnls.mean()), 2),
        'max_dd': round(float(dd.max()), 2),
    }


def filter_period(trades, start, end):
    ts_s = pd.Timestamp(start, tz='UTC')
    ts_e = pd.Timestamp(end, tz='UTC')
    return [t for t in trades if ts_s <= pd.Timestamp(t.entry_time) < ts_e]


def run_strat(h4_df, strat_name, params=None):
    """Run a single strategy with given or best params."""
    p = params or BEST_PARAMS.get(strat_name, {})
    sig_func = STRATEGY_MAP[strat_name]
    engine = H4BacktestEngine(
        h4_df,
        signal_funcs=[(strat_name, sig_func)],
        sl_atr_mult=p.get('sl', 3.0),
        tp_atr_mult=p.get('tp', 6.0),
        trailing_activate_atr=p.get('trail_act', 0.3),
        trailing_distance_atr=p.get('trail_dist', 0.08),
        max_hold=30,
        cooldown_bars=2,
        spread_cost=SPREAD,
    )
    trades = engine.run()
    return [t for t in trades if t.strategy == strat_name]


def kfold_6(trades):
    if len(trades) < 30:
        return {'skip': True, 'reason': f'n={len(trades)} < 30', 'verdict': 'SKIP'}
    pnls = np.array([t.pnl for t in trades])
    fold_size = len(pnls) // 6
    folds = []
    kf_pass = 0
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
    return {
        'folds': folds, 'pass_count': kf_pass,
        'total_folds': len(folds), 'pass_rate': round(rate, 3),
        'verdict': 'PASS' if rate >= 0.67 else 'FAIL',
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print('=' * 80)
    print('R221: H4 Top Candidates — Deep Validation')
    print('=' * 80)

    h4_df = load_h4_with_indicators()

    # ─── Phase 1: Walk-Forward Optimization ─────────────────────
    print('\n' + '=' * 80)
    print('Phase 1: Walk-Forward Optimization (Expanding Window, 6 Periods)')
    print('=' * 80)

    wf_cutoffs = [
        ("2015-01-01", "2017-01-01", "2017-01-01", "2018-10-01"),
        ("2015-01-01", "2018-10-01", "2018-10-01", "2020-07-01"),
        ("2015-01-01", "2020-07-01", "2020-07-01", "2022-04-01"),
        ("2015-01-01", "2022-04-01", "2022-04-01", "2024-01-01"),
        ("2015-01-01", "2024-01-01", "2024-01-01", "2025-07-01"),
        ("2015-01-01", "2025-07-01", "2025-07-01", "2026-06-01"),
    ]

    phase1 = {}
    for strat_name in TOP_CANDIDATES:
        print(f'\n  --- {strat_name} Walk-Forward ---')
        wf_results = []
        oos_sharpes = []

        for wf_i, (train_s, train_e, test_s, test_e) in enumerate(wf_cutoffs):
            # Train: find best SL among a small grid
            best_sh = -999
            best_p = None
            for sl_m in [2.0, 3.0, 4.0, 5.0]:
                for tp_m in [4.0, 6.0, 8.0]:
                    h4_train = h4_df[(h4_df.index >= pd.Timestamp(train_s, tz='UTC')) &
                                     (h4_df.index < pd.Timestamp(train_e, tz='UTC'))].copy()
                    if len(h4_train) < 200:
                        continue
                    trades = run_strat(h4_train, strat_name,
                                       {'sl': sl_m, 'tp': tp_m, 'trail_act': 0.3, 'trail_dist': 0.08})
                    s = calc_stats(trades)
                    if s['sharpe'] > best_sh and s['n'] >= 10:
                        best_sh = s['sharpe']
                        best_p = {'sl': sl_m, 'tp': tp_m, 'trail_act': 0.3, 'trail_dist': 0.08}

            if best_p is None:
                best_p = BEST_PARAMS.get(strat_name, {'sl': 3.0, 'tp': 6.0, 'trail_act': 0.3, 'trail_dist': 0.08})

            # Test: apply best params to OOS period
            h4_test = h4_df[(h4_df.index >= pd.Timestamp(test_s, tz='UTC')) &
                            (h4_df.index < pd.Timestamp(test_e, tz='UTC'))].copy()
            if len(h4_test) < 100:
                wf_results.append({
                    'period': f'{test_s}->{test_e}', 'skip': True,
                    'reason': f'OOS too short ({len(h4_test)} bars)'
                })
                continue

            oos_trades = run_strat(h4_test, strat_name, best_p)
            oos_stats = calc_stats(oos_trades)
            oos_sharpes.append(oos_stats['sharpe'])
            wf_results.append({
                'period': f'{test_s}->{test_e}',
                'train_best_params': best_p,
                'train_sharpe': round(best_sh, 3),
                'oos_n': oos_stats['n'],
                'oos_sharpe': oos_stats['sharpe'],
                'oos_pnl': oos_stats['pnl'],
            })
            print(f'    WF{wf_i+1} [{test_s}->{test_e}]: '
                  f'train_Sh={best_sh:.3f} -> OOS n={oos_stats["n"]} Sh={oos_stats["sharpe"]:.3f}')

        # WF verdict: majority of OOS periods should be positive
        valid_oos = [s for s in oos_sharpes if not np.isnan(s)]
        positive_oos = sum(1 for s in valid_oos if s > 0)
        wf_pass_rate = positive_oos / max(len(valid_oos), 1)
        wf_verdict = 'PASS' if wf_pass_rate >= 0.67 else 'FAIL'
        avg_oos = np.mean(valid_oos) if valid_oos else 0

        print(f'    WF Verdict: {wf_verdict} ({positive_oos}/{len(valid_oos)} positive, avg OOS={avg_oos:.3f})')

        phase1[strat_name] = {
            'walk_forward': wf_results,
            'oos_sharpes': [round(s, 3) for s in valid_oos],
            'avg_oos_sharpe': round(avg_oos, 3),
            'wf_pass_rate': round(wf_pass_rate, 3),
            'wf_verdict': wf_verdict,
        }

    save('phase1_walk_forward', phase1)

    # ─── Phase 2: Era Stability ─────────────────────────────────
    print('\n' + '=' * 80)
    print('Phase 2: Era Stability (4 Eras)')
    print('=' * 80)

    phase2 = {}
    for strat_name in TOP_CANDIDATES:
        print(f'\n  --- {strat_name} Era Stability ---')
        all_trades = run_strat(h4_df, strat_name)
        positive_eras = 0
        era_results = {}

        for era_name, (es, ee) in ERA_SEGMENTS.items():
            era_trades = filter_period(all_trades, es, ee)
            s = calc_stats(era_trades)
            era_results[era_name] = s
            if s['sharpe'] > 0:
                positive_eras += 1
            print(f'    {era_name:<30} n={s["n"]:>4}  Sharpe={s["sharpe"]:.3f}  PnL=${s["pnl"]:.0f}')

        era_verdict = 'PASS' if positive_eras >= 3 else 'FAIL'
        print(f'    Era Verdict: {era_verdict} ({positive_eras}/4 positive)')

        phase2[strat_name] = {
            'eras': era_results,
            'positive_eras': positive_eras,
            'era_verdict': era_verdict,
        }

    save('phase2_era_stability', phase2)

    # ─── Phase 3: Parameter Sensitivity ─────────────────────────
    print('\n' + '=' * 80)
    print('Phase 3: Parameter Sensitivity (±20% perturbation)')
    print('=' * 80)

    phase3 = {}
    for strat_name in TOP_CANDIDATES:
        print(f'\n  --- {strat_name} Sensitivity ---')
        base_p = BEST_PARAMS.get(strat_name, {'sl': 3.0, 'tp': 6.0, 'trail_act': 0.3, 'trail_dist': 0.08})
        base_trades = run_strat(h4_df, strat_name, base_p)
        base_stats = calc_stats(base_trades)
        base_sharpe = base_stats['sharpe']

        perturbations = []
        sharpe_drops = []

        for param_name in ['sl', 'tp', 'trail_act', 'trail_dist']:
            base_val = base_p.get(param_name, 0)
            if base_val == 0:
                continue
            for factor in [0.8, 1.2]:
                perturbed_p = dict(base_p)
                perturbed_p[param_name] = round(base_val * factor, 4)
                p_trades = run_strat(h4_df, strat_name, perturbed_p)
                p_stats = calc_stats(p_trades)
                drop = base_sharpe - p_stats['sharpe']
                drop_pct = (drop / max(abs(base_sharpe), 1e-9)) * 100
                perturbations.append({
                    'param': param_name,
                    'base': base_val,
                    'perturbed': perturbed_p[param_name],
                    'factor': factor,
                    'sharpe': p_stats['sharpe'],
                    'drop': round(drop, 3),
                    'drop_pct': round(drop_pct, 1),
                })
                sharpe_drops.append(abs(drop_pct))
                label = f'{param_name}={perturbed_p[param_name]}'
                print(f'    {label:<20} Sharpe={p_stats["sharpe"]:.3f} (drop={drop:+.3f}, {drop_pct:+.1f}%)')

        max_drop = max(sharpe_drops) if sharpe_drops else 0
        avg_drop = np.mean(sharpe_drops) if sharpe_drops else 0
        sens_verdict = 'STABLE' if max_drop < 50 else ('MODERATE' if max_drop < 80 else 'FRAGILE')
        print(f'    Sensitivity: {sens_verdict} (max_drop={max_drop:.1f}%, avg_drop={avg_drop:.1f}%)')

        phase3[strat_name] = {
            'base_sharpe': base_sharpe,
            'perturbations': perturbations,
            'max_drop_pct': round(max_drop, 1),
            'avg_drop_pct': round(avg_drop, 1),
            'verdict': sens_verdict,
        }

    save('phase3_sensitivity', phase3)

    # ─── Phase 4: Monte Carlo Bootstrap ─────────────────────────
    print('\n' + '=' * 80)
    print('Phase 4: Monte Carlo Bootstrap (1000 resamples)')
    print('=' * 80)

    N_BOOTSTRAP = 1000
    phase4 = {}
    for strat_name in TOP_CANDIDATES:
        print(f'\n  --- {strat_name} Monte Carlo ---')
        all_trades = run_strat(h4_df, strat_name)
        pnls = np.array([t.pnl for t in all_trades])
        n = len(pnls)

        if n < 20:
            print(f'    SKIP (n={n} < 20)')
            phase4[strat_name] = {'skip': True, 'n': n}
            continue

        rng = np.random.default_rng(42)
        boot_sharpes = []
        for _ in range(N_BOOTSTRAP):
            sample = rng.choice(pnls, size=n, replace=True)
            sh = float(sample.mean() / max(sample.std(ddof=1), 1e-9) * np.sqrt(252))
            boot_sharpes.append(sh)

        boot_arr = np.array(boot_sharpes)
        p_value = (boot_arr <= 0).sum() / N_BOOTSTRAP
        ci_5 = np.percentile(boot_arr, 5)
        ci_95 = np.percentile(boot_arr, 95)
        median_sh = np.median(boot_arr)

        mc_verdict = 'PASS' if p_value < 0.05 else 'FAIL'
        print(f'    p-value(Sharpe<=0): {p_value:.4f}  median={median_sh:.3f}  '
              f'CI=[{ci_5:.3f}, {ci_95:.3f}]  -> {mc_verdict}')

        phase4[strat_name] = {
            'n': n,
            'p_value': round(float(p_value), 4),
            'median_sharpe': round(float(median_sh), 3),
            'ci_5': round(float(ci_5), 3),
            'ci_95': round(float(ci_95), 3),
            'mc_verdict': mc_verdict,
        }

    save('phase4_monte_carlo', phase4)

    # ─── Phase 5: Drawdown Stress Test ──────────────────────────
    print('\n' + '=' * 80)
    print('Phase 5: Drawdown Stress Test')
    print('=' * 80)

    phase5 = {}
    for strat_name in TOP_CANDIDATES:
        print(f'\n  --- {strat_name} Drawdown ---')
        all_trades = run_strat(h4_df, strat_name)
        pnls = np.array([t.pnl for t in all_trades])
        n = len(pnls)

        if n < 10:
            phase5[strat_name] = {'skip': True}
            continue

        cum = np.cumsum(pnls)
        peaks = np.maximum.accumulate(cum)
        drawdowns = peaks - cum
        max_dd = float(drawdowns.max())
        max_dd_idx = int(drawdowns.argmax())

        # Worst losing streak
        streak = 0
        max_streak = 0
        for p in pnls:
            if p < 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

        # Recovery: bars from max DD to next peak
        recovery_bars = 0
        if max_dd_idx < n - 1:
            for ri in range(max_dd_idx + 1, n):
                if cum[ri] >= peaks[max_dd_idx]:
                    recovery_bars = ri - max_dd_idx
                    break

        # Calmar ratio (annualized return / max DD)
        years = n / (252 * 6 / 4)  # ~63 H4 bars per month
        annual_return = float(cum[-1]) / max(years, 0.5)
        calmar = annual_return / max(max_dd, 1e-9) if max_dd > 0 else 0

        print(f'    MaxDD=${max_dd:.0f}  Worst streak={max_streak}  '
              f'Recovery={recovery_bars} trades  Calmar={calmar:.2f}')

        phase5[strat_name] = {
            'max_dd': round(max_dd, 2),
            'worst_losing_streak': max_streak,
            'recovery_trades': recovery_bars,
            'calmar': round(calmar, 2),
            'total_pnl': round(float(cum[-1]), 2),
        }

    save('phase5_drawdown', phase5)

    # ─── Phase 6: Strategy Correlation ──────────────────────────
    print('\n' + '=' * 80)
    print('Phase 6: Strategy Correlation (Pairwise Daily PnL)')
    print('=' * 80)

    daily_pnl_map = {}
    for strat_name in TOP_CANDIDATES:
        trades = run_strat(h4_df, strat_name)
        if not trades:
            continue
        daily = {}
        for t in trades:
            day = pd.Timestamp(t.exit_time).date()
            daily[day] = daily.get(day, 0) + t.pnl
        daily_pnl_map[strat_name] = pd.Series(daily)

    all_days = sorted(set().union(*[set(s.index) for s in daily_pnl_map.values()]))
    corr_df = pd.DataFrame({name: s.reindex(all_days, fill_value=0)
                            for name, s in daily_pnl_map.items()})

    phase6 = {'pairs': []}
    for s1, s2 in combinations(TOP_CANDIDATES, 2):
        if s1 in corr_df.columns and s2 in corr_df.columns:
            r = float(corr_df[s1].corr(corr_df[s2]))
            label = 'LOW' if abs(r) < 0.3 else ('MODERATE' if abs(r) < 0.6 else 'HIGH')
            phase6['pairs'].append({
                'pair': f'{s1} vs {s2}',
                'correlation': round(r, 3),
                'label': label,
            })
            print(f'  {s1} vs {s2}: r={r:.3f} ({label})')

    save('phase6_correlation', phase6)

    # ─── Phase 7: Final 3-Gate Verdict ──────────────────────────
    print('\n' + '=' * 80)
    print('Phase 7: Final 3-Gate Verdict')
    print('=' * 80)

    phase7 = {}
    for strat_name in TOP_CANDIDATES:
        # K-Fold (rerun for clean result)
        all_trades = run_strat(h4_df, strat_name)
        kf = kfold_6(all_trades)

        wf_v = phase1.get(strat_name, {}).get('wf_verdict', 'N/A')
        era_v = phase2.get(strat_name, {}).get('era_verdict', 'N/A')
        kf_v = kf.get('verdict', 'N/A')
        mc_v = phase4.get(strat_name, {}).get('mc_verdict', 'N/A')
        sens_v = phase3.get(strat_name, {}).get('verdict', 'N/A')

        gates_passed = sum(1 for v in [kf_v, wf_v, era_v] if v == 'PASS')
        stats = calc_stats(all_trades)

        if gates_passed >= 3 and mc_v == 'PASS' and sens_v != 'FRAGILE':
            final = 'STRONG_PASS'
        elif gates_passed >= 2 and mc_v == 'PASS':
            final = 'CONDITIONAL_PASS'
        elif gates_passed >= 2:
            final = 'WEAK_PASS'
        else:
            final = 'REJECT'

        phase7[strat_name] = {
            'stats': stats,
            'kfold': kf_v,
            'walk_forward': wf_v,
            'era_stability': era_v,
            'monte_carlo': mc_v,
            'sensitivity': sens_v,
            'gates_passed': gates_passed,
            'final_verdict': final,
        }
        print(f'  {strat_name:<15} KF={kf_v:<5} WF={wf_v:<5} Era={era_v:<5} '
              f'MC={mc_v:<5} Sens={sens_v:<8} -> {final}')

    save('R221_final_verdict', phase7)

    # ─── Summary ────────────────────────────────────────────────
    print('\n' + '=' * 80)
    print('FINAL SUMMARY')
    print('=' * 80)

    strong = [s for s, v in phase7.items() if v['final_verdict'] == 'STRONG_PASS']
    conditional = [s for s, v in phase7.items() if v['final_verdict'] == 'CONDITIONAL_PASS']
    weak = [s for s, v in phase7.items() if v['final_verdict'] == 'WEAK_PASS']
    rejected = [s for s, v in phase7.items() if v['final_verdict'] == 'REJECT']

    print(f'  STRONG PASS:       {strong if strong else "None"}')
    print(f'  CONDITIONAL PASS:  {conditional if conditional else "None"}')
    print(f'  WEAK PASS:         {weak if weak else "None"}')
    print(f'  REJECTED:          {rejected if rejected else "None"}')

    # Portfolio suggestion (low-correlation strong candidates)
    low_corr_pairs = [p for p in phase6.get('pairs', []) if p['label'] == 'LOW']
    if low_corr_pairs:
        print(f'\n  Low-correlation pairs (good for portfolio):')
        for p in low_corr_pairs:
            print(f'    {p["pair"]}: r={p["correlation"]}')

    elapsed = time.time() - t_start
    print(f'\n  Total runtime: {elapsed:.0f}s')


if __name__ == '__main__':
    main()
