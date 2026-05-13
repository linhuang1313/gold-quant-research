#!/usr/bin/env python3
"""R219: Squeeze Release Strategy — BacktestEngine Validation
===============================================================
Engine: BacktestEngine (monkey-patch scan_all_signals)
Filters: ALL active via LIVE_PARITY_KWARGS

R21 showed K-Fold 6/6 for Squeeze Straddle but used standalone SimTrade loop.
R12 tested squeeze as KC filter (K-Fold 0/6 — rejected).

R219 tests squeeze RELEASE as an INDEPENDENT entry signal:
  - BB inside KC for N consecutive bars = squeeze
  - Squeeze releases -> enter in direction of KC breakout
  - Full filter stack applies (Choppy, ATR Pctl, regime, etc.)
  - NOT a straddle (engine only supports single-direction entries)

Phase 1: Baseline (no squeeze signal)
Phase 2: Squeeze Release signal with parameter sweep
         (min_squeeze_bars, SL/TP, trail params)
Phase 3: 6-Fold K-Fold on best config
Phase 4: Keltner correlation analysis
Phase 5: Sanity Gate + Decision
"""
from __future__ import annotations
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine
import indicators as signals_mod
from indicators import get_orb_strategy

OUTPUT_DIR = Path("results/r219_squeeze_engine")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERA_SEGMENTS = {
    "Pre-COVID (2015-2019)":      ("2015-01-01", "2020-01-01"),
    "COVID+Recovery (2020-2021)": ("2020-01-01", "2022-01-01"),
    "Tightening (2022-2023)":     ("2022-01-01", "2024-01-01"),
    "Recent (2024-2026)":         ("2024-01-01", "2026-06-01"),
}

MIN_SQUEEZE_BARS_OPTIONS = [3, 5, 8, 12]
SL_ATR_OPTIONS = [1.5, 2.5, 3.5]
TRAIL_OPTIONS = [
    (0.15, 0.03),
    (0.20, 0.04),
    (0.30, 0.06),
    (0.14, 0.025),
]
MAX_HOLD_OPTIONS = [8, 12, 20]

# Global state for squeeze detection across bars
_squeeze_consecutive = 0
_min_squeeze_bars = 5


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
    sharpe = float(pnls.mean() / max(pnls.std(ddof=1), 1e-9) * np.sqrt(252 * 4)) if n > 1 else 0
    return {
        'n': n,
        'pnl': round(float(pnls.sum()), 2),
        'sharpe': round(sharpe, 3),
        'win_rate': round(100 * (pnls > 0).sum() / n, 2),
        'avg_pnl': round(float(pnls.mean()), 2),
        'max_dd': round(float(dd.max()), 2),
    }


def filter_period(trades, start, end, strat=None):
    ts_s = pd.Timestamp(start, tz='UTC')
    ts_e = pd.Timestamp(end, tz='UTC')
    out = [t for t in trades if ts_s <= pd.Timestamp(t.entry_time) < ts_e]
    if strat:
        out = [t for t in out if t.strategy == strat]
    return out


def check_squeeze_signal(df: pd.DataFrame) -> Optional[Dict]:
    """Squeeze Release signal: BB was inside KC for N bars, just released."""
    global _squeeze_consecutive

    if len(df) < 50:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest

    bb_upper = float(latest.get('BB_upper', 0))
    bb_lower = float(latest.get('BB_lower', 0))
    kc_upper = float(latest.get('KC_upper', 0))
    kc_lower = float(latest.get('KC_lower', 0))

    if pd.isna(bb_upper) or pd.isna(kc_upper) or kc_upper == 0:
        return None

    is_squeeze = (bb_upper < kc_upper) and (bb_lower > kc_lower)

    prev_bb_upper = float(prev.get('BB_upper', 0))
    prev_bb_lower = float(prev.get('BB_lower', 0))
    prev_kc_upper = float(prev.get('KC_upper', 0))
    prev_kc_lower = float(prev.get('KC_lower', 0))
    was_squeeze = (not pd.isna(prev_bb_upper) and not pd.isna(prev_kc_upper)
                   and prev_kc_upper > 0
                   and prev_bb_upper < prev_kc_upper and prev_bb_lower > prev_kc_lower)

    if is_squeeze:
        _squeeze_consecutive += 1
        return None

    if was_squeeze and _squeeze_consecutive >= _min_squeeze_bars:
        close = float(latest['Close'])
        kc_mid = float(latest.get('KC_mid', 0))
        atr = float(latest.get('ATR', 0))

        if atr <= 0 or pd.isna(kc_mid):
            _squeeze_consecutive = 0
            return None

        if close > kc_upper:
            direction = 'BUY'
        elif close < kc_lower:
            direction = 'SELL'
        elif close > kc_mid:
            direction = 'BUY'
        else:
            direction = 'SELL'

        sl = round(atr * 2.5, 2)
        tp = round(atr * 6.0, 2)

        _squeeze_consecutive = 0
        return {
            'strategy': 'squeeze_release',
            'signal': direction,
            'reason': f"Squeeze释放{direction}: {_min_squeeze_bars}+bars压缩后突破",
            'close': close,
            'sl': sl,
            'tp': tp,
            'trailing_activate': round(atr * 0.20, 2),
            'trailing_distance': round(atr * 0.04, 2),
        }

    if not is_squeeze:
        _squeeze_consecutive = 0
    return None


_original_scan = signals_mod.scan_all_signals


def patched_scan_all_signals(df, timeframe='H1', h1_adx=None):
    signals = _original_scan(df, timeframe, h1_adx)
    if timeframe == 'H1':
        sig = check_squeeze_signal(df)
        if sig:
            signals.append(sig)
    return signals


def run_engine(data, **extra_kwargs):
    global _squeeze_consecutive
    _squeeze_consecutive = 0
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    get_orb_strategy().reset_daily()
    kwargs = {**LIVE_PARITY_KWARGS, **extra_kwargs}
    engine = BacktestEngine(data.m15_df, data.h1_df, **kwargs)
    return engine.run(), engine


def main():
    global _min_squeeze_bars
    t_start = time.time()
    print('=' * 80)
    print('R219: Squeeze Release Strategy — BacktestEngine Validation')
    print('=' * 80)
    print('  Engine: BacktestEngine (monkey-patch scan_all_signals)')
    print('  Filters: LIVE_PARITY_KWARGS — ALL ACTIVE')

    data = DataBundle.load_default()
    signals_mod.scan_all_signals = patched_scan_all_signals

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Baseline (with squeeze signal, default params)
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 1: Baseline with Squeeze Release signal')
    print('=' * 80)

    _min_squeeze_bars = 5
    trades_all, eng = run_engine(data)
    sq_trades = [t for t in trades_all if t.strategy == 'squeeze_release']
    kc_trades = [t for t in trades_all if t.strategy == 'keltner']

    sq_stats = calc_stats(sq_trades)
    kc_stats = calc_stats(kc_trades)

    print(f'  Keltner:         n={kc_stats["n"]:>5}  Sharpe={kc_stats["sharpe"]:.3f}  PnL=${kc_stats["pnl"]:.0f}')
    print(f'  Squeeze Release: n={sq_stats["n"]:>5}  Sharpe={sq_stats["sharpe"]:.3f}  '
          f'PnL=${sq_stats["pnl"]:.0f}  WR={sq_stats["win_rate"]:.1f}%')

    sq_eras = {}
    for era_name, (es, ee) in ERA_SEGMENTS.items():
        era_t = filter_period(sq_trades, es, ee)
        s = calc_stats(era_t)
        sq_eras[era_name] = s
        print(f'    {era_name:<30} n={s["n"]:>4}  Sharpe={s["sharpe"]:.3f}  PnL=${s["pnl"]:.0f}')

    phase1 = {'squeeze': sq_stats, 'keltner': kc_stats, 'eras': sq_eras}
    save('phase1_baseline', phase1)

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Parameter Sweep
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 2: Parameter Sweep')
    print('=' * 80)

    phase2 = []
    best_sharpe = -999
    best_config = None

    # min_squeeze_bars sweep (SL=2.5, trail=0.20/0.04, MH=12)
    print(f'\n  Min squeeze bars sweep:')
    for msb in MIN_SQUEEZE_BARS_OPTIONS:
        _min_squeeze_bars = msb
        trades, _ = run_engine(data, sl_atr_mult=0, tp_atr_mult=0)
        sq = [t for t in trades if t.strategy == 'squeeze_release']
        s = calc_stats(sq)
        label = f'MSB_{msb}'
        print(f'    {label:<20} n={s["n"]:>4}  Sharpe={s["sharpe"]:.3f}  '
              f'PnL=${s["pnl"]:.0f}  WR={s["win_rate"]:.1f}%')
        phase2.append({'label': label, **s})
        if s['sharpe'] > best_sharpe and s['n'] >= 20:
            best_sharpe = s['sharpe']
            best_config = {'msb': msb, 'sl': 2.5, 'trail': (0.20, 0.04), 'mh': 12}

    # SL sweep with best msb
    best_msb = best_config['msb'] if best_config else 5
    _min_squeeze_bars = best_msb
    print(f'\n  SL sweep (MSB={best_msb}):')
    for sl_m in SL_ATR_OPTIONS:
        trades, _ = run_engine(data, sl_atr_mult=sl_m, tp_atr_mult=0)
        sq = [t for t in trades if t.strategy == 'squeeze_release']
        s = calc_stats(sq)
        label = f'SL_{sl_m}'
        print(f'    {label:<20} n={s["n"]:>4}  Sharpe={s["sharpe"]:.3f}  '
              f'PnL=${s["pnl"]:.0f}  WR={s["win_rate"]:.1f}%')
        phase2.append({'label': label, **s})
        if s['sharpe'] > best_sharpe and s['n'] >= 20:
            best_sharpe = s['sharpe']
            best_config = {'msb': best_msb, 'sl': sl_m, 'trail': (0.20, 0.04), 'mh': 12}

    print(f'\n  Best config: {best_config}  Sharpe={best_sharpe:.3f}')
    save('phase2_sweep', {'results': phase2, 'best': best_config, 'best_sharpe': best_sharpe})

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: 6-Fold K-Fold on best config
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 3: 6-Fold K-Fold Validation')
    print('=' * 80)

    if best_config is None or best_sharpe <= 0:
        print('  No viable config — SKIP')
        phase3 = {'skip': True, 'reason': 'no viable config'}
    else:
        bc = best_config
        _min_squeeze_bars = bc['msb']
        sl_kw = {'sl_atr_mult': bc['sl'], 'tp_atr_mult': 0} if bc['sl'] != 2.5 else {'sl_atr_mult': 0, 'tp_atr_mult': 0}
        trades, _ = run_engine(data, **sl_kw)
        sq_only = [t for t in trades if t.strategy == 'squeeze_release']

        if len(sq_only) < 30:
            print(f'  Only {len(sq_only)} trades — insufficient for K-Fold')
            phase3 = {'skip': True, 'reason': f'n={len(sq_only)} < 30'}
        else:
            pnls = np.array([t.pnl for t in sq_only])
            n_folds = 6
            fold_size = len(pnls) // n_folds
            fold_results = []
            kf_pass = 0

            for fold in range(n_folds):
                start = fold * fold_size
                end = start + fold_size if fold < n_folds - 1 else len(pnls)
                fp = pnls[start:end]
                if len(fp) < 5:
                    continue
                s = float(fp.mean() / max(fp.std(ddof=1), 1e-9) * np.sqrt(252 * 4))
                fold_results.append({'fold': fold + 1, 'n': len(fp), 'sharpe': round(s, 3)})
                print(f'    Fold {fold+1}: n={len(fp):>4}  Sharpe={s:.3f}')
                if s > 0:
                    kf_pass += 1

            kf_rate = kf_pass / max(len(fold_results), 1)
            phase3 = {
                'config': bc,
                'folds': fold_results,
                'pass_count': kf_pass,
                'total_folds': len(fold_results),
                'pass_rate': round(kf_rate, 3),
                'verdict': 'PASS' if kf_rate >= 0.67 else 'FAIL',
            }
            print(f'    K-Fold: {kf_pass}/{len(fold_results)} positive -> {phase3["verdict"]}')

    save('phase3_kfold', phase3)

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Keltner Correlation Analysis
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 4: Keltner Correlation Analysis')
    print('=' * 80)

    if best_config:
        _min_squeeze_bars = best_config['msb']
        trades, _ = run_engine(data, sl_atr_mult=0, tp_atr_mult=0)
        sq_only = [t for t in trades if t.strategy == 'squeeze_release']
        kc_only = [t for t in trades if t.strategy == 'keltner']

        # Build daily PnL series
        def daily_pnl(trades_list):
            pnl_by_day = {}
            for t in trades_list:
                day = pd.Timestamp(t.entry_time).strftime('%Y-%m-%d')
                pnl_by_day[day] = pnl_by_day.get(day, 0) + t.pnl
            return pd.Series(pnl_by_day).sort_index()

        sq_daily = daily_pnl(sq_only)
        kc_daily = daily_pnl(kc_only)

        # Align on common dates
        common_idx = sq_daily.index.intersection(kc_daily.index)
        if len(common_idx) >= 10:
            corr = float(sq_daily.loc[common_idx].corr(kc_daily.loc[common_idx]))
        else:
            corr = 0.0

        # Entry time overlap
        sq_entry_days = set(pd.Timestamp(t.entry_time).strftime('%Y-%m-%d') for t in sq_only)
        kc_entry_days = set(pd.Timestamp(t.entry_time).strftime('%Y-%m-%d') for t in kc_only)
        overlap_days = sq_entry_days & kc_entry_days
        overlap_rate = len(overlap_days) / max(len(sq_entry_days), 1)

        print(f'  Squeeze daily PnL dates: {len(sq_daily)}')
        print(f'  Keltner daily PnL dates: {len(kc_daily)}')
        print(f'  Common dates: {len(common_idx)}')
        print(f'  Daily PnL correlation: {corr:.3f}')
        print(f'  Entry day overlap: {len(overlap_days)}/{len(sq_entry_days)} = {100*overlap_rate:.1f}%')

        diversification = 'GOOD' if corr < 0.3 else ('MODERATE' if corr < 0.5 else 'HIGH')
        print(f'  Diversification: {diversification}')

        phase4 = {
            'correlation': round(corr, 3),
            'sq_days': len(sq_daily),
            'kc_days': len(kc_daily),
            'common_days': len(common_idx),
            'entry_overlap_rate': round(overlap_rate, 3),
            'diversification': diversification,
        }
    else:
        phase4 = {'skip': True}
        print('  No viable squeeze config — SKIP')

    save('phase4_correlation', phase4)

    # ═══════════════════════════════════════════════════════════════
    # Phase 5: Sanity Gate + Final Verdict
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 5: Sanity Gate + Final Verdict')
    print('=' * 80)

    alerts = []
    if sq_stats['sharpe'] > 15:
        alerts.append(f'Sharpe {sq_stats["sharpe"]:.1f} suspiciously high')
    if sq_stats['win_rate'] > 95:
        alerts.append(f'WR {sq_stats["win_rate"]:.1f}% suspiciously high')

    if alerts:
        print(f'  ALERTS: {"; ".join(alerts)}')
    else:
        print(f'  No sanity alerts.')

    # Decision
    decision = 'REJECT'
    reasons = []

    if sq_stats['sharpe'] > 1.0:
        reasons.append(f'Full Sharpe {sq_stats["sharpe"]:.2f} > 1.0')
    else:
        reasons.append(f'Full Sharpe {sq_stats["sharpe"]:.2f} <= 1.0 (weak)')

    kf_verdict = phase3.get('verdict', 'SKIP')
    reasons.append(f'K-Fold: {kf_verdict}')

    if phase4.get('diversification') == 'GOOD':
        reasons.append(f'Low KC correlation ({phase4.get("correlation", "?")})')
    elif phase4.get('diversification') == 'HIGH':
        reasons.append(f'HIGH KC correlation ({phase4.get("correlation", "?")}) — adds no diversification')

    if sq_stats['sharpe'] > 1.0 and kf_verdict == 'PASS':
        decision = 'CONSIDER'
    elif sq_stats['sharpe'] > 0.5 and kf_verdict == 'PASS':
        decision = 'WEAK-CONSIDER'

    print(f'\n  Decision: {decision}')
    for r in reasons:
        print(f'    - {r}')

    phase5 = {'alerts': alerts, 'decision': decision, 'reasons': reasons}
    save('phase5_verdict', phase5)

    # Summary
    summary = {
        'engine': 'BacktestEngine (LIVE_PARITY_KWARGS)',
        'filters': 'Choppy Gate + ATR Pctl + regime + ADX — ALL ACTIVE',
        'squeeze_baseline': sq_stats,
        'keltner_baseline': kc_stats,
        'best_config': best_config,
        'kfold_verdict': kf_verdict,
        'kc_correlation': phase4.get('correlation'),
        'diversification': phase4.get('diversification'),
        'decision': decision,
        'reasons': reasons,
    }
    save('R219_summary', summary)

    # Restore original
    signals_mod.scan_all_signals = _original_scan

    elapsed = time.time() - t_start
    print(f'\n  Total runtime: {elapsed:.0f}s')


if __name__ == '__main__':
    main()
