#!/usr/bin/env python3
"""R232: SL vs Cap Alignment — Live Parameter Audit
=====================================================
The live system has SL=6.0×ATR but Cap=$70 fixed (=$17.5 price distance at 0.04 lots).
Cap fires 4-10x before SL ever can. This experiment validates:

  Config A: Baseline — LIVE_PARITY (SL=3.5×ATR, no Cap)
  Config B: Live-Actual — SL=6.0×ATR + Cap $70 fixed (reproducing live)
  Config C: Fix Option 1 — SL=6.0×ATR + Cap removed (SL is the only stop)
  Config D: Fix Option 2 — SL=6.0×ATR + Cap = dynamic only (4.0×ATR, no fixed floor)
  Config E: Fix Option 3 — SL=3.5×ATR + Cap $70 (Cap as true safety net)
  Config F: Fix Option 4 — SL=4.0×ATR + no Cap (tighter SL, no Cap needed)
  Config G: Fix Option 5 — SL=3.0×ATR + no Cap

For each config: full stats, K-Fold 6/6, era breakdown, Cap trigger count.
Also runs SL sweep: sl_atr_mult ∈ {2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0} with no Cap.
"""
from __future__ import annotations
import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine, TradeRecord

OUTPUT_DIR = Path("results/r232_sl_cap_alignment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERA_SEGMENTS = {
    "Pre-COVID (2015-2019)":      ("2015-01-01", "2020-01-01"),
    "COVID+Recovery (2020-2021)": ("2020-01-01", "2022-01-01"),
    "Tightening (2022-2023)":     ("2022-01-01", "2024-01-01"),
    "Recent (2024-2026)":         ("2024-01-01", "2026-06-01"),
}


def pf(msg): print(msg, flush=True)

def save(name, data):
    p = OUTPUT_DIR / f'{name}.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    pf(f'  -> saved {p}')

def filter_period(trades, start, end):
    ts_s, ts_e = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    return [t for t in trades if ts_s <= pd.Timestamp(t.entry_time) < ts_e]

def trade_stats(trades):
    if not trades:
        return {'n': 0, 'pnl': 0, 'sharpe': 0, 'win_rate': 0, 'max_dd': 0, 'profit_factor': 0, 'avg_pnl': 0}
    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    cum = np.cumsum(pnls)
    dd = np.maximum.accumulate(cum) - cum
    sharpe = float(pnls.mean() / max(pnls.std(ddof=1), 1e-9) * np.sqrt(252)) if n > 1 else 0
    wins, losses = pnls[pnls > 0], pnls[pnls < 0]
    pf_val = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 99.9
    return {'n': n, 'pnl': round(float(pnls.sum()), 2), 'sharpe': round(sharpe, 3),
            'win_rate': round(100 * (pnls > 0).sum() / n, 2), 'avg_pnl': round(float(pnls.mean()), 4),
            'max_dd': round(float(dd.max()), 2), 'profit_factor': round(pf_val, 3)}

def kfold_6(trades):
    if len(trades) < 60:
        return {'skip': True, 'verdict': 'SKIP'}
    pnls = np.array([t.pnl for t in trades])
    fold_size = len(pnls) // 6
    folds, kf_pass = [], 0
    for fold in range(6):
        s = fold * fold_size
        e = s + fold_size if fold < 5 else len(pnls)
        fp = pnls[s:e]
        if len(fp) < 5: continue
        sh = float(fp.mean() / max(fp.std(ddof=1), 1e-9) * np.sqrt(252))
        folds.append({'fold': fold+1, 'n': len(fp), 'sharpe': round(sh, 3)})
        if sh > 0: kf_pass += 1
    rate = kf_pass / max(len(folds), 1)
    return {'folds': folds, 'pass_count': kf_pass, 'total_folds': len(folds),
            'pass_rate': round(rate, 3), 'verdict': 'PASS' if rate >= 0.67 else 'FAIL'}


def run_config(data, label, sl_mult, cap_fixed, cap_atr_mult, extra_kw=None):
    """Run one configuration and return full evaluation.
    
    Fixed lots=0.04 to match live system. This is critical because:
    - Cap $70 at 0.04 lots = $17.5 price distance
    - Cap $70 at 0.01 lots = $70 price distance (completely different behavior)
    """
    kw = dict(LIVE_PARITY_KWARGS)
    kw['sl_atr_mult'] = sl_mult
    kw['maxloss_cap'] = cap_fixed
    kw['maxloss_cap_atr_mult'] = cap_atr_mult
    kw['min_lot_size'] = 0.04
    kw['max_lot_size'] = 0.04
    if extra_kw:
        kw.update(extra_kw)

    engine = BacktestEngine(data.m15_df, data.h1_df, **kw)
    engine.run()
    kc_trades = [t for t in engine.trades if t.strategy == 'keltner']

    s = trade_stats(kc_trades)
    kf = kfold_6(kc_trades)
    eras = {en: trade_stats(filter_period(kc_trades, es, ee)) for en, (es, ee) in ERA_SEGMENTS.items()}

    exit_reasons = {}
    for t in kc_trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    cap_trades = [t for t in kc_trades if t.exit_reason == 'MaxLossCap']
    cap_pnl = sum(t.pnl for t in cap_trades)

    return {
        'label': label, 'sl_atr_mult': sl_mult, 'cap_fixed': cap_fixed, 'cap_atr_mult': cap_atr_mult,
        'stats': s, 'kfold': kf, 'eras': eras,
        'exit_reasons': exit_reasons, 'cap_count': len(cap_trades), 'cap_pnl': round(cap_pnl, 2),
    }


def main():
    t0 = time.time()
    pf('='*80)
    pf('R232: SL vs Cap Alignment — Live Parameter Audit')
    pf('='*80)

    data = DataBundle.load_default()

    # Live system as of 5/14: R202 trail (0.06/0.015), SL=6.0×ATR, Cap=$70, lots=0.04
    LIVE_ACTUAL_TRAIL = {
        'trailing_activate_atr': 0.06,
        'trailing_distance_atr': 0.015,
        'regime_config': {
            'low':    {'trail_act': 0.06, 'trail_dist': 0.015},
            'normal': {'trail_act': 0.06, 'trail_dist': 0.015},
            'high':   {'trail_act': 0.06, 'trail_dist': 0.015},
        },
    }

    configs = [
        # label, sl_mult, cap_fixed, cap_atr_mult, extra_kw
        ('A: LIVE_PARITY baseline (SL=3.5, no Cap)',        3.5,  0,    0, None),
        ('B: Live-Actual (SL=6.0, Cap=$70, R202 trail)',     6.0,  70,   0, LIVE_ACTUAL_TRAIL),
        ('B2: Live w/ LIVE_PARITY trail (SL=6.0, Cap=$70)',  6.0,  70,   0, None),
        ('C: SL=6.0, no Cap, R202 trail',                    6.0,  0,    0, LIVE_ACTUAL_TRAIL),
        ('D: SL=6.0, Cap=4.0×ATR dynamic, R202 trail',      6.0,  0,    4.0, LIVE_ACTUAL_TRAIL),
        ('E: SL=3.5, Cap=$70, R202 trail',                   3.5,  70,   0, LIVE_ACTUAL_TRAIL),
        ('F: SL=4.0, no Cap, R202 trail',                    4.0,  0,    0, LIVE_ACTUAL_TRAIL),
        ('G: SL=3.0, no Cap, R202 trail',                    3.0,  0,    0, LIVE_ACTUAL_TRAIL),
        ('H: SL=3.5, no Cap, R202 trail',                    3.5,  0,    0, LIVE_ACTUAL_TRAIL),
    ]

    results = {}
    for label, sl, cap_f, cap_a, extra in configs:
        pf(f'\n{"="*80}\n{label}\n{"="*80}')
        r = run_config(data, label, sl, cap_f, cap_a, extra)
        s = r['stats']
        pf(f'  n={s["n"]:>5}  Sh={s["sharpe"]:.3f}  PnL=${s["pnl"]:.0f}  WR={s["win_rate"]:.1f}%  '
           f'PF={s["profit_factor"]:.2f}  MaxDD=${s["max_dd"]:.0f}  AvgPnL=${s["avg_pnl"]:.3f}')
        pf(f'  K-Fold: {r["kfold"]["verdict"]} ({r["kfold"].get("pass_count",0)}/6)')
        pf(f'  Exit reasons: {r["exit_reasons"]}')
        pf(f'  Cap triggers: {r["cap_count"]}  Cap PnL: ${r["cap_pnl"]:.0f}')
        for en, es in r['eras'].items():
            pf(f'    {en:<30} n={es["n"]:>4} Sh={es["sharpe"]:.3f}')
        results[label] = r

    save('configs_a_to_g', results)

    # ── SL Sweep (no Cap, R202 trail, fixed 0.04 lots) ──
    pf(f'\n{"="*80}\nSL ATR Multiplier Sweep (no Cap, R202 trail, 0.04 lots)\n{"="*80}')
    sl_sweep = {}
    for sl_m in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]:
        label = f'SL={sl_m}'
        r = run_config(data, label, sl_m, 0, 0, LIVE_ACTUAL_TRAIL)
        s = r['stats']
        pf(f'  SL={sl_m:.1f}×ATR: n={s["n"]:>5}  Sh={s["sharpe"]:.3f}  PnL=${s["pnl"]:.0f}  '
           f'WR={s["win_rate"]:.1f}%  AvgPnL=${s["avg_pnl"]:.3f}  MaxDD=${s["max_dd"]:.0f}  '
           f'KF={r["kfold"]["verdict"]}')
        sl_sweep[label] = r

    save('sl_sweep', sl_sweep)

    # ── Cap Sweep with SL=6.0 (mimicking live SL, R202 trail) ──
    pf(f'\n{"="*80}\nCap Sweep with SL=6.0×ATR (R202 trail, 0.04 lots)\n{"="*80}')
    cap_sweep = {}
    for cap_v in [0, 35, 50, 70, 100, 150, 200]:
        label = f'Cap=${cap_v}'
        r = run_config(data, label, 6.0, cap_v, 0, LIVE_ACTUAL_TRAIL)
        s = r['stats']
        pf(f'  SL=6.0 Cap=${cap_v:>3}: n={s["n"]:>5}  Sh={s["sharpe"]:.3f}  PnL=${s["pnl"]:.0f}  '
           f'WR={s["win_rate"]:.1f}%  Caps={r["cap_count"]:>3}  CapPnL=${r["cap_pnl"]:.0f}  '
           f'KF={r["kfold"]["verdict"]}')
        cap_sweep[label] = r

    save('cap_sweep', cap_sweep)

    # ── Trail Comparison: R202 vs R155 vs LIVE_PARITY with optimal SL ──
    pf(f'\n{"="*80}\nTrail × SL Combinations\n{"="*80}')
    trail_configs = [
        ('R155 trail (0.14/0.025) + SL=3.5',  3.5, 0, 0, {'trailing_activate_atr': 0.14, 'trailing_distance_atr': 0.025}),
        ('R202 trail (0.06/0.015) + SL=3.5',  3.5, 0, 0, {'trailing_activate_atr': 0.06, 'trailing_distance_atr': 0.015}),
        ('R155 trail + SL=6.0 + no Cap',      6.0, 0, 0, {'trailing_activate_atr': 0.14, 'trailing_distance_atr': 0.025}),
        ('R202 trail + SL=6.0 + no Cap',      6.0, 0, 0, {'trailing_activate_atr': 0.06, 'trailing_distance_atr': 0.015}),
        ('R202 trail + SL=6.0 + Cap $70',     6.0, 70, 0, {'trailing_activate_atr': 0.06, 'trailing_distance_atr': 0.015}),
    ]
    trail_results = {}
    for label, sl, cap_f, cap_a, extra in trail_configs:
        r = run_config(data, label, sl, cap_f, cap_a, extra)
        s = r['stats']
        pf(f'  {label:<40} n={s["n"]:>5}  Sh={s["sharpe"]:.3f}  PnL=${s["pnl"]:.0f}  '
           f'WR={s["win_rate"]:.1f}%  Caps={r["cap_count"]:>3}  KF={r["kfold"]["verdict"]}')
        trail_results[label] = r

    save('trail_combinations', trail_results)

    # ── Final Summary ──
    pf(f'\n{"="*80}\nFINAL SUMMARY\n{"="*80}')
    pf(f'\n  Main configs comparison:')
    pf(f'  {"Config":<45} {"n":>5} {"Sharpe":>7} {"PnL":>10} {"WR":>6} {"Caps":>5} {"KF":>5}')
    pf(f'  {"-"*85}')
    for label, r in results.items():
        s = r['stats']
        pf(f'  {label:<45} {s["n"]:>5} {s["sharpe"]:>7.3f} ${s["pnl"]:>8.0f} {s["win_rate"]:>5.1f}% {r["cap_count"]:>5} {r["kfold"]["verdict"]:>5}')

    pf(f'\n  SL sweep (no Cap):')
    pf(f'  {"SL":>6} {"n":>5} {"Sharpe":>7} {"PnL":>10} {"WR":>6} {"MaxDD":>8} {"KF":>5}')
    pf(f'  {"-"*55}')
    for label, r in sl_sweep.items():
        s = r['stats']
        pf(f'  {label:>6} {s["n"]:>5} {s["sharpe"]:>7.3f} ${s["pnl"]:>8.0f} {s["win_rate"]:>5.1f}% ${s["max_dd"]:>7.0f} {r["kfold"]["verdict"]:>5}')

    # Recommendation
    best_sl = max(sl_sweep.items(), key=lambda x: x[1]['stats']['sharpe'] if x[1]['kfold']['verdict'] == 'PASS' else -999)
    pf(f'\n  ═══ Recommendation ═══')
    pf(f'  Best SL (no Cap, K-Fold PASS): {best_sl[0]} — Sharpe={best_sl[1]["stats"]["sharpe"]:.3f}')

    config_b = results.get('B: Live-Actual (SL=6.0, Cap=$70)', {})
    pf(f'  Current live config (B): Sharpe={config_b.get("stats",{}).get("sharpe",0):.3f}, Caps={config_b.get("cap_count",0)}')

    elapsed = time.time() - t0
    pf(f'\n  Total runtime: {elapsed:.0f}s ({elapsed/3600:.1f}h)')


if __name__ == '__main__':
    main()
