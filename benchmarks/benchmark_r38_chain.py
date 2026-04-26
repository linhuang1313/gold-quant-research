"""R38-style 3-variant x 6-fold full-chain benchmark.

Purpose
-------
After P1 single-backtest optimization, measure the actual full-chain
runtime for a multi-variant K-Fold scenario (the real R38 workflow).

Variants (subset of R38 — 3 picked for stability):
  1. BASELINE       : LIVE_PARITY_KWARGS + L7-style overrides
  2. A1_SL3.0_TP6.0 : SL=3.0, TP=6.0
  3. A2_SL3.0_TP8.0 : SL=3.0, TP=8.0

Folds: 6 (2015-2017, 2017-2019, 2019-2021, 2021-2023, 2023-2025, 2025-2026)

Modes
-----
  python -m benchmarks.benchmark_r38_chain seq      # sequential single-thread
  python -m benchmarks.benchmark_r38_chain par      # K-Fold parallel (ProcessPool)
  python -m benchmarks.benchmark_r38_chain both     # run both, report ratio
"""
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import indicators
from backtest.runner import (
    DataBundle,
    LIVE_PARITY_KWARGS,
    run_kfold,
)


RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_l7_base() -> Dict:
    kw = {**LIVE_PARITY_KWARGS}
    kw['regime_config'] = {
        'low':    {'trail_act': 0.30, 'trail_dist': 0.06},
        'normal': {'trail_act': 0.20, 'trail_dist': 0.04},
        'high':   {'trail_act': 0.08, 'trail_dist': 0.01},
    }
    kw['time_adaptive_trail'] = True
    kw['time_adaptive_trail_start'] = 2
    kw['time_adaptive_trail_decay'] = 0.75
    kw['time_adaptive_trail_floor'] = 0.003
    kw['min_entry_gap_hours'] = 1.0
    kw['keltner_max_hold_m15'] = 8
    return kw


def build_variants() -> List[tuple]:
    """Return [(label, sl_max, kwargs), ...] (mirrors R38 subset)."""
    out = []

    kw = get_l7_base()
    out.append(("BASELINE", 150, kw))

    kw = get_l7_base()
    kw['sl_atr_mult'] = 3.0
    kw['tp_atr_mult'] = 6.0
    out.append(("A1_SL3.0_TP6.0", 150, kw))

    kw = get_l7_base()
    kw['sl_atr_mult'] = 3.0
    out.append(("A2_SL3.0_TP8.0", 150, kw))

    return out


def run_chain(data: DataBundle, parallel: bool, label_suffix: str) -> Dict:
    variants = build_variants()
    print(f"\n  ===== Chain mode: {'parallel folds' if parallel else 'sequential'} =====", flush=True)

    chain_start = time.perf_counter()
    per_variant = []

    for v_label, sl_max, kw in variants:
        t0 = time.perf_counter()
        original_max = indicators.ATR_SL_MAX
        try:
            indicators.ATR_SL_MAX = sl_max
            kf = run_kfold(
                data, kw,
                n_folds=6,
                label_prefix=f"{v_label}_",
                parallel=parallel,
            )
        finally:
            indicators.ATR_SL_MAX = original_max
        elapsed = time.perf_counter() - t0
        n = sum(r['n'] for r in kf)
        pnl = sum(r['total_pnl'] for r in kf)
        sh_avg = sum(r['sharpe'] for r in kf) / len(kf)
        per_variant.append({
            'label': v_label,
            'elapsed_s': round(elapsed, 2),
            'n_trades_total': int(n),
            'pnl_total': round(float(pnl), 2),
            'sharpe_mean': round(float(sh_avg), 4),
            'fold_breakdown': [
                {'fold': r['fold'],
                 'elapsed': r.get('elapsed_s', 0),
                 'n': int(r['n']),
                 'sharpe': round(float(r['sharpe']), 3),
                 'pnl': round(float(r['total_pnl']), 2)}
                for r in kf
            ],
        })
        print(f"    [{v_label}] elapsed={elapsed:.1f}s  trades={n}  "
              f"pnl=${pnl:.0f}  sharpe_mean={sh_avg:.2f}", flush=True)

    chain_total = time.perf_counter() - chain_start

    summary = {
        'mode': 'parallel' if parallel else 'sequential',
        'chain_total_s': round(chain_total, 2),
        'chain_total_min': round(chain_total / 60.0, 2),
        'n_variants': len(variants),
        'n_folds_each': 6,
        'per_variant': per_variant,
        'timestamp': datetime.now().isoformat(),
    }
    out_path = RESULTS_DIR / f"r38_chain_{label_suffix}.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  ===> Chain TOTAL: {chain_total:.1f}s ({chain_total/60:.2f} min)", flush=True)
    print(f"       Saved: {out_path}")
    return summary


def main():
    args = sys.argv[1:]
    mode = args[0] if args else 'seq'

    print(f"  Loading default DataBundle (full 2015-... data)...", flush=True)
    data = DataBundle.load_default()
    print(f"  M15: {len(data.m15_df)} bars   H1: {len(data.h1_df)} bars", flush=True)

    if mode in ('seq', 'sequential'):
        run_chain(data, parallel=False, label_suffix='seq')
    elif mode in ('par', 'parallel'):
        run_chain(data, parallel=True, label_suffix='par')
    elif mode == 'both':
        seq = run_chain(data, parallel=False, label_suffix='seq')
        par = run_chain(data, parallel=True, label_suffix='par')
        ratio = seq['chain_total_s'] / par['chain_total_s'] if par['chain_total_s'] > 0 else float('inf')
        print()
        print(f"  ====================================================")
        print(f"  Sequential: {seq['chain_total_s']:.1f}s  ({seq['chain_total_min']:.2f} min)")
        print(f"  Parallel  : {par['chain_total_s']:.1f}s  ({par['chain_total_min']:.2f} min)")
        print(f"  Speedup   : {ratio:.2f}x")
        print(f"  ====================================================")
    else:
        print(f"Unknown mode: {mode}. Use seq | par | both")
        sys.exit(2)


if __name__ == '__main__':
    main()
