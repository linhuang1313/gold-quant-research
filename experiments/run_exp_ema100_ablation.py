#!/usr/bin/env python3
"""
EXP-EMA100: EMA100 Trend Filter Ablation
==========================================
Test: what happens if we remove the EMA100 filter from Keltner and M15 RSI?
- Baseline: current production (close > EMA100 for BUY, close < EMA100 for SELL)
- NoEMA_KC: remove EMA100 from Keltner only
- NoEMA_RSI: remove EMA100 from M15 RSI only
- NoEMA_ALL: remove EMA100 from both

Parallel version — uses multiprocessing for K-Fold.
"""
import sys, os, time, multiprocessing as mp
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_FILE = "exp_ema100_ablation_output.txt"

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-01"),
]


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def patch_no_ema_keltner():
    """Monkey-patch check_keltner_signal to remove EMA100 filter."""
    import indicators as sig_mod
    import pandas as pd

    original_func = sig_mod.check_keltner_signal

    def check_keltner_no_ema(df):
        if len(df) < 105:
            return None
        latest = df.iloc[-1]
        close = float(latest['Close'])
        kc_upper = float(latest['KC_upper'])
        kc_lower = float(latest['KC_lower'])
        adx = float(latest['ADX'])
        if any(pd.isna(v) for v in [kc_upper, kc_lower, adx]):
            return None
        if adx < sig_mod.ADX_TREND_THRESHOLD:
            return None
        sl = sig_mod._calc_atr_stop(df)
        tp = sig_mod._calc_atr_tp(df)
        if close > kc_upper:
            return {'strategy': 'keltner', 'signal': 'BUY', 'close': close, 'sl': sl, 'tp': tp,
                    'reason': f"KC BUY no_ema: {close:.2f}>{kc_upper:.2f} ADX={adx:.1f}"}
        if close < kc_lower:
            return {'strategy': 'keltner', 'signal': 'SELL', 'close': close, 'sl': sl, 'tp': tp,
                    'reason': f"KC SELL no_ema: {close:.2f}<{kc_lower:.2f} ADX={adx:.1f}"}
        return None

    sig_mod.check_keltner_signal = check_keltner_no_ema
    return original_func


def patch_no_ema_rsi(engine_class):
    """Monkey-patch BacktestEngine._check_m15_rsi to remove EMA100 filter."""
    import pandas as pd
    import numpy as np

    original_method = engine_class.__dict__.get('_check_m15_rsi')

    def _check_m15_rsi_no_ema(self, m15_window, h1_window, bar_time):
        latest = m15_window.iloc[-1]
        close = float(latest['Close'])
        rsi2 = float(latest['RSI2'])
        sma50 = float(latest['SMA50'])
        if pd.isna(rsi2) or pd.isna(sma50):
            return
        h1_adx_val = 0
        if h1_window is not None and len(h1_window) > 0:
            h1_adx_val = float(h1_window.iloc[-1].get('ADX', 0))
        if self._rsi_adx_filter > 0 and h1_adx_val > self._rsi_adx_filter:
            return
        sl = float(latest.get('ATR', 5.0)) * 2.0
        if np.isnan(sl) or sl <= 0:
            sl = 5.0
        buy_th = self._rsi_buy_threshold or 15
        sell_th = self._rsi_sell_threshold or 85
        sig = None
        if rsi2 < buy_th and close > sma50:
            sig = {'strategy': 'm15_rsi', 'signal': 'BUY', 'close': close, 'sl': sl, 'tp': 0,
                   'reason': f"RSI BUY no_ema: RSI2={rsi2:.1f}<{buy_th}"}
        elif rsi2 > sell_th and close < sma50:
            sig = {'strategy': 'm15_rsi', 'signal': 'SELL', 'close': close, 'sl': sl, 'tp': 0,
                   'reason': f"RSI SELL no_ema: RSI2={rsi2:.1f}>{sell_th}"}
        if sig:
            self._pending_signals.append(([sig], 'M15'))

    engine_class._check_m15_rsi = _check_m15_rsi_no_ema
    return original_method


def restore_keltner(original_func):
    import indicators as sig_mod
    sig_mod.check_keltner_signal = original_func


def restore_rsi(engine_class, original_method):
    if original_method is not None:
        engine_class._check_m15_rsi = original_method


def run_variant_with_patches(args):
    """Run a single variant. Patches applied per-process."""
    variant_name, patch_kc, patch_rsi, start, end = args

    from backtest import DataBundle, run_variant
    from backtest.runner import LIVE_PARITY_KWARGS
    from backtest.engine import BacktestEngine

    BASE = {**LIVE_PARITY_KWARGS}
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    if start and end:
        data = data.slice(start, end)

    orig_kc = None
    orig_rsi = None
    if patch_kc:
        orig_kc = patch_no_ema_keltner()
    if patch_rsi:
        orig_rsi = patch_no_ema_rsi(BacktestEngine)

    s = run_variant(data, variant_name, verbose=False, **BASE, spread_cost=0.30)

    if orig_kc:
        restore_keltner(orig_kc)
    if orig_rsi:
        restore_rsi(BacktestEngine, orig_rsi)

    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    return (variant_name, n, s['sharpe'], s['total_pnl'], s['win_rate'], avg, s['max_dd'])


def main():
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        def p(msg=""):
            print(msg, flush=True)
            f.write(msg + "\n")
            f.flush()

        p("=" * 80)
        p("EXP-EMA100: EMA100 Trend Filter Ablation")
        p(f"CPUs: {mp.cpu_count()}")
        p(f"Started: {datetime.now()}")
        p("=" * 80)

        t_total = time.time()

        # Part 1: Full-sample comparison (4 variants, parallel)
        p(f"\n--- Part 1: Full-sample ablation ---")
        p(f"{'Variant':<20s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'$/trade':>8s}  {'MaxDD':>11s}")
        p("-" * 80)

        variants = [
            ("Baseline", False, False, None, None),
            ("NoEMA_KC", True, False, None, None),
            ("NoEMA_RSI", False, True, None, None),
            ("NoEMA_ALL", True, True, None, None),
        ]

        with mp.Pool(4) as pool:
            results_raw = pool.map(run_variant_with_patches, variants)

        results = {}
        for name, n, sharpe, pnl, wr, avg, maxdd in results_raw:
            results[name] = {'sharpe': sharpe, 'n': n}
            marker = " <-- production" if name == "Baseline" else ""
            p(f"  {name:<18s}  {n:>6d}  {sharpe:>7.2f}  "
              f"{fmt(pnl)}  {wr:>5.1f}%  "
              f"${avg:>7.2f}  {fmt(maxdd)}{marker}")

        elapsed1 = time.time() - t_total
        p(f"\n  Part 1 done in {elapsed1/60:.1f} minutes")

        # Part 2: K-Fold for any variant that beats baseline
        baseline_sharpe = results['Baseline']['sharpe']
        better = [(n, pc, pr) for n, pc, pr in [("NoEMA_KC", True, False),
                   ("NoEMA_RSI", False, True), ("NoEMA_ALL", True, True)]
                  if results[n]['sharpe'] > baseline_sharpe + 0.05]

        if better:
            p(f"\n--- Part 2: K-Fold for variants beating baseline by >0.05 ---")
            for var_name, pc, pr in better:
                p(f"\n  {var_name}:")
                tasks = []
                for fold_name, start, end in FOLDS:
                    tasks.append((f"base_{fold_name}", False, False, start, end))
                    tasks.append((f"{var_name}_{fold_name}", pc, pr, start, end))

                with mp.Pool(min(12, mp.cpu_count())) as pool:
                    fold_results = pool.map(run_variant_with_patches, tasks)

                fold_map = {r[0]: r for r in fold_results}
                wins = 0
                deltas = []
                for fold_name, _, _ in FOLDS:
                    base_r = fold_map[f"base_{fold_name}"]
                    test_r = fold_map[f"{var_name}_{fold_name}"]
                    delta = test_r[2] - base_r[2]
                    won = delta > 0
                    if won:
                        wins += 1
                    deltas.append(delta)
                    p(f"    {fold_name}: Base={base_r[2]:>6.2f}  "
                      f"{var_name}={test_r[2]:>6.2f}  delta={delta:>+.2f} {'V' if won else 'X'}")
                avg_d = sum(deltas) / len(deltas) if deltas else 0
                p(f"    Result: {wins}/6 {'PASS' if wins >= 5 else 'FAIL'}  avg_delta={avg_d:>+.3f}")
        else:
            p(f"\n--- Part 2: No variant beats baseline by >0.05 Sharpe, skip K-Fold ---")
            p(f"  EMA100 filter is confirmed effective.")

        # Part 3: Trade count analysis
        p(f"\n--- Part 3: Trade count impact ---")
        base_n = results['Baseline']['n']
        for name in ["NoEMA_KC", "NoEMA_RSI", "NoEMA_ALL"]:
            diff = results[name]['n'] - base_n
            pct = diff / base_n * 100 if base_n > 0 else 0
            p(f"  {name}: {results[name]['n']} trades ({diff:+d}, {pct:+.1f}% vs baseline)")

        elapsed = time.time() - t_total
        p(f"\nTotal runtime: {elapsed/60:.1f} minutes")
        p(f"Completed: {datetime.now()}")

    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
