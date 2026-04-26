"""R43: L8_BASE filter-combination K-Fold validation (post-P1 engine).

Tests 8 variants from the {KCBW5, Cap80, SkipUTC02-04} truth table on the
6-fold K-Fold scheme. Pre-corrects R42's main flaw (Cap30 → Cap80, since
Cap30 caps ~95% of losing trades and overrides the strategy SL).

KEY QUESTION
------------
Does V1 (KCBW5 alone) capture most of V7 (all 3 filters)' gain over V0?
If yes, Cap80 + SkipUTC are redundant decoration and we can ship simpler.

EFFICIENCY
----------
All 3 "filters" are post-hoc on the trade list, so we only run the engine
6 times (once per fold) and apply the 8 filter masks in-memory.
~3 minutes total instead of ~20.

USAGE
-----
    python -m experiments.run_round43
"""
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import research_config as config
from backtest.engine import TradeRecord
from backtest.runner import DataBundle, LIVE_PARITY_KWARGS, run_variant
from backtest.stats import calc_stats


OUT_DIR = ROOT / "results" / "round43_filter_combo"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# === L8_BASE config (replicated verbatim from run_r42_showdown.py) ===
L8_BASE = {
    **LIVE_PARITY_KWARGS,
    'keltner_adx_threshold': 14,
    'regime_config': {
        'low':    {'trail_act': 0.22, 'trail_dist': 0.04},
        'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
        'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
    },
    'min_entry_gap_hours': 1.0,
    'keltner_max_hold_m15': 20,
}


# === Fold definitions (matching backtest.runner.run_kfold) ===
FOLDS: List[Tuple[str, str, str]] = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-01"),
]


# === Filter parameters ===
CAP_USD = 80                      # NOT 30 — see OPTIMIZATION_NOTES context
SKIP_HOURS: Tuple[int, ...] = (2, 3, 4)   # UTC 02:00–04:59 (inclusive of 02, exclusive of 05)
KCBW_EMA = 25
KCBW_MULT = 1.2
KCBW_LOOKBACK = 5


# === 8-variant truth table ===
VARIANTS: List[Tuple[str, bool, bool, bool]] = [
    # (label,           use_kcbw5, use_cap80, use_skip_utc)
    ("V0_BASELINE",     False, False, False),
    ("V1_KCBW5",        True,  False, False),
    ("V2_Cap80",        False, True,  False),
    ("V3_SkipUTC",      False, False, True),
    ("V4_KCBW_Cap",     True,  True,  False),
    ("V5_KCBW_Skip",    True,  False, True),
    ("V6_Cap_Skip",     False, True,  True),
    ("V7_ALL",          True,  True,  True),
]


# ────────────────────────────────────────────────────────────────────────
# Filter implementations
# ────────────────────────────────────────────────────────────────────────

def filter_kcbw5(trades: List[TradeRecord], h1_df: pd.DataFrame) -> List[TradeRecord]:
    """Keep only entries where KC bandwidth at the entry's H1 bar is expanding
    (current BW > rolling-min over the previous KCBW_LOOKBACK bars).

    Mirrors filter_kcbw5() in experiments/run_r42_showdown.py for parity.
    """
    if h1_df is None or len(h1_df) == 0 or not trades:
        return list(trades)

    if 'ATR' in h1_df.columns:
        atr = h1_df['ATR']
    else:
        atr = (h1_df['High'] - h1_df['Low']).rolling(14).mean()
    ema = h1_df['Close'].ewm(span=KCBW_EMA).mean()
    bw = (KCBW_MULT * atr * 2.0) / ema.replace(0, np.nan)
    bw_min = bw.rolling(KCBW_LOOKBACK).min()
    expanding = (bw > bw_min.shift(1)).fillna(False)

    h1_index = h1_df.index
    out: List[TradeRecord] = []
    for t in trades:
        et = pd.Timestamp(t.entry_time)
        if et.tzinfo is None:
            et = et.tz_localize('UTC')
        pos = h1_index.searchsorted(et, side='right') - 1
        if pos < 0:
            continue
        ts = h1_index[pos]
        if ts in expanding.index and bool(expanding.loc[ts]):
            out.append(t)
    return out


def filter_skip_utc(trades: List[TradeRecord],
                    hours: Tuple[int, ...] = SKIP_HOURS) -> List[TradeRecord]:
    """Drop trades whose entry hour (UTC) is in `hours`."""
    skip = set(hours)
    return [t for t in trades if pd.Timestamp(t.entry_time).hour not in skip]


def apply_max_loss_cap(trades: List[TradeRecord], cap_usd: float) -> List[TradeRecord]:
    """Hard-cap losses at -cap_usd USD per trade (post-hoc PnL clip)."""
    out: List[TradeRecord] = []
    for t in trades:
        if t.pnl < -cap_usd:
            out.append(replace(t, pnl=-cap_usd))
        else:
            out.append(t)
    return out


def apply_variant(trades: List[TradeRecord], h1_df: pd.DataFrame,
                  use_kcbw5: bool, use_cap: bool, use_skip: bool) -> List[TradeRecord]:
    out = list(trades)
    if use_kcbw5:
        out = filter_kcbw5(out, h1_df)
    if use_skip:
        out = filter_skip_utc(out)
    if use_cap:
        out = apply_max_loss_cap(out, CAP_USD)
    return out


# ────────────────────────────────────────────────────────────────────────
# Metric helpers
# ────────────────────────────────────────────────────────────────────────

def reconstruct_equity_curve(trades: List[TradeRecord], capital: float) -> List[float]:
    """Trade-by-trade equity curve (sorted by exit_time). Sufficient for max_dd."""
    if not trades:
        return [capital]
    sorted_trades = sorted(trades, key=lambda t: t.exit_time)
    eq = [capital]
    cur = capital
    for t in sorted_trades:
        cur += t.pnl
        eq.append(cur)
    return eq


def corrected_sharpe(trades: List[TradeRecord]) -> float:
    """R42-parity Sharpe: aggregate by exit_date, fill non-trade business days
    with 0 PnL, then mean/std * sqrt(252).

    More conservative than calc_stats.sharpe (which only uses days with trades).
    Used by R42 K-Fold PASS criterion.
    """
    if not trades:
        return 0.0
    daily: Dict = {}
    for t in trades:
        d = pd.Timestamp(t.exit_time).date()
        daily[d] = daily.get(d, 0.0) + t.pnl
    if not daily:
        return 0.0
    start = min(daily.keys())
    end = max(daily.keys())
    full = [daily.get(d.date(), 0.0) for d in pd.bdate_range(start, end)]
    arr = np.array(full)
    if len(arr) < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std <= 0:
        return 0.0
    return float(np.mean(arr) / std * np.sqrt(252))


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            try:
                f.write(data)
            except Exception:
                pass
            try:
                f.flush()
            except Exception:
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass


def main() -> None:
    t_start = time.time()
    log_path = OUT_DIR / "R43_output.txt"
    log_f = open(log_path, 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.__stdout__, log_f)

    print("=" * 80, flush=True)
    print("  R43: L8_BASE filter-combination K-Fold validation", flush=True)
    print(f"  Variants: {len(VARIANTS)} (KCBW5 x Cap80 x SkipUTC truth table)", flush=True)
    print(f"  Folds:    {len(FOLDS)} (2015 to 2026-Q1)", flush=True)
    print(f"  CAP_USD = ${CAP_USD}    SKIP_HOURS (UTC) = {SKIP_HOURS}", flush=True)
    print(f"  KCBW: ema={KCBW_EMA} mult={KCBW_MULT} lookback={KCBW_LOOKBACK}", flush=True)
    print("=" * 80, flush=True)
    print(f"  Start: {datetime.now()}\n", flush=True)

    print("  Loading data once...", flush=True)
    data = DataBundle.load_default()
    print(f"  M15: {len(data.m15_df)} bars   H1: {len(data.h1_df)} bars\n", flush=True)

    print("  ===== Step 1: Run L8_BASE on each fold (engine fires 6 times only) =====",
          flush=True)
    fold_trades: Dict[str, List[TradeRecord]] = {}
    fold_h1: Dict[str, pd.DataFrame] = {}
    for fold_name, start, end in FOLDS:
        fold_data = data.slice(start, end)
        if len(fold_data.m15_df) < 1000 or len(fold_data.h1_df) < 200:
            print(f"  [{fold_name}] SKIPPED (insufficient data)", flush=True)
            continue
        stats = run_variant(fold_data, f"L8_BASE_{fold_name}", **L8_BASE)
        fold_trades[fold_name] = stats['_trades']
        fold_h1[fold_name] = fold_data.h1_df

    print("\n  ===== Step 2: Apply 8 filter combinations to each fold =====", flush=True)
    grid: Dict[Tuple[str, str], Dict] = {}
    for v_label, use_kcbw5, use_cap, use_skip in VARIANTS:
        print(f"\n  [{v_label}]  KCBW5={use_kcbw5} Cap80={use_cap} SkipUTC={use_skip}",
              flush=True)
        for fold_name in fold_trades:
            base_trades = fold_trades[fold_name]
            v_trades = apply_variant(base_trades, fold_h1[fold_name],
                                     use_kcbw5, use_cap, use_skip)
            eq = reconstruct_equity_curve(v_trades, config.CAPITAL)
            stats = calc_stats(v_trades, eq)
            csh = corrected_sharpe(v_trades)
            stats['corrected_sharpe'] = csh
            grid[(v_label, fold_name)] = stats
            print(f"    {fold_name}: n={stats['n']:>4}  "
                  f"Sh={stats['sharpe']:>5.2f}  CorrSh={csh:>5.2f}  "
                  f"PnL=${stats['total_pnl']:>7.0f}  "
                  f"DD=${stats['max_dd']:>5.0f}", flush=True)

    print("\n  ===== Step 3: Per-variant K-Fold summary =====", flush=True)
    summary: Dict[str, Dict] = {}
    for v_label, _, _, _ in VARIANTS:
        per_fold = []
        for fold_name in fold_trades:
            s = grid[(v_label, fold_name)]
            per_fold.append({
                'fold': fold_name,
                'n': int(s['n']),
                'sharpe': round(float(s['sharpe']), 3),
                'corrected_sharpe': round(float(s['corrected_sharpe']), 3),
                'total_pnl': round(float(s['total_pnl']), 2),
                'max_dd': round(float(s['max_dd']), 2),
                'win_rate': round(float(s['win_rate']), 2),
            })
        sharpes = [r['sharpe'] for r in per_fold]
        csharpes = [r['corrected_sharpe'] for r in per_fold]
        pnls = [r['total_pnl'] for r in per_fold]
        ns = [r['n'] for r in per_fold]
        # K-Fold PASS = corrected_sharpe > 0 in EVERY fold (R42-style)
        passed = sum(1 for s in csharpes if s > 0)
        summary[v_label] = {
            'n_folds': len(per_fold),
            'kfold_pass_count': passed,
            'kfold_pass': passed == len(per_fold),
            'sharpe_mean': round(float(np.mean(sharpes)), 3),
            'sharpe_min':  round(float(min(sharpes)), 3),
            'sharpe_max':  round(float(max(sharpes)), 3),
            'sharpe_std':  round(float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0, 3),
            'corr_sharpe_mean': round(float(np.mean(csharpes)), 3),
            'corr_sharpe_min':  round(float(min(csharpes)), 3),
            'pnl_total':       round(float(sum(pnls)), 2),
            'n_trades_total':  int(sum(ns)),
            'per_fold': per_fold,
        }

    # ── Headline tables ──
    print()
    print(f"  {'Variant':<16} {'Pass':>5}  {'Sh.Mean':>8} {'Sh.Min':>7} {'CSh.Mean':>9} {'CSh.Min':>8} "
          f"{'TotPnL':>10} {'N':>7}", flush=True)
    print(f"  {'-'*16} {'-'*5}  {'-'*8} {'-'*7} {'-'*9} {'-'*8} {'-'*10} {'-'*7}", flush=True)
    for v_label, _, _, _ in VARIANTS:
        s = summary[v_label]
        flag = "OK" if s['kfold_pass'] else "  "
        print(f"  {v_label:<16} {s['kfold_pass_count']}/{s['n_folds']} {flag} "
              f"{s['sharpe_mean']:>7.2f}  {s['sharpe_min']:>6.2f}  "
              f"{s['corr_sharpe_mean']:>8.2f}  {s['corr_sharpe_min']:>7.2f}  "
              f"${s['pnl_total']:>9.0f}  {s['n_trades_total']:>6}", flush=True)

    # ── Key contrast: V0 / V1 / V7 ──
    v0, v1, v7 = summary['V0_BASELINE'], summary['V1_KCBW5'], summary['V7_ALL']

    print(f"\n  ===== KEY QUESTION: Does V1 (KCBW5 alone) approx V7 (all 3) ? =====", flush=True)
    print(f"  Metric                   {'V0':>8} {'V1':>8} {'V7':>8}", flush=True)
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}", flush=True)
    print(f"  Sharpe mean              "
          f"{v0['sharpe_mean']:>8.2f} {v1['sharpe_mean']:>8.2f} {v7['sharpe_mean']:>8.2f}",
          flush=True)
    print(f"  Sharpe min               "
          f"{v0['sharpe_min']:>8.2f} {v1['sharpe_min']:>8.2f} {v7['sharpe_min']:>8.2f}",
          flush=True)
    print(f"  Corrected Sharpe mean    "
          f"{v0['corr_sharpe_mean']:>8.2f} {v1['corr_sharpe_mean']:>8.2f} {v7['corr_sharpe_mean']:>8.2f}",
          flush=True)
    print(f"  Corrected Sharpe min     "
          f"{v0['corr_sharpe_min']:>8.2f} {v1['corr_sharpe_min']:>8.2f} {v7['corr_sharpe_min']:>8.2f}",
          flush=True)
    print(f"  Total PnL (USD)          "
          f"{v0['pnl_total']:>8.0f} {v1['pnl_total']:>8.0f} {v7['pnl_total']:>8.0f}",
          flush=True)
    print(f"  Trades                   "
          f"{v0['n_trades_total']:>8} {v1['n_trades_total']:>8} {v7['n_trades_total']:>8}",
          flush=True)
    print(f"  K-Fold pass              "
          f"{v0['kfold_pass_count']:>4}/{v0['n_folds']:<3} "
          f"{v1['kfold_pass_count']:>4}/{v1['n_folds']:<3} "
          f"{v7['kfold_pass_count']:>4}/{v7['n_folds']:<3}", flush=True)

    sh_v0_v1 = v1['sharpe_mean'] - v0['sharpe_mean']
    sh_v0_v7 = v7['sharpe_mean'] - v0['sharpe_mean']
    sh_v1_v7 = v7['sharpe_mean'] - v1['sharpe_mean']
    print(f"\n  Sharpe gain V0 -> V1 (KCBW5 alone):                {sh_v0_v1:+.3f}", flush=True)
    print(f"  Sharpe gain V0 -> V7 (all 3 filters):              {sh_v0_v7:+.3f}", flush=True)
    print(f"  Sharpe extra V1 -> V7 (Cap80 + SkipUTC marginal): {sh_v1_v7:+.3f}", flush=True)

    if abs(sh_v0_v7) > 0.05:
        captured = sh_v0_v1 / sh_v0_v7 * 100.0
        print(f"  V1 captures {captured:.0f}% of V7's gain over V0.", flush=True)
        if captured >= 80:
            verdict = "V1 ~= V7  -> Cap80 + SkipUTC are decoration; KCBW5 alone suffices."
        elif captured >= 50:
            verdict = "V1 ~ V7  -> Most gain from KCBW5; Cap80/SkipUTC give marginal extra."
        else:
            verdict = "V1 << V7 -> All 3 filters meaningfully contribute (no redundancy)."
    elif abs(sh_v0_v7) <= 0.05:
        verdict = "V7 ~ V0 -> Filters do not improve Sharpe materially. Reject the combo."
    print(f"\n  >>> CONCLUSION: {verdict}", flush=True)

    # ── PnL retention contrast (filters reduce trade count -> total PnL drops, but is it healthier?) ──
    if v0['n_trades_total'] > 0:
        v1_n_pct = 100.0 * v1['n_trades_total'] / v0['n_trades_total']
        v7_n_pct = 100.0 * v7['n_trades_total'] / v0['n_trades_total']
        print(f"\n  Trade retention vs V0:    V1 keeps {v1_n_pct:.0f}%   V7 keeps {v7_n_pct:.0f}%", flush=True)
    if v0['pnl_total'] != 0:
        v1_pnl_pct = 100.0 * v1['pnl_total'] / v0['pnl_total']
        v7_pnl_pct = 100.0 * v7['pnl_total'] / v0['pnl_total']
        print(f"  PnL retention vs V0:      V1 keeps {v1_pnl_pct:.0f}%   V7 keeps {v7_pnl_pct:.0f}%", flush=True)

    # ── Save JSON ──
    out_json = OUT_DIR / "R43_filter_combo.json"
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'cap_usd': CAP_USD,
            'skip_hours_utc': list(SKIP_HOURS),
            'variants_tested': len(VARIANTS),
            'folds': len(FOLDS),
            'l8_base_kwargs': {k: v for k, v in L8_BASE.items() if isinstance(v, (int, float, bool, str, dict, list))},
            'summary': summary,
        }, fh, indent=2, default=str)
    print(f"\n  Saved: {out_json}", flush=True)
    print(f"  Total elapsed: {time.time() - t_start:.1f}s "
          f"({(time.time() - t_start) / 60.0:.2f} min)", flush=True)


if __name__ == '__main__':
    main()
