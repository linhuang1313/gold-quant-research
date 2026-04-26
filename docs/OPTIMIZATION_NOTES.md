# Backtest Engine Optimization Notes

> Sprint: P1 Hot-Path Vectorization
> Status: **CLOSED** (2026-04-27)
> Decision: Stop further optimization. Speed is no longer the bottleneck.

---

## TL;DR

| Metric | Pre-P1 | Post-P1 | Speedup |
|---|---|---|---|
| Single backtest (2025-H1 benchmark window, ~6 months) | 25.0s | 4.8s | **5.2x** |
| R38 full chain (3 variants × 6 folds, 2015–2026) | ~45 min (est.) | **7.90 min** (measured) | **~5.7x** |
| Workflow impact | "wait overnight" | "wait few minutes" | qualitative shift |

**P1 paid off. Sprint is closed. Moving on.**

---

## What changed in P1

Single commit: `f64b9c9` — *"P1: pre-extract H1 columns + vectorize intraday score (1.93x speedup)"*

### Hot-path changes in `backtest/engine.py`

1. **H1 column pre-extraction** — `_build_h1_arrays()` extracts `ATR`, `atr_percentile`, `ADX`, `EMA100`, `KC_upper/lower/mid`, `Close` from the H1 DataFrame into NumPy arrays once at engine init.
2. **ATR-percentile pre-computation** — `_compute_live_atr_pct_array()` pre-calculates the rolling-50 percentile rank for every H1 bar at init (was previously O(n) per bar inside the hot loop, ~28% of total runtime).
3. **`_update_intraday_score` vectorization** — `_compute_start_of_day()` + `_build_intraday_score_arrays()` pre-compute the intraday score and regime tag for every H1 bar via vectorized NumPy ops. Hot-loop reduces to O(1) array lookup.
4. **`h1_idx` threading** — `run()`, `_check_exits()`, `_check_h1_entries()`, `_check_m15_entries()`, `_check_m15_custom_rsi()`, `_process_signals()`, `_close_position()`, `_calc_dynamic_spread()` all accept an `h1_idx` parameter and use array accessors instead of repeated `df.iloc[]` / Series lookups.
5. **Filter conversions** — KC bandwidth, ADX gray-zone, EMA slope, `min_h1_bars_today` all switched to array-backed lookups.

### Strict correctness verification

Baseline benchmark (2025-01-01 → 2025-06-30, lead-in for warm-up):

| Metric | Baseline | P1 | Δ |
|---|---|---|---|
| n_trades | 1015 | 1015 | 0 |
| total_pnl | $4,117.19 | $4,117.19 | $0.00 |
| Sharpe | 11.097 | 11.097 | 0.000 |

**Results bit-identical**, well within tolerances (PnL <$0.01, Sharpe <0.001).

### Variance check (5-run on P1)

`benchmarks/variance_check.py`: mean=4.81s, std=0.21s, **CV = 4.44%** (target <8%). Stable.

---

## What's deliberately NOT done

These were on the original roadmap but **dropped after Step-1 R38 chain measurement showed 7.90 min — well under the 15-min decision threshold**.

| Phase | Estimated upside | Why dropped |
|---|---|---|
| **Multiprocessing K-Fold** (6-core pool) | ~2.5–3x → R38 chain to ~2.5 min | Renaming "fast enough → very fast" is gradient, not phase change. Implementation cost (Windows-compatible, disk-shared data) ~2h, ROI low. |
| **P2: M15→H1 index pre-mapping** (`np.searchsorted` + `clip`) | Marginal — `iloc` calls already mostly removed in P1 | Profile shows P1 already eliminated the bulk of `iloc` cost. Predicted gain <10%. |
| **P3: M15 OHLC NumPy arrays** | Marginal — M15 row reads now infrequent | Most M15 reads are at signal-fire moments only, not every bar. Predicted gain <5%. |

### Rationale for stopping

- **7.9 min/full-chain is in the "developer flow" zone** (submit → quick wait → see result). Going to 2.5 min is gradient, not qualitative.
- **The real bottleneck moved**: backtest Sharpe ~11 vs live Sharpe ~1.2 — a 9-10x gap that no amount of engine speed fixes.
- **Engineering discipline**: shipping and validating beats over-engineering.

---

## Future "if needed" optimizations

Reserved for later only if a specific use-case forces it.

| Trigger | Optimization to consider |
|---|---|
| Need to run >30 full chains/day (param sweep) | Multiprocessing K-Fold wrapper |
| Single backtest must drop below 2s | P2 (M15→H1 mapping) + P3 (M15 OHLC arrays) |
| Hot-loop needs to extend with new H1 features | Add the new column to `_build_h1_arrays()` for free O(1) access |

---

## Benchmark artifacts

Stored in `benchmarks/results/`:
- `baseline.json` — pre-P1 reference
- `p1_full.json`, `p1_full_run2.json` — post-P1 measurements
- `p1_h1_arrays.json`, `p1_h1_arrays_run2.json` — intermediate (H1 arrays only, before intraday-score vectorization)
- `r38_chain_seq.json` — Step-1 full-chain measurement (decisive evidence to close sprint)
- `r38_chain_seq.log` / `r38_chain_seq_attempt1.log` — runtime logs
- `profile.prof` / `profile_log.txt` — cProfile evidence used to plan P1

Scripts:
- `benchmarks/benchmark_engine.py` — single-backtest correctness + speed benchmark
- `benchmarks/variance_check.py` — 5-run CV check
- `benchmarks/profile_engine.py` — cProfile harness
- `benchmarks/benchmark_r38_chain.py` — multi-variant K-Fold chain benchmark
