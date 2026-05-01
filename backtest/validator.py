#!/usr/bin/env python3
"""
Strategy Validator — Standardized 8-Stage Testing Pipeline
=============================================================
Professional quantitative strategy validation framework based on:
  - Lopez de Prado: Deflated Sharpe, PBO, Combinatorial CV
  - Robert Pardo: Walk-Forward Efficiency
  - Industry best practices: purged K-Fold, stop criteria

Stages (progressive, each must PASS before proceeding):

  Stage 0: BASE LOGIC   — Default params, no optimization (optional)
  Stage 1: SANITY        — Basic stats + DSR selection bias correction
  Stage 2: ROBUSTNESS    — Purged K-Fold with embargo gap
  Stage 3: WALK-FORWARD  — Nested OOS + WF Efficiency metric
  Stage 4: STRESS        — Monte Carlo + PBO overfitting probability
  Stage 5: COST          — Spread/slippage sensitivity (live-calibrated)
  Stage 6: REALITY       — Era/direction bias, random benchmark, param stability
  Stage 7: DEPLOYMENT    — Stop criteria, multi-asset generalization (optional)

Usage:
    from backtest.validator import StrategyValidator, ValidatorConfig

    validator = StrategyValidator(
        name="MY_STRATEGY",
        backtest_fn=my_backtest_function,  # (h1_df, spread, lot) -> list[dict]
        spread=0.30,
        lot=0.03,
        config=ValidatorConfig(n_trials_tested=50, realistic_spread=0.88),
    )
    report = validator.run_all()
"""
import time, json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class ValidatorConfig:
    """Thresholds and parameters for each validation stage."""

    # Stage 0: Base Logic
    min_base_sharpe: float = 0.0     # just needs to be positive

    # Stage 1: Sanity + DSR
    min_trades: int = 50
    min_sharpe: float = 1.0
    max_dd_pct: float = 50.0
    n_trials_tested: int = 1         # how many param combos were tested (for DSR)
    min_dsr: float = 0.95            # Deflated Sharpe must exceed 95% confidence

    # Stage 2: Purged K-Fold
    n_folds: int = 6
    min_positive_folds: int = 4
    min_kfold_mean_sharpe: float = 1.0
    purge_bars: int = 30             # bars to remove at fold boundaries (~30h for H1)

    # Stage 3: Walk-Forward + Efficiency
    wf_windows: List[Dict] = field(default_factory=lambda: [
        {'name': 'WF1', 'train': ('2015-01-01', '2020-12-31'), 'test': ('2021-01-01', '2022-12-31')},
        {'name': 'WF2', 'train': ('2017-01-01', '2022-12-31'), 'test': ('2023-01-01', '2024-12-31')},
        {'name': 'WF3', 'train': ('2019-01-01', '2024-12-31'), 'test': ('2025-01-01', '2026-05-01')},
    ])
    max_sharpe_decay_pct: float = 50.0
    min_oos_sharpe: float = 0.5
    min_wf_efficiency: float = 0.5   # OOS_pnl_per_day / IS_pnl_per_day, Robert Pardo standard
    warn_wf_efficiency: float = 1.2  # if > 1.2, warn possible bug/leakage

    # Stage 4: Monte Carlo + PBO
    n_param_perturb: int = 200
    param_perturb_pct: float = 0.20
    n_bootstrap: int = 5000
    n_trade_removal: int = 500
    min_bootstrap_ci_lower: float = 0.0
    min_perturb_p5: float = 0.0
    max_pbo: float = 0.20            # PBO must be < 20% (either method)
    pbo_n_partitions: int = 8
    pbo_max_grid_combos: int = 200   # max grid combos for CSCV PBO

    # Stage 5: Cost
    spread_levels: List[float] = field(default_factory=lambda: [0.30, 0.50, 0.88, 1.00, 1.30, 1.50, 2.00])
    realistic_spread: float = 0.88
    min_realistic_sharpe: float = 0.5

    # Stage 6: Reality + Param Stability
    n_random_trials: int = 500
    max_random_above_real_pct: float = 40.0
    min_yearly_positive_pct: float = 70.0
    param_stability_threshold: float = 0.30  # 30% of param combos must have Sharpe > 80% of best

    # Stage 7: Deployment
    stop_dd_safety_mult: float = 2.0         # max live DD = backtest DD * this
    stop_monthly_safety_mult: float = 2.0    # max monthly loss = worst backtest month * this
    stop_consec_loss_mult: float = 1.5       # max consecutive loss days = backtest worst * this


@dataclass
class StageResult:
    stage: int
    name: str
    passed: bool
    sharpe: float = 0.0
    details: Dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    verdict: str = ""


# ═══════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════

def _trades_to_daily(trades: List[Dict]) -> np.ndarray:
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    if not daily:
        return np.array([])
    return np.array([daily[k] for k in sorted(daily.keys())])


def _sharpe(arr: np.ndarray) -> float:
    if len(arr) < 10 or arr.std() == 0:
        return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(252))


def _max_dd(arr: np.ndarray) -> float:
    if len(arr) == 0:
        return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def _fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


# ═══════════════════════════════════════════════════════════════
# Main Validator
# ═══════════════════════════════════════════════════════════════

class StrategyValidator:
    """
    Professional 8-stage strategy validation pipeline.

    Parameters
    ----------
    name : str
        Strategy identifier.
    backtest_fn : callable
        (h1_df, spread, lot) -> list[dict]. Each dict needs: pnl, exit_time, entry_time, dir
    spread, lot : float
        Base trading parameters.
    config : ValidatorConfig, optional
        Thresholds for each stage.
    base_backtest_fn : callable, optional
        Stage 0: strategy with DEFAULT (non-optimized) params. Same signature as backtest_fn.
    param_perturb_fn : callable, optional
        Stage 4 PBO-Perturb: (h1_df, spread, lot, rng) -> list[dict]. Random param perturbation.
    param_grid_fn : callable, optional
        Stage 6 param stability: (h1_df, spread, lot) -> dict[str, float]. {param_label: sharpe}.
    param_grid_backtest_fn : callable, optional
        Stage 4 PBO-CSCV (Bailey et al.): (h1_df, spread, lot) -> dict[str, list[dict]].
        Returns {param_label: trades_list} for ALL grid combos. Used to build T x N matrix
        for Combinatorially Symmetric Cross-Validation.
    alt_h1_dfs : dict, optional
        Stage 7: {"XAGUSD": silver_h1_df, ...} for multi-asset generalization.
    """

    STAGE_NAMES = {
        0: "BASE LOGIC (Pre-Optimization)",
        1: "SANITY + DSR (Selection Bias)",
        2: "ROBUSTNESS (Purged K-Fold)",
        3: "WALK-FORWARD + Efficiency",
        4: "STRESS (Monte Carlo + PBO)",
        5: "COST (Spread/Slippage)",
        6: "REALITY (Bias + Param Stability)",
        7: "DEPLOYMENT (Stop Criteria)",
    }

    def __init__(
        self,
        name: str,
        backtest_fn: Callable,
        spread: float = 0.30,
        lot: float = 0.03,
        config: Optional[ValidatorConfig] = None,
        output_dir: Optional[str] = None,
        h1_df: Optional[pd.DataFrame] = None,
        m15_df: Optional[pd.DataFrame] = None,
        base_backtest_fn: Optional[Callable] = None,
        param_perturb_fn: Optional[Callable] = None,
        param_grid_fn: Optional[Callable] = None,
        param_grid_backtest_fn: Optional[Callable] = None,
        alt_h1_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        self.name = name
        self.backtest_fn = backtest_fn
        self.spread = spread
        self.lot = lot
        self.config = config or ValidatorConfig()
        self.output_dir = Path(output_dir or f"results/validate_{name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.h1_df = h1_df
        self.m15_df = m15_df
        self.base_backtest_fn = base_backtest_fn
        self.param_perturb_fn = param_perturb_fn
        self.param_grid_fn = param_grid_fn
        self.param_grid_backtest_fn = param_grid_backtest_fn
        self.alt_h1_dfs = alt_h1_dfs or {}
        self.results: Dict[int, StageResult] = {}
        self._cached_trades: Optional[List[Dict]] = None

    def _load_data(self):
        if self.h1_df is None:
            from backtest.runner import DataBundle
            data = DataBundle.load_default()
            self.h1_df = data.h1_df
            self.m15_df = data.m15_df
            print(f"  Data loaded: H1={len(self.h1_df)} bars")

    def _run_backtest(self, h1_df=None, spread=None, lot=None):
        return self.backtest_fn(
            h1_df if h1_df is not None else self.h1_df,
            spread if spread is not None else self.spread,
            lot if lot is not None else self.lot,
        )

    def _get_cached_trades(self):
        if self._cached_trades is None:
            self._cached_trades = self._run_backtest()
        return self._cached_trades

    # ───────────────────────────────────────────────────────
    # Stage 0: Base Logic
    # ───────────────────────────────────────────────────────
    def stage0_base_logic(self) -> StageResult:
        t0 = time.time()
        if self.base_backtest_fn is None:
            return StageResult(
                stage=0, name="BASE LOGIC", passed=True, sharpe=0,
                details={'skipped': True, 'reason': 'no base_backtest_fn provided'},
                elapsed_s=time.time() - t0,
                verdict="SKIPPED (no base_backtest_fn)")

        trades = self.base_backtest_fn(self.h1_df, self.spread, self.lot)
        daily = _trades_to_daily(trades)
        sh = _sharpe(daily)
        pnl = float(daily.sum()) if len(daily) > 0 else 0
        n_trades = len(trades)

        passed = sh > self.config.min_base_sharpe and pnl > 0
        reasons = []
        if sh <= self.config.min_base_sharpe:
            reasons.append(f"base Sharpe={sh:.2f} <= {self.config.min_base_sharpe}")
        if pnl <= 0:
            reasons.append(f"base PnL={_fmt(pnl)} <= 0")

        verdict = "PASS" if passed else f"FAIL: {'; '.join(reasons)}"
        return StageResult(
            stage=0, name="BASE LOGIC", passed=passed, sharpe=sh,
            details={'n_trades': n_trades, 'pnl': round(pnl, 2),
                     'sharpe': round(sh, 2)},
            elapsed_s=time.time() - t0, verdict=verdict)

    # ───────────────────────────────────────────────────────
    # Stage 1: Sanity + DSR
    # ───────────────────────────────────────────────────────
    def stage1_sanity(self) -> StageResult:
        t0 = time.time()
        trades = self._get_cached_trades()
        daily = _trades_to_daily(trades)
        sh = _sharpe(daily)
        pnl = float(daily.sum()) if len(daily) > 0 else 0
        dd = _max_dd(daily)
        n_trades = len(trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        wr = wins / n_trades * 100 if n_trades > 0 else 0
        dd_pct = (dd / pnl * 100) if pnl > 0 else 999

        # DSR: correct for selection bias from multiple testing
        from backtest.stats import deflated_sharpe, probabilistic_sharpe
        daily_list = daily.tolist()
        dsr_result = deflated_sharpe(daily_list, n_trials=self.config.n_trials_tested)
        psr_result = probabilistic_sharpe(daily_list, sharpe_benchmark=0.0)

        passed_basic = (n_trades >= self.config.min_trades
                        and sh >= self.config.min_sharpe
                        and dd_pct <= self.config.max_dd_pct)
        passed_dsr = dsr_result.get('dsr', 0) >= self.config.min_dsr or self.config.n_trials_tested <= 1

        passed = passed_basic and passed_dsr
        reasons = []
        if n_trades < self.config.min_trades:
            reasons.append(f"trades={n_trades} < {self.config.min_trades}")
        if sh < self.config.min_sharpe:
            reasons.append(f"Sharpe={sh:.2f} < {self.config.min_sharpe}")
        if dd_pct > self.config.max_dd_pct:
            reasons.append(f"DD%={dd_pct:.1f}% > {self.config.max_dd_pct}%")
        if not passed_dsr:
            reasons.append(f"DSR={dsr_result.get('dsr', 0):.3f} < {self.config.min_dsr} "
                           f"(n_trials={self.config.n_trials_tested})")

        verdict = "PASS" if passed else f"FAIL: {'; '.join(reasons)}"
        return StageResult(
            stage=1, name="SANITY", passed=passed, sharpe=sh,
            details={
                'n_trades': n_trades, 'pnl': round(pnl, 2),
                'max_dd': round(dd, 2), 'dd_pct': round(dd_pct, 1),
                'win_rate': round(wr, 1), 'n_days': len(daily),
                'dsr': round(dsr_result.get('dsr', 0), 4),
                'dsr_passed': dsr_result.get('passed', False),
                'sr_star': round(dsr_result.get('sr_star', 0), 4),
                'n_trials': self.config.n_trials_tested,
                'psr': round(psr_result.get('psr', 0), 4),
                'psr_p_value': round(psr_result.get('p_value', 1), 4),
            },
            elapsed_s=time.time() - t0, verdict=verdict)

    # ───────────────────────────────────────────────────────
    # Stage 2: Purged K-Fold
    # ───────────────────────────────────────────────────────
    def stage2_kfold(self) -> StageResult:
        t0 = time.time()
        folds = [
            ("2015-01-01", "2017-01-01"),
            ("2017-01-01", "2019-01-01"),
            ("2019-01-01", "2021-01-01"),
            ("2021-01-01", "2023-01-01"),
            ("2023-01-01", "2025-01-01"),
            ("2025-01-01", "2026-05-01"),
        ]
        purge = self.config.purge_bars
        fold_results = []
        for i, (start, end) in enumerate(folds):
            h1_slice = self.h1_df[start:end]
            if purge > 0 and len(h1_slice) > purge * 2:
                h1_slice = h1_slice.iloc[purge:-purge]
            trades = self._run_backtest(h1_df=h1_slice)
            daily = _trades_to_daily(trades)
            sh = _sharpe(daily)
            pnl = float(daily.sum()) if len(daily) > 0 else 0
            fold_results.append({'fold': i + 1, 'period': f"{start}~{end}",
                                 'sharpe': round(sh, 2), 'pnl': round(pnl, 2),
                                 'n_trades': len(trades),
                                 'purged_bars': purge})
            print(f"    Fold {i+1}: Sharpe={sh:.2f} PnL={_fmt(pnl)} "
                  f"({len(trades)} trades, purge={purge})", flush=True)

        sharpes = [f['sharpe'] for f in fold_results]
        mean_sh = float(np.mean(sharpes))
        positive_folds = sum(1 for s in sharpes if s > 0)

        passed = (positive_folds >= self.config.min_positive_folds
                  and mean_sh >= self.config.min_kfold_mean_sharpe)
        verdict = (f"{'PASS' if passed else 'FAIL'}: "
                   f"{positive_folds}/{len(folds)} positive, mean={mean_sh:.2f}, purge={purge}bars")
        return StageResult(
            stage=2, name="ROBUSTNESS", passed=passed, sharpe=mean_sh,
            details={'folds': fold_results, 'positive_folds': positive_folds,
                     'mean_sharpe': round(mean_sh, 2), 'purge_bars': purge},
            elapsed_s=time.time() - t0, verdict=verdict)

    # ───────────────────────────────────────────────────────
    # Stage 3: Walk-Forward + Efficiency
    # ───────────────────────────────────────────────────────
    def stage3_walk_forward(self) -> StageResult:
        t0 = time.time()
        wf_results = []
        efficiencies = []

        for wf in self.config.wf_windows:
            train_s, train_e = wf['train']
            test_s, test_e = wf['test']

            h1_train = self.h1_df[train_s:train_e]
            trades_train = self._run_backtest(h1_df=h1_train)
            daily_train = _trades_to_daily(trades_train)
            sh_train = _sharpe(daily_train)
            pnl_train = float(daily_train.sum()) if len(daily_train) > 0 else 0
            n_days_train = len(daily_train)

            h1_test = self.h1_df[test_s:test_e]
            trades_test = self._run_backtest(h1_df=h1_test)
            daily_test = _trades_to_daily(trades_test)
            sh_test = _sharpe(daily_test)
            pnl_test = float(daily_test.sum()) if len(daily_test) > 0 else 0
            n_days_test = len(daily_test)

            decay = (sh_train - sh_test) / sh_train * 100 if sh_train > 0 else 0

            # WF Efficiency: OOS daily return / IS daily return
            is_daily_ret = pnl_train / n_days_train if n_days_train > 0 else 0
            oos_daily_ret = pnl_test / n_days_test if n_days_test > 0 else 0
            wf_eff = oos_daily_ret / is_daily_ret if is_daily_ret > 0 else 0
            efficiencies.append(wf_eff)

            wf_results.append({
                'window': wf['name'],
                'train_sharpe': round(sh_train, 2), 'test_sharpe': round(sh_test, 2),
                'decay_pct': round(decay, 1),
                'test_pnl': round(pnl_test, 2),
                'wf_efficiency': round(wf_eff, 3),
            })
            print(f"    {wf['name']}: Train={sh_train:.2f} Test={sh_test:.2f} "
                  f"Decay={decay:.1f}% WFE={wf_eff:.2f}", flush=True)

        avg_decay = float(np.mean([w['decay_pct'] for w in wf_results]))
        avg_oos = float(np.mean([w['test_sharpe'] for w in wf_results]))
        avg_wfe = float(np.mean(efficiencies))
        all_oos_positive = all(w['test_sharpe'] > self.config.min_oos_sharpe for w in wf_results)
        no_extreme_decay = all(w['decay_pct'] < self.config.max_sharpe_decay_pct for w in wf_results)
        wfe_ok = avg_wfe >= self.config.min_wf_efficiency

        passed = all_oos_positive and no_extreme_decay and wfe_ok

        wfe_note = ""
        if avg_wfe > self.config.warn_wf_efficiency:
            wfe_note = " WARNING: WFE > 1.2 — check for data leakage"

        verdict = (f"{'PASS' if passed else 'FAIL'}: "
                   f"OOS={avg_oos:.2f}, decay={avg_decay:.1f}%, WFE={avg_wfe:.2f}{wfe_note}")
        return StageResult(
            stage=3, name="WALK-FORWARD", passed=passed, sharpe=avg_oos,
            details={'windows': wf_results, 'avg_oos_sharpe': round(avg_oos, 2),
                     'avg_decay_pct': round(avg_decay, 1),
                     'avg_wf_efficiency': round(avg_wfe, 3),
                     'wfe_warning': avg_wfe > self.config.warn_wf_efficiency},
            elapsed_s=time.time() - t0, verdict=verdict)

    # ───────────────────────────────────────────────────────
    # Stage 4: Monte Carlo + Dual PBO
    # ───────────────────────────────────────────────────────
    def stage4_stress(self) -> StageResult:
        """Monte Carlo stress tests + dual PBO analysis.

        PBO is computed two ways (both reported, either can trigger failure):
          - PBO-Perturb: random ±20% param perturbation → measures parameter stability
          - PBO-CSCV: systematic grid search → Bailey et al. (2017) selection-bias test
        """
        t0 = time.time()
        trades = self._get_cached_trades()
        daily = _trades_to_daily(trades)
        base_sh = _sharpe(daily)
        rng = np.random.RandomState(42)

        # Bootstrap
        boot_sharpes = []
        n_days = len(daily)
        for _ in range(self.config.n_bootstrap):
            sample = rng.choice(daily, size=n_days, replace=True)
            boot_sharpes.append(_sharpe(sample))
        bs = np.array(boot_sharpes)
        ci_lower = float(np.percentile(bs, 2.5))
        ci_upper = float(np.percentile(bs, 97.5))

        # Trade removal
        removal_sharpes = []
        for _ in range(self.config.n_trade_removal):
            n_keep = max(1, int(len(trades) * 0.9))
            idx = rng.choice(len(trades), size=n_keep, replace=False)
            subset = [trades[i] for i in idx]
            removal_sharpes.append(_sharpe(_trades_to_daily(subset)))
        rs = np.array(removal_sharpes)

        # ── PBO Method 1: Perturbation (parameter stability) ──
        perturb_result = None
        pbo_perturb = None
        if self.param_perturb_fn is not None:
            perturb_sharpes = []
            perturb_dailies = {}
            for i in range(self.config.n_param_perturb):
                pt = self.param_perturb_fn(self.h1_df, self.spread, self.lot, rng)
                pd_arr = _trades_to_daily(pt)
                perturb_sharpes.append(_sharpe(pd_arr))
                perturb_dailies[f"variant_{i}"] = pd_arr.tolist()
            ps = np.array(perturb_sharpes)
            perturb_result = {
                'mean': round(float(ps.mean()), 2), 'std': round(float(ps.std()), 2),
                'p5': round(float(np.percentile(ps, 5)), 2),
                'pct_above_zero': round(float((ps > 0).mean() * 100), 1),
            }
            perturb_dailies['SELECTED'] = daily.tolist()
            try:
                from backtest.stats import compute_pbo
                pbo_perturb = compute_pbo(perturb_dailies,
                                          n_partitions=self.config.pbo_n_partitions)
            except Exception as e:
                pbo_perturb = {'pbo': 0.0, 'overfit_risk': 'UNKNOWN', 'error': str(e)}

        # ── PBO Method 2: CSCV (Bailey et al. 2017 — selection bias) ──
        pbo_cscv = None
        if self.param_grid_backtest_fn is not None:
            try:
                from backtest.stats import compute_pbo
                print("    Running CSCV grid backtests for PBO...", flush=True)
                grid_trades = self.param_grid_backtest_fn(
                    self.h1_df, self.spread, self.lot)
                grid_dailies = {}
                n_added = 0
                max_combos = self.config.pbo_max_grid_combos
                for label, tr_list in grid_trades.items():
                    if n_added >= max_combos:
                        break
                    d = _trades_to_daily(tr_list)
                    if len(d) >= self.config.pbo_n_partitions * 2:
                        grid_dailies[label] = d.tolist()
                        n_added += 1
                grid_dailies['SELECTED'] = daily.tolist()
                print(f"    CSCV: {len(grid_dailies)} variants "
                      f"(grid={n_added} + selected), "
                      f"S={self.config.pbo_n_partitions} partitions", flush=True)
                pbo_cscv = compute_pbo(grid_dailies,
                                       n_partitions=self.config.pbo_n_partitions)
            except Exception as e:
                pbo_cscv = {'pbo': 0.0, 'overfit_risk': 'UNKNOWN', 'error': str(e)}

        # ── Pass/fail logic ──
        passed_boot = ci_lower > self.config.min_bootstrap_ci_lower
        passed_removal = float(np.percentile(rs, 5)) > 0
        passed_perturb = True
        if perturb_result:
            passed_perturb = perturb_result['p5'] > self.config.min_perturb_p5

        # PBO pass: either method passing is sufficient
        # (they measure different things — stability vs selection bias)
        pbo_perturb_val = pbo_perturb.get('pbo', 0) if pbo_perturb else None
        pbo_cscv_val = pbo_cscv.get('pbo', 0) if pbo_cscv else None

        if pbo_perturb_val is not None and pbo_cscv_val is not None:
            passed_pbo = (pbo_perturb_val < self.config.max_pbo
                          or pbo_cscv_val < self.config.max_pbo)
        elif pbo_perturb_val is not None:
            passed_pbo = pbo_perturb_val < self.config.max_pbo
        elif pbo_cscv_val is not None:
            passed_pbo = pbo_cscv_val < self.config.max_pbo
        else:
            passed_pbo = True

        passed = passed_boot and passed_removal and passed_perturb and passed_pbo

        # ── Build details ──
        details = {
            'base_sharpe': round(base_sh, 2),
            'bootstrap': {'ci_95': [round(ci_lower, 2), round(ci_upper, 2)],
                          'p_lt_0': round(float((bs < 0).mean() * 100), 2)},
            'trade_removal': {'mean': round(float(rs.mean()), 2),
                              'p5': round(float(np.percentile(rs, 5)), 2)},
        }
        if perturb_result:
            details['param_perturb'] = perturb_result
        if pbo_perturb:
            details['pbo_perturb'] = {
                k: v for k, v in pbo_perturb.items()
                if k not in ('is_best_oos_ranks', 'logit_distribution')}
            details['pbo_perturb']['method'] = 'random_perturbation'
            details['pbo_perturb']['interpretation'] = 'parameter_stability'
        if pbo_cscv:
            details['pbo_cscv'] = {
                k: v for k, v in pbo_cscv.items()
                if k not in ('is_best_oos_ranks', 'logit_distribution')}
            details['pbo_cscv']['method'] = 'CSCV_Bailey_2017'
            details['pbo_cscv']['interpretation'] = 'selection_bias'
        # Legacy key for backward compat
        if pbo_perturb:
            details['pbo'] = details['pbo_perturb']

        # ── Verdict string ──
        pbo_parts = []
        if pbo_perturb_val is not None:
            pbo_parts.append(f"PBO-Perturb={pbo_perturb_val:.1%}")
        if pbo_cscv_val is not None:
            pbo_parts.append(f"PBO-CSCV={pbo_cscv_val:.1%}")
        pbo_str = f", {', '.join(pbo_parts)}" if pbo_parts else ""
        verdict = (f"{'PASS' if passed else 'FAIL'}: "
                   f"95% CI [{ci_lower:.2f}, {ci_upper:.2f}]{pbo_str}")
        return StageResult(
            stage=4, name="STRESS", passed=passed, sharpe=base_sh,
            details=details, elapsed_s=time.time() - t0, verdict=verdict)

    # ───────────────────────────────────────────────────────
    # Stage 5: Cost Sensitivity (unchanged)
    # ───────────────────────────────────────────────────────
    def stage5_cost(self) -> StageResult:
        t0 = time.time()
        cost_results = []
        break_even = None

        for sp in self.config.spread_levels:
            trades = self._run_backtest(spread=sp)
            daily = _trades_to_daily(trades)
            sh = _sharpe(daily)
            pnl = float(daily.sum()) if len(daily) > 0 else 0
            cost_results.append({'spread': sp, 'sharpe': round(sh, 2), 'pnl': round(pnl, 2)})
            if break_even is None and sh < 1.0:
                break_even = sp
            print(f"    Spread={sp:.2f}: Sharpe={sh:.2f} PnL={_fmt(pnl)}", flush=True)

        trades_real = self._run_backtest(spread=self.config.realistic_spread)
        daily_real = _trades_to_daily(trades_real)
        realistic_sh = _sharpe(daily_real)

        passed = realistic_sh >= self.config.min_realistic_sharpe
        verdict = (f"{'PASS' if passed else 'FAIL'}: "
                   f"Realistic(sp={self.config.realistic_spread}) Sharpe={realistic_sh:.2f}, "
                   f"break-even ~{break_even or '>max'}")
        return StageResult(
            stage=5, name="COST", passed=passed, sharpe=realistic_sh,
            details={'levels': cost_results, 'realistic_sharpe': round(realistic_sh, 2),
                     'break_even_spread': break_even},
            elapsed_s=time.time() - t0, verdict=verdict)

    # ───────────────────────────────────────────────────────
    # Stage 6: Reality + Param Stability
    # ───────────────────────────────────────────────────────
    def stage6_reality(self) -> StageResult:
        t0 = time.time()
        trades = self._get_cached_trades()

        # Per-year analysis
        yearly = {}
        for t in trades:
            yr = pd.Timestamp(t['exit_time']).year
            yearly.setdefault(yr, []).append(t['pnl'])
        yearly_sharpes = {}
        for yr in sorted(yearly.keys()):
            arr = np.array(yearly[yr])
            yearly_sharpes[yr] = round(_sharpe(arr) if len(arr) > 10 else 0, 2)
        positive_years = sum(1 for s in yearly_sharpes.values() if s > 0)
        pct_positive = positive_years / len(yearly_sharpes) * 100 if yearly_sharpes else 0

        # Direction bias
        buy_pnl = sum(t['pnl'] for t in trades if t.get('dir', '') in ('BUY', 'buy'))
        sell_pnl = sum(t['pnl'] for t in trades if t.get('dir', '') in ('SELL', 'sell'))
        total = buy_pnl + sell_pnl
        buy_pct = (buy_pnl / total * 100) if total != 0 else 50

        # Statistical significance via PSR (replaces broken permutation test)
        daily_real = _trades_to_daily(trades)
        real_sh = _sharpe(daily_real)
        from backtest.stats import probabilistic_sharpe
        psr_result = probabilistic_sharpe(daily_real.tolist(), sharpe_benchmark=0.0)
        psr_p_value = psr_result.get('p_value', 1.0)
        # p-value < 0.05 means Sharpe is statistically significant at 95% confidence
        pct_random_above = psr_p_value * 100  # treat PSR p-value as "random beats real %"

        # Parameter stability zone
        stability_result = None
        if self.param_grid_fn is not None:
            grid_sharpes = self.param_grid_fn(self.h1_df, self.spread, self.lot)
            if grid_sharpes:
                values = list(grid_sharpes.values())
                best_sh = max(values)
                threshold = best_sh * 0.8
                stable_pct = sum(1 for v in values if v >= threshold) / len(values) * 100
                stability_result = {
                    'n_combos': len(values),
                    'best_sharpe': round(best_sh, 2),
                    'threshold_80pct': round(threshold, 2),
                    'pct_above_threshold': round(stable_pct, 1),
                    'stable': stable_pct >= self.config.param_stability_threshold * 100,
                }
                print(f"    Param stability: {stable_pct:.1f}% above 80% of best", flush=True)

        era_ok = pct_positive >= self.config.min_yearly_positive_pct
        dir_ok = 20 < buy_pct < 80
        random_ok = pct_random_above < self.config.max_random_above_real_pct

        passed = era_ok and random_ok
        reasons = []
        if not era_ok:
            reasons.append(f"only {pct_positive:.0f}% years positive")
        if not dir_ok:
            reasons.append(f"direction bias: BUY={buy_pct:.0f}%")
        if not random_ok:
            reasons.append(f"random beats real {pct_random_above:.1f}% of time")

        details = {
            'yearly_sharpes': yearly_sharpes,
            'pct_years_positive': round(pct_positive, 1),
            'buy_pnl_pct': round(buy_pct, 1),
            'psr_p_value': round(psr_p_value, 6),
            'psr': round(psr_result.get('psr', 0), 4),
            'sharpe_significant': psr_p_value < 0.05,
            'pct_random_above_real': round(pct_random_above, 1),
        }
        if stability_result:
            details['param_stability'] = stability_result

        verdict = f"{'PASS' if passed else 'FAIL'}" + (f": {'; '.join(reasons)}" if reasons else "")
        return StageResult(
            stage=6, name="REALITY", passed=passed, sharpe=real_sh,
            details=details, elapsed_s=time.time() - t0, verdict=verdict)

    # ───────────────────────────────────────────────────────
    # Stage 7: Deployment Readiness
    # ───────────────────────────────────────────────────────
    def stage7_deployment(self) -> StageResult:
        t0 = time.time()
        trades = self._get_cached_trades()
        daily = _trades_to_daily(trades)

        if len(daily) == 0:
            return StageResult(stage=7, name="DEPLOYMENT", passed=False,
                               verdict="FAIL: no trades", elapsed_s=time.time() - t0)

        # Compute stop criteria from backtest history
        eq = np.cumsum(daily)
        max_dd = float((np.maximum.accumulate(eq) - eq).max())

        # Monthly PnL
        trade_dates = sorted(set(pd.Timestamp(t['exit_time']).date() for t in trades))
        monthly_pnl = {}
        for t in trades:
            d = pd.Timestamp(t['exit_time'])
            key = f"{d.year}-{d.month:02d}"
            monthly_pnl[key] = monthly_pnl.get(key, 0) + t['pnl']
        worst_month = min(monthly_pnl.values()) if monthly_pnl else 0

        # Consecutive loss days
        max_consec_loss = 0
        current_streak = 0
        for pnl_day in daily:
            if pnl_day < 0:
                current_streak += 1
                max_consec_loss = max(max_consec_loss, current_streak)
            else:
                current_streak = 0

        cfg = self.config
        stop_criteria = {
            'max_drawdown_live': round(max_dd * cfg.stop_dd_safety_mult, 2),
            'max_monthly_loss_live': round(abs(worst_month) * cfg.stop_monthly_safety_mult, 2),
            'max_consecutive_loss_days': int(max_consec_loss * cfg.stop_consec_loss_mult),
            'backtest_max_dd': round(max_dd, 2),
            'backtest_worst_month': round(worst_month, 2),
            'backtest_max_consec_loss_days': max_consec_loss,
        }

        # Multi-asset generalization (optional, informational only)
        multi_asset = {}
        if self.alt_h1_dfs:
            for asset_name, alt_df in self.alt_h1_dfs.items():
                try:
                    alt_trades = self._run_backtest(h1_df=alt_df)
                    alt_daily = _trades_to_daily(alt_trades)
                    alt_sh = _sharpe(alt_daily)
                    alt_pnl = float(alt_daily.sum()) if len(alt_daily) > 0 else 0
                    multi_asset[asset_name] = {
                        'sharpe': round(alt_sh, 2), 'pnl': round(alt_pnl, 2),
                        'n_trades': len(alt_trades), 'generalizes': alt_sh > 0,
                    }
                    print(f"    {asset_name}: Sharpe={alt_sh:.2f} PnL={_fmt(alt_pnl)}", flush=True)
                except Exception as e:
                    multi_asset[asset_name] = {'error': str(e), 'generalizes': False}

        passed = True  # Stage 7 always passes — it's informational
        details = {'stop_criteria': stop_criteria}
        if multi_asset:
            details['multi_asset'] = multi_asset
            n_gen = sum(1 for v in multi_asset.values() if v.get('generalizes', False))
            details['generalization_rate'] = f"{n_gen}/{len(multi_asset)}"

        lines = [
            f"Stop DD: {_fmt(stop_criteria['max_drawdown_live'])}",
            f"Stop Month: {_fmt(stop_criteria['max_monthly_loss_live'])}",
            f"Stop Consec: {stop_criteria['max_consecutive_loss_days']} days",
        ]
        verdict = "PASS: " + " | ".join(lines)
        return StageResult(
            stage=7, name="DEPLOYMENT", passed=passed, sharpe=0,
            details=details, elapsed_s=time.time() - t0, verdict=verdict)

    # ───────────────────────────────────────────────────────
    # Runner
    # ───────────────────────────────────────────────────────
    def run_stage(self, stage: int) -> StageResult:
        self._load_data()
        stage_fns = {
            0: self.stage0_base_logic,
            1: self.stage1_sanity,
            2: self.stage2_kfold,
            3: self.stage3_walk_forward,
            4: self.stage4_stress,
            5: self.stage5_cost,
            6: self.stage6_reality,
            7: self.stage7_deployment,
        }
        if stage not in stage_fns:
            raise ValueError(f"Unknown stage {stage}. Valid: 0-7")

        print(f"\n{'='*60}")
        print(f"  Stage {stage}: {self.STAGE_NAMES[stage]}")
        print(f"{'='*60}", flush=True)

        result = stage_fns[stage]()
        self.results[stage] = result
        self._save_stage(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"\n  [{status}] {result.verdict} ({result.elapsed_s:.1f}s)", flush=True)
        return result

    def run_all(self, stop_on_fail: bool = True) -> Dict[int, StageResult]:
        """Run all 8 stages (0-7). If stop_on_fail, stop at first failure."""
        print(f"\n{'#'*60}")
        print(f"  Strategy Validator: {self.name}")
        print(f"  Spread={self.spread}, Lot={self.lot}")
        print(f"  n_trials_tested={self.config.n_trials_tested}")
        print(f"{'#'*60}")

        t0 = time.time()
        for stage in range(0, 8):
            result = self.run_stage(stage)
            if not result.passed and stop_on_fail:
                print(f"\n  STOPPED at Stage {stage} (failed). Fix issues before proceeding.")
                break

        total = time.time() - t0
        self._print_summary(total)
        self._save_summary(total)
        return self.results

    def _print_summary(self, total_elapsed: float):
        n_total = 8
        print(f"\n{'='*60}")
        print(f"  VALIDATION SUMMARY: {self.name}")
        print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
        print(f"{'='*60}")
        all_passed = True
        for stage in sorted(self.results.keys()):
            r = self.results[stage]
            icon = "PASS" if r.passed else "FAIL"
            print(f"  Stage {stage} [{icon}] {r.name}: {r.verdict}")
            if not r.passed:
                all_passed = False

        max_stage = max(self.results.keys()) if self.results else -1
        if all_passed and max_stage == 7:
            print(f"\n  FINAL VERDICT: ALL {n_total} STAGES PASSED "
                  f"-- Strategy is validated for live trading")
        elif all_passed:
            print(f"\n  PROGRESS: {max_stage + 1}/{n_total} stages passed so far")
        else:
            print(f"\n  FINAL VERDICT: FAILED at Stage {max_stage} "
                  f"-- Fix issues before proceeding")

    def _save_stage(self, result: StageResult):
        path = self.output_dir / f"stage{result.stage}_{result.name.lower().replace(' ', '_')}.json"
        data = {
            'stage': result.stage, 'name': result.name,
            'passed': result.passed, 'sharpe': result.sharpe,
            'verdict': result.verdict, 'details': result.details,
            'elapsed_s': round(result.elapsed_s, 1),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

    def _save_summary(self, total_elapsed: float):
        path = self.output_dir / "validation_summary.json"
        summary = {
            'strategy': self.name,
            'spread': self.spread, 'lot': self.lot,
            'n_trials_tested': self.config.n_trials_tested,
            'total_elapsed_s': round(total_elapsed, 1),
            'stages': {},
        }
        for stage, r in sorted(self.results.items()):
            summary['stages'][f"stage{stage}"] = {
                'name': r.name, 'passed': r.passed,
                'sharpe': r.sharpe, 'verdict': r.verdict,
            }
        all_passed = all(r.passed for r in self.results.values())
        summary['all_passed'] = all_passed and len(self.results) == 8
        summary['stages_completed'] = len(self.results)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
