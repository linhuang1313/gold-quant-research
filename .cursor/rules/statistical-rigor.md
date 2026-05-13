---
description: Statistical rigor requirements for backtest experiments — Bootstrap CI, fat-tail checks, and autocorrelation guards
globs: experiments/run_*.py
alwaysApply: true
---

# Statistical Rigor Requirements

Based on Central Limit Theorem (CLT) principles and their known failure modes in financial data. Every experiment MUST apply these guards to avoid false discoveries.

## 1. Bootstrap Confidence Intervals (必须)

Point estimates (Sharpe, PnL, Win Rate) are meaningless without uncertainty bounds. Every experiment reporting a Sharpe ratio MUST also report its 95% Bootstrap CI.

### Implementation

```python
def bootstrap_ci(trade_pnls, n_boot=2000, ci=0.95):
    """Bootstrap 95% CI for Sharpe ratio."""
    if len(trade_pnls) < 10:
        return {'sharpe': 0, 'ci_lo': 0, 'ci_hi': 0, 'se': 0}
    
    rng = np.random.default_rng(42)
    sharpes = []
    for _ in range(n_boot):
        sample = rng.choice(trade_pnls, size=len(trade_pnls), replace=True)
        daily = {}
        # group by date if trades have dates, otherwise treat each as independent
        mu, sigma = np.mean(sample), np.std(sample, ddof=1)
        s = (mu / sigma * np.sqrt(252)) if sigma > 0 else 0
        sharpes.append(s)
    
    alpha = (1 - ci) / 2
    lo = np.percentile(sharpes, alpha * 100)
    hi = np.percentile(sharpes, (1 - alpha) * 100)
    return {
        'sharpe_mean': round(np.mean(sharpes), 3),
        'ci_lo': round(lo, 3),
        'ci_hi': round(hi, 3),
        'se': round(np.std(sharpes), 3),
    }
```

### Decision Rule

- **CI does not contain 0**: Strategy has statistically significant edge
- **CI contains 0**: Cannot reject the hypothesis that Sharpe = 0 (may be luck)
- **CI width > 1.5**: Insufficient data or high variance — need more trades

### When Comparing Two Parameter Sets

Use paired Bootstrap: resample the SAME trade indices for both configs, compute the Sharpe difference distribution. If the 95% CI of the difference contains 0, the improvement is NOT statistically significant.

```python
def bootstrap_sharpe_diff(pnls_a, pnls_b, n_boot=2000):
    """Is the Sharpe difference between A and B statistically significant?"""
    rng = np.random.default_rng(42)
    diffs = []
    n = min(len(pnls_a), len(pnls_b))
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sa = _sharpe(pnls_a[idx])
        sb = _sharpe(pnls_b[idx])
        diffs.append(sa - sb)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    significant = (lo > 0) or (hi < 0)  # CI doesn't cross zero
    return {'diff_mean': np.mean(diffs), 'ci_lo': lo, 'ci_hi': hi, 'significant': significant}
```

## 2. Fat-Tail Detection (肥尾检查)

Normal distribution assumes kurtosis = 3. Financial returns routinely show kurtosis > 5, meaning extreme losses are FAR more likely than a bell curve predicts.

### Mandatory Checks

```python
from scipy import stats

def fat_tail_report(trade_pnls):
    """Report tail risk metrics. Call for every strategy variant."""
    pnls = np.array(trade_pnls)
    return {
        'kurtosis': round(stats.kurtosis(pnls, fisher=True), 2),  # excess kurtosis, normal = 0
        'skewness': round(stats.skew(pnls), 2),
        'jarque_bera_p': round(stats.jarque_bera(pnls).pvalue, 4),
        'worst_5pct_avg': round(np.mean(np.sort(pnls)[:max(1, len(pnls)//20)]), 2),
        'best_5pct_avg': round(np.mean(np.sort(pnls)[-max(1, len(pnls)//20):]), 2),
        'tail_ratio': round(abs(np.percentile(pnls, 95) / np.percentile(pnls, 5)), 2) if np.percentile(pnls, 5) != 0 else 999,
    }
```

### Interpretation

| Metric | Healthy | Warning | Danger |
|--------|---------|---------|--------|
| Excess Kurtosis | < 3 | 3–6 | > 6 |
| Skewness | -0.5 to +0.5 | -1 to -0.5 | < -1 (left tail) |
| Jarque-Bera p | > 0.05 | 0.01–0.05 | < 0.01 |
| Tail Ratio | > 1.0 | 0.7–1.0 | < 0.7 |

### Action Items

- **Kurtosis > 6**: Use t-distribution for VaR, not normal. MaxLoss Cap is essential.
- **Skewness < -1**: Strategy has large left-tail risk. Consider tighter stop-loss or smaller lot size.
- **Tail Ratio < 0.7**: Losses are systematically larger than gains. Check R:R ratios.

## 3. Sample Size Guards

CLT requires "enough" independent observations. Financial trades are NOT fully independent (autocorrelation, regime clustering).

### Minimum Sample Sizes

| Analysis | Min Trades | If Below Threshold |
|----------|-----------|-------------------|
| Sharpe point estimate | 30 | Report as "insufficient data", do NOT make GO/NO-GO decisions |
| Bootstrap CI | 50 | Use wider CI (90% instead of 95%) |
| K-Fold (per fold) | 30 | Reduce folds or merge adjacent folds |
| Parameter comparison | 50 per variant | Use Bootstrap diff test, not point comparison |
| Era segment | 20 | Flag as "low-N era", do not use for rejection |

### Effective Sample Size for Autocorrelated Data

When trades cluster in time (multiple trades same day/week), the effective N is lower than the raw count.

```python
def effective_n(trade_pnls, max_lag=10):
    """Estimate effective sample size accounting for autocorrelation."""
    n = len(trade_pnls)
    if n < 20:
        return n
    acf_sum = 0
    pnls = np.array(trade_pnls) - np.mean(trade_pnls)
    var = np.var(pnls)
    if var == 0:
        return n
    for lag in range(1, min(max_lag + 1, n)):
        r = np.sum(pnls[lag:] * pnls[:-lag]) / (var * n)
        if abs(r) < 2 / np.sqrt(n):  # insignificant
            break
        acf_sum += r
    return max(1, int(n / (1 + 2 * acf_sum)))
```

## 4. Output Format Enhancement

Every experiment summary MUST include these additional columns alongside existing metrics:

```
  Variant          N   N_eff  Sharpe  [95% CI]         Kurt   Skew   Verdict
  --------------- --- ------ ------- --------------- ------ ------ ---------
  Live baseline   998    820   8.200  [6.51, 9.89]    2.14  -0.32   GO
  Paper params    450    380   9.100  [5.22, 12.98]   4.87  -0.68   REVIEW
```

### Verdict Rules (updated)

- **GO**: 3-Gate passes AND Sharpe CI lower bound > 0 AND Kurtosis < 6
- **CONDITIONAL GO**: 3-Gate passes but CI lower bound < 0.5 OR Kurtosis 3-6
- **NO-GO**: 3-Gate fails OR CI contains 0 OR Kurtosis > 10
- **INSUFFICIENT**: N_eff < 30 in any fold/era — cannot make a decision

## 5. Comparing Parameter Sets (A vs B)

When comparing two configurations (e.g., TSMOM Paper vs Live), do NOT rely on point Sharpe differences alone:

1. Run Bootstrap paired difference test (`bootstrap_sharpe_diff`)
2. If `significant == False`, the difference is within noise — keep the simpler/existing config
3. If `significant == True` AND the better config also passes 3-Gate, consider switching
4. Always report: `Sharpe_A=X.XX [CI], Sharpe_B=Y.YY [CI], Diff=Z.ZZ [CI], significant={True/False}`

## Rationale

Financial data violates CLT assumptions in 3 ways:
- **Autocorrelation**: Today's return depends on yesterday's (especially intraday)
- **Non-stationarity**: Volatility regimes shift (2020 vs 2024 vs 2025 gold markets)
- **Fat tails**: Extreme moves (black swans) occur far more often than normal distribution predicts

These guards prevent deploying strategies whose apparent edge is actually statistical noise, regime artifact, or tail-risk exposure.
