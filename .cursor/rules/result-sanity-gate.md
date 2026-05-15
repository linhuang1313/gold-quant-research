---
description: MANDATORY — Sanity gate before reporting any experiment results to user
globs: experiments/run_*.py
alwaysApply: true
---

# Result Sanity Gate

## The Rule

Before presenting ANY experiment result or recommendation to the user, you MUST pass these sanity checks. If any check fails, investigate and fix BEFORE reporting.

## Gate 1: Numbers Must Be Plausible

| Metric | Plausible Range (H1 gold) | Investigate If |
|--------|--------------------------|----------------|
| Sharpe (annualized) | 0.5 – 5.0 | > 5.0 or < -2.0 |
| Win Rate | 40% – 75% | > 85% or < 25% |
| Trades per year (single strategy) | 50 – 2000 | > 5000 or < 10 |
| Avg PnL per trade | -$50 to +$50 | > $100 (on 0.03 lots) |
| MaxDD (10-year) | $100 – $5000 | < $50 (unrealistically low) |

These ranges are for a realistic H1 gold strategy with 0.03–0.15 lot sizes. If numbers fall outside, the backtest methodology is likely wrong.

## Gate 2: Backtest Must Match Live Reality

If live trading data exists for the same period:
- **Trade count**: Backtest and live should be within 2x of each other. A 10x discrepancy = broken backtest.
- **PnL direction**: If live is negative and backtest is strongly positive (or vice versa), explain why before reporting.
- **Filter effects**: If backtest shows no filter rejections but live logs show frequent filter blocks, the backtest is missing filters.

## Gate 3: Methodology Must Be Stated

Every result report must explicitly state:
1. **Which engine was used** (BacktestEngine, standalone script, or other)
2. **Which filters are active** (Choppy Gate, ATR Pctl, Rule B, ML filter, etc.)
3. **Whether it matches live configuration** (yes/no, with specific differences if no)

## Gate 4: Recommendations Must Be Conservative

- **Never recommend deploying** based on a single experiment
- **Never recommend parameter changes** without 3-Gate validation (K-Fold, Walk-Forward, Era)
- **Never say "passes validation"** if the validation was run on a simplified backtest
- **If uncertain whether results are reliable, say so explicitly** — "these results may not be reliable because X" is far better than a wrong recommendation

## What This Prevents

- Reporting Sharpe 16+ as if it's meaningful
- Recommending "TUNE" for strategies that are losing money live
- Giving confident conclusions from broken methodology
- The user making trading decisions based on wrong data
