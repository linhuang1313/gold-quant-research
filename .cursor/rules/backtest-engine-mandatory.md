---
description: MANDATORY — All backtests must use BacktestEngine. No standalone backtest loops.
globs: experiments/run_*.py
alwaysApply: true
---

# Backtest Engine Mandatory Rule

## The Rule

**Every experiment that backtests a trading strategy MUST use `backtest.engine.BacktestEngine` or `backtest.runner`.**

Writing a standalone backtest loop (custom `for i in range(...)` with entry/exit logic) is **FORBIDDEN**.

## Why This Rule Exists

On 2026-05-13, R209 (Non-Keltner Strategy Audit) produced completely wrong results because it used a 130-line standalone backtest instead of the 2000+ line BacktestEngine. The standalone version:
- Had no Choppy Gate filter
- Had no ATR Percentile Floor
- Had no Rule B sigma filter
- Had no R173 ML filter
- Had no ADX gray zone
- Had no multi-strategy slot contention
- Had no realistic slippage model
- Produced Sharpe 16+ and WR 95%+ (impossible in real trading)
- Led to "TUNE" recommendations for strategies that were losing money in live trading

## What You Must Do

### For strategy-level backtests:
```python
from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine

data = DataBundle.load_default()
engine = BacktestEngine(data.m15_df, data.h1_df, **LIVE_PARITY_KWARGS)
trades = engine.run()
```

### If you need to test a single strategy in isolation:
Modify `LIVE_PARITY_KWARGS` to disable other strategies, do NOT rewrite the backtest loop. Example:
```python
kwargs = {**LIVE_PARITY_KWARGS}
kwargs['enabled_strategies'] = ['psar']  # only test PSAR
engine = BacktestEngine(data.m15_df, data.h1_df, **kwargs)
```

If `enabled_strategies` doesn't exist yet, add it to BacktestEngine — do NOT bypass the engine.

### If you need to sweep parameters:
Override specific kwargs per run, still using BacktestEngine:
```python
for sl in [3.0, 4.5, 6.0]:
    kwargs = {**LIVE_PARITY_KWARGS, 'psar_sl_atr': sl}
    engine = BacktestEngine(data.m15_df, data.h1_df, **kwargs)
    trades = engine.run()
```

### If BacktestEngine doesn't support what you need:
**Extend BacktestEngine**, do not write a parallel implementation. Document the extension.

## Sanity Checks Before Reporting Results

Before reporting ANY backtest result, verify:

1. **Trade count plausibility**: Compare against live trading rate. If backtest shows 5000+ trades/year for a strategy that does 3 trades/month live, something is wrong.
2. **Sharpe plausibility**: Sharpe > 5.0 for any single strategy on H1 data is suspicious. Sharpe > 10.0 is almost certainly a bug. Investigate before reporting.
3. **Win rate plausibility**: WR > 90% on H1 data with ATR-based SL/TP is suspicious. Check if filters are actually applied.
4. **Cross-reference with live**: If live data exists for the same period, the backtest should produce similar trade counts and PnL direction. A 10x difference in trade count means the backtest is wrong.

## What Happens If You Break This Rule

The experiment results are invalid and must be discarded. Re-run using BacktestEngine before drawing any conclusions.
