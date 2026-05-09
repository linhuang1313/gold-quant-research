---
description: Required 4-era time segmentation for all backtest experiments
globs: experiments/run_*.py
alwaysApply: true
---

# Era-Segmented Testing Requirements

Every backtest experiment MUST report results across 4 time dimensions. A single full-sample result is NOT sufficient.

## Mandatory Time Segments

### 1. Full Sample (全量)
- Use ALL available H1 data (typically 2015-01-01 to present)
- This is the baseline

### 2. Rate Hike Cycle (加息周期)
Periods when the Fed was actively raising rates. Gold typically faces headwinds.
```python
RATE_HIKE_PERIODS = [
    ("Hike1_Normalization", "2015-12-01", "2019-01-01"),  # 2015.12-2018.12: 0.25% → 2.50%
    ("Hike2_PostCOVID",     "2022-03-01", "2023-08-01"),  # 2022.03-2023.07: 0.25% → 5.50%
]
```

### 3. Rate Cut / Easing Cycle (降息周期)
Periods when the Fed was cutting rates or holding at zero. Gold typically benefits.
```python
RATE_CUT_PERIODS = [
    ("Cut1_Insurance",  "2019-07-01", "2020-03-15"),  # 2019.07-2020.03: 2.50% → 0.25% (3 cuts + emergency)
    ("Cut2_ZeroFloor",  "2020-03-15", "2022-03-01"),  # 2020.03-2022.03: held at 0-0.25%
    ("Cut3_Current",    "2024-09-01", "2026-06-01"),  # 2024.09-present: 5.50% → 3.50%
]
```

### 4. Recent 3 Years (近三年)
```python
RECENT_3Y_PERIOD = ("Recent_3Y", "2023-06-01", "2026-06-01")
```

## Implementation

Use this helper constant block in every experiment:

```python
ERA_SEGMENTS = {
    'full':       None,  # Use all data
    'hike':       [("2015-12-01", "2019-01-01"), ("2022-03-01", "2023-08-01")],
    'cut':        [("2019-07-01", "2022-03-01"), ("2024-09-01", "2026-06-01")],
    'recent_3y':  [("2023-06-01", "2026-06-01")],
}

def filter_trades_by_era(trades, era_name):
    """Filter trades to only those with entry_time in the given era."""
    if era_name == 'full' or ERA_SEGMENTS[era_name] is None:
        return trades
    periods = ERA_SEGMENTS[era_name]
    filtered = []
    for t in trades:
        entry = pd.Timestamp(t['entry_time'])
        for start, end in periods:
            if pd.Timestamp(start) <= entry < pd.Timestamp(end):
                filtered.append(t)
                break
    return filtered
```

## Output Format

Every experiment's summary table MUST include:

```
  Era           N    Sharpe      PnL     WR%   MaxDD
  ---------- ---- -------- -------- ------- -------
  Full       XXXX    X.XXX  $X,XXX   XX.X%  $X,XXX
  Hike       XXXX    X.XXX  $X,XXX   XX.X%  $X,XXX
  Cut        XXXX    X.XXX  $X,XXX   XX.X%  $X,XXX
  Recent_3Y  XXXX    X.XXX  $X,XXX   XX.X%  $X,XXX
```

## Interpretation Guide

- **Hike era Sharpe significantly lower** → Strategy is rate-sensitive, needs regime filter
- **Cut era Sharpe much higher than Full** → Strategy may be biased toward easing regimes
- **Recent 3Y diverges from Full** → Check for regime shift or parameter decay
- **All 4 eras positive Sharpe** → Strong evidence of structural edge (not regime-dependent)

## Rationale

Gold price dynamics are heavily influenced by monetary policy:
- **Hike cycles**: Rising real yields → gold headwind → trend signals may underperform
- **Cut cycles**: Falling real yields → gold tailwind → trend signals typically shine
- **Recent**: Captures current market structure, high ATR regime (gold above $2000)

Testing across all 4 eras prevents deploying a strategy that only works in one regime.
