---
description: MANDATORY — Protocol for any parameter change recommendation. Prevents the SL=6.0/Cap=$70 mistake.
globs: experiments/run_*.py
alwaysApply: true
---

# Parameter Change Protocol

## Why This Rule Exists

On 2026-05-14, R233/R234 revealed that the L8→R202 parameter migration contained a hidden regression:
- SL 3.5→6.0 was **individually negative** (-0.7% Sharpe), masked by the beneficial trail change (+4.8%)
- Cap=$70 was calibrated under SL=3.5 + ATR dynamic lots, but deployed under SL=6.0 + fixed 0.04 lots — completely different behavior
- No ablation analysis was performed, so the harmful changes were never isolated
- Backtest used ATR dynamic lots (~0.01) while live used fixed 0.04 lots — Cap $70 meant $70 price distance in backtest but only $17.5 in live

The "Trail-First" config (keep only the trail change, revert SL and Cap) scored Sharpe 5.675 vs R202's 5.532 — proving that less is more when changes aren't properly validated.

## The Protocol

### Before recommending ANY parameter change to live:

#### Step 1: Environment Parity
- [ ] **Lot size matches live** — use `min_lot_size` / `max_lot_size` to force fixed lots
- [ ] **All engine filters match live** — check `LIVE_PARITY_KWARGS` is current
- [ ] **Spread/slippage model stated** — zero-cost results must be labeled as such

#### Step 2: Ablation Analysis (MANDATORY for multi-parameter changes)
If changing more than one parameter, you MUST test each change in isolation:

```python
base = {**LIVE_PARITY_KWARGS}  # current live params

# Test EACH change alone
results = {}
results['A: baseline'] = run(base)
results['B: +change_1 only'] = run({**base, **change_1})
results['C: +change_2 only'] = run({**base, **change_2})
results['D: +change_1 + change_2'] = run({**base, **change_1, **change_2})
```

Report the individual delta of each change. If a change is negative in isolation, **flag it explicitly** even if the combined result is positive.

#### Step 3: Baseline Comparison
Every recommendation must include a direct comparison against the current live config:

| Config | Sharpe | PnL | MaxDD | KFold |
|--------|--------|-----|-------|-------|
| **Current live** | X.XXX | $XXX | $XXX | X/6 |
| **Proposed change** | X.XXX | $XXX | $XXX | X/6 |
| **Delta** | +X.X% | +$XXX | ... | ... |

Never report only the new config's results. The user must see whether it's better or worse than what's running now.

#### Step 4: Interaction Check
Parameters interact. When changing param A, verify that existing params B and C still make sense:

- SL and Cap: tighter SL may make Cap redundant; wider SL may make Cap too aggressive
- SL and Trail: tighter SL can mask trail differences; wider SL amplifies trail impact
- Lot size and Cap: Cap price distance = Cap$ / (lots × POINT_VALUE) — changes with lot size
- Trail and MaxHold: tighter trail may exit before MaxHold; looser trail makes MaxHold more relevant

#### Step 5: Multi-Gate Deep Validation
Before recommending deployment, the proposed config must pass:

1. **K-Fold 6/6** — at least 5/6 positive Sharpe
2. **Walk-Forward >70%** — majority of OOS windows positive
3. **Era stability** — all major eras positive
4. **Parameter sensitivity Δ < 2.0** — not on a cliff edge
5. **Monte Carlo P(Sharpe>0) > 99%** — statistically robust
6. **Realistic cost test (Sp$0.75) Sharpe > 1.0** — survives transaction costs

### What You Must NOT Do

- **Never recommend a parameter change based only on "it passed K-Fold"** — K-Fold PASS means stable, not optimal. SL=2.0 through SL=8.0 all pass K-Fold.
- **Never test with different lot sizes than live** — this silently changes Cap behavior, position sizing, and PnL scale
- **Never bundle parameter changes without ablation** — the beneficial change masks the harmful one
- **Never assume Cap/SL calibration survives other parameter changes** — recalibrate after every SL, trail, or lot size change
- **Never ignore live trading feedback** — if the user reports unexpected behavior (e.g. "Cap triggers too often"), treat it as a bug report, not a tuning request

### Template for Parameter Change Reports

```
## Parameter Change Proposal: [name]

### Current vs Proposed
| Param | Current | Proposed | Reason |
|-------|---------|----------|--------|
| ...   | ...     | ...      | ...    |

### Ablation (each change isolated)
| Change | Δ Sharpe | Δ PnL | Individual verdict |
|--------|----------|-------|--------------------|
| ...    | +X.X%    | +$XXX | Beneficial/Harmful  |

### Head-to-Head
| Metric | Current | Proposed | Winner |
|--------|---------|----------|--------|
| Sharpe | ...     | ...      | ...    |
| PnL    | ...     | ...      | ...    |
| MaxDD  | ...     | ...      | ...    |

### Validation Gates: X/6 passed
### Recommendation: DEPLOY / HOLD / REJECT
```
