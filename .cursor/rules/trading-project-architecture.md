---
description: Architecture of the live trading system (gold-quant-trading project)
globs: "**/*"
alwaysApply: true
---

# Live Trading System Architecture

## Project Location
`C:\Users\hlin2\gold-quant-trading`

## Architecture: Python-Driven with MT4 Execution Bridge

The live trading system is **entirely Python-driven**. MT4 only acts as an order executor.

```
Python (gold_trader.py) — 主进程，每30秒扫描
  ├── strategies/signals.py — 所有策略信号计算（Keltner, TSMOM, PSAR, SESS_BO）
  ├── strategies/exit_logic.py — 出场逻辑（trailing stop, maxloss cap）
  ├── mt4_bridge.py — 通过文件桥接与MT4通信
  ├── config.py — 所有参数集中管理
  ├── risk_manager.py — 风控
  ├── data_provider.py — 数据获取
  └── position_tracker.py — 持仓跟踪

GoldBridge_EA.mq4 — MT4端，纯"下单执行器"
  - 每500ms检查 DWX/commands.json
  - 执行 OPEN/CLOSE/MODIFY 指令
  - 写回 heartbeat/positions/account/bars JSON
  - 不做任何策略逻辑
```

## Communication: File Bridge (DWX)

- 路径: `C:\Users\hlin2\AppData\Roaming\MetaQuotes\Terminal\35EEC3EFDB656AF6FC775F21FEAD053B\MQL4\Files\DWX\`
- Python写 `commands.json` → EA读取执行 → EA写 `response.json`
- EA每30秒写 `bars_h1.json` 和 `bars_m15.json`（200根K线）
- EA每5秒写 `heartbeat.json`, `account.json`, `positions.json`

## Live Portfolio (6-slot, synced 2026-05-09)

| Strategy | Lots | MaxLoss Cap | SL/TP (ATR) | Trail Act/Dist | Max Hold | Notes |
|----------|------|-------------|-------------|----------------|----------|-------|
| Keltner (L8_MAX) | 0.02 | $35 | 3.5 / 8.0 | 0.14 / 0.025 | 2 bars | ML exit filter (0.65), R178 session ADX |
| PSAR | 0.09 | $60 | 4.0 / 6.0 | 0.08 / 0.015 | 15 bars | Skip hours {3,7,22} UTC |
| TSMOM | 0.15 | $60 | 6.0 / 8.0 | 0.14 / 0.025 | 12 bars | Score 480/720, Friday flatten guard |
| SESS_BO | 0.13 | $60 | 4.5 / 4.0 | 0.14 / 0.025 | 20 bars | GMT12 breakout, D1 EMA20 filter ON |
| Dual Thrust | 0.04 | $18 | 4.5 / 8.0 | 0.14 / 0.025 | 20 bars | k=0.5, n_bars=6 |
| Chandelier | 0.08 | $25 | 4.5 / 8.0 | 0.14 / 0.025 | 20 bars | period=22, mult=3.0, RSI filter ON |

Capital: $5,000 | MAX_POSITIONS: **4** (6策略竞争4槽位) | MAX_TOTAL_LOSS: $3,750
Max simultaneous exposure: 0.51 lots (0.15+0.09+0.13+0.08+0.04+0.02), R162 halves in high vol

### Risk Controls
- MAX_LOT_CAP_BY_LOSSES: {0 losses: 0.13, 1: 0.08, 2: 0.05, 3: 0.03}
- Rule B: ATR > 3σ (60-bar) → skip 8 hours
- R162: ATR > 2× rolling mean → halve lot size
- COOLDOWN_MINUTES: 30 (per strategy after losing close)
- DAILY_MAX_LOSS / DAILY_MAX_LOSSES: $9999 / 9999 (effectively disabled)

### ML Filters (live)
- **Keltner**: ml_filter.py, 12 features, threshold=0.65, model=l8_ml_exit_model.json
- **Non-Keltner**: ml_filter_r173.py, 4 features (ATR/ADX/RSI14/squeeze), threshold=0.70, model=r173_ml_filter.json, **LIVE** (not shadow)

### Session ADX (R178, deployed)
- Asia (0-7 UTC): ADX=14
- London (8-12 UTC): ADX=10
- NY (13-17 UTC): ADX=16
- Evening (18-23 UTC): ADX=14

## Key Implementation Details

### Signal Generation (strategies/signals.py)
- All indicators computed in Python using pandas (ATR, ADX, RSI, KC, EMA, PSAR, etc.)
- `prepare_indicators(df)` computes all technical indicators on H1 DataFrame
- `check_all_signals(df_h1, df_m15)` returns list of signal dicts
- Each signal: `{'strategy': 'keltner', 'signal': 'BUY'/'SELL', 'sl': ..., 'tp': ..., 'reason': ...}`

### Order Execution (mt4_bridge.py)
- `MT4Bridge.send_order(action, type, lots, sl, tp, ...)` writes commands.json
- Atomic writes via temp file + os.replace to prevent EA reading partial JSON

### Data Source
- EA writes H1/M15 bar data to `bars_h1.json`/`bars_m15.json` every 30 seconds
- Python reads these files via `data_provider.py`

### TSMOM Signal (IMPORTANT - R53/R56 Score method, NOT SMA crossover)
```python
def _tsmom_score(closes, idx, fast_lb, slow_lb):
    s = 0.0
    if idx >= fast_lb and closes[idx - fast_lb] > 0:
        ret = closes[idx] / closes[idx - fast_lb] - 1.0
        s += 0.5 * (1.0 if ret > 0 else (-1.0 if ret < 0 else 0.0))
    if idx >= slow_lb and closes[idx - slow_lb] > 0:
        ret = closes[idx] / closes[idx - slow_lb] - 1.0
        s += 0.5 * (1.0 if ret > 0 else (-1.0 if ret < 0 else 0.0))
    return s
```

## ML Exit Filter Integration Point

Since ALL strategy logic is in Python, ML Exit filter should be implemented as:
1. A Python module `strategies/ml_filter.py` with trained XGBoost model
2. Called AFTER signal generation, BEFORE sending to bridge
3. No MQL4 changes needed, no HTTP service needed, no separate process needed

```python
# In the signal processing flow:
sig = check_keltner_signal(df)
if sig and ml_filter.allows(sig, df):
    signals.append(sig)
```
