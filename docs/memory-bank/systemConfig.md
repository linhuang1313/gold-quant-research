# 系统配置与核心认知 (System Config)

> **读取频率: 每次对话必读**
> 当前生效参数、参数来源、核心策略认知

---

## 当前实盘参数 (2026-04-13 L5.1 部署)

| 参数 | 值 | 来源回测 |
|------|-----|------|
| KC EMA | 25 | EXP13/EXP20 K-Fold 4/6折赢 |
| KC Multiplier | 1.2 | EXP13/EXP20 |
| TrailAct | 由 V3 ATR Regime 控制 | R3-1 AllTight K-Fold 6/6 |
| TrailDist | 由 V3 ATR Regime 控制 | R3-1 AllTight K-Fold 6/6 |
| V3 ATR Regime | **L5 AllTight** (低波T0.40/D0.10, 正常T0.28/D0.06, 高波T0.12/D0.02) | R3-1 K-Fold 6/6, avg delta +1.31, Sharpe 4.19→5.43 |
| Time Decay TP | **OFF** | R2-R1 K-Fold 12/12, Sharpe 4.07→4.19 |
| ADX_TREND_THRESHOLD | 18 | EXP09/ADX阈值测试 确认最优 |
| ATR_SL_MULTIPLIER | **3.5** | **L5.1**: R6-A5 K-Fold 6/6, Sharpe 4.07→4.24, PnL +$4,699 (原4.5) |
| ATR_TP_MULTIPLIER | 8.0 | EXP06: TP5.0仅1.4%触发, TP8.0 Sharpe+0.04 |
| COOLDOWN_MINUTES | 30 | 20变体交叉测试, 所有点差下均优于3h; R6-A6 CD=15min K-Fold 4/6 FAIL 确认保持 |
| INTRADAY_TREND_ENABLED | True | Phase 5 回测 |
| INTRADAY_TREND_THRESHOLD | **0.50** (choppy) | EXP-G K-Fold 6/6 (原0.35) |
| INTRADAY_TREND_KC_ONLY_THRESHOLD | 0.60 (trending) | Phase 5 |
| Keltner max_hold_bars | **5** (=20 M15 bars) | Phase B K-Fold 6/6, Sharpe 2.29→2.62 (原3=12 M15) |
| RSI_ADX_BLOCK_THRESHOLD | 40 | M15 回测 Sharpe +0.21 |
| DAILY_MAX_LOSSES | 5 | 风控 |
| MAX_LOT_SIZE | 0.05 | 绝对安全上限 |
| 亏损递减上限 | 0笔→0.05, 1笔→0.03, 2笔→0.02, 3+→0.01 | 替代旧lot_scale |
| Max Positions | **1** | **L5.1**: R6-A4 K-Fold 6/6 (含$0.50), Sharpe +0.43, MaxDD -$72 (原2) |
| Risk per Trade | $50 (2.5%) | 风控 |

## 实盘策略组合

| 策略 | 时间框架 | 状态 | 核心逻辑 |
|---|---|---|---|
| Keltner通道突破 | H1 | 启用（主力） | 4阶段状态机, EMA100趋势过滤, ADX>18, ATR自适应止损, V3追踪止盈 |
| NY开盘区间突破(ORB) | M15 | 启用 | 纽约开盘前15分钟高低点突破, 2小时窗口 |
| M15 RSI均值回归 | M15 | 启用（实质消亡） | RSI2超卖/超买, Adaptive下仅6笔/11年 |
| 周一跳空回补 | H1 | 启用 | 周一开盘跳空后回补方向交易 |

## 核心认知 (2026-04-16 更新)

1. **交易成本是策略杀手**: 无成本 Sharpe 3.46 加 $0.50 点差后仅 0.35。所有回测必须包含成本
2. **少交易比好参数更重要**: C12 无 Adaptive 15,770 笔 Sharpe -0.53; 加 Adaptive 7,365 笔 Sharpe 1.03。砍掉一半交易反而从亏到赚
3. **避开震荡比抓住趋势更重要**: 利润主要来自大量小额追踪止盈（11,611 次 trail 触发，97.7%WR，中位持仓 1.5h）。震荡时段仍是亏损源，Adaptive 门控砍掉震荡时段交易仍然关键
4. **追踪止盈是核心 alpha**: 5,542 笔 trailing +$41,088 (97.7%WR), 赢家快进快出(中位 1.5h), 输家拖很久(中位 11.5h)
5. **过拟合风险低, 结构风险高**: PBO=0.00, 参数平滑, DSR 通过。真正风险是低波动盘整期持续数月小额亏损(1/3 半年窗口为负)
6. **Keltner+ADX+EMA100 信号集已饱和**: Strategy A/C/D 共 596 种组合全部无法超越基线。改善方向应聚焦于出场机制和仓位管理
7. **策略 alpha 极其鲁棒**: VIX/DXY/日线趋势/K线形态/ATR regime/时段/整数关口 — 都无法显著改善
8. **宏观 Regime 过滤无效（三次确认）**: 策略在所有 6 种 regime 下都盈利，是 regime-agnostic
9. **V3 ATR Regime 是唯一有效的 Regime 类调节**: 调整止盈参数而非过滤信号方向，本质是波动率自适应
10. **回测 preset 说明 (2026-04-09)**:
    - `C12_KWARGS`: 旧版参数（trail 0.8/0.25 base, regime 1.0/0.35→0.8/0.25→0.6/0.20），**不代表当前实盘行为**
    - `LIVE_PARITY_KWARGS`: 新增，精确匹配实盘（trail 0.5/0.15 base, regime 0.7/0.25→0.5/0.15→0.4/0.10, time_decay_tp=True, rsi_adx_filter=40, live_atr_percentile=True）
    - **所有新实验应使用 `LIVE_PARITY_KWARGS`**，`C12_KWARGS` 仅用于历史对比
    - `historical` spread 模型可用: 传入 `spread_model="historical", spread_series=load_spread_series()` 使用真实 Dukascopy 时变 spread
11. **Spread 数据验证 (2026-04-10)**: Dukascopy M15 spread 中位数 $0.33，均值 $0.39。固定 $0.30 是合理下界。引擎新增 `historical` 和 `session_aware` spread 模型
12. **K线形态无法改善KC入场 (R11, 2026-04-16)**: Pinbar/Fractal/InsideBar/Engulfing 作为入场过滤器全部 K-Fold 0/6。KC突破信号已饱和，不存在能改善入场质量的K线形态过滤器
13. **PA共振是伪信号 (R11)**: 要求越多形态共振，Sharpe越低 (Confluence≥2 → 1.38, ≥3 → -0.20)。好交易不需要"多重确认"
14. **出场系统已接近最优 (R12, 2026-04-16)**: 利润回吐止盈(ProfitDD)和自适应MaxHold均无法改善Trailing。Trailing 95.6%WR/中位1bar，是系统核心alpha，不应再调整
15. **快速交易最赚钱 (R12)**: 1-2bar持仓 94.5%WR/+$4.61/trade; 20+bar持仓 7.9%WR/-$14.42/trade。策略的alpha在于快速捕获突破动量
16. **Timeout是最大亏损源 (R12)**: -$24,032总亏损, 中位20bar。改善方向应是让更多Timeout交易更早被Trailing或其他机制处理
17. **时段信号质量差异显著 (R12)**: NY Sharpe=4.53 > London=3.86 > Asia=2.89 > OffHours=0.92。但时段过滤已被否决(所有时段都赚钱，过滤任何时段总Sharpe下降)
18. **尾部风险集中在SL (R12)**: CVaR(1%)的192笔中191笔是SL出场，说明极端亏损来自真正的反向趋势突破

## 因子有效性摘要

### 有效因子 (IC 显著且稳定)
- `RSI2 × ret_1`: IC=-0.0314, WF=100% — M15 RSI 策略的因子基础
- `day_of_week × ret_4/8`: IC=+0.033, WF=100% — 但回测显示跳过任何一天都更差
- `ATR × ret_4/8`: IC=+0.032~0.036, WF=60% — 高波动后正收益（趋势延续）
- `momentum_5/10 × ret_1`: IC=-0.019~0.021, WF=100% — 短期动量反转
- `KC_position/breakout_strength × ret_1`: IC=-0.016, WF=100% — KC 位置短期反转
- `MACD_hist_change × ret_1`: IC=-0.025, WF=100% — MACD 加速度反转

### 无效因子 (|IC| < 0.005)
- `ADX`, `close_ema100_dist`, `ema9_ema21_cross`, `volume_ratio`
- `Pinbar`, `Fractal`, `InsideBar`, `Engulfing` — R11 全样本 IC<0.01, K-Fold 0/6 (2026-04-16)
- `Squeeze (BB inside KC)` — R12 IC 扫描有条件收益但过滤后 K-Fold 0/6 (2026-04-16)
- ADX 作为线性预测因子无效，但作为条件筛选器（ADX>18 门槛）仍有用

### Gradient Boosting 因子重要性
- ATR(22.9%) > EMA100_dist(16.8%) > KC_pos(14.5%) > RSI14(11.5%) > KC_bw(8.8%)

## IntradayTrendMeter

- 4 个子因子: ADX(30%) + KC突破比例(25%) + EMA排列一致性(25%) + 趋势强度(20%)
- 三级门控: ≥0.60 TRENDING (全策略), 0.35-0.60 NEUTRAL (仅H1), <0.35 CHOPPY (禁止开仓)
- 27.2% 的交易日至少出现一次 choppy 窗口
- choppy 门控有价值: 无门控 Sharpe 从 4.46 降到 4.05
- **EXP-G 验证**: choppy_threshold 从 0.35 提升到 0.50，K-Fold 6/6 PASS — **等组合测试后部署**

## L6 候选版本 (2026-04-13 确认, 待部署)

L6 = L5.1 + UltraTight2 regime trail。在 L5.1 基础上进一步收紧所有 regime 的追踪止盈参数。

| 参数 | L5.1 值 | L6 值 | 变化 |
|------|---------|-------|------|
| regime trail low | 0.40/0.10 | **0.30/0.06** | 更紧 |
| regime trail normal | 0.28/0.06 | **0.20/0.04** | 更紧 |
| regime trail high | 0.12/0.02 | **0.08/0.01** | 更紧 |
| trail fallback | 0.28/0.06 | **0.20/0.04** | 同步 normal |

### L6 验证数据 (R7-3 + R6B-B1)
- **全样本 $0.30**: Sharpe=7.15 (vs L5.1=6.17, **+0.98**), PnL=$44,674, MaxDD=$215
- **全样本 $0.50**: Sharpe=4.84 (vs L5.1=4.03, **+0.81**)
- **K-Fold $0.30**: 6/6 PASS, 每 fold 都正 delta (+0.56 ~ +2.13)
- **K-Fold $0.50**: 6/6 PASS, Fold2 从 -0.47 翻正到 +0.01
- **Walk-Forward**: 11/11 年全盈利, 最弱年 2019 Sharpe=4.75
- **逐年对比**: 12/12 年 L6 都优于 L5.1, 总计多赚 $5,648
- **交叉验证**: 两台服务器 (Server A 16核, Server B 8核) 结果一致

### 部署条件
- [ ] L5.1 运行 ≥ 2 周 (最早 2026-04-27) 或 ≥ 30 笔交易
- [ ] Paper trade L6 ≥ 20 笔 → **P10_l6_ultratight 已加入模拟盘 (2026-04-14)**，开始积累
- [ ] 用户明确同意

---

## L5.1 部署记录 (2026-04-13)

L5.1 = L5 + SL收紧 + MaxPos缩减。基于 Round 6 验证，L5 特殊条款允许的唯一一次微调。

| 改进 | L5 值 | L5.1 值 | 修改位置 | 验证 |
|---|---|---|---|---|
| ATR_SL_MULTIPLIER | 4.5 | **3.5** | strategies/signals.py | R6-A5 K-Fold 6/6, Sharpe 4.07→4.24, PnL +$4,699 |
| MAX_POSITIONS | 2 | **1** | config.py | R6-A4 K-Fold 6/6 ($0.30+$0.50), Sharpe +0.43, MaxDD -$72 |
| LIVE_PARITY_KWARGS | L5 | **L5.1** | backtest/runner.py | sl_atr_mult=3.5, max_positions=1 |

### L5.1 预期验证数据 (基于 R6 回测)
- **SL=3.5 全样本**: Sharpe=4.24, PnL=$32,070, MaxDD=$423
- **MaxPos=1 全样本**: Sharpe=4.50, PnL=$26,753, MaxDD=$285
- **两者叠加预期**: Sharpe ~4.5+, MaxDD < $300

### L5.1 后部署纪律
- **L5 特殊条款已用完**，从 L5.1 开始严格执行部署前置条件
- 下一次部署需满足: 运行 ≥ 2 周 / ≥ 30 笔交易, K-Fold 通过, paper trade ≥ 20 笔

## L5 部署记录 (2026-04-12, 已被 L5.1 替代)

L5 = L3 + TDTP OFF + AllTight Trail。基于 Round 2/3/4 完整验证。

| 改进 | L3 值 | L5 值 | 修改位置 | 验证 |
|---|---|---|---|---|
| Time Decay TP | ON | **OFF** | gold_trader.py | R2-R1 K-Fold 12/12, Sharpe +0.12 |
| regime trail high | 0.20/0.03 | **0.12/0.02** | exit_logic.py | R3-1 AllTight K-Fold 6/6, +1.31 |
| regime trail normal | 0.35/0.10 | **0.28/0.06** | exit_logic.py | 同上 |
| regime trail low | 0.50/0.15 | **0.40/0.10** | exit_logic.py | 同上 |
| trail fallback (config) | 0.35/0.10 | **0.28/0.06** | config.py | 同步 normal regime |
| LIVE_PARITY_KWARGS | L3 | **L5** | backtest/runner.py | 回测基准同步 |

### L5 关键验证数据
- **全样本**: Sharpe=5.43, PnL=$34,400, N=21,473, WR=82.4%, MaxDD=$296
- **OOS 11/11 年全盈利**, 样本外 Sharpe > 样本内 (无过拟合)
- **破产概率 0%** (503 起点模拟)
- **盈亏平衡 spread $0.80**, 安全边际 167%
- **46 季度仅 1 季亏损** (2018-Q2 -$97)

## L3 部署记录 (2026-04-11, 已被 L5 替代)

| 改进 | 旧值 | L3 值 | 状态 |
|---|---|---|---|
| keltner max_hold_bars | 3 (12 M15) | 5 (20 M15) | ✅ 保留于 L5 |
| choppy_threshold | 0.35 | 0.50 | ✅ 保留于 L5 |
| regime trail | Tight_all | AllTight (L5 更紧) | 🔄 L5 升级 |
| KC mid reversion exit | — | — | ❌ 永久否决 |

## 舆情系统配置 (v5, 2026-04-15)

| 参数 | 值 | 说明 |
|------|-----|------|
| 基础更新间隔 | **180s** | v4 为 300s |
| 高波动加速间隔 | **60s** | VIX>25 或 |macro_score|>=0.3 时触发 |
| 关键词词典 | 42 bull / 69 bear | v5 重平衡: 描述性词降至 ±0.05 |
| BULLISH/BEARISH 阈值 | ±0.25 | 未变 |
| confidence 触发阈值 | 0.15 | 未变 |
| breaking 关键词 | 11 条规则 | v5: 需金融上下文词共现 |
| 数据源 | 12 RSS feeds | v5: 新增 Reuters/CNBC/MW/GoldBroker |
| 宏观信号 | DXY/VIX/US10Y/Brent | v5 新增，与 NLP 融合 |
| NLP vs 宏观冲突 | 宏观优先 | 一致时 lot_multiplier +0.1 |
| direction_bias | 观察模式 | 仅记录不过滤，待验证 |
| 采集失败告警 | 连续 3 次空 | Telegram 通知 |
| lot_multiplier 范围 | [0.3, 1.5] | v5 新增上下限夹紧 |

## EUR/USD 策略配置

| 参数 | 值 | 与黄金差异 |
|---|---|---|
| KC mult | 2.0 | 黄金 1.2 (EUR/USD 更宽通道最优) |
| MaxHold | 20 bars | 黄金 60 |
| lots | 0.05 | 匹配 $50 风险 |
| point_value | 100,000 | 外汇标准手 |
| 点差 | 1.8 pips | — |
| 11年 Sharpe | 1.91 | 含点差，12/12年全正 |
