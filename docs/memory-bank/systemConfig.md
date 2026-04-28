# 回测引擎与研究配置 (System Config)

> **读取频率: 每次对话必读**
> 回测引擎参数、preset 说明、策略版本定义、核心认知

---

## 回测引擎 Preset

### LIVE_PARITY_KWARGS (2026-04-29 更新，对齐 L8_BASE+Cap80)

✅ 已更新至 L8_BASE+Cap80 参数: ADX=14, UT3g_micro regime trail (0.22/0.14/0.06), TATrail OFF, MH=20。
旧 L5.1 参数保留为 `L51_PARITY_KWARGS` 供历史对比用。
注意: KCBW5 和 MaxLoss Cap $80 不在 engine kwargs 中，需用 `filter_kcbw5()` 和 `apply_max_loss_cap()` 后处理。
回测 L8_BASE 时请使用 `run_l7_l8_compare.py` 中的 `L8_BASE` dict 定义。

| 参数 | 值 | 来源 |
|------|-----|------|
| KC EMA | 25 | EXP13/EXP20 |
| KC Multiplier | 1.2 | EXP13/EXP20 |
| ATR_SL_MULTIPLIER | 3.5 | R6-A5 K-Fold 6/6 |
| ATR_TP_MULTIPLIER | 8.0 | EXP06 |
| ADX_TREND_THRESHOLD | 18 | EXP09 |
| COOLDOWN_MINUTES | 30 | 20变体交叉测试 |
| keltner_max_hold_m15 | 20 (5小时) | Phase B K-Fold 6/6 |
| INTRADAY_TREND_THRESHOLD | 0.50 (choppy) | EXP-G K-Fold 6/6 |
| INTRADAY_TREND_KC_ONLY | 0.60 (trending) | Phase 5 |
| RSI_ADX_BLOCK_THRESHOLD | 40 | M15回测 |
| Max Positions | 1 | R6-A4 K-Fold 6/6 |
| V3 ATR Regime | AllTight (L5) | R3-1 K-Fold 6/6 |
| Time Decay TP | OFF | R2-R1 K-Fold 12/12 |
| live_atr_percentile | True | rolling-50 对齐实盘 |
| rsi_adx_filter | 40 | 对齐实盘 |

### Spread 模型
- `fixed` (默认): 固定点差，$0.30 为标准测试值
- `historical`: 真实 Dukascopy 时变 spread (中位数 $0.33, 均值 $0.39)
- `session_aware`: 按时段设定不同 spread

### C12_KWARGS (旧版，仅历史对比)
- trail 0.8/0.25 base, 旧 regime config
- **不代表当前实盘行为，不要用于新实验**

---

## 策略版本定义

### 当前实盘: L8_BASE + Cap80 (2026-04-28 部署)
L8_BASE = LIVE_PARITY 基础上修改 ADX/Trail/TATrail/MH

**EA 文件**: `deploy/L8_BASE_EA.mq4` (MagicNumber=20250427)

| 参数 | 值 | 与 L7 差异 |
|------|-----|-----------|
| 信号 | H1 Keltner 通道突破 (EMA25, Mult 1.2) | 不变 |
| ADX 过滤 | >14 | L7 为 18，**L8 更宽松** |
| EMA100 趋势过滤 | 价格必须在 EMA100 同侧 | 不变 |
| Choppy 过滤 | trend_score ≥ 0.50 | 不变 |
| 入场间隔 | 1 小时 | 不变 |
| 止损 | 3.5 × ATR | 不变 |
| 止盈 | 8.0 × ATR | 不变 |
| Trailing (低波动 ATR<25%) | 激活 0.22xATR, 距离 0.04xATR | L7 为 0.30/0.06 |
| Trailing (正常波动) | 激活 0.14xATR, 距离 0.025xATR | L7 为 0.28/0.06，**大幅收紧** |
| Trailing (高波动 ATR>75%) | 激活 0.06xATR, 距离 0.008xATR | L7 为 0.12/0.02，**大幅收紧** |
| TATrail | **OFF** | L7 为 ON (s2/d0.75/f0.003) |
| MaxHold | 5 小时 (20 M15 bars) | L7 为 8 bars (2小时) |
| MaxLoss Cap | **$80** (灾难险) | |
| KCBW | **OFF** | |
| 手数 | 固定 0.03 | |
| 最大同时持仓 | 1 笔 | 不变 |
| live_atr_percentile | True (rolling-50) | 不变 |

**L8_BASE vs L7 核心差异总结:**
1. ADX 门槛降低 (18→14): 更多入场信号
2. Regime Trail 全面收紧: 正常波动 0.28/0.06→0.14/0.025，高波动 0.12/0.02→0.06/0.008
3. TATrail 关闭: 不使用时间衰减追踪止盈
4. MaxHold 恢复 20 bars: 给趋势更多空间 (L7 曾缩短到 8)

**L8_BASE 验证数据 (R41):**
- K-Fold 6/6 PASS (CorrSh Mean=6.27, Min=1.14, spread=$0.50)
- 对比: L7 vs L8_BASE vs L8_HYBRID vs L8c_R39 全面对比

**备注:** EA 源码默认值已更新为 `KCBW_Enabled=false, MaxLoss_USD=80`，与实盘一致。

### 历史版本: L7
L7 = L6 + TATrail(s2/d0.75/f0.003) + min_entry_gap_hours=1.0

| 参数 | 说明 | 值 |
|------|------|-----|
| time_adaptive_trail | 启用时间自适应追踪止盈 | True |
| time_adaptive_trail_start | 持仓超过N根bar后开始收紧 | 2 |
| time_adaptive_trail_decay | 每bar衰减系数 | 0.75 |
| time_adaptive_trail_floor | 最小trail距离(ATR倍数) | 0.003 |
| min_entry_gap_hours | 两次入场最小间隔 | 1.0 |
| keltner_max_hold_m15 | 最大持仓时间 | 8 (2小时) |

**L7 验证数据:**
- 全样本 $0.30: Sharpe 7.18→7.46 (+0.28), PnL $45,075→$46,468
- 全样本 $0.50: Sharpe 4.88→5.18 (+0.30)
- 逐年: 11/11年 L7 均优于 L6，无一例外
- K-Fold $0.30: 6/6 PASS (delta +0.12~+0.48)
- K-Fold $0.50: 6/6 PASS (delta +0.10~+0.60)
- Walk-Forward: 12/12年全盈利 (含2026)

### 历史版本: L6
L6 = L5.1 + UltraTight2 regime trail

| 参数 | 值 |
|------|-----|
| Trailing (低波动) | 0.30/0.06 |
| Trailing (正常波动) | 0.20/0.04 |
| Trailing (高波动) | 0.08/0.01 |
| ADX 过滤 | 18 |
| RSI ADX 过滤 | 40 |

**L6 验证数据:**
- Sharpe: 6.17→7.18 (+1.01), K-Fold 6/6 (双点差), WF 11/11
- 12/12年 L6 都优于 L5.1, 总计多赚 $5,648, MaxDD -25%

### 历史版本: L5.1
L5.1 = L5 + SL 3.5x + MaxPos=1

| 参数 | 值 |
|------|-----|
| regime trail low | 0.40/0.10 |
| regime trail normal | 0.28/0.06 |
| regime trail high | 0.12/0.02 |
| trail fallback | 0.28/0.06 |

### 待部署: D1/H4 Keltner (独立策略)

**D1 Keltner (日线级别, 独立 EA)**

| 参数 | 值 | 来源 |
|------|-----|------|
| EMA period | 20 | R25 网格搜索 |
| ATR period | 14 | 标准 |
| KC Multiplier | 2.0 | R25 网格搜索 |
| ADX threshold | 18 | R25 网格搜索 |
| SL | 3.5 x ATR | 与 L7 一致 |
| TP | 8.0 x ATR | 与 L7 一致 |
| Trail activate | 0.40 x ATR | R25 |
| Trail distance | 0.10 x ATR | R25 |
| MaxHold | 8 bars (8天) | R27 P1-5 最优 |
| Spread | $0.30 | 标准 |

- Sharpe: 6.03 (全样本), K-Fold 6/6 (mean=9.16)
- WR: 90.0%, N=289, PnL=$13,642, MaxDD=$497
- 与 L7 相关性: 0.17
- 参数 cliff test: EMA15-25/Mult1.5-2.5 全区域 Sharpe 3.9-6.6, 无 cliff

**H4 Keltner (4小时级别, 独立 EA)**

| 参数 | 值 | 来源 |
|------|-----|------|
| EMA period | 20 | R25 网格搜索 |
| ATR period | 14 | 标准 |
| KC Multiplier | 2.0 | R25 网格搜索 |
| ADX threshold | 18 | R25 网格搜索 |
| SL | 3.5 x ATR | 与 L7 一致 |
| TP | 8.0 x ATR | 与 L7 一致 |
| Trail activate | 0.28 x ATR | R25 |
| Trail distance | 0.06 x ATR | R25 |
| MaxHold | 20 bars (80小时) | R27 P1-5 最优 |
| Spread | $0.30 | 标准 |

- Sharpe: 4.64 (全样本), K-Fold 6/6 (mean=6.27)
- WR: 84.1%, N=1995, PnL=$24,712, MaxDD=$927
- 与 L7 相关性: 0.22
- 参数 cliff test: EMA15-25/Mult1.5-2.5 全区域 Sharpe 3.7-5.1, 无 cliff

---

## 回测引擎能力

### 性能优化 (2026-04-28)

**skip_non_h1_bars=True (默认开启)**
- 跳过"无持仓 + 无挂起信号 + 非H1边界"的 M15 bar
- 加速 1.6x (342s → 213s，267K bars 全样本)
- 跳过了 ~46% 的 bars (M15 RSI 信号不再在非 H1 边界 bar 触发)
- 设 `skip_non_h1_bars=False` 回退完整行为

**两层快筛架构 (backtest/fast_screen.py)**
- `fast_backtest_signals()`: 纯 NumPy H1 单时间框架回测，~2-3s/次
- `screen_grid()`: 批量扫描参数网格，支持淘汰器模式 (`min_sharpe=0`)
- `screen_then_validate()`: 自动化 "快筛淘汰 → 完整验证 → K-Fold" 全流程
- 快筛定位: **淘汰器**（排除 Sharpe<0），不做最终排名选择

**实验 SOP 流程**
1. Phase 1: `screen_grid(min_sharpe=0)` 快速淘汰明显差的组合 (~1000 组合/小时)
2. Phase 2: 全部存活候选用完整引擎验证 + K-Fold (只对 Sharpe>1 做 K-Fold)
3. Phase 3: Top 候选做压力确认 (Spread/Crisis/MonteCarlo)
- 模板脚本: `experiments/_template_experiment.py`

### 核心引擎参数 (engine.py / runner.py)
- KC 参数: `kc_ema_override`, `kc_mult_override`
- 出场: `sl_atr_mult`, `tp_atr_mult`, `keltner_max_hold_m15`, `time_decay_tp`
- Trail: `trail_activate_atr`, `trail_distance_atr`, `regime_config`
- TATrail: `time_adaptive_trail_start/decay/floor` (R14/R15新增)
- 门控: `intraday_adaptive`, `choppy_threshold`, `rsi_adx_filter`
- 双KC: `dual_kc_mode`, `dual_kc_fast`, `dual_kc_slow` (R13新增, 已否决)
- MA类型: `kc_ma_type` (ema/hma/kama) (R13新增, 已否决)
- Spread: `spread_model`, `spread_series`
- 验证: `purge_embargo_bars` (R13新增)
- 性能: `skip_non_h1_bars` (默认 True, 1.6x 加速)

### 指标 (indicators.py)
- Keltner Channel (EMA/HMA/KAMA + ATR)
- Bollinger Bands (Squeeze 检测用)
- Swing High/Low 支撑阻力位
- Pinbar/Fractal/InsideBar/Engulfing 形态检测
- IntradayTrendMeter (ADX+KC突破+EMA排列+趋势强度)
- Dual KC (快慢双通道)

### 统计 (stats.py)
- Sharpe (ddof=1), PnL, WR, MaxDD, $/trade
- K-Fold 验证, Walk-Forward, Monte Carlo
- 逐年/逐季度分解

---

## 核心研究认知

1. **追踪止盈是核心 alpha**: 95.6%WR, 中位1bar, 贡献$75,357
2. **KC+ADX+EMA100 入场信号已饱和**: 596种变体全部不如基线, PA形态(R11 24实验)全0/6
3. **出场系统已接近最优**: ProfitDD/AdaptHold/KC mid revert 全部否决
4. **快速交易最赚钱**: 1-2bar WR=94.5%(+$4.61/t), 20+bar WR=7.9%(-$14.42/t)
5. **Timeout 是最大亏损源**: -$24,032, 中位20bar → **MH=8 大幅改善** (R25/R27)
6. **参数极其鲁棒**: MC 80次±15%扰动, 100%盈利, min Sharpe=5.49
7. **V3 ATR Regime 是唯一有效的 Regime 调节**: 调参数不过滤信号
8. **宏观 Regime 过滤无效 (3次确认)**: 策略在所有 regime 下都盈利
9. **$0.30 spread 合理**: Dukascopy 中位数 $0.33, 盈亏平衡 ~$0.60
10. **EMA25/Mult1.2 已近最优**: R13 EMA18仅+0.14, 参数空间平滑无悬崖
11. **TATrail 已确认有效**: R15 全30种组合正delta; R19 L7逐年11/11优于L6, K-Fold 6/6
12. **D1/H4 Keltner 是独立 alpha 源** (R25-R27): KC 框架在多时间尺度有效, 与 L7 相关性仅 0.17-0.22
13. **EqCurve LB=30 风控层有效** (R27): K-Fold 6/6 改善, 平均 +0.25 Sharpe, 非过拟合
14. **D1/H4 参数极其鲁棒** (R27): 125 组合 cliff test 无悬崖, Sharpe 3.7-6.6 全区域平滑
15. **多策略组合价值明确** (R27): L7+H4(0.5x) Sharpe 8.99, PnL +20%, 独立运行无需互斥

## 因子有效性摘要

### 有效因子
- `RSI2 × ret_1`: IC=-0.031, WF=100%
- `ATR × ret_4/8`: IC=+0.032~0.036, WF=60%
- `momentum_5/10 × ret_1`: IC=-0.019~0.021, 短期反转
- `KC_position/breakout_strength × ret_1`: IC=-0.016
- `kc_bw × ret_8`: IC=+0.018, 宽带后正收益

### 无效因子
- ADX, volume_ratio, close_ema100_dist — |IC|<0.005
- Pinbar/Fractal/InsideBar/Engulfing — IC<0.01, K-Fold 0/6
- Squeeze (BB inside KC) — 过滤后 K-Fold 0/6

### Gradient Boosting 重要性
ATR(22.9%) > EMA100_dist(16.8%) > KC_pos(14.5%) > RSI14(11.5%) > KC_bw(8.8%)

---

## 服务器信息

| 服务器 | 连接 | 用途 |
|--------|------|------|
| **Server Westd (当前)** | `ssh -p 41109 root@connect.westd.seetacloud.com` | **R47+ (当前活跃)** |
| 密码 (Westd) | `3sCdENtzYfse` | |
| Server C | `ssh -p 16005 root@connect.westc.seetacloud.com` | R15 TATrail (可能已过期) |
| Server BJB1 | `ssh -p 45411 root@connect.bjb1.seetacloud.com` | R25-R28 (可能已过期) |
| 项目路径 | `/root/gold-quant-research` | 服务器上的工作目录 |

**已部署到服务器的优化引擎** (2026-04-28):
- `backtest/engine.py` (skip_non_h1_bars)
- `backtest/fast_screen.py` (两层快筛)
- `backtest/runner.py` (screen_then_validate)
- 同步脚本: `deploy/_sync_engine_to_server.py`
