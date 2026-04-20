# 回测引擎与研究配置 (System Config)

> **读取频率: 每次对话必读**
> 回测引擎参数、preset 说明、策略版本定义、核心认知

---

## 回测引擎 Preset

### LIVE_PARITY_KWARGS (标准 preset，所有新实验必须使用)

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

### 当前实盘: L6 (2026-04-18 部署)
L6 = L5.1 + UltraTight2 regime trail

| 参数 | 值 |
|------|-----|
| 止损 | 3.5 x ATR |
| 止盈 | 8.0 x ATR |
| 最大持仓时间 | 5 小时 (20 x M15) |
| 最大同时持仓 | 1 笔 |
| Trailing (低波动 ATR<30%) | 激活 0.30xATR, 距离 0.06xATR |
| Trailing (正常波动) | 激活 0.20xATR, 距离 0.04xATR |
| Trailing (高波动 ATR>70%) | 激活 0.08xATR, 距离 0.01xATR |
| IntradayTrendMeter | Choppy < 0.50 禁止开仓 |
| 时间衰减止盈 (TDTP) | OFF |
| 冷却期 | 30 分钟 |
| RSI ADX 过滤 | 40 |
| live_atr_percentile | True (rolling-50) |

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

### Paper Trade 中: L7
L7 = L6 + TATrail(s2/d0.75/f0.003) + min_entry_gap_hours=1.0

| 参数 | 说明 | 值 |
|------|------|-----|
| time_adaptive_trail | 启用时间自适应追踪止盈 | True |
| time_adaptive_trail_start | 持仓超过N根bar后开始收紧 | 2 |
| time_adaptive_trail_decay | 每bar衰减系数 | 0.75 |
| time_adaptive_trail_floor | 最小trail距离(ATR倍数) | 0.003 |
| min_entry_gap_hours | 两次入场最小间隔 | 1.0 |

**L7 验证数据:**
- 全样本 $0.30: Sharpe 7.18→7.46 (+0.28), PnL $45,075→$46,468
- 全样本 $0.50: Sharpe 4.88→5.18 (+0.30)
- 逐年: 11/11年 L7 均优于 L6，无一例外
- K-Fold $0.30: 6/6 PASS (delta +0.12~+0.48)
- K-Fold $0.50: 6/6 PASS (delta +0.10~+0.60)
- Walk-Forward: 12/12年全盈利 (含2026)

---

## 回测引擎能力

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
5. **Timeout 是最大亏损源**: -$24,032, 中位20bar, 占亏损69.9%
6. **参数极其鲁棒**: MC 80次±15%扰动, 100%盈利, min Sharpe=5.49
7. **V3 ATR Regime 是唯一有效的 Regime 调节**: 调参数不过滤信号
8. **宏观 Regime 过滤无效 (3次确认)**: 策略在所有 regime 下都盈利
9. **$0.30 spread 合理**: Dukascopy 中位数 $0.33, 盈亏平衡 ~$0.60
10. **EMA25/Mult1.2 已近最优**: R13 EMA18仅+0.14, 参数空间平滑无悬崖
11. **TATrail 已确认有效**: R15 全30种组合正delta; R19 L7逐年11/11优于L6, K-Fold 6/6
12. **单策略系统风险可控**: 经19轮实验锤炼, 上百种新策略/因子全部不如Keltner基线; 品种多元化优先于策略堆叠

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
| Server C | `ssh -p 16005 root@connect.westc.seetacloud.com` | R15 TATrail |
| Server D | `ssh -p 35258 root@connect.westd.seetacloud.com` | R13 Alpha |
| 密码 | `r1zlTZQUb+E4` | 两台相同 |
| 项目路径 | `/root/gold-quant-trading` | 服务器上的工作目录 |
