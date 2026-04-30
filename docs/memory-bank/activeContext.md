# 当前上下文 (Active Context)

> **读取频率: 每次对话必读**
> 研究焦点、进行中的实验、回测引擎状态、核心结论

---

## 项目定位

- **本仓库 `gold-quant-research`**: 回测引擎、实验脚本、部署工具、结果归档
- 实盘系统在独立仓库 `gold-quant-trading`，两者完全解耦
- 研究仓库使用 `research_config.py` + `indicators.py`（替代实盘的 config.py / signals.py）

## 回测引擎状态

### 引擎修复 (2026-04-09)
- **H1 look-ahead bias 已修复**: 入场信号只使用已收盘的 H1 bar (`closed_only=True`)
- **入场价 look-ahead 已修复**: 信号在下一根 M15 bar 的 Open 执行
- **⚠️ 重要**: 修复前的所有 Sharpe 数字不可信，R2 以后的实验都基于修复后引擎

### 回测 preset
- **`LIVE_PARITY_KWARGS`**: 当前标准 preset，精确对齐实盘参数。**所有新实验必须使用此 preset**
- `C12_KWARGS`: 旧版，仅用于历史对比
- 引擎支持 `historical` / `session_aware` / `fixed` 三种 spread 模型

### 数据 (2026-04-10)
- BID: M15 266,882 bars, H1 98,808 bars (2015-01 ~ 2026-04-09)
- ASK: M15 395,232 bars, H1 98,808 bars
- Spread 时序: Dukascopy 中位数 $0.33，**$0.30 固定 spread 是合理下界**

---

## 进行中 / 已完成实验 (2026-04-30 更新)

### R25-R28 D1/H4 Keltner + 综合验证 — 已完成

- **R25 Phase A: L7 MaxHold 优化** — MH=8 Sharpe 10.14 (vs MH=20 的 9.41), K-Fold 5/5 通过
  - R26 完整数据确认: L7(MH=8) Sharpe=9.61
- **R25 Phase B: Dynamic Sizing** — EqCurve LB=30 Sharpe +0.41 (唯一有效); Kelly/Streak/Regime sizing 均无改善
- **R25 Phase C + R26: D1/H4 Keltner 发现** — 全新时间尺度 alpha
  - D1 Keltner (EMA20/M2.0/ADX18): Sharpe=5.83, K-Fold 6/6, WR=92.4%
  - H4 Keltner (EMA20/M2.0/ADX18): Sharpe=4.55, K-Fold 6/6, WR=85.1%
  - 与 L7 相关性极低: D1-L7 r=0.17, H4-L7 r=0.22 → 独立 alpha 源
- **R27 完整验证套件 (6 项全部完成)**:
  - P0-1 引擎平价检查: D1 在 spread=1.00 仍 Sharpe 5.03, H4 仍 3.19, 对成本高度鲁棒
  - P0-2 持仓冲突: L7 与 H4 重叠时 88% 同方向, 三策略应独立运行
  - P1-3 EqCurve K-Fold: **6/6 改善**, 平均 +0.25 Sharpe, 确认非过拟合
  - P1-4 参数 cliff test: D1/H4 参数极其稳定, EMA15-25/Mult1.5-2.5 全区域 Sharpe 3.7-6.4, 无 cliff
  - P1-5 MaxHold sweep: D1 最优 MH=8 (Sharpe 6.03, K-Fold 6/6 mean=9.16); H4 最优 MH=20 (Sharpe 4.64, K-Fold 6/6 mean=6.27)
  - P2-6 Lot 优化: L7(1.0x)+H4(0.5x) 最佳平衡 (Sharpe 8.99, PnL $72,664, MaxDD $311)
- **R28 L7(MH=8) 完整数据 K-Fold** — 运行中, 验证 L7 核心策略在 MH=8 下的 6-Fold 鲁棒性

### R21 四个新赚钱维度 — 已完成

- **S1 Squeeze Straddle: 通过 K-Fold 6/6** — Sharpe=1.67, PnL=$1163, WR=82%, 12/12年全正
  - 最优: SqzB3_T0.2/0.04_MH20 (Squeeze 3bar后释放双开, ATR trailing)
  - K-Fold: Fold1~6 全正, Sharpe 1.60~3.16, 极其稳健
- **S2 Event-Driven (NFP/FOMC): 有信号但不独立建模** — NFP Straddle WR=82%, Avg $5/trade, 12/12年正
  - FOMC 效果弱 (WR=35.6%), Continuation 策略无优势
- **S3 Overnight Hold: 通过 K-Fold 6/6** — Sharpe=0.88, PnL=$3148, 10/12年正
  - NYclose→LDNopen 做多, 学术 overnight anomaly 在黄金上验证有效
  - OffHours (21-0 UTC) CumRet=140%, Sharpe=2.47 (session level)
  - K-Fold: 6/6 全正, Sharpe 0.05~1.72 (Fold2/Fold4 较弱但仍正)
- **S4 Extreme Reversal: 全部否决** — 所有 81 种参数组合 Sharpe 全负, 最好 -0.43
  - 逆向做反转在黄金上不可行: 趋势持续性太强, 极端偏离后不均值回归

### R15 TATrail — 已完成，结论纳入 L7

- TATrail 所有 30 种参数组合都优于 Baseline，无负面配置
- 最优 Start=2, Decay=0.75, Floor=0.003
- L5.1 Sharpe +0.20, L6 +0.10, 高点差下提升更大

### R9-2 L7 全面验证 — 已完成

- L7 vs L6 K-Fold 6/6 ($0.30), 6/6 ($0.50)
- Walk-Forward 12/12 年全盈利 (含2026)
- L7-L6 delta: +0.12 ~ +0.48 Sharpe (K-Fold $0.30)

### R19-E1 L7 逐年确认 — 已完成

- 全样本 $0.30: L6 Sharpe 7.18 → L7 **7.46** (+0.28)
- 全样本 $0.50: L6 4.88 → L7 **5.18** (+0.30)
- **11/11 年 L7 均优于 L6，无一例外**

### R13 Alpha 精炼 — 已完成，全部否决

- EMA18 仅 +0.14 vs EMA25，不值得改
- 保本止损/双KC/HMA/KAMA 全部否决

---

## 已完成实验一览

| Round | 方向 | 核心结论 | 详见 |
|-------|------|---------|------|
| R2-R4 | L5组合验证/OOS/存活模拟 | L3 Sharpe=4.07, OOS 11/11年盈利, 破产率0% | backtestArchive |
| R5 | Monte Carlo | 100/100 Sharpe>4, 极其鲁棒 | backtestArchive |
| R6A | 历史点差/MaxPos/SL/Cooldown | L5.1 部署依据 (SL=3.5, MaxPos=1) | backtestArchive |
| R6B | L6 逐年对比 | L6 12/12年优于L5.1, +$5,648 | backtestArchive |
| R7 | L5.1基准/EntryGap/L6增量/MC | L6 Sharpe=7.15(+0.98), K-Fold 6/6, WF 11/11 | backtestArchive |
| R8 | TP×SL/L6+Gap | L6全面优于L5.1, Gap=1h K-Fold 6/6 | backtestArchive |
| R9 | L7 K-Fold + Walk-Forward | L7 K-Fold 6/6双点差, WF 12/12年, L7-L6 +0.12~+0.48 | backtestArchive |
| R11 | PA形态全面验证 | **全部否决** — 24实验, K-Fold全0/6 | backtestArchive |
| R12 | 深水区探索 | Squeeze/连续突破/出场前沿全否决, 行为画像有价值 | backtestArchive |
| R13 | Alpha精炼(EMA/Mult/BE/双KC/MA) | **全部否决** — EMA25/Mult1.2已近最优 | backtestArchive |
| R15 | TATrail深化验证 | 30种参数全正delta, 最优s2/d0.75 +0.20 | backtestArchive |
| R19 | L7综合确认 | L7 Sharpe 7.46(+0.28), 11/11年优于L6 | results/round19 |
| R21 | 四个新赚钱维度 | **S1 Squeeze K-Fold 6/6, S3 Overnight K-Fold 6/6**; S2部分有效, S4全否决 | results/round21 |
| R22 | EUR/USD 品种扩展 | Sharpe 1.6~2.3 (仅1.3年数据), MC 100%正, **需Dukascopy长期数据** | results/round22_23 |
| R23 | L7+S1+S3 组合 | 相关性≈0, 组合Sharpe=7.23, 12/12年全正, L7贡献93%利润 | results/round22_23 |
| R25 | MaxHold优化+DynamicSizing+D1/H4网格 | **MH=8 Sharpe 10.14**; EqCurve +0.41; D1/H4 KC发现 | results/round25 |
| R26 | D1/H4 K-Fold+相关性+组合分析 | D1 6/6 Sharpe 5.83, H4 6/6 Sharpe 4.55, 与L7低相关 | results/round26 |
| R27 | 完整验证套件(6项) | Spread鲁棒/冲突/EqCurve 6/6/Cliff无/MH优化/Lot优化 | results/round27 |
| R28 | L7(MH=8) 完整K-Fold | 6/6 通过, Sharpe 9.61~10.14 | results/round28 |
| R29 | 新因子探索 | **TSMOM 发现** Sharpe 5.40, ATR Ratio/Vol-Rank/Volume 否决 | results/round29 |
| R30 | TSMOM 深度验证 | K-Fold 6/6, 与L7相关<0.05, 组合L7(1.25)+TS(0.5)=Sharpe 10.55 | results/round30 |
| R31 | EqCurve 深度优化 | **LB=10,Cut=0,Red=0** 最优: Sharpe+1.93, MaxDD-50%, K-Fold 6/6 | results/round31 |
| R32 | 多TF确认/波动率/替代指标/ML | **H1 KC同向过滤 Sharpe 13.78**, K-Fold 6/6; ML/Vol微弱 | results/round32 |
| R33 | 跨资产相关性/GVZ | DXY/US10Y/SPX/GVZ 均无显著过滤价值 | results/round33 |
| R34 | 订单流/微观结构 | 25.7M ticks, spread均值$0.92, 大跳后无方向偏差 | results/round34 |
| R35 | 全策略深度验证 | **H1 KC参数无cliff(11.57-14.19)**, TSMOM最优SL4/TP12/MH80, 组合K-Fold 6/6(9.44-11.64), ST/PSAR均6/6通过 | results/round35 |
| R36 | 实盘边际优化 | 22/22半年窗口全正Sharpe(12.9-17.8), MC 200次@$1.00 spread最差Sharpe仍9.99, 亚洲盘排除+1.23 | results/round36 |
| R50 | L8全参数暴力搜索 | **48,300组合搜索, 0/50通过K-Fold**, L8_MAX仍为全局最优, 参数空间已充分搜索 | results/round50_results |

---

## 核心结论汇总

### 策略版本演进
| 版本 | 改进 | Sharpe (sp$0.3) | 状态 |
|------|------|-----------------|------|
| L3 | MaxHold20+Choppy0.50+Tight_all | 4.07 | 历史 |
| L5 | L3+TDTP OFF+AllTight Trail | 5.43 | 历史 |
| L5.1 | L5+SL3.5+MaxPos1 | 6.17 | 历史 |
| L6 | L5.1+UltraTight2 regime trail | 7.18 | 历史 |
| L7 | L6+TATrail(s2/d0.75/f0.003)+Gap1h+MH=8 | 7.46 | 历史 |
| **L8_BASE+Cap80** | ADX14, Tight Trail, TATrail OFF, MH=20, Cap$80 | K-Fold 6/6 (CorrSh 6.27) | **当前实盘 (2026-04-28)** |

### 新发现的独立策略
| 策略 | Sharpe | K-Fold | 与L7相关性 | 状态 |
|------|--------|--------|-----------|------|
| **D1 Keltner** (EMA20/M2.0/ADX18/MH=8) | 6.03 | 6/6 (mean=9.16) | 0.17 | 已验证, 待部署 |
| **H4 Keltner** (EMA20/M2.0/ADX18/MH=20) | 4.64 | 6/6 (mean=6.27) | 0.22 | 已验证, 待部署 |
| **TSMOM** (20d+60d, H1, SL3.5/TP12/MH50) | 5.40 | 6/6 (mean=6.02) | <0.05 | 已验证, 模拟盘运行中 |
| S1 Squeeze Straddle | 1.67 | 6/6 | -0.01 | 待引擎实现 |
| S3 Overnight Hold | 0.88 | 6/6 | -0.03 | 待部署 |

### 关键认知
1. **追踪止盈是核心 alpha**: 95.6%WR, 中位1bar, 快进快出
2. **KC+ADX+EMA100 信号集已饱和**: 596种入场变体全部不如基线
3. **出场系统已接近最优**: ProfitDD/AdaptHold/Trailing调参均无法改善
4. **快速交易最赚钱**: 1-2bar WR=94.5%, 20+bar WR=7.9%
5. **Timeout是最大亏损源**: -$24,032, 中位20bar → **MaxHold 20→8 大幅减少 Timeout 亏损, Sharpe +0.7~1.0**
6. **参数极其鲁棒**: MC 80次±15%扰动, min Sharpe=5.49
7. **EMA25/Mult1.2 已近最优**: R13 EMA18仅+0.14, 不值得改
8. **TATrail 已确认有效**: R15 全30种组合正delta, 最优s2/d0.75 +0.20; R19 L7逐年11/11优于L6
9. **1h Entry Gap 已确认有效**: 过滤低质量连续突破, K-Fold 6/6双点差通过
10. **R21 发现两个独立新策略**: Squeeze Straddle (K-Fold 6/6, Sharpe 1.67) 和 Overnight Hold (K-Fold 6/6, Sharpe 0.88)
11. **黄金趋势持续性太强**: 极端反转策略 81 种组合全负 Sharpe, 不适合做逆向
12. **L7+S1+S3 三策略近零相关**: L7-S1 r=-0.01, L7-S3 r=-0.03, S1-S3 r=-0.04
13. **D1/H4 Keltner 是独立 alpha 源** (R25-R27): 与 L7 相关性仅 0.17-0.22, K-Fold 6/6, 参数无 cliff
14. **EqCurve LB=30 风控层已通过 K-Fold 6/6** (R27): 6/6 folds 均改善 Sharpe, 平均 +0.25, 非过拟合
15. **多策略组合价值**: L7(0.03)+H4(0.015) → Sharpe 8.99, PnL +20%, MaxDD $311; 三策略独立运行无需互斥
16. **D1/H4 参数极其鲁棒** (R27 cliff test): EMA15-25/Mult1.5-2.5/ADX15-25 全区域 Sharpe 3.7-6.4, 无悬崖
17. **TSMOM 独立alpha** (R29-R30): 20d+60d动量, H1, 与L7相关<0.05, K-Fold 6/6 (mean Sharpe 6.02)
18. **L7(1.25)+TSMOM(0.5) 最优组合** (R30): Sharpe 10.55, PnL $88,607, K-Fold 6/6 (mean 11.24)
19. **EqCurve LB=10,Cut=0,Red=0** (R31): 比LB=30更优, Sharpe +1.93, MaxDD减半至$59, K-Fold 6/6
20. **H1 KC(EMA20/M2.0) 多TF同向过滤** (R32): L7 Sharpe 9.61→13.78, K-Fold 6/6 (每fold +3.3~5.3), **最重要的R32发现**
21. **H1 KC(EMA25/M1.2) 同向过滤** (R32): L7→12.34, K-Fold 6/6, 保留更多交易
22. **替代趋势指标可作独立策略但弱于KC** (R32): SuperTrend 3.55, PSAR 3.43, Ichimoku 2.96
23. **ML元模型不如EqCurve** (R32): XGBoost Walk-Forward仅+0~0.3, EqCurve LB=10达12.26
24. **跨资产/GVZ对L7无边际价值** (R33): L7在所有DXY/US10Y相关性regime下Sharpe稳定(9.3~10.3)
25. **Tick微观结构无可用信号** (R34): 大跳后continuation=49%(≈随机), spread均值$0.92(高于M15估计)
26. **H1 KC多TF过滤参数极其鲁棒** (R35-A): 36种EMA/Mult组合全在Sharpe 11.57-14.19, 无参数悬崖; 最优EMA15/M2.0=14.19
27. **L7+H1filter+EqCurve三层叠加** (R35-A): Sharpe=13.97, MaxDD仅$64, 是当前已知最强组合
28. **TSMOM出场优化** (R35-B): 最优SL=4.0/TP=12/MH=80/Trail0.28/0.06, Sharpe=4.40, WR=88.3%; K-Fold 6/6(5.08-6.76)
29. **TSMOM+EqCurve LB=5** (R35-B): Sharpe从4.40→8.39, PnL +49%, **重大发现**
30. **TSMOM spread敏感** (R35-B): $0.50仍可用(Sharpe 3.62), $1.00以上衰减严重
31. **最优组合L7(1.5)+TS(0.25)** (R35-C): Sharpe=10.53, PnL=$83,912, MaxDD=$137; K-Fold 6/6 (9.44-11.64)
32. **SuperTrend K-Fold 6/6** (R35-D): P20/F3.0, Sharpe 2.19-5.99, 与L7相关仅0.088
33. **PSAR K-Fold 6/6** (R35-D): AF0.01/Max0.10, Sharpe 3.43-5.12, 与L7相关仅0.050
34. **ST/PSAR参数均无cliff** (R35-D): ST全区域2.65-3.80, PSAR全区域2.23-3.43
35. **亚洲盘是最弱时段** (R36-A): Asian Sharpe 11.11 vs NY 15.70, 排除亚洲盘Sharpe 13.78→15.00(+1.23), 但损失$15k PnL
36. **12-14点(UTC)是最强时段** (R36-A): Hour12 WR=99.4% AvgPnL=$9.67, Hour14 WR=97.7% AvgPnL=$7.76
37. **DOW无显著效应** (R36-B): 周一到周五Sharpe 12.98-15.59, 排除任一天delta<±0.4, 不值得过滤
38. **ATR Lot Sizing无改善** (R36-C): MildInv Sharpe仅+0.17, 四分位Sharpe都在12-13.8, 保持flat lot最优
39. **跳过22-1点(UTC)最有效** (R36-D): Sharpe 13.78→14.57(+0.79), 仅损失722笔低质量交易
40. **联合EqCurve不如独立EqCurve** (R36-E): 独立Sharpe 11.62 > 联合10.25, L7的EqCurve几乎不触发(96次)
41. **22/22半年窗口全正Sharpe** (R36-F): 最差12.90(2018H2), 最佳17.82(2023H1), 均值15.00, 无系统性衰减
42. **最大连续亏损仅2天** (R36-F): 无3天以上连亏, MaxDD仅$64
43. **MC 200次@$1.00spread仍Sharpe≥10** (R36-G): 最极端条件(均值$1, std $0.5)最差仍9.99, 100%概率>0
44. **Session-aware spread模型** (R36-G): 亚洲盘$0.80/NFP $1.50/其他$0.35, 均值Sharpe=12.93, 100%>0
45. **L8_MAX已是全局最优** (R50): 48,300组合暴力搜索(5,100核心+43,200叠加), Top50全部K-Fold FAIL(0/50), 参数空间充分覆盖, 无法超越L8_MAX(Sharpe 11.23)

### 已否决方向汇总
- PA形态 (R11): Pinbar/Fractal/InsideBar/Engulfing 全部 K-Fold 0/6
- Squeeze过滤 (R12): K-Fold 0/6 (但 R21 Squeeze 作为独立信号源有效)
- 保本止损 (R13): delta全为0, 无效
- 双KC通道 (R13): 交集降低表现, Fast Confirmed仅+0.01
- HMA/KAMA替代EMA (R13): 全部不如标准EMA25
- Strategy A/C/D: 596种组合全不如基线
- London Breakout / 宏观因子: 全部无效
- 极端反转/流动性供给 (R21-S4): 81种组合全负Sharpe, 黄金不适合逆向
- ATR Ratio regime (R29): IC<0.01, 无预测力
- Volume Anomaly (R29): 无显著alpha
- Vol-Rank Sizing (R29): 仅+0.42 Sharpe
- ATR Term Structure (R32-B): IC<0.01, 独立策略最高Sharpe 3.69但远不如KC
- ML Meta-model (R32-D): XGBoost Walk-Forward +0~0.3, 不如简单EqCurve
- DXY/US10Y/SPX 相关性过滤 (R33-A): Sharpe变化-0.15~+0.47, 不值得增加复杂度
- GVZ 仓位管理 (R33-B): 最优仅+0.47, 但损失大量PnL
- Tick 微观结构 (R34): 大跳无方向偏差, spread spikes集中在亚洲盘(23:00), 无可用信号
- L8 全参数暴力替代 (R50): 48,300组合(Mult0.8-1.6/SL2.0-3.5/各种叠加), K-Fold 0/50通过, Fold3全面崩溃, M0.8等替代参数时间不稳定

---

## 待办事项

- [x] ~~R13/R15 完成后汇总结论~~ → L7 定义已锁定
- [x] ~~TATrail与L6叠加~~ → L7 = L6 + TATrail(s2/d0.75/f0.003) + Gap1h
- [x] ~~R21 新维度探索~~ → S1 + S3 通过 K-Fold 6/6
- [x] ~~R25-R27 D1/H4 Keltner 发现与验证~~ → D1/H4 KC 独立 alpha, 全部通过 K-Fold 6/6
- [x] ~~R27 EqCurve K-Fold~~ → 6/6 改善, 平均 +0.25 Sharpe
- [x] ~~R27 参数 cliff test~~ → 无 cliff, 参数极其鲁棒
- [x] ~~R27 MaxHold 优化~~ → D1 MH=8 (6.03), H4 MH=20 (4.64), 均 K-Fold 6/6
- [x] ~~R28 L7(MH=8) 完整 K-Fold~~ → 6/6 通过
- [x] ~~R29 新因子探索~~ → TSMOM发现, 其余否决
- [x] ~~R30 TSMOM深度验证~~ → K-Fold 6/6, 组合最优
- [x] ~~R31 EqCurve深度优化~~ → LB=10,Cut=0,Red=0 最优
- [x] ~~R32 多TF/波动率/替代指标/ML~~ → **H1 KC同向过滤 重大发现 Sharpe 13.78**
- [x] ~~R33 跨资产/GVZ~~ → 无边际价值
- [x] ~~R34 Tick微观结构~~ → 无可用信号
- [x] ~~R35 全策略深度验证~~ → H1KC无cliff, TSMOM优化确认, 组合K-Fold 6/6, ST/PSAR均通过
- [x] ~~R36 实盘边际优化~~ → Session/DOW/ATR sizing/Spread/联合EqCurve/WF/MC 全面完成
- [ ] **L7(MH=8) + H1 KC同向过滤 + EqCurve 部署实盘**: R35确认三层叠加Sharpe 13.97, MaxDD $64, 参数鲁棒无cliff
- [ ] **TSMOM(SL4/TP12/MH80) + EqCurve LB=5 部署**: Sharpe 8.39, K-Fold 6/6, 与L7相关<0.05
- [ ] **SuperTrend(P20/F3.0) 独立EA**: Sharpe 3.55, K-Fold 6/6, 与L7相关0.088
- [ ] **PSAR(AF0.01/Max0.10) 独立EA**: Sharpe 3.43, K-Fold 6/6, 与L7相关0.050
- [ ] **TSMOM 模拟盘观察**: 代码已提供, 等待观察期结果
- [ ] **D1/H4 Keltner 部署**: 作为独立 EA 运行
- [ ] S1 Squeeze / S3 Overnight: 待部署
