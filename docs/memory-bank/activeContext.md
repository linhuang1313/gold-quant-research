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

## 进行中 / 已完成实验 (2026-04-20 更新)

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

---

## 核心结论汇总

### 策略版本演进
| 版本 | 改进 | Sharpe (sp$0.3) | 状态 |
|------|------|-----------------|------|
| L3 | MaxHold20+Choppy0.50+Tight_all | 4.07 | 历史 |
| L5 | L3+TDTP OFF+AllTight Trail | 5.43 | 历史 |
| L5.1 | L5+SL3.5+MaxPos1 | 6.17 | 历史 |
| L6 | L5.1+UltraTight2 regime trail | 7.18 | **当前实盘 (2026-04-18 部署)** |
| L7 | L6+TATrail(s2/d0.75/f0.003)+Gap1h | 7.46 | **paper trade 中** |

### 关键认知
1. **追踪止盈是核心 alpha**: 95.6%WR, 中位1bar, 快进快出
2. **KC+ADX+EMA100 信号集已饱和**: 596种入场变体全部不如基线
3. **出场系统已接近最优**: ProfitDD/AdaptHold/Trailing调参均无法改善
4. **快速交易最赚钱**: 1-2bar WR=94.5%, 20+bar WR=7.9%
5. **Timeout是最大亏损源**: -$24,032, 中位20bar
6. **参数极其鲁棒**: MC 80次±15%扰动, min Sharpe=5.49
7. **EMA25/Mult1.2 已近最优**: R13 EMA18仅+0.14, 不值得改
8. **TATrail 已确认有效**: R15 全30种组合正delta, 最优s2/d0.75 +0.20; R19 L7逐年11/11优于L6
9. **1h Entry Gap 已确认有效**: 过滤低质量连续突破, K-Fold 6/6双点差通过
10. **单策略系统风险可控**: Keltner 经19轮实验锤炼, 鲁棒性极强; 品种多元化优先于策略堆叠
11. **R21 发现两个独立新策略**: Squeeze Straddle (K-Fold 6/6, Sharpe 1.67) 和 Overnight Hold (K-Fold 6/6, Sharpe 0.88) 可作为 L7 的叠加策略
12. **黄金趋势持续性太强**: 极端反转策略 81 种组合全负 Sharpe, 不适合做逆向
13. **L7+S1+S3 三策略近零相关**: L7-S1 r=-0.01, L7-S3 r=-0.03, S1-S3 r=-0.04, 真正独立
14. **组合增加收益但不改善 Sharpe**: L7 alone $61,769 Sharpe 7.24 → L7+S1+S3 $66,080 Sharpe 7.23 (L7 太强, S1/S3 体量小)
15. **EUR/USD Keltner 信号有效**: 1.3 年 yfinance 数据 Sharpe 1.6~2.3, MC 100%正, 但**需要 Dukascopy 11 年数据做完整验证**

### 已否决方向汇总
- PA形态 (R11): Pinbar/Fractal/InsideBar/Engulfing 全部 K-Fold 0/6
- Squeeze过滤 (R12): K-Fold 0/6 (但 R21 Squeeze 作为独立信号源有效)
- 保本止损 (R13): delta全为0, 无效
- 双KC通道 (R13): 交集降低表现, Fast Confirmed仅+0.01
- HMA/KAMA替代EMA (R13): 全部不如标准EMA25
- Strategy A/C/D: 596种组合全不如基线
- London Breakout / 宏观因子: 全部无效
- 极端反转/流动性供给 (R21-S4): 81种组合全负Sharpe, 黄金不适合逆向

---

## 待办事项

- [x] ~~R13/R15 完成后汇总结论~~ → L7 定义已锁定
- [x] ~~TATrail与L6叠加~~ → L7 = L6 + TATrail(s2/d0.75/f0.003) + Gap1h
- [x] ~~R21 新维度探索~~ → S1 Squeeze Straddle + S3 Overnight Hold 通过 K-Fold 6/6
- [ ] L7 paper trade 积累样本，达标后切换实盘
- [ ] S1 Squeeze Straddle: 需要在引擎中实现独立通道, 进入 paper trade
- [ ] S3 Overnight Hold: 极简策略可直接部署 paper trade (每天 21:00 UTC 买入, 07:00 UTC 平仓)
- [ ] EUR/USD: 需下载 Dukascopy 2015-2026 H1 数据做 K-Fold 6/6 验证 (yfinance 只有 1.3 年)
- [ ] 组合仓位优化: S1/S3 可考虑加大手数以提升组合贡献 (当前 S1+S3 仅占总利润 7%)
