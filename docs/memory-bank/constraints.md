# 研究约束与否决方向 (Constraints)

> **读取频率: 每次对话必读，尤其在提出任何优化建议前**
> 已否决的研究方向（禁止重复）、回测框架陷阱、方法论纪律

---

## ⛔ 被否决的研究方向 — 禁止重复

> 以下方向已经过回测验证确认无效或有害，**不再投入时间重复研究**。
> 如果某个新想法本质上是以下方向的变体，直接跳过。

### 入场过滤类
- **宏观 Regime 过滤** (3次确认): 策略在所有 regime 下都盈利
- **日前趋势预判**: 准确率 ~55% ≈ 抛硬币
- **D1 日线方向过滤**: 逆D1反而更好 ($/t=$3.45 > $2.86)
- **K线实体/影线过滤**: 与直觉相反 (Low body > High body)
- **KC Bandwidth 扩张过滤**: BW3-BW12 全降 Sharpe (-0.40 ~ -1.60)
- **London Breakout**: 7变体全负 Sharpe
- **Strategy A/C/D**: 596种组合全不如基线
- **EMA150/EMA100 趋势过滤**: K-Fold Fold4 崩溃 / skipped=0
- **R11 PA形态全系** (2026-04-16): Pinbar/Fractal/InsideBar/Engulfing 作为过滤器 K-Fold 全0/6
- **R11 PA共振**: Confluence≥2 Sharpe=1.38, ≥3 → -0.20 (越多越差)
- **R11 PA+SR独立策略**: S/R zone太窄, 无额外信号
- **R12 Squeeze(BB inside KC)**: K-Fold 0/6, 过滤太多好信号
- **R12 连续突破确认**: K-Fold 0/6, Sharpe降至5.25/4.75
- **R13 保本止损(Breakeven)** (2026-04-16): delta全为0, 完全无效
- **R13 双KC通道** (2026-04-16): 交集降低表现, Fast Confirmed仅+0.01
- **R13 HMA/KAMA替代EMA** (2026-04-16): 全部不如标准EMA25 (HMA 5.82, KAMA 5.99 vs EMA 6.17)

### 时间/时段过滤类
- **周一降仓/跳过任何星期**: 每天都赚钱, 跳过任何一天 Sharpe 均下降
- **时段过滤**: K-Fold 仅 2/6, 所有时段都赚钱
- **降低交易频率 (min gap 2-8h)**: 好信号被跳过

### 仓位管理类
- **波动率过滤**: 跳过高波动 Sharpe 降到 0.79
- **ATR Regime 反波动率加权**: 11/12年降 Sharpe
- **禁用 SELL**: PnL -$2,127
- **连续亏损自适应减仓**: 连亏后下一笔期望仍为正

### 出场类
- **事件日防御 "带伞策略"**: 11伞×3触发×6折, 全无效
- **RSI 背离提前出场**: 后验非预测
- **ATR spike protection**: 真实引擎仅 +0.03
- **R12 利润回吐止盈(ProfitDD)**: K-Fold 0/6, Trailing已很紧
- **R12 自适应MaxHold**: 对结果完全无影响 (Trailing先于MaxHold触发)
- **KC Mid Reversion Exit (L4)**: look-ahead bias 审计否决, 修正后 0/6 FAIL
- **Trail Momentum 1.5x**: 修复后引擎全面有害, K-Fold 0/6

### 策略类
- **Donchian Channel**: 全部负 Sharpe
- **Keltner 均值回归 (H1)**: 15配置全负, 盈亏比不足(SL远大于到KC mid的TP空间)
- **XAUUSD 均值回归全面否决**: H1 KC(15配), M1 MeanRevert/Bounce, RSI均值回归全负; 黄金趋势持续性太强
- **RSI2 均值回归 (EUR/USD H1)**: 同样负 Sharpe, 与黄金结论一致
- **R21 S4 极端反转**: 81组参数全负 Sharpe, "逆向做反转在黄金上不可行"
- **R21 S2 NFP/FOMC Continuation**: 信号太弱, 不单独建模
- **Stochastic Mean Reversion**: 标记观察, 未验证有效

### ML / 高频 Scalper 类 (2026-04-20 全面否决)
- **M1 EMA Scalper (Grid/HF)**: M15→M1 数据降级无效, 所有变体亏损
- **M1 规则 Scalper (MeanRevert/Momentum/Bounce/Smart)**: 最佳 WR 66% 但 RR<1, 全亏
- **M1 ML Scalper v2 (XGBoost)**: Sharpe -4.10, WR 54.5%, RR 0.70
- **M1 ML Scalper v3 (XGB+LGB Ensemble + VolFilter + ATR自适应)**: 最佳 Sharpe -2.42
- **M1 ML Scalper v4 (智能出场 Lock/QCut)**: 最佳 Sharpe -0.85 (TP$5/SL$3.5)
- **M1 ML Scalper v5 (6m训练 + 精细网格搜索)**: 最佳 Sharpe -1.08, 2025+ AUC退化到0.5
- **结论**: M1 bar 级别统计特征不足以产生 alpha; 截图策略依赖 tick/order flow 基础设施, 不可复制

### Keltner 框架内微调 (2026-04-20 确认触顶, 2026-04-22 更新)
- **L系列进化已触顶**: L3(4.07)→L5(5.43)→L5.1(6.17)→L6(7.18)→L7(7.46), 边际递减
- **R13 EMA/Mult/MA 全扫描**: EMA25/Mult1.2 已近全局最优, 全部否决
- **R24 L8 候选**: L8c_max_tight Sharpe 10.03 (K-Fold 6/6), 但仍为同框架微调
- **不再投入时间做 Keltner 参数扫描/微调**, 除非有结构性新信息
- **⚠️ 例外: MaxHold 优化有效** (R25/R27): MH 20→8 Sharpe +0.7~1.0, 这不是参数微调而是结构性改善 (减少 Timeout 最大亏损源)

### R45 新信号源探索 — 放弃项 (2026-04-27)
- **S2 BB Squeeze-to-Expansion**: 11年仅3笔交易, K-Fold 1/6, 严重过拟合无统计意义
- **S6 Range Contraction Filter on L8**: 过滤掉83%交易, Sharpe从11.34全面下降至6.6~7.9, 无正向价值
- **S1 Donchian Channel Breakout**: K-Fold 6/6 Sharpe 7.83 但**与L8日收益相关性0.445**, 同为趋势突破, 组合无分散化价值; 不作为独立策略部署
- **S5 Z-Score Mean Reversion (激进配置)**: p=100/z=3.0/adx<20 虽 Sharpe 5.72 但仅198笔; 修复后(p=150/z=3.5/adx<20)虽6/6但仅40笔, 统计功效不足, **标记观察不部署**

### R45 新信号源探索 — 正面结论 (2026-04-27)
- **S4 Chandelier Exit Flip**: K-Fold 6/6, mean=6.35, min=6.02; **与L8相关性-0.021(零)**; 组合MaxDD降23%; 12组参数全>10.2组合Sharpe; 推荐部署
- **S3 Dual Thrust**: K-Fold 6/6, mean=5.88, min=4.27; L8相关性0.277(中低); 三策略组合min Sharpe 10.62 > L8单独10.29; 推荐观察
- **Spread抗性**: 全部4个候选策略在$1.50 spread下仍盈利, S1 Donchian最抗spread(衰减21%), S4/S3衰减28%

### R25-R27 已验证的正面结论 (2026-04-22)
- **D1/H4 Keltner 是独立 alpha**: EMA20/M2.0/ADX18, K-Fold 6/6, 与 L7 相关性 0.17-0.22
- **D1/H4 参数极其鲁棒**: R27 cliff test 125 组合, EMA15-25/Mult1.5-2.5/ADX15-25 全区域 Sharpe 3.7-6.6, 无悬崖
- **EqCurve LB=30 是有效风控层**: R27 K-Fold 6/6 改善, 平均 +0.25 Sharpe, 非过拟合
- **多策略独立运行无冲突**: R27 L7-H4 重叠时 88% 同方向, 不需要互斥逻辑

### Cross-Asset Z-Score 否决 (2026-05-05)
- **Z-Score 入场过滤器**: 脚本 `gold-quant-trading/scripts/cross_asset_zscore_backtest.py`, 2y 1h (11,332 bars), Gold/Brent/US10Y 24h 滚动 Z-Score. 7 种 Z 条件 + 4 种 KC+Z 联合 + 3 种 EMA+Z 联合, 全部 Sharpe < 纯 KC 基准 (1.28). **否决: Z-Score 无法改善 Keltner 入场质量**
- **Z-Score 仓位调节器**: 5 种方案 (金油分化加仓/超买减仓/避险加仓/混合/EMA+Z). 唯一亮点 `z_overbought_reduce` MaxDD -6.5% vs -8.9% (改善27%), 但总收益 +11.2% vs +25.2% (损失55%), Sharpe 0.86 vs 1.28. **否决: 用减仓换回撤在趋势市中代价太大**
- **定位**: Cross-Asset Z-Score 仅作 Dashboard 宏观 regime 温度计 (定性参考), 不纳入实盘入场/仓位/出场逻辑
- 数据报告: `gold-quant-trading/data/cross_asset_zscore_backtest.md` + `.json`

### Dynamic Sizing 否决 (R25 Phase B, 2026-04-21)
- **Streak-based sizing**: 连亏/连赢后调仓, 全部无改善 (连亏后下一笔期望仍为正)
- **Regime-based sizing**: ATR regime 加权, 无改善
- **Kelly Criterion sizing**: f=0.553, 但实际应用降低 Sharpe
- **⚠️ 例外: EqCurve LB=30 有效** — 近30笔均值<0 时缩仓0.5x, K-Fold 6/6 验证

---

## ⚠️ 回测框架陷阱 — 必须避免

### H1 Look-Ahead (已修复但需警惕)
- Dukascopy H1 时间戳 = bar 开盘时间
- **任何使用 H1 Close/KC_mid 的新规则必须用 `closed_only=True` 或 `iloc[-2]`**
- `_check_exits` 传入的 `h1_window` 默认含未收盘 bar
- SL/TP 用 M15 High/Low (安全), Trailing 用 ATR (安全)

### 入场价
- 信号在当前 bar 产生, 但入场价必须是下一根 M15 Open (pending queue)
- 绝不能用当前 bar Close 作为入场价

### post-hoc vs 真实引擎
- post-hoc 分析与真实引擎差异可达 19 倍 (EXP52: +0.48 vs +0.03)
- 所有改进必须在 engine.py 中实现并回测, 不能只做 post-hoc 分析

### monkey-patching
- `staticmethod` 用 `Class.__dict__['method']` 保存原始描述器
- `from module import func` 创建本地绑定, patch 原模块无效

---

## 方法论纪律

### 提议新优化前必须检查
1. **查阅否决列表**: 是否已被否决？变体也算
2. **明确性质**: "假设"还是"已验证"？< 30 笔只能标"假设"
3. **查阅因子IC**: 与已有因子扫描矛盾就放弃
4. **先回测后改代码**: 带 $0.30-$0.50 点差
5. **不动已验证参数**: 除非新数据明确优于

### 检查表 (每次提议必须输出)
```
┌─ 优化提议检查表 ─────────────────────────┐
│ 性质: 假设(待验证) / 已验证(回测#XX)       │
│ 否决历史: 是否有相关结论？→ 引用           │
│ 样本量: N = ??                             │
│ IC因子一致性: 是否矛盾？                   │
│ 回测状态: 未开始 / 进行中 / 已完成(结果)    │
│ 涉及已验证参数: 是(哪个) / 否              │
└─────────────────────────────────────────┘
```

### RSI2 过滤器否决 (2026-05-05)
- **RSI2 > 85 过滤做多 — 否决**: 66 笔实盘数据 (2026-04-14~05-05)。RSI2>85 的 BUY 13 笔 PnL=+$77.62 (WR=85%), RSI2<=85 仅 4 笔 PnL=+$8.67。过滤高 RSI2 信号会损失 90% 做多利润。Keltner 突破时 RSI2 天然极高 (99+) = 动量确认
- **RSI2 < 15 过滤做空 — 否决**: RSI2<10 的做空是盈利主力 (20 笔, PnL=+$364.59, WR=90%)。极度超卖 = 空头动量确认而非反转信号
- **方法论教训**: IC 负值不能简单翻译为"高值时应跳过"。IC 是截面秩相关统计量，反映的是"因子值高/低与收益好/坏的对应关系"，必须按方向(BUY/SELL)分别验证才能确定可操作的过滤规则

### 参数变更联动验证规则 (2026-05-08, R176 系列发现, 永久生效)

> **任何核心参数变更后，必须重新验证所有依赖该参数的上层模块。违反此规则导致的部署视为无效。**

1. **SL/TP/MH/Trail 变更 → 重新验证所有入场过滤器**
   - 原因: R166b 改了 TSMOM 的 SL/TP/MH 后没有重测 D1 EMA20 过滤器，导致 Sharpe 损失 0.63（-11%）长达数周未被发现
   - R123 在旧参数下验证 D1 过滤 +22%，但 R166b 改参后 D1 过滤变为 -11%，**参数交互关系完全改变**
   - 规则: 改出场参数后，必须对所有入场过滤器跑 ON/OFF A/B 测试 + K-Fold

2. **手数变更 → 重新验证 MaxLoss Cap**
   - 原因: R89 手数优化器将手数从 0.01→0.09/0.15，但 Cap 是美元绝对值，`build_portfolio_daily()` 的线性缩放 `pnl × lot_multiplier` 无法正确反映 Cap 在不同手数下的行为
   - 例: PSAR Cap=$5 在 0.01 手下合理 ($5/oz 容忍度)，0.09 手下 = $0.56/oz，44% 交易被噪音强平
   - 规则: Cap 必须在**实际部署手数**下独立回测，禁止从其他手数线性推算

3. **新策略上线前 → 必须跑过滤器 shootout**
   - 原因: Chandelier 上线时沿用 EMA100 过滤从未被质疑，实际上 EMA100 过滤了 49.6% 交易且 Sharpe 排名倒数，RSI 30/70 仅过滤 9.5% 但 Sharpe 更高
   - 规则: 新策略部署前至少测: NoFilter + EMA50/100/200 + D1 EMA20 + RSI 30/70 + ADX>20，选 K-Fold 最稳定的

4. **变更影响矩阵** (当前):
   ```
   改 SL/TP/MH/Trail → 重测: 入场过滤器, Cap, K-Fold
   改 手数/lot       → 重测: Cap, 组合 MaxDD
   改 过滤器         → 重测: K-Fold, 近期表现
   改 Cap            → 重测: 触发率, K-Fold
   ```

### 踩过的坑
- **小样本归纳**: 3笔案例 cherry-picking → 结论自动作废
- **无视已有结论**: 必须先查 journal/否决列表
- **先改代码后验证**: 先有回测数据再改 engine
- **市场分析必须先查日历再归因**: 大跌≠策略问题 (如 Liberation Day)
- **IC 误读 (2026-05-05)**: RSI2 IC=-0.28 看似"高RSI2→低收益"，但分BUY/SELL验证后发现高RSI2做多反而赚钱。IC 是聚合指标，不能直接当单向过滤器用
- **R89 Cap 线性缩放缺陷 (2026-05-08)**: R89 在 0.01 手下优化 Cap 后将结果直接应用于 0.09/0.15 手，但 Cap 是美元绝对值阈值，手数变了 Cap 的价格容忍度 ($/oz) 完全不同。导致 PSAR/SESS_BO/Keltner 等策略实盘触发率远高于回测预期
- **参数改了不重测过滤器 (2026-05-08)**: R166b 改 TSMOM SL/TP/MH 后直接沿用 R123 的 D1 过滤器结论，没有重新验证。R176b 发现 D1 过滤在新参数下从 +22% 变成 -11%。**过滤器的有效性依赖于它被验证时的参数环境**
- **默认配置不等于最优配置 (2026-05-08)**: Chandelier 的 EMA100 过滤是从 R45 就自带的"默认设置"，从未被当作变量做过 A/B 测试。R176e 测了 17 个过滤器后发现 EMA100 排名垫底，RSI 30/70 才是最优
