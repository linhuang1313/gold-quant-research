# 当前上下文 (Active Context)

> **读取频率: 每次对话必读**
> 当前焦点、进行中的实验、待办事项、最近系统状态

---

## 当前实盘状态 (2026-04-15)

### 舆情系统 v5 升级 (2026-04-15)
- **已部署**: 7 项修复全部完成，需重启 `gold_runner.py` 生效
- **direction_bias 观察模式**: 仅记录不过滤，待积累 30+ 笔样本后评估是否启用硬过滤
- **跨资产宏观信号**: DXY/VIX/US10Y/Brent 现在直接影响 direction_bias 和 lot_multiplier
- **动态更新间隔**: 正常 180s，高波动时自动降至 60s

### 历史 (2026-04-13)

- 实盘运行中: `gold_runner.py` → MT4 桥接
- 本金 $2,000，最大总亏损保护 $1,500
- **L5.1 部署 (2026-04-13)**: L5 + SL 3.5x + MaxPos=1
  - SL: 4.5→3.5 (R6-A5 K-Fold 6/6, Sharpe +0.17)
  - MaxPos: 2→1 (R6-A4 K-Fold 6/6, Sharpe +0.43, MaxDD -$72)
  - L5 特殊条款已用完，从此开始严格执行部署纪律
- 最近连续亏损: 4/6-4/8 共 5 笔止损（~$163），4/6 两笔 Keltner + 4/8 三笔(Keltner+RSI)

## 🚨 回测引擎重大修复 (2026-04-09)

**H1 look-ahead bias + 入场价 look-ahead 已修复**（commit 7f02772）

修复内容：
1. `_get_h1_window(closed_only=True)` — 入场信号只使用已收盘的 H1 bar
2. pending signal 队列 — 信号在下一根 M15 bar 的 Open 执行，不用当前 bar Close

修复前后初步对比（Current 无成本）：
- 修复前: N=18,544 Sharpe=5.06 PnL=$35,251
- 修复后: N=25,677 Sharpe=3.18 PnL=$26,206
- Sharpe 下降 37%，但仍为强正值

**⚠️ 重要**: 之前所有实验的 Sharpe 数字都基于有 look-ahead 的旧引擎，不再可信。
在服务器运行 `run_lookahead_fix_verify.py` 获取 6 配置完整基准线后，
之前发现的优化（T7 ExtremeRegime、Trail Momentum 1.5x 等）需要在修复后的引擎上重新验证。

## 回测/实盘路径对齐 (2026-04-09)

完成 16 项不对齐审查，修复了最关键的 6 项：
1. **`LIVE_PARITY_KWARGS`** — 新增实盘对齐参数 preset（Mega Trail 参数、time_decay、rsi_adx_filter 等）
2. **`live_atr_percentile`** — 新引擎选项，用实盘方式的 rolling-50 ATR rank
3. **`_get_atr_percentile()`** — 统一方法消除 8 处重复取值
4. **Sharpe ddof=1** — 统计更准确
5. **数据缺口检测** — 加载时自动报告
6. **⚠️ 后续实验应使用 `LIVE_PARITY_KWARGS` 而非 `C12_KWARGS`**

## 进行中的实验 (2026-04-13 14:15 更新)

### Server A (12核/90GB) — R7 运行中
| Phase | 状态 | 耗时 | 关键结果 |
|-------|------|------|---------|
| R7-1 L5.1 基准 | **完成** | 12.7 min | Sharpe=6.17($0.30), K-Fold 6/6, WF 11/11, 12/12年盈利 |
| R7-2 Entry Gap | **完成** | 11.4 min | Gap=1h Sharpe=6.28(+0.11), K-Fold 6/6($0.30+$0.50) |
| R7-3 L6 增量 | **完成** | 12.3 min | L6 Sharpe=7.15(+0.98), K-Fold 6/6, WF 11/11, MaxDD $271→$215 |
| R7-4 Monte Carlo | **完成** | 77.7 min | **80/80 全盈利**, Sharpe均值=6.14, std=0.20, min=5.49 |
| R7-5 TP交互 | **运行中** | — | 17个进程, ~50%进度 |
| R7-6 近期放大镜 | 等待中 | — | — |

### Server B (32核/120GB) — R6B(重启) + R7 并行
| 实验 | 状态 | 备注 |
|------|------|------|
| R6B-B1 L6 评估 | Part A **完成** | L6 每年都优于 L5.1, 12年 delta 全正, 总计 +$5,648 |
| R6B-B2~B6 | **运行中** (dotenv修复后重启) | ~40% 进度, B2出场/B3组合/B4交互/B5近期/B6热力图 |
| R7 (旧版) | R7-1/2/3 完成, R7-4+ 运行中 | 10个进程, 结果与 Server A 一致 |

### ⚠️ 修复记录
- **R6B dotenv 失败**: Server B 的 `config.py` 新增 `from dotenv import load_dotenv` 导致 B2~B6 全部 `ModuleNotFoundError`
- **修复**: 安装 `python-dotenv`, 清理失败结果, 重启 R6B (2026-04-13 14:10)
- B1 Part A 在旧代码版本下成功完成（不受影响）

## 已完成的修复后验证

- **`run_lookahead_fix_verify.py`** — 已完成 (2026-04-09 18:54)
  - Current $0: Sharpe=3.18, $0.30: Sharpe=0.10, $0.50: Sharpe=-2.00
  - **Mega $0: Sharpe=4.04, $0.30: Sharpe=1.20**, $0.50: Sharpe=-0.80
  - 结论: Mega (Adaptive门控) 是关键，策略 alpha 真实存在（$0.30 Sharpe=1.20）
- **`run_t7_extreme_validation.py`** — 已完成 (2026-04-09)
  - T7 OnlyHigh K-Fold: $0.30 6/6, $0.50 6/6 全通过
  - **已部署到实盘**: high regime T0.4/D0.10 → T0.25/D0.05
- **`run_post_fix_validation.py`** — 已完成 (2026-04-10 02:01)
  - **Phase A — LIVE_PARITY 新基准线**:
    - $0.00: Sharpe=5.05, PnL=$36,261, N=20,831
    - **$0.30: Sharpe=2.29, PnL=$15,690, N=20,712** ← 实盘参考值
    - $0.50: Sharpe=0.36, PnL=$2,381
  - **Phase B — MaxHold=20 通过 K-Fold 6/6**: Sharpe 2.29→2.62, PnL +$3,033, MaxDD $949→$716
  - **Phase C — Trail Momentum 1.5x K-Fold 0/6 FAIL**: 正式否决

## 待部署的验证结果

- **MaxHold=20** (5小时): K-Fold 6/6 全赢，Sharpe +0.33, MaxDD -25%，可部署到实盘
  - 当前实盘: keltner_max_hold_m15=12 (3小时)
  - 建议改为: keltner_max_hold_m15=20 (5小时)

## 已否决方向（修复后引擎确认）

- Trail Momentum 1.5x: 全面有害，K-Fold 0/6，每个 fold delta -0.54 ~ -1.04
- Entry Quality Filters: 全部无效（旧引擎否决 + 修复后无需重验）
- Session/SL/TP 优化: 全部无效（旧引擎否决 + 修复后无需重验）

## 模拟盘策略

| 策略 | 状态 | 观察重点 |
|---|---|---|
| P4_atr_regime | 运行中 | 不同波动率环境下胜率差异 |
| P5_volume_breakout | 运行中 | volume_ratio IC≈0，预测力存疑 |
| ~~P6_dxy_filtered~~ | **已停用 (2026-04-14)** | 5笔WR=20% PnL=-$8.70，替换为P10 |
| P7_mega_trail | 运行中 | T0.5/D0.15 实盘 trailing 触发频率 |
| P8_mega_h20 | 运行中 | 短持仓(5h)是否减少SL损失 |
| P9_eurusd_keltner | 运行中 | EUR/USD KC mult=2.0 实盘验证 |
| **P10_l6_ultratight** | **新增 (2026-04-14)** | L6 UltraTight2 trailing 实盘验证, low(0.30/0.06) normal(0.20/0.04) high(0.08/0.01), SL=3.5x, MaxPos=1, MaxHold=5 |
| **P12_pinbar_keltner** | **新增 (2026-04-15)** | Keltner突破 + Pinbar形态确认，仅在突破方向有Pinbar时入场 |
| **P13_pinbar_sr** | **新增 (2026-04-15)** | Pinbar + 支撑阻力位独立策略，在S/R zone出现Pinbar时入场 |

## 数据更新 (2026-04-10)

- **BID 数据更新到 2026-04-09**: M15 266,882 bars, H1 98,808 bars（新增关税暴跌行情 4/1-4/9）
- **ASK 数据全量下载**: M15 395,232 bars, H1 98,808 bars（2015-2026 全量）
- **真实 Spread 时序构建完成**: `xauusd-m15-spread-2015-01-01-2026-04-10.csv`
  - Dukascopy spread 中位数 $0.33（接近我们假设的 $0.30）
  - 时段差异: London/NY ~$0.36, Off-hours ~$0.60, Hour 21 ~$0.60
  - 引擎新增 `historical` spread 模型，可用真实时变 spread 回测
- **Historical Spread 回测 (2024-2026)**: Sharpe 2.80（介于 Fixed $0.30 的 3.64 和 $0.50 的 2.97 之间）
  - 交易时实际 Dukascopy spread 均值 $0.62（高于固定 $0.30 假设）
  - 但 Dukascopy spread 含其零售加价，我们实际经纪商 spread 更低
  - **结论: $0.30 固定 spread 是合理的下界估计，实际成本在 $0.30-$0.50 之间**
- **数据路径已更新**: `runner.py` 自动 fallback 到旧文件名

## 实验结果汇总 (2026-04-10 最终更新)

### 通过验证的改进（可部署到实盘）
1. **MaxHold=20** — Sharpe 2.29→2.62, K-Fold 6/6 PASS
2. **EXP-G Choppy 阈值 0.50** — 每 fold 都改善, K-Fold 6/6 PASS
3. **EXP-K Tight_all Trail 强度** — Sharpe 2.62→3.62 (+38%), **K-Fold 6/6 PASS**
   - 参数: low(0.5/0.15), normal(0.35/0.10), high(0.20/0.03)
   - 关键洞察: 64% 交易在 high regime, 贡献 83% PnL
4. **EXP-V 突破强度 Sizing** — Spearman IC=0.080, Sharpe 2.29→2.71, K-Fold 6/6 PASS

### 组合测试结果（L3 stack = MaxHold20 + Choppy0.50 + Tight_all）
- **L3: Sharpe=4.07, PnL=$27,371, N=19,806, MaxDD=$357** ← 可部署版本
- L0→L1→L2→L3 累积叠加: Sharpe 2.29→2.62→3.10→4.07

### 🚨 L4 (KC mid reversion) — 已否决 (2026-04-11 审计)
- **L4 Original Sharpe 5.22 存在严重 H1 look-ahead bias**
- KC mid exit 使用 `h1_window.iloc[-1]`（未收盘 H1 bar），偷看了未来数据
- 修正 look-ahead 后 Sharpe 5.22 → 3.99（低于 L3 的 4.07）
- kc_mid_revert 出场: 822 笔, WR=0.9%, avg=-$21.44（几乎全亏）
- 修正后 K-Fold: **0/6 FAIL**（原始 6/6 PASS 是虚假的）
- 所有调参（min_bars 1~12、profit_filter、出场顺序）均无法超过 L3
- **结论: KC mid exit 完全否决，不部署**

### 重要发现（不改参数但影响认知）
6. **EXP-C Time Decay TP**: TD OFF Sharpe=2.85 vs TD ON=2.62 → **TDTP 降低了 Sharpe -0.22**（待完整 K-Fold 验证）
7. **EXP-H SL 灵敏度**: SL=3.0 Sharpe 最高 2.76 但 SL 触发从 318→577 笔，当前 4.5x 是合理平衡
8. **EXP-H TP 灵敏度**: TP 几乎不影响结果（8x vs OFF 仅差 0.02），因为 TP 很少触发
9. **EXP-H 出场分布**: Trailing 贡献 $64,890（主力），Timeout 亏 -$34,573（最大亏损源），SL 318 笔亏 -$16,122
10. **EXP-I Spread 压力**: 盈亏平衡在 ~$0.60 spread，11/12 年盈利（仅 2018 微亏 -$295）
11. **EXP-A ORB 消融**: K-Fold 3/6 FAIL，ORB 不显著贡献也不伤害
12. **EXP-T Donchian**: 全部负 Sharpe，否决
13. **EXP-W 亏损画像**: bars_held 是最强区分因子（d=-1.95），亏损单平均持仓 10.3 bars vs 赢利单 3.2 bars；Timeout 占亏损 69.9%
14. **EXP-R 基准确认**: 旧/新数据基线一致（Sharpe=2.29），11/12 年盈利，2026Q1 Sharpe=5.85

### 服务器实验状态 (2026-04-13 14:15 更新)

#### 已完成
- **Round 2/3/4** — 本地 `round2_results/` ~ `round4_results/`
- **Round 5 (A+C)** — `round5_results/`, R5-8 Monte Carlo 100% robust
- **Round 6A** — `round6_results/`, R6-A1~A6 全部完成 → L5.1 部署依据
- **R7-1~R7-4** — Server A 完成, Server B 交叉确认 (R7-1/2/3)

#### 进行中
- **Round 6B** — Server B 重新运行（dotenv修复后重启）
  - R6-B1 Part A 完成: L6 每年都优于 L5.1, 12年 delta 全正, 总计 +$5,648
  - R6-B2~B6 重新运行中 (~40%)
- **Round 7** — Server A R7-5/R7-6 运行中 (17进程), Server B R7-4+ 运行中 (10进程)
  - 脚本: `scripts/experiments/run_round7.py`

### R7 已确认核心结论 (2026-04-13)

1. **L5.1 参数鲁棒性 A+**: Monte Carlo 80次±15%扰动, Sharpe 最低5.49, 全部>4, 100%盈利
2. **L6 = 最强候选版本**: Sharpe 6.17→7.15 (+0.98), K-Fold 6/6, WF 11/11, MaxDD 降25%
3. **Entry Gap 1h 有效但边际**: Sharpe +0.11, K-Fold 6/6, 可与 L6 叠加测试
4. **两台服务器交叉验证**: R7-1/2/3 结果高度一致, 确认非随机

## Round 8 服务器实验 (2026-04-14 20:34 启动)

- 服务器: `westd.seetacloud.com:30367` (25核, Python 3.10)
- **R8-1 TP x SL Grid**: ✅ 完成 (45min) — L6 全面优于 L5.1, TP=7≈TP=8, K-Fold 6/6
- **R8-2 L6+Entry Gap**: ✅ 完成 (12min) — Gap=1h Sharpe 7.18→7.35, K-Fold 6/6 ($0.30+$0.50双验证)
- R8-3 Monte Carlo 200次: 运行中
- R8-4~R8-10: 排队中

## Round 11 — Price Action 因子全面验证 ✅ 完成 (2026-04-16)

- **灵感来源**: 张峻齐 (张Mr stock) 的裸K价格行为交易方法论 + 课程笔记
- **结果**: 24个实验，**全部否决**
- **关键数据**:
  - PA 过滤器 K-Fold: Pinbar 0/6, Fractal 0/6, InsideBar 0/6, Engulf 0/6, AnyPA 0/6
  - PA共振: Confluence≥2 Sharpe 1.38, ≥3 Sharpe=-0.20（越多越差）
  - PA+SR 独立策略: S/R zone 太窄，无额外信号触发（所有结果=基线）
  - 全局最优 AnyPA+SR1.5: K-Fold 2/6 ($0.30) — FAIL
- **结论**: KC突破信号已饱和，K线形态无法改善入场质量。详见 `constraints.md`
- **耗时**: 2,477s (~41min) — Phase 6-8

## Round 12 — "深水区探索" ✅ 完成 (2026-04-16)

- **目标**: 探索系统未覆盖的全新方向（微观结构/Squeeze/连续突破/出场前沿/因子IC/行为画像）
- **耗时**: 8,457s (~2.3h)
- **结果总览**:
  | Phase | 方向 | 结论 |
  |-------|------|------|
  | A 微观结构 | NY>London>Asia>OffHours | **有价值认知**（但时段过滤已否决） |
  | B Squeeze | K-Fold 0/6 | **否决** |
  | C 连续突破 | K-Fold 0/6 | **否决** |
  | D 出场前沿 | ProfitDD/AdaptHold 全部 K-Fold 0/6 | **否决**（Trailing已最优） |
  | E 因子IC | gold_mom IC=-0.020, kc_bw IC=+0.018 | **验证已有认知** |
  | F 行为画像 | CVaR(1%)=-$52.41, 1-2bar WR=94.5% | **重要洞察** |
- **新增认知**:
  1. 时段信号质量: NY Sharpe=4.53, London=3.86, Asia=2.89, OffHours=0.92
  2. 出场画像: Trailing 95.6%WR 中位1bar, Timeout -$24,032 中位20bar, SL -$13,915 中位10bar
  3. 持仓时间: 1-2bar 94.5%WR +$4.61/trade; 20+bar 7.9%WR -$14.42/trade
  4. 尾部风险: Worst 1% 全是SL(191/192), Worst 5% 主要是Timeout(653/962)

## Round 13 — "Alpha 淬炼" (2026-04-16 设计)

- **目标**: 在入场信号已饱和、出场已最优的基础上，开辟全新维度寻找边际 alpha
- **预计耗时**: ~20小时
- **核心思路**: 不重复已否决方向，回到基础参数扫描 + 引入新维度
- **实验结构**: 7 Phase, ~18 个实验

| Phase | 方向 | 实验 | 核心思路 |
|-------|------|------|---------|
| A | KC参数空间精细扫描 | A1~A4 | EMA/Mult网格搜索+热力图+K-Fold (从未在修复后引擎精细搜索) |
| B | Breakeven Stop | B1~B3 | 引擎已有但从未测试, 可能减少Timeout亏损 |
| C | 多速度KC信号 | C1~C3 | 快KC+慢KC并行(union/intersect/confirmed), 学术灵感 |
| D | 自适应MA中轨 | D1~D3 | HMA/KAMA替换EMA, 减少滞后 |
| E | 滚动窗口自适应 | E1 | 2Y/3Y/5Y滚动最优trail vs 固定参数 |
| F | Purged Walk-Forward | F1 | 更严格的验证(embargo=20bars), L5.1 vs L6 |
| G | 组合+L7候选 | G1~G3 | 叠加通过验证的改进, MC 100次, 全面对比 |

- **新增引擎参数**: `kc_ema_override`, `kc_mult_override`, `dual_kc_mode/fast/slow`, `kc_ma_type`, `gsr_filter_enabled/series`, `purge_embargo_bars`
- **新增指标**: `_hma()` (Hull MA), `_kama()` (Kaufman Adaptive MA), `add_dual_kc()` (双KC通道)
- **脚本**: `scripts/experiments/run_round13.py`, `scripts/deploy_r13.py`, `scripts/check_r13.py`
- **状态**: 已编写完成, 待部署

## 待办事项

### 紧急
- [x] **修复 PermissionError 误判平仓 BUG (2026-04-14)** — 已修复 mt4_bridge + position_tracker + gold_trader

### 中期
- [ ] M15 RSI 策略评估 — R4-8 显示 PnL=-$339, 考虑关闭
- [ ] Timeout 出场优化（Timeout 是最大亏损来源）
- [ ] EUR/USD paper trade 至少 20 笔后评估实盘切换

### 低优先级
- [x] ~~Telegram Token 从代码移到 .env 文件~~ — 已完成 (2026-04-13)
- [ ] 舆情系统 30 天后（~4/30）做第一次正式评估

### 已完成 (最近)
- [x] **L5.1 部署到实盘 (2026-04-13)** — SL 4.5→3.5 + MaxPos 2→1, R6-A4/A5 K-Fold 6/6
- [x] **Round 5C+6A 全部完成 (2026-04-13)** — R5-8 Monte Carlo 100% robust, R6-A1~A6 全部完成
- [x] **L5 部署到实盘 (2026-04-12)** — TDTP OFF + AllTight Trail, Sharpe 4.07→5.43
- [x] Round 2/3/4 完整实验 (2026-04-12) — 9.3 小时无错误完成
- [x] OOS 验证 (R4-0): 11/11 年全盈利, 样本外 Sharpe > 样本内
- [x] 存活模拟 (R4-2): 破产概率 0% (503 起点)
- [x] 24H Marathon 测试 (2026-04-11)
- [x] L3 部署到实盘 (2026-04-11)
- [x] L4 审计 + KC mid 永久否决 (2026-04-11)
