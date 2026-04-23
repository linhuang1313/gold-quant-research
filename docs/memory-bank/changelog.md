# 研究系统变更记录 (Changelog)

> **读取频率: 需要时查阅**
> 仅记录回测引擎、实验脚本、研究工具的变更。实盘变更见 `gold-quant-trading` 仓库。

---

## 2026-04-22

- **R28 L7(MH=8) 完整数据 K-Fold 验证**: 运行中
  - 对比: L6 vs L7(MH=20) vs L7(MH=8) 全样本 + 6-Fold K-Fold (真实引擎)
  - 脚本: `experiments/run_round28_l7mh8_kfold.py`

## 2026-04-21

- **R27 完整验证套件 (6项全部完成, 18.8分钟)**:
  - P0-1 引擎平价检查: D1 spread=0.50 Sharpe 5.62, spread=1.00 Sharpe 5.03; H4 spread=0.50 Sharpe 4.16, spread=1.00 Sharpe 3.19 → 对执行成本高度鲁棒
  - P0-2 持仓冲突: L7(20878 trades) vs D1(288) vs H4(1947); L7-H4 重叠 88% 同方向; 三策略应独立运行, 最大敞口 0.09 lot
  - P1-3 EqCurve LB=30 K-Fold: **6/6 改善**, Fold1~6 delta +0.11~+0.33, mean +0.25 Sharpe → 非过拟合
  - P1-4 参数 cliff test: D1 125 组合 (EMA15-25/Mult1.5-2.5/ADX15-25) Sharpe 3.9-6.6; H4 125 组合 Sharpe 3.7-5.1 → 无 cliff, 极其稳定
  - P1-5 MaxHold sweep: D1 最优 MH=8 Sharpe 6.03 (K-Fold 6/6, mean=9.16); H4 最优 MH=20 Sharpe 4.64 (K-Fold 6/6, mean=6.27)
  - P2-6 Lot 优化: L7(1.0x)+H4(0.5x) Sharpe 8.99, PnL $72,664, MaxDD $311; 纯 Sharpe 最优仍是 L7 alone (9.42)
  - 脚本: `experiments/run_round27.py`, 结果: `results/round27_results/R27_output.txt`
- **R25/R26 成果汇总** (此前完成, 此处补记):
  - R25 Phase A: L7 MaxHold sweep → MH=8 Sharpe 10.14 (截断数据), K-Fold 5/5
  - R25 Phase B: Dynamic Sizing → EqCurve LB=30 Sharpe +0.41; Kelly/Streak/Regime 均无改善
  - R25 Phase C: D1/H4 Keltner 网格搜索 → 发现 D1(EMA20/M2.0/ADX18) 和 H4(EMA20/M2.0/ADX18)
  - R26: D1/H4 K-Fold 验证 → 均 6/6 通过; 与 L7 相关性 0.17-0.22; 组合 L7+H4 PnL +40%
  - 脚本: `experiments/run_round25.py`, `experiments/run_round26.py`
- **服务器信息更新**: 新增 Server BJB1 `ssh -p 45411 root@connect.bjb1.seetacloud.com` (密码 `5zQ8khQzttDN`)

## 2026-04-20

- **R22 EUR/USD 品种扩展**: 81 参数组合网格搜索 (KC 1.2/1.5/2.0 × ADX 18/22/25 × Trail × MH)
  - yfinance 数据仅 2025-01 ~ 2026-04 (1.3 年, 7961 bars), 不够做 K-Fold 6/6
  - 最优 KC1.2_ADX22 Sharpe=2.27, 全部组合 Sharpe>1.28, MC 100% 正
  - **结论**: 信号有效但数据不足; 需 Dukascopy 11 年 H1 数据才能严格验证
- **R23 L7+S1+S3 组合分析**: 三策略日收益相关性矩阵
  - L7-S1: **-0.014**, L7-S3: **-0.033**, S1-S3: **-0.041** → 近乎完全独立
  - L7 alone: $61,769, Sharpe=7.24, MaxDD=$179
  - L7+S1+S3: $66,080, Sharpe=7.23, MaxDD=$281
  - 12/12 年每年组合都正; S1+S3 额外贡献 $4,311 但 L7 占 93% 收益
  - 最差 30 天窗口: -$12; 最差 90 天窗口: +$509 (组合极其稳健)
- 脚本: `experiments/run_round22_23.py`
- 结果: `results/round22_23_results/R22_R23_output.txt`

- **R21 四个新赚钱维度完成**: 跳出 Keltner 框架，探索四个独立策略方向
  - **S1 Squeeze Straddle: K-Fold 6/6 通过** — Sharpe=1.67, 736 trades, WR=82%, 12/12年全正, MaxDD=$71
    - 参数: min_squeeze_bars=3, trail_act=0.2, trail_dist=0.04, max_hold=20
    - K-Fold Sharpe: 2.20 / 2.88 / 1.90 / 3.16 / 1.60 / 1.62
  - **S2 Event-Driven (NFP/FOMC)**: NFP Straddle WR=82% Avg $5/trade 有效; FOMC 效果弱, 不独立建模
  - **S3 Overnight Hold: K-Fold 6/6 通过** — Sharpe=0.88, 4116 trades, PnL=$3148, 10/12年正
    - 策略: NYclose (21:00 UTC) BUY → LDNopen (07:00 UTC) 平仓
    - OffHours (21-0 UTC) session CumRet=140%, Sharpe=2.47
    - K-Fold Sharpe: 1.43 / 0.38 / 1.15 / 0.05 / 1.52 / 1.72
  - **S4 Extreme Reversal: 全部否决** — 81种参数组合 Sharpe 全负 (最好 -0.43), 黄金趋势性太强不适合逆向
- 脚本: `experiments/run_round21.py`, `experiments/run_round21_phase3.py`
- 结果: `results/round21_results/R21_full_output.txt`, `results/round21_results/R21_phase3_kfold.txt`

## 2026-04-18

- **L6 部署实盘**: UltraTight2 regime trail, 全参数对齐 LIVE_PARITY_KWARGS
- **L7 定义锁定, 进入 paper trade**: L7 = L6 + TATrail(start=2, decay=0.75, floor=0.003) + min_entry_gap_hours=1.0
- **R9-2 完成**: L7 vs L6 K-Fold 6/6 (双点差), Walk-Forward 12/12 年, delta +0.12~+0.48
- **R13 全部完成**: EMA/Mult/保本止损/双KC/HMA/KAMA — 全部否决, EMA25/Mult1.2 确认最优
- **R15 TATrail 完成**: 30 种 Start×Decay 全正 delta, 最优 s2/d0.75 +0.20 Sharpe
- **R19-E1 完成**: L7 全样本 Sharpe 7.46(+0.28 vs L6), 11/11 年逐年优于 L6, $0.50 下 5.18(+0.30)
- **策略多元化评估**: 单策略系统经 19 轮实验锤炼, 上百种新方向全部不如 Keltner 基线; 结论: 品种扩展优先于策略堆叠

## 2026-04-16

- **仓库拆分完成**: 从 `gold-quant-trading` 独立出研究仓库
  - 迁移: 104 实验脚本, 119 部署脚本, 28 legacy 脚本, 8 轮结果, 11 CSV 数据文件
  - `research_config.py` + `indicators.py` 替代实盘 config/signals 引用
  - Bugfix: `research_config.py` 补充 `LOT_SIZE=0.03`, 28个legacy脚本+2个factor脚本添加 `sys.path`
- **R11 PA形态全面验证完成**: 24实验, 全部否决 (K-Fold 全0/6)
  - 新增引擎参数: `pinbar_confirmation`, `sr_filter_atr`, `fractal_confirmation`, `inside_bar_confirmation`, `engulf_confirmation`, `any_pa_confirmation`, `pa_confluence_min`
  - 新增指标: Pinbar/Fractal/InsideBar/Engulfing, Swing High/Low, PA共振计数
- **R12 深水区探索完成** (8,457s): 6方向探索
  - 新增引擎参数: `squeeze_filter`, `consecutive_outside_bars`, `partial_tp_atr`, `profit_drawdown_pct`, `adaptive_max_hold`
  - 新增指标: BB (布林带), Squeeze 检测
- **R13 Alpha 淬炼设计+部署**: 7 Phase, ~18 实验
  - 新增引擎参数: `kc_ema_override`, `kc_mult_override`, `dual_kc_mode/fast/slow`, `kc_ma_type`, `purge_embargo_bars`
  - 新增指标: `_hma()`, `_kama()`, `add_dual_kc()`
  - 脚本: `run_round13.py`, `deploy_r13.py`, `_check_r13_on_d.py`
- **R15 TATrail 深化验证设计+部署**
  - TATrail 参数: `time_adaptive_trail_start/decay/floor`
  - 脚本: `run_round15.py`, `deploy_r15.py`, `_check_r15.py`

## 2026-04-14
- **R8 L6全面验证启动**: TP×SL Grid + L6+EntryGap 完成
  - R8-1: L6全面优于L5.1, K-Fold 6/6
  - R8-2: Gap=1h Sharpe 7.18→7.35, K-Fold 6/6

## 2026-04-13
- **R7 L5.1验证+L6发现**: 双服务器交叉确认
  - R7-1: L5.1基准 Sharpe=6.17, K-Fold 6/6, WF 11/11
  - R7-2: Entry Gap 1h +0.11, K-Fold 6/6
  - R7-3: L6 Sharpe=7.15(+0.98), K-Fold 6/6, MaxDD -25%
  - R7-4: MC 80次±15%, 100%盈利, min=5.49
- **R5C+R6A 完成**: MC 100% robust, L5.1部署参数验证
- **R6B-B1**: L6 12/12年优于L5.1, +$5,648

## 2026-04-12
- **R2/R3/R4 全部完成** (9.3小时): L5组合验证, OOS 11/11年盈利, 破产率0%

## 2026-04-10
- **数据更新**: BID/ASK 更新到 2026-04-09, Spread 时序构建
- **引擎新增 `historical` spread 模型**
- **EXP-U/K/W 完成**: KC mid revert 6/6→审计发现look-ahead→否决; Tight_all 6/6; 亏损画像

## 2026-04-09
- **🚨 回测引擎重大修复**: H1 look-ahead bias + 入场价 look-ahead
  - `_get_h1_window(closed_only=True)`, pending signal 队列
  - 修复前 Sharpe=5.06 → 修复后 3.18 (-37%)
- **回测/实盘路径对齐**: 新增 `LIVE_PARITY_KWARGS`, `live_atr_percentile`, Sharpe ddof=1
- **T7 OnlyHigh 验证通过**: K-Fold 6/6 ($0.30+$0.50)

## 2026-04-08
- 修复 IntradayTrendMeter 索引 bug (choppy 之前未生效)
- 修复 monkey-patch 信号注入 bug

## 2026-04-05
- 引擎参数: KC EMA 25, Mult 1.2, SL 4.5, TP 8.0, Cooldown 30min
- V3 ATR Regime 启用 (K-Fold 6/6)
- Mega Grid Search 1440 组合完成

## 2026-04-04
- backtest/ 统一回测包重构 (engine.py + stats.py + runner.py)

## 2026-04-03
- 追踪止盈系统化: C12冠军 Trail0.8/0.25+SL3.5+TP5+ADX18
- 过拟合检测 4项全通过 (PBO/PSR/DSR/WF)
