# 研究系统变更记录 (Changelog)

> **读取频率: 需要时查阅**
> 仅记录回测引擎、实验脚本、研究工具的变更。实盘变更见 `gold-quant-trading` 仓库。

---

## 2026-05-13

- **R204 Keltner Trail M1-Resolution Replay**: `experiments/run_r204_keltner_m1_replay.py`
  - 用 242MB Dukascopy M1 数据（比 M15 高 15 倍分辨率）重放 R202 winner-D/F trail 参数。每个 Keltner trade 拿 entry_time/price/ATR，在 M1 bar 上完整重放 SL/TP/Trail/Timeout 出场逻辑
  - 3 配置对比: baseline (`ta=0.14/td=0.025`) vs D (`ta=0.06/td=0.015`) vs F (`ta=0.06/td=0.008`)
  - 输出: PnL/Win rate/Sharpe estimate + Hair-trigger 率(≤15/≤30 M1 bars) + Trail 激活 bar 分布
  - 适配 Dukascopy 格式 (`Gmt time` 列、`dd.mm.yyyy HH:MM:SS.fff` 时间格式)
  - 用途: 验证 `td=0.008` 在更高分辨率下是否仍稳健，决定是否冒险部署 F vs 保守选 D

- **R203/R203b TSMOM 实盘 0 触发深度诊断**: `experiments/run_r203_tsmom_signal_diag.py`, `experiments/run_r203b_tsmom_filter_cascade.py`
  - **R203**: 历史信号频率分析 — TSMOM crossover 在 11 年里出现 1405 次, **平均 10-11 信号/月**, 完全不是"信号稀缺"
  - **R203b**: 完整模拟 EA 过滤器级联 — 2025-01 至今 210 个 crossover, **113 笔应该已经入场（53.8% 通过率），最近 60 天 9 笔**, 但实盘是 0 笔
  - **根因发现 — TSMOM EA silent-failure bug**:
    - `TSMOM_H1_EA.mq4` 第 82/100 行: `if(Bars < Slow_Lookback + 2) return 0`
    - 如果 MT4 H1 图表加载 bar < 722（30 个交易日），`MomentumScore()` 永远返回 0，crossover 永不发生
    - **EA 无任何错误日志**，silent 失败
  - **修复版**: `deploy/TSMOM_H1_EA_v1.20.mq4`
    - `EnsureHistoryLoaded()`: OnInit 时主动请求历史数据，不足则 `INIT_FAILED` + Alert
    - 每 24 H1 bars 输出 `STATE` 诊断（score/prev/atr/rule_b_skip/position 状态）
    - 信号触发输出 `SIGNAL` 日志
    - OrderSend 失败输出 `GetLastError` 码

- **R202 Keltner Regime-Config Optimization**: `experiments/run_r202_regime_optim.py`
  - 测试 8 种 `regime_config` 配置(含 regime ON/OFF, 全局/分 regime tightening), 三关验证 Top-2
  - **结果**: 所有 7 种替代配置都优于 baseline (`ta=0.14/td=0.025`, Sharpe 8.741)
  - **冠军 F** (`ta=0.06/td=0.008`, regime OFF): Sharpe **9.451** (+0.71), MaxDD 187 (-24%), 胜率 91.27%
    - K-Fold 6/6, WF **19/19**, Era 4/4 PASS → **GO**
    - Bootstrap CI [8.84, 10.03] 下界远超 baseline 上界 9.32
  - **亚军 D** (`ta=0.06/td=0.015`, regime 全 tight): Sharpe **9.391** (+0.65), MaxDD 187 (-24%)
    - K-Fold 6/6, WF 19/19, Era 4/4 PASS → **GO**
    - trail_dist 是 F 的近 2 倍，对 tick noise 更鲁棒，**保守部署首选**
  - 全部 4 个时代 (Pre-COVID/COVID/Tightening/Recent) 均正提升
  - 关键发现: regime adaptation 在统一紧 trail 下几乎无差异(D vs A), 真正的 alpha 在 "所有 regime 都用更紧的 trail"
  - 结果: `results/r202_regime_optim/R202_regime_optim.json`

- **R201 Keltner Tight-Trail Validation (regime_config 覆盖 bug 发现)**: `experiments/run_r201_keltner_tight_trail_validation.py`
  - 验证 D2 top Keltner 候选 (`ta=0.06/td=0.015`) 的 3-Gate
  - **关键发现 — regime_config 覆盖 bug**: 当 `regime_config=ON`，BacktestEngine 内的 regime 判断会**覆盖**显式传入的 `trailing_activate_atr`/`trailing_distance_atr`，导致所有 ta/td sweep 结果完全相同
  - 这解释了 R200 D2 Keltner 175 点中 km 维度完全无效（5 个 km 值产生完全相同的 Sharpe）
  - 修复: R200 D2/D3 Keltner variants 显式设置 `v['regime_config'] = None`
  - 触发了 R202 的 regime-config 专项优化

- **R200 Mega Research — 关键发现总结**:
  - **A3b: H1 vs M15 解析度交叉验证** ★最有价值的输出★
    - 4 策略 (PSAR/TSMOM/SESS_BO/Chandelier) **bias_flags 全 True**
    - Hair-trigger rate **95-99%**: trail exit 绝大多数发生在入场 bar 或紧邻 bar
    - H1 vs M15 Sharpe gap 22-37%: chandelier +22%, psar +29%, tsmom +37%
    - 越紧 trail 偏差越大 → 完美解释为何参数扫描总推荐最紧 trail
    - **结论**: 任何 H1 + tight trail 的回测都系统性高估 Sharpe 1-2 个单位
  - **TSMOM "+8.22 Sharpe" 不是真改进**:
    - A1 报告的是绝对 Sharpe 而非 delta，原 baseline 在 grid 之外
    - TSMOM 所有 25 个参数点 Sharpe 都在 5.79-7.45 之间，N 集中在 938-978
    - "+8.22" 等于 0 trades baseline → 998 trades 的 delta，非真实改进
  - **Keltner 的 km 参数完全无效**: 5 个 km 值产生**完全相同**的 Sharpe（到小数点后 6 位），D2 实际是 2D (ta×td) 而非 3D
  - **Keltner 真实改进梯度**: 25 真实点中 ta=0.06/td=0.008 最优（+0.91 Sharpe），梯度单调，非孤立尖峰
  - **B 系列零增量**: B1 初次跑时 dxy/gvz/us10y 全 False（缺数据），B3/B4 高 Sharpe 是 regime stats artifact
  - **E2 实盘对账暴露真问题**: Keltner 占实盘 74% 流量、贡献 90% 利润，但 R200 火力都在 tsmom/psar/chandelier/sess_bo 这些实盘几乎不触发的策略
  - 部署前置条件: 任何 GO 候选实盘样本 ≥20 笔；TSMOM 0 笔(EA bug)、psar/chandelier ≤3 笔 → 全部不可部署

- **R200 B1 Rerun (宏观数据完整上传)**: 已重新启动远程跑
  - 上传 17 个 macro CSV 到 `data/external/` (DXY/GVZ/VIX/US10Y/SPX/Copper/WTI/HYG/Real Yield + GSR)
  - 修复 SFTP 中断导致的 `run_r200_mega_research.py` 0 字节问题（原子化上传 + size 校验）
  - 修复远程 python 路径（`/usr/bin/python3` 而非 `/root/miniconda3/bin/python`，后者无 pandas）
  - 远程 B1 现已加载 10 个 macro 源: `['dxy', 'GVZ', 'vix', 'us10y', 'SPX', 'copper', 'crude_wti', 'hyg', 'real_yield', 'gsr']`
  - Baseline Sharpe: Keltner 8.79, PSAR 6.04, TSMOM 6.60

- **TSMOM EA v1.20 修复版**: `deploy/TSMOM_H1_EA_v1.20.mq4` — 部署建议
  - 替换原 `TSMOM_H1_EA.mq4`
  - 新增 history-load 检查 + 信号诊断日志（每 24 H1 bars 输出 STATE）
  - 实盘部署后可通过 Experts 日志直接看到 score/atr/rule_b 状态

---

## 2026-05-10

- **R196 — 100-Hour Deep Strategy Parameter Exploration**: `experiments/run_r196_100h_params.py`
  - **Phase 1-6: 全策略核心参数 Sweep**
    - Keltner: kc_mult=[0.8~2.5], ema_span=[10~50], adx_min=[0~30], trend_ema=[50~200]
      - 全样本最优: kc=1.5, ema=30 → Sharpe 7.536 (baseline 7.191)
    - PSAR: af_step=[0.005~0.03], af_max=[0.03~0.20]
      - 全样本最优: step=0.03, max=0.10 → Sharpe 7.550 (baseline 7.155)
    - TSMOM: fast_lb=[120~720], slow_lb=[360~1200]
      - 全样本最优: fast=480, slow=1200 → Sharpe 6.658 (baseline 6.388)
    - Dual Thrust: k_factor=[0.3~1.0], nb=[3~20]
      - 全样本最优: k=0.4, nb=10 → Sharpe 7.717 (baseline 6.997)
    - Chandelier: atr_period=[10~35], mult=[2~5], ema=[50~200]
      - 全样本最优: atr_p=14 → Sharpe 6.700 (baseline 6.320)
    - Session BO: entry_hour=[8~16], lb=[2~12]
      - 全样本最优: hour=13, lb=3 → Sharpe 7.636 (baseline 7.570)
  - **Phase 7: Cap Optimization**
    - L8_MAX: cap=70 是 Sharpe 最优 (7.191), cap过高(>100)降低 Sharpe
    - PSAR: cap=OFF 最佳(7.557), 但 cap=60 已经很好(7.155), 去掉 cap 增加 maxDD
    - SESS_BO: cap越高越好, cap=OFF 达 8.010, 但当前 cap=60(7.570) 保守合理
    - DT: cap=100 最优(7.332), 当前 cap=18 过紧限制了收益
    - CHANDELIER: cap=50 最优(6.440), 当前 cap=25(6.320) 偏保守
  - **Phase 8: Intraday Patterns**
    - Range autocorrelation: lag-1=0.30, lag-24=0.14 → 强波动聚集效应
    - Momentum persistence: 0.608 (>0.5 → H1级别有动量延续)
    - Spread/Range 最差小时: Hour 21 (0.672!), Hour 23 (0.156), Hour 22 (0.196)
  - **Phase 9: Trade Management**
    - Post-loss cooldown: 全部降低 Sharpe → 不应用
    - Streak reversal: 无效 (streak logic 对 Keltner 不适用)
    - Hold duration: bars≤1 的 WR=92.8% → Keltner 极快出场是正常的
  - **Phase 10: Stress Test**
    - Worst day: -$574 (2025-08-26), Worst week: -$78, Worst month: +$201 (无负月!)
    - VaR95: -$27/day, VaR99: -$88/day, CVaR95: -$68, CVaR99: -$136
    - Monte Carlo MaxDD: mean=$589, 95th=$660, 99th=$733
    - Ruin probability (20%/30%/50%): 全部 0.00%
    - Calmar ratio: 309.64

- **R196b — K-Fold + Walk-Forward + Era Validation**: `experiments/run_r196b_validation.py`
  - **结论: 全部 6 个参数变化均 NO-GO**
  - Keltner kc=1.5 ema=30: KF 3/6, WF 7/19, Era FAIL → **NO-GO**
  - PSAR step=0.03 max=0.10: KF 4/6, WF 12/19, Era PASS → **NO-GO** (WF不足)
  - TSMOM fast=480 slow=1200: KF 3/6, WF 4/19, Era FAIL → **NO-GO**
  - DualThrust k=0.4 nb=10: KF 2/6, WF 12/19, Era FAIL → **NO-GO**
  - Chandelier atr_p=14: KF 4/6, WF 15/19, Era FAIL (cut_covid -1.448) → **NO-GO**
  - SessionBO hour=13 lb=3: KF 3/6, WF 9/19, Era FAIL → **NO-GO**
  - **核心结论: 当前参数配置已是 robust 最优, 单纯参数调整无法通过时间稳定性验证**

- **R196c — New Alpha Sources & Exit Optimization**: `experiments/run_r196c_new_alpha.py`
  - **Phase 1 (Dynamic SL/TP by ATR pctl)**: 效果微弱, 最佳仅 +0.034 → 不值得
  - **Phase 2 (Time-decay TP/Ratchet/Break-even)**: 对 Keltner 几乎无影响(max_hold=2, 大部分交易1-2bar结束)
  - **Phase 3 (ATR Spike Entry)**: 高 Sharpe 但 N<300, 样本不足不可靠
  - **Phase 4 (SL/TP Ratio Grid)**: SL=6~8区间最优, TP对Keltner无影响(trail先触发)
  - **Phase 5 (Trail Act/Dist Fine Sweep)**: ★★★ 重大发现 ★★★
    - ta=0.02, td=0.005 → Sharpe **7.751** (+0.560), MaxDD $135 (vs $211)
    - ta=0.02, td=0.008 → Sharpe 7.730 (+0.539)
    - ta=0.03, td=0.005 → Sharpe 7.578 (+0.387)
    - 趋势: trail_act 越小(越早激活) → 越好; trail_dist 越小(越紧跟) → 越好
  - **Phase 6 (Calendar)**: 
    - 周四最强(8.618), 周五最弱(5.958)
    - Skip Friday: +0.206 Sharpe
    - 月份: Jul(10.25), Sep(10.44) 最强; Jan(5.41) 最弱
  - **Phase 7 (Max Hold Sweep)**: mh=2 接近最优, mh=3/8 微略好但差异 <0.05
  - **Phase 8 (RSI/ADX Filter)**: RSI>50 +0.209; 高ADX过滤反而降低 (-0.74~-1.10)
  - **Phase 9 (Portfolio Exit Combos)**: 所有enhancement影响 <0.003 → 不值得
  - **Phase 10: 最终 3-Gate Validation** ★★★
    - **Skip Hours {1,20,22,23}**: KF **6/6**, WF **16/19**, Era **全正** → **GO**
      - Era details: pre_hike +0.37, hike +0.52, cut_covid +0.17, hike_2022 +0.47, recent +0.08
    - **Trail ta=0.02, td=0.005**: KF **6/6**, WF **19/19** (完美!), Era **全正** → **GO**
      - Era details: pre_hike +1.13, hike +0.70, cut_covid +0.70, hike_2022 +1.21, recent +0.67
    - **这是目前唯一通过全套严格验证的可部署优化**

---

## 2026-05-09

- **R192 R191 Findings Rigorous Validation**: `experiments/run_r192_rigorous_validation.py`. 对R191的每个发现做隔离变量+K-Fold+WF+Era三关验证, 纠正R191多个方法论缺陷:
  - **Phase 1 Baseline**: A(live)=7.120, B(live+R187)=8.299 — 确认与R190一致
  - **Phase 2 Trail 0.06/0.01 隔离验证**:
    - L8_MAX: **GO** (KF 6/6, WF 19/19, 全era正)
    - PSAR: **GO** (KF 4/6, WF 14/19, 全era正)
    - TSMOM: **NO-GO** (KF 5/6但WF仅9/19 — WF FAIL)
    - SESS_BO: **GO** (KF 6/6, WF 14/19, 全era正, 低ATR改善最大+2.489)
    - DUAL_THRUST: **GO** (KF 6/6, WF 19/19 — 最强通过)
    - CHANDELIER: **GO** (KF 6/6, WF 13/19, 全era正)
    - 结论: 5/6策略通过, 仅TSMOM WF不足. **R191称"6策略全收敛"是错误的**
  - **Phase 3 SL=6.0 隔离验证**:
    - L8_MAX: **GO** (KF 6/6, WF 16/19) — SL从2.6%降至1.2%
    - PSAR: **GO** (KF 6/6, WF 14/19) — SL从6.6%降至3.8%
    - TSMOM: **CAP-DOMINATED** — SL<2%触发, 改SL无意义
    - SESS_BO: **NO-GO** (KF 3/6, WF 6/19) — 全量好看但折叠不稳定
    - DUAL_THRUST: **GO** (KF 5/6, WF 14/19)
    - CHANDELIER: **NO-GO** (KF 4/6但WF仅10/19)
    - SESS_BO TP=4.0 bug: 确认TP<SL, 但TP=6.0改善KF仅1/6 **NO-GO** — 当前TP=4.0虽低但不伤害Sharpe
    - 结论: **R191称"SL=6.0通用最优"是错误的**, 仅3/6策略通过
    - **关键发现: Cap/SL交叉分析** — 每策略在不同ATR下SL和Cap的binding切换点:
      - 低ATR: SL binds (SL先触发) → SL值有影响
      - 高ATR: Cap binds (Cap先触发) → SL值无关
      - DT的Cap=$18最紧, 中位ATR以上已Cap-dominated(Cap% 9.7~26%)
  - **Phase 4 Max Hold — 与R182一致性确认**:
    - Keltner MH3: **NO-GO** (KF 3/6, WF 11/19) — 与R182一致
    - Keltner MH4: **NO-GO** (KF 3/6, WF 12/19)
    - Keltner MH5: **NO-GO** (KF 4/6但WF 12/19) — **与R182(KF 2/6)不完全一致但结论相同**
    - Keltner MH6: **NO-GO** (KF 5/6但WF 12/19)
    - Keltner MH8: **NO-GO** (KF 5/6但WF 12/19)
    - Keltner MH10: **NO-GO** (KF 4/6但WF 11/19)
    - **全部MH值WF均FAIL** — Keltner MH=2是正确的. **R191称MH=10最优是错误的**
    - 其他策略: TSMOM(7.6%TO)和Chandelier(11.5%TO)增加MH无改善
  - **Phase 5 Chandelier带MAX_POSITIONS模拟**:
    - MAX_POS=4: 6策略=7.122, 5策略(去CH)=7.169 (+0.047)
    - MAX_POS=5: delta=+0.050, MAX_POS=6: delta=+0.049
    - **与位限无关, CH本身就是净拖累**
    - KF 5/6支持去掉CH, 48.4%的CH信号与其他策略重叠
    - CH在回撤期平均PnL=-$2.18, 不提供避险价值
  - **Phase 6 Cap/SL Regime Diagnostic**:
    - 低ATR(<Q25=0.63): SL binds全部策略 — SL改动在此区间有效
    - 中ATR(Q25-Q75): 混合 — 小cap策略已Cap-dominated
    - 高ATR(>=Q75=4.48): Cap binds大多策略 — SL改动在此区间无效
    - DT的Cap=$18在Q50以上就Cap-dominated(Cap%=9.7%), 问题最严重
  - **Phase 7 Combined Optimal (仅GO项)**:
    - GO项: Trail(5策略) + SL=6.0(L8/PSAR/DT) = 改动5策略共8个参数
    - D(combined+R187): Sharpe **8.610** vs B(live+R187): 8.299 (+0.312)
    - KF: **5/6 PASS**, WF: **19/19 PASS**, Era: 全正
    - 2015-2026每年都正且大多高于B
    - **VERDICT: GO** — 但改善幅度(+0.312)远小于R191声称的(+1.365), 因为只含验证通过的改动
  - 结果: `results/r192_validation/`

- **R191 10-Hour Mega Test Suite**: `experiments/run_r191_mega_test.py`. 12批次广泛探索, 发现多个潜在优化方向, 但部分结论存在方法论缺陷:
  - Batch 1确认live config一致性, Batch 2 Session ADX影响极小, Batch 3 ML R173无alpha
  - Batch 4固定Cap优于动态Cap, Batch 6 TSMOM贡献正面
  - **方法论缺陷(已被R192纠正)**: Batch 8 trail无K-Fold/Era验证, Batch 9 SL未考虑Cap/SL交互, Batch 5 max_hold与R182矛盾未reconcile, Batch 7 portfolio无MAX_POSITIONS模拟, Batch 12多参数同时改动无隔离验证
  - 结果: `results/r191_mega/`

- **R190 Selective Lot Reduction**: `experiments/run_r190_selective_lot.py`. 测试5个手数配置(从只降TSMOM/SESS到全降):
  - 5个配置: A=现有, B=只降TSMOM+SESS(0.04), C=+降CH(0.03), D=+降PSAR(0.04), E=全降
  - **Phase 1 Full Period**: 每降一档Sharpe都提升 — A 6.814 → B 6.929 → C 7.108 → D 7.195 → E 7.281
  - **Phase 2 Yearly**: 2024-26年PnL保留率 — B保留82-88%, C保留73-79%, D保留62-67%, E保留49-53%
  - **Phase 3 K-Fold**: B=5/6, **C=6/6**, **D=6/6**, E=5/6 — 全部PASS
  - **Phase 4 Walk-Forward**: B=17/19(89%), C=17/19(89%), **D=18/19(95%)**, E=15/19(79%) — 全部PASS, D最强
  - **Phase 5 Era**: 所有config所有era Sharpe均高于A_current
  - **Phase 6 Combined R187**: 配合ATR Pctl Floor全策略 — A→7.686, B→7.941, **C→8.234**, D→8.210, E→8.239
  - **Phase 7 Risk**: MaxLoss — A $490, B $490, C $385, D $258, E $184
  - **最佳平衡点: C_3worst** (降TSMOM 0.04, SESS 0.04, CH 0.03):
    - Sharpe 7.108 (+0.294 vs A), 配合R187达 **8.234** (+1.420 vs A)
    - PnL $178K (保留A的75%), 2024-26保留73-79%
    - K-Fold 6/6, WF 17/19, MaxLoss $385 (7.7% capital)
    - 近3年PnL $92K vs A $121K (76%), 但Sharpe 8.628 vs 8.275
  - **次选: D_4worst** (再降PSAR到0.04):
    - Sharpe 7.195, 配合R187达 8.210
    - PnL $147K (保留62%), WF 18/19(最高), MaxLoss $258 (5.2%)
  - 结果: `results/r190_selective_lot/r190_results.json`

- **R189 Adaptive Cap Validation**: `experiments/run_r189_adaptive_cap.py`. 验证方案C (Cap = cap_atr_mult × ATR × lot × PV) + 新手数:
  - **Phase 2 Sweep**: cap_atr_mult 1.0~4.0, 新手数(L8=0.02, PSAR=0.04, TSMOM=0.04, SESS=0.04, DT=0.02, CH=0.03)
    - Best by Sharpe: cap_atr_mult=4.0 (Sharpe 6.673), 但Fixed Cap (Sharpe 7.281)更高
    - **关键发现**: 固定Cap + 新手数 Sharpe=7.281 > 任何adaptive cap组合. 降低手数本身就解决了Cap/SL不匹配的大部分问题
  - **Phase 4 K-Fold**: ADAP wins **2/6** — FAIL
  - **Phase 5 Walk-Forward**: ADAP wins **7/19** — FAIL
  - **Phase 6 Era**: 所有era adaptive cap Sharpe均低于fixed cap (full -0.607, hike -0.340, cut -0.545, recent_3y -0.541)
  - **Phase 8 Risk**: 新手数 + adaptive cap最大单笔亏损$306 (6.1% capital), P99=$86, worst day -$207 — 风险可控
  - **Phase 9 Combined**: 
    - B (newlots+fixedcap): Sharpe **7.281**, Cap%=2.8%, MaxLoss=$184
    - D (newlots+adapcap+r187): Sharpe 7.460, Cap%=0.3%, MaxLoss=$342
    - E (curlots+r187_all): Sharpe **7.686**, PnL=$234K, MaxLoss=$769
  - **Verdict: ADAPTIVE CAP = NO-GO**. 理由:
    - K-Fold FAIL, WF FAIL, 全era Sharpe下降
    - Adaptive cap让亏损交易亏更多(MaxLoss从$184增至$306), 换来的PnL微增($120K→$128K)不值得
    - 降低手数本身(B配置)已经把Cap%从7.9%降至2.8%, Sharpe从6.814升至7.281
  - **修正建议**:
    1. **方案B(降手数+保留固定Cap)才是最优解**: Sharpe 7.281 > 现有6.814, Cap%仅2.8%
    2. 配合R187 ATR Pctl Floor全策略扩展: 预期Sharpe ~7.5+
    3. 固定Cap不需要改动, 只需把手数调低
  - 结果: `results/r189_adaptive_cap/r189_results.json`

- **R188 Lot/Cap/SL Mismatch Audit + R187 Full-Strategy Expansion**: `experiments/run_r188_lot_risk_audit.py`. 针对实盘全景审查发现的3大问题:
  - **Phase 1 Cap vs SL Mismatch Audit**: 在当前ATR≈20环境下, **全部6策略都存在Cap<<SL** (Cap/SL ratio: L8_MAX 25%, PSAR 8%, TSMOM 3%, SESS_BO 5%, DT 5%, Chandelier 3%) — 所有策略的实际止损都是Cap而非SL, R/R严重扭曲
    - 历史中位ATR=2.57时仅PSAR/TSMOM出现TIGHT, 说明手数是在低ATR环境标定的
  - **Phase 2 TSMOM Lot Sweep**: 0.01~0.15手sweep, 手数越小Cap%越低且Sharpe越高 (0.02 lot: Sharpe 6.690, Cap%=1.9%; 0.15 lot: Sharpe 6.193, Cap%=16.4%). 最优:0.05手 (Sharpe 6.490, Cap/SL=10%, 仍受Cap支配)
    - **关键发现**: 即使0.05手在ATR=20时SL=$600 >> Cap=$60, Cap/SL ratio仅10%, Cap仍先触发. 要在ATR=20时让SL≈Cap需要lot=0.005 → 不现实
  - **Phase 3 Risk-Targeted Recalibration**: 按公式 max_lot = Cap / (SL_mult × ATR × PV), ATR=20时6策略全部需降至0.01手 (最小可交易单位)
    - 这揭示了**根本问题**: 不是手数太大, 而是**Cap太低**. 当前Cap($18~$60)是低ATR时代设定的, ATR涨4倍但Cap没跟着调整
  - **Phase 4 Portfolio对比**: Current Sharpe=6.814 vs Recal(0.01) Sharpe=6.803 (几乎相同), 但PnL从$238K降至$47K. TSMOM的Cap%从16.4%降至0.0%, Sharpe从6.193升至6.915 (+0.722)
    - 年度稳定性: 所有年份Sharpe正, Recent years(2024-2026)recal版Sharpe更高 (11.0 vs 9.4 in 2024)
  - **Phase 5 R187 ATR Pctl Floor: 全策略扩展验证**:
    - Full period: No filter 6.814 → Keltner-only 6.939 → **All-strategy 7.686** (+0.872)
    - K-Fold: **ALL wins 6/6** (每个fold全策略都优于Keltner-only)
    - Walk-Forward: **ALL wins 19/19** (每个OOS窗口全策略都赢)
    - **Verdict: EXPAND_TO_ALL** — 压倒性证据支持R187 ATR Pctl Floor从Keltner扩展到全部6策略
  - **Phase 6 Combined Config**: Recal lots + Full-strategy R187 → Sharpe 7.636, Cap%=0.7%
    - 对比current live: Sharpe +0.822, Cap exits从7.9%降至0.7%
  - **Phase 7 Era Validation**: 4个era全部改善 (hike +0.859, cut +1.140, recent_3y +1.048)
  - **部署建议**:
    1. **立即**: R187 ATR Pctl Floor从Keltner-only扩展到全部6策略 (K-Fold 6/6, WF 19/19)
    2. **立即**: 根据当前ATR环境调整Cap或Lot. 两个方案:
       - A) 提高Cap: 让Cap与SL匹配 (e.g. Cap = SL_mult × ATR × lot × PV ≈ 当前lot × ATR20 × SL × 100)
       - B) 降低Lot: 已验证0.01-0.02手在高ATR时Sharpe更好, 但PnL绝对值大幅降低
    3. **讨论**: TSMOM 0.15手→Cap$60的扭曲最严重, 优先调整
  - 结果: `results/r188_lot_risk_audit/r188_results.json`

- **R187d ATR Pctl Floor: lb=200 Live Validation**: `experiments/run_r187d_lb200_validation.py`. 实盘data provider只有200根H1, 验证短lookback可行性:
  - Phase 16 多lookback sweep(100/150/200/300/500): **lb=300, pctl=30最优** (Sharpe 7.686, +0.872)
    - lb=200也有效(7.100, +0.286), lb=150更好(7.661 at pctl=35)
    - pctl=30在lb=200~500全部为sweet spot, 非常稳定
  - Phase 17 K-Fold: **6/6 PASS (100%)** at lb=300, pctl=30
  - Phase 18 Walk-Forward: **19/19 PASS (100%)** — 每个OOS都赢
  - Phase 19 ATR实现对比: TR_ATR vs HL_ATR相关性**0.9998**, pctl相关性0.9995, 一致率**99.9%** → 用哪种ATR都行
  - Phase 20 Per-strategy: 6策略全改善, Chandelier(+0.731)/PSAR(+0.783)/DT(+0.755)获益最大
  - **部署建议**: lb=300(~12.5 trading days), pctl_floor=30; 实盘需将bars_h1.json扩展到>=300根, 或用lb=200(效果略弱但可用)
  - 结果: `results/r187_live_stress_test/r187d_results.json`

- **R187c ATR Percentile Floor Validation**: `experiments/run_r187c_atr_pctl_validation.py`. R187b用绝对ATR($1.2)有Rule B同样的缺陷(随金价变化失效), 改用rolling percentile rank:
  - Phase 11 百分位Sweep: 3个lookback(500/1000/2000) x 8个阈值(0-40)
    - **最优: lookback=500, pctl_floor=30** → Sharpe 6.803→**7.602** (+0.799/+11.7%)
    - 单调性好: 0→25逐步上升, 30最优, 35开始拐头
    - 3个lookback结果一致: pctl_30在所有lookback下都是峰值(7.602/7.590/7.587)
  - Phase 12 K-Fold: **6/6 PASS (100%)** — 每个fold filter都赢
  - Phase 13 Walk-Forward: **19/19 PASS (100%)** — 每个OOS期都赢, 最小改善+0.257
  - Phase 14 Era: 4个era全部改善(full+0.788, hike+0.745, cut+0.907, recent_3y+0.853)
  - Phase 15 Yearly: **12/12年Sharpe均提升**, PnL微降但Sharpe全胜(低ATR交易本身WR差)
  - **VERDICT: GO** — K-Fold 100% + WF 100% = 史上最干净的验证结果
  - 实现: `atr_pctl_rank = rolling_rank(ATR, lookback=500)`, skip if pctl < 30
  - 远程服务器1.5min完成, 结果: `results/r187_live_stress_test/r187c_results.json`

- **R187b Era Segmented + ATR Floor Filter Sweep**: `experiments/run_r187b_era_atr_floor.py`. R187扩展:
  - Phase 9 Era分段: 4个时期Portfolio Sharpe全部正值 — **结构性edge确认**
    - Full=6.814, Hike=7.667, Cut=7.695, Recent3Y=8.275
    - Hike/Full=1.13, Cut/Full=1.13 — 非常平衡, 不依赖任何单一利率周期
    - 6策略在所有era均为正Sharpe, 最低Chandelier Hike=5.520
  - Phase 10 ATR Floor Sweep: 测试ATR下限过滤消除低波动亏损
    - **最优ATR floor=1.2**: Sharpe 6.814→**7.320** (+0.507/+7.4%), 仅减少2199笔交易(8.5%)
    - Chandelier获益最大(+0.583), PSAR(+0.371), DT(+0.418), SESS_BO几乎无影响(+0.002)
    - Era check: 4个era全部改善(hike+0.665最大)
    - Yearly: 12年中7年PnL持平/改善, 5年微降(<$200), Sharpe全部提升
    - **VERDICT: GO** — ATR floor=1.2是低风险高回报的改进, 建议部署

- **R187 Live Portfolio Stress Test & Fragility Audit**: `experiments/run_r187_live_stress_test.py`. 使用真实实盘lot/cap对6策略组合进行全面压力测试:
  - Phase 1 基线: Portfolio Sharpe=**6.814**, PnL=$238,536, MaxDD=$634 (12.7% of $5k), 12年全部盈利
    - 单策略: L8_MAX(5.455), PSAR(6.275), TSMOM(6.193), SESS_BO(6.223), DT(6.061), CHANDELIER(4.265)
  - Phase 2 回撤: Top DD=$634(2025-08-21, 恢复8天), 最长连亏6天(仅-$100), 单日最大亏损-$471(占资本9.4%)
  - Phase 3 ATR压力: Extreme ATR(Q90+)反而Sharpe最高(**8.140**), 但Low ATR(Q0-25)为负(-2.827)
    - Cap充足性: Chandelier **DANGER**(0.43x), PSAR/TSMOM/SESS_BO/DT均 **TIGHT**
  - Phase 4 相关性: 日损益平均相关0.173, 亏损日反降至-0.048 (**GOOD**), 最大并发亏损-$550
  - Phase 5 MC(300次): ±15%参数扰动Sharpe 5th=6.725, 95th=6.872(**极稳定**); 任意drop 1-2策略最低仍5.653
  - Phase 6 Alpha衰减: 0 RED, 3 YELLOW(L8_MAX/TSMOM/DT trend略负但recent>hist), 3 GREEN
  - Phase 8 Scorecard: **6 GREEN, 3 YELLOW, 0 RED → ADEQUATE**
    - YELLOW: 单日最大亏损9.4%(>5%), 最长连亏6天(>5), 3策略alpha trend微降
  - **VERDICT**: 系统不脆弱, 但需监控: (1) Chandelier Cap在高ATR时偏紧 (2) 低ATR环境表现差 (3) Alpha trend
  - 远程服务器12.2min完成, 结果: `results/r187_live_stress_test/r187_results.json`

- **R186 KCBW Replace-Not-Stack: Engine Filter Substitution**: `experiments/run_r186_kcbw_replace_filters.py`. R185发现KCBW在standalone有效但在Engine中负面, 测试用KCBW替换(而非叠加)现有过滤器:
  - Phase 1: 12种配置全量对比
    - **A_baseline** (choppy+rsi_adx+无KCBW): Sharpe=**4.297** — 最优
    - D3_kcbw3_replace_rsi: 4.141 (-0.156) — 最接近但仍不如
    - B_kcbw_stacked: 4.022 (-0.275) — 叠加反而更差
    - C_kcbw_replace_choppy: 3.337 (-0.960) — 关choppy最致命
    - E_kcbw_replace_both: 3.301 (-0.996) — 关两个filter最差
    - F_no_filters: 3.793 (-0.504) — 全关比替换好(因为没KCBW开销)
  - Phase 2 K-Fold: 所有候选FAIL (D3 2/6, B 1/6, D 1/6)
  - Phase 3 Era: baseline在hike/cut/recent_3y全面领先
  - Phase 4 Walk-Forward: 所有候选FAIL (D3 1/5, B 0/5, D 0/5)
  - Phase 5 Cost: spread $0.20-$1.00全部baseline领先, 仅$1.00时D3追平
  - Phase 6 Yearly: C 0/12年赢, D 2/12年赢, E 0/12年赢
  - **核心发现**: Choppy gate是最重要的过滤器(关掉Sharpe暴跌~1.0), KCBW的alpha已被choppy gate捕获
  - **VERDICT: NO-GO** — 维持现有配置不变
  - 部署: `deploy/_deploy_r186.py`, 远程208核服务器41.5min完成
  - 结果: `results/r186_kcbw_replace_filters/r186_results.json`

- **R185 KCBW Deep Validation**: `experiments/run_r185_kcbw_deep_validation.py`. R184发现KCBW Sharpe+16.6%, 部署前8-phase深度验证:
  - Phase 1 Lookback敏感度: lb=2~15全部>5.0, lb=5最优(5.850), lb=20(4.963)略低于基线
  - Phase 2 定义一致性: 3种KCBW实现(rolling_mean/lag_compare/rolling_min)全部优于基线5.015
  - Phase 3 **Engine交叉验证**: KCBW在全功能Engine中**反而降低Sharpe**(4.30→4.02, 4.00, 3.79) — **FAIL**
    - 原因: Engine已有choppy gate + RSI-ADX filter, KCBW功能重叠, 叠加=过度过滤
  - Phase 4 Walk-Forward: 5/5 OOS wins, avg delta=+0.516, OOS/IS ratio=1.04 — PASS
  - Phase 5 参数扰动MC: P(better)=85.4%, delta 5th=-0.105 — PASS
  - Phase 6 成本敏感度: 5/5 spread全正向, spread越高优势越大(+0.79~+1.11) — PASS
  - Phase 7 逐年稳定性: 8/12年正向, 优势不集中 — PASS
  - Phase 8 PBO: 0.000 (0/70 combinatorial splits) — PASS
  - **Overall: 5/6 tests passed, STRONG GO for standalone但Engine不适用**
  - 结果: `results/r185_kcbw_deep_validation/r185_results.json`
  - **后续R186确认**: KCBW无论叠加还是替换Engine过滤器均不如baseline → **最终决定: 不部署KCBW到Engine**

- **R184 Keltner Entry Filter Impact on R:R**: `experiments/run_r184_keltner_filters.py`. 全面测试入场过滤器对Keltner交易质量的影响:
  - Phase 1 Grid Scan: 6类过滤器共30+配置测试
    - ADX阈值(0-25): 影响极小, ADX14 Sharpe=5.015 vs ADX0 5.051, ADX25 4.653
    - Session: NY最强Sharpe=5.267(WR 82.7%), Asia最弱3.917, London+NY综合4.692
    - **KCBW带宽扩张**: Sharpe从5.015→5.850(+16.6%), WR从71.8%→80.4%, 交易减少39%
    - EMA: EMA200 Sharpe=5.380略好于EMA100=5.015, 但无EMA=5.107也不差
    - KC乘数: KC2.5 Sharpe=5.293 R=1.449(最高R:R), 但交易仅3898笔
    - 组合: ADX18+London+KCBW最高Sharpe=5.878但N=2226太少
  - Phase 2 Robustness: Top 5 候选 vs 基线 (6-Fold CV + 1000x MC)
    - **KCBW_ON**: K-Fold 5/6 PASS, MC 100% CI[+0.43,+1.75] PASS → **GO**
    - ADX14_KCBW / ADX18_KCBW: 同样PASS (实质等同KCBW_ON)
    - ADX18_London_KCBW: K-Fold 4/6 PASS, MC PASS → GO (但N=2226偏少)
    - EMA_200: K-Fold 2/6 FAIL, MC 67.3% FAIL → **NO-GO**
  - **核心结论**: KCBW过滤器是唯一稳健通过的改进, 值得部署到实盘
  - 部署: `deploy/deploy_r184.py`, 服务器18s完成

- **MT4 Log Excel审计**: 分析`log.xlsx`与系统`gold_trade_log.json`交叉验证:
  - Excel 5月数据52笔中47笔(90%)盈亏计算正确, 5笔有误(Row 145平仓价错误, 4笔小偏差)
  - 系统log实盘R:R: Keltner 0.468(102笔), M15_RSI 0.831(19笔), DT 1.231(9笔)
  - Keltner出场分布: Trail 78%, Timeout 8%, MaxLoss Cap 8%, KC中轨 6%
  - 实盘vs回测R:R差异(0.468 vs 1.121)主因: 102笔小样本+2026高波动期

- **R183 Keltner R:R Optimization**: `experiments/run_r183_keltner_rr.py`. 4-phase experiment addressing Keltner's R:R imbalance (SL=3.5xATR vs trailing captures 0.06-0.14xATR):
  - Phase 1 Grid: 收紧SL无效(SL 1.5->3.5 Sharpe单调递增), 放宽trailing激活严重降Sharpe(0.5xATR Act Sharpe 2.76), 放宽trail距离也降Sharpe, 延长MH小幅提Sharpe但恶化R:R
  - Phase 2 Top: MH8 Sharpe 5.277 > baseline 5.015, 但 R从1.121降到0.539
  - Phase 3 Robustness: MH8 K-Fold 5/6 PASS, MC P=80.2% MARGINAL(CI含0) -> CAUTION
  - Phase 4 Anatomy: Baseline R=1.121(avg_win=$6.79/avg_loss=$6.05), safety margin=24.7pp; MH8 R=0.539, safety margin=19.4pp
  - **核心发现**: 当前MH2配置实际上R:R=1.12(不是0.12)，24.7pp安全边际，是所有配置中最佳平衡
  - 部署: `deploy/deploy_r183.py`, 服务器17s完成

- **R182 Robustness Validation**: `experiments/run_r182_robustness.py`. 6-Fold CV + 1000x Monte Carlo Bootstrap + 6-strategy portfolio interaction for R181 proposed changes:
  - **Keltner MH2->MH5**: K-Fold 2/6 FAIL, MC P=87.4% CI含0 MARGINAL -> **NO-GO** (全样本好看但折叠不稳定)
  - **Chandelier ATR14->22**: K-Fold 3/6 FAIL, MC P=62.4% FAIL -> **NO-GO** (折叠方差大，优势不显著)
  - **Dual Thrust current-bar vs confirmed-bar**: K-Fold 5/6 PASS, MC P=98.4% CI[+0.09,+2.07] PASS -> **GO** (但实盘 confirmed-bar 出于执行安全保留)
  - Portfolio 组合测试: 4/4 eras PASS (Sharpe 6.967 vs 6.863)
  - 结论: 当前实盘配置已是最优平衡点，不建议改动
  - 部署: `deploy/deploy_r182.py`, 服务器21s完成

- **R181 Full Live Strategy Audit**: `experiments/run_r181_full_audit.py`. Comprehensive A/B test of all 6 live strategies comparing research baseline vs live config vs live+filters across 4 eras. 20 individual test configs + 3 portfolio variants. Key findings:
  - **Chandelier RSI>EMA100**: Sharpe 5.015 vs 3.869 (+1.146), confirms R176e/f RSI filter is superior
  - **PSAR skip hours**: Sharpe 7.075 vs 6.480 (+0.595), live {3,7,22} skip validated
  - **PSAR live params**: Sharpe 6.480 vs 4.913 (+1.567), new SL/TP/MH vastly better than old
  - **SESS_BO D1 EMA20**: Sharpe 6.905 vs 6.447 (+0.458), filter validated (halves trades, higher Sharpe)
  - **TSMOM live params**: Sharpe 6.933 vs 6.613 (+0.320), new params validated
  - **Keltner Session ADX**: Negligible impact (Sharpe +0.001), R178 has no measurable effect in backtest
  - **Keltner MH5>MH2**: Sharpe 5.239 vs 5.015 (+0.224), consider increasing from 2 to 5
  - **DT confirmed_bar**: Sharpe 4.510 vs 4.883 (-0.373), live crossover logic slightly worse
  - **Chandelier ATR14>ATR22**: Sharpe 4.328 vs 3.869 (+0.459), ATR14 for lines is better
  - Portfolio C (live+filters): Sharpe 6.863 vs A (research baseline) 6.403 (+0.460)
  - 部署: `deploy/deploy_r181.py`, 服务器34s完成
  - 结果: `results/r181_full_audit/`

## 2026-05-05

- **Cross-Asset Z-Score 回测 (完整三阶段)**: `gold-quant-trading/scripts/cross_asset_zscore_backtest.py`. 2y 1h Gold/Brent/US10Y, 24h 滚动 Z-Score. Phase 1: 7 种 Z 前瞻收益统计. Phase 2: 14 种信号带成本做多模拟. Phase 3: 5 种 KC+Z 仓位调节方案. **全面否决**: 纯 KC Sharpe=1.28 是最优, 所有 Z 变体均不超越. Z-Score 定位为 Dashboard 宏观参考. 详见 backtestArchive.md + constraints.md

## 2026-05-02

- **R90 Full External Data Integration — 服务器5 Phase全部完成**:
  - Phase A: 宏观Regime检测(9.3s) — Rule-Based最佳(ANOVA F=10.81, p=0.00002), 3 Regime
  - Phase B: 因子增强信号过滤(58min) — 720种组合, 仅TSMOM+COPPER_GOLD_RATIO(Q20)通过K-Fold
  - Phase C: ML方向预测(15s) — AUC≈0.53, 回测亏损, 纯方向预测对黄金无效
  - Phase D: ML Exit优化(97s) — TSMOM+XGBoost AUC=0.781(>R62基线0.76), Sharpe+49%; 所有策略过滤后Sharpe均提升
  - Phase E: 动态组合配置(134s) — Dynamic Sharpe 6.62 vs Static 6.37, K-Fold仅2/6, 推荐静态
  - 修复多个timezone/数据对齐bug(Phase B allowed_dates, C dropna, D tz-aware比较, E regime lookup)
  - 外部数据源: VIX, DXY, US10Y, GLD, COT, SPX, GVZ, HYG, TIPS, Fed Funds, US2Y, USDJPY, USDCNH, WTI, Copper, GLD Holdings, M2
  - 脚本: `experiments/run_r90_full.py` + `run_r90a~e_*.py`
  - 部署: `deploy/deploy_r90.py`, `_r90_check.py`, `_r90_download.py` 等
  - 结果: `results/r90_external_data/`

- **R91 Warsh Regime Analysis — 本地完成(142s)**:
  - 沃什三情景Regime分类: A(渐进正常化)66%, B(激进紧缩)15%, C(政治化宽松)19%
  - Taylor Rule偏差对比: 简单偏差与金价相关性+0.100 > 市场基础偏差-0.043
  - 政治化风险得分: 均值26.4, P75=32.8, 最大58.8
  - 黄金在Regime C最强: 年化+28.9%, Sharpe 1.45
  - TSMOM在非正常化Regime爆发: B=17.18, C=11.01
  - 脚本: `experiments/run_r91_warsh_regime.py`
  - 结果: `results/r91_warsh_regime/`

- **R88 Per-Strategy Cap Grid完成(本地)**:
  - 实盘4策略(L8_MAX 0.05, TSMOM 0.04, SESS_BO 0.02, PSAR 0.01)逐策略MaxLoss Cap网格搜索
  - 最优: L8_MAX=$35, PSAR=$5, TSMOM=NoCap, SESS_BO=$35
  - 脚本: `experiments/run_r88_cap_grid.py`

- **R89 Lot Size Optimizer完成(本地)**:
  - $5,000本金, 组合MaxDD<$1,000约束
  - 推荐: L8=0.02, PSAR=0.09, TSMOM=0.08, SESS_BO=0.08
  - Portfolio Sharpe=6.37, PnL=$125,689, MaxDD=$420
  - 脚本: `experiments/run_r89_lot_optimizer.py`

- **外部数据库扩展**:
  - 新增10个数据源: Copper, Crude WTI, Fed Funds, GLD Holdings, HYG, M2, Real Yield, US2Y, USDJPY, USDCNH
  - 更新download_external.py支持全部17个数据源
  - aligned_daily.csv整合所有外部数据为统一日频DataFrame

## 2026-05-01

- **新增 `docs/memory-bank/literature_notes.md` 知识沉淀文档**:
  - 石川博士「川流不息」专栏系统性梳理: 回测过拟合(PBO/CSCV)、金融数学基础(GARCH/布朗运动/BS/凯利)、趋势跟踪策略、资产配置
  - Clever Liu「量化交易系列」7篇完整学习: Alpha来源、三大经典策略、风险管理三层防御、多因子模型、五大认知陷阱、散户生存之道、策略公开悖论
  - Clever Liu TradingView 236策略回测: 低频策略优势验证(年交易>200次全部亏损转正)
  - Clever Liu Agentic AI因子挖掘: 统计弹性(Statistical Resilience)概念
  - 知识映射总表: 17项外部知识点 vs 我们项目实践的对照
  - 识别待深入方向: 市场适应机制、CVaR、策略拥挤度、半凯利动态仓位、因子分解归因

- **R69 P6 参数对账完成** (本地, 16.5min):
  - 背景: EA部署参数来自R56, R61在Cap$37下探索了不同PSAR/SESS_BO参数, 需确认哪组更优
  - 参数差异: PSAR SL 4.5→2.0, MH 20→80; SESS_BO LB 4→3, SL 4.5→3.0, TP 4.0→6.0
  - Path A (Portfolio Lot Grid + K-Fold):
    - R56 params: 10/10 K-Fold PASS, 最优KF mean=8.031
    - R61 params: 10/10 K-Fold PASS, 最优KF mean=7.813
  - Path B (Per-Strategy K-Fold + Walk-Forward):
    - PSAR: R56 KF mean=5.592 > R61 5.213; WF OOS avg 6.419 > 5.760
    - SESS_BO: R56 KF mean=7.877 > R61 6.381; WF OOS avg 9.467 > 6.997
  - **结论: P6的R56(EA)参数在所有维度上优于R61, 实盘无需更改**
  - R61参数变化(更小SL/更大MH/TP)是full-sample过拟合
  - 脚本: `experiments/run_r69_param_reconcile.py`
  - 结果: `results/r69_param_reconcile/r69_summary.txt`, `r69_results.json`

## 2026-04-30 (晚间)

- **R51 独立策略全参数暴力搜索完成** (Westd 208核, ~50min):
  - D1 KC: 57,600组合 → 56,720正Sharpe → Top 50 K-Fold **50/50 PASS**
    - 最优: E10/M2.5/ADX18/SL4.5/Trail0.2/0.05, Sharpe 18.38, KF=34.66
    - 注意: N=47, 100%WR, 交易频率低但Sharpe极高
  - H4 KC: 48,000组合 → 全部正Sharpe → Top 50 K-Fold **50/50 PASS**
    - 最优: E15/M2.5/ADX10/SL4.5/MH50/Trail0.2/0.04, Sharpe 5.12, KF=6.58
    - 1199笔交易, 85.7%WR, PnL $15K, MaxDD $337
  - PSAR: 6,000组合 → 全部正Sharpe → Top 50 K-Fold **50/50 PASS**
    - 最优: AF0.01/MX0.05/SL4.5/MH20/Trail0.2/0.04, Sharpe 4.13, KF=4.36
    - 3155笔交易, 79.2%WR, PnL $17K, MaxDD $311
  - **SuperTrend H1: 8,640组合, 0个正Sharpe — 完全否决**
  - 性能优化: ST/PSAR指标预计算避免worker重复计算, 从卡住→2min完成

- **R52 多策略Lot组合优化完成** (Westd, ~23min):
  - 4策略(L8_MAX+D1KC+H4KC+PSAR) × 9种lot × 4维 = 6,560组合
  - Grid Sharpe Top 30 → K-Fold **30/30 全PASS**
  - 最优组合: L8=0.02, D1=0.06, H4=0.02, PSAR=0.05, Total=0.15 lot
    - Sharpe=5.18, KF_mean=5.75, KF_min=3.79
    - PnL=$88,407, MaxDD=$330, PnL/lot=$589K
  - 关键发现: D1 KC是组合中lot占比最高的策略(0.06-0.10)

- 新增脚本: `experiments/run_round51_independent_grids.py`, `experiments/run_round52_lot_portfolio.py`
- 新增部署: `deploy/deploy_r51.py`, `deploy/deploy_r52.py`

## 2026-04-30

- **R50 L8 全参数暴力搜索完成** (服务器 Westd, 208核):
  - Layer 1: 5,100 核心参数网格, 7.2h → 最优 E25_M0.8_SL2.0 Sharpe=5.78
  - Layer 2: 43,200 叠加组合 (Top 20 × 7维叠加层)
  - Layer 3: Top 50 K-Fold 6-Fold → **0/50 通过**, Fold 3 全面崩溃
  - Layer 4: 跳过 (无候选通过 K-Fold)
  - **结论: 48,300 组合搜索未超越 L8_MAX (Sharpe 11.23), 确认 L8_MAX 为全局最优**
  - 结果归档: `results/round50_results/`
  - 脚本: `experiments/run_round50_brute_force.py`

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
