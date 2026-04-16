# 系统变更记录 (Changelog)

> **读取频率: 需要时查阅，不需每次读**

---

## 2026-04-16
- **Round 13 "Alpha 淬炼" 设计完成**:
  - 7 Phase, ~18 实验, 预计 20 小时
  - 回测引擎新增 7 个参数: `kc_ema_override`, `kc_mult_override`, `dual_kc_mode`, `kc_ma_type`, `gsr_filter_enabled`, `gsr_series`, `purge_embargo_bars`
  - runner.py 新增: `_hma()`, `_kama()`, `add_dual_kc()`, `DataBundle.load_custom` 支持 `kc_ma_type`
  - 脚本: `run_round13.py`, `deploy_r13.py`, `check_r13.py`
  - 新方向: KC 参数网格搜索、Breakeven Stop、多速度 KC、自适应 MA(HMA/KAMA)、滚动窗口、Purged WF
- **Round 11 全部完成, 结论: 全部否决**
  - 24 个实验, 8 Phase, K-Fold 全部 0/6
  - PA形态(Pinbar/Fractal/InsideBar/Engulfing)无论作为过滤器还是独立策略均无效
  - PA共振(Confluence)越多反而越差: ≥2 Sharpe=1.38, ≥3 Sharpe=-0.20
  - 已录入 `constraints.md` 永久否决
- **Round 12 全部完成 (8,457s), 关键发现**:
  - A1 时段分析: NY最优(Sharpe=4.53), OffHours最差(0.92) — 有价值认知
  - B/C/D 新过滤器/出场全部 K-Fold 0/6 否决 — Squeeze/连续突破/ProfitDD/AdaptiveHold
  - D6 出场画像: Trailing 主导利润(95.6%WR), Timeout是最大亏损源(-$24,032)
  - E1 因子IC: gold_mom(-0.020短期反转), kc_bw(+0.018正收益) 验证已有认知
  - F1/F5 行为画像: 快速交易(1-2bar)最赚钱(94.5%WR), CVaR(1%)=-$52.41
  - 结果已录入 `backtestArchive.md` 和 `constraints.md`
- **R12 原始启动记录**:
  - 6 个全新研究方向: 微观结构分析、Squeeze 波动率压缩、连续突破确认、出场策略前沿、跨资产因子、持仓行为分析
  - 回测引擎新增 8 个参数: `squeeze_filter`, `squeeze_lookback`, `consecutive_outside_bars`, `entry_session_tag`, `partial_tp_atr`, `profit_drawdown_pct`, `adaptive_max_hold`, `adaptive_max_hold_profit_bars`
  - 新增指标: `BB_upper/lower` (布林带), `squeeze` (BB inside KC 压缩检测)
  - 脚本: `scripts/experiments/run_round12.py`, `scripts/check_r12.py`, `scripts/deploy_r12.py`

## 2026-04-15
- **Round 11 Phase 6-8 扩展 (Price Action 全面验证)**:
  - 新增 K线形态: 顶底分型 (`top/bot_fractal`), 孕线 (`inside_bar_bull/bear`), 2B吞没 (`engulf_bull/bear`)
  - 新增指标: 日幅波动 (`daily_range_up/down/max`), PA共振计数 (`pa_bull/bear_count`)
  - 回测引擎新增 9 个参数: `fractal_confirmation`, `inside_bar_confirmation`, `engulf_confirmation`, `any_pa_confirmation`, `pa_confluence_min`, `daily_range_filter`, `fractal_sr_strategy`, `inside_bar_sr_strategy`, `engulf_sr_strategy`
  - 扩展实验: Phase 6 (R11-15~R11-18 新形态IC+过滤器), Phase 7 (R11-19~R11-21 日幅+共振), Phase 8 (R11-22~R11-24 独立策略+全局最优)
  - 模拟盘新增: P14_fractal_sr, P15_inside_bar_sr, P16_engulf_sr
  - 灵感来源: 张峻齐课程笔记 (顶底分型/孕线/2B/日幅$15规则/共振理论)
- **Round 11 Phase 1-5 原始框架**:
  - 新增 Pinbar 形态检测指标 (`pinbar_bull`, `pinbar_bear`) — 基于实体/影线比率量化定义
  - 新增 Swing High/Low 支撑阻力位 (`swing_high/low`, `nearest_resistance/support`, `dist_to_resistance/support`)
  - 回测引擎新增 4 个参数: `pinbar_confirmation` (入场确认), `sr_filter_atr` (S/R过滤), `pinbar_sr_strategy` (独立策略), `pinbar_sr_atr_zone` (S/R zone)
  - 实验脚本: `scripts/experiments/run_round11.py` (8 Phase, 24 实验)
  - 模拟盘: P12_pinbar_keltner (KC+Pinbar确认), P13_pinbar_sr (Pinbar+S/R独立策略)
  - 涉及文件: `strategies/signals.py`, `backtest/engine.py`, `backtest/runner.py`, `paper_trader.py`
  - 灵感来源: 张峻齐裸K价格行为交易方法论
- **舆情系统 v5 全面优化** — 7 项结构性修复:
  1. **breaking 误报修复** (`calendar_guard.py`): 从简单子串匹配改为 (关键词 + 上下文词) 组合匹配。"heartbreak" 类误报消除，只有同时含金融上下文词（rate, fed, war 等）才触发暂停
  2. **采集失败告警** (`sentiment_engine.py`): 连续 3 次（约 9 分钟）采集为空时通过 Telegram 发送告警通知，防止 4/8 全天零数据无人知晓
  3. **关键词词典 v5 重平衡** (`analyzer.py`): 描述性价格词（gold surge/rally/fall/drop 等）从 ±0.15~0.25 降至 ±0.05；新增 11 个利空驱动词（strong economy, job growth, payroll beat 等）；近零分数标题过滤（|avg|<0.02 丢弃）。总计 42 bull / 69 bear
  4. **跨资产宏观信号融合** (`sentiment_engine.py`): 新增 `_compute_macro_signal()` 基于 DXY/VIX/US10Y/Brent 变化率计算宏观方向。NLP 与宏观冲突时以宏观为准；一致时增强仓位系数 +0.1。VIX>30 或油价日变>10% 直接触发保护性调整
  5. **数据源扩展** (`news_collector.py`): 从 6 条 RSS 扩展到 12 条，新增 Reuters/CNBC/MarketWatch 源过滤的 Google News 查询、DXY/treasury 专题查询、GoldBroker 专业 RSS
  6. **反应速度提升** (`sentiment_engine.py`): 基础更新间隔从 300s 降至 180s；VIX>25 或极端宏观时自动降至 60s；新闻按发布时间降序排列确保 FinBERT 优先处理最新标题
  7. **direction_bias 观察模式** (`gold_trader.py`): 当信号方向与舆情偏好冲突时记录日志（含宏观信号详情），不实际过滤。开仓记录新增 sentiment_bias/macro_bias/macro_score 字段供后续 IC 分析
  - 涉及文件: `sentiment/calendar_guard.py`, `sentiment/sentiment_engine.py`, `sentiment/analyzer.py`, `sentiment/news_collector.py`, `gold_trader.py`

## 2026-04-14
- **🚨 BUG修复: PermissionError → 误判平仓**: `mt4_bridge._read_json()` 读取 positions.json 遇到文件锁 (PermissionError) 时静默返回 None → `get_positions()` 返回空列表 → `sync_positions()` 误判所有持仓已被MT4平仓 → 删除 tracking 数据 → 持仓变成 "unknown"，丢失策略/时间信息，Trailing 和时间止损全部失效
  - 实际案例: #16463889 keltner 做多，14:50 读文件 Permission denied → 误判平仓 → 追踪丢失 → 持仓 8+ 小时无法被系统管理
  - 修复1: `_read_json()` 遇 PermissionError 自动重试3次 (间隔0.3/0.6/0.9s)，仍失败则抛出异常而非返回 None
  - 修复2: `get_positions()` PermissionError 时抛出 IOError 而非返回空列表
  - 修复3: `sync_positions()` 捕获 IOError → 跳过本轮同步，保留现有 tracking 数据
  - 修复4: `_check_exits()` / `_check_entries()` / `_get_current_direction()` / `_can_add_position()` 全部加 IOError 防护
  - 涉及文件: `mt4_bridge.py`, `position_tracker.py`, `gold_trader.py`
- **Round 8 启动**: 25核服务器 (`westd.seetacloud.com:30367`) 运行 L6 全面验证 10 阶段
  - R8-1 TP x SL Grid 完成 (45min): L6 全面优于 L5.1，TP=7/8 差异极小，K-Fold 6/6
  - R8-2 L6+Entry Gap 完成 (12min): Gap=1h Sharpe 7.18→7.35, K-Fold 6/6 ($0.30+$0.50)
  - R8-3~R8-10 运行中
- **P10_l6_ultratight 加入模拟盘**: L6 UltraTight2 trailing 策略，替换表现最差的 P6_dxy_filtered (5笔WR=20% PnL=-$8.70)
- P6_dxy_filtered 停用 (enabled=False)
- P10 参数: SL=3.5x, TP=8x, MaxPos=1, MaxHold=5, regime trail low(0.30/0.06) normal(0.20/0.04) high(0.08/0.01)
- 目标: 积累 ≥20 笔实盘级 paper trade 验证 L6

## 2026-04-13
- **Telegram Token 安全迁移**: 明文硬编码 → `.env` 文件 + `python-dotenv`
  - 发现 GitHub 仓库曾为 public，Token 泄露导致第三方往 Telegram bot 发消息（品种 `GOLD#` / 总限亏 `$15000`）
  - 用户已将仓库改为 private，并通过 BotFather `/revoke` 重新生成 Token
  - `config.py` 改为 `os.environ.get()` 读取，`.env` 在 `.gitignore` 中
- **Round 7 实验启动 + R7-1~R7-4 完成**: 双服务器运行 L5.1 验证 + 探索
  - Server A: `run_round7.py` (16核), R7-1~R7-4 已完成, R7-5/R7-6 运行中
  - Server B: R6B + R7 并行运行 (10核 R7 + R6B)
  - **R7-1**: L5.1 基准 Sharpe=6.17($0.30), K-Fold 6/6, WF 11/11, 12/12年盈利
  - **R7-2**: Entry Gap 1h Sharpe=6.28(+0.11), K-Fold 6/6
  - **R7-3**: L6 Sharpe=7.15(+0.98 vs L5.1), K-Fold 6/6(双点差), WF 11/11, MaxDD $271→$215
  - **R7-4**: Monte Carlo 80次±15%, 80/80 全盈利, Sharpe均值6.14, min=5.49
- **R6B dotenv 修复**: Server B 的 `config.py` 新增 `python-dotenv` 依赖导致 B2~B6 全部失败
  - 修复: 远程 `pip install python-dotenv`, 清理失败结果, 重启 R6B
- **R6B 语法错误修复** (早): f-string 反斜杠问题 → 提取变量避免
- **L6 确认为最强候选版本**: UltraTight2 regime trail (低0.30/0.06, 正常0.20/0.04, 高0.08/0.01)
  - 部署需等待纪律窗口 (2026-04-27 或 30 笔交易)
- **L5.1 部署到实盘**: L5 特殊条款唯一一次微调
  - `strategies/signals.py`: ATR_SL_MULTIPLIER 4.5→3.5 (R6-A5 K-Fold 6/6, Sharpe +0.17, PnL +$4,699)
  - `config.py`: MAX_POSITIONS 2→1 (R6-A4 K-Fold 6/6, Sharpe +0.43, MaxDD -$72)
  - `backtest/runner.py`: LIVE_PARITY_KWARGS 同步 (sl_atr_mult=3.5, max_positions=1)
  - 从 L5.1 起严格执行部署纪律 (≥2周/30笔, paper trade ≥20笔, K-Fold 通过, 用户同意)
- **Round 5C + 6A 完成** (双服务器 ~3.5h)
  - R5-8: Monte Carlo 100次扰动, 100/100 Sharpe>4 (极其鲁棒)
  - R6-A1~A6: 历史点差/危机年份/成本梯度/MaxPos/SL/Cooldown 全验证
  - R6B: 因 f-string 语法错误未执行 (不影响部署)
- **部署节奏纪律写入 constraints.md**: 防止频繁部署，严格前置条件

## 2026-04-12
- **L5 部署到实盘**: L3 基础上 2 项改进
  - `gold_trader.py`: TDTP 关闭（注释掉 check_time_decay_tp 调用）
  - `exit_logic.py`: AllTight trail — high 0.20/0.03→0.12/0.02, normal 0.35/0.10→0.28/0.06, low 0.50/0.15→0.40/0.10
  - `config.py`: fallback trail 0.35/0.10→0.28/0.06
  - `backtest/runner.py`: LIVE_PARITY_KWARGS 同步更新（time_decay_tp=False, regime AllTight）
  - 验证: R3-1 AllTight K-Fold 6/6 PASS (avg delta +1.31), OOS 11/11年盈利, 破产率0%
- **Round 2/3/4 实验全部完成** (9.3小时, 零错误)
  - R2: L5组合验证 + KC参数扫描 + SL/TP微调 + Walk-Forward + 出场审计
  - R3: 三维Trail联合优化 + MaxHold + ORB + ADX + Cooldown + Robustness + 回撤分析
  - R4: OOS样本外验证 + 危机期 + 存活模拟 + 频率 + 因子衰减 + 滑点 + Regime切换 + 序列分析
- **根目录整理**: 194 .py → 16 核心文件，其余归档到 scripts/experiments/(88), scripts/server/(62), scripts/legacy/(28), output/(56)

## 2026-04-11
- **L3 部署到实盘**: 三项改进一次性部署
  - `config.py`: max_hold_bars 3→5, INTRADAY_TREND_THRESHOLD 0.35→0.50, TRAILING_ACTIVATE_ATR 0.5→0.35, TRAILING_DISTANCE_ATR 0.15→0.10
  - `exit_logic.py`: regime trail 全收紧 — high 0.25/0.05→0.20/0.03, normal 0.50/0.15→0.35/0.10, low 0.70/0.25→0.50/0.15
  - `backtest/runner.py`: LIVE_PARITY_KWARGS 同步更新, keltner_max_hold_m15 12→20, choppy_threshold 0.50
  - 验证: L3 组合 Sharpe=4.07 (vs L0=2.29), K-Fold 6/6 PASS
- **L4 (KC mid reversion exit) 永久否决**: look-ahead bias 审计确认 Sharpe 虚高 1.24, 修复后劣于 L3

## 2026-04-10

- **数据更新**: BID 数据更新到 2026-04-09（M15 +1,076 bars, H1 +384 bars），覆盖关税暴跌行情
- **ASK 数据全量下载**: M15 395,232 bars + H1 98,808 bars (Dukascopy, 2015-2026)
- **Spread 时序构建**: `xauusd-m15-spread-2015-01-01-2026-04-10.csv` — BID/ASK 差值，中位数 $0.33
- **引擎新增 `historical` spread 模型**: `spread_model="historical"` + `spread_series` 参数
  - 支持真实时变 spread 回测
  - Historical Spread 回测 Sharpe=2.80（vs Fixed $0.30 的 3.64），交易时实际均值 $0.62
- **runner.py 路径 fallback**: 自动检测新旧数据文件名，兼容远程服务器旧数据
- **EXP-L Bug 修复**: `run_exp_l_trend_weights.py` 第69行 `_calc_realtime_score` staticmethod 描述器恢复错误，导致 `TypeError: takes 1 positional argument but 2 were given`。修复为 `BacktestEngine.__dict__['_calc_realtime_score']` 保留原始 staticmethod
- **数据上传到服务器**: SFTP 上传 6 个 CSV（Spread M15/H1 + BID M15/H1 + ASK M15/H1）共 ~71 MB
- **组合测试脚本**: `run_exp_combo.py` — 5 层叠加测试（MaxHold20 + Choppy0.50 + Tight_all + KC mid revert）+ K-Fold + 压力测试
- **新完成实验结果**:
  - EXP-U KC Mid Reversion: K-Fold 6/6 PASS, Sharpe 2.29→3.42 (+49%)
  - EXP-K Tight_all Trail: K-Fold 6/6 PASS, Sharpe 2.62→3.62 (+38%)
  - EXP-W 亏损画像: bars_held 是最强亏损预测因子 (Cohen's d=-1.95)
  - EXP-R 基准: 与之前完全一致 (Sharpe=2.29), 11/12 年盈利

## 2026-04-09
- **T7 OnlyHigh 部署到实盘**: exit_logic.py high regime T0.4/D0.10 → T0.25/D0.05
  - K-Fold 验证: $0.30 下 6/6 全赢, $0.50 下 6/6 全赢
  - 全样本 $0.30: Sharpe 1.20→1.87 (+0.68)
  - 同步更新 paper_trader.py P7 策略 + LIVE_PARITY_KWARGS
- **回测/实盘路径对齐**: 全面审查发现 16 个不对齐点，修复了以下关键项:
  - 新增 `LIVE_PARITY_KWARGS` preset (runner.py) — 精确匹配实盘 config.py + exit_logic.py 参数
    - Trailing: Mega Trail (0.5/0.15 base, regime 0.7/0.25 → 0.5/0.15 → 0.4/0.10) 替代旧 C12 (0.8/0.25)
    - time_decay_tp=True, rsi_adx_filter=40, intraday_adaptive=True, keltner_max_hold_m15=12
  - 新增 `live_atr_percentile` 引擎选项 — 用实盘方式的 rolling-50 rank 替代预计算 rolling-500
  - 统一 `_get_atr_percentile()` 方法，消除引擎中 8 处重复的 atr_percentile 取值代码
  - Sharpe 计算 `ddof=0` → `ddof=1` (stats.py)
  - 新增数据缺口检测 `check_data_gaps()` (runner.py)
  - **未修复（记录为已知差异）**: time_decay 时间单位(bar数 vs wall-clock), H1评估时机(minute==0 vs 每次轮询), SL/TP同bar优先级
- **🚨 重大修复: H1 look-ahead bias + 入场价 look-ahead** (commit 7f02772)
  - `_get_h1_window(closed_only=True)`: 入场只用已收盘 H1
  - pending signal 队列: 入场价用下一根 M15 Open，不用当前 bar Close
  - 修复前 Current Sharpe=5.06 → 修复后 3.18（-37%）
  - 根因: H1 时间戳=开盘时间(Dukascopy), 引擎在14:00就用了15:00才有的Close
  - 13.8% 的 KC 突破信号是 phantom（只因看到未来 Close 才触发）
- 框架审计: 发现 7 个问题（3 严重 + 3 中等 + 1 ORB patch bug）
- 全量实验分析完成: Trail Momentum 1.5x K-Fold通过, T7 ExtremeRegime 待验证, 其余全部否决
- 新增 run_lookahead_fix_verify.py（修复后基准线）和 run_t7_extreme_validation.py
- 修复 run_t7_extreme_validation.py 中 stats key 名称错误（n_trades→n 等）
- 新增 entry quality filters 到 backtest engine (min_h1_bars_today, adx_gray_zone, escalating_cooldown) — 回测确认无效
- Memory Bank 迁移: trading_journal.md → docs/memory-bank/ 6 文件分离

## 2026-04-08
- 修复 backtest engine IntradayTrendMeter 索引 bug — choppy 门控之前在回测中从未生效
- 修复 monkey-patch 信号注入 bug — Strategy A/C/D 的信号注入之前无效
- EXP-MOM 短窗口动量旁路回测: 效果微弱(±0.01 Sharpe), 不上线

## 2026-04-07
- Strategy A/C/D 三策略 596 种组合回测完成, 全部无法超越基线
- EUR/USD Keltner 深度回测(11年), P9_eurusd_keltner 注册模拟盘
- P7_mega_trail, P8_mega_h20 注册模拟盘
- EXP36-47 实验完成(时段/分批止盈/RSI背离/D1方向/K线形态/ATR反波动率/宏观/波动率聚集/整数关口/隔夜收益)

## 2026-04-05
- **SL 4.5 ATR + Cooldown 30min 实装** (commit 379cb17)
- **日内亏损递减手数上限** — MAX_LOT_CAP_BY_LOSSES 替代旧 lot_scale
- **KC EMA 25 + KC mult 1.2 实装** (EXP13/EXP20)
- **V3 ATR Regime 启用** (EXP07 K-Fold 6/6 折全赢)
- **TP 8.0 ATR 实装** (EXP06)
- Mega Grid Search 1440 组合 + Walk-Forward 18 窗口验证
- EXP28 事件日防御全部否决
- Polymarket 监控 v2 (pizzint.watch NEH API)
- 旧引擎 7 个文件删除 (~3055 行)

## 2026-04-04
- backtest/ 统一回测包重构 (engine.py + stats.py + runner.py)
- 4 个核心实验脚本迁移至统一包
- P2 动态点差模型, P3 高级统计检验, P4 宏观Regime自动识别
- macro/ 宏观数据管道 (yfinance + FRED)
- 成本调整回测: $0.50 点差下当时仅 Combo 勉强存活
- 大趋势日识别研究: Oracle Sharpe 6.44 但技术指标无法预判
- 盘中自适应系统设计 + Phase 5 回测

## 2026-04-03
- 追踪止盈 Round 2 精调 (22 变体)
- R3 组合测试 — C12 冠军: Trail0.8/0.25 + SL3.5 + TP5 + ADX18, Sharpe 2.54
- 过拟合检测 4 项全通过
- 高级回测套件: 蒙特卡洛/K-Fold/Regime自适应/参数探索
- 组合验证: Adaptive Trail + KC1.25 + KC_EMA30 三合一通过
- 策略压力测试: **发现无成本回测的致命缺陷** (18,606笔交易成本 $25,043)

## 2026-04-02
- 架构大重构: gold_trader.py → data_provider + risk_manager + position_tracker + gold_trader
- IC 监控上线, 42 个单元测试
- Cursor Rule 创建, trading_journal.md 创建
- 首次全量因子 IC 扫描 (27因子×3窗口=81次检验)
- M15 RSI ADX>40 过滤实装
- 修复 CLOSE_DETECTED 重复 PnL bug
- 新增 backtest.py, factor_scanner.py, factor_deep_dive.py, backtest_ab_test.py
