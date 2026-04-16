# 约束与纪律 (Constraints)

> **读取频率: 每次对话必读，尤其在提出任何优化建议前**
> 优化纪律、踩过的坑、被否决的方向

---

## ⛔ 部署节奏纪律 (2026-04-13 制定，永久生效)

**此规则优先级最高。测试可以每天跑，但部署必须克制。**

### 部署前置条件（全部满足才能部署到实盘）
1. **当前实盘版本已运行 ≥ 2 周**，或已完成 ≥ 30 笔实盘交易
2. **新版本 K-Fold 通过**（≥ 5/6 folds）
3. **新版本在 paper trade 中观察 ≥ 20 笔交易**，且表现优于当前实盘版本
4. **用户明确同意部署**

### 发现改进时的正确流程
```
发现改进 → 记录到 backtestArchive → 加入 paper trade → 
等待 20+ 笔 → 对比实盘表现 → 确认优于当前 → 提议部署 → 用户同意 → 部署
```

### 特殊条款
- **L5 允许一次微调**：~~Round 6 结果出来后~~ → **已使用** (2026-04-13 L5.1 部署)
- **从 L5.1 开始**，严格执行上述部署前置条件，不再有例外
- L5.1 部署日期: 2026-04-13，下次部署最早: 2026-04-27 或 30 笔交易后

### 禁止行为
- 禁止在当前版本运行不满 2 周时提议部署新版本（紧急 bug 修复除外）
- 禁止跳过 paper trade 阶段直接部署
- 禁止以"回测更好"为唯一理由推动部署

### 手数/风险参数调整
- RISK_PER_TRADE 调整同样需要实盘验证 ≥ 2 周后才能进行
- 每次调整幅度不超过 50%（如 $50 → $75）

---

## ⛔ 优化提议纪律 (2026-04-09 制定，永久生效)

**此规则优先级最高。提出优化建议是好的，但必须诚实、严谨。**

### 1. 提议前 — 数据基础检查
- [ ] **明确标注**: 这是"假设"还是"已验证结论"？不允许模糊表述
- [ ] **查阅历史**: 该方向是否已有结论？如果已被否定过，必须说明为什么这次不同
- [ ] **样本量**: 支撑该提议的数据有多少笔？< 30 笔只能标"假设"
- [ ] **IC/因子一致性**: 是否与已有因子扫描结论矛盾？

### 2. 提议时 — 诚实表达
- [ ] 分开陈述"数据事实"和"我的推测"，不混为一谈
- [ ] 禁止伪确定性（未统计验证的观察禁用"通常"、"都是"、"一般"）
- [ ] 禁止倒果为因（正确: 观察→假设→验证→结论）

### 3. 实施前 — 强制回测
- [ ] **先回测后改代码**: 任何变更必须先有全量带成本回测数据证明有效
- [ ] **带成本测试**: 必须包含 $0.30-$0.50 点差
- [ ] **不动已验证参数**: 除非新数据**明确优于**现有参数

### 4. 强制输出格式
**每次提出任何策略/参数/过滤器优化建议时，必须先输出以下检查表。没有检查表的建议视为无效。**

```
┌─ 优化提议检查表 ─────────────────────────┐
│ 性质: 假设(待验证) / 已验证(回测#XX)       │
│ Journal历史: 是否有相关结论？→ 引用位置     │
│ 样本量: N = ??                             │
│ IC因子一致性: 是否矛盾？→ 说明             │
│ 回测状态: 未开始 / 进行中 / 已完成(结果)    │
│ 涉及已验证参数: 是(哪个) / 否              │
└─────────────────────────────────────────┘
```

### 5. 违反后果
- 违反任何一条，该提议自动作废
- 记录到"踩过的坑"

---

## 被否决的优化方向 — 禁止重复研究

> **硬约束**: 以下方向已经过回测验证确认无效或有害，**不再投入时间重复研究**。
> 如果某个新想法本质上是以下方向的变体，直接跳过。

### 入场过滤类
- **宏观 Regime 过滤** (3次确认): Sharpe 下降，策略在所有 regime 下都盈利
- **日前趋势预判**: 准确率 ~55% ≈ 抛硬币，趋势日由事件驱动
- **D1 日线方向过滤**: Against(逆D1) $/t=$3.45 > Aligned $2.86，突破往往在反转点
- **K线实体/影线过滤**: Low body $/t=$3.88 > High body $2.11，与直觉相反
- **KC Bandwidth 扩张过滤**: BW3-BW12 全部降低 Sharpe（-0.40 到 -1.60）
- **London Breakout**: 7 个变体全部负 Sharpe（-0.15 到 -0.39）
- **SELL ADX>=28**: 效果几乎为零，减少 309 笔交易样本
- **Strategy A 动量追击**: 180 组合，全部 Sharpe < baseline
- **Strategy C D1趋势过滤**: 32 组合，全部与基线一致
- **Strategy D 趋势回调入场**: 384 组合，全部与基线一致
- **EMA150 趋势过滤**: K-Fold Fold4 崩溃至 -13.56
- **EMA100 斜率过滤**: skipped=0，Keltner 自带过滤已覆盖
- **R11 Pinbar 过滤** (2026-04-16): K-Fold 0/6 ($0.30+$0.50)，Sharpe 6.17→2.04，过滤掉好交易
- **R11 顶底分型(Fractal)过滤**: K-Fold 0/6，Sharpe 6.17→3.04
- **R11 孕线(InsideBar)过滤**: K-Fold 0/6，Sharpe 6.17→1.65，$0.50 下 Sharpe=-0.02
- **R11 2B吞没(Engulfing)过滤**: K-Fold 0/6，Sharpe 6.17→2.17
- **R11 AnyPA(任意PA形态)过滤**: K-Fold 0/6 ($0.30+$0.50)，Sharpe 6.17→4.69
- **R11 PA共振(Confluence≥2)**: Sharpe 6.17→1.38，共振越多越差(Confluence≥3 Sharpe=-0.20)
- **R11 Daily Range 过滤($15/$20/$25)**: 全部降低 Sharpe (5.01~5.96 vs 6.17)，过滤掉高波动时的好交易
- **R11 PA+SR独立策略**: 所有配置 = 基线(S/R zone太窄，无额外信号触发)
- **R11 全局最优组合 AnyPA+SR1.5**: K-Fold 仅 2/6 ($0.30) 和 3/6 ($0.50) — FAIL
- **R12 Squeeze过滤(BB inside KC)**: K-Fold 0/6 ($0.30+$0.50)，Sharpe 6.17→3.87，过滤太多好信号
- **R12 连续突破确认(2-3 bar外通道)**: K-Fold 0/6，Sharpe 6.17→5.25(2bar)/4.75(3bar)

### 时间/时段过滤类
- **周一降仓**: Sharpe -0.07，2025-2026 周一反而是强日
- **跳过任何星期**: 每天都赚钱，跳过任何一天 Sharpe 均下降
- **时段过滤**: K-Fold 仅 2/6 折，所有时段都赚钱
- **缩短 ORB 持仓**: C12+Adaptive 下 ORB 默认最优
- **降低交易频率 (min gap 2-8h)**: 全部变差，好信号被跳过

### 仓位管理类
- **波动率过滤**: 跳过高波动 Sharpe 降到 0.79
- **ATR Regime 反波动率加权**: 12 年中 11 年降低 Sharpe
- **Trump/波动率 Sizing**: 所有方案效果在 ±0.05 以内
- **禁用 SELL**: PnL -$2,127，下跌年份 SELL 是唯一利润来源
- **连续亏损自适应减仓**: 连亏后下一笔期望仍为正

### 出场类
- **事件日防御 "带伞策略"**: EXP28 穷举 11 把伞×3 触发器×6 折，全部无效
- **RSI 背离提前出场**: 后验分析，非预测信号
- **ATR spike protection (真实引擎)**: 仅 +0.03 Sharpe，噪声级别
- **R12 利润回吐止盈(ProfitDD 30%~70%)**: K-Fold 0/6，Sharpe 6.17→5.89~6.06。Trailing已非常紧，利润回吐反而提前平掉好交易
- **R12 自适应MaxHold(无利润时缩短持仓)**: 对结果完全无影响(Sharpe/PnL=基线)，因为Trailing已在MaxHold前触发

### 策略类
- **Combo (KC1.25+EMA30+Adaptive Trail)**: 无成本 Sharpe 3.46, $0.50 仅 0.35
- **Keltner 均值回归 (H1)**: 盈亏比不足，15 个配置全负 Sharpe
- **波动率聚集→sizing**: 统计上显著但交易上无用
- **大波动后方向偏倚→sizing**: 日线反转效应与日内策略时间尺度不匹配
- **RSI 参数调整**: Adaptive 下 RSI 仅 6 笔/11 年，已自然消亡
- **Choppy 阈值调整**: range=0.000，完全无影响
- **kc_only=0.65**: 本质是关闭 M15 RSI，非渐进优化
- **点差优化/换交易商**: 短期不换，以当前点差为既定条件
- **R11 PA形态类全系否决 (2026-04-16)**: Pinbar/Fractal/InsideBar/Engulfing 作为独立策略或KC入场过滤器均无效。24个实验、6种组合方式、K-Fold全部0/6。结论: KC突破信号已饱和，任何基于K线形态的入场过滤只会减少交易数量而不改善质量

---

## 踩过的坑

### 技术类
- `_check_exits` 平仓后必须同时更新 `risk_manager` 的 `daily_pnl` 和 `cooldown`
- JSON 文件写入必须用原子操作（`tempfile` + `os.replace`）
- PowerShell 不支持 bash heredoc 语法
- **CLOSE_DETECTED 重复检测**: MT4 桥接文件读写竞争导致重复计 PnL，已加 `_ticket_already_closed()`
- **IntradayTrendMeter 索引 bug**: 回测引擎中 `_update_intraday_score` 用相对索引匹配绝对索引导致 choppy 从未触发（2026-04-08 修复）
- **monkey-patch 信号注入 bug**: `from strategies.signals import scan_all_signals` 创建本地绑定导致 monkey-patch 无效（2026-04-08 修复）
- **服务器测试最佳实践**: 独立 `.py` 脚本 + `sys.path.insert(0, ...)` + `python -u run_expXX.py 2>&1 | tee logs/expXX.log`

### 🚨 回测框架审计 (2026-04-09 发现)

**严重 1 — H1 Look-Ahead Bias（未来函数）**:
- Dukascopy H1 数据时间戳 = bar 开盘时间（已验证: Open[i] ≈ Close[i-1], avg diff 0.037）
- 引擎 `_get_h1_window` 在 M15 xx:00 时取 `floor('h')` 对应的 H1 bar → 取到的是当前小时尚未收盘的 bar
- 例: 14:00 时用了 14:00-15:00 这根 H1 的 Close=2998.49（Open=2988.12, 差 $10）
- **所有基于 H1 Close 的信号（KC 突破、ADX、EMA100）都提前 1 小时看到了未来**
- H1 信号占总交易 72%，**所有 Sharpe 数字系统性高估**
- **修复方案**: `_get_h1_window` 中 `h1_idx -= 1`（只用已收盘的 H1 bar）

**严重 2 — 入场价格用 bar Close**:
- `sig['close']` = 产生信号的 bar 收盘价 → 实盘不可能在 bar 收盘前以 Close 成交
- **修复方案**: 改为下一根 M15 bar 的 Open 作为入场价

**严重 3 — 回测与实盘路径不对齐** (2026-04-09 部分修复):
- ~~`regime_config` 参数表与实盘不匹配~~ → 新增 `LIVE_PARITY_KWARGS` preset 精确匹配实盘
- ~~`time_decay_tp` 回测需显式打开，实盘默认开启~~ → `LIVE_PARITY_KWARGS` 默认 `time_decay_tp=True`
- ~~ATR percentile 计算方法不同（回测 rolling-500 vs 实盘 rolling-50）~~ → 新增 `live_atr_percentile` 选项
- ~~`rsi_adx_filter` 回测默认 0，实盘 40~~ → `LIVE_PARITY_KWARGS` 设为 40
- `escalating_cooldown / min_entry_gap` 仅回测有，实盘无 — **保持现状**，`LIVE_PARITY_KWARGS` 不启用
- H1 评估时机: 回测仅 minute==0，实盘每次轮询 — **已知差异，未修复**
- Time decay TP 时间单位: 回测用 M15 bar 数，实盘用 wall-clock hours — **已知差异，未修复**
- 实盘 `_check_exits` 对 keltner 先用 H1 ATR 再用 M15 ATR（双重检查），回测仅用 H1 ATR — **已知差异，未修复**
- **建议**: 新实验使用 `LIVE_PARITY_KWARGS` 而非 `C12_KWARGS`，以确保回测→实盘可复现性

**中等 — ~~无数据缺口检测~~**: 已添加 `check_data_gaps()` (2026-04-09)
**中等 — SL/TP 同 bar 固定 SL 优先**: 不区分 bar 内真实触发先后
**中等 — ~~Sharpe 用 ddof=0~~**: 已改为 ddof=1 (2026-04-09)

**ORB monkey-patch 失效原因**: `check_orb_signal` 直接用 `_orb_strategy` 模块级单例，patch `get_orb_strategy` 无效
**T1-T6 Sharpe 相同原因**: `regime_config` 在 `_check_exits` 开头覆盖 `_trail_act/_trail_dist`，基线参数从未被使用

### 策略类
- **2026-04-10**: Trail Momentum 1.5x (trail_dist * 1.5) 在修复后引擎 + LIVE_PARITY 上全面有害。全样本 Sharpe -0.78 (sp$0.30), K-Fold 0/6 FAIL，每个 fold 均为负 delta (-0.54 ~ -1.04)。**永久否决**，不再测试 trail_dist 放大方向
- **2026-04-10**: EXP-T Donchian Channel 突破策略完全无效。所有配置（20/40/50/60 周期，0.3/0.5/1.0x 成本）全部负 Sharpe，K-Fold 0/6 PASS。**永久否决** Donchian 作为独立信号源
- **2026-04-10**: EXP-A ORB 消融 — 去掉 ORB 后 Sharpe +0.05（2.62→2.67），但 K-Fold 3/6 FAIL。ORB 贡献不显著，保留但不投入优化资源
- **2026-04-16**: **R11 PA形态全系否决**: Pinbar/Fractal/InsideBar/Engulfing + S/R zone 共 24 个实验，全部 K-Fold 0/6。PA形态无法改善 KC 突破策略。**永久否决**任何基于 K 线形态的入场过滤器方向
- **2026-04-16**: **R12 Squeeze(BB inside KC)否决**: K-Fold 0/6，Sharpe -2.30。波动率压缩虽然释放后绝对收益更大(22bp/4bar)，但过滤掉太多好信号

### 出场类
- **2026-04-11**: **KC Mid Reversion Exit 永久否决**。L4 审计 (`run_exp_l4_audit.py`) 发现严重 H1 look-ahead bias:
  - 原始版本用 `h1_window.iloc[-1]` (未收盘 H1 bar) 的 Close 和 KC_mid 做出场判断 = 偷看未来数据
  - 修正后 (使用 `iloc[-2]` 已收盘 H1) Sharpe 从 5.22 暴跌至 3.99，**低于 L3 基线的 4.07**
  - kc_mid_revert 出场: 822 笔, WR=0.9%, avg=-$21.44（几乎 100% 亏钱）
  - 修正后 K-Fold **0/6 FAIL**（原始 6/6 PASS 完全是 look-ahead 的虚假结果）
  - 穷尽测试: min_bars 1~12、profit_filter (all/loss/profit)、出场优先级 (before/after trailing) — 全部无法超过 L3
  - **教训: 任何使用 H1 bar 数据的新出场规则，必须用 `closed_only=True` 或 `iloc[-2]` 避免 look-ahead**

### 架构/安全类
- **2026-04-14**: **PermissionError 误判平仓 BUG (已修复)**: `_read_json()` 读文件遇到 Windows 文件锁时静默返回 None → 下游误判持仓已消失 → 删除 tracking → Trailing/时间止损全部失效 → 持仓可能被无限持有。**教训: 任何读取 positions.json 的失败必须区分"文件真的没内容"和"无法读取"。后者必须抛异常让调用方跳过，绝不能当作空列表处理。** 修复: 重试+异常传播+全链路 IOError 防护

### 重要经验
- **2026-04-10**: monkey-patching `staticmethod` 时必须用 `Class.__dict__['method_name']` 保存原始描述器，不能用 `Class.method_name.__func__` 解包——后者丢失 staticmethod 包装导致 self 被注入
- **2026-04-10**: 服务器实验必须在启动前确认数据文件存在（EXP-S 因服务器缺 spread CSV 回退到 $0.00 无效结果，浪费 40 分钟）
- **2026-04-10**: ~~出场类改进（KC mid revert, Tight_all trail）效果远超入场类过滤器~~ → **修正**: KC mid revert 的效果是 look-ahead 虚假结果。Tight_all trail 仍然有效。bars_held 是亏损最强预测因子 (Cohen's d=-1.95)
- **2026-04-11**: **出场规则 look-ahead 检查清单**: 回测引擎 `_check_exits` 传入的 `h1_window` 默认 `closed_only=False`（包含未收盘 H1 bar）。SL/TP 用 M15 High/Low（安全），Trailing 用 ATR（滞后指标，安全），但任何直接比较 H1 Close 或 KC_mid 的新规则**必须**使用 `iloc[-2]` 或修改调用为 `_get_h1_window(bar_time, closed_only=True)`

### 分析类
- **市场分析必须先查日历再归因**: 大跌时遗漏 Liberation Day 关税生效
- **post-hoc vs 真实引擎差异巨大**: EXP52 post-hoc +0.48 Sharpe，真实引擎仅 +0.03（差异 19 倍）

### 方法论类 (2026-04-09 严重错误)
- **"伪数据驱动"事件**: 从 3 笔亏损案例中归纳"ADX灰区"规律，声称"盈利大单ADX>28-30"但从未做过统计验证
- **错误 1 — 小样本归纳**: 3 笔案例 cherry-picking
- **错误 2 — 无视已有结论**: journal 明确记录 "SELL ADX>=28 不采纳"，ADX 因子 IC<0.005
- **错误 3 — 推翻已验证参数**: 30 分钟冷却期经 20 变体确认，提议 x4 渐进冷却无数据支撑
- **错误 4 — 先改代码后验证**: 直接修改 engine.py，被用户指出后才验证
- **根因**: 先有结论再找证据，用专业术语包装主观臆断
