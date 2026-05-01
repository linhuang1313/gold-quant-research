# Gold Quant Research

XAUUSD (黄金) 量化交易研究与回测系统 — 从策略发现到实盘部署的完整工具链。

## 系统架构

```
gold-quant-research/
├── backtest/              # 核心引擎
│   ├── engine.py          # M15/H1 双时间框架回测引擎
│   ├── runner.py          # DataBundle 数据管理 + 批量运行
│   ├── validator.py       # 8 阶段专业验证管线 (核心)
│   └── stats.py           # DSR, PSR, PBO 等高级统计
├── experiments/           # 实验脚本 (R1-R71)
├── deploy/                # MT4 EA + 服务器部署
│   ├── *.mq4              # MetaTrader 4 Expert Advisors
│   ├── TradeLogger.mqh    # 交易日志导出模块
│   └── deploy_*.py        # 远程服务器部署脚本
├── monitor/               # 实时监控系统
│   ├── live_monitor.py    # 实盘交易监控 + 止损预警
│   ├── auto_revalidate.py # 自动策略衰退检测
│   ├── update_data.py     # 数据自动更新
│   └── run_monitor.py     # 统一入口
├── data/download/         # 历史行情 CSV (M15/H1, 2015-2026)
├── results/               # 实验结果 JSON
├── indicators.py          # 技术指标计算
└── research_config.py     # 研究配置
```

## 8 阶段策略验证管线

核心组件 `backtest/validator.py` 实现了专业级的反过拟合验证框架，每个策略必须通过以下 8 个阶段才能部署：

```
Stage 0: BASE LOGIC          无优化参数的基础信号测试，确认策略有内在边际
Stage 1: SANITY + DSR         基本健全性 + Deflated Sharpe Ratio (校正选择偏差)
Stage 2: ROBUSTNESS           Purged K-Fold 交叉验证 (带净化间隔防数据泄漏)
Stage 3: WALK-FORWARD         Walk-Forward Efficiency 滚动前向验证
Stage 4: STRESS + PBO         Monte Carlo 参数扰动 + Probability of Backtest Overfitting
Stage 5: COST                 多级点差/滑点压力测试 (找到盈亏平衡点)
Stage 6: REALITY              年度一致性 + 参数稳定性高原 + PSR 显著性
Stage 7: DEPLOYMENT           自动生成止损标准 (最大回撤/月亏损/连亏天数)
```

### 关键统计指标

| 指标 | 说明 | 来源 |
|------|------|------|
| **DSR** (Deflated Sharpe Ratio) | 校正多重测试和选择偏差后的 Sharpe | Bailey & López de Prado, 2014 |
| **PSR** (Probabilistic Sharpe Ratio) | 观测 Sharpe 显著超过基准的概率 | Bailey & López de Prado, 2012 |
| **PBO** (Probability of Backtest Overfitting) | 参数选择导致样本外亏损的概率 | Bailey et al., 2017 |
| **WFE** (Walk-Forward Efficiency) | OOS/IS 收益比，衡量参数泛化能力 | - |
| **Purged K-Fold** | 带净化间隔的时间序列交叉验证 | López de Prado, 2018 |

### 使用示例

```python
from backtest.validator import StrategyValidator, ValidatorConfig

config = ValidatorConfig(
    n_trials_tested=72,        # 测试过的参数组合数 (用于 DSR 校正)
    realistic_spread=0.88,     # 真实交易成本 (点)
    purge_bars=30,             # K-Fold 净化间隔
)

validator = StrategyValidator(
    name="MY_STRATEGY",
    backtest_fn=my_backtest_fn,  # fn(h1_df, spread, lot) -> list[dict]
    spread=0.30, lot=0.03,
    h1_df=h1_data,
    config=config,
)

results = validator.run_all(stop_on_fail=False)
# results: {0: StageResult, 1: StageResult, ..., 7: StageResult}
```

## 已验证策略

经过 R69-R71 轮完整 8 阶段验证，以下策略已部署到 MT4 实盘：

| 策略 | 时间框架 | 信号 | 验证结果 | OOS Sharpe |
|------|---------|------|---------|-----------|
| **PSAR** | H1 | Parabolic SAR 方向翻转 | 7/8 PASS (PBO=1.4%) | 5.26 |
| **SESS_BO** | H1 | 12:00-14:00 UTC 区间突破 | 7/8 PASS | 3.90 |
| **TSMOM** | H1 | 时序动量 (480/720h 双窗口) | 8/8 PASS | 3.31 |
| **L8_MAX** | M15+H1 | 多信号融合 (KC/ORB/Gap) | 8/8 PASS | - |

OOS Sharpe 来自 R71 纯样本外测试 (2024-2026 holdout, 参数未接触此数据段)。

## 实时监控系统

部署后通过 `monitor/` 模块持续监控：

```bash
# 检查实盘交易状态
python -m monitor.run_monitor --mode live

# 持续监控 (每15分钟)
python -m monitor.run_monitor --mode live --loop --interval 15

# 更新数据 + 重验证策略
python -m monitor.run_monitor --mode refresh

# 全套监控
python -m monitor.run_monitor --mode all --loop --interval 60
```

每个策略都有自动止损标准 (Stage 7 生成)，触发时控制台报警：
- 最大回撤超限
- 月度亏损超限
- 连续亏损天数超限

## 研究历程 (R1-R71)

| 阶段 | 轮次 | 内容 |
|------|------|------|
| 信号发现 | R1-R15 | 趋势跟踪、均值回归、ORB、Gap 等信号筛选 |
| 引擎优化 | R16-R38 | M15+H1 双框架引擎、指标优化、性能加速 |
| 参数网格 | R39-R52 | 大规模参数搜索 (最高 57,600 组合) |
| 组合优化 | R53-R56 | 策略组合配比、手数分配、TSMOM 独立验证 |
| 深度验证 | R57-R68 | Monte Carlo、K-Fold、Walk-Forward |
| 专业验证 | R69-R70 | 8 阶段完整管线 (DSR/PBO/Purged KFold/WFE) |
| 样本外 | R71 | 2024-2026 纯 holdout 前向测试 |

## 安装与使用

```bash
git clone https://github.com/linhuang1313/gold-quant-research.git
cd gold-quant-research
pip install -r requirements.txt

# 运行 8 阶段验证示例
python experiments/example_validate_psar.py
```

### 依赖

- Python 3.10+
- pandas, numpy, scipy
- paramiko (远程服务器部署)
- histdata (数据下载)

## 与实盘系统的关系

```
gold-quant-research (本仓库)     gold-quant-trading (实盘)
     策略研发 + 回测验证     -->     MT4 桥接交易执行
     8 阶段验证管线          -->     EA 参数部署
     实时监控系统            <--     交易日志导出
```

两个仓库代码独立，通过 EA 参数和交易日志 CSV 连接。
