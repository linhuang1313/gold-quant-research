# Gold Quant Research

XAUUSD 量化交易研究与回测系统。

## 目录结构

```
gold-quant-research/
  backtest/          # 回测引擎 (engine, runner, stats)
  experiments/       # 实验脚本 (run_round*.py)
  deploy/            # 服务器部署/监控脚本
  legacy/            # 历史回测脚本
  data/download/     # 历史行情CSV数据
  results/           # 实验结果 (round*_results/)
  research_config.py # 研究用配置 (独立于实盘)
  indicators.py      # 技术指标计算 (从signals.py提取)
```

## 使用方法

```bash
pip install -r requirements.txt
python experiments/run_round14.py
```

## 与实盘系统的关系

本仓库与 [gold-quant-trading](https://github.com/linhuang1313/gold-quant-trading) 完全独立：
- 实盘系统: `gold-quant-trading` — 运行 MT4 桥接交易
- 研究系统: `gold-quant-research` — 回测、实验、策略研发

两者共享策略逻辑思路，但代码独立，互不影响。
