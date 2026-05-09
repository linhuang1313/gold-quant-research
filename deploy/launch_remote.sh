#!/bin/bash
cd /root/gold-quant-research

mkdir -p results/r132 results/r133 results/r134 results/r135 results/r136
mkdir -p results/r137 results/r138 results/r139 results/r140 results/r141
mkdir -p results/r142 results/r143

echo "Launching all R132-R143..."

nohup python3 -u experiments/run_r132_s4_chandelier_validation.py > results/r132/r132_stdout.txt 2>&1 &
echo "r132: PID=$!"

nohup python3 -u experiments/run_r133_production_v3.py > results/r133/r133_stdout.txt 2>&1 &
echo "r133: PID=$!"

nohup python3 -u experiments/run_r134_overnight_session.py > results/r134/r134_stdout.txt 2>&1 &
echo "r134: PID=$!"

nohup python3 -u experiments/run_r135_s3_s4_multi.py > results/r135/r135_stdout.txt 2>&1 &
echo "r135: PID=$!"

nohup python3 -u experiments/run_r136_cot_macro_strategy.py > results/r136/r136_stdout.txt 2>&1 &
echo "r136: PID=$!"

nohup python3 -u experiments/run_r137_tft_entry.py > results/r137/r137_stdout.txt 2>&1 &
echo "r137: PID=$!"

nohup python3 -u experiments/run_r138_rl_exit.py > results/r138/r138_stdout.txt 2>&1 &
echo "r138: PID=$!"

nohup python3 -u experiments/run_r139_extreme_detection.py > results/r139/r139_stdout.txt 2>&1 &
echo "r139: PID=$!"

nohup python3 -u experiments/run_r140_dd_recovery.py > results/r140/r140_stdout.txt 2>&1 &
echo "r140: PID=$!"

nohup python3 -u experiments/run_r141_tail_risk_budget.py > results/r141/r141_stdout.txt 2>&1 &
echo "r141: PID=$!"

nohup python3 -u experiments/run_r142_execution_analysis.py > results/r142/r142_stdout.txt 2>&1 &
echo "r142: PID=$!"

nohup python3 -u experiments/run_r143_paper_trade_framework.py > results/r143/r143_stdout.txt 2>&1 &
echo "r143: PID=$!"

echo "ALL 12 EXPERIMENTS LAUNCHED"
