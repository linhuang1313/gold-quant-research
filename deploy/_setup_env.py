#!/usr/bin/env python3
"""Setup Python environment on new server and start R8."""
import paramiko, sys, io, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def ssh_exec(client, cmd, timeout=300):
    print(f"  > {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if out.strip():
        for line in out.strip().split('\n')[-20:]:
            print(f"    {line}")
    if err.strip():
        for line in err.strip().split('\n')[-5:]:
            print(f"    [err] {line}")
    return out, err

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("connect.westd.seetacloud.com", port=30367, username="root", password="r1zlTZQUb+E4", timeout=15)
print("Connected!\n")

print("=== Find Python/pip ===")
ssh_exec(client, "which python3 && python3 --version")
ssh_exec(client, "which pip3 || which pip || echo 'NO PIP'")
ssh_exec(client, "which conda || echo 'NO CONDA'")
ssh_exec(client, "ls /usr/bin/pip* /usr/local/bin/pip* 2>/dev/null || echo 'no pip binaries'")
ssh_exec(client, "python3 -m pip --version 2>&1 | head -2")

print("\n=== Install pip if missing ===")
ssh_exec(client, "python3 -m ensurepip --upgrade 2>&1 | tail -3 || apt-get install -y python3-pip 2>&1 | tail -5", timeout=120)
ssh_exec(client, "python3 -m pip --version 2>&1 | head -2")

print("\n=== Install deps ===")
ssh_exec(client, "python3 -m pip install numpy pandas python-dotenv yfinance 2>&1 | tail -5", timeout=300)

print("\n=== Verify imports ===")
ssh_exec(client, "python3 -c 'import numpy, pandas, dotenv; print(\"All imports OK\")'")

print("\n=== Create .env ===")
ssh_exec(client, "echo 'GOLD_TRADING_MODE=backtest' > /root/gold-quant-trading/.env")

print("\n=== Quick DataBundle test ===")
ssh_exec(client, "cd /root/gold-quant-trading && python3 -c \"from backtest.runner import DataBundle; d=DataBundle.load_custom(kc_ema=25,kc_mult=1.2); print(f'M15={len(d.m15_df)} H1={len(d.h1_df)}')\" 2>&1", timeout=60)

print("\n=== Start R8 ===")
ssh_exec(client, "pkill -f run_round8 2>/dev/null; sleep 1")
ssh_exec(client, "mkdir -p /root/gold-quant-trading/round8_results")
launcher = """#!/bin/bash
cd /root/gold-quant-trading
export PYTHONIOENCODING=utf-8
nohup python3 -u scripts/experiments/run_round8.py > round8_results/round8_stdout.txt 2>&1 &
echo $!
"""
ssh_exec(client, f"cat > /tmp/start_r8.sh << 'SCRIPT'\n{launcher}\nSCRIPT")
ssh_exec(client, "chmod +x /tmp/start_r8.sh && bash /tmp/start_r8.sh")

time.sleep(8)
print("\n=== Verify running ===")
ssh_exec(client, "ps aux | grep run_round8 | grep -v grep")
ssh_exec(client, "head -20 /root/gold-quant-trading/round8_results/round8_stdout.txt 2>/dev/null")
ssh_exec(client, "wc -l /root/gold-quant-trading/round8_results/round8_stdout.txt 2>/dev/null")

client.close()
print("\nDone!")
