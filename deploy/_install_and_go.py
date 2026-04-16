#!/usr/bin/env python3
"""Install pip via apt, then deps, then start R8."""
import paramiko, sys, io, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def ssh_exec(client, cmd, timeout=600):
    print(f"  > {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if out.strip():
        for line in out.strip().split('\n')[-15:]:
            print(f"    {line}")
    if err.strip():
        for line in err.strip().split('\n')[-5:]:
            print(f"    [err] {line}")
    return out, err

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("connect.westd.seetacloud.com", port=30367, username="root", password="r1zlTZQUb+E4", timeout=15)
print("Connected!\n")

print("=== Step 1: Install pip3 via apt ===")
ssh_exec(client, "apt-get update -qq 2>&1 | tail -3", timeout=120)
ssh_exec(client, "apt-get install -y python3-pip 2>&1 | tail -5", timeout=300)

print("\n=== Step 2: Verify pip3 ===")
out, _ = ssh_exec(client, "pip3 --version 2>&1")
if "pip" not in out:
    ssh_exec(client, "python3 -m pip --version 2>&1")

print("\n=== Step 3: Install Python deps ===")
ssh_exec(client, "pip3 install numpy pandas python-dotenv yfinance 2>&1 | tail -10", timeout=600)

print("\n=== Step 4: Verify all imports ===")
out, err = ssh_exec(client, "python3 -c 'import numpy, pandas, dotenv; print(f\"numpy={numpy.__version__} pandas={pandas.__version__} dotenv OK\")'")
if "OK" not in out:
    print("!!! DEPS STILL MISSING, aborting")
    client.close()
    sys.exit(1)

print("\n=== Step 5: DataBundle test ===")
out, _ = ssh_exec(client, "cd /root/gold-quant-trading && python3 -c \"from backtest.runner import DataBundle; d=DataBundle.load_custom(kc_ema=25,kc_mult=1.2); print(f'OK M15={len(d.m15_df)} H1={len(d.h1_df)}')\" 2>&1", timeout=60)
if "OK" not in out:
    print("!!! DataBundle failed, check output")
    client.close()
    sys.exit(1)

print("\n=== Step 6: Start R8 ===")
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

time.sleep(10)

print("\n=== Step 7: Verify running ===")
ssh_exec(client, "ps aux | grep run_round8 | grep -v grep")
ssh_exec(client, "head -30 /root/gold-quant-trading/round8_results/round8_stdout.txt 2>/dev/null")

client.close()
print("\nAll done!")
