#!/usr/bin/env python3
"""Install remaining deps (scipy etc) and start R8."""
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

print("=== Install all missing deps ===")
ssh_exec(client, "pip3 install scipy ta-lib 2>&1 | tail -10", timeout=600)

print("\n=== Full import test ===")
out, _ = ssh_exec(client, "cd /root/gold-quant-trading && python3 -c \"\nfrom backtest.runner import DataBundle, LIVE_PARITY_KWARGS, run_variant\nd = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)\nprint(f'OK M15={len(d.m15_df)} H1={len(d.h1_df)}')\n\" 2>&1", timeout=60)

if "OK" not in out:
    print("Still failing, check what's missing...")
    ssh_exec(client, "cd /root/gold-quant-trading && python3 -c 'from backtest.runner import DataBundle' 2>&1")
    client.close()
    sys.exit(1)

print("\n=== Start R8 ===")
ssh_exec(client, "pkill -f run_round8 2>/dev/null; sleep 1")
ssh_exec(client, "rm -f /root/gold-quant-trading/round8_results/round8_stdout.txt /root/gold-quant-trading/round8_results/00_master_log.txt")
ssh_exec(client, "mkdir -p /root/gold-quant-trading/round8_results")

launcher = """#!/bin/bash
cd /root/gold-quant-trading
export PYTHONIOENCODING=utf-8
nohup python3 -u scripts/experiments/run_round8.py > round8_results/round8_stdout.txt 2>&1 &
echo $!
"""
ssh_exec(client, f"cat > /tmp/start_r8.sh << 'SCRIPT'\n{launcher}\nSCRIPT")
ssh_exec(client, "chmod +x /tmp/start_r8.sh && bash /tmp/start_r8.sh")

time.sleep(12)

print("\n=== Verify ===")
ssh_exec(client, "ps aux | grep run_round8 | grep -v grep")
ssh_exec(client, "head -30 /root/gold-quant-trading/round8_results/round8_stdout.txt 2>/dev/null")

client.close()
print("\nDone!")
