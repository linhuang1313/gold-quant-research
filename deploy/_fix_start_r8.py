#!/usr/bin/env python3
"""Fix deps, pull code, start R8."""
import paramiko, sys, io, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def ssh_exec(client, cmd, timeout=120):
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

print("=== Install deps ===")
ssh_exec(client, "pip3 install python-dotenv 2>&1 | tail -3", timeout=120)

print("\n=== Pull fix ===")
ssh_exec(client, "cd /root/gold-quant-trading && git pull origin main 2>&1 | tail -5", timeout=60)

print("\n=== Quick test ===")
ssh_exec(client, "cd /root/gold-quant-trading && python3 -c \"from backtest.runner import DataBundle, LIVE_PARITY_KWARGS; d=DataBundle.load_custom(kc_ema=25, kc_mult=1.2); print(f'OK: M15={len(d.m15_df)} H1={len(d.h1_df)}')\" 2>&1", timeout=60)

print("\n=== Start R8 ===")
ssh_exec(client, "pkill -f run_round8 2>/dev/null; sleep 1")
ssh_exec(client, "mkdir -p /root/gold-quant-trading/round8_results")
ssh_exec(client, "cd /root/gold-quant-trading && PYTHONIOENCODING=utf-8 nohup python3 -u scripts/experiments/run_round8.py > round8_results/round8_stdout.txt 2>&1 & echo PID=$!")

time.sleep(5)
print("\n=== Verify ===")
ssh_exec(client, "ps aux | grep run_round8 | grep -v grep | head -2")
ssh_exec(client, "tail -10 /root/gold-quant-trading/round8_results/round8_stdout.txt 2>/dev/null")

client.close()
print("\nDone!")
