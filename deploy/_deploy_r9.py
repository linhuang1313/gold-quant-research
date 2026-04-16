#!/usr/bin/env python3
"""Deploy and start Round 9 on server."""
import paramiko, sys, io, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def ssh_exec(client, cmd, timeout=300):
    print(f"  > {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if out.strip():
        for line in out.strip().split('\n')[-10:]:
            print(f"    {line}")
    if err.strip():
        for line in err.strip().split('\n')[-3:]:
            print(f"    [err] {line}")
    return out, err

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("connect.westd.seetacloud.com", port=30367, username="root", password="r1zlTZQUb+E4", timeout=15)
print("Connected!\n")

print("=== Pull latest code ===")
ssh_exec(client, "cd /root/gold-quant-trading; git pull origin main 2>&1 | tail -5")

print("\n=== Verify R9 script ===")
ssh_exec(client, "head -20 /root/gold-quant-trading/scripts/experiments/run_round9.py")

print("\n=== Quick import test ===")
out, _ = ssh_exec(client, "cd /root/gold-quant-trading; python3 -c 'from backtest.runner import DataBundle; print(\"OK\")'")
if "OK" not in out:
    print("Import failed!")
    client.close()
    sys.exit(1)

print("\n=== Kill old processes ===")
ssh_exec(client, "pkill -f run_round 2>/dev/null; sleep 2")

print("\n=== Start R9 ===")
ssh_exec(client, "mkdir -p /root/gold-quant-trading/round9_results")
launcher = """#!/bin/bash
cd /root/gold-quant-trading
export PYTHONIOENCODING=utf-8
nohup python3 -u scripts/experiments/run_round9.py > round9_results/round9_stdout.txt 2>&1 &
echo $!
"""
ssh_exec(client, f"cat > /tmp/start_r9.sh << 'SCRIPT'\n{launcher}\nSCRIPT")
ssh_exec(client, "chmod +x /tmp/start_r9.sh; bash /tmp/start_r9.sh")

time.sleep(10)

print("\n=== Verify ===")
ssh_exec(client, "ps aux | grep run_round9 | grep -v grep | wc -l")
ssh_exec(client, "head -20 /root/gold-quant-trading/round9_results/round9_stdout.txt 2>/dev/null")

client.close()
print("\nDone!")
