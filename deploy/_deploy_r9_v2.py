#!/usr/bin/env python3
"""Deploy R9 with autodl proxy for git pull."""
import paramiko, sys, io, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westd.seetacloud.com"
PORT = 30367
USER = "root"
PASS = "r1zlTZQUb+E4"

def run(client, cmd, timeout=120):
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
    return out

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
print("Connected!\n")

# Step 1: Setup proxy and pull
print("=== Step 1: Git pull with proxy ===")
run(c, "source /etc/network_turbo 2>/dev/null; cd /root/gold-quant-trading; git pull origin main 2>&1 | tail -5", timeout=60)

# Step 2: Verify script exists
print("\n=== Step 2: Verify R9 ===")
run(c, "ls -la /root/gold-quant-trading/scripts/experiments/run_round9.py")

# Step 3: Kill old
print("\n=== Step 3: Kill old ===")
run(c, "pkill -f 'run_round[89]' 2>/dev/null; sleep 1; echo done")

# Step 4: Start R9
print("\n=== Step 4: Start R9 ===")
run(c, "mkdir -p /root/gold-quant-trading/round9_results")

launcher = """#!/bin/bash
cd /root/gold-quant-trading
export PYTHONIOENCODING=utf-8
nohup python3 -u scripts/experiments/run_round9.py > round9_results/round9_stdout.txt 2>&1 &
echo PID=$!
"""
run(c, f"cat > /tmp/start_r9.sh << 'SCRIPT'\n{launcher}\nSCRIPT")
run(c, "chmod +x /tmp/start_r9.sh; bash /tmp/start_r9.sh")

time.sleep(10)

# Step 5: Verify
print("\n=== Step 5: Verify ===")
run(c, "ps aux | grep run_round9 | grep -v grep | head -3")
run(c, "head -15 /root/gold-quant-trading/round9_results/round9_stdout.txt 2>/dev/null")

c.close()
print("\nDone!")
