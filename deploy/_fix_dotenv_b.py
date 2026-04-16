#!/usr/bin/env python3
"""Fix dotenv on Server B - install for ALL python versions."""
import paramiko, sys, io, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westb.seetacloud.com", port=25821, username="root", password="r1zlTZQUb+E4", timeout=30)
print("Connected to Server B!")

def run(cmd):
    _, stdout, stderr = c.exec_command(cmd, timeout=120)
    out = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()
    print(f">>> {cmd[:120]}")
    if out: print(f"  {out[-300:]}")
    if err and 'WARNING' not in err: print(f"  ERR: {err[-200:]}")
    return out

# Check which python the R6B process is using
run("ps aux | grep 'round6b' | grep -v grep")

# Install dotenv for ALL possible pythons
run("/root/miniconda3/bin/pip install python-dotenv 2>&1 | tail -2")
run("/root/miniconda3/bin/python -c 'from dotenv import load_dotenv; print(\"miniconda OK\")'")

# Also try the env python
run("/root/miniconda3/envs/3.10/bin/pip install python-dotenv 2>&1 | tail -2")

# Kill R6B and restart with explicit miniconda python
run("pkill -f 'run_round6b' 2>/dev/null")
time.sleep(2)

# Make sure .env exists
run("touch /root/gold-quant-trading/.env")

# Start with explicit path
run("cd /root/gold-quant-trading && nohup /root/miniconda3/bin/python -u scripts/experiments/run_round6b.py > round6_results/round6b_stdout.txt 2>&1 &")
time.sleep(10)

# Verify
run("ps aux | grep 'round6b' | grep -v grep")
run("tail -10 /root/gold-quant-trading/round6_results/round6b_stdout.txt 2>/dev/null")

c.close()
print("\nDone!")
