#!/usr/bin/env python3
"""Restart R6B on Server B with proper timeout handling."""
import paramiko, sys, io, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westb.seetacloud.com", port=25821, username="root", password="r1zlTZQUb+E4", timeout=30)
print("Connected!")

def run(cmd, timeout=30):
    _, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    return stdout.read().decode('utf-8', errors='replace').strip()

# Check if R6B is running
out = run("ps aux | grep 'round6b' | grep -v grep | wc -l")
print(f"Current R6B processes: {out}")

if int(out.strip() or "0") == 0:
    print("No R6B running. Starting...")
    # Use exec_command with get_pty to avoid nohup issues
    # Write a small launcher script
    launcher = """#!/bin/bash
cd /root/gold-quant-trading
mkdir -p round6_results
nohup /root/miniconda3/bin/python -u scripts/experiments/run_round6b.py > round6_results/round6b_stdout.txt 2>&1 &
echo $!
"""
    run(f"cat > /tmp/start_r6b.sh << 'SCRIPT'\n{launcher}\nSCRIPT")
    run("chmod +x /tmp/start_r6b.sh")
    
    pid = run("bash /tmp/start_r6b.sh")
    print(f"Started with PID: {pid}")
    
    time.sleep(8)
    
    out = run("ps aux | grep 'round6b' | grep -v grep")
    print(f"Process check: {out}")
    
    out = run("tail -10 /root/gold-quant-trading/round6_results/round6b_stdout.txt 2>/dev/null || echo 'no output yet'")
    print(f"Output: {out[-500:]}")
else:
    print("R6B already running, checking progress...")
    out = run("tail -5 /root/gold-quant-trading/round6_results/round6b_stdout.txt 2>/dev/null")
    print(f"Last output: {out[-300:]}")

# Also check R7
out = run("ps aux | grep 'round7' | grep -v grep | wc -l")
print(f"\nR7 processes: {out}")
out = run("cat /root/gold-quant-trading/round7_results/00_master_log.txt 2>/dev/null || echo 'no log'")
print(f"R7 log: {out}")

c.close()
print("\nDone!")
