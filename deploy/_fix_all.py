#!/usr/bin/env python3
"""Fix all issues on both servers and restart experiments."""
import paramiko, sys, io, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SERVERS = {
    "A": {"host": "connect.westb.seetacloud.com", "port": 42894, "user": "root", "password": "r1zlTZQUb+E4"},
    "B": {"host": "connect.westb.seetacloud.com", "port": 25821, "user": "root", "password": "r1zlTZQUb+E4"},
}
PYTHON = "/root/miniconda3/bin/python"
PIP = "/root/miniconda3/bin/pip"

def run(c, cmd, timeout=120):
    _, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()
    return out, err

# === Server B: Fix dotenv + restart R6B ===
print("="*60)
print("Server B: Fix + Restart R6B")
print("="*60)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(SERVERS["B"]["host"], port=SERVERS["B"]["port"], username="root", password="r1zlTZQUb+E4", timeout=30)

# Kill broken R6B
print("[B] Killing broken R6B...")
run(c, "pkill -f 'run_round6b' 2>/dev/null")
time.sleep(2)

# Install dotenv
print("[B] Installing python-dotenv...")
out, _ = run(c, f"{PIP} install python-dotenv 2>&1 | tail -2")
print(f"  {out}")

# Verify
out, _ = run(c, f"{PYTHON} -c 'from dotenv import load_dotenv; print(\"OK\")'")
print(f"  dotenv: {out}")

# Create .env
run(c, "touch /root/gold-quant-trading/.env")

# Clean old failed R6B results (B2-B6 were all failures)
print("[B] Cleaning failed R6B results...")
for f in ["r6_b2_exit_analysis.txt", "r6_b3_strategy_combo.txt", "r6_b4_interaction.txt", "r6_b5_recent_zoom.txt", "r6_b6_heatmap.txt"]:
    run(c, f"rm -f /root/gold-quant-trading/round6_results/{f}")

# Start R6B fresh
print("[B] Starting R6B...")
# Write a launcher script to avoid SSH timeout issues
launcher = f"""#!/bin/bash
cd /root/gold-quant-trading
mkdir -p round6_results
{PYTHON} -u scripts/experiments/run_round6b.py > round6_results/round6b_stdout.txt 2>&1
"""
run(c, f"echo '{launcher}' > /tmp/r6b.sh && chmod +x /tmp/r6b.sh")
run(c, "nohup bash /tmp/r6b.sh &")
time.sleep(8)

out, _ = run(c, "ps aux | grep 'round6b' | grep python | grep -v grep | wc -l")
print(f"[B] R6B processes: {out}")

out, _ = run(c, "tail -5 /root/gold-quant-trading/round6_results/round6b_stdout.txt 2>/dev/null || echo 'waiting...'")
print(f"[B] R6B output: {out[-200:]}")

# Check R7 on B
out, _ = run(c, "ps aux | grep 'round7' | grep python | grep -v grep | wc -l")
print(f"[B] R7 processes: {out}")

c.close()

# === Server A: Install dotenv + check R7 status ===
print(f"\n{'='*60}")
print("Server A: Fix dotenv + Check R7")
print("="*60)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(SERVERS["A"]["host"], port=SERVERS["A"]["port"], username="root", password="r1zlTZQUb+E4", timeout=30)

# Install dotenv (for when R7-4+ needs it)
print("[A] Installing python-dotenv...")
out, _ = run(c, f"{PIP} install python-dotenv 2>&1 | tail -2")
print(f"  {out}")
run(c, "touch /root/gold-quant-trading/.env")

# Check R7 progress detail
out, _ = run(c, "ps aux | grep 'round7' | grep python | grep -v grep | wc -l")
print(f"[A] R7 processes: {out}")

# How long has R7 been running?
out, _ = run(c, "ps aux | grep 'round7' | grep python | grep -v grep | head -3")
print(f"[A] R7 process details:\n  {out}")

# Check if R7-4 has started writing results
out, _ = run(c, "ls -la /root/gold-quant-trading/round7_results/ 2>/dev/null")
print(f"[A] R7 results:\n  {out}")

# R7 stdout tail for progress
out, _ = run(c, "tail -3 /root/gold-quant-trading/round7_stdout.txt 2>/dev/null")
print(f"[A] R7 last output: {out[-200:]}")

c.close()
print("\nAll fixes applied!")
