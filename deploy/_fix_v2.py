#!/usr/bin/env python3
"""Fix R6B on Server B using non-blocking nohup, check R7 on Server A."""
import paramiko, sys, io, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SERVERS = {
    "A": {"host": "connect.westb.seetacloud.com", "port": 42894, "user": "root", "password": "r1zlTZQUb+E4"},
    "B": {"host": "connect.westb.seetacloud.com", "port": 25821, "user": "root", "password": "r1zlTZQUb+E4"},
}
PYTHON = "/root/miniconda3/bin/python"

def run(c, cmd, timeout=60):
    _, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    return stdout.read().decode('utf-8', errors='replace').strip()

def run_fire_and_forget(c, cmd):
    """Send command without waiting for output."""
    c.exec_command(cmd)

# === Server B ===
print("="*60)
print("Server B: Fix + Restart R6B")
print("="*60)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(SERVERS["B"]["host"], port=SERVERS["B"]["port"], username="root", password="r1zlTZQUb+E4", timeout=30)

# dotenv already installed from last run, verify
out = run(c, f"{PYTHON} -c 'from dotenv import load_dotenv; print(\"dotenv_OK\")'")
print(f"[B] dotenv: {out}")

# Kill any remaining R6B
out = run(c, "pkill -f 'run_round6b' 2>/dev/null; sleep 2; ps aux | grep 'round6b' | grep python | grep -v grep | wc -l")
print(f"[B] After kill, R6B procs: {out}")

# Clean failed results
for f in ["r6_b2_exit_analysis.txt", "r6_b3_strategy_combo.txt", "r6_b4_interaction.txt", "r6_b5_recent_zoom.txt", "r6_b6_heatmap.txt", "00_master_log.txt", "round6b_stdout.txt"]:
    run(c, f"rm -f /root/gold-quant-trading/round6_results/{f}")
print("[B] Cleaned failed results")

# Create .env
run(c, "touch /root/gold-quant-trading/.env")

# Write launcher script on server
launcher_script = f"""#!/bin/bash
cd /root/gold-quant-trading
mkdir -p round6_results
nohup {PYTHON} -u scripts/experiments/run_round6b.py > round6_results/round6b_stdout.txt 2>&1 &
echo "R6B PID: $!"
"""
run(c, f"cat > /tmp/start_r6b.sh << 'ENDSCRIPT'\n{launcher_script}\nENDSCRIPT")
run(c, "chmod +x /tmp/start_r6b.sh")

# Execute launcher
print("[B] Launching R6B...")
out = run(c, "bash /tmp/start_r6b.sh", timeout=15)
print(f"  {out}")

time.sleep(5)

out = run(c, "ps aux | grep 'run_round6b' | grep python | grep -v grep | wc -l")
print(f"[B] R6B processes: {out}")

out = run(c, "tail -3 /root/gold-quant-trading/round6_results/round6b_stdout.txt 2>/dev/null || echo 'starting...'")
print(f"[B] R6B output: {out}")

# R7 on B
out = run(c, "ps aux | grep 'run_round7' | grep python | grep -v grep | wc -l")
print(f"[B] R7 processes: {out}")

c.close()

# === Server A ===
print(f"\n{'='*60}")
print("Server A: Install dotenv + Check R7")
print("="*60)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(SERVERS["A"]["host"], port=SERVERS["A"]["port"], username="root", password="r1zlTZQUb+E4", timeout=30)

# Install dotenv (precaution for R7-4+ that imports config)
out = run(c, f"{PYTHON} -c 'from dotenv import load_dotenv; print(\"dotenv_OK\")' 2>/dev/null || {PYTHON} -m pip install python-dotenv -q 2>&1 | tail -2")
print(f"[A] dotenv: {out}")
run(c, "touch /root/gold-quant-trading/.env")

# R7 progress
out = run(c, "cat /root/gold-quant-trading/round7_results/00_master_log.txt 2>/dev/null")
print(f"[A] R7 master log:\n{out}")

out = run(c, "ls -la /root/gold-quant-trading/round7_results/ 2>/dev/null")
print(f"[A] R7 files:\n{out}")

# R7 stdout for progress info
out = run(c, "tail -5 /root/gold-quant-trading/round7_stdout.txt 2>/dev/null")
print(f"[A] R7 last output: {out[-300:]}")

out = run(c, "ps aux | grep 'run_round7' | grep python | grep -v grep | wc -l")
print(f"[A] R7 processes: {out}")

c.close()
print("\nDone!")
