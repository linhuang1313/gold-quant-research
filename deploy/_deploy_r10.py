#!/usr/bin/env python3
"""Deploy and launch Round 10 experiments on remote server."""
import paramiko, sys, io, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westd.seetacloud.com"
PORT = 30367
USER = "root"
PASS = "r1zlTZQUb+E4"

def run_cmd(c, cmd, timeout=60, read_timeout=30):
    print(f"  $ {cmd}")
    _, o, e = c.exec_command(cmd, timeout=timeout)
    o.channel.settimeout(read_timeout)
    e.channel.settimeout(read_timeout)
    try:
        out = o.read().decode("utf-8", errors="replace")
    except Exception:
        out = "(read timeout - command may still be running)"
    try:
        err = e.read().decode("utf-8", errors="replace")
    except Exception:
        err = ""
    if out.strip(): print(out.strip())
    if err.strip(): print(f"  STDERR: {err.strip()}")
    return out

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

print("="*60)
print("Round 10 Deployment")
print("="*60)

# Step 1: Enable autodl proxy for git pull
print("\n--- Step 1: Enable autodl proxy ---")
run_cmd(c, "source /etc/network_turbo 2>/dev/null; echo 'proxy enabled'")

# Step 2: Git pull latest code
print("\n--- Step 2: Git pull ---")
run_cmd(c, "cd /root/gold-quant-trading && source /etc/network_turbo 2>/dev/null && git stash && git pull origin main", timeout=120)

# Step 3: Check required files
print("\n--- Step 3: Verify files ---")
run_cmd(c, "ls -la /root/gold-quant-trading/scripts/experiments/run_round10.py")
run_cmd(c, "ls -la /root/gold-quant-trading/backtest/engine.py")
run_cmd(c, "ls -la /root/gold-quant-trading/data/download/xauusd-m15-bid-*.csv | head -3")

# Step 4: Verify syntax
print("\n--- Step 4: Syntax check ---")
run_cmd(c, 'cd /root/gold-quant-trading && python3 -c "import ast; ast.parse(open(\'scripts/experiments/run_round10.py\').read()); print(\'R10 syntax OK\')"')
run_cmd(c, 'cd /root/gold-quant-trading && python3 -c "import ast; ast.parse(open(\'backtest/engine.py\').read()); print(\'Engine syntax OK\')"')

# Step 5: Create output dir
print("\n--- Step 5: Setup ---")
run_cmd(c, "mkdir -p /root/gold-quant-trading/round10_results")

# Step 6: Kill existing round10 if any
print("\n--- Step 6: Kill existing ---")
run_cmd(c, "pkill -f run_round10 2>/dev/null; echo 'cleaned'")
time.sleep(2)

# Step 7: Launch
print("\n--- Step 7: Launch! ---")
launch_cmd = (
    "cd /root/gold-quant-trading && "
    "nohup python3 scripts/experiments/run_round10.py "
    "> round10_results/round10_stdout.txt 2>&1 &"
)
run_cmd(c, launch_cmd)
time.sleep(3)

# Step 8: Verify running
print("\n--- Step 8: Verify ---")
run_cmd(c, "ps aux | grep run_round10 | grep -v grep")
run_cmd(c, "cat /root/gold-quant-trading/round10_results/00_master_log.txt 2>/dev/null | head -10")
run_cmd(c, "tail -5 /root/gold-quant-trading/round10_results/round10_stdout.txt 2>/dev/null")

c.close()
print("\n" + "="*60)
print("Round 10 deployed! Monitor with: python scripts/server/_check_r10.py")
