"""Quick deploy + start Phase 6-8."""
import paramiko, os, time
from pathlib import Path

HOST = "connect.westd.seetacloud.com"
PORT = 30367
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE = "/root/gold-quant-trading"
ROOT = Path(__file__).parent.parent

files = [
    "strategies/signals.py",
    "backtest/engine.py",
    "backtest/runner.py",
    "scripts/experiments/run_round11.py",
    "paper_trader.py",
]

def run(client, cmd, t=15):
    _, out, err = client.exec_command(cmd, timeout=t)
    o = out.read().decode("utf-8", "replace").strip()
    e = err.read().decode("utf-8", "replace").strip()
    return o, e

print("Connecting...")
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=10)
except Exception as ex:
    print(f"FAILED: {ex}")
    print("Server may be sleeping. Wake it from AutoDL console.")
    raise SystemExit(1)
print("Connected!")

# Upload
sftp = client.open_sftp()
for f in files:
    local = str(ROOT / f)
    remote = f"{REMOTE}/{f}"
    rdir = os.path.dirname(remote)
    try:
        sftp.stat(rdir)
    except FileNotFoundError:
        parts = rdir.split("/")
        for i in range(2, len(parts) + 1):
            d = "/".join(parts[:i])
            try:
                sftp.stat(d)
            except FileNotFoundError:
                sftp.mkdir(d)
    sftp.put(local, remote)
    print(f"  OK: {f} ({os.path.getsize(local):,} bytes)")
sftp.close()

# Kill old
run(client, "pkill -f run_round11 || true")
time.sleep(1)

# Verify
o, e = run(client, f"cd {REMOTE} && python3 -c 'from backtest.engine import BacktestEngine; print(\"engine OK\")'")
print(o or e)
o, e = run(client, f"cd {REMOTE} && python3 -c 'from strategies.signals import prepare_indicators; print(\"signals OK\")'")
print(o or e)

# Write a shell runner on the server
runner_script = """#!/bin/bash
cd /root/gold-quant-trading
echo "=== Phase 6 ===" >> logs/round11_ext.log
python3 -u scripts/experiments/run_round11.py 6 >> logs/round11_ext.log 2>&1
echo "=== Phase 7 ===" >> logs/round11_ext.log
python3 -u scripts/experiments/run_round11.py 7 >> logs/round11_ext.log 2>&1
echo "=== Phase 8 ===" >> logs/round11_ext.log
python3 -u scripts/experiments/run_round11.py 8 >> logs/round11_ext.log 2>&1
echo "=== ALL DONE ===" >> logs/round11_ext.log
"""
sftp = client.open_sftp()
run(client, f"mkdir -p {REMOTE}/round11_results {REMOTE}/logs")
with sftp.open(f"{REMOTE}/scripts/run_r11_ext.sh", "w") as fh:
    fh.write(runner_script)
sftp.close()
run(client, f"chmod +x {REMOTE}/scripts/run_r11_ext.sh")

# Launch
print("\nStarting Phase 6-8...")
client.exec_command(f"nohup {REMOTE}/scripts/run_r11_ext.sh > /dev/null 2>&1 &", timeout=5)
time.sleep(4)
o, _ = run(client, "ps aux | grep run_round11 | grep -v grep")
if "run_round11" in o:
    print("Phase 6-8 RUNNING!")
else:
    # Check if runner script is running
    o2, _ = run(client, "ps aux | grep run_r11_ext | grep -v grep")
    if "run_r11_ext" in o2:
        print("Runner script started, waiting for Phase 6 to begin...")
    else:
        print("Not running. Log:")
        o3, _ = run(client, f"tail -20 {REMOTE}/logs/round11_ext.log")
        print(o3)

time.sleep(3)
print("\n--- Initial log ---")
o, _ = run(client, f"tail -30 {REMOTE}/logs/round11_ext.log")
print(o or "(empty)")

client.close()
print("\nDone!")
