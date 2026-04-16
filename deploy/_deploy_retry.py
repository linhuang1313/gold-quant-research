"""Retry deploy of remaining files and start Phase 6-8."""
import paramiko
import os
import time
from pathlib import Path

HOST = "connect.westd.seetacloud.com"
PORT = 30367
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE = "/root/gold-quant-trading"
ROOT = Path(__file__).parent.parent

files = [
    "scripts/experiments/run_round11.py",
    "paper_trader.py",
]

def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
    return c

def run_cmd(client, cmd, t=15):
    _, out, err = client.exec_command(cmd, timeout=t)
    o = out.read().decode("utf-8", "replace").strip()
    e = err.read().decode("utf-8", "replace").strip()
    return o, e

# Upload with retry
for attempt in range(3):
    try:
        print(f"Attempt {attempt + 1}...")
        client = connect()
        print("Connected!")
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
        print("Upload complete!")
        break
    except Exception as e:
        print(f"  Error: {e}")
        try:
            client.close()
        except:
            pass
        time.sleep(5)
else:
    print("All attempts failed!")
    raise SystemExit(1)

# Kill old process
print("\nKilling old R11 process...")
run_cmd(client, "pkill -f run_round11 || true")
time.sleep(1)

# Verify
print("Verifying imports...")
o, e = run_cmd(client, "cd /root/gold-quant-trading && python3 -c 'from backtest.engine import BacktestEngine; print(\"engine OK\")'")
print(o or e)
o, e = run_cmd(client, "cd /root/gold-quant-trading && python3 -c 'from strategies.signals import prepare_indicators; print(\"signals OK\")'")
print(o or e)

# Create output dirs
run_cmd(client, f"mkdir -p {REMOTE}/round11_results {REMOTE}/logs")

# Write runner shell script
RUNNER = (
    "#!/bin/bash\n"
    "cd /root/gold-quant-trading\n"
    "for phase in 6 7 8; do\n"
    "  echo \"=== Phase $phase ===\"\n"
    "  python3 -u scripts/experiments/run_round11.py $phase\n"
    "done\n"
    "echo \"=== ALL DONE ===\"\n"
)
sftp = client.open_sftp()
with sftp.open(f"{REMOTE}/scripts/run_r11_ext.sh", "w") as fh:
    fh.write(RUNNER)
sftp.close()
run_cmd(client, f"chmod +x {REMOTE}/scripts/run_r11_ext.sh")

# Start
print("\nStarting Phase 6-8...")
client.exec_command(
    f"nohup {REMOTE}/scripts/run_r11_ext.sh > {REMOTE}/logs/round11_ext.log 2>&1 &",
    timeout=5,
)
time.sleep(5)

o, _ = run_cmd(client, "ps aux | grep 'run_r11_ext\\|run_round11' | grep -v grep")
if o:
    print("RUNNING!")
    for line in o.split("\n")[:3]:
        print(f"  {line.strip()[:120]}")
else:
    print("Not running. Checking log:")
    o, _ = run_cmd(client, f"tail -20 {REMOTE}/logs/round11_ext.log")
    print(o or "(empty)")

time.sleep(2)
print("\n--- Initial log ---")
o, _ = run_cmd(client, f"tail -20 {REMOTE}/logs/round11_ext.log")
print(o or "(empty)")

client.close()
print("\nDone!")
