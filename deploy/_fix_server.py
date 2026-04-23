"""Upload missing files and fix data truncation."""
import paramiko
from scp import SCPClient
from pathlib import Path

HOST = "connect.bjb1.seetacloud.com"
PORT = 45411
USER = "root"
PASSWD = "5zQ8khQzttDN"
LOCAL = Path(r"c:\Users\hlin2\gold-quant-research")
REMOTE = "/root/gold-quant-research"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWD, timeout=30)
scp = SCPClient(ssh.get_transport())

PY = "/root/miniconda3/bin/python"

def run(cmd, timeout=60):
    print(f">>> {cmd}")
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    out = o.read().decode(errors='replace').strip()
    err = e.read().decode(errors='replace').strip()
    if out: print(out)
    if err: print(f"[ERR] {err}")
    return out

def upload(local_rel, remote_path):
    lp = LOCAL / local_rel
    if lp.exists():
        sz = lp.stat().st_size / 1024 / 1024
        print(f"  Uploading {local_rel} ({sz:.1f} MB)...")
        scp.put(str(lp), remote_path)
        return True
    else:
        print(f"  SKIP: {local_rel} not found")
        return False

print("=" * 60)
print("[1] Uploading missing code files...")
upload("research_config.py", f"{REMOTE}/research_config.py")
upload("indicators.py", f"{REMOTE}/indicators.py")

print("\n[2] Uploading correct data files (these are large)...")
upload("data/download/xauusd-m15-bid-2015-01-01-2026-04-10.csv",
       f"{REMOTE}/data/download/xauusd-m15-bid-2015-01-01-2026-04-10.csv")
upload("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv",
       f"{REMOTE}/data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv")

print("\n[3] Verifying data line counts...")
run(f"wc -l {REMOTE}/data/download/xauusd-m15-bid-2015-01-01-2026-04-10.csv")
run(f"wc -l {REMOTE}/data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv")
run(f"tail -2 {REMOTE}/data/download/xauusd-m15-bid-2015-01-01-2026-04-10.csv")

print("\n[4] Import test...")
run(f"cd {REMOTE} && {PY} -c 'from backtest.runner import DataBundle; d = DataBundle.load_default(); print(f\"M15: {{len(d.m15_df)}} bars, H1: {{len(d.h1_df)}} bars\")' 2>&1",
    timeout=120)

print("\n[5] Launching R25...")
run(f"cd {REMOTE} && nohup {PY} -u experiments/run_round25.py > results/round25_results/R25_stdout.txt 2>&1 &")

import time; time.sleep(5)
print("\n[6] Checking process...")
out = run(f"ps aux | grep run_round25 | grep -v grep")
if "run_round25" in out:
    print("\n>>> R25 is running!")
    print(f">>> Monitor: tail -f {REMOTE}/results/round25_results/R25_stdout.txt")
else:
    print("\nWARNING: Not started. Checking log...")
    run(f"head -30 {REMOTE}/results/round25_results/R25_stdout.txt")

scp.close()
ssh.close()
print("\nDone!")
