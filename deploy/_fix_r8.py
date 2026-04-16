#!/usr/bin/env python3
"""Fix and restart Round 8 on server."""
import paramiko
import sys, io, time, os, glob

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westd.seetacloud.com"
PORT = 30367
USER = "root"
PASS = "r1zlTZQUb+E4"

def ssh_exec(client, cmd, timeout=120):
    print(f"  > {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if out.strip():
        for line in out.strip().split('\n')[-20:]:
            print(f"    {line}")
    if err.strip():
        for line in err.strip().split('\n')[-10:]:
            print(f"    [err] {line}")
    return out, err

def main():
    print(f"Connecting to {HOST}:{PORT}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    print("Connected!\n")

    print("=== Fix 1: Install deps with pip3 ===")
    ssh_exec(client, "pip3 install python-dotenv pandas numpy yfinance 2>&1 | tail -5", timeout=300)

    print("\n=== Fix 2: Create output dir ===")
    ssh_exec(client, "mkdir -p /root/gold-quant-trading/round8_results")

    print("\n=== Fix 3: Upload new data files ===")
    sftp = client.open_sftp()
    ssh_exec(client, "mkdir -p /root/gold-quant-trading/data/download")

    local_dir = r"c:\Users\hlin2\gold-quant-trading\data\download"
    needed = [
        "xauusd-m15-bid-2015-01-01-2026-04-10.csv",
        "xauusd-h1-bid-2015-01-01-2026-04-10.csv",
    ]

    for fname in needed:
        local_path = os.path.join(local_dir, fname)
        remote_path = f"/root/gold-quant-trading/data/download/{fname}"
        if os.path.exists(local_path):
            fsize = os.path.getsize(local_path) / 1024 / 1024
            out, _ = ssh_exec(client, f"ls -la {remote_path} 2>/dev/null || echo 'MISSING'")
            if 'MISSING' in out:
                print(f"  Uploading {fname} ({fsize:.1f} MB)...")
                t0 = time.time()
                sftp.put(local_path, remote_path)
                elapsed = time.time() - t0
                print(f"    Done in {elapsed:.0f}s!")
            else:
                print(f"  {fname} already exists, skipping")
        else:
            print(f"  !!! Local file not found: {local_path}")
    sftp.close()

    print("\n=== Fix 4: Kill any old R8 processes ===")
    ssh_exec(client, "pkill -f run_round8 2>/dev/null; sleep 1")

    print("\n=== Fix 5: Verify data ===")
    ssh_exec(client, "ls -lh /root/gold-quant-trading/data/download/xauusd-*2026-04-10.csv 2>/dev/null || echo 'NEW DATA NOT FOUND'")

    print("\n=== Fix 6: Quick import test ===")
    ssh_exec(client, "cd /root/gold-quant-trading && python3 -c 'from backtest.runner import DataBundle, LIVE_PARITY_KWARGS; d=DataBundle.load_custom(kc_ema=25, kc_mult=1.2); print(f\"M15: {len(d.m15_df)} bars, H1: {len(d.h1_df)} bars\")' 2>&1", timeout=120)

    print("\n=== Fix 7: Start Round 8 ===")
    launcher = """#!/bin/bash
cd /root/gold-quant-trading
mkdir -p round8_results
export PYTHONIOENCODING=utf-8
nohup python3 -u scripts/experiments/run_round8.py > round8_results/round8_stdout.txt 2>&1 &
echo "PID=$!"
"""
    ssh_exec(client, f"cat > /tmp/start_r8.sh << 'SCRIPT'\n{launcher}\nSCRIPT")
    ssh_exec(client, "chmod +x /tmp/start_r8.sh && bash /tmp/start_r8.sh")

    time.sleep(5)
    print("\n=== Verify ===")
    ssh_exec(client, "ps aux | grep run_round8 | grep -v grep | head -3")
    ssh_exec(client, "tail -20 /root/gold-quant-trading/round8_results/round8_stdout.txt 2>/dev/null || echo 'Waiting for output...'")
    ssh_exec(client, "cat /root/gold-quant-trading/round8_results/00_master_log.txt 2>/dev/null || echo 'Log not yet created'")

    print("\n=== DONE ===")
    client.close()

if __name__ == "__main__":
    main()
