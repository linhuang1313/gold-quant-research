#!/usr/bin/env python3
"""Deploy Round 8 to new server (25 cores) and start experiment."""
import paramiko
import sys, io, time

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

    print("=== Step 1: Check environment ===")
    ssh_exec(client, "nproc && free -h | head -3")
    ssh_exec(client, "python3 --version 2>&1 || python --version 2>&1")
    ssh_exec(client, "which conda 2>/dev/null || echo 'no conda'")

    print("\n=== Step 2: Check if repo exists ===")
    out, _ = ssh_exec(client, "ls -la /root/gold-quant-trading 2>/dev/null || echo 'NOT_FOUND'")

    if 'NOT_FOUND' in out:
        print("\n=== Step 2b: Clone repo ===")
        ssh_exec(client, "cd /root && git clone https://github.com/linhuang1313/gold-quant-trading.git", timeout=300)
    else:
        print("\n=== Step 2b: Pull latest ===")
        ssh_exec(client, "cd /root/gold-quant-trading && git stash && git pull origin main", timeout=120)

    print("\n=== Step 3: Check data files ===")
    ssh_exec(client, "ls -lh /root/gold-quant-trading/data/download/*.csv 2>/dev/null | tail -5 || echo 'NO DATA'")

    print("\n=== Step 4: Install dependencies ===")
    ssh_exec(client, "cd /root/gold-quant-trading && pip install python-dotenv pandas numpy yfinance 2>&1 | tail -5", timeout=300)

    print("\n=== Step 5: Create .env ===")
    ssh_exec(client, 'echo "TELEGRAM_BOT_TOKEN=\nTELEGRAM_CHAT_ID=" > /root/gold-quant-trading/.env')

    print("\n=== Step 6: Verify R8 script ===")
    ssh_exec(client, "ls -la /root/gold-quant-trading/scripts/experiments/run_round8.py")

    print("\n=== Step 7: Check data availability ===")
    out, _ = ssh_exec(client, "ls /root/gold-quant-trading/data/download/ 2>/dev/null || echo 'NO_DIR'")
    if 'NO_DIR' in out or 'xauusd' not in out.lower():
        print("\n  !!! Data files missing, need to transfer from local")
        print("  Run: scp -P 30367 data/download/xauusd-*.csv root@connect.westd.seetacloud.com:/root/gold-quant-trading/data/download/")
        print("  Or use the SFTP transfer below...")

        print("\n=== Step 7b: Transfer data via SFTP ===")
        sftp = client.open_sftp()
        ssh_exec(client, "mkdir -p /root/gold-quant-trading/data/download")

        import glob, os
        local_data = glob.glob("data/download/xauusd-*.csv")
        if not local_data:
            local_data = glob.glob(r"c:\Users\hlin2\gold-quant-trading\data\download\xauusd-*.csv")

        for f in local_data:
            fname = os.path.basename(f)
            remote = f"/root/gold-quant-trading/data/download/{fname}"
            fsize = os.path.getsize(f) / 1024 / 1024
            print(f"  Uploading {fname} ({fsize:.1f} MB)...")
            sftp.put(f, remote)
            print(f"    Done!")
        sftp.close()
        print("  All data files transferred!")

    print("\n=== Step 8: Start Round 8 ===")
    launcher = """#!/bin/bash
cd /root/gold-quant-trading
export PYTHONIOENCODING=utf-8
nohup python -u scripts/experiments/run_round8.py > round8_results/round8_stdout.txt 2>&1 &
echo $!
"""
    ssh_exec(client, f'cat > /tmp/start_r8.sh << \'SCRIPT\'\n{launcher}\nSCRIPT')
    ssh_exec(client, "chmod +x /tmp/start_r8.sh")
    out, _ = ssh_exec(client, "bash /tmp/start_r8.sh")
    print(f"\n  Round 8 launched! PID: {out.strip()}")

    time.sleep(3)
    print("\n=== Step 9: Verify running ===")
    ssh_exec(client, "ps aux | grep run_round8 | grep -v grep")
    ssh_exec(client, "cat /root/gold-quant-trading/round8_results/00_master_log.txt 2>/dev/null || echo 'Not started yet'")

    print("\n=== DEPLOYMENT COMPLETE ===")
    print(f"Monitor: ssh -p {PORT} {USER}@{HOST}")
    print(f"  tail -f /root/gold-quant-trading/round8_results/round8_stdout.txt")
    print(f"  cat /root/gold-quant-trading/round8_results/00_master_log.txt")

    client.close()

if __name__ == "__main__":
    main()
