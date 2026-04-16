#!/usr/bin/env python3
"""Quick fix: install dotenv + start R6B on Server B."""
import paramiko, sys, io, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PYTHON = "/root/miniconda3/bin/python"

for name, port in [("A", 42894), ("B", 25821)]:
    print(f"\n=== Server {name} (:{port}) ===")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("connect.westb.seetacloud.com", port=port, username="root", password="r1zlTZQUb+E4", timeout=30)
    
    # Single command: install dotenv + create .env + verify
    cmd = f"{PYTHON} -m pip install -q python-dotenv 2>&1 | tail -3; touch /root/gold-quant-trading/.env; {PYTHON} -c 'from dotenv import load_dotenv; print(\"dotenv OK\")' 2>&1"
    print(f"  Installing dotenv...")
    stdin, stdout, stderr = c.exec_command(cmd, timeout=120)
    print(f"  {stdout.read().decode('utf-8', errors='replace').strip()}")
    
    if name == "B":
        # Kill old R6B, start new
        c.exec_command("pkill -f 'run_round6b' 2>/dev/null", timeout=10)
        time.sleep(2)
        
        start_cmd = f"cd /root/gold-quant-trading && mkdir -p round6_results && nohup {PYTHON} -u scripts/experiments/run_round6b.py > round6_results/round6b_stdout.txt 2>&1 &"
        print(f"  Starting R6B...")
        c.exec_command(start_cmd, timeout=10)
        time.sleep(8)
        
        # Check
        _, stdout, _ = c.exec_command("ps aux | grep 'round6b' | grep -v grep | wc -l", timeout=10)
        cnt = stdout.read().decode().strip()
        print(f"  R6B processes: {cnt}")
        
        _, stdout, _ = c.exec_command("tail -5 /root/gold-quant-trading/round6_results/round6b_stdout.txt 2>/dev/null", timeout=10)
        print(f"  Last output: {stdout.read().decode('utf-8', errors='replace').strip()[-300:]}")
    
    # R7 status
    _, stdout, _ = c.exec_command("cat /root/gold-quant-trading/round7_results/00_master_log.txt 2>/dev/null", timeout=10)
    log = stdout.read().decode('utf-8', errors='replace').strip()
    print(f"  R7 master log:\n    {log.replace(chr(10), chr(10) + '    ')}")
    
    c.close()

print("\nDone!")
