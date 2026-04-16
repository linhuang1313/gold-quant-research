#!/usr/bin/env python3
"""Install missing deps + start R6B on both servers."""
import paramiko
import sys
import io
import time
import threading

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SERVERS = {
    "A": {"host": "connect.westb.seetacloud.com", "port": 42894, "user": "root", "password": "r1zlTZQUb+E4"},
    "B": {"host": "connect.westb.seetacloud.com", "port": 25821, "user": "root", "password": "r1zlTZQUb+E4"},
}

PYTHON = "/root/miniconda3/bin/python"
PIP = "/root/miniconda3/bin/pip"

def run_cmd(client, cmd, timeout=300):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()
    return out, err

def deploy(name, run_r6b=False):
    info = SERVERS[name]
    print(f"\n[Server {name}] Connecting...")
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(info["host"], port=info["port"], username=info["user"], password=info["password"], timeout=30)
    
    # Install python-dotenv
    print(f"[Server {name}] Installing python-dotenv...")
    out, err = run_cmd(client, f"{PIP} install python-dotenv 2>&1")
    print(f"[Server {name}]   {out.split(chr(10))[-1]}")
    
    # Verify
    out, _ = run_cmd(client, f"{PYTHON} -c \"from dotenv import load_dotenv; print('dotenv OK')\" 2>&1")
    print(f"[Server {name}]   {out}")
    
    # Create .env file (empty is fine for backtest, no Telegram needed)
    run_cmd(client, "touch /root/gold-quant-trading/.env")
    
    if run_r6b:
        # Kill old R6B attempts
        run_cmd(client, "pkill -f 'run_round6b' 2>/dev/null; sleep 1")
        
        # Start R6B
        print(f"[Server {name}] Starting R6B...")
        run_cmd(client, f"cd /root/gold-quant-trading && mkdir -p round6_results && nohup {PYTHON} -u scripts/experiments/run_round6b.py > round6_results/round6b_stdout.txt 2>&1 &")
        
        time.sleep(10)
        
        # Verify
        out, _ = run_cmd(client, "ps aux | grep 'round6b' | grep -v grep")
        if out:
            print(f"[Server {name}] R6B running!")
            for line in out.split('\n'):
                print(f"  {line}")
        else:
            print(f"[Server {name}] R6B NOT running, checking error...")
            out, _ = run_cmd(client, "tail -30 /root/gold-quant-trading/round6_results/round6b_stdout.txt 2>/dev/null")
            print(f"  {out[:500]}")
    
    # Check R7 status
    out, _ = run_cmd(client, "cat /root/gold-quant-trading/round7_results/00_master_log.txt 2>/dev/null || echo 'NO R7 LOG'")
    print(f"[Server {name}] R7 log: {out}")
    
    out, _ = run_cmd(client, "ps aux | grep 'round7' | grep -v grep | wc -l")
    print(f"[Server {name}] R7 processes: {out}")
    
    client.close()
    print(f"[Server {name}] Done!")


# Install deps on both, start R6B only on Server B
t_a = threading.Thread(target=deploy, args=("A", False))
t_b = threading.Thread(target=deploy, args=("B", True))

t_a.start()
t_b.start()
t_a.join()
t_b.join()

print("\nAll done!")
