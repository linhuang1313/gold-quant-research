#!/usr/bin/env python3
"""Deploy on both servers: force git pull + start experiments."""
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

def run_cmd(client, cmd, timeout=300):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    return out.strip(), err.strip()

def deploy_server(name, task):
    info = SERVERS[name]
    print(f"\n{'='*60}")
    print(f"[Server {name}] Starting deployment (task: {task})")
    print(f"{'='*60}")
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(info["host"], port=info["port"], username=info["user"], password=info["password"], timeout=30)
    print(f"[Server {name}] Connected!")
    
    # Step 1: Backup local results before reset
    print(f"[Server {name}] Backing up local results...")
    run_cmd(client, "cd /root/gold-quant-trading && cp -r round7_results round7_results_bak 2>/dev/null; cp -r round6_results round6_results_bak 2>/dev/null; cp round7_stdout.txt round7_stdout_bak.txt 2>/dev/null")
    
    # Step 2: Force git pull (reset untracked conflicts)
    print(f"[Server {name}] Force git pull...")
    out, err = run_cmd(client, "cd /root/gold-quant-trading && git fetch origin main && git reset --hard origin/main 2>&1")
    print(f"[Server {name}] git reset: {out[-200:]}")
    
    # Step 3: Restore backed up results
    print(f"[Server {name}] Restoring local results...")
    run_cmd(client, "cd /root/gold-quant-trading && cp -rn round7_results_bak/* round7_results/ 2>/dev/null; cp -rn round6_results_bak/* round6_results/ 2>/dev/null")
    
    # Step 4: Verify version
    out, _ = run_cmd(client, "cd /root/gold-quant-trading && git log --oneline -3")
    print(f"[Server {name}] Git version:\n  {out}")
    
    # Step 5: Check R6B script exists
    out, _ = run_cmd(client, "ls /root/gold-quant-trading/scripts/experiments/run_round6b.py 2>/dev/null && echo EXISTS || echo MISSING")
    print(f"[Server {name}] R6B script: {out}")
    
    # Step 6: Find python path
    out, _ = run_cmd(client, "which python3 2>/dev/null || echo /root/miniconda3/bin/python")
    python_path = out.split('\n')[-1].strip()
    print(f"[Server {name}] Python: {python_path}")
    
    if task == "r6b":
        # Start R6B
        print(f"[Server {name}] Starting R6B (16 workers)...")
        run_cmd(client, f"cd /root/gold-quant-trading && mkdir -p round6_results && nohup {python_path} -u scripts/experiments/run_round6b.py > round6_results/round6b_stdout.txt 2>&1 &")
        time.sleep(5)
        out, _ = run_cmd(client, f"ps aux | grep round6b | grep -v grep")
        print(f"[Server {name}] R6B process: {out}")
        out, _ = run_cmd(client, "head -30 /root/gold-quant-trading/round6_results/round6b_stdout.txt 2>/dev/null || echo 'NO OUTPUT YET'")
        print(f"[Server {name}] R6B output:\n{out[:500]}")
    
    elif task == "check_r7":
        # Just check R7 progress
        out, _ = run_cmd(client, "ps aux | grep 'round7' | grep -v grep | wc -l")
        print(f"[Server {name}] R7 processes: {out}")
        out, _ = run_cmd(client, "cat /root/gold-quant-trading/round7_results/00_master_log.txt 2>/dev/null || echo 'NO LOG'")
        print(f"[Server {name}] R7 master log:\n  {out}")
    
    client.close()
    print(f"\n[Server {name}] Done!")


# Server A: just update code (R7 is running, don't kill it - check progress)
# Server B: update code + start R6B

t_a = threading.Thread(target=deploy_server, args=("A", "check_r7"))
t_b = threading.Thread(target=deploy_server, args=("B", "r6b"))

t_a.start()
t_b.start()
t_a.join()
t_b.join()

print("\n" + "="*60)
print("Deployment complete!")
print("="*60)
