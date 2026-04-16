#!/usr/bin/env python3
"""SSH deploy script: git pull + run experiments on remote servers."""
import paramiko
import sys
import time
import threading

SERVERS = {
    "A": {"host": "connect.westb.seetacloud.com", "port": 42894, "user": "root", "password": "r1zlTZQUb+E4"},
    "B": {"host": "connect.westb.seetacloud.com", "port": 25821, "user": "root", "password": "r1zlTZQUb+E4"},
}

def ssh_exec(name, info, commands):
    """Connect to server and run commands sequentially."""
    print(f"\n{'='*60}")
    print(f"[Server {name}] Connecting to {info['host']}:{info['port']}...")
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(info["host"], port=info["port"], username=info["user"], password=info["password"], timeout=30)
        print(f"[Server {name}] Connected!")
        
        for cmd in commands:
            print(f"\n[Server {name}] >>> {cmd}")
            stdin, stdout, stderr = client.exec_command(cmd, timeout=300)
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            if out.strip():
                for line in out.strip().split('\n'):
                    print(f"  [Server {name}] {line}")
            if err.strip():
                for line in err.strip().split('\n'):
                    print(f"  [Server {name}] ERR: {line}")
    except Exception as e:
        print(f"[Server {name}] ERROR: {e}")
    finally:
        client.close()
        print(f"\n[Server {name}] Disconnected.")


def check_status():
    """Check both servers status."""
    commands = [
        "hostname && nproc && free -h | head -2",
        "ls /root/gold-quant-trading/ 2>/dev/null && echo 'REPO_EXISTS' || echo 'REPO_NOT_FOUND'",
        "cd /root/gold-quant-trading && git log --oneline -3 2>/dev/null || echo 'NOT_A_GIT_REPO'",
        "ps aux | grep python | grep -v grep | head -10 || echo 'NO_PYTHON_PROCESSES'",
    ]
    
    threads = []
    for name, info in SERVERS.items():
        t = threading.Thread(target=ssh_exec, args=(name, info, commands))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def deploy_and_run(server_name, experiment_script, workers=None):
    """Git pull and run experiment on a specific server."""
    info = SERVERS[server_name]
    
    worker_env = f"export MAX_WORKERS={workers} && " if workers else ""
    script_basename = experiment_script.split('/')[-1].replace('.py', '')
    
    commands = [
        # Setup proxy for GitHub access
        "export https_proxy=http://127.0.0.1:7890 && export http_proxy=http://127.0.0.1:7890 && cd /root/gold-quant-trading && git pull origin main 2>&1 | tail -5",
        # Check data files exist
        "ls -la /root/gold-quant-trading/data/xauusd-m15-bid-*.csv /root/gold-quant-trading/data/xauusd-h1-bid-*.csv 2>/dev/null | wc -l",
        # Run experiment in background with nohup
        f"cd /root/gold-quant-trading && {worker_env}nohup python -u {experiment_script} > round{script_basename.split('round')[-1].split('_')[0]}_results/{script_basename}_stdout.txt 2>&1 &",
        # Verify it started
        "sleep 2 && ps aux | grep python | grep -v grep",
    ]
    
    ssh_exec(server_name, info, commands)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python ssh_deploy.py status          - Check both servers")
        print("  python ssh_deploy.py pull             - Git pull on both servers")
        print("  python ssh_deploy.py run_r6b          - Run R6B on Server B")
        print("  python ssh_deploy.py run_r7           - Run R7 on Server A")
        print("  python ssh_deploy.py run_all          - Run R7 on A, R6B on B")
        sys.exit(1)
    
    action = sys.argv[1]
    
    if action == "status":
        check_status()
    
    elif action == "pull":
        for name, info in SERVERS.items():
            ssh_exec(name, info, [
                "export https_proxy=http://127.0.0.1:7890 && export http_proxy=http://127.0.0.1:7890 && cd /root/gold-quant-trading && git pull origin main 2>&1"
            ])
    
    elif action == "run_r6b":
        deploy_and_run("B", "scripts/experiments/run_round6b.py", workers=16)
    
    elif action == "run_r7":
        deploy_and_run("A", "scripts/experiments/run_round7.py", workers=8)
    
    elif action == "run_all":
        threads = []
        t1 = threading.Thread(target=deploy_and_run, args=("A", "scripts/experiments/run_round7.py", 8))
        t2 = threading.Thread(target=deploy_and_run, args=("B", "scripts/experiments/run_round6b.py", 16))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    
    else:
        print(f"Unknown action: {action}")
