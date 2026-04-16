#!/usr/bin/env python3
"""Check R7/R6B progress on both servers."""
import paramiko
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SERVERS = {
    "A": {"host": "connect.westb.seetacloud.com", "port": 42894, "user": "root", "password": "r1zlTZQUb+E4"},
    "B": {"host": "connect.westb.seetacloud.com", "port": 25821, "user": "root", "password": "r1zlTZQUb+E4"},
}

def run_cmd(client, cmd):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=60)
    return stdout.read().decode('utf-8', errors='replace')

for name, info in SERVERS.items():
    print(f"\n{'='*60}")
    print(f"Server {name} ({info['port']})")
    print('='*60)
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(info["host"], port=info["port"], username=info["user"], password=info["password"], timeout=30)
    
    # Git version
    print("\n--- Git Version ---")
    print(run_cmd(client, "cd /root/gold-quant-trading && git log --oneline -3"))
    
    # R7 stdout (last 80 lines)
    print("--- R7 Progress (last 80 lines) ---")
    print(run_cmd(client, "tail -80 /root/gold-quant-trading/round7_stdout.txt 2>/dev/null || echo 'NO R7 STDOUT'"))
    
    # R7 results
    print("--- R7 Results ---")
    print(run_cmd(client, "ls -la /root/gold-quant-trading/round7_results/ 2>/dev/null || echo 'NO R7 DIR'"))
    
    # R6B (Server B only)
    if name == "B":
        print("--- R6B Status ---")
        print(run_cmd(client, "cat /root/gold-quant-trading/round6_results/round6b_stdout.txt 2>/dev/null || echo 'NO R6B'"))
    
    client.close()

print("\nDone.")
