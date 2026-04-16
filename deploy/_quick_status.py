#!/usr/bin/env python3
"""Quick status check after fix."""
import paramiko, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SERVERS = {
    "A": {"host": "connect.westb.seetacloud.com", "port": 42894},
    "B": {"host": "connect.westb.seetacloud.com", "port": 25821},
}

for name, info in SERVERS.items():
    print(f"\n{'='*50} Server {name} {'='*50}")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(info["host"], port=info["port"], username="root", password="r1zlTZQUb+E4", timeout=30)
    
    def run(cmd):
        _, stdout, _ = c.exec_command(cmd, timeout=30)
        return stdout.read().decode('utf-8', errors='replace').strip()
    
    r7 = run("ps aux | grep 'run_round7' | grep python | grep -v grep | wc -l")
    r6b = run("ps aux | grep 'run_round6b' | grep python | grep -v grep | wc -l")
    print(f"  R7 procs: {r7}, R6B procs: {r6b}")
    
    # R7 master log
    print(f"  R7 master: {run('cat /root/gold-quant-trading/round7_results/00_master_log.txt 2>/dev/null | tail -5')}")
    
    # R7 latest file
    print(f"  R7 files: {run('ls -lt /root/gold-quant-trading/round7_results/ 2>/dev/null | head -5')}")
    
    if name == "B":
        # R6B master log
        print(f"  R6B master: {run('cat /root/gold-quant-trading/round6_results/00_master_log.txt 2>/dev/null | tail -5')}")
        # R6B stdout tail
        out = run("tail -5 /root/gold-quant-trading/round6_results/round6b_stdout.txt 2>/dev/null || echo 'N/A'")
        print(f"  R6B stdout: {out[-200:]}")
        # R6B result files
        print(f"  R6B files: {run('ls -lt /root/gold-quant-trading/round6_results/ 2>/dev/null | head -10')}")
    
    c.close()
