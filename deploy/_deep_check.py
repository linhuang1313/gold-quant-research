#!/usr/bin/env python3
"""Deep check: R7 remaining phases + R6B actual state."""
import paramiko, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SERVERS = {
    "A": {"host": "connect.westb.seetacloud.com", "port": 42894, "user": "root", "password": "r1zlTZQUb+E4"},
    "B": {"host": "connect.westb.seetacloud.com", "port": 25821, "user": "root", "password": "r1zlTZQUb+E4"},
}

def run(c, cmd, timeout=30):
    _, stdout, _ = c.exec_command(cmd, timeout=timeout)
    return stdout.read().decode('utf-8', errors='replace').strip()

for name, info in SERVERS.items():
    print(f"\n{'='*70}")
    print(f"Server {name} (:{info['port']})")
    print('='*70)
    
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(info["host"], port=info["port"], username=info["user"], password=info["password"], timeout=30)
    
    # R7 full results
    print("\n[R7 Master Log]")
    print(run(c, "cat /root/gold-quant-trading/round7_results/00_master_log.txt 2>/dev/null || echo 'N/A'"))
    
    # Check all R7 result files
    print("\n[R7 Result Files]")
    print(run(c, "ls -la /root/gold-quant-trading/round7_results/ 2>/dev/null"))
    
    # R7-4 through R7-6
    for f in ["r7_4_monte_carlo.txt", "r7_5_tp_interact.txt", "r7_6_recent_zoom.txt"]:
        content = run(c, f"cat /root/gold-quant-trading/round7_results/{f} 2>/dev/null")
        if content:
            print(f"\n[{f}]")
            print(content[:3000])
    
    # R7 stdout tail
    print("\n[R7 stdout last 20 lines]")
    print(run(c, "tail -20 /root/gold-quant-trading/round7_stdout.txt 2>/dev/null || echo 'N/A'"))
    
    # R6B specific checks (Server B)
    if name == "B":
        print("\n[R6B Master Log]")
        print(run(c, "cat /root/gold-quant-trading/round6_results/00_master_log.txt 2>/dev/null || echo 'N/A'"))
        
        # R6B result files
        print("\n[R6B Result Files]")
        print(run(c, "ls -la /root/gold-quant-trading/round6_results/ 2>/dev/null"))
        
        # New R6B results
        for f in ["r6_b1_l6.txt", "r6_b2_exit.txt", "r6_b3_combo.txt", "r6_b4_interact.txt", "r6_b5_recent.txt"]:
            content = run(c, f"cat /root/gold-quant-trading/round6_results/{f} 2>/dev/null")
            if content:
                print(f"\n[{f}]")
                print(content[:3000])
        
        # R6B stdout last lines
        print("\n[R6B stdout last 30 lines]")
        print(run(c, "tail -30 /root/gold-quant-trading/round6_results/round6b_stdout.txt 2>/dev/null || echo 'N/A'"))
    
    # Process status
    r7 = run(c, "ps aux | grep 'round7' | grep python | grep -v grep | wc -l")
    r6b = run(c, "ps aux | grep 'round6b' | grep python | grep -v grep | wc -l")
    print(f"\n[Processes] R7={r7}, R6B={r6b}")
    
    c.close()

print("\nDone!")
