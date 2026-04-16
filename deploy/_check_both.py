#!/usr/bin/env python3
"""Check both servers with error handling."""
import paramiko, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SERVERS = {
    "A": {"host": "connect.westb.seetacloud.com", "port": 42894},
    "B": {"host": "connect.westb.seetacloud.com", "port": 25821},
}

def run(c, cmd):
    _, stdout, _ = c.exec_command(cmd, timeout=30)
    return stdout.read().decode('utf-8', errors='replace').strip()

for name, info in SERVERS.items():
    print(f"\n{'='*50} Server {name} (:{info['port']}) {'='*50}")
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(info["host"], port=info["port"], username="root", password="r1zlTZQUb+E4", timeout=15)
        
        # R6B results
        for f in ["r6_b1_l6_eval.txt", "r6_b2_exit_analysis.txt", "r6_b3_strategy_combo.txt",
                   "r6_b4_interaction.txt", "r6_b5_recent_zoom.txt", "r6_b6_heatmap.txt"]:
            content = run(c, f"cat /root/gold-quant-trading/round6_results/{f} 2>/dev/null")
            if content and len(content) > 100:
                print(f"\n--- {f} ({len(content)} bytes) ---")
                print(content[:4000])
        
        # R6B master
        print(f"\n--- R6B Master ---")
        print(run(c, "cat /root/gold-quant-trading/round6_results/00_master_log.txt 2>/dev/null || echo 'N/A'"))
        
        # R7 master
        print(f"\n--- R7 Master ---")
        print(run(c, "cat /root/gold-quant-trading/round7_results/00_master_log.txt 2>/dev/null || echo 'N/A'"))
        
        # R7-5, R7-6
        for f in ["r7_5_tp_interact.txt", "r7_6_recent_zoom.txt"]:
            content = run(c, f"cat /root/gold-quant-trading/round7_results/{f} 2>/dev/null")
            if content and len(content) > 100:
                print(f"\n--- {f} ---")
                print(content[:3000])
        
        # Process count
        r7 = run(c, "ps aux | grep 'run_round7' | grep python | grep -v grep | wc -l")
        r6b = run(c, "ps aux | grep 'run_round6b' | grep python | grep -v grep | wc -l")
        print(f"\n[Processes] R7={r7}, R6B={r6b}")
        
        c.close()
    except Exception as e:
        print(f"  CONNECTION FAILED: {e}")
