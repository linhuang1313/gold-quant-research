#!/usr/bin/env python3
"""Read R7 results + R6B results from both servers."""
import paramiko, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SERVERS = {
    "A": {"host": "connect.westb.seetacloud.com", "port": 42894, "user": "root", "password": "r1zlTZQUb+E4"},
    "B": {"host": "connect.westb.seetacloud.com", "port": 25821, "user": "root", "password": "r1zlTZQUb+E4"},
}

def run_cmd(client, cmd, timeout=60):
    _, stdout, _ = client.exec_command(cmd, timeout=timeout)
    return stdout.read().decode('utf-8', errors='replace').strip()

for name, info in SERVERS.items():
    print(f"\n{'='*70}")
    print(f"Server {name} (:{info['port']})")
    print('='*70)
    
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(info["host"], port=info["port"], username=info["user"], password=info["password"], timeout=30)
    
    # R7 master log
    print("\n--- R7 Master Log ---")
    print(run_cmd(c, "cat /root/gold-quant-trading/round7_results/00_master_log.txt 2>/dev/null || echo 'N/A'"))
    
    # R7 result files
    for f in ["r7_1_baseline.txt", "r7_2_entry_gap.txt", "r7_3_l6_on_l51.txt", "r7_4_monte_carlo.txt", "r7_5_tp_interact.txt", "r7_6_recent_zoom.txt"]:
        content = run_cmd(c, f"cat /root/gold-quant-trading/round7_results/{f} 2>/dev/null")
        if content:
            print(f"\n--- {f} ---")
            print(content)
    
    # R6B results (Server B only)
    if name == "B":
        print("\n--- R6B Master Log ---")
        print(run_cmd(c, "cat /root/gold-quant-trading/round6_results/00_master_log.txt 2>/dev/null || echo 'N/A'"))
        
        for f in ["r6_b1_l6.txt", "r6_b2_exit.txt", "r6_b3_combo.txt", "r6_b4_interact.txt", "r6_b5_recent.txt", "r6_b6_heatmap.txt"]:
            content = run_cmd(c, f"cat /root/gold-quant-trading/round6_results/{f} 2>/dev/null")
            if content:
                print(f"\n--- {f} ---")
                print(content)
        
        # R6B stdout tail
        print("\n--- R6B stdout (last 30 lines) ---")
        print(run_cmd(c, "tail -30 /root/gold-quant-trading/round6_results/round6b_stdout.txt 2>/dev/null || echo 'N/A'"))
    
    # Process count
    r7_cnt = run_cmd(c, "ps aux | grep 'round7' | grep -v grep | wc -l")
    r6b_cnt = run_cmd(c, "ps aux | grep 'round6b' | grep -v grep | wc -l")
    print(f"\n--- Active: R7={r7_cnt} processes, R6B={r6b_cnt} processes ---")
    
    c.close()

print("\nDone!")
