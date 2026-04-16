#!/usr/bin/env python3
"""Read R6B result files from Server B."""
import paramiko, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westb.seetacloud.com", port=25821, username="root", password="r1zlTZQUb+E4", timeout=30)

def run(cmd):
    _, stdout, _ = c.exec_command(cmd, timeout=30)
    return stdout.read().decode('utf-8', errors='replace').strip()

for f in ["r6_b1_l6_eval.txt", "r6_b2_exit_analysis.txt", "r6_b3_strategy_combo.txt", 
          "r6_b4_interaction.txt", "r6_b5_recent_zoom.txt", "r6_b6_heatmap.txt"]:
    content = run(f"cat /root/gold-quant-trading/round6_results/{f} 2>/dev/null")
    if content:
        print(f"\n{'='*60}")
        print(f"  {f}")
        print('='*60)
        print(content)

c.close()
