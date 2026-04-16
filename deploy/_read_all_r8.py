#!/usr/bin/env python3
"""Read all R8 result files."""
import paramiko, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("connect.westd.seetacloud.com", port=30367, username="root", password="r1zlTZQUb+E4", timeout=15)

files = [
    "r8_1_tp_sl_grid.txt", "r8_2_l6_entry_gap.txt", "r8_3_l6_monte_carlo.txt",
    "r8_4_timeout_opt.txt", "r8_5_hist_spread.txt", "r8_6_recent_heatmap.txt",
    "r8_7_sl_sensitivity.txt", "r8_8_maxhold.txt", "r8_9_choppy.txt", "r8_10_adx.txt",
]

for f in files:
    print(f"\n{'='*70}")
    print(f"  {f}")
    print(f"{'='*70}")
    stdin, stdout, stderr = client.exec_command(f"cat /root/gold-quant-trading/round8_results/{f} 2>/dev/null", timeout=30)
    content = stdout.read().decode('utf-8', errors='replace').strip()
    print(content if content else "NOT FOUND")

client.close()
