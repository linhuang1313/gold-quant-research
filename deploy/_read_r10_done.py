#!/usr/bin/env python3
"""Read completed R10 result files from server."""
import paramiko, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westd.seetacloud.com", port=30367, username="root", password="r1zlTZQUb+E4", timeout=15)

for f in ["r10_1_l8_construction.txt", "r10_2_l8_kfold.txt"]:
    _, o, _ = c.exec_command(f"cat /root/gold-quant-trading/round10_results/{f}", timeout=30)
    print(f"\n{'='*80}\n=== {f} ===\n{'='*80}")
    print(o.read().decode("utf-8", errors="replace"))

c.close()
