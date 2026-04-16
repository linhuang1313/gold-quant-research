#!/usr/bin/env python3
"""Read R7-4 Monte Carlo results from Server A."""
import paramiko, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westb.seetacloud.com", port=42894, username="root", password="r1zlTZQUb+E4", timeout=30)

_, stdout, _ = c.exec_command("cat /root/gold-quant-trading/round7_results/r7_4_monte_carlo.txt 2>/dev/null", timeout=30)
print(stdout.read().decode('utf-8', errors='replace'))

c.close()
