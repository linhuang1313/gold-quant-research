#!/usr/bin/env python3
"""Check R10 experiment status on remote server."""
import paramiko, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westd.seetacloud.com", port=30367, username="root", password="r1zlTZQUb+E4", timeout=15)

print("=== Process Status ===")
_, o, _ = c.exec_command("ps aux | grep run_round10 | grep -v grep")
print(o.read().decode())

print("=== Result Files ===")
_, o, _ = c.exec_command("ls -lhS /root/gold-quant-trading/round10_results/*.txt 2>/dev/null || echo 'No results yet'")
print(o.read().decode())

print("=== Master Log ===")
_, o, _ = c.exec_command("cat /root/gold-quant-trading/round10_results/00_master_log.txt 2>/dev/null || echo 'No master log yet'")
print(o.read().decode())

print("=== Last 30 lines stdout ===")
_, o, _ = c.exec_command("tail -30 /root/gold-quant-trading/round10_results/round10_stdout.txt 2>/dev/null || echo 'No stdout yet'")
print(o.read().decode())

c.close()
