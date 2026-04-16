#!/usr/bin/env python3
"""Check R9 progress."""
import paramiko, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westd.seetacloud.com", port=30367, username="root", password="r1zlTZQUb+E4", timeout=15)

i,o,e = c.exec_command("ps aux | grep run_round9 | grep -v grep | wc -l", timeout=10)
print(f"Workers: {o.read().decode().strip()}")

i,o,e = c.exec_command("ls -lhS /root/gold-quant-trading/round9_results/*.txt 2>/dev/null", timeout=10)
print(f"\nResult files:\n{o.read().decode().strip()}")

i,o,e = c.exec_command("cat /root/gold-quant-trading/round9_results/00_master_log.txt 2>/dev/null", timeout=10)
print(f"\nMaster log:\n{o.read().decode().strip()}")

i,o,e = c.exec_command("tail -20 /root/gold-quant-trading/round9_results/round9_stdout.txt 2>/dev/null", timeout=10)
print(f"\nStdout tail:\n{o.read().decode().strip()}")

c.close()
