#!/usr/bin/env python3
"""Read remaining R8 results (R8-8, R8-9, R8-10)."""
import paramiko, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("connect.westd.seetacloud.com", port=30367, username="root", password="r1zlTZQUb+E4", timeout=15)

for f in ["r8_8_maxhold.txt", "r8_9_choppy.txt", "r8_10_adx.txt"]:
    print(f"\n{'='*70}")
    print(f"  {f}")
    print(f"{'='*70}")
    stdin, stdout, stderr = client.exec_command(f"cat /root/gold-quant-trading/round8_results/{f}", timeout=15)
    print(stdout.read().decode('utf-8', errors='replace').strip())

client.close()
