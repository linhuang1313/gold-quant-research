#!/usr/bin/env python3
"""Read completed R8 result files."""
import paramiko, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def ssh_exec(client, cmd, timeout=30):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    return stdout.read().decode('utf-8', errors='replace')

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("connect.westd.seetacloud.com", port=30367, username="root", password="r1zlTZQUb+E4", timeout=15)

for f in ["r8_1_tp_sl_grid.txt", "r8_2_l6_entry_gap.txt"]:
    print(f"\n{'='*70}")
    print(f"  {f}")
    print(f"{'='*70}")
    content = ssh_exec(client, f"cat /root/gold-quant-trading/round8_results/{f} 2>/dev/null")
    print(content.strip() if content.strip() else "NOT FOUND")

client.close()
