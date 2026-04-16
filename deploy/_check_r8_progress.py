#!/usr/bin/env python3
"""Check R8 progress on server."""
import paramiko, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def ssh_exec(client, cmd, timeout=30):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    return stdout.read().decode('utf-8', errors='replace')

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("connect.westd.seetacloud.com", port=30367, username="root", password="r1zlTZQUb+E4", timeout=15)
print("Connected!\n")

ps = ssh_exec(client, "ps aux | grep run_round8 | grep -v grep | wc -l")
print(f"R8 worker processes: {ps.strip()}")

print("\n=== Result files ===")
files = ssh_exec(client, "ls -lhS /root/gold-quant-trading/round8_results/*.txt 2>/dev/null")
print(files.strip() if files.strip() else "No result files yet")

print("\n=== Master log ===")
log = ssh_exec(client, "cat /root/gold-quant-trading/round8_results/00_master_log.txt 2>/dev/null")
print(log.strip() if log.strip() else "No master log yet")

print("\n=== Stdout tail (last 40 lines) ===")
tail = ssh_exec(client, "tail -40 /root/gold-quant-trading/round8_results/round8_stdout.txt 2>/dev/null")
print(tail.strip() if tail.strip() else "No stdout")

client.close()
