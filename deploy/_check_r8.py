#!/usr/bin/env python3
"""Quick check R8 status on server."""
import paramiko, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westd.seetacloud.com"
PORT = 30367
USER = "root"
PASS = "r1zlTZQUb+E4"

def ssh_exec(client, cmd, timeout=30):
    print(f"  > {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if out.strip():
        for line in out.strip().split('\n')[-30:]:
            print(f"    {line}")
    if err.strip():
        for line in err.strip().split('\n')[-5:]:
            print(f"    [err] {line}")
    return out, err

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
    print("Connected!\n")

    print("=== R8 Process ===")
    ssh_exec(client, "ps aux | grep round8 | grep -v grep")

    print("\n=== Data files ===")
    ssh_exec(client, "ls -lh /root/gold-quant-trading/data/download/xauusd-*bid*.csv 2>/dev/null")

    print("\n=== pip3 check ===")
    ssh_exec(client, "python3 -c 'import dotenv; print(\"dotenv OK\")' 2>&1")

    print("\n=== R8 output dir ===")
    ssh_exec(client, "ls -la /root/gold-quant-trading/round8_results/ 2>/dev/null || echo 'NOT FOUND'")

    print("\n=== Master log ===")
    ssh_exec(client, "cat /root/gold-quant-trading/round8_results/00_master_log.txt 2>/dev/null || echo 'No log'")

    print("\n=== Stdout tail ===")
    ssh_exec(client, "tail -30 /root/gold-quant-trading/round8_results/round8_stdout.txt 2>/dev/null || echo 'No stdout'")

    client.close()

if __name__ == "__main__":
    main()
