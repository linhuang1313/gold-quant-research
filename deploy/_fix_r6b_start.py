#!/usr/bin/env python3
"""Fix R6B startup on Server B: use correct python with numpy."""
import paramiko
import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SERVER_B = {"host": "connect.westb.seetacloud.com", "port": 25821, "user": "root", "password": "r1zlTZQUb+E4"}

def run_cmd(client, cmd, timeout=300):
    print(f">>> {cmd[:150]}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()
    if out:
        for line in out.split('\n')[-15:]:
            print(f"  {line}")
    if err:
        for line in err.split('\n')[-5:]:
            print(f"  ERR: {line}")
    return out

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(SERVER_B["host"], port=SERVER_B["port"], username=SERVER_B["user"], password=SERVER_B["password"], timeout=30)
print("Connected to Server B!")

# Find the correct python with numpy
print("\n--- Finding python with numpy ---")
run_cmd(client, "/usr/bin/python3 -c 'import numpy; print(numpy.__version__)' 2>&1 || echo 'system python3: NO numpy'")
run_cmd(client, "/root/miniconda3/bin/python -c 'import numpy; print(numpy.__version__)' 2>&1 || echo 'miniconda python: NO numpy'")
run_cmd(client, "find / -name 'python3*' -type f 2>/dev/null | head -10")

# Use miniconda python
python_path = "/root/miniconda3/bin/python"

# Kill any failed R6B process
run_cmd(client, "pkill -f 'run_round6b' 2>/dev/null; sleep 1")

# Start R6B with correct python
print("\n--- Starting R6B ---")
run_cmd(client, f"cd /root/gold-quant-trading && mkdir -p round6_results && nohup {python_path} -u scripts/experiments/run_round6b.py > round6_results/round6b_stdout.txt 2>&1 &")

time.sleep(8)

# Verify
print("\n--- Verify ---")
run_cmd(client, "ps aux | grep 'round6b' | grep -v grep")
run_cmd(client, "wc -l /root/gold-quant-trading/round6_results/round6b_stdout.txt 2>/dev/null")
run_cmd(client, "head -30 /root/gold-quant-trading/round6_results/round6b_stdout.txt 2>/dev/null || echo 'NO OUTPUT'")

client.close()
print("\nDone!")
