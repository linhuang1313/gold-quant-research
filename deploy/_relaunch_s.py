"""Relaunch EXP-S with spread data now available on server."""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
PYTHON = "/root/miniconda3/bin/python"
PROJECT = "/root/gold-quant-trading"

def run_cmd(ssh, cmd, timeout=15):
    try:
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        return stdout.read().decode('utf-8', errors='replace').strip()
    except:
        return ""

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

# Kill old EXP-S if still running
out = run_cmd(ssh, "ps aux | grep 'run_exp_s' | grep -v grep | awk '{print $2}'")
if out:
    for pid in out.strip().split('\n'):
        print(f"  Killing old EXP-S: {pid}")
        run_cmd(ssh, f"kill {pid}")
    time.sleep(1)

# Verify spread file exists
out = run_cmd(ssh, f"ls -la {PROJECT}/data/download/*spread*m15* | head -1")
print(f"Spread file: {out}")

# Launch
cmd = f"cd {PROJECT} && screen -dmS exp_s_rerun bash -c '{PYTHON} -u run_exp_s_historical_spread.py > exp_s_spread_output.txt 2>&1'"
run_cmd(ssh, cmd)
time.sleep(2)

out = run_cmd(ssh, "screen -ls | grep exp_s")
print(f"Screen: {out}")
out = run_cmd(ssh, f"head -10 {PROJECT}/exp_s_spread_output.txt 2>/dev/null")
print(f"Output start:\n{out}")

ssh.close()
print("\nDone!")
