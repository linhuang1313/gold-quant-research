"""Launch combo test on remote server."""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
PYTHON = "/root/miniconda3/bin/python"
PROJECT = "/root/gold-quant-trading"

def run_cmd(ssh, cmd, timeout=30):
    try:
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        return stdout.read().decode('utf-8', errors='replace').strip()
    except Exception as e:
        return f"ERROR: {e}"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

print("=== Git Pull ===")
out = run_cmd(ssh, f"cd {PROJECT} && git pull")
print(out)

print("\n=== Launch Combo Test ===")
cmd = f"cd {PROJECT} && screen -dmS exp_combo bash -c '{PYTHON} -u run_exp_combo.py > exp_combo_output.txt 2>&1'"
run_cmd(ssh, cmd)
time.sleep(2)

print("\n=== Verify ===")
out = run_cmd(ssh, "screen -ls | grep combo")
print(out if out else "  Not found!")
out = run_cmd(ssh, f"wc -c < {PROJECT}/exp_combo_output.txt 2>/dev/null")
print(f"  Output size: {out} bytes")

out = run_cmd(ssh, "uptime")
print(f"\nServer: {out}")

ssh.close()
print("\nDone!")
