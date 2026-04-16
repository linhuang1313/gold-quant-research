"""Check which server we're actually connecting to."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"

def run_cmd(ssh, cmd, timeout=10):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    return stdout.read().decode('utf-8', errors='replace').strip()

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

print(f"Connected to: {HOST}:{PORT}")
print(f"nproc: {run_cmd(ssh, 'nproc')}")
print(f"hostname: {run_cmd(ssh, 'hostname')}")
print(f"free -h (line 1-2):")
print(run_cmd(ssh, "free -h | head -2"))
print(f"\nGPU:")
print(run_cmd(ssh, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi'"))
print(f"\nCPU model:")
print(run_cmd(ssh, "cat /proc/cpuinfo | grep 'model name' | head -1"))
print(f"\n/etc/motd or banner:")
print(run_cmd(ssh, "head -20 /etc/motd 2>/dev/null; cat /etc/autodl-motd 2>/dev/null; ls /root/autodl* 2>/dev/null || echo 'no autodl files'"))

ssh.close()
