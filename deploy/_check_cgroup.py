"""Check actual cgroup CPU quota vs nproc."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"

def run_cmd(ssh, cmd, timeout=10):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()
    return out, err

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

# Check cgroup v1
out1, _ = run_cmd(ssh, "cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null")
out2, _ = run_cmd(ssh, "cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null")

if out1 and out2:
    quota = int(out1)
    period = int(out2)
    effective_cores = quota / period
    print(f"cgroup v1:")
    print(f"  cfs_quota_us  = {quota}")
    print(f"  cfs_period_us = {period}")
    print(f"  Effective CPU cores = {effective_cores}")
else:
    # Check cgroup v2
    out3, _ = run_cmd(ssh, "cat /sys/fs/cgroup/cpu.max 2>/dev/null")
    if out3:
        parts = out3.split()
        quota = int(parts[0])
        period = int(parts[1])
        effective_cores = quota / period
        print(f"cgroup v2:")
        print(f"  cpu.max = {out3}")
        print(f"  Effective CPU cores = {effective_cores}")

# Check memory limit
out4, _ = run_cmd(ssh, "cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null")
if out4:
    mem_gb = int(out4) / 1024 / 1024 / 1024
    print(f"  Memory limit = {mem_gb:.0f} GB")

# nproc for comparison
out5, _ = run_cmd(ssh, "nproc")
print(f"\nnproc (visible cores) = {out5}")

# Check how many cores python sees
out6, _ = run_cmd(ssh, "/root/miniconda3/envs/3.10/bin/python -c \"import os; print('os.cpu_count():', os.cpu_count()); import multiprocessing; print('mp.cpu_count():', multiprocessing.cpu_count())\"")
print(f"\nPython sees:\n{out6}")

# Check running experiments
out7, _ = run_cmd(ssh, "ps aux | grep python | grep -v grep | wc -l")
print(f"\nRunning python processes: {out7}")

out8, _ = run_cmd(ssh, "ps aux | grep python | grep -v grep")
print(f"\nProcess list:")
for line in out8.split('\n')[:20]:
    print(f"  {line}")

ssh.close()
