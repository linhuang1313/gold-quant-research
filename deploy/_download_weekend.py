"""Download all weekend experiment results from remote server."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
PROJECT = "/root/gold-quant-trading"

FILES = [
    "exp_weekend_batch_output.txt",
    "exp_choppy_fine_output.txt",
    "exp_ema100_ablation_output.txt",
]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
sftp = ssh.open_sftp()

for fname in FILES:
    remote = f"{PROJECT}/{fname}"
    local = fname
    try:
        sftp.stat(remote)
        sftp.get(remote, local)
        size = os.path.getsize(local)
        print(f"  Downloaded {fname} ({size:,} bytes)")
    except FileNotFoundError:
        print(f"  {fname}: not found on server")

sftp.close()
ssh.close()
print("\nDone.")
