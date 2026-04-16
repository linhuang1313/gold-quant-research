"""Download Round 5 results from remote server."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westb.seetacloud.com"
PORT = 25821
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"
LOCAL_DIR = "round5_results"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
sftp = ssh.open_sftp()

os.makedirs(LOCAL_DIR, exist_ok=True)

remote_dir = f"{REMOTE_DIR}/round5_results"
try:
    files = sftp.listdir(remote_dir)
    for f in sorted(files):
        remote_path = f"{remote_dir}/{f}"
        local_path = os.path.join(LOCAL_DIR, f)
        print(f"  Downloading {f}...", end='', flush=True)
        sftp.get(remote_path, local_path)
        size = os.path.getsize(local_path)
        print(f" {size/1024:.1f}KB")
except Exception as e:
    print(f"  Error: {e}")

try:
    sftp.get(f"{REMOTE_DIR}/round5_stdout.txt", os.path.join(LOCAL_DIR, "round5_stdout.txt"))
    print("  Downloaded round5_stdout.txt")
except:
    pass

sftp.close()
ssh.close()
print("\nDone!")
