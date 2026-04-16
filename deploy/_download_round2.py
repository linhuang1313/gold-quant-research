"""Download Round 2 results from remote server."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"
LOCAL_DIR = "round2_results"

os.makedirs(LOCAL_DIR, exist_ok=True)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
sftp = ssh.open_sftp()

# Download all result files
remote_dir = f"{REMOTE_DIR}/round2_results"
try:
    files = sftp.listdir(remote_dir)
    print(f"Found {len(files)} files in {remote_dir}")
    for f in sorted(files):
        remote_path = f"{remote_dir}/{f}"
        local_path = os.path.join(LOCAL_DIR, f)
        attr = sftp.stat(remote_path)
        sftp.get(remote_path, local_path)
        print(f"  {f}: {attr.st_size:,} bytes")
except Exception as e:
    print(f"Error: {e}")

# Also grab stdout
try:
    sftp.get(f"{REMOTE_DIR}/round2_stdout.txt", os.path.join(LOCAL_DIR, "round2_stdout.txt"))
    print(f"  round2_stdout.txt downloaded")
except:
    pass

sftp.close()
ssh.close()
print(f"\nAll results downloaded to {LOCAL_DIR}/")
