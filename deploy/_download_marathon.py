"""Download all marathon results from remote server."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading/marathon_results"
LOCAL_DIR = "marathon_results"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

os.makedirs(LOCAL_DIR, exist_ok=True)

sftp = ssh.open_sftp()
try:
    files = sftp.listdir(REMOTE_DIR)
except FileNotFoundError:
    print("No marathon results directory on server!")
    ssh.close()
    sys.exit(1)

print(f"Found {len(files)} files in {REMOTE_DIR}")
for f in sorted(files):
    remote_path = f"{REMOTE_DIR}/{f}"
    local_path = os.path.join(LOCAL_DIR, f)
    try:
        attrs = sftp.stat(remote_path)
        size = attrs.st_size
        sftp.get(remote_path, local_path)
        print(f"  ✅ {f} ({size:,} bytes)")
    except Exception as e:
        print(f"  ❌ {f}: {e}")

# Also download stdout
try:
    sftp.get("/root/gold-quant-trading/marathon_stdout.txt",
             os.path.join(LOCAL_DIR, "marathon_stdout.txt"))
    print(f"  ✅ marathon_stdout.txt")
except Exception as e:
    print(f"  ❌ marathon_stdout.txt: {e}")

# Also download any remaining weekend batch results
for remote_file in ["exp_weekend_batch_output.txt", "exp_choppy_fine_output.txt",
                     "exp_ema100_ablation_output.txt"]:
    try:
        sftp.get(f"/root/gold-quant-trading/{remote_file}",
                 os.path.join(LOCAL_DIR, remote_file))
        print(f"  ✅ {remote_file}")
    except:
        pass

sftp.close()
ssh.close()
print(f"\nAll results downloaded to {LOCAL_DIR}/")
