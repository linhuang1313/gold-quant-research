"""Download R14 results from Server C."""
import paramiko, sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westc.seetacloud.com"
PORT = 16005
REMOTE_DIR = "/root/gold-quant-trading/round14_results"
LOCAL_DIR = r"c:\Users\hlin2\gold-quant-research\results\round14_results"

os.makedirs(LOCAL_DIR, exist_ok=True)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username="root", password="r1zlTZQUb+E4", timeout=15)

sftp = c.open_sftp()
files = sftp.listdir(REMOTE_DIR)
txt_files = sorted([f for f in files if f.endswith('.txt') and f != 'r14_log.txt'])

print(f"Found {len(txt_files)} result files\n")
for fname in txt_files:
    remote = f"{REMOTE_DIR}/{fname}"
    local = os.path.join(LOCAL_DIR, fname)
    stat = sftp.stat(remote)
    sftp.get(remote, local)
    print(f"  {fname} ({stat.st_size/1024:.1f} KB)")

sftp.close()
c.close()
print(f"\nAll files saved to {LOCAL_DIR}")
