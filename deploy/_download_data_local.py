"""Download 11-year data files from server to local."""
import paramiko, sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westc.seetacloud.com"
PORT = 16005
REMOTE_DIR = "/root/gold-quant-trading/data/download"
LOCAL_DIR = r"c:\Users\hlin2\gold-quant-research\data\download"

os.makedirs(LOCAL_DIR, exist_ok=True)

FILES = [
    "xauusd-m15-bid-2015-01-01-2026-04-10.csv",
    "xauusd-h1-bid-2015-01-01-2026-04-10.csv",
]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username="root", password="r1zlTZQUb+E4", timeout=15)

sftp = c.open_sftp()

for fname in FILES:
    remote_path = f"{REMOTE_DIR}/{fname}"
    local_path = os.path.join(LOCAL_DIR, fname)
    print(f"Downloading {fname}...")
    
    remote_stat = sftp.stat(remote_path)
    size_mb = remote_stat.st_size / 1024 / 1024
    print(f"  Remote size: {size_mb:.1f} MB")
    
    sftp.get(remote_path, local_path)
    
    local_size = os.path.getsize(local_path) / 1024 / 1024
    print(f"  Downloaded: {local_size:.1f} MB -> {local_path}")

sftp.close()
c.close()

print("\nAll files downloaded!")
for fname in FILES:
    local_path = os.path.join(LOCAL_DIR, fname)
    print(f"  {local_path} ({os.path.getsize(local_path)/1024/1024:.1f} MB)")
