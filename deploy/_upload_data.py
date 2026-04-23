"""Upload large data files using SFTP with progress."""
import paramiko
import os
from pathlib import Path

HOST = "connect.bjb1.seetacloud.com"
PORT = 45411
USER = "root"
PASSWD = "5zQ8khQzttDN"
LOCAL = Path(r"c:\Users\hlin2\gold-quant-research")
REMOTE = "/root/gold-quant-research"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWD, timeout=30,
            banner_timeout=30)

sftp = ssh.open_sftp()

files = [
    ("data/download/xauusd-m15-bid-2015-01-01-2026-04-10.csv",
     f"{REMOTE}/data/download/xauusd-m15-bid-2015-01-01-2026-04-10.csv"),
    ("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv",
     f"{REMOTE}/data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv"),
]

for local_rel, remote_path in files:
    local_path = LOCAL / local_rel
    if not local_path.exists():
        print(f"SKIP: {local_rel} not found")
        continue

    sz = local_path.stat().st_size
    print(f"Uploading {local_rel} ({sz / 1024 / 1024:.1f} MB)...")

    def progress(transferred, total, name=local_rel):
        pct = transferred / total * 100
        print(f"\r  {name}: {pct:.1f}% ({transferred // 1024}KB / {total // 1024}KB)", end="", flush=True)

    sftp.put(str(local_path), remote_path, callback=progress)
    print(f"\n  Done!")

sftp.close()

# Verify
print("\nVerifying...")
_, o, _ = ssh.exec_command(f"wc -l {REMOTE}/data/download/xauusd-m15-bid-2015-01-01-2026-04-10.csv {REMOTE}/data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv")
print(o.read().decode())

ssh.close()
print("All done!")
