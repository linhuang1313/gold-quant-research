"""Upload spread CSV files to remote server via SFTP."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading/data/download"

files = [
    "data/download/xauusd-m15-spread-2015-01-01-2026-04-10.csv",
    "data/download/xauusd-h1-spread-2015-01-01-2026-04-10.csv",
]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
sftp = ssh.open_sftp()

for f in files:
    local = f
    remote = f"{REMOTE_DIR}/{os.path.basename(f)}"
    sz = os.path.getsize(local)
    print(f"  Uploading {os.path.basename(f)} ({sz/1024/1024:.1f} MB)...", end="", flush=True)
    sftp.put(local, remote)
    remote_sz = sftp.stat(remote).st_size
    ok = "OK" if remote_sz == sz else f"MISMATCH ({remote_sz} vs {sz})"
    print(f" {ok}")

# Also upload BID and ASK data if newer versions exist locally
extra = [
    "data/download/xauusd-m15-bid-2015-01-01-2026-04-10.csv",
    "data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv",
    "data/download/xauusd-m15-ask-2015-01-01-2026-04-10.csv",
    "data/download/xauusd-h1-ask-2015-01-01-2026-04-10.csv",
]
for f in extra:
    if os.path.exists(f):
        remote = f"{REMOTE_DIR}/{os.path.basename(f)}"
        sz = os.path.getsize(f)
        print(f"  Uploading {os.path.basename(f)} ({sz/1024/1024:.1f} MB)...", end="", flush=True)
        sftp.put(f, remote)
        remote_sz = sftp.stat(remote).st_size
        ok = "OK" if remote_sz == sz else f"MISMATCH ({remote_sz} vs {sz})"
        print(f" {ok}")

sftp.close()

# Verify
stdin, stdout, stderr = ssh.exec_command(f"ls -la {REMOTE_DIR}/*spread* {REMOTE_DIR}/*2026-04-10*")
print(f"\n=== Remote files ===\n{stdout.read().decode()}")

ssh.close()
print("Done!")
