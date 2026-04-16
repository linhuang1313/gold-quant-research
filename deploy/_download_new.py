"""Download newly completed experiment outputs."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko, os

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
PROJECT = "/root/gold-quant-trading"

files = [
    "exp_r_baseline_output.txt",
    "exp_s_spread_output.txt",
    "exp_u_kc_reentry_output.txt",
    "exp_k_regime_bounds_output.txt",
    "exp_w_loss_profile_output.txt",
]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
sftp = ssh.open_sftp()

for f in files:
    remote = f"{PROJECT}/{f}"
    local = f
    try:
        sftp.get(remote, local)
        sz = os.path.getsize(local)
        print(f"  OK {f} ({sz:,} bytes)")
    except Exception as e:
        print(f"  FAIL {f}: {e}")

sftp.close()
ssh.close()
print("Done!")
