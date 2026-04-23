"""Pull R30 results from remote server."""
import paramiko, sys, io, os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root',
            password='5zQ8khQzttDN', timeout=120, banner_timeout=120)

BASE = "/root/gold-quant-research"
LOCAL = r'c:\Users\hlin2\gold-quant-research\results\round30_results'
os.makedirs(LOCAL, exist_ok=True)

sftp = ssh.open_sftp()
for fname in ['R30_output.txt', 'R30_stdout.txt']:
    remote = f'{BASE}/results/round30_results/{fname}'
    local = os.path.join(LOCAL, fname)
    try:
        sftp.get(remote, local)
        print(f"Downloaded {fname}")
    except Exception as e:
        print(f"Skip {fname}: {e}")
sftp.close()
ssh.close()
print("Done!")
