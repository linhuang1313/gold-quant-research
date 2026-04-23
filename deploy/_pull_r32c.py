"""Pull R32c results."""
import paramiko, sys, io, time, os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
for attempt in range(3):
    try:
        ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root',
                    password='5zQ8khQzttDN', timeout=120, banner_timeout=120)
        break
    except Exception as e:
        print(f"  SSH attempt {attempt+1} failed: {e}")
        time.sleep(5)

BASE = "/root/gold-quant-research"
LOCAL = r"c:\Users\hlin2\gold-quant-research\results\round32_results"

sftp = ssh.open_sftp()
for f in ['R32c_output.txt', 'R32c_stdout.txt']:
    try:
        sftp.get(f'{BASE}/results/round32_results/{f}', os.path.join(LOCAL, f))
        print(f"  Downloaded: {f}")
    except Exception as e:
        print(f"  Skip {f}: {e}")

sftp.close()
ssh.close()
print("Done!")
