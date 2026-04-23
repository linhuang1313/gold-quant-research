"""Upload and run tick API test."""
import paramiko, sys, io, time

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

PY = "/root/miniconda3/bin/python"
BASE = "/root/gold-quant-research"

# Kill old download
def run(cmd, timeout=30):
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors='replace').strip()

run('pkill -f download_tick 2>/dev/null')

sftp = ssh.open_sftp()
sftp.put(r'c:\Users\hlin2\gold-quant-research\experiments\test_tick_api.py',
         f'{BASE}/experiments/test_tick_api.py')
sftp.close()

_, o, e = ssh.exec_command(f'{PY} {BASE}/experiments/test_tick_api.py', timeout=60)
print(o.read().decode(errors='replace'))
err = e.read().decode(errors='replace')
if err: print("STDERR:", err)

ssh.close()
