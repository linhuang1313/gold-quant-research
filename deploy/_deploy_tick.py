"""Deploy tick data download to server."""
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

def run(cmd, timeout=30):
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors='replace').strip()

run(f'mkdir -p {BASE}/data/tick')

print("[TICK] Uploading...")
sftp = ssh.open_sftp()
sftp.put(r'c:\Users\hlin2\gold-quant-research\experiments\download_tick_data.py',
         f'{BASE}/experiments/download_tick_data.py')
sftp.close()

print("[TICK] Launching download (will run in background)...")
chan = ssh.get_transport().open_session()
chan.exec_command(
    f'cd {BASE} && nohup {PY} -u experiments/download_tick_data.py '
    f'> data/tick/download_log.txt 2>&1 &'
)
time.sleep(5)

ps = run('ps aux | grep download_tick | grep -v grep')
if 'download_tick' in ps:
    print("TICK download is RUNNING!")
    print(run(f'head -10 {BASE}/data/tick/download_log.txt'))
else:
    print("WARNING: download may not have started")
    err = run(f'cat {BASE}/data/tick/download_log.txt 2>/dev/null')
    if err: print(err[:1000])

ssh.close()
print("\nDone! (Download will take ~30-60 min for 15 months)")
