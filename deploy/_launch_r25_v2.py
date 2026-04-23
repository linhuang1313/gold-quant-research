"""Launch R25 - fire and forget."""
import paramiko, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root', password='5zQ8khQzttDN', timeout=30)

PY = '/root/miniconda3/bin/python'
BASE = '/root/gold-quant-research'

# Use exec_command but don't wait for output on the nohup command
chan = ssh.get_transport().open_session()
chan.exec_command(
    f'cd {BASE} && pkill -f run_round25 2>/dev/null; sleep 2; '
    f'nohup {PY} -u experiments/run_round25.py > results/round25_results/R25_stdout.txt 2>&1 &'
    f' && sleep 3 && ps aux | grep run_round25 | grep -v grep'
)
chan.settimeout(20)
try:
    data = b''
    while True:
        chunk = chan.recv(4096)
        if not chunk:
            break
        data += chunk
    print(data.decode(errors='replace'))
except Exception:
    print(data.decode(errors='replace') if data else "(timeout - process likely backgrounded)")

time.sleep(2)

# Check in a new channel
_, o, _ = ssh.exec_command('ps aux | grep run_round25 | grep -v grep', timeout=10)
out = o.read().decode(errors='replace').strip()
if 'run_round25' in out:
    print("\nR25 is RUNNING on server!")
    print(out)
else:
    print("\nChecking log...")
    _, o, _ = ssh.exec_command(f'cat {BASE}/results/round25_results/R25_stdout.txt 2>&1 | head -30', timeout=10)
    print(o.read().decode(errors='replace'))

ssh.close()
print("\nDone!")
