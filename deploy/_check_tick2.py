"""Detailed check on tick download."""
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

BASE = "/root/gold-quant-research"

def run(cmd, timeout=30):
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors='replace').strip()

# Check process
ps = run('ps aux | grep download_tick | grep -v grep')
print(f"[PS] {ps}")

# Check output file
print(f"\n[LOG full]")
print(run(f'cat {BASE}/data/tick/download_log.txt 2>/dev/null'))

# Check if there are any files in tick dir
print(f"\n[TICK DIR]")
print(run(f'ls -la {BASE}/data/tick/ 2>/dev/null'))

# Check memory/CPU
print(f"\n[TOP]")
print(run(f'ps aux | grep python | head -5'))

ssh.close()
