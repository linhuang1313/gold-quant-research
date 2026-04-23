"""Check tick download + deploy R34 if ready."""
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
PY = "/root/miniconda3/bin/python"

def run(cmd, timeout=30):
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors='replace').strip()

ps = run('ps aux | grep download_tick | grep -v grep')
if ps:
    print("[TICK] STILL DOWNLOADING")
else:
    print("[TICK] Download completed or not running")

print("\n[Download log - last 20 lines]")
print(run(f'tail -20 {BASE}/data/tick/download_log.txt 2>/dev/null'))

# Check if tick CSV exists
tick_file = run(f'ls -la {BASE}/data/tick/xauusd_ticks_2025_2026.csv 2>/dev/null')
if tick_file:
    print(f"\n[TICK CSV] {tick_file}")

# Check R34
ps_r34 = run('ps aux | grep run_round34 | grep -v grep')
if ps_r34:
    print("\n[R34] RUNNING")
else:
    print("\n[R34] Not running")

print("\n[R34 output - last 20 lines]")
print(run(f'tail -20 {BASE}/results/round34_results/R34_output.txt 2>/dev/null'))

ssh.close()
