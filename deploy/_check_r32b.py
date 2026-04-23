"""Check R32b status."""
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

ps = run('ps aux | grep run_round32b | grep -v grep')
if ps:
    print("[R32b] RUNNING")
    print(ps[:200])
else:
    print("[R32b] NOT running (may be completed)")

print("\n[R32b STDOUT - last 80 lines]")
print(run(f'tail -80 {BASE}/results/round32_results/R32b_stdout.txt 2>/dev/null'))

print("\n[R32b OUTPUT - last 50 lines]")
print(run(f'tail -50 {BASE}/results/round32_results/R32b_output.txt 2>/dev/null'))

ssh.close()
