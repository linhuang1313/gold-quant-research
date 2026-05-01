"""Deploy R53 TSMOM brute-force to Westd server."""
import paramiko, os, time

HOST = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASS = '3sCdENtzYfse'

LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES_TO_SYNC = [
    'experiments/run_round53_tsmom_brute.py',
    'backtest/engine.py',
    'backtest/runner.py',
    'backtest/fast_screen.py',
    'backtest/stats.py',
    'backtest/__init__.py',
    'indicators.py',
    'research_config.py',
]

def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=60, banner_timeout=120)
    return c

def ssh(c, cmd):
    _, out, _ = c.exec_command(cmd, timeout=30)
    return out.read().decode('utf-8', errors='replace').strip()

print("=== Syncing files ===")
for rel_path in FILES_TO_SYNC:
    local = os.path.join(LOCAL_BASE, rel_path)
    remote = f"/root/gold-quant-research/{rel_path}"
    if not os.path.exists(local):
        print(f"  SKIP: {rel_path}"); continue
    for attempt in range(5):
        try:
            time.sleep(2)
            c = connect()
            sftp = c.open_sftp()
            sftp.put(local, remote)
            sftp.close(); c.close()
            print(f"  OK: {rel_path} ({os.path.getsize(local)/1024:.1f} KB)")
            break
        except Exception as e:
            print(f"  Retry {attempt+1}: {rel_path} ({e})")
            time.sleep(10)

print("\n=== Start R53 ===")
time.sleep(3)
c = connect()
ssh(c, 'mkdir -p /root/gold-quant-research/results/round53_results')
ssh(c, 'pkill -f round53 2>/dev/null; sleep 1')
c.exec_command(
    'cd /root/gold-quant-research && '
    'nohup python3 -u experiments/run_round53_tsmom_brute.py '
    '> results/round53_results/stdout.txt 2>&1 &'
)
time.sleep(5)
print(ssh(c, 'ps aux | grep round53 | grep python | grep -v grep | head -3') or '!! NOT RUNNING !!')
c.close()
print("\nDeploy complete!")
