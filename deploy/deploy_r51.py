"""Deploy R51 independent strategy grids to Westd server."""
import paramiko, os

HOST = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASS = '3sCdENtzYfse'

LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE_BASE = '/root/gold-quant-research'

FILES_TO_SYNC = [
    'experiments/run_round51_independent_grids.py',
    'backtest/engine.py',
    'backtest/runner.py',
    'backtest/fast_screen.py',
    'backtest/stats.py',
    'backtest/__init__.py',
    'indicators.py',
    'research_config.py',
]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30, banner_timeout=60)

sftp = c.open_sftp()

for rel_path in FILES_TO_SYNC:
    local = os.path.join(LOCAL_BASE, rel_path)
    remote = f"{REMOTE_BASE}/{rel_path}"
    if not os.path.exists(local):
        print(f"  SKIP (not found): {rel_path}")
        continue
    remote_dir = os.path.dirname(remote)
    try:
        sftp.stat(remote_dir)
    except FileNotFoundError:
        _, out, _ = c.exec_command(f'mkdir -p {remote_dir}')
        out.channel.recv_exit_status()
    print(f"  Uploading {rel_path}...", end=' ')
    sftp.put(local, remote)
    size = os.path.getsize(local)
    print(f"{size/1024:.1f} KB")

sftp.close()

def ssh(cmd):
    _, out, _ = c.exec_command(cmd, timeout=30)
    return out.read().decode('utf-8', errors='replace').strip()

print("\n=== Create output dir ===")
print(ssh('mkdir -p /root/gold-quant-research/results/round51_results'))

print("\n=== Kill any existing R51 ===")
print(ssh('pkill -f run_round51 2>/dev/null; echo done'))

print("\n=== Start R51 ===")
cmd = ('cd /root/gold-quant-research && '
       'nohup python3 -u experiments/run_round51_independent_grids.py '
       '> results/round51_results/stdout.txt 2>&1 &')
c.exec_command(cmd)

import time; time.sleep(5)

print("\n=== Verify ===")
print(ssh('ps aux | grep round51 | grep python | grep -v grep | head -1') or '!! NOT RUNNING !!')

print("\n=== Load ===")
print(ssh('uptime'))

c.close()
print("\nDeploy complete!")
