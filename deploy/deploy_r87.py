"""Deploy R87 Advanced Risk Research to Westd server."""
import paramiko, os, time

HOST = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASS = '3sCdENtzYfse'

LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES_TO_SYNC = [
    'experiments/run_r87_advanced_risk.py',
    'backtest/stats.py',
    'backtest/validator.py',
    'backtest/engine.py',
    'backtest/runner.py',
    'backtest/__init__.py',
    'indicators.py',
    'research_config.py',
]

def connect(retries=5):
    for a in range(retries):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=90, banner_timeout=180)
            c.get_transport().set_keepalive(30)
            return c
        except Exception as e:
            print(f"  Connect {a+1}/{retries}: {e}")
            time.sleep(10 * (a + 1))
    raise RuntimeError("Cannot connect")

def ssh(c, cmd):
    _, out, _ = c.exec_command(cmd, timeout=60)
    return out.read().decode('utf-8', errors='replace').strip()

# Upload files
print("=== Syncing files ===")
for rel_path in FILES_TO_SYNC:
    local = os.path.join(LOCAL_BASE, rel_path)
    remote = f"/root/gold-quant-research/{rel_path}"
    if not os.path.exists(local):
        print(f"  SKIP: {rel_path}")
        continue
    for attempt in range(5):
        try:
            time.sleep(2)
            c = connect()
            sftp = c.open_sftp()
            sftp.put(local, remote)
            sftp.close()
            c.close()
            print(f"  OK: {rel_path} ({os.path.getsize(local)/1024:.1f} KB)")
            break
        except Exception as e:
            print(f"  Retry {attempt+1}: {rel_path} ({e})")
            time.sleep(10)

# Create output dir and launch
print("\n=== Launching R87 ===")
time.sleep(3)
c = connect()
ssh(c, "mkdir -p /root/gold-quant-research/results/r87_advanced_risk")
print(ssh(c, "kill $(pgrep -f r87_advanced_risk) 2>/dev/null; echo 'old process killed'"))
c.close()

time.sleep(3)
c = connect()
cmd = ("cd /root/gold-quant-research && "
       "nohup python3 experiments/run_r87_advanced_risk.py "
       "> results/r87_stdout.txt 2>&1 &")
ssh(c, cmd)
time.sleep(3)
proc = ssh(c, "ps aux | grep r87 | grep -v grep")
print(f"Process: {proc}")
c.close()
print("\nR87 launched! Use _r87_check.py to monitor progress.")
