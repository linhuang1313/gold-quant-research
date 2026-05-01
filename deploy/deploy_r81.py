"""Deploy R81 SESS_BO definitive validation to server."""
import paramiko, os, time

HOST = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASS = '3sCdENtzYfse'
LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = [
    'experiments/run_r81_sess_bo_validation.py',
    'backtest/validator.py',
    'backtest/stats.py',
    'backtest/__init__.py',
    'backtest/runner.py',
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
            time.sleep(10*(a+1))
    raise RuntimeError("Cannot connect")

c = connect()
print("Connected. Uploading files...")
sftp = c.open_sftp()
for f in FILES:
    local = os.path.join(LOCAL_BASE, f)
    if not os.path.exists(local):
        print(f"  SKIP: {f}"); continue
    sftp.put(local, f"/root/gold-quant-research/{f}")
    print(f"  OK: {f}")
sftp.close()

_, out, _ = c.exec_command("mkdir -p /root/gold-quant-research/results/r81_sess_bo_validation", timeout=15)
out.read()

print("\nLaunching R81...", flush=True)
c.exec_command(
    "cd /root/gold-quant-research && "
    "nohup python3 -u experiments/run_r81_sess_bo_validation.py "
    "> results/r81_stdout.txt 2>&1 &",
    timeout=10
)
time.sleep(5)
_, out, _ = c.exec_command("ps aux | grep r81 | grep -v grep", timeout=15)
try:
    proc = out.read().decode('utf-8', errors='replace').strip()
    print(f"LAUNCHED OK: {proc[:120]}" if proc else "WARNING: process not found")
except:
    print("Process check timeout")
c.close()
print("\nDeploy complete!")
