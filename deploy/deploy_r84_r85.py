"""Deploy R84+R85 to server."""
import paramiko, os, time

HOST = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASS = '3sCdENtzYfse'
LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = [
    'experiments/run_r84_r85_volsize_triple.py',
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

def upload_one(f):
    local = os.path.join(LOCAL_BASE, f)
    if not os.path.exists(local):
        print(f"  SKIP: {f}"); return True
    for attempt in range(3):
        try:
            c = connect()
            sftp = c.open_sftp()
            sftp.put(local, f"/root/gold-quant-research/{f}")
            sftp.close(); c.close()
            print(f"  OK: {f}"); return True
        except Exception as e:
            print(f"  FAIL (attempt {attempt+1}): {f}: {e}")
            time.sleep(5)
    return False

print("Uploading files...")
for f in FILES:
    if not upload_one(f):
        print(f"FATAL: Could not upload {f}"); exit(1)
    time.sleep(2)

print("\nLaunching R84+R85...")
c = connect()
c.exec_command("mkdir -p /root/gold-quant-research/results/r84_r85/r85_triple_tsmom", timeout=15)
time.sleep(1)
c.exec_command(
    "cd /root/gold-quant-research && "
    "nohup python3 -u experiments/run_r84_r85_volsize_triple.py "
    "> results/r84_r85_stdout.txt 2>&1 &",
    timeout=10
)
time.sleep(5)
_, out, _ = c.exec_command("ps aux | grep r84_r85 | grep -v grep", timeout=15)
proc = out.read().decode('utf-8', errors='replace').strip()
print(f"LAUNCHED OK: {proc[:150]}" if proc else "WARNING: process not found")
c.close()
print("\nDeploy complete!")
