"""Deploy R66-R68 deep anti-overfitting tests to server and launch."""
import paramiko, os, time

HOST = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASS = '3sCdENtzYfse'
LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = [
    'experiments/run_r66_era_bias.py',
    'experiments/run_r67_random_entry.py',
    'experiments/run_r68_h1_fix_impact.py',
    'experiments/run_r66_68_deep_antioverfitting.py',
    'backtest/engine.py',
    'backtest/runner.py',
    'backtest/fast_screen.py',
    'backtest/stats.py',
    'backtest/__init__.py',
    'indicators.py',
    'research_config.py',
]

DIRS = ['results/r66_era_bias', 'results/r67_random_entry', 'results/r68_h1_fix']

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

_, out, _ = c.exec_command("mkdir -p " + " ".join(f"/root/gold-quant-research/{d}" for d in DIRS), timeout=15)
out.read()
print("Dirs created.")

# Kill any old research processes
_, out, _ = c.exec_command("pkill -f 'run_r6[3-8]' 2>/dev/null; pkill -f 'antioverfitting' 2>/dev/null", timeout=10)
try: out.read()
except: pass
time.sleep(2)

c.exec_command(
    "cd /root/gold-quant-research && "
    "nohup python3 -u experiments/run_r66_68_deep_antioverfitting.py "
    "> results/r66_68_stdout.txt 2>&1 &",
    timeout=10
)
time.sleep(5)

_, out, _ = c.exec_command("ps aux | grep r66_68 | grep -v grep", timeout=15)
try:
    proc = out.read().decode('utf-8', errors='replace').strip()
    if proc:
        print(f"LAUNCHED OK: {proc[:120]}")
    else:
        print("WARNING: process not found")
except:
    print("Process check timeout, but launch command was sent")

c.close()
print("Deploy complete!")
