"""Deploy R57-R62 overnight research to server and launch."""
import paramiko, os, time, sys

HOST = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASS = '3sCdENtzYfse'

LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES_TO_SYNC = [
    'experiments/run_r57_cooldown_gap.py',
    'experiments/run_r58_sl_tp_cap37.py',
    'experiments/run_r59_session_weight.py',
    'experiments/run_r60_mean_rev_validate.py',
    'experiments/run_r61_6strat_portfolio.py',
    'experiments/run_r62_ml_exit_filter.py',
    'experiments/run_overnight_research.py',
    'backtest/engine.py',
    'backtest/runner.py',
    'backtest/fast_screen.py',
    'backtest/stats.py',
    'backtest/__init__.py',
    'indicators.py',
    'research_config.py',
]

RESULT_DIRS = [
    'results/r57_cooldown_gap',
    'results/r58_sl_tp_cap37',
    'results/r59_session_weight',
    'results/r60_mean_rev_validate',
    'results/r61_6strat_portfolio',
    'results/r62_ml_exit_filter',
]

def ssh_connect(retries=3):
    for attempt in range(retries):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(HOST, port=PORT, username=USER, password=PASS,
                      timeout=90, banner_timeout=180)
            c.get_transport().set_keepalive(30)
            return c
        except Exception as e:
            print(f"  SSH attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
    raise RuntimeError("Cannot connect to server")

def ssh_exec(c, cmd, timeout=120):
    try:
        _, out, err = c.exec_command(cmd, timeout=timeout)
        return out.read().decode('utf-8', errors='replace').strip()
    except Exception as e:
        return f"[ERROR] {e}"

def sftp_upload(c, local_path, remote_path):
    for attempt in range(3):
        try:
            sftp = c.open_sftp()
            sftp.put(local_path, remote_path)
            sftp.close()
            return True
        except Exception as e:
            print(f"    Upload retry {attempt+1}: {e}")
            if attempt < 2:
                time.sleep(5)
                try:
                    c.close()
                except:
                    pass
                c = ssh_connect()
    return False


# ============================================================
# Phase 1: Upload files
# ============================================================
print("=" * 60)
print("  R57-R62 Deployment")
print("=" * 60)

c = ssh_connect()

print("\n--- Phase 1: Upload files ---")
for rel_path in FILES_TO_SYNC:
    local = os.path.join(LOCAL_BASE, rel_path)
    remote = f"/root/gold-quant-research/{rel_path}"
    if not os.path.exists(local):
        print(f"  SKIP (not found): {rel_path}")
        continue
    try:
        sftp = c.open_sftp()
        sftp.put(local, remote)
        sftp.close()
        print(f"  OK: {rel_path}")
    except Exception as e:
        print(f"  FAIL: {rel_path} -> {e}")
        c.close()
        c = ssh_connect()
        try:
            sftp = c.open_sftp()
            sftp.put(local, remote)
            sftp.close()
            print(f"  OK (retry): {rel_path}")
        except Exception as e2:
            print(f"  FAIL (retry): {rel_path} -> {e2}")

# ============================================================
# Phase 2: Create result directories
# ============================================================
print("\n--- Phase 2: Create result directories ---")
for d in RESULT_DIRS:
    remote_dir = f"/root/gold-quant-research/{d}"
    ssh_exec(c, f"mkdir -p {remote_dir}")
    print(f"  OK: {d}")

# ============================================================
# Phase 3: Kill any old research processes
# ============================================================
print("\n--- Phase 3: Kill old research processes ---")
kill_output = ssh_exec(c, "pkill -f 'run_overnight_research\\|run_r5[789]\\|run_r6[012]' 2>/dev/null; echo 'Done'")
print(f"  {kill_output}")
time.sleep(2)

# ============================================================
# Phase 4: Clear old done flags (force re-run)
# ============================================================
print("\n--- Phase 4: Clear old done flags ---")
for d in RESULT_DIRS:
    ssh_exec(c, f"rm -f /root/gold-quant-research/{d}/_done.flag")
print("  Cleared all _done.flag files")

# ============================================================
# Phase 5: Launch orchestrator
# ============================================================
print("\n--- Phase 5: Launch overnight research ---")
launch_cmd = (
    "cd /root/gold-quant-research && "
    "nohup python3 -u experiments/run_overnight_research.py "
    "> results/overnight_stdout.txt 2>&1 &"
)
ssh_exec(c, launch_cmd)
time.sleep(3)

proc_check = ssh_exec(c, "ps aux | grep run_overnight_research | grep -v grep")
if 'run_overnight_research' in proc_check:
    print("  LAUNCHED OK!")
    print(f"  {proc_check}")
else:
    print("  WARNING: Process not found! Checking stdout...")
    print(ssh_exec(c, "tail -20 /root/gold-quant-research/results/overnight_stdout.txt 2>/dev/null"))

# ============================================================
# Phase 6: Verify start
# ============================================================
print("\n--- Phase 6: Quick verify (5s wait) ---")
time.sleep(5)
stdout_tail = ssh_exec(c, "tail -30 /root/gold-quant-research/results/overnight_stdout.txt 2>/dev/null")
print(stdout_tail)

c.close()
print("\n" + "=" * 60)
print("  Deployment complete!")
print("  Monitor: tail -f results/overnight_stdout.txt")
print("=" * 60)
