"""
Deploy R132-R143 experiments to remote server.
100h Deep Research v2 — 5 batch scheduling.

Batch 1 (~15h): R132 (S4 validation) + R133 (prod v3) — must finish first
Batch 2 (~25h): R134 (overnight/session) + R135 (S3+S4 multi) + R136 (COT macro)
Batch 3 (~20h): R137 (TFT entry) + R138 (RL exit) — ML heavy, serial
Batch 4 (~20h): R139 (extreme detection) + R140 (DD recovery) + R141 (tail risk)
Batch 5 (~20h): R142 (execution analysis) + R143 (paper trade framework)

Usage:
  python deploy/deploy_r132_r143.py all        — upload all + launch batch1
  python deploy/deploy_r132_r143.py status     — check running status
  python deploy/deploy_r132_r143.py batch2     — launch batch2
  python deploy/deploy_r132_r143.py batch3     — launch batch3
  python deploy/deploy_r132_r143.py batch4     — launch batch4
  python deploy/deploy_r132_r143.py batch5     — launch batch5
  python deploy/deploy_r132_r143.py upload     — upload files only
  python deploy/deploy_r132_r143.py download   — download all results
"""
import paramiko, os, time, base64, sys

SERVER = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASSWD = '3sCdENtzYfse'
REMOTE_BASE = '/root/gold-quant-research'
LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALL_EXPERIMENTS = [
    ('r132', 'run_r132_s4_chandelier_validation.py'),
    ('r133', 'run_r133_production_v3.py'),
    ('r134', 'run_r134_overnight_session.py'),
    ('r135', 'run_r135_s3_s4_multi.py'),
    ('r136', 'run_r136_cot_macro_strategy.py'),
    ('r137', 'run_r137_tft_entry.py'),
    ('r138', 'run_r138_rl_exit.py'),
    ('r139', 'run_r139_extreme_detection.py'),
    ('r140', 'run_r140_dd_recovery.py'),
    ('r141', 'run_r141_tail_risk_budget.py'),
    ('r142', 'run_r142_execution_analysis.py'),
    ('r143', 'run_r143_paper_trade_framework.py'),
]

BATCHES = {
    'batch1': ['r132', 'r133'],
    'batch2': ['r134', 'r135', 'r136'],
    'batch3': ['r137', 'r138'],
    'batch4': ['r139', 'r140', 'r141'],
    'batch5': ['r142', 'r143'],
}

SHARED_FILES = [
    ('backtest/__init__.py', 'backtest/__init__.py'),
    ('backtest/engine.py', 'backtest/engine.py'),
    ('backtest/runner.py', 'backtest/runner.py'),
    ('backtest/stats.py', 'backtest/stats.py'),
    ('backtest/validator.py', 'backtest/validator.py'),
    ('indicators.py', 'indicators.py'),
    ('research_config.py', 'research_config.py'),
]

DATA_FILES = [
    ('data/cot_gold_weekly.csv', 'data/cot_gold_weekly.csv'),
]


def connect():
    for attempt in range(5):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(SERVER, port=PORT, username=USER, password=PASSWD,
                      timeout=120, banner_timeout=60, auth_timeout=60)
            return c
        except Exception as e:
            print(f"  connect attempt {attempt+1} failed: {e}")
            time.sleep(5)
    raise RuntimeError("Cannot connect after 5 attempts")


def upload_file(local_path, remote_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            return _upload_file_impl(local_path, remote_path)
        except Exception as e:
            print(f"    Upload attempt {attempt+1} failed for {os.path.basename(local_path)}: {e}")
            if attempt < max_retries - 1:
                time.sleep(10)
            else:
                raise


def _upload_file_impl(local_path, remote_path):
    with open(local_path, 'rb') as f:
        content = f.read()
    b64 = base64.b64encode(content).decode('ascii')
    chunk_size = 40000
    chunks = [b64[i:i+chunk_size] for i in range(0, len(b64), chunk_size)]

    c = connect()
    _, out, _ = c.exec_command(f"rm -f {remote_path}.b64", timeout=15)
    out.channel.recv_exit_status()
    c.close()
    time.sleep(0.5)

    for i, chunk in enumerate(chunks):
        for retry in range(3):
            try:
                c = connect()
                op = ">>" if i > 0 else ">"
                cmd = f"echo '{chunk}' {op} {remote_path}.b64"
                _, out, _ = c.exec_command(cmd, timeout=30)
                out.channel.recv_exit_status()
                c.close()
                break
            except Exception as e:
                if retry < 2:
                    time.sleep(5)
                else:
                    raise
        time.sleep(0.5)

    c = connect()
    _, out, err = c.exec_command(
        f"base64 -d {remote_path}.b64 > {remote_path} && rm {remote_path}.b64", timeout=30)
    rc = out.channel.recv_exit_status()
    if rc != 0:
        raise RuntimeError(f"base64 decode failed: {err.read().decode()}")
    _, out, _ = c.exec_command(f"wc -c {remote_path}", timeout=15)
    remote_size = out.read().decode().strip().split()[0]
    c.close()
    print(f"    Uploaded: {os.path.basename(local_path)} ({len(content)} -> {remote_size} bytes)")


def launch_experiment(tag, script_name):
    result_dir = f"{REMOTE_BASE}/results/{tag}"
    c = connect()
    c.exec_command(f"mkdir -p {result_dir}", timeout=15)
    time.sleep(0.5)
    _, out, _ = c.exec_command(
        f"cd {REMOTE_BASE} && nohup python3 -u experiments/{script_name} "
        f"> {result_dir}/{tag}_stdout.txt 2>&1 &", timeout=30)
    out.channel.recv_exit_status()
    c.close()
    time.sleep(2)

    c = connect()
    _, out, _ = c.exec_command(f"pgrep -f {script_name}", timeout=15)
    pid = out.read().decode().strip()
    c.close()
    if pid:
        print(f"    {tag} running (PID: {pid.split()[0]})")
        return True
    else:
        print(f"    WARNING: {tag} not found running!")
        c = connect()
        _, out, _ = c.exec_command(f"tail -5 {result_dir}/{tag}_stdout.txt", timeout=15)
        print(f"    Last output: {out.read().decode()[:200]}")
        c.close()
        return False


def upload_all():
    print("\n  Creating remote directories...")
    dirs = [
        f"{REMOTE_BASE}/experiments",
        f"{REMOTE_BASE}/backtest",
        f"{REMOTE_BASE}/data",
        f"{REMOTE_BASE}/data/external",
        f"{REMOTE_BASE}/data/download",
    ]
    for d in dirs:
        c = connect()
        _, out, _ = c.exec_command(f"mkdir -p {d}", timeout=15)
        out.channel.recv_exit_status()
        c.close()
        time.sleep(0.3)

    print("\n  Uploading shared framework files...")
    for local_rel, remote_rel in SHARED_FILES:
        local_path = os.path.join(LOCAL_BASE, local_rel)
        remote_path = f"{REMOTE_BASE}/{remote_rel}"
        if os.path.exists(local_path):
            upload_file(local_path, remote_path)
        else:
            print(f"    SKIP (not found): {local_rel}")

    print("\n  Uploading data files...")
    for local_rel, remote_rel in DATA_FILES:
        local_path = os.path.join(LOCAL_BASE, local_rel)
        remote_path = f"{REMOTE_BASE}/{remote_rel}"
        if os.path.exists(local_path):
            upload_file(local_path, remote_path)
        else:
            print(f"    SKIP (not found): {local_rel}")

    print("\n  Uploading experiment scripts...")
    for tag, script in ALL_EXPERIMENTS:
        local_path = os.path.join(LOCAL_BASE, 'experiments', script)
        remote_path = f"{REMOTE_BASE}/experiments/{script}"
        upload_file(local_path, remote_path)


def launch_batch(batch_name):
    tags = BATCHES.get(batch_name)
    if not tags:
        print(f"  Unknown batch: {batch_name}")
        return
    print(f"\n  Launching {batch_name}: {tags}")
    script_map = dict(ALL_EXPERIMENTS)
    for tag in tags:
        print(f"\n  Starting {tag}...")
        launch_experiment(tag, script_map[tag])
        time.sleep(3)


def check_status():
    print("\n  Checking experiment status...")
    c = connect()
    _, out, _ = c.exec_command("ps aux | grep 'run_r1[3-4]' | grep -v grep", timeout=15)
    procs = out.read().decode().strip()
    c.close()

    if procs:
        print(f"\n  Running processes:")
        for line in procs.split('\n'):
            parts = line.split()
            if len(parts) > 10:
                pid = parts[1]
                cmd = ' '.join(parts[10:])
                print(f"    PID {pid}: {cmd}")
    else:
        print("    No R132-R143 experiments currently running.")

    print("\n  Result files:")
    c = connect()
    for tag, _ in ALL_EXPERIMENTS:
        result_dir = f"{REMOTE_BASE}/results/{tag}"
        _, out, _ = c.exec_command(f"ls -la {result_dir}/ 2>/dev/null | tail -3", timeout=15)
        files = out.read().decode().strip()
        if files:
            _, out2, _ = c.exec_command(
                f"test -f {result_dir}/{tag}_results.json && echo 'DONE' || echo 'RUNNING'",
                timeout=15)
            status = out2.read().decode().strip()
            print(f"    {tag}: {status}")
        else:
            print(f"    {tag}: NOT STARTED")
    c.close()


def download_results():
    print("\n  Downloading results...")
    os.makedirs(os.path.join(LOCAL_BASE, 'results'), exist_ok=True)

    for tag, _ in ALL_EXPERIMENTS:
        local_dir = os.path.join(LOCAL_BASE, 'results', tag)
        os.makedirs(local_dir, exist_ok=True)
        remote_dir = f"{REMOTE_BASE}/results/{tag}"

        c = connect()
        _, out, _ = c.exec_command(f"ls {remote_dir}/ 2>/dev/null", timeout=15)
        files = out.read().decode().strip().split('\n')
        c.close()

        if not files or files == ['']:
            print(f"    {tag}: no results yet")
            continue

        for fname in files:
            fname = fname.strip()
            if not fname:
                continue
            remote_file = f"{remote_dir}/{fname}"
            local_file = os.path.join(local_dir, fname)

            c = connect()
            _, out, _ = c.exec_command(f"base64 {remote_file}", timeout=60)
            b64_data = out.read().decode().strip()
            c.close()

            if b64_data:
                content = base64.b64decode(b64_data)
                with open(local_file, 'wb') as f:
                    f.write(content)
        print(f"    {tag}: downloaded {len(files)} files")


def install_deps():
    """Install required Python packages on remote server."""
    print("\n  Installing dependencies...")
    c = connect()
    deps = "xgboost scikit-learn scipy hmmlearn"
    _, out, err = c.exec_command(
        f"pip install {deps} -q 2>&1 | tail -3", timeout=120)
    print(f"    {out.read().decode().strip()}")
    c.close()

    c = connect()
    _, out, _ = c.exec_command("pip install torch --index-url https://download.pytorch.org/whl/cpu -q 2>&1 | tail -3", timeout=300)
    print(f"    PyTorch: {out.read().decode().strip()}")
    c.close()


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if action == 'all':
        upload_all()
        install_deps()
        launch_batch('batch1')
    elif action == 'upload':
        upload_all()
    elif action == 'deps':
        install_deps()
    elif action.startswith('batch'):
        launch_batch(action)
    elif action == 'status':
        check_status()
    elif action == 'download':
        download_results()
    elif action == 'launch_all':
        for batch in ['batch1', 'batch2', 'batch3', 'batch4', 'batch5']:
            launch_batch(batch)
            time.sleep(5)
    else:
        print(f"  Unknown action: {action}")
        print("  Usage: python deploy/deploy_r132_r143.py [all|upload|deps|batch1-5|status|download|launch_all]")


if __name__ == '__main__':
    main()
