"""
Deploy R119-R130 experiments to remote server.
Batch scheduling for ~90h total runtime.

Batch 1 (~25h): R119 (ML entry) + R120 (ML exit) — ML-heavy, run first
Batch 2 (~15h): R121 (macro regime) + R122 (COT deep) — macro/COT analysis
Batch 3 (~10h): R123 (D1 overlay) — independent timeframe work
Batch 4 (~15h): R125 (dynamic alloc) + R126 (DD rotation) — portfolio construction
Batch 5 (~15h): R127 (param sensitivity) + R128 (regime params) — parameter sweeps
Batch 6 (~10h): R129 (unified validation) + R130 (prod v2) — needs prior results

Usage:
  python deploy/deploy_r119_r130.py all        — upload all + launch batch1
  python deploy/deploy_r119_r130.py status     — check running status
  python deploy/deploy_r119_r130.py batch2     — launch batch2 (after batch1 done)
  python deploy/deploy_r119_r130.py upload     — upload files only
  python deploy/deploy_r119_r130.py download   — download all results
"""
import paramiko, os, time, base64, sys

SERVER = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASSWD = '3sCdENtzYfse'
REMOTE_BASE = '/root/gold-quant-research'
LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALL_EXPERIMENTS = [
    ('r119', 'run_r119_ml_entry_v2.py'),
    ('r120', 'run_r120_ml_exit_v2.py'),
    ('r121', 'run_r121_macro_regime.py'),
    ('r122', 'run_r122_cot_deep.py'),
    ('r123', 'run_r123_daily_overlay.py'),
    ('r125', 'run_r125_dynamic_allocation.py'),
    ('r126', 'run_r126_drawdown_rotation.py'),
    ('r127', 'run_r127_param_sensitivity.py'),
    ('r128', 'run_r128_regime_params.py'),
    ('r129', 'run_r129_unified_validation.py'),
    ('r130', 'run_r130_prod_v2.py'),
]

BATCHES = {
    'batch1': ['r119', 'r120'],
    'batch2': ['r121', 'r122'],
    'batch3': ['r123'],
    'batch4': ['r125', 'r126'],
    'batch5': ['r127', 'r128'],
    'batch6': ['r129', 'r130'],
}

SHARED_FILES = [
    ('backtest/__init__.py', 'backtest/__init__.py'),
    ('backtest/engine.py', 'backtest/engine.py'),
    ('backtest/runner.py', 'backtest/runner.py'),
    ('backtest/stats.py', 'backtest/stats.py'),
    ('indicators.py', 'indicators.py'),
    ('research_config.py', 'research_config.py'),
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
    c = connect()
    c.exec_command(f"mkdir -p {REMOTE_BASE}/experiments {REMOTE_BASE}/backtest "
                   f"{REMOTE_BASE}/data/download {REMOTE_BASE}/data/external", timeout=15)
    c.close()
    time.sleep(1)

    print("\n  Uploading shared dependencies...")
    for local_rel, remote_rel in SHARED_FILES:
        local = os.path.join(LOCAL_BASE, local_rel)
        remote = f"{REMOTE_BASE}/{remote_rel}"
        if os.path.exists(local):
            upload_file(local, remote)
        else:
            print(f"    SKIP (not found): {local_rel}")

    print(f"\n  Uploading {len(ALL_EXPERIMENTS)} experiment scripts...")
    for tag, script in ALL_EXPERIMENTS:
        local = os.path.join(LOCAL_BASE, 'experiments', script)
        remote = f"{REMOTE_BASE}/experiments/{script}"
        upload_file(local, remote)


def deploy_batch(batch_name):
    tags = BATCHES.get(batch_name)
    if not tags:
        print(f"  Unknown batch: {batch_name}. Available: {list(BATCHES.keys())}")
        return

    experiments = [(t, s) for t, s in ALL_EXPERIMENTS if t in tags]
    print(f"\n  Launching {batch_name}: {[e[0] for e in experiments]}")
    for tag, script in experiments:
        launch_experiment(tag, script)
        time.sleep(3)


def check_status():
    print("\n  Current experiment status:")
    print("  " + "-" * 60)
    c = connect()
    for tag, script in ALL_EXPERIMENTS:
        result_dir = f"{REMOTE_BASE}/results/{tag}"
        _, out, _ = c.exec_command(f"pgrep -f {script}", timeout=10)
        pid = out.read().decode().strip()

        _, out2, _ = c.exec_command(
            f"ls -la {result_dir}/{tag}_results.json 2>/dev/null | awk '{{print $5}}'",
            timeout=10)
        json_size = out2.read().decode().strip()

        if pid:
            _, out3, _ = c.exec_command(
                f"tail -1 {result_dir}/{tag}_stdout.txt 2>/dev/null", timeout=10)
            last = out3.read().decode().strip()[:60]
            print(f"    {tag:6s}: RUNNING (PID {pid.split()[0]:>6s}) | {last}")
        elif json_size:
            print(f"    {tag:6s}: DONE    (results: {int(json_size):,} bytes)")
        else:
            _, out3, _ = c.exec_command(
                f"tail -3 {result_dir}/{tag}_stdout.txt 2>/dev/null", timeout=10)
            tail = out3.read().decode().strip()
            if tail:
                status = "STOPPED" if tail else "NOT STARTED"
                print(f"    {tag:6s}: {status:8s} | {tail[-60:]}")
            else:
                print(f"    {tag:6s}: NOT STARTED")
    c.close()


RESULT_DIRS = {
    'r119': 'r119_ml_entry_v2',
    'r120': 'r120_ml_exit_v2',
    'r121': 'r121_macro_regime',
    'r122': 'r122_cot_deep',
    'r123': 'r123_daily_overlay',
    'r125': 'r125_dynamic_allocation',
    'r126': 'r126_drawdown_rotation',
    'r127': 'r127_param_sensitivity',
    'r128': 'r128_regime_params',
    'r129': 'r129_unified_validation',
    'r130': 'r130_prod_v2',
}


def download_results():
    print("\n  Downloading results...")
    c = connect()
    for tag, _ in ALL_EXPERIMENTS:
        result_dir = RESULT_DIRS.get(tag, tag)
        remote_json = f"{REMOTE_BASE}/results/{result_dir}/{tag}_results.json"
        remote_stdout = f"{REMOTE_BASE}/results/{result_dir}/{tag}_stdout.txt"
        # Also check the short-name dir for stdout
        remote_stdout_alt = f"{REMOTE_BASE}/results/{tag}/{tag}_stdout.txt"
        local_dir = os.path.join(LOCAL_BASE, 'results', result_dir)
        os.makedirs(local_dir, exist_ok=True)

        for remote_file in [remote_json, remote_stdout, remote_stdout_alt]:
            fname = os.path.basename(remote_file)
            local_file = os.path.join(local_dir, fname)
            if os.path.exists(local_file) and fname.endswith('_stdout.txt'):
                continue
            try:
                _, out, _ = c.exec_command(f"cat {remote_file} | base64", timeout=120)
                b64_data = out.read().decode().strip()
                if b64_data:
                    content = base64.b64decode(b64_data)
                    with open(local_file, 'wb') as f:
                        f.write(content)
                    print(f"    {result_dir}/{fname}: {len(content):,} bytes")
                else:
                    if not fname.endswith('_stdout.txt'):
                        print(f"    {result_dir}/{fname}: not found")
            except Exception as e:
                print(f"    {result_dir}/{fname}: error ({e})")
                c.close()
                c = connect()
    c.close()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'

    print("=" * 70)
    print("  R119-R130 Mega Research Deployment Manager")
    print("  Total experiments: 11 | Estimated runtime: ~90h")
    print("=" * 70)

    if mode == 'status':
        check_status()
        return

    if mode == 'download':
        download_results()
        return

    if mode == 'upload':
        upload_all()
        print("\n  Upload complete. Use 'batch1' to start first batch.")
        return

    if mode == 'all':
        upload_all()
        print("\n" + "=" * 60)
        print("  BATCH 1: R119 + R120 (ML entry/exit, ~25h)")
        print("=" * 60)
        deploy_batch('batch1')
        print("\n" + "=" * 60)
        print("  Batch 1 launched. After completion, run:")
        print("    python deploy/deploy_r119_r130.py batch2")
        print("    python deploy/deploy_r119_r130.py batch3")
        print("    python deploy/deploy_r119_r130.py batch4")
        print("    python deploy/deploy_r119_r130.py batch5")
        print("  Then finally:")
        print("    python deploy/deploy_r119_r130.py batch6")
        print("=" * 60)

    elif mode.startswith('batch'):
        deploy_batch(mode)

    print("\n  Checking status...")
    time.sleep(3)
    check_status()


if __name__ == '__main__':
    main()
