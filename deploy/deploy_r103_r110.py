"""
Deploy R103-R110 experiments to remote server.
Batch 1 (parallel): R103, R104, R108, R110
Batch 2 (parallel): R105, R106
Batch 3 (sequential): R107, R109
"""
import paramiko, os, time, base64, sys

SERVER = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASSWD = '3sCdENtzYfse'
REMOTE_BASE = '/root/gold-quant-research'
LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALL_EXPERIMENTS = [
    ('r103', 'run_r103_adaptive_sizing.py'),
    ('r104', 'run_r104_mtf_filter.py'),
    ('r105', 'run_r105_dynamic_rotation.py'),
    ('r106', 'run_r106_production_sim.py'),
    ('r107', 'run_r107_ml_entry.py'),
    ('r108', 'run_r108_tail_hedge.py'),
    ('r109', 'run_r109_meta_ensemble.py'),
    ('r110', 'run_r110_stress_scenarios.py'),
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


def upload_file(local_path, remote_path):
    with open(local_path, 'rb') as f:
        content = f.read()
    b64 = base64.b64encode(content).decode('ascii')
    chunk_size = 50000
    chunks = [b64[i:i+chunk_size] for i in range(0, len(b64), chunk_size)]

    c = connect()
    _, out, _ = c.exec_command(f"rm -f {remote_path}.b64", timeout=15)
    out.channel.recv_exit_status()
    c.close()

    for i, chunk in enumerate(chunks):
        c = connect()
        op = ">>" if i > 0 else ">"
        cmd = f"echo '{chunk}' {op} {remote_path}.b64"
        _, out, _ = c.exec_command(cmd, timeout=30)
        out.channel.recv_exit_status()
        c.close()
        time.sleep(0.2)

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


def deploy_batch(experiments):
    print(f"\n  Uploading {len(experiments)} scripts...")
    for tag, script in experiments:
        local = os.path.join(LOCAL_BASE, 'experiments', script)
        remote = f"{REMOTE_BASE}/experiments/{script}"
        upload_file(local, remote)

    print(f"\n  Launching {len(experiments)} experiments...")
    for tag, script in experiments:
        launch_experiment(tag, script)
        time.sleep(3)


def check_status():
    print("\n  Current running experiments:")
    c = connect()
    for tag, script in ALL_EXPERIMENTS:
        _, out, _ = c.exec_command(f"pgrep -f {script}", timeout=10)
        pid = out.read().decode().strip()
        if pid:
            _, out2, _ = c.exec_command(
                f"tail -1 {REMOTE_BASE}/results/{tag}/{tag}_stdout.txt 2>/dev/null", timeout=10)
            last = out2.read().decode().strip()[:80]
            print(f"    {tag}: RUNNING (PID {pid.split()[0]}) — {last}")
        else:
            _, out2, _ = c.exec_command(
                f"tail -3 {REMOTE_BASE}/results/{tag}/{tag}_stdout.txt 2>/dev/null", timeout=10)
            tail = out2.read().decode().strip()
            if 'COMPLETE' in tail or 'Saved' in tail:
                print(f"    {tag}: DONE")
            elif tail:
                print(f"    {tag}: STOPPED — {tail[-80:]}")
            else:
                print(f"    {tag}: NOT STARTED")
    c.close()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'

    print("=" * 70)
    print("  R103-R110 Deployment Manager")
    print("=" * 70)

    if mode == 'status':
        check_status()
        return

    if mode in ('all', 'batch1'):
        print("\n" + "=" * 60)
        print("  BATCH 1: R103, R104, R108, R110 (parallel, ~3h each)")
        print("=" * 60)
        batch1 = [e for e in ALL_EXPERIMENTS if e[0] in ('r103', 'r104', 'r108', 'r110')]
        deploy_batch(batch1)

    if mode in ('all', 'batch2'):
        print("\n" + "=" * 60)
        print("  BATCH 2: R105, R106 (parallel, ~8h each)")
        print("=" * 60)
        batch2 = [e for e in ALL_EXPERIMENTS if e[0] in ('r105', 'r106')]
        deploy_batch(batch2)

    if mode in ('all', 'batch3'):
        print("\n" + "=" * 60)
        print("  BATCH 3: R107, R109 (parallel, ~15h each)")
        print("=" * 60)
        batch3 = [e for e in ALL_EXPERIMENTS if e[0] in ('r107', 'r109')]
        deploy_batch(batch3)

    print("\n" + "=" * 60)
    print("  Deployment complete. Checking status...")
    print("=" * 60)
    time.sleep(5)
    check_status()


if __name__ == '__main__':
    main()
