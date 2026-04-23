"""Deploy R25 experiment to remote server and run it."""
import paramiko
import os
import sys
from scp import SCPClient
from pathlib import Path

HOST = "connect.bjb1.seetacloud.com"
PORT = 45411
USER = "root"
PASSWD = "5zQ8khQzttDN"

LOCAL_BASE = Path(r"c:\Users\hlin2\gold-quant-research")
REMOTE_BASE = "/root/gold-quant-research"


def ssh_connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASSWD, timeout=30)
    return ssh


def run_cmd(ssh, cmd, show=True):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=60)
    out = stdout.read().decode(errors='replace')
    err = stderr.read().decode(errors='replace')
    if show:
        if out.strip(): print(out.strip())
        if err.strip(): print(f"[STDERR] {err.strip()}")
    return out, err


def upload_files(ssh):
    scp = SCPClient(ssh.get_transport())

    run_cmd(ssh, f"mkdir -p {REMOTE_BASE}/backtest {REMOTE_BASE}/experiments {REMOTE_BASE}/data/download {REMOTE_BASE}/results/round25_results")

    local_files = [
        ("backtest/engine.py", "backtest/engine.py"),
        ("backtest/runner.py", "backtest/runner.py"),
        ("backtest/__init__.py", "backtest/__init__.py"),
        ("experiments/run_round25.py", "experiments/run_round25.py"),
    ]

    for local_rel, remote_rel in local_files:
        local_path = LOCAL_BASE / local_rel
        if local_path.exists():
            remote_path = f"{REMOTE_BASE}/{remote_rel}"
            print(f"  Uploading {local_rel} ...")
            scp.put(str(local_path), remote_path)
        else:
            print(f"  SKIP (not found): {local_rel}")

    data_files = list((LOCAL_BASE / "data" / "download").glob("xauusd-*.csv"))
    for f in data_files:
        print(f"  Uploading data/{f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)...")
        scp.put(str(f), f"{REMOTE_BASE}/data/download/{f.name}")

    scp.close()


def main():
    print("=" * 60)
    print("Deploying R25 to remote server")
    print("=" * 60)

    print("\n[1/4] Connecting...")
    ssh = ssh_connect()
    print("  Connected!")

    print("\n[2/4] Checking server environment...")
    run_cmd(ssh, "python3 --version && pip3 list 2>/dev/null | grep -iE 'numpy|pandas' && free -h | head -2 && nproc")

    print("\n[3/4] Uploading files...")
    upload_files(ssh)

    print("\n[4/4] Installing deps & launching experiment...")
    run_cmd(ssh, f"cd {REMOTE_BASE} && pip3 install numpy pandas --quiet 2>&1 | tail -3")

    # Check __init__.py exists
    init_path = LOCAL_BASE / "backtest" / "__init__.py"
    if not init_path.exists():
        print("  Creating backtest/__init__.py on server...")
        run_cmd(ssh, f"touch {REMOTE_BASE}/backtest/__init__.py")

    # Launch in background with nohup
    launch_cmd = (
        f"cd {REMOTE_BASE} && "
        f"nohup python3 -u experiments/run_round25.py "
        f"> results/round25_results/R25_stdout.txt 2>&1 &"
    )
    run_cmd(ssh, launch_cmd)

    import time
    time.sleep(3)
    print("\n  Checking if process started...")
    out, _ = run_cmd(ssh, "ps aux | grep run_round25 | grep -v grep")
    if "run_round25" in out:
        print("\n  >>> R25 is running on remote server!")
        print(f"  >>> Monitor: ssh -p {PORT} {USER}@{HOST}")
        print(f"  >>> tail -f {REMOTE_BASE}/results/round25_results/R25_stdout.txt")
        print(f"  >>> Output also in: {REMOTE_BASE}/results/round25_results/R25_output.txt")
    else:
        print("\n  WARNING: Process may not have started. Checking log...")
        run_cmd(ssh, f"cat {REMOTE_BASE}/results/round25_results/R25_stdout.txt | head -30")

    ssh.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
