#!/usr/bin/env python3
"""Download R170-R174 results from remote server."""
import paramiko, time, os, sys
from pathlib import Path

HOST = "connect.westd.seetacloud.com"
PORT = 41109
USER = "root"
PASS = "3sCdENtzYfse"

REMOTE_BASE = "/root/gold-quant-research"

RESULT_DIRS = {
    "R170": "results/r170_ml_exit_v3",
    "R171": "results/r171_p6_monte_carlo",
    "R172": "results/r172_regime_weight_optimizer",
    "R173": "results/r173_m1_feature_enhancement",
    "R174": "results/r174_alpha_decay_monitor",
}


def connect():
    for attempt in range(5):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(HOST, port=PORT, username=USER, password=PASS,
                      timeout=30, banner_timeout=30)
            return c
        except Exception as e:
            print(f"  Connect attempt {attempt+1} failed: {e}")
            time.sleep(3)
    raise RuntimeError("Cannot connect after 5 attempts")


def run(ssh, cmd, timeout=30):
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    return o.read().decode(), e.read().decode()


def download_dir(sftp, remote_dir, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    count = 0
    try:
        for item in sftp.listdir_attr(remote_dir):
            remote_path = f"{remote_dir}/{item.filename}"
            local_path = os.path.join(local_dir, item.filename)
            if item.filename.startswith('.'):
                continue
            if stat_is_dir(item):
                download_dir(sftp, remote_path, local_path)
            else:
                sftp.get(remote_path, local_path)
                size_kb = item.st_size / 1024
                print(f"    {item.filename} ({size_kb:.1f} KB)")
                count += 1
    except FileNotFoundError:
        print(f"    (directory not found: {remote_dir})")
    return count


def stat_is_dir(attr):
    import stat
    return stat.S_ISDIR(attr.st_mode)


def main():
    base = Path(__file__).parent.parent

    print("=" * 70)
    print("  Downloading R170-R174 Results")
    print("=" * 70)

    # Check status first
    ssh = connect()
    print("\n  Process status:")
    out, _ = run(ssh, "ps aux | grep 'run_r17[0-4]' | grep -v grep")
    running = []
    for line in out.strip().split('\n'):
        if line.strip():
            parts = line.split()
            for name in ['r170', 'r171', 'r172', 'r173', 'r174']:
                if name in line:
                    running.append(name.upper())
                    break
            pid = parts[1]
            cmd_part = ' '.join(parts[10:])
            print(f"    PID {pid}: {cmd_part[:80]}")

    if running:
        print(f"\n  WARNING: {', '.join(running)} still running. Results may be partial.")
    else:
        print(f"    All experiments completed.")
    ssh.close()

    # Download each
    total_files = 0
    for name, remote_dir in RESULT_DIRS.items():
        local_dir = str(base / remote_dir)
        remote_full = f"{REMOTE_BASE}/{remote_dir}"

        print(f"\n  {name}: {remote_dir}/")
        ssh = connect()
        sftp = ssh.open_sftp()
        count = download_dir(sftp, remote_full, local_dir)
        total_files += count
        sftp.close()
        ssh.close()

    # Also download stdout logs
    print(f"\n  Downloading stdout logs...")
    ssh = connect()
    sftp = ssh.open_sftp()
    for name, remote_dir in RESULT_DIRS.items():
        rnum = name[1:]
        stdout_patterns = [
            f"{REMOTE_BASE}/{remote_dir}/{name.lower()}_stdout.txt",
            f"{REMOTE_BASE}/{remote_dir}/r{rnum}_stdout.txt",
        ]
        for remote_path in stdout_patterns:
            try:
                local_path = str(base / remote_dir / os.path.basename(remote_path))
                sftp.get(remote_path, local_path)
                stat = sftp.stat(remote_path)
                print(f"    {name}: {os.path.basename(remote_path)} ({stat.st_size/1024:.1f} KB)")
                break
            except FileNotFoundError:
                continue
    sftp.close()
    ssh.close()

    print(f"\n{'=' * 70}")
    print(f"  Downloaded {total_files} files total")
    print("=" * 70)


if __name__ == '__main__':
    main()
