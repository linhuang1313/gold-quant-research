"""Pull ALL experiment results from Server D1 (port 35258)."""
import paramiko
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westd.seetacloud.com"
PORT = 35258
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_BASE = "/root/gold-quant-trading"
LOCAL_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'd1_all')


def ssh(client, cmd):
    _, o, _ = client.exec_command(cmd, timeout=30)
    return o.read().decode('utf-8', errors='replace').strip()


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    sftp = client.open_sftp()

    # List all result directories
    print("=== Scanning D1 for results ===\n")
    dirs_raw = ssh(client, f"find {REMOTE_BASE} -maxdepth 2 -type d -name '*result*' | sort")
    result_dirs = [d.strip() for d in dirs_raw.split('\n') if d.strip()]
    print(f"Found {len(result_dirs)} result directories:\n")
    for d in result_dirs:
        print(f"  {d}")

    # Pull each directory
    for remote_dir in result_dirs:
        dir_name = remote_dir.replace(REMOTE_BASE + '/', '').replace('/', '_')
        local_dir = os.path.join(LOCAL_BASE, dir_name)
        os.makedirs(local_dir, exist_ok=True)

        try:
            files = sftp.listdir(remote_dir)
        except Exception as e:
            print(f"\n  [ERR] Cannot list {remote_dir}: {e}")
            continue

        txt_files = [f for f in files if f.endswith('.txt') or f.endswith('.json') or f.endswith('.csv')]
        if not txt_files:
            print(f"\n  {dir_name}: no result files")
            continue

        print(f"\n  {dir_name}: {len(txt_files)} files")
        for f in sorted(txt_files):
            remote_path = f"{remote_dir}/{f}"
            local_path = os.path.join(local_dir, f)
            try:
                sftp.get(remote_path, local_path)
                size = os.path.getsize(local_path)
                print(f"    [OK] {f} ({size:,} bytes)")
            except Exception as e:
                print(f"    [ERR] {f}: {e}")

    # Also pull standalone output files
    print("\n=== Standalone output files ===")
    standalone = ssh(client, f"find {REMOTE_BASE} -maxdepth 1 -name '*output*' -o -name '*stdout*' | sort")
    for f in standalone.split('\n'):
        f = f.strip()
        if not f:
            continue
        fname = os.path.basename(f)
        local_path = os.path.join(LOCAL_BASE, fname)
        try:
            sftp.get(f, local_path)
            size = os.path.getsize(local_path)
            print(f"  [OK] {fname} ({size:,} bytes)")
        except Exception as e:
            print(f"  [ERR] {fname}: {e}")

    sftp.close()
    client.close()
    print(f"\n=== All saved to {LOCAL_BASE} ===")


if __name__ == "__main__":
    main()
