"""Deploy R112 via SFTP (more reliable for large data files)."""
import paramiko, time, os

SERVER = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASSWD = '3sCdENtzYfse'
REMOTE_BASE = '/root/gold-quant-research'
LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def conn():
    for i in range(8):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            t = paramiko.Transport((SERVER, PORT))
            t.connect(username=USER, password=PASSWD)
            t.set_keepalive(15)
            c._transport = t
            return c, t
        except Exception as e:
            wait = min(5 * (i + 1), 30)
            print(f"  retry {i+1}: {e}, wait {wait}s")
            time.sleep(wait)
    raise Exception("Cannot connect")


def ssh_conn():
    for i in range(8):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(SERVER, port=PORT, username=USER, password=PASSWD,
                      timeout=120, banner_timeout=60, auth_timeout=60)
            return c
        except Exception as e:
            wait = min(5 * (i + 1), 30)
            print(f"  retry {i+1}: {e}, wait {wait}s")
            time.sleep(wait)
    raise Exception("Cannot connect")


def upload_sftp(local_path, remote_path):
    """Upload file via SFTP with retry."""
    size = os.path.getsize(local_path)
    print(f"  {os.path.basename(local_path)}: {size:,} bytes")
    for attempt in range(3):
        try:
            _, t = conn()
            sftp = paramiko.SFTPClient.from_transport(t)
            sftp.put(local_path, remote_path)
            remote_size = sftp.stat(remote_path).st_size
            sftp.close()
            t.close()
            if remote_size == size:
                print(f"  OK ({remote_size:,} bytes)")
                return True
            else:
                print(f"  SIZE MISMATCH: local={size} remote={remote_size}")
        except Exception as e:
            print(f"  attempt {attempt+1} failed: {e}")
            time.sleep(5)
    return False


def main():
    print("=" * 50)
    print("  Deploying R112 via SFTP")
    print("=" * 50)

    c = ssh_conn()
    c.exec_command(f"mkdir -p {REMOTE_BASE}/data {REMOTE_BASE}/results/r112_pair_trading",
                   timeout=10)[1].channel.recv_exit_status()
    c.close()

    files = [
        ('data/xauusd_daily_yf.csv', f'{REMOTE_BASE}/data/xauusd_daily_yf.csv'),
        ('data/xagusd_daily_yf.csv', f'{REMOTE_BASE}/data/xagusd_daily_yf.csv'),
        ('data/xauusd_h1_yf.csv', f'{REMOTE_BASE}/data/xauusd_h1_yf.csv'),
        ('data/xagusd_h1_yf.csv', f'{REMOTE_BASE}/data/xagusd_h1_yf.csv'),
        ('experiments/run_r112_pair_trading.py', f'{REMOTE_BASE}/experiments/run_r112_pair_trading.py'),
    ]

    for local_rel, remote in files:
        local = os.path.join(LOCAL_BASE, local_rel)
        print(f"\n  Uploading {local_rel}...")
        if not upload_sftp(local, remote):
            print(f"  FAILED! Aborting.")
            return

    print("\n  Launching R112...")
    script = 'run_r112_pair_trading.py'
    c = ssh_conn()
    c.exec_command(f"pkill -f {script}", timeout=10)[1].channel.recv_exit_status()
    c.close()
    time.sleep(2)

    c = ssh_conn()
    _, out, _ = c.exec_command(
        f"cd {REMOTE_BASE} && nohup python3 -u experiments/{script} "
        f"> results/r112_pair_trading/r112_stdout.txt 2>&1 &", timeout=30)
    out.channel.recv_exit_status()
    c.close()
    time.sleep(5)

    c = ssh_conn()
    _, out, _ = c.exec_command(f"pgrep -f {script}", timeout=10)
    pid = out.read().decode().strip()
    c.close()
    if pid:
        print(f"  R112 RUNNING (PID: {pid.split()[0]})")
    else:
        print(f"  R112 NOT RUNNING!")
        c = ssh_conn()
        _, out, _ = c.exec_command(
            f"tail -20 {REMOTE_BASE}/results/r112_pair_trading/r112_stdout.txt", timeout=15)
        print(out.read().decode('utf-8', errors='replace')[:500])
        c.close()

    print("\nDone!")


if __name__ == '__main__':
    main()
