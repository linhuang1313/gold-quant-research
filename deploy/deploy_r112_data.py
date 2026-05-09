"""Upload daily CSVs + launch R112. Skip H1 (not needed for daily backtest)."""
import paramiko, base64, time, os

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
            c.connect(SERVER, port=PORT, username=USER, password=PASSWD,
                      timeout=120, banner_timeout=60, auth_timeout=60)
            return c
        except Exception as e:
            wait = min(5 * (i + 1), 30)
            print(f"  retry {i+1}: {e}, wait {wait}s")
            time.sleep(wait)
    raise Exception("Cannot connect")


def upload_b64(local_path, remote_path):
    with open(local_path, 'rb') as f:
        content = f.read()
    b64 = base64.b64encode(content).decode('ascii')
    chunk_size = 15000
    chunks = [b64[i:i+chunk_size] for i in range(0, len(b64), chunk_size)]
    print(f"  {os.path.basename(local_path)}: {len(content):,} bytes, {len(chunks)} chunks")

    c = conn()
    c.exec_command(f"rm -f {remote_path}.b64", timeout=15)[1].channel.recv_exit_status()
    c.close()
    time.sleep(1)

    for i, chunk in enumerate(chunks):
        op = ">>" if i > 0 else ">"
        for attempt in range(5):
            try:
                c = conn()
                _, out, _ = c.exec_command(f"echo '{chunk}' {op} {remote_path}.b64", timeout=30)
                out.channel.recv_exit_status()
                c.close()
                break
            except Exception as e:
                print(f"    chunk {i+1}/{len(chunks)} attempt {attempt+1}: {e}")
                time.sleep(5)
        if (i + 1) % 10 == 0 or i == len(chunks) - 1:
            print(f"    {i+1}/{len(chunks)} chunks uploaded")
        time.sleep(0.5)

    c = conn()
    _, out, _ = c.exec_command(
        f"base64 -d {remote_path}.b64 > {remote_path} && rm {remote_path}.b64", timeout=30)
    rc = out.channel.recv_exit_status()
    c.close()
    if rc != 0:
        print(f"  DECODE FAILED!")
        return False
    c = conn()
    _, out, _ = c.exec_command(f"wc -c {remote_path}", timeout=15)
    remote_sz = out.read().decode().strip()
    c.close()
    print(f"  OK: {remote_sz}")
    return True


def main():
    print("=" * 50)
    print("  R112 Deploy: upload daily data + launch")
    print("=" * 50)

    c = conn()
    c.exec_command(f"mkdir -p {REMOTE_BASE}/data {REMOTE_BASE}/results/r112_pair_trading",
                   timeout=10)[1].channel.recv_exit_status()
    c.close()

    files = [
        ('data/xauusd_daily_yf.csv', f'{REMOTE_BASE}/data/xauusd_daily_yf.csv'),
        ('data/xagusd_daily_yf.csv', f'{REMOTE_BASE}/data/xagusd_daily_yf.csv'),
    ]

    for local_rel, remote in files:
        local = os.path.join(LOCAL_BASE, local_rel)
        print(f"\nUploading {local_rel}...")
        if not upload_b64(local, remote):
            print(f"  FAILED! Aborting.")
            return

    print(f"\nLaunching R112...")
    script = 'run_r112_pair_trading.py'
    c = conn()
    c.exec_command(f"pkill -f {script}", timeout=10)[1].channel.recv_exit_status()
    c.close()
    time.sleep(2)

    c = conn()
    _, out, _ = c.exec_command(
        f"cd {REMOTE_BASE} && nohup python3 -u experiments/{script} "
        f"> results/r112_pair_trading/r112_stdout.txt 2>&1 &", timeout=30)
    out.channel.recv_exit_status()
    c.close()
    time.sleep(5)

    c = conn()
    _, out, _ = c.exec_command(f"pgrep -f {script}", timeout=10)
    pid = out.read().decode().strip()
    c.close()
    if pid:
        print(f"  R112 RUNNING (PID: {pid.split()[0]})")
    else:
        print(f"  R112 NOT RUNNING!")
        c = conn()
        _, out, _ = c.exec_command(
            f"tail -20 {REMOTE_BASE}/results/r112_pair_trading/r112_stdout.txt", timeout=15)
        print(out.read().decode('utf-8', errors='replace')[:500])
        c.close()

    print("\nDone!")


if __name__ == '__main__':
    main()
