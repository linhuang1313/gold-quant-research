"""Deploy R112: upload script only, server downloads data via yfinance."""
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
    print(f"  {os.path.basename(local_path)}: {len(content)} bytes, {len(chunks)} chunks")

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
                print(f"    chunk {i+1} attempt {attempt+1} failed: {e}")
                time.sleep(5)
        time.sleep(0.8)

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
    print(f"  OK: {out.read().decode().strip()}")
    c.close()
    return True


def main():
    print("=" * 50)
    print("  R112 Deploy v2: script + server-side data download")
    print("=" * 50)

    # 1. Upload experiment script (small file, ~15KB)
    script = 'run_r112_pair_trading.py'
    local = os.path.join(LOCAL_BASE, 'experiments', script)
    remote = f"{REMOTE_BASE}/experiments/{script}"
    print(f"\n1. Uploading {script}...")
    if not upload_b64(local, remote):
        return

    # 2. Create data download script to run on server
    download_script = '''
import sys
try:
    import yfinance as yf
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'yfinance', '-q'])
    import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/root/gold-quant-research/data")
DATA_DIR.mkdir(exist_ok=True)

for symbol, fname in [("GC=F","xauusd_daily_yf.csv"),("SI=F","xagusd_daily_yf.csv")]:
    print(f"Downloading {symbol} daily...")
    df = yf.download(symbol, period="max", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.to_csv(DATA_DIR / fname)
    print(f"  {fname}: {len(df)} rows")

for symbol, fname in [("GC=F","xauusd_h1_yf.csv"),("SI=F","xagusd_h1_yf.csv")]:
    print(f"Downloading {symbol} H1...")
    df = yf.download(symbol, period="730d", interval="1h", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.to_csv(DATA_DIR / fname)
    print(f"  {fname}: {len(df)} rows")

print("All data downloaded!")
'''

    # Write download script to remote
    print("\n2. Creating data download script on server...")
    c = conn()
    c.exec_command(f"mkdir -p {REMOTE_BASE}/data {REMOTE_BASE}/results/r112_pair_trading",
                   timeout=10)[1].channel.recv_exit_status()
    c.close()

    escaped = download_script.replace("'", "'\\''")
    c = conn()
    _, out, err = c.exec_command(
        f"cat > {REMOTE_BASE}/data/_download.py << 'DLEOF'\n{download_script}\nDLEOF",
        timeout=30)
    out.channel.recv_exit_status()
    c.close()

    # 3. Run data download on server
    print("\n3. Downloading data on server (yfinance)...")
    c = conn()
    _, out, err = c.exec_command(
        f"cd {REMOTE_BASE} && python3 data/_download.py", timeout=120)
    stdout = out.read().decode('utf-8', errors='replace')
    stderr = err.read().decode('utf-8', errors='replace')
    c.close()
    print(f"  {stdout}")
    if stderr.strip():
        print(f"  STDERR: {stderr[:300]}")

    # Verify data files exist
    c = conn()
    _, out, _ = c.exec_command(
        f"ls -la {REMOTE_BASE}/data/*.csv 2>/dev/null | tail -6", timeout=15)
    print(f"  Files: {out.read().decode().strip()}")
    c.close()

    # 4. Launch R112
    print(f"\n4. Launching R112...")
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
