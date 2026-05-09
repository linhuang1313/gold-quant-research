"""Deploy R102 Donchian50 experiment to remote server."""
import paramiko, os, time, base64

SERVER = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASSWD = '3sCdENtzYfse'
REMOTE_BASE = '/root/gold-quant-research'
LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    raise RuntimeError("Cannot connect")


def upload_via_base64(local_path, remote_path):
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
        time.sleep(0.3)
    c = connect()
    _, out, err = c.exec_command(
        f"base64 -d {remote_path}.b64 > {remote_path} && rm {remote_path}.b64", timeout=30)
    rc = out.channel.recv_exit_status()
    if rc != 0:
        raise RuntimeError(f"base64 decode failed: {err.read().decode()}")
    _, out, _ = c.exec_command(f"wc -c {remote_path}", timeout=15)
    remote_size = out.read().decode().strip().split()[0]
    c.close()
    print(f"  Uploaded: {os.path.basename(local_path)} ({len(content)} -> {remote_size} bytes)")


def main():
    print("=" * 60)
    print("Deploying R102 Donchian50")
    print("=" * 60)

    local_file = os.path.join(LOCAL_BASE, 'experiments', 'run_r102_donchian50.py')
    remote_file = f"{REMOTE_BASE}/experiments/run_r102_donchian50.py"

    print("\n[1/3] Uploading...")
    upload_via_base64(local_file, remote_file)

    print("\n[2/3] Launching...")
    c = connect()
    c.exec_command(f"mkdir -p {REMOTE_BASE}/results/r102_donchian50", timeout=15)
    _, out, _ = c.exec_command(
        f"cd {REMOTE_BASE} && nohup python3 -u experiments/run_r102_donchian50.py "
        f"> results/r102_donchian50/r102_stdout.txt 2>&1 &", timeout=30)
    out.channel.recv_exit_status()
    c.close()

    time.sleep(5)
    print("\n[3/3] Verifying...")
    c = connect()
    _, out, _ = c.exec_command("pgrep -f run_r102", timeout=15)
    pid = out.read().decode().strip()
    c.close()
    if pid:
        print(f"  R102 running (PID: {pid})")
    else:
        print("  WARNING: process not found")
        c = connect()
        _, out, _ = c.exec_command(
            f"tail -20 {REMOTE_BASE}/results/r102_donchian50/r102_stdout.txt", timeout=15)
        print(out.read().decode())
        c.close()
        return

    time.sleep(10)
    c = connect()
    _, out, _ = c.exec_command(
        f"head -20 {REMOTE_BASE}/results/r102_donchian50/r102_stdout.txt", timeout=15)
    print(f"\n--- Initial output ---")
    print(out.read().decode()[:500])
    c.close()
    print(f"\nDeploy complete!")


if __name__ == '__main__':
    main()
