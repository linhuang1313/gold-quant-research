"""
Deploy R24 via GitHub clone (with academic proxy) and run on remote server.
"""
import paramiko
import os
import time
from pathlib import Path

HOST = "connect.westd.seetacloud.com"
PORT = 45630
USER = "root"
PASS = "r1zlTZQUb+E4"

REMOTE_DIR = "/root/gold-quant-research"
REPO_URL = "https://github.com/linhuang1313/gold-quant-research.git"
LOCAL_ROOT = Path(__file__).parent.parent


def ssh_connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=60)
    client.get_transport().set_keepalive(5)
    time.sleep(0.5)
    return client


def run_cmd(client, cmd, timeout=600):
    print(f"  $ {cmd[:150]}")
    transport = client.get_transport()
    chan = transport.open_session()
    chan.settimeout(timeout)
    chan.exec_command(cmd)

    out = b''
    err = b''
    while True:
        time.sleep(0.1)
        if chan.recv_ready():
            out += chan.recv(65536)
        if chan.recv_stderr_ready():
            err += chan.recv_stderr(65536)
        if chan.exit_status_ready():
            while chan.recv_ready():
                out += chan.recv(65536)
            while chan.recv_stderr_ready():
                err += chan.recv_stderr(65536)
            break

    rc = chan.recv_exit_status()
    chan.close()
    out_s = out.decode('utf-8', errors='replace')
    err_s = err.decode('utf-8', errors='replace')
    if out_s.strip():
        lines = out_s.strip().split('\n')
        for line in lines[:20]:
            print(f"    {line}")
        if len(lines) > 20:
            print(f"    ... ({len(lines)} lines)")
    if err_s.strip() and rc != 0:
        for line in err_s.strip().split('\n')[:10]:
            print(f"    [ERR] {line}")
    print(f"    [rc={rc}]")
    return out_s, err_s, rc


def main():
    print("=" * 60)
    print("R24: Deploy via GitHub + Academic Proxy")
    print("=" * 60)

    # 1. Check env + setup proxy
    print("\n[1/5] Environment check + proxy setup")
    c = ssh_connect()
    run_cmd(c, "python3 --version && nproc && free -h | head -2")
    run_cmd(c, "pip3 install numpy pandas scipy 2>&1 | tail -3", timeout=180)

    # Setup git proxy via academic acceleration
    run_cmd(c, "which git || apt-get install -y git 2>&1 | tail -3", timeout=120)

    # Check if academic proxy (clash/v2ray) is available
    run_cmd(c, "curl -s --connect-timeout 5 http://127.0.0.1:7890 >/dev/null 2>&1 && echo 'proxy:7890 OK' || echo 'no proxy on 7890'")
    run_cmd(c, "curl -s --connect-timeout 5 http://127.0.0.1:7891 >/dev/null 2>&1 && echo 'proxy:7891 OK' || echo 'no proxy on 7891'")

    # Try common academic acceleration proxy configs
    # SeetaCloud typically has proxy on specific ports
    run_cmd(c, "env | grep -i proxy || echo 'no proxy env vars'")
    c.close()

    # 2. Git clone (or pull)
    print("\n[2/5] Git clone / pull via academic proxy")
    c = ssh_connect()

    # Check if repo exists
    out, _, rc = run_cmd(c, f"test -d {REMOTE_DIR}/.git && echo 'EXISTS' || echo 'NEED_CLONE'")

    if 'NEED_CLONE' in out:
        print("  Cloning repository...")
        # Try with and without proxy
        clone_cmd = (
            f"export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890; "
            f"git clone {REPO_URL} {REMOTE_DIR} 2>&1 || "
            f"(unset https_proxy http_proxy && git clone {REPO_URL} {REMOTE_DIR} 2>&1)"
        )
        run_cmd(c, clone_cmd, timeout=600)
    else:
        print("  Repo exists, pulling latest...")
        pull_cmd = (
            f"cd {REMOTE_DIR} && "
            f"export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890; "
            f"git pull origin main 2>&1 || "
            f"(unset https_proxy http_proxy && git pull origin main 2>&1)"
        )
        run_cmd(c, pull_cmd, timeout=300)

    # Verify files
    run_cmd(c, f"ls -la {REMOTE_DIR}/backtest/engine.py {REMOTE_DIR}/experiments/run_round24.py {REMOTE_DIR}/data/download/ 2>&1")
    run_cmd(c, f"wc -l {REMOTE_DIR}/data/download/xauusd-m15-bid-*.csv 2>/dev/null || echo 'no M15 data'")
    c.close()

    # 3. Create results dir + launch
    print("\n[3/5] Launching R24 experiment")
    c = ssh_connect()
    run_cmd(c, f"mkdir -p {REMOTE_DIR}/results/round24_results")
    run_cmd(c, (
        f"cd {REMOTE_DIR} && "
        f"PYTHONIOENCODING=utf-8 nohup python3 -u experiments/run_round24.py "
        f"> results/round24_results/stdout.txt 2>&1 &"
    ))
    time.sleep(3)
    run_cmd(c, f"ps aux | grep run_round24 | grep -v grep")
    run_cmd(c, f"head -20 {REMOTE_DIR}/results/round24_results/stdout.txt 2>/dev/null")
    c.close()

    # 4. Poll until done
    print("\n[4/5] Monitoring (poll every 30s, max 60min)")
    for i in range(120):
        time.sleep(30)
        try:
            c = ssh_connect()
            out, _, _ = run_cmd(c, f"tail -5 {REMOTE_DIR}/results/round24_results/stdout.txt 2>/dev/null")
            out2, _, _ = run_cmd(c, f"ps aux | grep 'python3.*run_round24' | grep -v grep | wc -l")
            c.close()
        except Exception as e:
            print(f"  Poll error: {e}")
            time.sleep(10)
            continue

        running = int(out2.strip()) if out2.strip().isdigit() else 0
        elapsed_min = (i + 1) * 0.5
        if running == 0:
            print(f"\n  Experiment completed! ({elapsed_min:.0f} min)")
            break
        print(f"  ... still running ({elapsed_min:.0f} min)")

    # 5. Download results
    print("\n[5/5] Downloading results")
    local_out = LOCAL_ROOT / "results" / "round24_results"
    local_out.mkdir(parents=True, exist_ok=True)

    for rname in ["R24_output.txt", "stdout.txt"]:
        for attempt in range(3):
            try:
                c = ssh_connect()
                sftp = c.open_sftp()
                remote = f"{REMOTE_DIR}/results/round24_results/{rname}"
                local = str(local_out / rname)
                sftp.get(remote, local)
                sftp.close()
                c.close()
                size_kb = os.path.getsize(local) / 1024
                print(f"  Downloaded: {rname} ({size_kb:.1f} KB)")
                break
            except Exception as e:
                print(f"  {rname} attempt {attempt+1}: {e}")
                try:
                    c.close()
                except:
                    pass
                time.sleep(3)

    print("\n" + "=" * 60)
    print("Done! Check results/round24_results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
