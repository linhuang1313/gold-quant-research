"""Step 1: Clone repo on remote server via git."""
import paramiko, time, sys

HOST = "connect.westd.seetacloud.com"
PORT = 45630
USER = "root"
PASS = "r1zlTZQUb+E4"
REPO = "https://github.com/linhuang1313/gold-quant-research.git"
DIR  = "/root/gold-quant-research"

sys.stdout.reconfigure(line_buffering=True)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print("Connecting...", flush=True)
c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=60)
c.get_transport().set_keepalive(5)
print("Connected!", flush=True)

def run(cmd, t=600):
    print(f"$ {cmd[:200]}", flush=True)
    tr = c.get_transport()
    ch = tr.open_session()
    ch.settimeout(t)
    ch.exec_command(cmd)
    out = b""
    err = b""
    while True:
        time.sleep(0.3)
        if ch.recv_ready():
            chunk = ch.recv(65536)
            out += chunk
            sys.stdout.write(chunk.decode("utf-8", errors="replace"))
            sys.stdout.flush()
        if ch.recv_stderr_ready():
            chunk = ch.recv_stderr(65536)
            err += chunk
            sys.stderr.write(chunk.decode("utf-8", errors="replace"))
            sys.stderr.flush()
        if ch.exit_status_ready():
            while ch.recv_ready():
                out += ch.recv(65536)
            while ch.recv_stderr_ready():
                err += ch.recv_stderr(65536)
            break
    rc = ch.recv_exit_status()
    ch.close()
    print(f"\n[rc={rc}]", flush=True)
    return out.decode("utf-8", errors="replace"), rc

run("test -d /root/gold-quant-research/.git && echo EXISTS || echo NEED_CLONE")
run("rm -rf /root/gold-quant-research")
print("\n=== Cloning... ===", flush=True)
run(f"git clone --depth 1 {REPO} {DIR} 2>&1", 600)
print("\n=== Verify ===", flush=True)
run(f"ls {DIR}/experiments/run_round24.py {DIR}/backtest/engine.py")
run(f"wc -l {DIR}/data/download/xauusd-m15-bid-*.csv 2>/dev/null || echo 'no M15 data'")
run(f"wc -l {DIR}/data/download/xauusd-h1-bid-*.csv 2>/dev/null || echo 'no H1 data'")
c.close()
print("Done!", flush=True)
