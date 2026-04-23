"""Step 2: Launch R24 on remote and poll until done."""
import paramiko, time, sys, os
from pathlib import Path

HOST = "connect.westd.seetacloud.com"
PORT = 45630
USER = "root"
PASS = "r1zlTZQUb+E4"
DIR  = "/root/gold-quant-research"

sys.stdout.reconfigure(line_buffering=True)

def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=60)
    c.get_transport().set_keepalive(5)
    time.sleep(0.3)
    return c

def run(c, cmd, t=300):
    print(f"$ {cmd[:200]}", flush=True)
    tr = c.get_transport()
    ch = tr.open_session()
    ch.settimeout(t)
    ch.exec_command(cmd)
    out = b""
    err = b""
    while True:
        time.sleep(0.2)
        if ch.recv_ready(): out += ch.recv(65536)
        if ch.recv_stderr_ready(): err += ch.recv_stderr(65536)
        if ch.exit_status_ready():
            while ch.recv_ready(): out += ch.recv(65536)
            while ch.recv_stderr_ready(): err += ch.recv_stderr(65536)
            break
    rc = ch.recv_exit_status()
    ch.close()
    out_s = out.decode("utf-8", errors="replace")
    if out_s.strip():
        for line in out_s.strip().split("\n")[:30]:
            print(f"  {line}", flush=True)
    if err.strip() and rc != 0:
        for line in err.decode("utf-8", errors="replace").strip().split("\n")[:10]:
            print(f"  [ERR] {line}", flush=True)
    print(f"  [rc={rc}]", flush=True)
    return out_s, rc

print("=== Launch R24 ===", flush=True)
c = connect()
run(c, f"mkdir -p {DIR}/results/round24_results")
run(c, f"cd {DIR} && PYTHONIOENCODING=utf-8 nohup python3 -u experiments/run_round24.py > results/round24_results/stdout.txt 2>&1 &")
time.sleep(3)
run(c, "ps aux | grep run_round24 | grep -v grep")
run(c, f"head -20 {DIR}/results/round24_results/stdout.txt 2>/dev/null")
c.close()

print("\n=== Monitoring (30s intervals) ===", flush=True)
for i in range(120):
    time.sleep(30)
    try:
        c = connect()
        out, _ = run(c, f"tail -8 {DIR}/results/round24_results/stdout.txt 2>/dev/null")
        out2, _ = run(c, "ps aux | grep 'python3.*run_round24' | grep -v grep | wc -l")
        c.close()
    except Exception as e:
        print(f"  Poll error: {e}", flush=True)
        time.sleep(10)
        continue

    running = int(out2.strip()) if out2.strip().isdigit() else 0
    elapsed = (i + 1) * 0.5
    if running == 0:
        print(f"\n=== Completed! ({elapsed:.0f} min) ===", flush=True)
        break
    print(f"  ... running ({elapsed:.0f} min)", flush=True)

# Download
print("\n=== Download results ===", flush=True)
local_out = Path(__file__).parent.parent / "results" / "round24_results"
local_out.mkdir(parents=True, exist_ok=True)
for rname in ["R24_output.txt", "stdout.txt"]:
    for attempt in range(3):
        try:
            c = connect()
            sftp = c.open_sftp()
            sftp.get(f"{DIR}/results/round24_results/{rname}", str(local_out / rname))
            sftp.close()
            c.close()
            sz = os.path.getsize(str(local_out / rname)) / 1024
            print(f"  Downloaded: {rname} ({sz:.1f} KB)", flush=True)
            break
        except Exception as e:
            print(f"  {rname} attempt {attempt+1}: {e}", flush=True)
            try: c.close()
            except: pass
            time.sleep(3)

print("\nDone!", flush=True)
