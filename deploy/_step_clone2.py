"""Clone repo using AutoDL academic acceleration proxy."""
import paramiko, time, sys
sys.stdout.reconfigure(line_buffering=True)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print("Connecting...", flush=True)
c.connect('connect.westd.seetacloud.com', port=45630, username='root',
          password='r1zlTZQUb+E4', timeout=60)
c.get_transport().set_keepalive(5)
print("Connected!", flush=True)

def run(cmd, t=600):
    print(f"$ {cmd[:250]}", flush=True)
    tr = c.get_transport(); ch = tr.open_session(); ch.settimeout(t)
    ch.exec_command(cmd)
    out = b''
    t0 = time.time()
    while True:
        time.sleep(0.5)
        if ch.recv_ready():
            chunk = ch.recv(65536)
            out += chunk
            text = chunk.decode('utf-8', 'replace').strip()
            if text:
                for line in text.split('\n')[-5:]:
                    print(f"  {line}", flush=True)
        if ch.recv_stderr_ready():
            chunk = ch.recv_stderr(65536)
            out += chunk
            text = chunk.decode('utf-8', 'replace').strip()
            if text:
                for line in text.split('\n')[-5:]:
                    print(f"  [err] {line}", flush=True)
        if ch.exit_status_ready():
            while ch.recv_ready(): out += ch.recv(65536)
            while ch.recv_stderr_ready(): out += ch.recv_stderr(65536)
            break
    rc = ch.recv_exit_status(); ch.close()
    elapsed = time.time() - t0
    print(f"  [{rc}] ({elapsed:.1f}s)", flush=True)
    return out.decode('utf-8', 'replace'), rc

# Remove old empty dir
run('rm -rf /root/gold-quant-research')

# Clone with academic proxy
print("\n=== Git clone with academic acceleration ===", flush=True)
clone_cmd = (
    'source /etc/network_turbo && '
    'git clone --depth 1 --progress '
    'https://github.com/linhuang1313/gold-quant-research.git '
    '/root/gold-quant-research 2>&1'
)
run(clone_cmd, 600)

# Verify
print("\n=== Verify ===", flush=True)
run('ls /root/gold-quant-research/experiments/run_round24.py '
    '/root/gold-quant-research/backtest/engine.py '
    '/root/gold-quant-research/backtest/runner.py 2>&1')
run('wc -l /root/gold-quant-research/data/download/xauusd-m15-bid-*.csv '
    '/root/gold-quant-research/data/download/xauusd-h1-bid-*.csv 2>/dev/null || echo NO_DATA')
run('du -sh /root/gold-quant-research/')

c.close()
print("\nDone!", flush=True)
