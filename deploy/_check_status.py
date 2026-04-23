"""Quick check if R24 is still running on remote."""
import paramiko, time, sys
sys.stdout.reconfigure(line_buffering=True)
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('connect.westd.seetacloud.com', port=45630, username='root',
          password='r1zlTZQUb+E4', timeout=60)
c.get_transport().set_keepalive(5)

def run(cmd, t=120):
    tr = c.get_transport(); ch = tr.open_session(); ch.settimeout(t)
    ch.exec_command(cmd)
    out = b''
    while True:
        time.sleep(0.2)
        if ch.recv_ready(): out += ch.recv(65536)
        if ch.recv_stderr_ready(): out += ch.recv_stderr(65536)
        if ch.exit_status_ready():
            while ch.recv_ready(): out += ch.recv(65536)
            while ch.recv_stderr_ready(): out += ch.recv_stderr(65536)
            break
    rc = ch.recv_exit_status(); ch.close()
    return out.decode('utf-8', 'replace'), rc

out, _ = run("ps aux | grep run_round24 | grep -v grep")
print(f"Process:\n{out}", flush=True)
out, _ = run("wc -l /root/gold-quant-research/results/round24_results/stdout.txt")
print(f"Lines: {out.strip()}", flush=True)
out, _ = run("tail -20 /root/gold-quant-research/results/round24_results/stdout.txt")
print(f"Last output:\n{out}", flush=True)
c.close()
