"""Start R14 on remote server."""
import paramiko
import time

HOST = "connect.westc.seetacloud.com"
PORT = 16005
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"

def ssh_exec(ssh, cmd, timeout=60):
    print(f"  >>> {cmd[:120]}...")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    rc = stdout.channel.recv_exit_status()
    if out.strip():
        for line in out.strip().split('\n')[-15:]:
            print(f"      {line}")
    if err.strip() and rc != 0:
        for line in err.strip().split('\n')[-5:]:
            print(f"  ERR {line}")
    return out, err, rc

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
print("Connected!")

print("\n--- Check if already running ---")
out, _, _ = ssh_exec(ssh, "ps aux | grep run_round14 | grep -v grep")
if "run_round14" in out:
    print("  R14 already running!")
else:
    print("\n--- Starting R14 via screen ---")
    ssh_exec(ssh, "which screen || apt-get install -y screen 2>/dev/null | tail -1", timeout=30)
    ssh_exec(ssh,
        f"cd {REMOTE_DIR} && "
        f"screen -dmS r14 bash -c 'python3 -u scripts/experiments/run_round14.py "
        f"> round14_results/r14_log.txt 2>&1'")

time.sleep(5)

print("\n--- Verify ---")
ssh_exec(ssh, "ps aux | grep run_round14 | grep -v grep")
ssh_exec(ssh, "screen -ls 2>/dev/null || true")
ssh_exec(ssh, f"wc -l {REMOTE_DIR}/round14_results/r14_log.txt 2>/dev/null; "
              f"tail -30 {REMOTE_DIR}/round14_results/r14_log.txt 2>/dev/null")

print("\nDone!")
ssh.close()
