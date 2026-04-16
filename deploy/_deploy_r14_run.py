"""Deploy and run Round 14 on remote server via SSH."""
import paramiko
import time
import sys

HOST = "connect.westc.seetacloud.com"
PORT = 16005
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"

def ssh_exec(ssh, cmd, timeout=120):
    print(f"  >>> {cmd[:100]}...")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    rc = stdout.channel.recv_exit_status()
    if out.strip():
        for line in out.strip().split('\n')[-10:]:
            print(f"      {line}")
    if err.strip() and rc != 0:
        for line in err.strip().split('\n')[-5:]:
            print(f"  ERR {line}")
    return out, err, rc


def main():
    print("="*60)
    print("Deploying Round 14 to server")
    print(f"Server: {HOST}:{PORT}")
    print("="*60)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    for attempt in range(3):
        try:
            print(f"\nConnecting (attempt {attempt+1})...")
            ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
            print("  Connected!")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            if attempt < 2:
                time.sleep(5)
            else:
                print("Cannot connect after 3 attempts")
                return

    print("\n--- Step 1: Check environment ---")
    ssh_exec(ssh, "python3 --version; which python3; df -h / | tail -1")
    ssh_exec(ssh, "nproc; free -h | head -2")

    print("\n--- Step 2: Clone or update repo ---")
    out, _, rc = ssh_exec(ssh, f"test -d {REMOTE_DIR} && echo EXISTS || echo MISSING")
    if "MISSING" in out:
        print("  Cloning repo...")
        ssh_exec(ssh,
            "source /etc/network_turbo 2>/dev/null; "
            f"git clone https://github.com/linhuang1313/gold-quant-trading.git {REMOTE_DIR}",
            timeout=180)
    else:
        print("  Updating repo...")
        ssh_exec(ssh,
            f"cd {REMOTE_DIR} && source /etc/network_turbo 2>/dev/null; "
            f"git stash 2>/dev/null; git pull; git stash drop 2>/dev/null; echo PULL_DONE",
            timeout=120)

    print("\n--- Step 3: Install dependencies ---")
    ssh_exec(ssh,
        "source /etc/network_turbo 2>/dev/null; "
        "pip install pandas numpy scipy yfinance paramiko python-dotenv 2>/dev/null | tail -3",
        timeout=120)

    print("\n--- Step 4: Check data files ---")
    ssh_exec(ssh,
        f"ls -la {REMOTE_DIR}/data/download/xauusd-m15-bid*.csv "
        f"{REMOTE_DIR}/data/download/xauusd-h1-bid*.csv "
        f"{REMOTE_DIR}/data/download/xauusd-m15-spread*.csv 2>/dev/null | head -10")

    print("\n--- Step 5: Quick syntax check ---")
    _, _, rc = ssh_exec(ssh,
        f"cd {REMOTE_DIR} && python3 -c \"import scripts.experiments.run_round14; print('Import OK')\"")
    if rc != 0:
        print("  Syntax check failed! Trying direct file check...")
        ssh_exec(ssh, f"cd {REMOTE_DIR} && python3 -m py_compile scripts/experiments/run_round14.py")

    print("\n--- Step 6: Start Round 14 ---")
    ssh_exec(ssh, f"mkdir -p {REMOTE_DIR}/round14_results")
    ssh_exec(ssh,
        f"cd {REMOTE_DIR} && "
        f"nohup python3 -u scripts/experiments/run_round14.py "
        f"> round14_results/r14_log.txt 2>&1 &"
        f" && sleep 2 && echo 'Process started'")

    time.sleep(3)
    print("\n--- Step 7: Verify process ---")
    ssh_exec(ssh, "ps aux | grep run_round14 | grep -v grep | head -5")
    ssh_exec(ssh, f"tail -20 {REMOTE_DIR}/round14_results/r14_log.txt 2>/dev/null")

    print("\n" + "="*60)
    print("Round 14 deployment complete!")
    print(f"Monitor: ssh -p {PORT} {USER}@{HOST}")
    print(f"  tail -f {REMOTE_DIR}/round14_results/r14_log.txt")
    print(f"  ls -la {REMOTE_DIR}/round14_results/")
    print("="*60)

    ssh.close()


if __name__ == "__main__":
    main()
