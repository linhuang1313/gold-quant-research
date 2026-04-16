"""Deploy R13 via git pull and start on server."""
import paramiko
import time

HOST = "connect.westd.seetacloud.com"
PORT = 35258
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"

def ssh_exec(client, cmd, timeout=60):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return out.strip(), err.strip()

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    for attempt in range(5):
        try:
            print(f"Connecting (attempt {attempt+1}/5)...")
            client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
            print("Connected!")
            break
        except Exception as e:
            print(f"Failed: {e}")
            if attempt < 4:
                time.sleep(10)
    else:
        print("Cannot connect after 5 attempts")
        return

    print("\n--- Stash local changes + Git pull ---")
    out, err = ssh_exec(client,
        f"cd {REMOTE_DIR} && source /etc/network_turbo 2>/dev/null; "
        f"git stash && git pull && git stash drop 2>/dev/null; echo PULL_DONE",
        timeout=120)
    print(f"stdout: {out}")
    if err:
        print(f"stderr: {err}")
    if "PULL_DONE" not in out:
        print("ERROR: git pull failed!")
        # Try harder - force reset
        print("\n--- Force reset to origin/main ---")
        out, err = ssh_exec(client,
            f"cd {REMOTE_DIR} && source /etc/network_turbo 2>/dev/null; "
            f"git fetch origin && git reset --hard origin/main && echo RESET_DONE",
            timeout=120)
        print(f"stdout: {out}")
        if err:
            print(f"stderr: {err}")

    print("\n--- Verifying new files ---")
    out, _ = ssh_exec(client, f"ls -la {REMOTE_DIR}/scripts/experiments/run_round13.py 2>&1")
    print(out)

    print("\n--- Verifying imports ---")
    verify_cmd = (
        f"cd {REMOTE_DIR} && python3 -c "
        "'from backtest.runner import DataBundle, _hma, _kama, add_dual_kc; "
        "print(\"IMPORT OK\")'"
    )
    out, err = ssh_exec(client, verify_cmd, timeout=30)
    print(f"Import: {out}")
    if "OK" not in out:
        print(f"Error: {err}")
        client.close()
        return

    print("\n--- Data files ---")
    out, _ = ssh_exec(client, f"ls -lh {REMOTE_DIR}/data/download/xauusd-*bid*.csv 2>&1 | head -3")
    print(out)

    ssh_exec(client, f"mkdir -p {REMOTE_DIR}/logs {REMOTE_DIR}/round13_results")

    ssh_exec(client, "pkill -f run_round13 2>/dev/null")
    time.sleep(2)

    print("\n--- Starting R13 ---")
    cmd = (f"cd {REMOTE_DIR} && nohup python3 -u scripts/experiments/run_round13.py "
           f"> logs/round13.log 2>&1 &")
    client.exec_command(cmd)
    time.sleep(10)

    out, _ = ssh_exec(client, "ps aux | grep run_round13 | grep -v grep")
    if out:
        print("R13 started successfully!")
        print(out)
    else:
        print("WARNING: R13 may not have started. Checking log...")

    print("\n--- Initial log ---")
    out, _ = ssh_exec(client, f"tail -30 {REMOTE_DIR}/logs/round13.log 2>/dev/null", timeout=30)
    print(out if out else "(empty)")

    client.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
