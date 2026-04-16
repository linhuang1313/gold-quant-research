"""Check Round 12 experiment status on server."""
import paramiko
import sys
import time

HOST = "connect.westd.seetacloud.com"
PORT = 30367
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"

def ssh_exec(client, cmd, timeout=15):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    return out.strip(), err.strip()

def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for attempt in range(3):
        try:
            client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=20)
            return client
        except Exception as e:
            print(f"  Attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(5)
    raise RuntimeError("Cannot connect after 3 attempts")

def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    client = connect()

    if action == "status":
        out, _ = ssh_exec(client, "ps aux | grep run_round12 | grep -v grep")
        if out:
            print("R12 Running:")
            print(out)
        else:
            print("R12 NOT running")

        # Also check R11
        out, _ = ssh_exec(client, "ps aux | grep run_round11 | grep -v grep")
        if out:
            print("\nR11 Still Running:")
            print(out)

        print("\n--- Last 50 lines of R12 log ---")
        out, _ = ssh_exec(client, f"tail -50 {REMOTE_DIR}/logs/round12.log 2>/dev/null", timeout=10)
        print(out if out else "(no log)")

    elif action == "log":
        n = sys.argv[2] if len(sys.argv) > 2 else "100"
        out, _ = ssh_exec(client, f"tail -{n} {REMOTE_DIR}/logs/round12.log 2>/dev/null", timeout=30)
        print(out if out else "(no log)")

    elif action == "results":
        out, _ = ssh_exec(client, f"ls -la {REMOTE_DIR}/round12_results/ 2>/dev/null")
        print(out if out else "(no results yet)")
        result_files = [
            'R12-A1_sessions.txt', 'R12-A2_dow.txt', 'R12-A3_monthly.txt',
            'R12-B1_squeeze_ic.txt', 'R12-B3_squeeze_filter.txt', 'R12-B4_squeeze_kfold.txt',
            'R12-C1_consecutive.txt', 'R12-C2_consecutive_kfold.txt', 'R12-C3_strength.txt',
            'R12-D1_profit_dd.txt', 'R12-D2_adapt_hold.txt', 'R12-D3_profitdd_kfold.txt',
            'R12-D5_exit_combo.txt', 'R12-D6_exit_profile.txt',
            'R12-E1_cross_asset_ic.txt',
            'R12-F1_behavior.txt', 'R12-F5_tail_risk.txt',
        ]
        for f in result_files:
            out, _ = ssh_exec(client, f"cat {REMOTE_DIR}/round12_results/{f} 2>/dev/null")
            if out:
                print(f"\n{'='*60}")
                print(f"=== {f} ===")
                print(f"{'='*60}")
                print(out)

    elif action == "start":
        out, _ = ssh_exec(client, "ps aux | grep run_round12 | grep -v grep")
        if out:
            print("R12 already running!")
            print(out)
        else:
            out11, _ = ssh_exec(client, "ps aux | grep run_round11 | grep -v grep")
            if out11:
                print("WARNING: R11 still running. Starting R12 anyway (parallel)...")

            print("Starting R12...")
            client.exec_command(
                f"cd {REMOTE_DIR} && mkdir -p logs round12_results && "
                f"nohup python3 -u scripts/experiments/run_round12.py "
                f"> logs/round12.log 2>&1 &"
            )
            time.sleep(3)
            out, _ = ssh_exec(client, "ps aux | grep run_round12 | grep -v grep")
            print(out if out else "Failed to start!")

    elif action == "kill":
        ssh_exec(client, "pkill -f run_round12")
        print("Killed R12")

    client.close()

if __name__ == '__main__':
    main()
