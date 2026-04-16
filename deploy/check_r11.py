"""Check Round 11 experiment status on server."""
import paramiko
import sys

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

def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "status"

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

    if action == "status":
        out, _ = ssh_exec(client, "ps aux | grep run_round11 | grep -v grep")
        if out:
            print("Running:")
            print(out)
        else:
            print("NOT running")

        print("\n--- Last 50 lines of log ---")
        out, _ = ssh_exec(client, f"tail -50 {REMOTE_DIR}/logs/round11.log 2>/dev/null", timeout=10)
        print(out if out else "(no log)")

    elif action == "start":
        out, _ = ssh_exec(client, "ps aux | grep run_round11 | grep -v grep")
        if out:
            print("Already running!")
            print(out)
        else:
            print("Starting R11...")
            client.exec_command(
                f"cd {REMOTE_DIR} && nohup python3 -u scripts/experiments/run_round11.py "
                f"> logs/round11.log 2>&1 &"
            )
            import time; time.sleep(3)
            out, _ = ssh_exec(client, "ps aux | grep run_round11 | grep -v grep")
            print(out if out else "Failed to start!")

    elif action == "log":
        n = sys.argv[2] if len(sys.argv) > 2 else "100"
        out, _ = ssh_exec(client, f"tail -{n} {REMOTE_DIR}/logs/round11.log 2>/dev/null", timeout=10)
        print(out if out else "(no log)")

    elif action == "log_ext":
        n = sys.argv[2] if len(sys.argv) > 2 else "100"
        out, _ = ssh_exec(client, f"tail -{n} {REMOTE_DIR}/logs/round11_ext.log 2>/dev/null", timeout=10)
        print(out if out else "(no ext log)")

    elif action == "results":
        out, _ = ssh_exec(client, f"ls -la {REMOTE_DIR}/round11_results/ 2>/dev/null")
        print(out if out else "(no results yet)")
        result_files = [
            'R11-1_pinbar_ic.txt', 'R11-2_pinbar_freq.txt', 'R11-3_pinbar_confirm.txt',
            'R11-15_pa_ic.txt', 'R11-16_pa_freq.txt', 'R11-17_pa_filters.txt',
            'R11-18_pa_kfold.txt', 'R11-19_daily_range.txt', 'R11-20_confluence.txt',
            'R11-21_dr_pa_combo.txt', 'R11-22_pa_sr_strats.txt',
            'R11-23_global_best.txt', 'R11-24_global_kfold.txt',
        ]
        for f in result_files:
            out, _ = ssh_exec(client, f"cat {REMOTE_DIR}/round11_results/{f} 2>/dev/null")
            if out:
                print(f"\n=== {f} ===")
                print(out)

    elif action == "kill":
        ssh_exec(client, "pkill -f run_round11")
        print("Killed")

    client.close()

if __name__ == '__main__':
    main()
