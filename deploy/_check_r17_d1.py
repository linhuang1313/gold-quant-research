"""Check R17 progress on Server D1 (port 35258)."""
import paramiko
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westd.seetacloud.com"
PORT = 35258
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"


def ssh(client, cmd):
    _, o, _ = client.exec_command(cmd, timeout=30)
    return o.read().decode('utf-8', errors='replace').strip()


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)

    print("=== D1: R17 Progress ===\n")

    procs = ssh(client, "ps aux | grep run_round17 | grep python | grep -v grep")
    if procs:
        for line in procs.split("\n"):
            parts = line.split()
            if len(parts) > 3:
                print(f"  PID={parts[1]}  CPU={parts[2]}%  MEM={parts[3]}%")
    else:
        print("  No R17 process running!")

    # Result files
    print("\n--- Result files ---")
    results = ssh(client, f"ls -lhS {REMOTE_DIR}/results/round17_results/ 2>/dev/null")
    print(results or "  No results yet")

    # Log - phase progress
    print("\n--- Phase progress ---")
    phases = ssh(client, f"grep -E '>>> Starting|<<< Phase|Round 17' {REMOTE_DIR}/logs/round17.log 2>/dev/null")
    print(phases or "  No phase info yet")

    # Log tail
    print("\n--- Last 20 log lines ---")
    log = ssh(client, f"tail -20 {REMOTE_DIR}/logs/round17.log 2>/dev/null")
    print(log)

    client.close()


if __name__ == "__main__":
    main()
