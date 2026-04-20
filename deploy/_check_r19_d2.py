"""Check R19 progress on Server D2 (port 45630)."""
import paramiko
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westd.seetacloud.com"
PORT = 45630
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

    print("=== D2: R19 Progress ===\n")

    procs = ssh(client, "ps aux | grep run_round19 | grep python | grep -v grep")
    if procs:
        for line in procs.split("\n"):
            parts = line.split()
            if len(parts) > 3:
                print(f"  PID={parts[1]}  CPU={parts[2]}%  MEM={parts[3]}%")
    else:
        print("  No R19 process running!")

    print("\n--- Result files ---")
    results = ssh(client, f"ls -lhS {REMOTE_DIR}/results/round19_results/ 2>/dev/null")
    print(results or "  No results yet")

    print("\n--- Last 30 log lines ---")
    log = ssh(client, f"tail -30 {REMOTE_DIR}/logs/round19.log 2>/dev/null")
    print(log)

    client.close()


if __name__ == "__main__":
    main()
