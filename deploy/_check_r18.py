"""Check Round 18 (Temporal Relevance) progress on Server C."""
import paramiko
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westc.seetacloud.com"
PORT = 16005
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)

    def ssh(cmd):
        _, o, _ = client.exec_command(cmd, timeout=30)
        return o.read().decode('utf-8', errors='replace').strip()

    print("=== Server C: R18 Progress ===\n")

    procs = ssh("ps aux | grep run_round18 | grep python | grep -v grep")
    if procs:
        print("Running:")
        for line in procs.split("\n"):
            parts = line.split()
            if len(parts) > 10:
                print(f"  PID={parts[1]} CPU={parts[2]}% MEM={parts[3]}% CMD={' '.join(parts[10:])}")
    else:
        print("  No R18 processes running!")

    print(f"\n--- Result files ---")
    files = ssh(f"ls -la {REMOTE_DIR}/results/round18_results/ 2>/dev/null")
    print(f"  {files}" if files else "  No results yet")

    print(f"\n--- Log size ---")
    print(f"  {ssh(f'wc -l {REMOTE_DIR}/logs/round18.log 2>/dev/null')}")

    print(f"\n--- Phase progress ---")
    phases = ssh(f"grep -E '(Phase|done|COMPLETE|FAIL|R18-)' {REMOTE_DIR}/logs/round18.log 2>/dev/null | tail -20")
    print(f"  {phases}" if phases else "  No phase markers yet")

    print(f"\n--- Last 10 log lines ---")
    tail = ssh(f"tail -10 {REMOTE_DIR}/logs/round18.log 2>/dev/null")
    print(f"  {tail}" if tail else "  No log yet")

    summary = ssh(f"cat {REMOTE_DIR}/results/round18_results/R18_summary.txt 2>/dev/null")
    if summary:
        print(f"\n--- R18 Summary ---")
        print(summary)

    client.close()


if __name__ == '__main__':
    main()
