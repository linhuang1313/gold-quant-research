"""Check EXP-CHOPPY-OC progress on Server D2 (port 45630)."""
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

    print("=== D2: EXP-CHOPPY-OC Progress ===\n")

    procs = ssh(client, "ps aux | grep run_exp_choppy | grep python | grep -v grep")
    if procs:
        for line in procs.split("\n"):
            parts = line.split()
            if len(parts) > 3:
                print(f"  PID={parts[1]}  CPU={parts[2]}%  MEM={parts[3]}%")
    else:
        print("  No process running! (may be finished)")

    log_size = ssh(client, f"wc -c < {REMOTE_DIR}/logs/exp_choppy_oc.log 2>/dev/null || echo 0")
    print(f"\n  Log size: {log_size} bytes")

    print("\n--- Last 30 log lines ---")
    log = ssh(client, f"tail -30 {REMOTE_DIR}/logs/exp_choppy_oc.log 2>/dev/null")
    print(log)

    # Check output file
    out_size = ssh(client, f"wc -c < {REMOTE_DIR}/exp_choppy_opportunity_cost_output.txt 2>/dev/null || echo 0")
    print(f"\n  Output file size: {out_size} bytes")

    print("\n--- Output tail ---")
    out = ssh(client, f"tail -20 {REMOTE_DIR}/exp_choppy_opportunity_cost_output.txt 2>/dev/null")
    print(out)

    client.close()


if __name__ == "__main__":
    main()
