"""Check EXP-CHOPPY-OC progress on Server C."""
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

    print("=== Server C: EXP-CHOPPY-OC Progress ===\n")

    procs = ssh("ps aux | grep run_exp_choppy | grep python | grep -v grep")
    if procs:
        print("Running:")
        for line in procs.split("\n"):
            parts = line.split()
            if len(parts) > 1:
                print(f"  PID={parts[1]}  CPU={parts[2]}%  MEM={parts[3]}%")
    else:
        print("  No process running!")

    log_size = ssh(f"wc -c < {REMOTE_DIR}/logs/exp_choppy_oc.log 2>/dev/null || echo 0")
    print(f"\n--- Log size: {log_size} bytes ---")

    print("\n--- Last 50 log lines ---")
    log = ssh(f"tail -50 {REMOTE_DIR}/logs/exp_choppy_oc.log 2>/dev/null")
    print(log)

    # Check output file
    output = ssh(f"cat {REMOTE_DIR}/exp_choppy_opportunity_cost_output.txt 2>/dev/null | tail -30")
    if output:
        print("\n--- Output file (last 30 lines) ---")
        print(output)

    client.close()


if __name__ == "__main__":
    main()
