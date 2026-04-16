"""Quick check R13 progress."""
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
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    print("Connected!")

    out, _ = ssh_exec(client, "ps aux | grep run_round13 | grep -v grep")
    if out:
        print(f"\nR13 RUNNING:\n{out}")
    else:
        print("\nR13 NOT RUNNING!")

    print("\n--- Result files ---")
    out, _ = ssh_exec(client, f"ls -lh {REMOTE_DIR}/round13_results/ 2>&1")
    print(out)

    print("\n--- Last 40 lines of log ---")
    out, _ = ssh_exec(client, f"tail -40 {REMOTE_DIR}/logs/round13.log 2>/dev/null", timeout=30)
    print(out)

    client.close()

if __name__ == "__main__":
    main()
