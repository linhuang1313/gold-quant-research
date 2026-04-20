"""Check Server D2 (port 45630) for any experiments."""
import paramiko
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westd.seetacloud.com"
PORT = 45630
USER = "root"
PASS = "r1zlTZQUb+E4"


def ssh(client, cmd):
    _, o, _ = client.exec_command(cmd, timeout=30)
    return o.read().decode('utf-8', errors='replace').strip()


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)

    print("=== Server D2 (port 45630) ===\n")

    print("--- Python processes ---")
    print(ssh(client, "ps aux | grep python | grep -v grep") or "  None")

    print("\n--- Project directories ---")
    print(ssh(client, "ls -la /root/gold-quant-trading/ 2>/dev/null || echo 'No gold-quant-trading dir'"))

    print("\n--- Result directories ---")
    print(ssh(client, "find /root -maxdepth 3 -type d -name '*result*' 2>/dev/null") or "  None found")

    print("\n--- Log files ---")
    print(ssh(client, "find /root -maxdepth 3 -name '*.log' 2>/dev/null | head -20") or "  None found")

    print("\n--- Output/experiment files ---")
    print(ssh(client, "find /root -maxdepth 3 -name '*output*' -o -name '*exp_*' 2>/dev/null | head -20") or "  None found")

    print("\n--- Disk usage ---")
    print(ssh(client, "df -h / | tail -1"))

    client.close()


if __name__ == "__main__":
    main()
