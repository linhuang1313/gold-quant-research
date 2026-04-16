"""Download and display R13 results."""
import paramiko

HOST = "connect.westd.seetacloud.com"
PORT = 35258
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading/round13_results"

def ssh_exec(client, cmd, timeout=60):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    return stdout.read().decode("utf-8", errors="replace").strip()

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)

    files = ["R13-A1_ema_scan.txt", "R13-A2_mult_scan.txt", "R13-A3_heatmap.txt",
             "R13-A4_kfold.txt", "R13-B1_breakeven.txt", "R13-B2_be_kfold.txt"]

    for f in files:
        print(f"\n{'='*80}")
        print(f"  {f}")
        print(f"{'='*80}")
        content = ssh_exec(client, f"cat {REMOTE_DIR}/{f} 2>/dev/null")
        if content:
            print(content)
        else:
            print("(not found)")

    client.close()

if __name__ == "__main__":
    main()
