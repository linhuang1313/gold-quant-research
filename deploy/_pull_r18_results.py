"""Pull R18 key results from Server C."""
import paramiko
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westc.seetacloud.com"
PORT = 16005
USER = "root"
PASS = "r1zlTZQUb+E4"
BASE = "/root/gold-quant-trading/results/round18_results/"

FILES = [
    "R18_summary.txt",
    "R18_F2_summary.txt",
    "R18_A1_per-year_profile.txt",
    "R18_C3_optimal_start_year.txt",
    "R18_E1_recency-weighted_sharpe.txt",
    "R18_E2_recency-weighted_params.txt",
    "R18_B1_old_train_→_new_test.txt",
]

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)

    def ssh(cmd):
        _, o, _ = client.exec_command(cmd, timeout=30)
        return o.read().decode('utf-8', errors='replace').strip()

    for f in FILES:
        print(f"=== {f} ===")
        content = ssh(f"cat {BASE}{f}")
        print(content)
        print()

    client.close()

if __name__ == "__main__":
    main()
