"""Pull R17 results - one connection per file to handle unstable D1."""
import paramiko
import sys
import io
import os
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westd.seetacloud.com"
PORT = 35258
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading/results/round17_results"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'round17_results')

FILES = [
    "R17_A1_compounding_vs_fixed.txt",
    "R17_A2_compounding_spreads.txt",
    "R17_A3_compounding_kfold.txt",
    "R17_B1_kelly_scan.txt",
    "R17_B2_kelly_compounding.txt",
    "R17_B3_kelly_walkforward.txt",
    "R17_B4_kelly_kfold.txt",
    "R17_C1_drawdown_threshold.txt",
    "R17_C2_reduction_grid.txt",
    "R17_C3_dd_kfold.txt",
    "R17_D1_antimartingale_scan.txt",
    "R17_D2_triple_stack.txt",
    "R17_E1_reinvest_scan.txt",
    "R17_E2_equity_filter.txt",
    "R17_F1_combo_matrix.txt",
    "R17_F2_montecarlo.txt",
    "R17_F3_spread_sensitivity.txt",
    "R17_G1_final_kfold.txt",
    "R17_G2_ruin_probability.txt",
    "R17_summary.txt",
]


def pull_one(fname):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS,
                   timeout=15, banner_timeout=15, auth_timeout=15)
    _, o, _ = client.exec_command(f"cat {REMOTE_DIR}/{fname}", timeout=10)
    content = o.read().decode('utf-8', errors='replace')
    client.close()
    return content


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    ok = 0
    fail = 0

    for fname in FILES:
        local = os.path.join(LOCAL_DIR, fname)
        if os.path.exists(local) and os.path.getsize(local) > 100:
            print(f"  [SKIP] {fname} (already local)")
            ok += 1
            continue
        for attempt in range(2):
            try:
                content = pull_one(fname)
                with open(local, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  [OK] {fname} ({len(content):,} chars)")
                ok += 1
                time.sleep(1)
                break
            except Exception as e:
                print(f"  [FAIL attempt {attempt+1}] {fname}: {e}")
                time.sleep(3)
        else:
            fail += 1

    print(f"\nDone: {ok} OK, {fail} FAIL")


if __name__ == "__main__":
    main()
