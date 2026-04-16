"""Check EXP-U completion, fix EXP-L, relaunch crashed experiments."""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
PYTHON = "/root/miniconda3/bin/python"
PROJECT = "/root/gold-quant-trading"

def run_cmd(ssh, cmd, timeout=15):
    try:
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        return stdout.read().decode('utf-8', errors='replace').strip()
    except:
        return ""

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

    # 1. Check EXP-U status
    print("=== EXP-U Status ===")
    out = run_cmd(ssh, f"wc -c < {PROJECT}/exp_u_kc_reentry_output.txt")
    print(f"  File size: {out} bytes")
    out = run_cmd(ssh, f"ps aux | grep 'run_exp_u' | grep -v grep")
    if out:
        print(f"  STILL RUNNING: {out.split()[1]}")
    else:
        print("  Process NOT running")
    out = run_cmd(ssh, f"tail -5 {PROJECT}/exp_u_kc_reentry_output.txt")
    print(f"  Last lines:\n  {out[-300:]}")
    
    # 2. Check full EXP-U for K-Fold results
    print("\n=== EXP-U K-Fold Search ===")
    out = run_cmd(ssh, f"grep -n 'K-Fold\\|SUMMARY\\|PASS\\|FAIL\\|best_alt\\|kc_mid_min.*Sharpe\\|kc_band.*Sharpe' {PROJECT}/exp_u_kc_reentry_output.txt")
    print(out[:2000] if out else "  No K-Fold results found yet")

    # 3. Check EXP-L error
    print("\n=== EXP-L Error ===")
    out = run_cmd(ssh, f"grep -A3 'TypeError' {PROJECT}/exp_l_trend_weights_output.txt")
    print(f"  {out}")
    
    # 4. Check EXP-W status
    print("\n=== EXP-W Status ===")
    out = run_cmd(ssh, f"wc -c < {PROJECT}/exp_w_loss_profile_output.txt")
    print(f"  File size: {out} bytes")
    out = run_cmd(ssh, f"ps aux | grep 'run_exp_w' | grep -v grep")
    if out:
        print(f"  STILL RUNNING")
    else:
        print("  Process NOT running")
    out = run_cmd(ssh, f"tail -3 {PROJECT}/exp_w_loss_profile_output.txt")
    print(f"  Last: {out[-200:]}")

    # 5. Check EXP-K status  
    print("\n=== EXP-K Status ===")
    out = run_cmd(ssh, f"wc -c < {PROJECT}/exp_k_regime_output.txt 2>/dev/null")
    print(f"  File size: {out if out else '0'} bytes")

    # 6. All running processes
    print("\n=== All Running ===")
    out = run_cmd(ssh, "ps aux | grep 'python.*run_exp' | grep -v grep | awk '{print $2, $11, $12}'")
    if out:
        for line in out.strip().split('\n'):
            print(f"  {line}")
    else:
        print("  None running")
    
    ssh.close()

if __name__ == "__main__":
    main()
