"""Check file sizes and completions on remote server."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
PROJECT = "/root/gold-quant-trading"

def run_cmd(ssh, cmd, timeout=15):
    try:
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        return stdout.read().decode('utf-8', errors='replace').strip()
    except:
        return ""

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

files = [
    "exp_batch_postfix_output.txt",
    "exp_m_slippage_output.txt",
    "exp_r_baseline_output.txt",
    "exp_s_spread_output.txt",
    "exp_u_kc_reentry_output.txt",
    "exp_l_trend_weights_output.txt",
    "exp_k_regime_bounds_output.txt",
    "exp_w_loss_profile_output.txt",
]

print(f"{'File':<40s} {'Size':>10s}  Last line")
print("-" * 100)
for f in files:
    sz = run_cmd(ssh, f"wc -c < {PROJECT}/{f} 2>/dev/null")
    last = run_cmd(ssh, f"tail -3 {PROJECT}/{f} 2>/dev/null")
    last_short = last[-120:] if last else "N/A"
    print(f"  {f:<38s} {sz:>10s}  {last_short}")

# Check running processes
print("\n=== Running ===")
out = run_cmd(ssh, "ps aux | grep 'python.*run_exp' | grep -v grep | awk '{for(i=11;i<=NF;i++) printf $i\" \"; print \"\"}'")
print(out if out else "  None")

ssh.close()
