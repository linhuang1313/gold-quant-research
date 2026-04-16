"""Download key experiment results from remote server."""
import paramiko, sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westc.seetacloud.com', port=30886, username='root',
            password='r1zlTZQUb+E4', timeout=15)

def run(cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=30)
    return stdout.read().decode('utf-8', errors='replace').strip()

# EXP-U: get lines with variant sweep results (skip RSI exit noise)
print("=" * 80)
print(">>> EXP-U KC Reentry — Key Results")
print("=" * 80)
out = run("""grep -E '(kc_mid|baseline|SUMMARY|K-Fold|Best beating|Result:|Sharpe=|Fold[0-9])' /root/gold-quant-trading/exp_u_kc_reentry_output.txt | head -40""")
print(out)

# EXP Batch: get all EXP sections
print("\n" + "=" * 80)
print(">>> EXP Batch — B/C/D Results (lines 100-400)")
print("=" * 80)
# Get lines from after EXP-A section
out2 = run("sed -n '100,500p' /root/gold-quant-trading/exp_batch_postfix_output.txt | grep -v 'M15 RSI' | head -200")
print(out2)

# EXP-L: get Part 3 K-Fold result summary
print("\n" + "=" * 80) 
print(">>> EXP-L Trend Weights — Summary")
print("=" * 80)
out3 = run("""grep -E '(Part|weights|PASS|FAIL|Fold[0-9]|Result:|best|0\\.[0-9]+ +0\\.[0-9]+)' /root/gold-quant-trading/exp_l_trend_weights_output.txt | head -30""")
print(out3)

ssh.close()
