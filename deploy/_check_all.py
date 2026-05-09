"""Check status of all R103-R110 experiments."""
import paramiko, time

SERVER = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASSWD = '3sCdENtzYfse'
REMOTE_BASE = '/root/gold-quant-research'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(SERVER, port=PORT, username=USER, password=PASSWD,
          timeout=120, banner_timeout=60, auth_timeout=60)

experiments = [
    ('r103', 'r103_adaptive_sizing', 'run_r103'),
    ('r104', 'r104_mtf_filter', 'run_r104'),
    ('r105', 'r105_dynamic_rotation', 'run_r105'),
    ('r106', 'r106_production_sim', 'run_r106'),
    ('r107', 'r107_ml_entry', 'run_r107'),
    ('r108', 'r108_tail_hedge', 'run_r108'),
    ('r109', 'r109_meta_ensemble', 'run_r109'),
    ('r110', 'r110_stress_scenarios', 'run_r110'),
]

print("=" * 70)
print("  R103-R110 Status Check")
print("=" * 70)

for tag, result_dir, proc_name in experiments:
    _, pout, _ = c.exec_command(f"pgrep -f {proc_name}", timeout=10)
    pid = pout.read().decode().strip()
    status = f"RUNNING (PID {pid.split()[0]})" if pid else "DONE/STOPPED"

    _, out, _ = c.exec_command(
        f"tail -5 {REMOTE_BASE}/results/{result_dir}/{tag}_stdout.txt 2>/dev/null",
        timeout=15)
    text = out.read().decode('utf-8', errors='replace').strip()
    lines = text.split('\n') if text else ['(no output)']
    last_lines = lines[-3:] if len(lines) >= 3 else lines

    print(f"\n  {tag} [{status}]:")
    for ln in last_lines:
        print(f"    {ln[:80]}")

c.close()
