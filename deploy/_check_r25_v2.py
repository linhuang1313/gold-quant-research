"""Check R25 - with retry logic."""
import paramiko, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

for attempt in range(3):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root',
                    password='5zQ8khQzttDN', timeout=60, banner_timeout=60,
                    auth_timeout=60)

        BASE = '/root/gold-quant-research'

        # Get tail of stdout
        _, o, _ = ssh.exec_command(f'tail -80 {BASE}/results/round25_results/R25_stdout.txt 2>&1', timeout=30)
        print(o.read().decode(errors='replace'))

        # Check process
        _, o, _ = ssh.exec_command('ps aux | grep run_round25 | grep -v grep', timeout=10)
        out = o.read().decode(errors='replace').strip()
        if out:
            print(f"\n[STILL RUNNING]\n{out}")
        else:
            print("\n[FINISHED]")
            _, o, _ = ssh.exec_command(f'wc -l {BASE}/results/round25_results/R25_output.txt {BASE}/results/round25_results/R25_stdout.txt 2>&1', timeout=10)
            print(o.read().decode(errors='replace'))

        ssh.close()
        break
    except Exception as e:
        print(f"Attempt {attempt+1} failed: {e}")
        if attempt < 2:
            print(f"Retrying in 15s...")
            time.sleep(15)
