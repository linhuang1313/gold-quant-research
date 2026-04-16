"""Launch EXP-R through EXP-X on remote server in parallel."""
import sys
import io
import time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
PYTHON = "/root/miniconda3/bin/python"
PROJECT = "/root/gold-quant-trading"

EXPERIMENTS = [
    ("run_exp_r_baseline_update.py", "exp_r_baseline_output.txt"),
    ("run_exp_s_historical_spread.py", "exp_s_spread_output.txt"),
    ("run_exp_t_donchian.py", "exp_t_donchian_output.txt"),
    ("run_exp_u_kc_reentry.py", "exp_u_kc_reentry_output.txt"),
    ("run_exp_v_breakout_sizing.py", "exp_v_sizing_output.txt"),
    ("run_exp_w_loss_profile.py", "exp_w_loss_profile_output.txt"),
]

def run_cmd(ssh, cmd, timeout=30):
    try:
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode('utf-8', errors='replace').strip()
        err = stderr.read().decode('utf-8', errors='replace').strip()
        return out, err
    except Exception as e:
        return "", str(e)

def launch_bg(ssh, script, output):
    """Launch a background process using screen (fire-and-forget)."""
    screen_name = script.replace('.py', '').replace('run_', '')
    cmd = f'screen -dmS {screen_name} bash -c "cd {PROJECT} && {PYTHON} -u {script} > {output} 2>&1"'
    try:
        ssh.exec_command(cmd, timeout=5)
        time.sleep(0.5)
    except:
        pass

def main():
    print("Connecting to remote server...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
    print("Connected!")

    # 1. Git pull
    print("\n--- Git Pull ---")
    out, err = run_cmd(ssh, f"cd {PROJECT} && git pull origin main 2>&1", timeout=30)
    if out:
        lines = out.split('\n')
        for l in lines[-10:]:
            print(f"  {l}")
    
    # 2. Check running processes
    print("\n--- Currently running ---")
    out, _ = run_cmd(ssh, "ps aux | grep 'python.*run_exp' | grep -v grep | awk '{print $2, $11, $12}'", timeout=10)
    running_scripts = set()
    if out:
        for line in out.strip().split('\n'):
            print(f"  {line}")
            for s, _ in EXPERIMENTS:
                if s in line:
                    running_scripts.add(s)
    else:
        print("  None")

    # 3. Launch experiments one by one
    print("\n--- Launching new experiments ---")
    launched = []
    for script, output in EXPERIMENTS:
        if script in running_scripts:
            print(f"  SKIP {script} (already running)")
            continue
        launch_bg(ssh, script, output)
        launched.append(script)
        print(f"  LAUNCHED {script}")

    # 4. Check EUR/USD
    print("\n--- EUR/USD data check ---")
    out, _ = run_cmd(ssh, f"ls {PROJECT}/data/download/eurusd* 2>/dev/null | head -3", timeout=5)
    if out.strip():
        print(f"  Found: {out}")
        script, output = "run_exp_x_eurusd_stress.py", "exp_x_eurusd_output.txt"
        if script not in running_scripts:
            launch_bg(ssh, script, output)
            launched.append(script)
            print(f"  LAUNCHED {script}")
    else:
        print("  No EUR/USD data, skipping EXP-X")

    # 5. Wait a bit and verify
    time.sleep(3)
    print("\n--- Verification ---")
    out, _ = run_cmd(ssh, "ps aux | grep 'python.*run_exp' | grep -v grep | wc -l", timeout=10)
    print(f"  Total python experiments running: {out}")
    
    out, _ = run_cmd(ssh, "screen -ls 2>/dev/null | grep -c Detached", timeout=5)
    print(f"  Screen sessions: {out}")

    out, _ = run_cmd(ssh, "uptime", timeout=5)
    print(f"  Server load: {out}")

    out, _ = run_cmd(ssh, "nproc", timeout=5)
    print(f"  CPU cores: {out}")

    # 6. Summary
    print("\n" + "=" * 60)
    print(f"LAUNCHED {len(launched)} NEW EXPERIMENTS:")
    for s in launched:
        print(f"  - {s}")
    print("=" * 60)

    ssh.close()

if __name__ == "__main__":
    main()
