"""Relaunch EXP-L (fixed), EXP-K, EXP-W on remote server."""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
PYTHON = "/root/miniconda3/bin/python"
PROJECT = "/root/gold-quant-trading"

def run_cmd(ssh, cmd, timeout=30):
    try:
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        return stdout.read().decode('utf-8', errors='replace').strip()
    except Exception as e:
        return f"ERROR: {e}"

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

    # 1. Pull latest code (includes EXP-L fix)
    print("=== Git Pull ===")
    out = run_cmd(ssh, f"cd {PROJECT} && git pull", timeout=30)
    print(out)

    # 2. Check what's currently running
    print("\n=== Currently Running ===")
    out = run_cmd(ssh, "ps aux | grep 'python.*run_exp' | grep -v grep | awk '{print $2, $NF}'")
    print(out if out else "  None")

    # 3. Kill old crashed EXP-L and EXP-W if they left zombie processes
    for pattern in ['run_exp_l', 'run_exp_w', 'run_exp_k']:
        out = run_cmd(ssh, f"ps aux | grep '{pattern}' | grep -v grep | awk '{{print $2}}'")
        if out:
            for pid in out.strip().split('\n'):
                print(f"  Killing old {pattern} process: {pid}")
                run_cmd(ssh, f"kill {pid}")
                time.sleep(0.3)

    # 4. Launch EXP-L (fixed), EXP-K, EXP-W
    experiments = [
        ("exp_l", "run_exp_l_trend_weights.py", "exp_l_trend_weights_output.txt"),
        ("exp_k", "run_exp_k_regime_bounds.py", "exp_k_regime_bounds_output.txt"),
        ("exp_w", "run_exp_w_loss_profile.py", "exp_w_loss_profile_output.txt"),
    ]

    for screen_name, script, output in experiments:
        out = run_cmd(ssh, f"ls -la {PROJECT}/{script}")
        if "No such file" in out:
            print(f"  SKIP {script} - not found!")
            continue
        
        cmd = f"cd {PROJECT} && screen -dmS {screen_name} bash -c '{PYTHON} -u {script} > {output} 2>&1'"
        print(f"\n  Launching {screen_name}: {script}")
        run_cmd(ssh, cmd)
        time.sleep(1)

    # 5. Verify launches
    time.sleep(2)
    print("\n=== Verification ===")
    out = run_cmd(ssh, "screen -ls")
    print(out)
    
    out = run_cmd(ssh, "ps aux | grep 'python.*run_exp' | grep -v grep | awk '{print $2, $NF}'")
    print(f"\nRunning python processes:")
    print(out if out else "  None!")

    # 6. Server load
    out = run_cmd(ssh, "uptime")
    print(f"\nServer: {out}")
    out = run_cmd(ssh, "nproc")
    print(f"CPU cores: {out}")

    ssh.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
