"""Launch EMA100 ablation test on remote server (parallel with choppy sweep)."""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
PROJECT = "/root/gold-quant-trading"
PYTHON = "/root/miniconda3/envs/3.10/bin/python"
SCRIPT = "run_exp_ema100_ablation.py"
OUTPUT = "exp_ema100_ablation_output.txt"

def run_cmd(ssh, cmd, timeout=30):
    try:
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode('utf-8', errors='replace').strip()
        err = stderr.read().decode('utf-8', errors='replace').strip()
        return out, err
    except Exception as e:
        return f"ERROR: {e}", ""

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print("Connecting...")
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

    # Check choppy sweep still running
    ps_choppy, _ = run_cmd(ssh, "ps aux | grep run_exp_choppy_fine | grep -v grep | wc -l")
    print(f"Choppy sweep processes: {ps_choppy}")

    # Upload script
    print("\n--- Upload ---")
    sftp = ssh.open_sftp()
    sftp.put(SCRIPT, f"{PROJECT}/{SCRIPT}")
    sftp.close()
    print(f"  Uploaded {SCRIPT}")

    # Launch (don't kill choppy!)
    print(f"\n--- Launching EMA100 ablation (parallel with choppy) ---")
    launch_cmd = f"cd {PROJECT} && nohup {PYTHON} -u {SCRIPT} > {OUTPUT} 2>&1 &"
    run_cmd(ssh, launch_cmd, timeout=10)
    time.sleep(5)

    # Verify
    out, _ = run_cmd(ssh, f"ps aux | grep {SCRIPT} | grep -v grep | head -3")
    if out:
        print("RUNNING:")
        for line in out.split('\n')[:3]:
            print(f"  {line.strip()}")
    else:
        print("NOT RUNNING! Checking output...")
        out2, _ = run_cmd(ssh, f"head -30 {PROJECT}/{OUTPUT} 2>/dev/null")
        print(out2)
        ssh.close()
        return

    time.sleep(10)
    out, _ = run_cmd(ssh, f"tail -10 {PROJECT}/{OUTPUT} 2>/dev/null")
    print(f"\n--- Initial Output ---")
    print(out)

    # Show total processes
    ps_all, _ = run_cmd(ssh, "ps aux | grep 'run_exp_' | grep -v grep | wc -l")
    print(f"\nTotal experiment processes: {ps_all}")

    ssh.close()
    print(f"\nExpected runtime: ~20-30 minutes")

if __name__ == "__main__":
    main()
