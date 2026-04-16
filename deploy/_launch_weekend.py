"""Launch weekend batch experiments on remote server."""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
PROJECT = "/root/gold-quant-trading"
PYTHON = "/root/miniconda3/envs/3.10/bin/python"
SCRIPT = "run_weekend_batch.py"
OUTPUT = "exp_weekend_batch_output.txt"

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

    # Check running experiments
    ps, _ = run_cmd(ssh, "ps aux | grep 'run_exp_' | grep -v grep | wc -l")
    print(f"Currently running experiment processes: {ps}")

    # Git pull to get latest LIVE_PARITY_KWARGS
    print("\n--- Git pull ---")
    out, _ = run_cmd(ssh, f"cd {PROJECT} && git stash 2>/dev/null; git pull", timeout=30)
    print(f"  {out[:300]}")

    # Upload script
    print("\n--- Upload ---")
    sftp = ssh.open_sftp()
    sftp.put(SCRIPT, f"{PROJECT}/{SCRIPT}")
    sftp.close()
    print(f"  Uploaded {SCRIPT}")

    # Launch
    print(f"\n--- Launching weekend batch ---")
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
        out2, _ = run_cmd(ssh, f"head -40 {PROJECT}/{OUTPUT} 2>/dev/null")
        print(out2)
        ssh.close()
        return

    time.sleep(15)
    out, _ = run_cmd(ssh, f"tail -15 {PROJECT}/{OUTPUT} 2>/dev/null")
    print(f"\n--- Initial Output ---")
    print(out)

    # Total processes
    ps_all, _ = run_cmd(ssh, "ps aux | grep python | grep -v grep | wc -l")
    print(f"\nTotal python processes: {ps_all}")

    ssh.close()
    print(f"\nExpected runtime: ~60-90 minutes (6 experiments sequential, each parallel internally)")
    print(f"Monitor: python _check_weekend.py")

if __name__ == "__main__":
    main()
