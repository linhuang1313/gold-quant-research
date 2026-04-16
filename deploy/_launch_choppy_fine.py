"""Launch choppy fine sweep (parallel) on remote server."""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
PROJECT = "/root/gold-quant-trading"
PYTHON = "/root/miniconda3/envs/3.10/bin/python"
SCRIPT = "run_exp_choppy_fine.py"
OUTPUT = "exp_choppy_fine_output.txt"

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

    # Kill old run
    print("--- Kill old process ---")
    run_cmd(ssh, "pkill -f run_exp_choppy_fine 2>/dev/null")
    time.sleep(2)

    # Git pull
    print("--- Git pull ---")
    out, err = run_cmd(ssh, f"cd {PROJECT} && git stash && git pull", timeout=30)
    print(f"  {out[:300]}")

    # Upload script
    print("\n--- Upload ---")
    sftp = ssh.open_sftp()
    sftp.put(SCRIPT, f"{PROJECT}/{SCRIPT}")
    sftp.close()
    print(f"  Uploaded {SCRIPT}")

    # Launch
    print(f"\n--- Launching (parallel, 21 workers) ---")
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

    # Wait for initial output
    time.sleep(10)
    out, _ = run_cmd(ssh, f"tail -10 {PROJECT}/{OUTPUT} 2>/dev/null")
    print(f"\n--- Initial Output ---")
    print(out)

    ssh.close()
    print(f"\nExpected runtime: ~15-20 minutes (parallel).")
    print(f"Monitor: python _check_choppy_fine.py")

if __name__ == "__main__":
    main()
