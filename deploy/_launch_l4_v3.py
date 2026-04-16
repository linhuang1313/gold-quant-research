"""Launch L4 audit using correct conda python path."""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
PROJECT = "/root/gold-quant-trading"

# Conda env python with numpy installed
PYTHON = "/root/miniconda3/envs/3.10/bin/python"

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

    # Verify python has numpy
    print("--- Verify Python ---")
    out, err = run_cmd(ssh, f"{PYTHON} -c 'import numpy, pandas; print(f\"numpy={{numpy.__version__}}, pandas={{pandas.__version__}}\")'")
    print(f"  {out} {err}")
    if "Error" in (out + err) or "No module" in (out + err):
        # Try base conda
        alt = "/root/miniconda3/bin/python"
        out2, err2 = run_cmd(ssh, f"{alt} -c 'import numpy, pandas; print(f\"numpy={{numpy.__version__}}, pandas={{pandas.__version__}}\")'")
        print(f"  Trying base conda: {out2} {err2}")
        if "Error" not in (out2 + err2):
            print(f"  Using base conda python")
            PYTHON_USE = alt
        else:
            print("ERROR: No working Python with numpy found!")
            ssh.close()
            return
    else:
        PYTHON_USE = PYTHON

    # Upload script
    print("\n--- Upload ---")
    sftp = ssh.open_sftp()
    sftp.put("run_exp_l4_audit.py", f"{PROJECT}/run_exp_l4_audit.py")
    sftp.close()
    print("Uploaded run_exp_l4_audit.py")

    # Kill any old runs
    run_cmd(ssh, "pkill -f run_exp_l4_audit 2>/dev/null")
    time.sleep(1)

    # Launch
    print(f"\n--- Launching with {PYTHON_USE} ---")
    launch_cmd = f"cd {PROJECT} && nohup {PYTHON_USE} -u run_exp_l4_audit.py > exp_l4_audit_output.txt 2>&1 &"
    run_cmd(ssh, launch_cmd, timeout=10)
    time.sleep(5)

    # Verify
    out, _ = run_cmd(ssh, "ps aux | grep run_exp_l4_audit | grep -v grep")
    if out:
        print("RUNNING:")
        print(out)
    else:
        print("NOT RUNNING! Checking output...")
        out2, _ = run_cmd(ssh, f"head -30 {PROJECT}/exp_l4_audit_output.txt 2>/dev/null")
        print(out2)
        ssh.close()
        return

    # Wait a bit for initial output
    time.sleep(10)
    out, _ = run_cmd(ssh, f"tail -15 {PROJECT}/exp_l4_audit_output.txt 2>/dev/null")
    print("\n--- Initial Output ---")
    print(out)

    ssh.close()
    print("\nDone. Expected runtime: ~60-90 minutes.")

if __name__ == "__main__":
    main()
