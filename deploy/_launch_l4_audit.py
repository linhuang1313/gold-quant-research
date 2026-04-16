"""Upload and launch L4 audit experiment on remote server."""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
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
    print("Connecting to server...")
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

    # 1. Check server status
    print("\n--- Server Status ---")
    print("Uptime:", run_cmd(ssh, "uptime"))
    print("Cores:", run_cmd(ssh, "nproc"))

    # 2. Check running experiments
    print("\n--- Running Processes ---")
    out = run_cmd(ssh, "ps aux | grep 'python.*run_exp\\|python.*run_overnight\\|python.*run_trail' | grep -v grep")
    if out:
        print(out)
    else:
        print("No experiments running.")

    # 3. Check combo/overnight results
    print("\n--- Combo/Overnight Results ---")
    for f in ['exp_combo_output.txt', 'overnight_batch_output.txt', 'exp_k_regime_output.txt']:
        size = run_cmd(ssh, f"wc -c < {PROJECT}/{f} 2>/dev/null")
        last = run_cmd(ssh, f"tail -3 {PROJECT}/{f} 2>/dev/null | tail -1")
        print(f"  {f}: {size} bytes, last: {last[:80]}")

    # 4. Upload the audit script
    print("\n--- Uploading run_exp_l4_audit.py ---")
    sftp = ssh.open_sftp()
    sftp.put("run_exp_l4_audit.py", f"{PROJECT}/run_exp_l4_audit.py")
    sftp.close()
    print("Upload complete.")

    # 5. Git pull (to get latest engine code)
    print("\n--- Git Pull ---")
    out = run_cmd(ssh, f"cd {PROJECT} && git pull", timeout=30)
    print(out[:200])

    # 6. Launch the experiment
    print("\n--- Launching L4 Audit ---")
    launch_cmd = f"cd {PROJECT} && nohup python3 -u run_exp_l4_audit.py > exp_l4_audit_output.txt 2>&1 &"
    run_cmd(ssh, launch_cmd, timeout=10)
    time.sleep(3)

    # 7. Verify it's running
    out = run_cmd(ssh, f"ps aux | grep run_exp_l4_audit | grep -v grep")
    if out:
        print("L4 audit is RUNNING:")
        print(out)
    else:
        print("WARNING: L4 audit may not have started!")
        out2 = run_cmd(ssh, f"cat {PROJECT}/exp_l4_audit_output.txt 2>/dev/null | head -20")
        print("Output so far:", out2)

    # 8. Check initial output
    time.sleep(5)
    out = run_cmd(ssh, f"tail -10 {PROJECT}/exp_l4_audit_output.txt 2>/dev/null")
    print("\n--- Initial Output ---")
    print(out)

    ssh.close()
    print("\nDone. Script should run ~60-90 minutes.")

if __name__ == "__main__":
    main()
