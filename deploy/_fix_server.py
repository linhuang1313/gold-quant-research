"""Fix server python env and relaunch L4 audit."""
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

    # Find correct python
    print("\n--- Finding Python ---")
    for cmd in ["which python", "which python3", "which python3.10", "which python3.11",
                "ls /usr/bin/python*", "ls /opt/conda/bin/python*", "conda info --envs 2>/dev/null",
                "pip --version 2>/dev/null", "pip3 --version 2>/dev/null"]:
        out, err = run_cmd(ssh, cmd)
        print(f"  {cmd}: {out} {err}")

    # Check what python was used for previous experiments
    print("\n--- Previous experiment launch method ---")
    out, _ = run_cmd(ssh, f"head -1 {PROJECT}/exp_combo_output.txt 2>/dev/null")
    print(f"  combo first line: {out}")
    out, _ = run_cmd(ssh, f"head -5 {PROJECT}/exp_k_regime_output.txt 2>/dev/null")
    print(f"  k_regime first lines: {out}")

    # Check if conda environment
    out, _ = run_cmd(ssh, "conda activate base 2>/dev/null && which python")
    print(f"  conda base python: {out}")

    # Try to find numpy
    out, _ = run_cmd(ssh, "python3 -c 'import numpy; print(numpy.__version__)' 2>&1")
    print(f"  python3 numpy: {out}")

    out, _ = run_cmd(ssh, "/opt/conda/bin/python -c 'import numpy; print(numpy.__version__)' 2>&1")
    print(f"  conda python numpy: {out}")

    # Install if needed
    print("\n--- Installing deps ---")
    out, err = run_cmd(ssh, "pip3 install numpy pandas 2>&1", timeout=60)
    print(f"  pip3 install: {out[:200]}")

    # Retry launch
    print("\n--- Relaunching L4 Audit ---")
    launch_cmd = f"cd {PROJECT} && nohup python3 -u run_exp_l4_audit.py > exp_l4_audit_output.txt 2>&1 &"
    run_cmd(ssh, launch_cmd, timeout=10)
    time.sleep(5)

    out, _ = run_cmd(ssh, "ps aux | grep run_exp_l4_audit | grep -v grep")
    if out:
        print("L4 audit is RUNNING:")
        print(out)
    else:
        print("Still not running. Checking output...")
        out2, _ = run_cmd(ssh, f"head -20 {PROJECT}/exp_l4_audit_output.txt 2>/dev/null")
        print(out2)

    # Check initial output
    time.sleep(3)
    out, _ = run_cmd(ssh, f"tail -10 {PROJECT}/exp_l4_audit_output.txt 2>/dev/null")
    print("\n--- Output ---")
    print(out)

    ssh.close()

if __name__ == "__main__":
    main()
