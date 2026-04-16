"""Deep investigation and fix of server Python env."""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
PROJECT = "/root/gold-quant-trading"

def run_cmd(ssh, cmd, timeout=60):
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

    # Deep env scan
    print("--- Deep Python scan ---")
    cmds = [
        "find / -name 'python*' -type f 2>/dev/null | head -30",
        "find / -name 'pip*' -type f 2>/dev/null | head -20",
        "find / -name 'numpy' -type d 2>/dev/null | head -10",
        "cat /etc/os-release | head -5",
        "apt list --installed 2>/dev/null | grep -i python | head -20",
        "dpkg -l | grep python | head -20",
        "ls -la /root/.local/bin/ 2>/dev/null | head -10",
        "ls -la /root/venv/ 2>/dev/null | head -10",
        "ls -la /root/.venv/ 2>/dev/null | head -10",
        "find /root -name 'activate' -path '*/bin/activate' 2>/dev/null | head -5",
        "cat /root/.bashrc 2>/dev/null | grep -i 'python\\|pip\\|venv\\|conda\\|PATH' | head -10",
        "env | grep -i python | head -10",
        "env | grep -i PATH | head -5",
    ]
    for cmd in cmds:
        out, err = run_cmd(ssh, cmd, timeout=30)
        result = out or err
        if result:
            print(f"\n  $ {cmd}")
            for line in result.split('\n')[:10]:
                print(f"    {line}")

    # Check how previous experiments ran - maybe they used a venv
    print("\n--- Checking shell history ---")
    out, _ = run_cmd(ssh, "cat /root/.bash_history 2>/dev/null | grep -i 'python\\|pip\\|venv\\|nohup' | tail -30")
    if out:
        print(out)

    ssh.close()

if __name__ == "__main__":
    main()
