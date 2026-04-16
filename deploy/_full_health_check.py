"""Comprehensive health check for all running experiments."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"

def run_cmd(ssh, cmd, timeout=15):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    return stdout.read().decode('utf-8', errors='replace').strip()

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

    print("=" * 70)
    print("FULL HEALTH CHECK")
    print("=" * 70)

    # 1. Processes
    print("\n--- PYTHON PROCESSES ---")
    procs = run_cmd(ssh, "ps aux | grep -E 'run_round|chain' | grep -v grep")
    print(procs if procs else "NONE RUNNING!")

    # 2. Round 2 master log
    print("\n--- ROUND 2 MASTER LOG ---")
    print(run_cmd(ssh, f"cat {REMOTE_DIR}/round2_results/00_master_log.txt"))

    # 3. Round 2 files
    print("\n--- ROUND 2 FILES ---")
    print(run_cmd(ssh, f"ls -lhS {REMOTE_DIR}/round2_results/"))

    # 4. Chain log
    print("\n--- CHAINER LOG ---")
    print(run_cmd(ssh, f"cat {REMOTE_DIR}/chain_full_log.txt 2>/dev/null"))

    # 5. Scripts exist
    print("\n--- SCRIPTS ON SERVER ---")
    print(run_cmd(ssh, f"ls -lh {REMOTE_DIR}/run_round2.py {REMOTE_DIR}/run_round3.py {REMOTE_DIR}/run_round4.py {REMOTE_DIR}/_chain_full.sh"))

    # 6. Round3/4 dirs
    print("\n--- ROUND 3 DIR ---")
    print(run_cmd(ssh, f"ls -la {REMOTE_DIR}/round3_results/ 2>/dev/null"))
    print("\n--- ROUND 4 DIR ---")
    print(run_cmd(ssh, f"ls -la {REMOTE_DIR}/round4_results/ 2>/dev/null"))

    # 7. Check for errors in stdout
    print("\n--- ERRORS IN round2_stdout.txt ---")
    errors = run_cmd(ssh, f"grep -ciE 'error|traceback|exception' {REMOTE_DIR}/round2_stdout.txt 2>/dev/null")
    print(f"  Error/Traceback/Exception mentions: {errors}")
    if errors and int(errors) > 0:
        print(run_cmd(ssh, f"grep -iE 'error|traceback|exception' {REMOTE_DIR}/round2_stdout.txt | head -20"))

    # 8. CPU
    print("\n--- CPU LOAD ---")
    print(run_cmd(ssh, "uptime"))

    # 9. Disk space
    print("\n--- DISK SPACE ---")
    print(run_cmd(ssh, "df -h /root"))

    # 10. Current R2 stdout tail
    print("\n--- round2_stdout.txt (last 5 lines) ---")
    print(run_cmd(ssh, f"tail -5 {REMOTE_DIR}/round2_stdout.txt 2>/dev/null"))

    ssh.close()
    print("\n" + "=" * 70)
    print("HEALTH CHECK COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
