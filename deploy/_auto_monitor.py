"""
Auto-monitor: 每 15 分钟检查一次服务器状态，记录到本地日志。
运行方式: python _auto_monitor.py
在你睡觉期间让它跑着，明早查看 monitor_log.txt 即可了解全过程。
按 Ctrl+C 停止。
"""
import sys, io, time, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from datetime import datetime
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"
LOG_FILE = "monitor_log.txt"
CHECK_INTERVAL = 900  # 15 minutes

def log(msg, f=None):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if f:
        f.write(line + "\n")
        f.flush()

def check_once(f):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
    except Exception as e:
        log(f"CONNECTION FAILED: {e}", f)
        return

    def run_cmd(cmd, timeout=15):
        try:
            _, o, _ = ssh.exec_command(cmd, timeout=timeout)
            return o.read().decode('utf-8', errors='replace').strip()
        except Exception as e:
            return f"CMD_ERROR: {e}"

    r2_running = "run_round2" in run_cmd("ps aux | grep run_round2 | grep -v grep")
    r3_running = "run_round3" in run_cmd("ps aux | grep run_round3 | grep -v grep")
    r4_running = "run_round4" in run_cmd("ps aux | grep run_round4 | grep -v grep")
    chain_running = "chain" in run_cmd("ps aux | grep chain | grep -v grep")

    status = []
    if r2_running: status.append("R2:RUNNING")
    if r3_running: status.append("R3:RUNNING")
    if r4_running: status.append("R4:RUNNING")
    if chain_running: status.append("CHAINER:OK")
    if not any([r2_running, r3_running, r4_running]):
        if chain_running:
            status.append("WAITING_BETWEEN_ROUNDS")
        else:
            status.append("ALL_COMPLETE")

    log(f"Status: {' | '.join(status)}", f)

    # Show master logs
    for rnd in ["round2", "round3", "round4"]:
        master = run_cmd(f"cat {REMOTE_DIR}/{rnd}_results/00_master_log.txt 2>/dev/null")
        if master and "NOT_YET" not in master:
            lines = master.strip().split('\n')
            last_lines = lines[-3:] if len(lines) > 3 else lines
            for l in last_lines:
                log(f"  [{rnd}] {l.strip()}", f)

    # File counts
    for rnd in ["round2", "round3", "round4"]:
        count = run_cmd(f"ls {REMOTE_DIR}/{rnd}_results/*.txt 2>/dev/null | wc -l")
        log(f"  {rnd} result files: {count}", f)

    # Errors
    for rnd in ["round2", "round3", "round4"]:
        stdout_file = f"{REMOTE_DIR}/{rnd}_stdout.txt"
        err_count = run_cmd(f"grep -ciE 'FAILED|Traceback' {stdout_file} 2>/dev/null")
        if err_count and err_count != "0":
            log(f"  WARNING: {rnd} has {err_count} error(s)!", f)
            err_lines = run_cmd(f"grep -iE 'FAILED|Traceback' {stdout_file} | head -5")
            log(f"    {err_lines}", f)

    # CPU
    load = run_cmd("cat /proc/loadavg")
    log(f"  Load: {load}", f)

    # Chain log tail
    chain_log = run_cmd(f"tail -3 {REMOTE_DIR}/chain_full_log.txt 2>/dev/null")
    if chain_log:
        log(f"  Chainer: {chain_log}", f)

    ssh.close()


def main():
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILE)
    print(f"Monitor log: {log_path}")
    print(f"Check interval: {CHECK_INTERVAL}s ({CHECK_INTERVAL//60} min)")
    print("Press Ctrl+C to stop.\n")

    with open(log_path, 'a', encoding='utf-8') as f:
        log("=" * 60, f)
        log("AUTO MONITOR STARTED", f)
        log(f"Interval: {CHECK_INTERVAL}s", f)
        log("=" * 60, f)

        check_count = 0
        try:
            while True:
                check_count += 1
                log(f"--- Check #{check_count} ---", f)
                check_once(f)
                log("", f)
                time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            log("Monitor stopped by user.", f)


if __name__ == "__main__":
    main()
