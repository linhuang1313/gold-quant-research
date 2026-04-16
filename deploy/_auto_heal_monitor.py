"""
Self-Healing Monitor — 自愈式实验监控
=====================================
每 10 分钟检查一次远程服务器，能自动处理以下故障:

故障 1: Round X 进程崩溃 (Traceback / 非0退出) → 自动重启该 Round
故障 2: Chainer 进程消失 → 判断当前阶段，重建 chainer 继续
故障 3: 进程卡死 (同一个 phase 超过 45 分钟无新输出) → 杀死并重启
故障 4: 磁盘满 → 清理旧 stdout 日志，释放空间
故障 5: SSH 连接失败 → 重试 3 次，间隔递增

运行: python _auto_heal_monitor.py
停止: Ctrl+C
日志: monitor_log.txt (本地)
"""
import sys, io, time, os, traceback
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from datetime import datetime
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
PYTHON = "/root/miniconda3/envs/3.10/bin/python"
REMOTE_DIR = "/root/gold-quant-trading"
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor_log.txt")
CHECK_INTERVAL = 600  # 10 minutes
PHASE_TIMEOUT = 2700  # 45 minutes per phase max
MAX_RESTARTS = 2       # max restarts per round

# Track state across checks
last_master_log_lines = {"round2": 0, "round3": 0, "round4": 0}
last_file_sizes = {}
stall_counter = {"round2": 0, "round3": 0, "round4": 0}
restart_count = {"round2": 0, "round3": 0, "round4": 0}
rounds_completed = set()


def log(msg, f):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode('utf-8', errors='replace').decode('ascii', errors='replace'), flush=True)
    f.write(line + "\n")
    f.flush()


def ssh_connect(f, retries=3):
    for attempt in range(1, retries + 1):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=20)
            return ssh
        except Exception as e:
            log(f"SSH CONNECT FAILED (attempt {attempt}/{retries}): {e}", f)
            if attempt < retries:
                wait = 30 * attempt
                log(f"  Retrying in {wait}s...", f)
                time.sleep(wait)
    return None


def run_cmd(ssh, cmd, timeout=20):
    try:
        _, o, e = ssh.exec_command(cmd, timeout=timeout)
        return o.read().decode('utf-8', errors='replace').strip()
    except Exception as ex:
        return f"CMD_ERROR: {ex}"


def is_process_running(ssh, pattern):
    out = run_cmd(ssh, f"ps aux | grep '{pattern}' | grep -v grep")
    return bool(out and pattern in out)


def get_master_log_line_count(ssh, rnd):
    out = run_cmd(ssh, f"wc -l < {REMOTE_DIR}/{rnd}_results/00_master_log.txt 2>/dev/null")
    try:
        return int(out)
    except (ValueError, TypeError):
        return 0


def get_file_sizes(ssh, rnd):
    out = run_cmd(ssh, f"ls -la {REMOTE_DIR}/{rnd}_results/*.txt 2>/dev/null")
    sizes = {}
    if out and "CMD_ERROR" not in out:
        for line in out.split('\n'):
            parts = line.split()
            if len(parts) >= 5 and '.txt' in line:
                try:
                    sizes[parts[-1]] = int(parts[4])
                except (ValueError, IndexError):
                    pass
    return sizes


def check_for_errors(ssh, rnd):
    """Check if a round has fatal errors in its stdout."""
    exists = run_cmd(ssh, f"test -f {REMOTE_DIR}/{rnd}_stdout.txt && echo yes || echo no")
    if exists != "yes":
        return False
    out = run_cmd(ssh, f"grep -c 'FAILED' {REMOTE_DIR}/{rnd}_stdout.txt 2>/dev/null")
    try:
        return int(out) > 0
    except (ValueError, TypeError):
        return False


def check_master_log_completion(ssh, rnd):
    """Check if master log says ALL COMPLETE."""
    out = run_cmd(ssh, f"grep -c 'ALL COMPLETE' {REMOTE_DIR}/{rnd}_results/00_master_log.txt 2>/dev/null")
    try:
        return int(out) > 0
    except (ValueError, TypeError):
        return False


def restart_round(ssh, rnd, f):
    """Restart a specific round."""
    if restart_count[rnd] >= MAX_RESTARTS:
        log(f"  SKIP RESTART: {rnd} already restarted {MAX_RESTARTS} times", f)
        return False

    log(f"  ACTION: Restarting {rnd}...", f)

    # Kill any remnants
    run_cmd(ssh, f"pkill -f 'run_{rnd}.py' 2>/dev/null")
    time.sleep(3)

    # Backup old stdout
    run_cmd(ssh, f"mv {REMOTE_DIR}/{rnd}_stdout.txt {REMOTE_DIR}/{rnd}_stdout_bak.txt 2>/dev/null")

    # Restart
    cmd = f"cd {REMOTE_DIR} && nohup {PYTHON} -u run_{rnd}.py > {rnd}_stdout.txt 2>&1 &"
    run_cmd(ssh, cmd)
    time.sleep(5)

    if is_process_running(ssh, f"run_{rnd}.py"):
        restart_count[rnd] += 1
        stall_counter[rnd] = 0
        log(f"  RESTART SUCCESS: {rnd} (restart #{restart_count[rnd]})", f)
        return True
    else:
        log(f"  RESTART FAILED: {rnd} process did not start!", f)
        return False


def rebuild_chainer(ssh, current_round, f):
    """Rebuild chainer to continue from the current round."""
    log(f"  ACTION: Rebuilding chainer from {current_round}...", f)

    remaining = []
    for rnd in ["round2", "round3", "round4"]:
        if rnd not in rounds_completed and rnd != current_round:
            remaining.append(rnd)

    if not remaining and current_round in rounds_completed:
        log(f"  All rounds completed, no chainer needed.", f)
        return

    script_lines = ['#!/bin/bash', f'echo "[$(date)] Rebuilt chainer from {current_round}"']

    if current_round and current_round not in rounds_completed:
        rnd = current_round
        script_lines.append(f'echo "[$(date)] Waiting for {rnd}..."')
        script_lines.append(f'while pgrep -f "run_{rnd}.py" > /dev/null 2>&1; do sleep 30; done')
        script_lines.append(f'echo "[$(date)] {rnd} finished."')

    for rnd in remaining:
        script_lines.append(f'echo "[$(date)] Starting {rnd}..."')
        script_lines.append('sleep 5')
        script_lines.append(f'cd {REMOTE_DIR}')
        script_lines.append(f'{PYTHON} -u run_{rnd}.py > {rnd}_stdout.txt 2>&1')
        script_lines.append(f'echo "[$(date)] {rnd} finished."')

    script_lines.append('echo "[$(date)] ALL ROUNDS COMPLETE."')
    script_content = '\n'.join(script_lines)

    # Write and launch
    run_cmd(ssh, f"pkill -f '_chain_full.sh' 2>/dev/null")
    time.sleep(2)
    # Write script via echo
    for i, line in enumerate(script_lines):
        op = '>' if i == 0 else '>>'
        safe_line = line.replace("'", "'\\''")
        run_cmd(ssh, f"echo '{safe_line}' {op} {REMOTE_DIR}/_chain_rebuilt.sh")
    run_cmd(ssh, f"chmod +x {REMOTE_DIR}/_chain_rebuilt.sh")
    run_cmd(ssh, f"cd {REMOTE_DIR} && nohup bash _chain_rebuilt.sh > chain_rebuilt_log.txt 2>&1 &")
    time.sleep(3)

    if is_process_running(ssh, "_chain_rebuilt.sh"):
        log(f"  CHAINER REBUILT OK", f)
    else:
        log(f"  CHAINER REBUILD FAILED", f)


def detect_current_round(ssh):
    """Detect which round is currently running or should run next."""
    if is_process_running(ssh, "run_round2.py"):
        return "round2"
    if is_process_running(ssh, "run_round3.py"):
        return "round3"
    if is_process_running(ssh, "run_round4.py"):
        return "round4"
    return None


def check_disk(ssh, f):
    """Check disk and clean if needed."""
    out = run_cmd(ssh, "df -h /root | tail -1")
    if out and "CMD_ERROR" not in out:
        parts = out.split()
        if len(parts) >= 5:
            use_pct = parts[4].replace('%', '')
            try:
                if int(use_pct) > 90:
                    log(f"  DISK WARNING: {use_pct}% used! Cleaning up...", f)
                    run_cmd(ssh, f"rm -f {REMOTE_DIR}/*_stdout_bak.txt 2>/dev/null")
                    run_cmd(ssh, "find /tmp -name '*.py' -mmin +120 -delete 2>/dev/null")
                    log(f"  Cleanup done.", f)
                    return True
            except ValueError:
                pass
    return False


def determine_next_round():
    """Figure out what round should run next based on completion."""
    for rnd in ["round2", "round3", "round4"]:
        if rnd not in rounds_completed:
            return rnd
    return None


def check_once(f):
    global rounds_completed

    ssh = ssh_connect(f)
    if not ssh:
        log("ALL SSH RETRIES FAILED - will retry next cycle", f)
        return

    try:
        current_round = detect_current_round(ssh)
        chainer_alive = is_process_running(ssh, "_chain_full.sh") or is_process_running(ssh, "_chain_rebuilt.sh")

        # --- Update completion status ---
        for rnd in ["round2", "round3", "round4"]:
            if rnd not in rounds_completed and check_master_log_completion(ssh, rnd):
                rounds_completed.add(rnd)
                log(f"  COMPLETED: {rnd}", f)

        status_parts = []
        for rnd in ["round2", "round3", "round4"]:
            if rnd in rounds_completed:
                status_parts.append(f"{rnd}:DONE")
            elif current_round == rnd:
                status_parts.append(f"{rnd}:RUNNING")
            else:
                status_parts.append(f"{rnd}:PENDING")
        status_parts.append(f"chainer:{'OK' if chainer_alive else 'DEAD'}")
        log(f"Status: {' | '.join(status_parts)}", f)

        # --- Show progress ---
        if current_round:
            ml_lines = get_master_log_line_count(ssh, current_round)
            cur_sizes = get_file_sizes(ssh, current_round)
            last_log = run_cmd(ssh, f"tail -2 {REMOTE_DIR}/{current_round}_results/00_master_log.txt 2>/dev/null")
            file_count = len([k for k in cur_sizes if '00_master' not in k])
            log(f"  {current_round}: {file_count} result files, log lines={ml_lines}", f)
            if last_log:
                for ll in last_log.split('\n'):
                    log(f"    {ll.strip()}", f)

            # --- FAULT 1: Check for crash (FAILED/Traceback in stdout) ---
            has_errors = check_for_errors(ssh, current_round)
            if has_errors:
                # Check if it's still running despite errors
                if not is_process_running(ssh, f"run_{current_round}.py"):
                    log(f"  FAULT: {current_round} CRASHED (errors in stdout, process dead)", f)
                    # Check if master log shows it actually completed (phase-level try/except)
                    if check_master_log_completion(ssh, current_round):
                        log(f"  Actually completed despite phase errors - OK", f)
                        rounds_completed.add(current_round)
                    else:
                        restart_round(ssh, current_round, f)
                else:
                    # Running with errors = phase-level error caught by try/except, continuing
                    log(f"  Note: {current_round} has phase-level errors but still running (OK)", f)

            # --- FAULT 3: Check for stall ---
            # Also check stdout growth as progress indicator
            stdout_size = run_cmd(ssh, f"wc -c < {REMOTE_DIR}/{current_round}_stdout.txt 2>/dev/null")
            try:
                stdout_size = int(stdout_size)
            except (ValueError, TypeError):
                stdout_size = 0
            prev_stdout = last_file_sizes.get(f"{current_round}_stdout", 0)

            prev_sizes = last_file_sizes.get(current_round, {})
            stdout_growing = stdout_size > prev_stdout
            files_growing = cur_sizes != prev_sizes or ml_lines != last_master_log_lines.get(current_round, 0)

            if prev_sizes and not files_growing and not stdout_growing:
                stall_counter[current_round] += 1
                stall_minutes = stall_counter[current_round] * (CHECK_INTERVAL // 60)
                log(f"  STALL WARNING: {current_round} no progress for {stall_minutes} min (count={stall_counter[current_round]})", f)

                load = run_cmd(ssh, "cat /proc/loadavg")
                try:
                    load1 = float(load.split()[0])
                except (ValueError, IndexError):
                    load1 = 0

                if stall_minutes >= 45 and load1 < 3:
                    log(f"  FAULT: {current_round} STUCK ({stall_minutes} min, load={load1:.1f})", f)
                    run_cmd(ssh, f"pkill -f 'run_{current_round}.py' 2>/dev/null")
                    time.sleep(5)
                    restart_round(ssh, current_round, f)
                elif stall_minutes >= 90:
                    log(f"  FAULT: {current_round} STUCK >90 min even with load={load1:.1f}. Force restart.", f)
                    run_cmd(ssh, f"pkill -f 'run_{current_round}.py' 2>/dev/null")
                    time.sleep(5)
                    restart_round(ssh, current_round, f)
                elif load1 >= 3:
                    log(f"  Load={load1:.1f}, likely computing heavy phase — extending patience", f)
            else:
                if stall_counter[current_round] > 0:
                    log(f"  Progress resumed (stall counter reset)", f)
                stall_counter[current_round] = 0

            last_file_sizes[f"{current_round}_stdout"] = stdout_size

            last_file_sizes[current_round] = cur_sizes
            last_master_log_lines[current_round] = ml_lines

        # --- FAULT 2: No round running + chainer dead + not all complete ---
        if not current_round and not chainer_alive:
            next_round = determine_next_round()
            if next_round:
                log(f"  FAULT: Nothing running, chainer dead, {next_round} not done yet!", f)
                # Direct restart of next round + rebuild chainer for the rest
                restart_round(ssh, next_round, f)
                rebuild_chainer(ssh, next_round, f)
            else:
                log(f"  ALL EXPERIMENTS COMPLETE! No action needed.", f)

        # --- FAULT 2b: Round finished but chainer didn't start next ---
        if not current_round and chainer_alive:
            # Chainer is running but no python round process = chainer is in sleep/wait
            # This is normal transition between rounds
            log(f"  Chainer alive, no round running — likely transitioning between rounds", f)

        # --- FAULT 4: Disk check ---
        check_disk(ssh, f)

        # CPU load info
        load = run_cmd(ssh, "cat /proc/loadavg")
        log(f"  Load: {load}", f)

    except Exception as e:
        log(f"CHECK ERROR: {e}", f)
        log(traceback.format_exc(), f)
    finally:
        ssh.close()


def main():
    print(f"Self-Healing Monitor")
    print(f"Log: {LOG_FILE}")
    print(f"Check interval: {CHECK_INTERVAL}s ({CHECK_INTERVAL // 60} min)")
    print(f"Phase timeout: {PHASE_TIMEOUT}s ({PHASE_TIMEOUT // 60} min)")
    print(f"Max restarts per round: {MAX_RESTARTS}")
    print(f"Press Ctrl+C to stop.\n")

    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        log("=" * 70, f)
        log("SELF-HEALING MONITOR STARTED", f)
        log(f"Interval: {CHECK_INTERVAL}s, Phase timeout: {PHASE_TIMEOUT}s, Max restarts: {MAX_RESTARTS}", f)
        log("=" * 70, f)

        check_count = 0
        try:
            while True:
                check_count += 1
                log(f"{'='*40} Check #{check_count} {'='*40}", f)
                check_once(f)

                # Check if everything is done
                if len(rounds_completed) == 3:
                    log("ALL 3 ROUNDS COMPLETED. Monitor will keep checking for 1 more cycle then exit.", f)
                    time.sleep(CHECK_INTERVAL)
                    log(f"{'='*40} Final Check {'='*40}", f)
                    check_once(f)
                    log("MONITOR EXITING — all experiments done.", f)
                    break

                log("", f)
                time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            log("Monitor stopped by user (Ctrl+C).", f)

    print(f"\nMonitor ended. Full log: {LOG_FILE}")


if __name__ == "__main__":
    main()
