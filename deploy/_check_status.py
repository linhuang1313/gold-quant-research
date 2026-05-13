#!/usr/bin/env python3
"""Quick status check of all running tmux sessions and recent results on remote."""
import paramiko, time

HOST = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASS = '3sCdENtzYfse'
REMOTE_DIR = '/root/gold-quant-research'


def connect():
    for attempt in range(3):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(HOST, port=PORT, username=USER, password=PASS,
                      timeout=60, banner_timeout=60)
            return c
        except Exception as e:
            print(f"  [Attempt {attempt+1}] {e}")
            time.sleep(5)
    return None


def run(c, cmd, label=None):
    if label:
        print(f"\n=== {label} ===")
    _, o, e = c.exec_command(cmd)
    out = o.read().decode('utf-8', errors='replace')
    err = e.read().decode('utf-8', errors='replace')
    if out.strip():
        print(out.rstrip())
    if err.strip():
        print(f"[stderr] {err.rstrip()}")


def main():
    c = connect()
    if not c:
        print("Cannot connect")
        return
    try:
        run(c, 'tmux ls 2>&1', 'tmux sessions')
        run(c, f'ls -lt {REMOTE_DIR}/results/ 2>/dev/null | head -25', 'results/ (most recent)')
        run(c, f'ls -lt {REMOTE_DIR}/results/*/r*_stdout.txt 2>/dev/null | head -10',
            'stdout files (most recent)')
        run(c, 'ps -ef | grep -E "python|run_r" | grep -v grep | head -10', 'python processes')
        run(c, f'find {REMOTE_DIR}/results -name "*.json" -newer {REMOTE_DIR}/README.md 2>/dev/null '
              '-printf "%T@ %p\\n" | sort -rn | head -15', 'recent json results')
    finally:
        c.close()


if __name__ == '__main__':
    main()
