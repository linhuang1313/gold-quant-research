"""Check progress of R96-R100 experiments on remote server."""
import paramiko
import time

SERVER = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASSWD = '3sCdENtzYfse'
REMOTE_BASE = '/root/gold-quant-research/results'


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(SERVER, port=PORT, username=USER, password=PASSWD,
              timeout=120, banner_timeout=60, auth_timeout=60)
    return c


def main():
    experiments = [
        ('r96_slippage', 'r96_stdout.txt'),
        ('r97_correlation', 'r97_stdout.txt'),
        ('r98_ml_threshold', 'r98_stdout.txt'),
        ('r99_time_effects', 'r99_stdout.txt'),
        ('r100_circuit_breaker', 'r100_stdout.txt'),
    ]

    c = connect()

    # Check running processes
    _, out, _ = c.exec_command("ps aux | grep run_r | grep -v grep | awk '{print $NF}'", timeout=15)
    running = out.read().decode().strip()
    print("=== Running processes ===")
    print(running if running else "(none)")
    print()

    # Check tail output of each
    for result_dir, stdout_file in experiments:
        _, out, _ = c.exec_command(
            f"tail -5 {REMOTE_BASE}/{result_dir}/{stdout_file} 2>/dev/null", timeout=15)
        txt = out.read().decode().strip()
        print(f"--- {result_dir} ---")
        print(txt[-400:] if txt else "(no output yet)")
        print()

    # Check if JSON results exist
    print("=== Completed (JSON exists) ===")
    for result_dir, _ in experiments:
        json_name = result_dir.split('_', 1)[0] + '_results.json'
        _, out, _ = c.exec_command(
            f"ls -la {REMOTE_BASE}/{result_dir}/{json_name} 2>/dev/null", timeout=15)
        result = out.read().decode().strip()
        if result:
            print(f"  DONE: {result_dir}/{json_name}")

    # Check R101
    print()
    print("=== R101 Mega Experiment ===")
    _, out, _ = c.exec_command(
        f"tail -3 {REMOTE_BASE}/r101_mega_experiment/r101_stdout.txt 2>/dev/null", timeout=15)
    txt = out.read().decode().strip()
    print(txt[-300:] if txt else "(no output)")

    c.close()


if __name__ == '__main__':
    main()
