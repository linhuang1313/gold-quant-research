"""Night scheduler: wait for combo, then launch overnight batch + EXP-K."""
import paramiko, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PYTHON = '/root/miniconda3/bin/python'
WORKDIR = '/root/gold-quant-trading'
HOST = 'connect.westc.seetacloud.com'
PORT = 30886
USER = 'root'
PASS = 'r1zlTZQUb+E4'


def get_ssh():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    return ssh


def run(ssh, cmd, timeout=15):
    try:
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        stdout.channel.settimeout(timeout)
        return stdout.read().decode('utf-8', errors='replace').strip()
    except Exception as e:
        return f"ERROR: {e}"


def is_combo_running(ssh):
    out = run(ssh, "ps aux | grep run_exp_combo | grep python | grep -v grep")
    return bool(out and 'python' in out)


def wait_for_combo():
    """Poll every 5 minutes until combo finishes."""
    print(f"[{time.strftime('%H:%M:%S')}] Waiting for combo test to finish...")
    while True:
        try:
            ssh = get_ssh()
            running = is_combo_running(ssh)
            out = run(ssh, f"wc -l {WORKDIR}/exp_combo_output.txt 2>/dev/null")
            last = run(ssh, f"tail -3 {WORKDIR}/exp_combo_output.txt 2>/dev/null")
            ssh.close()

            if running:
                print(f"[{time.strftime('%H:%M:%S')}] Combo still running. Lines: {out}. Last: {last[:80]}")
                time.sleep(300)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Combo FINISHED! Lines: {out}")
                print(f"  Last: {last}")
                return True
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] SSH error: {e}, retrying in 60s...")
            time.sleep(60)


def upload_and_launch():
    """Upload overnight scripts, then launch them."""
    ssh = get_ssh()

    # Upload run_overnight_batch.py via sftp
    print(f"\n[{time.strftime('%H:%M:%S')}] Uploading scripts...")
    sftp = ssh.open_sftp()
    sftp.put('run_overnight_batch.py', f'{WORKDIR}/run_overnight_batch.py')
    print("  Uploaded run_overnight_batch.py")
    sftp.close()

    # Launch overnight batch
    print(f"[{time.strftime('%H:%M:%S')}] Launching overnight batch...")
    ssh.exec_command(
        f"cd {WORKDIR} && nohup {PYTHON} -u run_overnight_batch.py > overnight_batch_output.txt 2>&1 &"
    )
    time.sleep(3)

    # Launch EXP-K (was NOT STARTED)
    print(f"[{time.strftime('%H:%M:%S')}] Launching EXP-K Regime Bounds...")
    ssh.exec_command(
        f"cd {WORKDIR} && nohup {PYTHON} -u run_exp_k_regime_bounds.py > exp_k_regime_output.txt 2>&1 &"
    )
    time.sleep(5)

    # Verify both are running
    out = run(ssh, "ps aux | grep python | grep -E 'overnight|regime' | grep -v grep")
    print(f"\n[{time.strftime('%H:%M:%S')}] Running processes:")
    print(out if out else "  (NONE - launch failed!)")

    # Check initial output
    for f in ['overnight_batch_output.txt', 'exp_k_regime_output.txt']:
        out = run(ssh, f"head -5 {WORKDIR}/{f} 2>/dev/null")
        print(f"\n  {f}:")
        print(f"    {out[:150] if out else '(empty)'}")

    ssh.close()
    print(f"\n[{time.strftime('%H:%M:%S')}] All launched. Good night!")


if __name__ == '__main__':
    wait_for_combo()
    upload_and_launch()
