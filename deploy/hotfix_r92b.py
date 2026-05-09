"""Hotfix: re-upload and relaunch R92-B after timezone fix."""
import paramiko, os, time

HOST = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASS = '3sCdENtzYfse'
REMOTE = '/root/gold-quant-research'
LOCAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=60)
c.get_transport().set_keepalive(15)

# Upload fixed file
print("Uploading fixed script...", flush=True)
sftp = c.open_sftp()
sftp.put(os.path.join(LOCAL, 'experiments', 'run_r92b_multi_strategy.py'),
         REMOTE + '/experiments/run_r92b_multi_strategy.py')
sftp.close()
print("  Done.", flush=True)

# Kill old process
print("Killing old process...", flush=True)
_, out, _ = c.exec_command("pkill -f run_r92b 2>/dev/null; sleep 2", timeout=15)
out.read()
print("  Done.", flush=True)

# Relaunch
print("Relaunching R92-B...", flush=True)
chan = c.get_transport().open_session()
cmd = "cd /root/gold-quant-research && nohup python3 -u experiments/run_r92b_multi_strategy.py > results/r92b_multi_strategy/r92b_stdout.txt 2>&1 &"
chan.exec_command(cmd)
time.sleep(3)
chan.close()
print("  Launched.", flush=True)

# Verify
time.sleep(10)
_, out, _ = c.exec_command("ps aux | grep run_r92b | grep -v grep", timeout=10)
proc = out.read().decode('utf-8', errors='replace').strip()
if proc:
    print(f"  Process RUNNING", flush=True)
else:
    print("  WARNING: Process not found!", flush=True)

# Show initial output
time.sleep(5)
_, out, _ = c.exec_command("head -20 /root/gold-quant-research/results/r92b_multi_strategy/r92b_stdout.txt", timeout=10)
print("\nInitial output:")
print(out.read().decode('utf-8', errors='replace'))
c.close()
print("\nDone. Monitor with: tail -f results/r92b_multi_strategy/r92b_stdout.txt")
