"""Kill duplicate processes and verify clean state."""
import paramiko, time, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westc.seetacloud.com', port=30886, username='root',
            password='r1zlTZQUb+E4', timeout=15)

# Kill duplicate ef_parallel processes (keep 40957 which is the latest)
for pid in ['40562', '40563', '40773', '40774', '40956']:
    ssh.exec_command('kill ' + pid + ' 2>/dev/null')

time.sleep(3)

stdin, stdout, stderr = ssh.exec_command(
    "ps aux | grep 'python.*run_' | grep -v grep",
    timeout=10
)
output = stdout.read().decode('utf-8', errors='replace').strip()
print("=== Clean process list ===")
count = 0
for line in output.split('\n'):
    parts = line.split()
    if len(parts) > 10:
        pid = parts[1]
        cpu = parts[2]
        for p in parts[10:]:
            if 'run_' in p:
                print("  PID=" + pid + "  CPU=" + cpu + "%  " + p)
                count += 1
                break

print("\nTotal: " + str(count) + " processes")

stdin2, stdout2, _ = ssh.exec_command('uptime', timeout=10)
load_str = stdout2.read().decode('utf-8', errors='replace').strip()
print("Load: " + load_str)

ssh.close()
print("Done.")
