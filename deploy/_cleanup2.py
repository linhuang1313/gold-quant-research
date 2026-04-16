"""Kill all duplicate session_analysis processes except the latest."""
import paramiko, time, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westc.seetacloud.com', port=30886, username='root',
            password='r1zlTZQUb+E4', timeout=15)

# Kill all session_analysis duplicates except the last real worker (41398)
# Also kill the parent/wrapper processes (CPU=0.0%)
kill_pids = ['41094', '41095', '41267', '41268', '41397']
for pid in kill_pids:
    ssh.exec_command('kill ' + pid + ' 2>/dev/null')

time.sleep(3)

stdin, stdout, stderr = ssh.exec_command(
    "ps aux | grep 'python.*run_' | grep -v grep",
    timeout=10
)
output = stdout.read().decode('utf-8', errors='replace').strip()
print("=== Final process list ===")
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
print("Load: " + stdout2.read().decode('utf-8', errors='replace').strip())

ssh.close()
