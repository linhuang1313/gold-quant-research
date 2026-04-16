"""Check which experiments are completed vs still running."""
import paramiko, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westc.seetacloud.com', port=30886, username='root',
            password='r1zlTZQUb+E4', timeout=15)

# Check all output files for completion
check_script = r"""
import os, glob
files = sorted(glob.glob('/root/gold-quant-trading/exp_*_output.txt') + 
               glob.glob('/root/gold-quant-trading/trail_grid*_output.txt'))
done = []
running = []
for f in files:
    name = os.path.basename(f)
    size = os.path.getsize(f)
    with open(f, 'r', errors='replace') as fh:
        content = fh.read()
    if 'Completed:' in content or 'Total runtime:' in content:
        # Find the runtime line
        for line in content.split('\n'):
            if 'Total runtime:' in line:
                done.append((name, size, line.strip()))
                break
        else:
            done.append((name, size, 'Completed'))
    else:
        lines = content.strip().split('\n')
        last = lines[-1][:80] if lines else '(empty)'
        running.append((name, size, last))

print('=== COMPLETED ===')
for name, size, info in done:
    print(f'  {name:<45s}  {size:>8,}B  {info}')
print(f'\n  Total: {len(done)} completed')

print('\n=== STILL RUNNING ===')
for name, size, last in running:
    print(f'  {name:<45s}  {size:>8,}B  last: {last}')
print(f'\n  Total: {len(running)} running')
"""

# Write check script to remote
ssh.exec_command('cat > /tmp/check_status.py << PYEOF\n' + check_script + '\nPYEOF')
import time; time.sleep(1)

stdin, stdout, stderr = ssh.exec_command(
    '/root/miniconda3/bin/python /tmp/check_status.py', timeout=15)
print(stdout.read().decode('utf-8', errors='replace'))
err = stderr.read().decode('utf-8', errors='replace').strip()
if err:
    print('STDERR:', err[:200])

# Process list
print('=== ACTIVE PROCESSES ===')
stdin2, stdout2, stderr2 = ssh.exec_command(
    "ps aux | grep 'python.*run_' | grep -v grep | grep -v check_status", timeout=10)
lines = stdout2.read().decode('utf-8', errors='replace').strip().split('\n')
for line in lines:
    parts = line.split()
    if len(parts) > 10:
        pid = parts[1]
        cpu = parts[2]
        for p in parts[10:]:
            if 'run_' in p:
                print('  PID=' + pid + '  CPU=' + cpu + '%  ' + p)
                break

stdin3, stdout3, _ = ssh.exec_command('uptime', timeout=10)
print('\nLoad:', stdout3.read().decode('utf-8', errors='replace').strip())

ssh.close()
