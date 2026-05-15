import paramiko, time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

for i in range(5):
    try:
        print(f'Attempt {i+1}...')
        ssh.connect('connect.westd.seetacloud.com', port=41109, username='root',
                    password='3sCdENtzYfse', timeout=60, banner_timeout=60)
        print('Connected!')
        break
    except Exception as e:
        print(f'  Failed: {e}')
        time.sleep(5*(i+1))

# Kill old
print('Killing old processes...')
_, o, _ = ssh.exec_command('pkill -f run_r230_mega_overnight 2>/dev/null; echo KILLED')
print(o.read().decode().strip())
time.sleep(2)

# Start new - use exec_command but don't wait for nohup output
print('Starting R230...')
transport = ssh.get_transport()
channel = transport.open_session()
channel.exec_command(
    'cd /root/gold-quant-research && '
    'nohup /usr/bin/python3 -u experiments/run_r230_mega_overnight.py '
    '> results/r230_mega_overnight/r230_stdout.txt 2>&1 &'
)
time.sleep(3)
channel.close()

print('Waiting for process to start...')
time.sleep(10)

# Check process
print('Checking process...')
_, o, _ = ssh.exec_command('ps aux | grep run_r230 | grep -v grep | head -3')
procs = o.read().decode().strip()
print(f'Process: {procs}')

# Check output
_, o, _ = ssh.exec_command('wc -l /root/gold-quant-research/results/r230_mega_overnight/r230_stdout.txt 2>/dev/null; echo "---"; tail -15 /root/gold-quant-research/results/r230_mega_overnight/r230_stdout.txt 2>/dev/null')
print(f'Output:\n{o.read().decode().strip()}')

ssh.close()
print('\nDone!')
