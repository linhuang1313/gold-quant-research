"""Check what proxy/acceleration options are available on remote server."""
import paramiko, time, sys
sys.stdout.reconfigure(line_buffering=True)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('connect.westd.seetacloud.com', port=45630, username='root',
          password='r1zlTZQUb+E4', timeout=60)
c.get_transport().set_keepalive(5)
print('Connected', flush=True)

def run(cmd, t=60):
    print(f'$ {cmd}', flush=True)
    tr = c.get_transport(); ch = tr.open_session(); ch.settimeout(t)
    ch.exec_command(cmd)
    out = b''
    while True:
        time.sleep(0.3)
        if ch.recv_ready(): out += ch.recv(65536)
        if ch.recv_stderr_ready(): out += ch.recv_stderr(65536)
        if ch.exit_status_ready():
            while ch.recv_ready(): out += ch.recv(65536)
            while ch.recv_stderr_ready(): out += ch.recv_stderr(65536)
            break
    rc = ch.recv_exit_status(); ch.close()
    print(out.decode('utf-8', 'replace')[:2000], flush=True)
    print(f'[rc={rc}]', flush=True)
    return out.decode('utf-8', 'replace'), rc

# Kill any lingering git clone
run('pkill -f "git clone" 2>/dev/null; sleep 1; echo ok')

# AutoDL academic acceleration (source /etc/network_turbo)
run('cat /etc/network_turbo 2>/dev/null || echo "no /etc/network_turbo"')
run('bash -c "source /etc/network_turbo 2>/dev/null && env | grep -i proxy" || echo "no network_turbo env"')

# Check proxychains
run('which proxychains proxychains4 2>/dev/null || echo "no proxychains"')
run('ls /etc/proxychains* 2>/dev/null || echo "no proxychains conf"')

# Check clash/v2ray
run('which clash v2ray 2>/dev/null || echo "no clash/v2ray"')
run('ps aux | grep -E "clash|v2ray|proxy" | grep -v grep || echo "no proxy process"')

# Test some mirror URLs
run('curl -s --connect-timeout 10 -o /dev/null -w "ghproxy: %{http_code} %{time_total}s\n" https://ghproxy.com/ 2>&1 || echo "ghproxy fail"', 15)
run('curl -s --connect-timeout 10 -o /dev/null -w "mirror.ghproxy: %{http_code} %{time_total}s\n" https://mirror.ghproxy.com/ 2>&1 || echo "mirror fail"', 15)

# Check if direct github is just slow (small file test)
run('curl -s --connect-timeout 10 -o /dev/null -w "github raw: %{http_code} %{time_total}s\n" https://raw.githubusercontent.com/linhuang1313/gold-quant-research/main/README.md 2>&1 || echo "github raw fail"', 20)

c.close()
print('Done', flush=True)
