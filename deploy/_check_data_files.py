"""Check data files on both servers."""
import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SERVERS = [
    ("C", "connect.westc.seetacloud.com", 16005),
    ("D", "connect.westd.seetacloud.com", 35258),
]

def run(c, cmd):
    _, stdout, _ = c.exec_command(cmd, timeout=20)
    return stdout.read().decode('utf-8', errors='replace').strip()

for name, host, port in SERVERS:
    print(f"\n{'='*60}")
    print(f"Server {name}: {host}:{port}")
    print("=" * 60)
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(host, port=port, username="root", password="r1zlTZQUb+E4", timeout=15)

        print("\n--- Data files ---")
        files = run(c, "ls -lhS /root/gold-quant-trading/data/download/ 2>/dev/null")
        if files:
            for line in files.split("\n"):
                print(f"  {line}")
        else:
            print("  No files in data/download/")

        print("\n--- Other data locations ---")
        other = run(c, "find /root/gold-quant-trading/data -name '*.csv' -exec ls -lh {} + 2>/dev/null")
        if other:
            for line in other.split("\n"):
                print(f"  {line}")

        print(f"\n--- Total data size ---")
        print(f"  {run(c, 'du -sh /root/gold-quant-trading/data/ 2>/dev/null')}")

        print(f"\n--- Git remote ---")
        print(f"  {run(c, 'cd /root/gold-quant-trading && git remote -v 2>/dev/null')}")

        print(f"\n--- Git .gitignore data lines ---")
        print(f"  {run(c, 'grep data /root/gold-quant-trading/.gitignore 2>/dev/null')}")

        c.close()
    except Exception as e:
        print(f"  CONNECTION FAILED: {e}")
