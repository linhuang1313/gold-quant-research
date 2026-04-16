"""Upload 11-year gold data to GitHub from server."""
import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westc.seetacloud.com"
PORT = 16005
REMOTE_DIR = "/root/gold-quant-trading"

def run(c, cmd, timeout=60):
    print(f"  >>> {cmd[:120]}")
    _, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()
    if out:
        for line in out.split("\n")[-20:]:
            print(f"      {line}")
    if err and "warning" not in err.lower():
        for line in err.split("\n")[-10:]:
            print(f"  ERR {line}")
    return out

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username="root", password="r1zlTZQUb+E4", timeout=15)

print("=== Upload 11-year gold data to GitHub ===\n")

print("--- Step 1: Check current .gitignore ---")
run(c, f"cat {REMOTE_DIR}/.gitignore")

print("\n--- Step 2: Update .gitignore to allow data CSV files ---")
new_gitignore = """__pycache__/
*.pyc
.env
*.log
data/download/
data/bars_*.json
results/
.idea/
.vscode/
*.egg-info/
dist/
build/

# Allow core data files (force-added)
!data/download/xauusd-m15-bid-2015-01-01-2026-04-10.csv
!data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv
!data/download/xauusd-m15-spread-*.csv
"""

run(c, f"cat > {REMOTE_DIR}/.gitignore << 'GITIGNORE_EOF'\n{new_gitignore}\nGITIGNORE_EOF")

print("\n--- Step 3: Check if spread data exists ---")
run(c, f"ls -lh {REMOTE_DIR}/data/download/xauusd-m15-spread* 2>/dev/null || echo 'No spread files'")

print("\n--- Step 4: Force add data files ---")
run(c, f"cd {REMOTE_DIR} && git add -f data/download/xauusd-m15-bid-2015-01-01-2026-04-10.csv")
run(c, f"cd {REMOTE_DIR} && git add -f data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv")
run(c, f"cd {REMOTE_DIR} && git add .gitignore")

print("\n--- Step 5: Check what's staged ---")
run(c, f"cd {REMOTE_DIR} && git status")
run(c, f"cd {REMOTE_DIR} && git diff --cached --stat")

print("\n--- Step 6: Commit and push ---")
run(c, f'cd {REMOTE_DIR} && git commit -m "data: add 11-year XAUUSD M15 + H1 data (2015-2026)"', timeout=30)
run(c, f"cd {REMOTE_DIR} && source /etc/network_turbo 2>/dev/null; git push origin main", timeout=120)

print("\n--- Step 7: Verify ---")
run(c, f"cd {REMOTE_DIR} && git log --oneline -3")

c.close()
print("\n=== DONE ===")
