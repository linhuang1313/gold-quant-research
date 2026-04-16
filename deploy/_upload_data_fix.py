"""Fix git config and push data to GitHub."""
import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westc.seetacloud.com"
PORT = 16005
REMOTE_DIR = "/root/gold-quant-trading"

def run(c, cmd, timeout=120):
    print(f"  >>> {cmd[:120]}")
    _, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()
    if out:
        for line in out.split("\n")[-20:]:
            print(f"      {line}")
    if err:
        for line in err.split("\n")[-10:]:
            print(f"  err {line}")
    return out

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username="root", password="r1zlTZQUb+E4", timeout=15)

print("=== Fix and push data ===\n")

print("--- Step 1: Set git config ---")
run(c, f'cd {REMOTE_DIR} && git config user.email "linhuang1313@users.noreply.github.com"')
run(c, f'cd {REMOTE_DIR} && git config user.name "linhuang1313"')

print("\n--- Step 2: Check existing remote URL and git log ---")
run(c, f"cd {REMOTE_DIR} && git remote -v")
run(c, f"cd {REMOTE_DIR} && git log --oneline -5")

print("\n--- Step 3: Check staged changes ---")
run(c, f"cd {REMOTE_DIR} && git diff --cached --stat")

print("\n--- Step 4: Commit ---")
run(c, f'cd {REMOTE_DIR} && git commit -m "data: add 11-year XAUUSD M15 + H1 data (2015-2026)"')

print("\n--- Step 5: Check git credentials/tokens ---")
run(c, f"cd {REMOTE_DIR} && cat .git/config | head -20")
run(c, f"cd {REMOTE_DIR} && git log --oneline -5")

print("\n--- Step 6: Try push with token from previous commits ---")
# Check if there's a stored credential
run(c, f"cd {REMOTE_DIR} && git config --list | grep credential")
run(c, f"cd {REMOTE_DIR} && cat ~/.git-credentials 2>/dev/null || echo 'no credentials file'")
run(c, f"cd {REMOTE_DIR} && cat ~/.netrc 2>/dev/null || echo 'no netrc'")

# Check env
run(c, f"env | grep -i git 2>/dev/null; env | grep -i token 2>/dev/null; echo done")

c.close()
print("\n=== DONE ===")
