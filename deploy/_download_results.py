"""Download completed results from remote server."""
import paramiko, os, json, base64

SERVER = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASSWD = '3sCdENtzYfse'
REMOTE_BASE = '/root/gold-quant-research'
LOCAL_BASE = r'c:\Users\hlin2\gold-quant-research'

ALL = ['r132','r133','r134','r135','r136','r137','r138','r139','r140','r141','r142','r143']

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(SERVER, port=PORT, username=USER, password=PASSWD, timeout=60, banner_timeout=60)
sftp = c.open_sftp()

for tag in ALL:
    remote_json = f"{REMOTE_BASE}/results/{tag}/{tag}_results.json"
    local_dir = os.path.join(LOCAL_BASE, 'results', tag)
    os.makedirs(local_dir, exist_ok=True)
    local_json = os.path.join(local_dir, f"{tag}_results.json")

    try:
        sftp.stat(remote_json)
        sftp.get(remote_json, local_json)
        print(f"  {tag}: DOWNLOADED results")
    except FileNotFoundError:
        # Check if stdout exists (still running or crashed)
        try:
            remote_stdout = f"{REMOTE_BASE}/results/{tag}/{tag}_stdout.txt"
            sftp.stat(remote_stdout)
            print(f"  {tag}: still running (no results yet)")
        except FileNotFoundError:
            print(f"  {tag}: not started")

sftp.close()
c.close()
