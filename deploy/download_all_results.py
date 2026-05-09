"""Download all experiment results from remote server to local.
Uses per-folder reconnection to handle unstable connections."""
import paramiko
import os
import time
import stat

SERVER = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASSWD = '3sCdENtzYfse'
REMOTE_BASE = '/root/gold-quant-research/results'
LOCAL_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

FOLDERS_TO_SYNC = [
    'r62_ml_exit_filter',
    'r63_walk_forward',
    'r64_monte_carlo',
    'r65_spread_stress',
    'r66_era_bias',
    'r67_random_entry',
    'r68_h1_fix',
    'r69_validation',
    'r70_ea_validation',
    'r71_oos',
    'r72_generalization',
    'r73_silver_validation',
    'r74_79_comprehensive',
    'r80_macd_validation',
    'r81_sess_bo_validation',
    'r82_dual_pbo',
    'r83_advanced_research',
    'r84_r85',
    'r86_risk_parity',
    'r87_advanced_risk',
    'r88_cap_grid',
    'r89_lot_optimizer',
    'r90_external_data',
    'r92_robustness',
    'r92b_multi_strategy',
]

SINGLE_FILES = [
    'overnight_research_summary.txt',
    'overnight_stdout.txt',
    'r62_stdout.txt',
    'r63_65_stdout.txt',
    'r66_68_stdout.txt',
    'r67_stdout.txt',
    'r69_stdout.txt',
    'r70_stdout.txt',
    'r71_stdout.txt',
    'r72_stdout.txt',
    'r73_stdout.txt',
    'r74_79_stdout.txt',
    'r80_stdout.txt',
    'r81_stdout.txt',
    'r82_stdout.txt',
    'r83_stdout.txt',
    'r84_r85_stdout.txt',
    'r86_stdout.txt',
    'r87_stdout.txt',
    'r88_stdout.txt',
    'r89_stdout.txt',
]

def connect():
    for attempt in range(5):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(SERVER, port=PORT, username=USER, password=PASSWD,
                      timeout=120, banner_timeout=60, auth_timeout=60)
            sftp = c.open_sftp()
            return c, sftp
        except Exception as e:
            print(f"  [RETRY] Attempt {attempt+1} failed: {e}")
            time.sleep(5)
    raise RuntimeError("Failed to connect after 5 attempts")

def download_recursive(sftp, remote_path, local_path, stats):
    try:
        entries = sftp.listdir_attr(remote_path)
    except IOError:
        return
    os.makedirs(local_path, exist_ok=True)
    for entry in entries:
        remote_fp = f"{remote_path}/{entry.filename}"
        local_fp = os.path.join(local_path, entry.filename)
        if stat.S_ISDIR(entry.st_mode):
            download_recursive(sftp, remote_fp, local_fp, stats)
        else:
            if not entry.filename.endswith(('.json', '.txt', '.csv', '.log')):
                continue
            if os.path.exists(local_fp):
                local_size = os.path.getsize(local_fp)
                if local_size == entry.st_size:
                    stats['skipped'] += 1
                    continue
            try:
                sftp.get(remote_fp, local_fp)
                stats['downloaded'] += 1
                print(f"    + {entry.filename} ({entry.st_size:,}B)")
            except Exception as e:
                print(f"    ERR: {entry.filename}: {e}")
                stats['errors'] += 1

def sync_folder(folder, stats):
    """Connect, sync one folder, disconnect."""
    c, sftp = connect()
    remote_path = f"{REMOTE_BASE}/{folder}"
    local_path = os.path.join(LOCAL_BASE, folder)
    try:
        attr = sftp.stat(remote_path)
        if stat.S_ISDIR(attr.st_mode):
            download_recursive(sftp, remote_path, local_path, stats)
        else:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if os.path.exists(local_path) and os.path.getsize(local_path) == attr.st_size:
                stats['skipped'] += 1
            else:
                sftp.get(remote_path, local_path)
                stats['downloaded'] += 1
                print(f"    + {folder} ({attr.st_size:,}B)")
    except IOError:
        print(f"    (not found)")
    finally:
        sftp.close()
        c.close()

def main():
    stats = {'downloaded': 0, 'skipped': 0, 'errors': 0}
    total = len(FOLDERS_TO_SYNC) + len(SINGLE_FILES)

    print("=" * 60)
    print("Downloading experiment results from remote server")
    print("=" * 60)

    for i, folder in enumerate(FOLDERS_TO_SYNC, 1):
        print(f"\n[{i}/{total}] {folder}/")
        try:
            sync_folder(folder, stats)
        except Exception as e:
            print(f"    FAILED: {e}")
            stats['errors'] += 1
        time.sleep(1)

    for i, fname in enumerate(SINGLE_FILES, len(FOLDERS_TO_SYNC) + 1):
        print(f"\n[{i}/{total}] {fname}")
        try:
            sync_folder(fname, stats)
        except Exception as e:
            print(f"    FAILED: {e}")
            stats['errors'] += 1
        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"DONE  Downloaded: {stats['downloaded']}  Skipped: {stats['skipped']}  Errors: {stats['errors']}")

if __name__ == '__main__':
    main()
