"""Download all completed experiment outputs from remote server."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
PROJECT = "/root/gold-quant-trading"

FILES_TO_DOWNLOAD = [
    "exp_batch_postfix_output.txt",
    "exp_choppy_ablation_output.txt",
    "exp_sl_sensitivity_output.txt",
    "exp_spread_model_output.txt",
    "exp_m_slippage_output.txt",
    "exp_r_baseline_output.txt",
    "exp_s_spread_output.txt",
    "exp_t_donchian_output.txt",
    "exp_u_kc_reentry_output.txt",
    "exp_v_sizing_output.txt",
    "exp_w_loss_profile_output.txt",
    "exp_l_trend_weights_output.txt",
]

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
    sftp = ssh.open_sftp()
    
    for f in FILES_TO_DOWNLOAD:
        remote = f"{PROJECT}/{f}"
        local = f
        try:
            sftp.stat(remote)
            sftp.get(remote, local)
            import os
            size = os.path.getsize(local)
            print(f"  Downloaded {f} ({size//1024}KB)")
        except FileNotFoundError:
            print(f"  SKIP {f} (not found)")
        except Exception as e:
            print(f"  ERROR {f}: {e}")
    
    sftp.close()
    ssh.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
