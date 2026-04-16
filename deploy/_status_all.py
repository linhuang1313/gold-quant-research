"""Check status of ALL experiments on remote server."""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
PROJECT = "/root/gold-quant-trading"

ALL_OUTPUTS = [
    # Old batch
    ("run_trail_grid_validation.py", "trail_grid_output.txt", "Trail Grid"),
    ("run_exp_batch_postfix.py", "exp_batch_postfix_output.txt", "EXP-A~D Batch"),
    ("run_exp_choppy_ablation.py", "exp_choppy_ablation_output.txt", "EXP-G Choppy"),
    ("run_exp_sl_sensitivity.py", "exp_sl_sensitivity_output.txt", "EXP-H SL/TP"),
    ("run_exp_spread_model.py", "exp_spread_model_output.txt", "EXP-I Spread"),
    ("run_exp_k_regime_bounds.py", "exp_k_regime_output.txt", "EXP-K Regime"),
    ("run_exp_l_trend_weights.py", "exp_l_trend_weights_output.txt", "EXP-L Weights"),
    ("run_exp_m_slippage.py", "exp_m_slippage_output.txt", "EXP-M Slippage"),
    # New batch
    ("run_exp_r_baseline_update.py", "exp_r_baseline_output.txt", "EXP-R Baseline"),
    ("run_exp_s_historical_spread.py", "exp_s_spread_output.txt", "EXP-S HistSpread"),
    ("run_exp_t_donchian.py", "exp_t_donchian_output.txt", "EXP-T Donchian"),
    ("run_exp_u_kc_reentry.py", "exp_u_kc_reentry_output.txt", "EXP-U KC Reentry"),
    ("run_exp_v_breakout_sizing.py", "exp_v_sizing_output.txt", "EXP-V Sizing"),
    ("run_exp_w_loss_profile.py", "exp_w_loss_profile_output.txt", "EXP-W LossProfile"),
]

def run_cmd(ssh, cmd, timeout=15):
    try:
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        return stdout.read().decode('utf-8', errors='replace').strip(), stderr.read().decode('utf-8', errors='replace').strip()
    except:
        return "", ""

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

    # Server status
    out, _ = run_cmd(ssh, "uptime")
    print("Server: " + out)
    out, _ = run_cmd(ssh, "nproc")
    print("Cores: " + out)

    # Running processes
    out, _ = run_cmd(ssh, "ps aux | grep 'python.*run_exp\\|python.*run_trail' | grep -v grep | awk '{print $2, $11, $12}'")
    running_scripts = set()
    running_pids = {}
    if out:
        for line in out.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 3:
                for s, _, _ in ALL_OUTPUTS:
                    if s in line:
                        running_scripts.add(s)
                        running_pids[s] = parts[0]

    print("\n" + "=" * 90)
    print(f"{'Experiment':<20} {'Script':<35} {'Status':<12} {'Size':<10} {'Last Line'}")
    print("=" * 90)

    completed = []
    running = []
    not_started = []

    for script, output, label in ALL_OUTPUTS:
        # Check if output file exists and its size
        out, _ = run_cmd(ssh, f"wc -c < {PROJECT}/{output} 2>/dev/null")
        size = int(out.strip()) if out.strip().isdigit() else 0
        
        is_running = script in running_scripts
        
        # Get last meaningful line
        out, _ = run_cmd(ssh, f"tail -5 {PROJECT}/{output} 2>/dev/null | grep -v '^$' | tail -1")
        last_line = out.strip()[:60] if out.strip() else "(empty)"
        
        if is_running:
            status = "RUNNING"
            running.append(label)
            size_str = f"{size//1024}KB"
        elif size > 100:
            # Check if it finished (look for common end markers)
            out2, _ = run_cmd(ssh, f"tail -20 {PROJECT}/{output} 2>/dev/null | grep -ic 'summary\\|done\\|complete\\|error\\|traceback'")
            has_end = int(out2.strip()) if out2.strip().isdigit() else 0
            out3, _ = run_cmd(ssh, f"tail -20 {PROJECT}/{output} 2>/dev/null | grep -ic 'traceback\\|error'")
            has_error = int(out3.strip()) if out3.strip().isdigit() else 0
            
            if has_error > 0:
                status = "CRASHED"
            elif has_end > 0:
                status = "DONE"
                completed.append(label)
            else:
                status = "DONE?"
                completed.append(label)
            size_str = f"{size//1024}KB"
        else:
            status = "NOT STARTED"
            not_started.append(label)
            size_str = "-"
        
        print(f"{label:<20} {script:<35} {status:<12} {size_str:<10} {last_line}")

    print("\n" + "=" * 90)
    print(f"COMPLETED: {len(completed)}  |  RUNNING: {len(running)}  |  NOT STARTED: {len(not_started)}")
    print("=" * 90)
    
    if completed:
        print("\nCompleted: " + ", ".join(completed))
    if running:
        print("Running:   " + ", ".join(running))
    if not_started:
        print("Waiting:   " + ", ".join(not_started))

    ssh.close()

if __name__ == "__main__":
    main()
