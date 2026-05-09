#!/usr/bin/env python3
"""
Deploy R170-R174 to remote server and launch sequentially.

R170: ML Exit V3 Walk-Forward Pipeline (~30-60min)
R171: P6 Portfolio Monte Carlo Stress Test (~2-4h)
R172: Regime-Conditional Weight Optimization (~4-6h)
R173: M1 Feature Enhancement for H1 Entry (~6-10h)
R174: Alpha Decay Monitor (~2h)

Total estimated runtime: ~15-24h on 208-core server.
R170/R174 are fastest — launch first. R173 is heaviest — launch last.
"""
import paramiko, time, os, sys
from pathlib import Path

HOST = "connect.westd.seetacloud.com"
PORT = 41109
USER = "root"
PASS = "3sCdENtzYfse"

REMOTE_BASE = "/root/gold-quant-research"

SCRIPTS = [
    "experiments/run_r170_ml_exit_v3.py",
    "experiments/run_r171_p6_monte_carlo.py",
    "experiments/run_r172_regime_weight_optimizer.py",
    "experiments/run_r173_m1_feature_enhancement.py",
    "experiments/run_r174_alpha_decay_monitor.py",
]

SHARED_FILES = [
    "backtest/runner.py",
    "backtest/engine.py",
    "backtest/stats.py",
    "backtest/__init__.py",
    "research_config.py",
    "indicators.py",
]

RESULT_DIRS = [
    "results/r170_ml_exit_v3",
    "results/r171_p6_monte_carlo",
    "results/r172_regime_weight_optimizer",
    "results/r173_m1_feature_enhancement",
    "results/r174_alpha_decay_monitor",
    "results/r174_alpha_decay_monitor/snapshots",
]

EXPERIMENTS = [
    ("R170", "run_r170_ml_exit_v3.py",            "r170_ml_exit_v3/r170_stdout.txt"),
    ("R171", "run_r171_p6_monte_carlo.py",         "r171_p6_monte_carlo/r171_stdout.txt"),
    ("R172", "run_r172_regime_weight_optimizer.py", "r172_regime_weight_optimizer/r172_stdout.txt"),
    ("R173", "run_r173_m1_feature_enhancement.py",  "r173_m1_feature_enhancement/r173_stdout.txt"),
    ("R174", "run_r174_alpha_decay_monitor.py",     "r174_alpha_decay_monitor/r174_stdout.txt"),
]


def connect():
    for attempt in range(5):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(HOST, port=PORT, username=USER, password=PASS,
                      timeout=30, banner_timeout=30)
            return c
        except Exception as e:
            print(f"  Connect attempt {attempt+1} failed: {e}")
            time.sleep(3)
    raise RuntimeError("Cannot connect after 5 attempts")


def run(ssh, cmd, timeout=30):
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    return o.read().decode(), e.read().decode()


def upload_file(local_path, remote_path):
    ssh = connect()
    sftp = ssh.open_sftp()
    try:
        remote_dir = os.path.dirname(remote_path)
        try:
            sftp.stat(remote_dir)
        except FileNotFoundError:
            ssh2 = connect()
            run(ssh2, f"mkdir -p {remote_dir}")
            ssh2.close()
        sftp.put(str(local_path), remote_path)
    finally:
        sftp.close()
        ssh.close()


def main():
    base = Path(__file__).parent.parent

    print("=" * 70)
    print("  R170-R174 Deployment to Remote Server")
    print(f"  Server: {HOST}:{PORT}")
    print("=" * 70)

    # Step 1: Verify local files
    print("\n  Step 1: Verifying local files...")
    missing = []
    for script in SCRIPTS + SHARED_FILES:
        local_path = base / script
        if not local_path.exists():
            missing.append(script)
            print(f"    MISSING: {script}")
        else:
            print(f"    OK: {script}")
    if missing:
        print(f"\n  ERROR: {len(missing)} files missing. Aborting.")
        return

    # Step 2: Upload shared infrastructure
    print("\n  Step 2: Uploading shared infrastructure...")
    for f in SHARED_FILES:
        local_path = base / f
        remote_path = f"{REMOTE_BASE}/{f}"
        print(f"    Uploading {f}...", end=" ", flush=True)
        upload_file(local_path, remote_path)
        print("OK")

    # Step 3: Upload experiment scripts
    print("\n  Step 3: Uploading experiment scripts...")
    for script in SCRIPTS:
        local_path = base / script
        remote_path = f"{REMOTE_BASE}/{script}"
        print(f"    Uploading {script}...", end=" ", flush=True)
        upload_file(local_path, remote_path)
        print("OK")

    # Step 4: Create result directories
    print("\n  Step 4: Creating result directories...")
    ssh = connect()
    for d in RESULT_DIRS:
        run(ssh, f"mkdir -p {REMOTE_BASE}/{d}")
    ssh.close()
    print("    Done.")

    # Step 5: Install Python dependencies
    print("\n  Step 5: Checking Python dependencies...")
    ssh = connect()
    out, _ = run(ssh, "pip3 list 2>/dev/null | grep -i 'xgboost\\|lightgbm\\|scikit-learn'", timeout=15)
    print(f"    Current ML packages:\n{out.strip()}" if out.strip() else "    No ML packages found")
    if 'xgboost' not in out.lower():
        print("    Installing xgboost + lightgbm + scikit-learn...")
        run(ssh, "pip3 install xgboost lightgbm scikit-learn --quiet", timeout=300)
        print("    Done.")
    ssh.close()

    # Step 6: Launch experiments
    print(f"\n  Step 6: Launching experiments...")
    print("  Strategy: launch all 5 concurrently (they use different resources)")
    print()

    for name, script, stdout_file in EXPERIMENTS:
        print(f"  Launching {name} ({script})...")
        ssh = connect()
        cmd = (
            f"cd {REMOTE_BASE} && "
            f"nohup python3 experiments/{script} "
            f"> results/{stdout_file} 2>&1 &"
        )
        ssh.exec_command(cmd)
        time.sleep(2)

        out, _ = run(ssh, f"ps aux | grep {script} | grep -v grep")
        if out.strip():
            pid = out.strip().split()[1]
            print(f"    {name} running (PID: {pid})")
        else:
            print(f"    WARNING: {name} process not found, checking output...")
            out, _ = run(ssh, f"tail -5 {REMOTE_BASE}/results/{stdout_file} 2>/dev/null")
            print(f"    {out.strip()[:200]}")
        ssh.close()
        time.sleep(1)

    # Step 7: Status summary
    print(f"\n{'=' * 70}")
    print("  All experiments launched. Status:")
    print("=" * 70)
    time.sleep(3)

    ssh = connect()
    out, _ = run(ssh, "ps aux | grep 'run_r17[0-4]' | grep -v grep")
    running = 0
    for line in out.strip().split('\n'):
        if line.strip():
            parts = line.split()
            pid = parts[1]
            cmd_part = ' '.join(parts[10:])
            print(f"  PID {pid}: {cmd_part[:80]}")
            running += 1

    print(f"\n  {running}/5 experiments running")
    print(f"\n  Monitor commands:")
    for name, _, stdout_file in EXPERIMENTS:
        print(f"    {name}: tail -f {REMOTE_BASE}/results/{stdout_file}")

    print("\n  Quick check all:")
    check_cmd = (
        "for f in r170 r171 r172 r173 r174; do "
        "echo \"=== $f ===\"; "
        f"tail -3 {REMOTE_BASE}/results/${{f}}_*/r${{f#r}}_stdout.txt 2>/dev/null; "
        "done"
    )
    print(f"    {check_cmd}")

    print(f"\n  Estimated completion times:")
    print(f"    R170 (ML Exit V3):      ~30-60 min")
    print(f"    R171 (MC Stress Test):  ~2-4 hours")
    print(f"    R172 (Regime Weights):  ~4-6 hours")
    print(f"    R173 (M1 Features):     ~6-10 hours")
    print(f"    R174 (Alpha Monitor):   ~2 hours")
    ssh.close()


if __name__ == '__main__':
    main()
