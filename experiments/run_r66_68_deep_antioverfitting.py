#!/usr/bin/env python3
"""
R66-R68 Deep Anti-Overfitting Test Suite (Serial)
==================================================
  R66: Era Bias Diagnostic + Direction Bias (~5min)
  R67: Random Entry Benchmark x1000 (~15min)
  R68: H1 Lookahead Fix Impact (~3min)
Total: ~25 min
"""
import sys, os, io, time, subprocess
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
RESEARCH_ROOT = ROOT.parent

SCRIPTS = [
    ('R66', ROOT / 'run_r66_era_bias.py'),
    ('R67', ROOT / 'run_r67_random_entry.py'),
    ('R68', ROOT / 'run_r68_h1_fix_impact.py'),
]

def main():
    total_start = time.time()
    print(f"\n{'#'*70}")
    print(f"  Deep Anti-Overfitting Test Suite (R66-R68)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n", flush=True)

    results = {}
    for task_id, script in SCRIPTS:
        print(f"\n{'='*60}")
        print(f"  Starting {task_id}: {script.name}")
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(RESEARCH_ROOT),
                capture_output=False,
                timeout=7200,
            )
            elapsed = time.time() - t0
            success = result.returncode == 0
            print(f"\n  {task_id} {'OK' if success else 'FAILED'} in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
            results[task_id] = (elapsed, success, "" if success else f"exit={result.returncode}")
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"\n  {task_id} TIMEOUT after {elapsed:.0f}s", flush=True)
            results[task_id] = (elapsed, False, "timeout")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n  {task_id} ERROR: {e}", flush=True)
            results[task_id] = (elapsed, False, str(e))

    total_elapsed = time.time() - total_start
    print(f"\n{'#'*70}")
    print(f"  ALL DONE — {total_elapsed/60:.1f}min total")
    for tid, (e, s, err) in results.items():
        status = "OK" if s else f"FAILED ({err})"
        print(f"    {tid}: {e/60:.1f}min — {status}")
    print(f"{'#'*70}")

if __name__ == "__main__":
    main()
