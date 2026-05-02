#!/usr/bin/env python3
"""
R90 — Full External Data Integration Test (Orchestrator)
==========================================================
Runs all 5 phases sequentially with checkpoint/resume support.
Each phase saves results to disk; if a checkpoint exists, that phase is skipped.

Phase A: Macro Regime Detection          (~4h)   [parallel with C]
Phase B: Factor-Enhanced Signal Filtering (~6h)   [after A]
Phase C: ML Direction Prediction          (~16h)  [parallel with A]
Phase D: ML Exit Optimization             (~8h)   [after A+C]
Phase E: Dynamic Portfolio Allocation     (~6h)   [after A+D]

Total estimated: 32-40h on 25-core / 90GB / RTX 5090 server.
"""
import sys, os, io, time, json, subprocess, signal
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
RESULTS_DIR = BASE_DIR / "results" / "r90_external_data"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PHASES = [
    {
        "id": "A",
        "name": "Macro Regime Detection",
        "script": "experiments/run_r90a_regime.py",
        "checkpoint": RESULTS_DIR / "r90a_regime" / "r90a_results.json",
        "depends_on": [],
    },
    {
        "id": "C",
        "name": "ML Direction Prediction",
        "script": "experiments/run_r90c_ml_direction.py",
        "checkpoint": RESULTS_DIR / "r90c_ml_direction" / "r90c_results.json",
        "depends_on": [],
    },
    {
        "id": "B",
        "name": "Factor-Enhanced Signal Filtering",
        "script": "experiments/run_r90b_factor_filter.py",
        "checkpoint": RESULTS_DIR / "r90b_factor_filter" / "r90b_results.json",
        "depends_on": ["A"],
    },
    {
        "id": "D",
        "name": "ML Exit Optimization",
        "script": "experiments/run_r90d_ml_exit.py",
        "checkpoint": RESULTS_DIR / "r90d_ml_exit" / "r90d_results.json",
        "depends_on": ["A", "C"],
    },
    {
        "id": "E",
        "name": "Dynamic Portfolio Allocation",
        "script": "experiments/run_r90e_portfolio.py",
        "checkpoint": RESULTS_DIR / "r90e_portfolio" / "r90e_results.json",
        "depends_on": ["A", "D"],
    },
]


def run_phase(phase, python_cmd="python3"):
    """Run a single phase as a subprocess."""
    pid = phase["id"]
    name = phase["name"]
    script = str(BASE_DIR / phase["script"])
    checkpoint = phase["checkpoint"]

    print(f"\n{'='*80}", flush=True)
    print(f"  Phase {pid}: {name}", flush=True)
    print(f"{'='*80}", flush=True)

    if checkpoint.exists():
        size_kb = checkpoint.stat().st_size / 1024
        print(f"  SKIP: checkpoint exists ({checkpoint.name}, {size_kb:.1f} KB)", flush=True)
        print(f"  To re-run, delete: {checkpoint}", flush=True)
        return {"phase": pid, "status": "skipped", "elapsed_s": 0}

    print(f"  Script: {script}", flush=True)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  {'─'*60}", flush=True)

    t0 = time.time()
    try:
        result = subprocess.run(
            [python_cmd, "-u", script],
            cwd=str(BASE_DIR),
            timeout=86400,  # 24h max per phase
            capture_output=False,
        )
        elapsed = time.time() - t0
        status = "success" if result.returncode == 0 else f"failed (exit {result.returncode})"
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        status = "timeout"
    except Exception as e:
        elapsed = time.time() - t0
        status = f"error: {e}"

    print(f"\n  {'─'*60}", flush=True)
    print(f"  Phase {pid} {status} in {elapsed/3600:.2f}h ({elapsed:.0f}s)", flush=True)

    if checkpoint.exists():
        print(f"  Checkpoint saved: {checkpoint}", flush=True)
    else:
        print(f"  WARNING: checkpoint NOT found after run!", flush=True)

    return {"phase": pid, "status": status, "elapsed_s": round(elapsed, 1)}


def run_parallel_phases(phases, python_cmd="python3"):
    """Run independent phases in parallel using subprocesses."""
    results = {}
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(run_phase, p, python_cmd): p["id"] for p in phases}
        for future in as_completed(futures):
            pid = futures[future]
            try:
                results[pid] = future.result()
            except Exception as e:
                results[pid] = {"phase": pid, "status": f"error: {e}", "elapsed_s": 0}
    return results


def main():
    print("=" * 80, flush=True)
    print("  R90: Full External Data Integration Test", flush=True)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 80, flush=True)

    # Detect python command
    python_cmd = sys.executable or "python3"
    print(f"  Python: {python_cmd}", flush=True)
    print(f"  Results: {RESULTS_DIR}", flush=True)

    t0_total = time.time()
    phase_results = {}

    # ── Step 1: Run A + C in parallel (no dependencies) ──
    print(f"\n{'#'*80}", flush=True)
    print(f"  STEP 1: Phase A + Phase C (parallel)", flush=True)
    print(f"{'#'*80}", flush=True)

    parallel_phases = [p for p in PHASES if p["id"] in ("A", "C")]
    already_done = all(p["checkpoint"].exists() for p in parallel_phases)

    if already_done:
        for p in parallel_phases:
            phase_results[p["id"]] = {"phase": p["id"], "status": "skipped", "elapsed_s": 0}
            print(f"  Phase {p['id']}: SKIPPED (checkpoint exists)", flush=True)
    else:
        # Run sequentially if parallel causes issues (safer on shared GPU)
        for p in parallel_phases:
            result = run_phase(p, python_cmd)
            phase_results[p["id"]] = result

    # ── Step 2: Phase B (depends on A) ──
    print(f"\n{'#'*80}", flush=True)
    print(f"  STEP 2: Phase B (after A)", flush=True)
    print(f"{'#'*80}", flush=True)

    phase_b = next(p for p in PHASES if p["id"] == "B")
    phase_results["B"] = run_phase(phase_b, python_cmd)

    # ── Step 3: Phase D (depends on A + C) ──
    print(f"\n{'#'*80}", flush=True)
    print(f"  STEP 3: Phase D (after A + C)", flush=True)
    print(f"{'#'*80}", flush=True)

    phase_d = next(p for p in PHASES if p["id"] == "D")
    phase_results["D"] = run_phase(phase_d, python_cmd)

    # ── Step 4: Phase E (depends on A + D) ──
    print(f"\n{'#'*80}", flush=True)
    print(f"  STEP 4: Phase E (after A + D)", flush=True)
    print(f"{'#'*80}", flush=True)

    phase_e = next(p for p in PHASES if p["id"] == "E")
    phase_results["E"] = run_phase(phase_e, python_cmd)

    # ── Final Summary ──
    total_elapsed = time.time() - t0_total
    print(f"\n{'='*80}", flush=True)
    print(f"  R90 COMPLETE", flush=True)
    print(f"  Total time: {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s)", flush=True)
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'='*80}", flush=True)

    print(f"\n  Phase Results:", flush=True)
    for pid in ["A", "C", "B", "D", "E"]:
        r = phase_results.get(pid, {})
        status = r.get("status", "unknown")
        elapsed = r.get("elapsed_s", 0)
        hours = elapsed / 3600
        print(f"    Phase {pid}: {status:20s} ({hours:.2f}h)", flush=True)

    # Collect per-phase summaries
    summary = {
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_h": round(total_elapsed / 3600, 2),
        "phases": {},
    }

    for pid in ["A", "B", "C", "D", "E"]:
        phase_info = next(p for p in PHASES if p["id"] == pid)
        checkpoint = phase_info["checkpoint"]
        phase_summary = {"status": phase_results.get(pid, {}).get("status", "unknown"),
                         "elapsed_h": round(phase_results.get(pid, {}).get("elapsed_s", 0) / 3600, 2)}
        if checkpoint.exists():
            try:
                with open(checkpoint) as f:
                    data = json.load(f)
                # Extract key metrics from each phase
                if pid == "A":
                    phase_summary["best_method"] = data.get("best_method", "unknown")
                    phase_summary["regime_count"] = data.get("regime_stats", {}).get("n_regimes", 0)
                elif pid == "B":
                    top = data.get("top_filters", [])
                    if top:
                        phase_summary["best_filter"] = top[0].get("label", "unknown")
                        phase_summary["best_improvement"] = top[0].get("sharpe_improvement_pct", 0)
                elif pid == "C":
                    phase_summary["accuracy_1d"] = data.get("aggregate", {}).get("accuracy_1d", 0)
                    phase_summary["sharpe_signal"] = data.get("signal_backtest", {}).get("sharpe", 0)
                elif pid == "D":
                    phase_summary["best_auc"] = data.get("best_overall_auc", 0)
                elif pid == "E":
                    rec = data.get("final_recommendation", {})
                    phase_summary["dynamic_sharpe"] = rec.get("dynamic_sharpe", 0)
                    phase_summary["static_sharpe"] = rec.get("static_sharpe", 0)
            except Exception:
                pass
        summary["phases"][pid] = phase_summary

    summary_path = RESULTS_DIR / "r90_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
