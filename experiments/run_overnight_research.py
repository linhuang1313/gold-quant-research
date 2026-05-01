#!/usr/bin/env python3
"""
10-Hour Overnight Research Orchestrator
========================================
Runs R57-R62 in dependency-ordered batches:
  Batch 1 (parallel): R57, R58, R59        ~2h
  Batch 2 (parallel): R60, R61             ~3h
  Batch 3 (sequential): R62                ~2h

Each task writes a _done.flag on completion.
Sends Telegram notification when all tasks finish.
"""
import sys, os, io, time, subprocess, json, traceback
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
RESEARCH_ROOT = ROOT.parent
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(RESEARCH_ROOT.parent))

SCRIPTS = {
    'R57': ROOT / 'run_r57_cooldown_gap.py',
    'R58': ROOT / 'run_r58_sl_tp_cap37.py',
    'R59': ROOT / 'run_r59_session_weight.py',
    'R60': ROOT / 'run_r60_mean_rev_validate.py',
    'R61': ROOT / 'run_r61_6strat_portfolio.py',
    'R62': ROOT / 'run_r62_ml_exit_filter.py',
}

RESULT_DIRS = {
    'R57': RESEARCH_ROOT / 'results' / 'r57_cooldown_gap',
    'R58': RESEARCH_ROOT / 'results' / 'r58_sl_tp_cap37',
    'R59': RESEARCH_ROOT / 'results' / 'r59_session_weight',
    'R60': RESEARCH_ROOT / 'results' / 'r60_mean_rev_validate',
    'R61': RESEARCH_ROOT / 'results' / 'r61_6strat_portfolio',
    'R62': RESEARCH_ROOT / 'results' / 'r62_ml_exit_filter',
}

BATCHES = [
    {'name': 'Batch 1', 'tasks': ['R57', 'R58', 'R59'], 'parallel': False},
    {'name': 'Batch 2', 'tasks': ['R60', 'R61'], 'parallel': False},
    {'name': 'Batch 3', 'tasks': ['R62'], 'parallel': False},
]


def send_telegram(msg):
    try:
        from notifier import send_telegram_long
        send_telegram_long(msg)
    except Exception as e:
        print(f"[Telegram] Failed: {e}")


def flag_path(task_id):
    return RESULT_DIRS[task_id] / '_done.flag'


def is_done(task_id):
    return flag_path(task_id).exists()


def write_flag(task_id, elapsed, success, error_msg=""):
    fp = flag_path(task_id)
    fp.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'task': task_id,
        'completed': datetime.now().isoformat(),
        'elapsed_s': round(elapsed, 1),
        'success': success,
        'error': error_msg,
    }
    fp.write_text(json.dumps(data, indent=2), encoding='utf-8')


def run_task(task_id):
    """Run a single research task, return (task_id, elapsed, success, error)."""
    script = SCRIPTS[task_id]
    print(f"\n{'='*60}")
    print(f"  Starting {task_id}: {script.name}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}", flush=True)

    if is_done(task_id):
        print(f"  {task_id} already done (flag exists), skipping.")
        return task_id, 0, True, "skipped"

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
        error_msg = "" if success else f"exit_code={result.returncode}"
        write_flag(task_id, elapsed, success, error_msg)
        status = "OK" if success else f"FAILED (code {result.returncode})"
        print(f"\n  {task_id} {status} in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
        return task_id, elapsed, success, error_msg
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        write_flag(task_id, elapsed, False, "timeout")
        print(f"\n  {task_id} TIMEOUT after {elapsed:.0f}s", flush=True)
        return task_id, elapsed, False, "timeout"
    except Exception as e:
        elapsed = time.time() - t0
        error_msg = traceback.format_exc()
        write_flag(task_id, elapsed, False, str(e))
        print(f"\n  {task_id} ERROR: {e}", flush=True)
        return task_id, elapsed, False, str(e)


def run_batch_parallel(task_ids):
    """Run tasks in parallel using subprocess (each script manages its own multiprocessing)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = {}
    with ThreadPoolExecutor(max_workers=len(task_ids)) as executor:
        futures = {executor.submit(run_task, tid): tid for tid in task_ids}
        for future in as_completed(futures):
            tid = futures[future]
            try:
                results[tid] = future.result()
            except Exception as e:
                results[tid] = (tid, 0, False, str(e))
    return results


def run_batch_sequential(task_ids):
    """Run tasks sequentially."""
    results = {}
    for tid in task_ids:
        results[tid] = run_task(tid)
    return results


def main():
    total_start = time.time()
    print(f"\n{'#'*70}")
    print(f"  10-Hour Overnight Research Orchestrator")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Batches: {len(BATCHES)}")
    print(f"  Tasks: {', '.join(SCRIPTS.keys())}")
    print(f"{'#'*70}\n", flush=True)

    send_telegram(
        f"🔬 <b>过夜研究启动</b>\n\n"
        f"任务: R57-R62 (6个研究)\n"
        f"开始: {datetime.now().strftime('%H:%M')}\n"
        f"预计: ~10小时"
    )

    all_results = {}

    for batch in BATCHES:
        batch_name = batch['name']
        task_ids = batch['tasks']
        parallel = batch['parallel']

        print(f"\n{'*'*70}")
        print(f"  {batch_name}: {', '.join(task_ids)} ({'parallel' if parallel else 'sequential'})")
        print(f"{'*'*70}", flush=True)

        bt0 = time.time()
        if parallel and len(task_ids) > 1:
            batch_results = run_batch_parallel(task_ids)
        else:
            batch_results = run_batch_sequential(task_ids)
        all_results.update(batch_results)
        bt_elapsed = time.time() - bt0

        ok = sum(1 for _, _, s, _ in batch_results.values() if s)
        fail = len(batch_results) - ok
        print(f"\n  {batch_name} done in {bt_elapsed:.0f}s — {ok} OK, {fail} failed", flush=True)

        send_telegram(
            f"📊 <b>{batch_name} 完成</b>\n\n"
            + "\n".join(
                f"{'✅' if s else '❌'} {tid}: {e:.0f}s" + (f" ({err})" if not s else "")
                for tid, (_, e, s, err) in batch_results.items()
            )
        )

    total_elapsed = time.time() - total_start
    total_min = total_elapsed / 60
    total_hr = total_elapsed / 3600

    ok_tasks = [tid for tid, (_, _, s, _) in all_results.items() if s]
    fail_tasks = [tid for tid, (_, _, s, _) in all_results.items() if not s]

    summary_lines = [
        f"🏁 <b>过夜研究全部完成</b>",
        f"",
        f"⏱ 总耗时: {total_hr:.1f}h ({total_min:.0f}min)",
        f"✅ 成功: {len(ok_tasks)} — {', '.join(ok_tasks) if ok_tasks else 'none'}",
    ]
    if fail_tasks:
        summary_lines.append(f"❌ 失败: {len(fail_tasks)} — {', '.join(fail_tasks)}")

    summary_lines.append(f"\n<b>各任务耗时:</b>")
    for tid in SCRIPTS:
        if tid in all_results:
            _, elapsed, success, err = all_results[tid]
            status = "✅" if success else "❌"
            summary_lines.append(f"  {status} {tid}: {elapsed/60:.1f}min")

    send_telegram("\n".join(summary_lines))

    result_file = RESEARCH_ROOT / 'results' / 'overnight_research_summary.txt'
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"10-Hour Overnight Research Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Started: {datetime.fromtimestamp(total_start).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total: {total_hr:.1f}h ({total_min:.0f}min)\n\n")
        for tid in SCRIPTS:
            if tid in all_results:
                _, elapsed, success, err = all_results[tid]
                status = "OK" if success else f"FAILED ({err})"
                f.write(f"  {tid}: {elapsed/60:.1f}min — {status}\n")

    print(f"\n{'#'*70}")
    print(f"  ALL DONE — {total_hr:.1f}h total")
    print(f"  Success: {len(ok_tasks)}/{len(all_results)}")
    print(f"  Summary: {result_file}")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
