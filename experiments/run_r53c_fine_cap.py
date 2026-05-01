#!/usr/bin/env python3
"""
R53C — Fine-grained Cap Grid Search ($30-$50, step $1)
=====================================================
Full-sample + Recent (2023-2026) + Yearly Sharpe for top candidates
"""
import sys, os, io, time
import multiprocessing as mp
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "results/r53c_fine_cap"
MAX_WORKERS = 6
FIXED_LOT = 0.05


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def get_base():
    from backtest.runner import LIVE_PARITY_KWARGS
    return {**LIVE_PARITY_KWARGS, 'min_lot_size': FIXED_LOT, 'max_lot_size': FIXED_LOT}


def _run_one(args):
    label, kw, spread, start, end = args
    from backtest.runner import DataBundle, run_variant
    data = DataBundle.load_custom()
    if start and end:
        data = data.slice(start, end)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    return {
        'label': label, 'n': s['n'], 'sharpe': s['sharpe'],
        'total_pnl': s['total_pnl'], 'win_rate': s['win_rate'],
        'max_dd': s['max_dd'], 'maxloss_cap_count': s.get('maxloss_cap_count', 0),
    }


def run_pool(tasks):
    n = min(MAX_WORKERS, len(tasks))
    with mp.Pool(n) as pool:
        return pool.map(_run_one, tasks)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()
    caps = list(range(30, 51))  # 30,31,...,50

    periods = [
        ("Full", None, None),
        ("Recent_2023-2026", "2023-01-01", "2026-05-01"),
    ]

    print(f"\n{'='*70}")
    print(f"R53C: Fine Cap Grid $30-$50 (step $1) x {len(periods)} periods = {len(caps)*len(periods)} variants")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    tasks = []
    for pname, start, end in periods:
        for cap in caps:
            kw = get_base()
            kw['maxloss_cap'] = cap
            tasks.append((f"{pname}_Cap{cap}", kw, 0.30, start, end))

    results = run_pool(tasks)

    with open(f"{OUTPUT_DIR}/fine_cap_grid.txt", 'w', encoding='utf-8') as f:
        f.write(f"R53C: Fine Cap Grid $30-$50 (Lot={FIXED_LOT})\n")
        f.write("=" * 100 + "\n\n")

        for pname, _, _ in periods:
            f.write(f"\n--- {pname} ---\n")
            sub = [r for r in results if r['label'].startswith(pname + "_")]
            sub.sort(key=lambda x: x['sharpe'], reverse=True)
            header = f"{'Cap':>6} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10} {'CapHits':>8} {'CapHit%':>8}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for r in sub:
                cap_pct = (r['maxloss_cap_count'] / r['n'] * 100) if r['n'] > 0 else 0
                cap_val = r['label'].split('Cap')[1]
                f.write(f"  ${cap_val:>4} {r['n']:>6} {r['sharpe']:>8.2f} "
                        f"{fmt(r['total_pnl']):>12} {r['win_rate']:>6.1f}% "
                        f"{fmt(r['max_dd']):>10} {r['maxloss_cap_count']:>8} {cap_pct:>7.1f}%\n")

            best = sub[0]
            cap_val = best['label'].split('Cap')[1]
            f.write(f"\n  >>> Best: Cap${cap_val}  Sharpe={best['sharpe']:.2f}  "
                    f"PnL={fmt(best['total_pnl'])}  CapHit%={(best['maxloss_cap_count']/best['n']*100):.1f}%\n")

    # Phase 2: yearly breakdown for top 5
    full_sub = [r for r in results if r['label'].startswith("Full_")]
    full_sub.sort(key=lambda x: x['sharpe'], reverse=True)
    top5_caps = [int(r['label'].split('Cap')[1]) for r in full_sub[:5]]

    print(f"\nPhase 2: Yearly Sharpe for top 5 caps: {top5_caps}")

    years = [(str(y), f"{y}-01-01", f"{y+1}-01-01") for y in range(2015, 2026)]
    tasks2 = []
    for yr_name, start, end in years:
        for cap in top5_caps:
            kw = get_base()
            kw['maxloss_cap'] = cap
            tasks2.append((f"Y{yr_name}_Cap{cap}", kw, 0.30, start, end))

    results2 = run_pool(tasks2)

    with open(f"{OUTPUT_DIR}/fine_cap_grid.txt", 'a', encoding='utf-8') as f:
        f.write(f"\n\n--- Yearly Sharpe for Top 5 Caps ---\n")
        f.write(f"{'Year':<6}")
        for cap in top5_caps:
            f.write(f" {'Cap'+str(cap):>10}")
        f.write(f" {'Best':>10}\n")
        f.write("-" * (6 + 11 * (len(top5_caps) + 1)) + "\n")

        for yr_name, _, _ in years:
            f.write(f"{yr_name:<6}")
            sharpes = {}
            for cap in top5_caps:
                r = next((x for x in results2 if x['label'] == f"Y{yr_name}_Cap{cap}"), None)
                sh = r['sharpe'] if r else 0
                sharpes[cap] = sh
                f.write(f" {sh:>10.2f}")
            best_cap = max(sharpes, key=sharpes.get)
            f.write(f" {'Cap'+str(best_cap):>10}\n")

        f.write(f"\n{'Year':<6}")
        for cap in top5_caps:
            f.write(f" {'Cap'+str(cap):>10}")
        f.write("\n")
        f.write("-" * (6 + 11 * len(top5_caps)) + "\n")
        win_count = {cap: 0 for cap in top5_caps}
        for yr_name, _, _ in years:
            sharpes = {}
            for cap in top5_caps:
                r = next((x for x in results2 if x['label'] == f"Y{yr_name}_Cap{cap}"), None)
                sharpes[cap] = r['sharpe'] if r else 0
            best_cap = max(sharpes, key=sharpes.get)
            win_count[best_cap] += 1
        f.write(f"{'Wins':<6}")
        for cap in top5_caps:
            f.write(f" {win_count[cap]:>10}")
        f.write("\n")

    total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"R53C Complete — Total: {total:.0f}s ({total/60:.1f}min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
