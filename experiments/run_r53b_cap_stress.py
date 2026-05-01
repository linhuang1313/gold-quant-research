#!/usr/bin/env python3
"""
R53B — MaxLoss Cap Stress Test
================================
Test 1: Sub-sample Cap comparison (early/mid/recent)
Test 2: Yearly Sharpe heatmap (Cap x Year)
Test 3: Dynamic ATR Cap vs Fixed Cap
Test 4: False-kill analysis (Cap$30 truncated trades' no-cap PnL)
"""
import sys, os, io, time, json
import multiprocessing as mp
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "results/r53b_stress"
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
        'elapsed_s': s.get('elapsed_s', 0),
    }


def _run_one_trades(args):
    label, kw, spread, start, end = args
    from backtest.runner import DataBundle, run_variant
    data = DataBundle.load_custom()
    if start and end:
        data = data.slice(start, end)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    trades = s.get('_trades', [])
    trade_list = []
    for t in trades:
        trade_list.append({
            'entry_time': str(t.entry_time)[:19],
            'direction': t.direction,
            'entry_price': round(t.entry_price, 2),
            'exit_price': round(t.exit_price, 2) if t.exit_price else 0,
            'pnl': round(t.pnl, 2),
            'exit_reason': t.exit_reason or '',
            'bars_held': t.bars_held,
            'lots': t.lots,
        })
    return {
        'label': label, 'n': s['n'], 'sharpe': s['sharpe'],
        'total_pnl': s['total_pnl'], 'win_rate': s['win_rate'],
        'max_dd': s['max_dd'], 'maxloss_cap_count': s.get('maxloss_cap_count', 0),
        'trades': trade_list,
    }


def run_pool(tasks, func=_run_one):
    n = min(MAX_WORKERS, len(tasks))
    with mp.Pool(n) as pool:
        return pool.map(func, tasks)


# ═══════════════════════════════════════════════════════════════
# Test 1: Sub-sample Cap comparison
# ═══════════════════════════════════════════════════════════════

def run_test1(out):
    print("\n" + "=" * 70)
    print("Test 1: Sub-sample Cap Comparison")
    print("=" * 70)

    periods = [
        ("Early_2015-2019", "2015-01-01", "2020-01-01"),
        ("Mid_2020-2022", "2020-01-01", "2023-01-01"),
        ("Recent_2023-2026", "2023-01-01", "2026-05-01"),
    ]
    cap_values = [0, 30, 40, 50, 60, 70, 80]

    tasks = []
    for period_name, start, end in periods:
        for cap in cap_values:
            kw = get_base()
            if cap > 0:
                kw['maxloss_cap'] = cap
            cap_label = f"Cap{cap}" if cap > 0 else "NoCap"
            label = f"{period_name}_{cap_label}"
            tasks.append((label, kw, 0.30, start, end))

    results = run_pool(tasks)

    with open(f"{out}/test1_subsample.txt", 'w', encoding='utf-8') as f:
        f.write("Test 1: Sub-sample Cap Comparison (Lot=0.05)\n")
        f.write("=" * 100 + "\n\n")

        for period_name, _, _ in periods:
            f.write(f"\n--- {period_name} ---\n")
            sub = [r for r in results if r['label'].startswith(period_name)]
            sub.sort(key=lambda x: x['sharpe'], reverse=True)
            header = f"{'Cap':<10} {'N':>5} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10} {'CapHits':>8} {'CapHit%':>8}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for r in sub:
                cap_pct = (r['maxloss_cap_count'] / r['n'] * 100) if r['n'] > 0 else 0
                cap_label = r['label'].split('_')[-1]
                f.write(f"{cap_label:<10} {r['n']:>5} {r['sharpe']:>8.2f} "
                        f"{fmt(r['total_pnl']):>12} {r['win_rate']:>6.1f}% "
                        f"{fmt(r['max_dd']):>10} {r['maxloss_cap_count']:>8} {cap_pct:>7.1f}%\n")

        f.write("\n\n--- Cap x Period Matrix (Sharpe) ---\n")
        f.write(f"{'Cap':<10}")
        for pn, _, _ in periods:
            f.write(f" {pn:>20}")
        f.write("\n")
        for cap in cap_values:
            cap_label = f"Cap{cap}" if cap > 0 else "NoCap"
            f.write(f"{cap_label:<10}")
            for pn, _, _ in periods:
                r = next((x for x in results if x['label'] == f"{pn}_{cap_label}"), None)
                sh = r['sharpe'] if r else 0
                f.write(f" {sh:>20.2f}")
            f.write("\n")

    print("\n  Test 1 Results:")
    for pn, _, _ in periods:
        sub = [r for r in results if r['label'].startswith(pn)]
        best = max(sub, key=lambda x: x['sharpe'])
        cap_label = best['label'].split('_')[-1]
        print(f"    {pn}: Best={cap_label} Sharpe={best['sharpe']:.2f}")

    return results


# ═══════════════════════════════════════════════════════════════
# Test 2: Yearly Sharpe heatmap
# ═══════════════════════════════════════════════════════════════

def run_test2(out):
    print("\n" + "=" * 70)
    print("Test 2: Yearly Sharpe Heatmap")
    print("=" * 70)

    years = [(str(y), f"{y}-01-01", f"{y+1}-01-01") for y in range(2015, 2026)]
    cap_values = [0, 30, 50, 80]

    tasks = []
    for yr_name, start, end in years:
        for cap in cap_values:
            kw = get_base()
            if cap > 0:
                kw['maxloss_cap'] = cap
            cap_label = f"Cap{cap}" if cap > 0 else "NoCap"
            tasks.append((f"Y{yr_name}_{cap_label}", kw, 0.30, start, end))

    results = run_pool(tasks)

    with open(f"{out}/test2_yearly_heatmap.txt", 'w', encoding='utf-8') as f:
        f.write("Test 2: Yearly Sharpe Heatmap (Lot=0.05)\n")
        f.write("=" * 100 + "\n\n")

        f.write("Sharpe:\n")
        f.write(f"{'Year':<8}")
        for cap in cap_values:
            cl = f"Cap{cap}" if cap > 0 else "NoCap"
            f.write(f" {cl:>10}")
        f.write(f" {'Best':>10} {'Cap30-80':>10}\n")
        f.write("-" * 78 + "\n")

        for yr_name, _, _ in years:
            f.write(f"{yr_name:<8}")
            sharpes = {}
            for cap in cap_values:
                cl = f"Cap{cap}" if cap > 0 else "NoCap"
                r = next((x for x in results if x['label'] == f"Y{yr_name}_{cl}"), None)
                sh = r['sharpe'] if r else 0
                sharpes[cap] = sh
                f.write(f" {sh:>10.2f}")
            best_cap = max(sharpes, key=sharpes.get)
            best_label = f"Cap{best_cap}" if best_cap > 0 else "NoCap"
            delta = sharpes.get(30, 0) - sharpes.get(80, 0)
            f.write(f" {best_label:>10} {delta:>+9.2f}\n")

        f.write("\nPnL:\n")
        f.write(f"{'Year':<8}")
        for cap in cap_values:
            cl = f"Cap{cap}" if cap > 0 else "NoCap"
            f.write(f" {cl:>12}")
        f.write("\n")
        f.write("-" * 68 + "\n")
        for yr_name, _, _ in years:
            f.write(f"{yr_name:<8}")
            for cap in cap_values:
                cl = f"Cap{cap}" if cap > 0 else "NoCap"
                r = next((x for x in results if x['label'] == f"Y{yr_name}_{cl}"), None)
                pnl = r['total_pnl'] if r else 0
                f.write(f" {fmt(pnl):>12}")
            f.write("\n")

    print("  Yearly heatmap written")
    return results


# ═══════════════════════════════════════════════════════════════
# Test 3: Dynamic ATR Cap
# ═══════════════════════════════════════════════════════════════

def run_test3(out):
    print("\n" + "=" * 70)
    print("Test 3: Dynamic ATR Cap vs Fixed Cap")
    print("=" * 70)

    atr_mults = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    fixed_caps = [0, 30, 50, 80]

    periods = [
        ("Full", None, None),
        ("Recent_2023-2026", "2023-01-01", "2026-05-01"),
    ]

    tasks = []
    for period_name, start, end in periods:
        for mult in atr_mults:
            kw = get_base()
            kw['maxloss_cap_atr_mult'] = mult
            label = f"{period_name}_ATR{mult}"
            tasks.append((label, kw, 0.30, start, end))

        for cap in fixed_caps:
            kw = get_base()
            if cap > 0:
                kw['maxloss_cap'] = cap
            cap_label = f"Cap{cap}" if cap > 0 else "NoCap"
            label = f"{period_name}_{cap_label}"
            tasks.append((label, kw, 0.30, start, end))

    results = run_pool(tasks)

    with open(f"{out}/test3_dynamic_cap.txt", 'w', encoding='utf-8') as f:
        f.write("Test 3: Dynamic ATR Cap vs Fixed Cap (Lot=0.05)\n")
        f.write("=" * 100 + "\n\n")

        for period_name, _, _ in periods:
            f.write(f"\n--- {period_name} ---\n")
            sub = [r for r in results if r['label'].startswith(period_name + "_")]
            sub.sort(key=lambda x: x['sharpe'], reverse=True)
            header = f"{'Config':<18} {'N':>5} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10} {'CapHits':>8}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for r in sub:
                cfg = r['label'].replace(f"{period_name}_", "")
                f.write(f"{cfg:<18} {r['n']:>5} {r['sharpe']:>8.2f} "
                        f"{fmt(r['total_pnl']):>12} {r['win_rate']:>6.1f}% "
                        f"{fmt(r['max_dd']):>10} {r['maxloss_cap_count']:>8}\n")

    print("\n  Test 3 Results:")
    for period_name, _, _ in periods:
        sub = [r for r in results if r['label'].startswith(period_name + "_")]
        best = max(sub, key=lambda x: x['sharpe'])
        cfg = best['label'].replace(f"{period_name}_", "")
        print(f"    {period_name}: Best={cfg} Sharpe={best['sharpe']:.2f}")

    return results


# ═══════════════════════════════════════════════════════════════
# Test 4: False-kill analysis
# ═══════════════════════════════════════════════════════════════

def run_test4(out):
    print("\n" + "=" * 70)
    print("Test 4: False-Kill Analysis")
    print("=" * 70)

    kw_cap30 = get_base()
    kw_cap30['maxloss_cap'] = 30
    kw_nocap = get_base()

    tasks = [
        ("Cap30_Full", kw_cap30, 0.30, None, None),
        ("NoCap_Full", kw_nocap, 0.30, None, None),
    ]
    results = run_pool(tasks, func=_run_one_trades)

    cap30_trades = next(r for r in results if r['label'] == 'Cap30_Full')['trades']
    nocap_trades = next(r for r in results if r['label'] == 'NoCap_Full')['trades']

    nocap_by_entry = {}
    for t in nocap_trades:
        key = (t['entry_time'], t['direction'])
        nocap_by_entry[key] = t

    killed_trades = [t for t in cap30_trades if t['exit_reason'] == 'MaxLossCap']
    false_kills = []
    true_kills = []
    unmatched = 0

    for t in killed_trades:
        key = (t['entry_time'], t['direction'])
        nocap_t = nocap_by_entry.get(key)
        if nocap_t is None:
            unmatched += 1
            continue
        if nocap_t['pnl'] > 0:
            false_kills.append({
                'entry_time': t['entry_time'],
                'direction': t['direction'],
                'cap_pnl': t['pnl'],
                'nocap_pnl': nocap_t['pnl'],
                'nocap_exit': nocap_t['exit_reason'],
                'missed_profit': nocap_t['pnl'] - t['pnl'],
            })
        else:
            true_kills.append({
                'entry_time': t['entry_time'],
                'direction': t['direction'],
                'cap_pnl': t['pnl'],
                'nocap_pnl': nocap_t['pnl'],
                'saved_loss': t['pnl'] - nocap_t['pnl'],
            })

    total_killed = len(false_kills) + len(true_kills)
    false_kill_rate = len(false_kills) / total_killed * 100 if total_killed > 0 else 0

    by_year_fk = defaultdict(lambda: {'false': 0, 'true': 0})
    for fk in false_kills:
        yr = fk['entry_time'][:4]
        by_year_fk[yr]['false'] += 1
    for tk in true_kills:
        yr = tk['entry_time'][:4]
        by_year_fk[yr]['true'] += 1

    with open(f"{out}/test4_false_kill.txt", 'w', encoding='utf-8') as f:
        f.write("Test 4: False-Kill Analysis (Cap$30, Lot=0.05)\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Cap$30 trades: {len(cap30_trades)}\n")
        f.write(f"NoCap trades: {len(nocap_trades)}\n")
        f.write(f"MaxLossCap exits: {len(killed_trades)}\n")
        f.write(f"Matched: {total_killed}, Unmatched: {unmatched}\n\n")

        f.write(f"True kills (Cap saved money): {len(true_kills)} ({100-false_kill_rate:.1f}%)\n")
        avg_saved = np.mean([t['saved_loss'] for t in true_kills]) if true_kills else 0
        total_saved = sum(t['saved_loss'] for t in true_kills)
        f.write(f"  Avg saved per trade: ${avg_saved:.2f}\n")
        f.write(f"  Total saved: {fmt(total_saved)}\n\n")

        f.write(f"False kills (Cap truncated eventual winner): {len(false_kills)} ({false_kill_rate:.1f}%)\n")
        avg_missed = np.mean([t['missed_profit'] for t in false_kills]) if false_kills else 0
        total_missed = sum(t['missed_profit'] for t in false_kills)
        f.write(f"  Avg missed profit per trade: ${avg_missed:.2f}\n")
        f.write(f"  Total missed profit: {fmt(total_missed)}\n\n")

        net = total_saved - total_missed
        f.write(f"Net Cap value: {fmt(net)} ({'Cap helps' if net > 0 else 'Cap hurts'})\n\n")

        f.write("--- False-Kill Rate by Year ---\n")
        f.write(f"{'Year':<6} {'FalseKill':>10} {'TrueKill':>10} {'Total':>8} {'FalseKill%':>12}\n")
        f.write("-" * 50 + "\n")
        for yr in sorted(by_year_fk.keys()):
            fk = by_year_fk[yr]['false']
            tk = by_year_fk[yr]['true']
            total = fk + tk
            rate = fk / total * 100 if total > 0 else 0
            f.write(f"{yr:<6} {fk:>10} {tk:>10} {total:>8} {rate:>11.1f}%\n")

        f.write("\n--- Top 10 Worst False Kills (most missed profit) ---\n")
        top_fk = sorted(false_kills, key=lambda x: x['missed_profit'], reverse=True)[:10]
        for fk in top_fk:
            f.write(f"  {fk['entry_time']} {fk['direction']:>4} | "
                    f"Cap PnL=${fk['cap_pnl']:>8.2f} → NoCap PnL=${fk['nocap_pnl']:>8.2f} | "
                    f"Missed=${fk['missed_profit']:>8.2f} | NoCap exit: {fk['nocap_exit']}\n")

    print(f"\n  False-kill rate: {false_kill_rate:.1f}% ({len(false_kills)}/{total_killed})")
    print(f"  Net Cap value: {fmt(net)}")

    return {
        'false_kill_rate': false_kill_rate,
        'total_killed': total_killed,
        'net_value': net,
        'by_year': dict(by_year_fk),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()
    print(f"\n{'=' * 70}")
    print(f"R53B: MaxLoss Cap Stress Test")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")

    tests = [
        ("Test 1: Sub-sample", run_test1),
        ("Test 2: Yearly Heatmap", run_test2),
        ("Test 3: Dynamic ATR Cap", run_test3),
        ("Test 4: False-Kill", run_test4),
    ]

    all_results = {}
    for name, fn in tests:
        try:
            t_start = time.time()
            print(f"\n>>> {name}...")
            r = fn(OUTPUT_DIR)
            elapsed = time.time() - t_start
            all_results[name] = r
            print(f"<<< {name} done in {elapsed:.0f}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"<<< {name} FAILED: {e}")

    total = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"R53B Complete — Total: {total:.0f}s ({total / 60:.1f}min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
