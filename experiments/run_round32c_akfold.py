"""
R32-C Fix: K-Fold validation for Multi-TF Confirmation + correlation analysis.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from backtest.runner import DataBundle, run_variant, run_kfold, LIVE_PARITY_KWARGS

OUT_DIR = Path("results/round32_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            try: f.write(data)
            except: pass
            f.flush()
    def flush(self):
        for f in self.files: f.flush()


L7_MH8 = {
    **LIVE_PARITY_KWARGS,
    'time_adaptive_trail': {'start': 2, 'decay': 0.75, 'floor': 0.003},
    'min_entry_gap_hours': 1.0,
    'keltner_max_hold_m15': 8,
}


def make_daily(trades):
    daily = {}
    for t in trades:
        exit_t = t.exit_time if hasattr(t, 'exit_time') else t['exit_time']
        pnl = t.pnl if hasattr(t, 'pnl') else t['pnl']
        d = pd.Timestamp(exit_t).date()
        daily.setdefault(d, 0); daily[d] += pnl
    return pd.Series(daily).sort_index()


def add_h1_kc_dir(h1_df, ema_period=20, mult=2.0):
    h1 = h1_df.copy()
    h1['EMA_kc'] = h1['Close'].ewm(span=ema_period, adjust=False).mean()
    tr = pd.DataFrame({
        'hl': h1['High'] - h1['Low'],
        'hc': (h1['High'] - h1['Close'].shift(1)).abs(),
        'lc': (h1['Low'] - h1['Close'].shift(1)).abs(),
    }).max(axis=1)
    h1['ATR_kc'] = tr.rolling(14).mean()
    h1['KC_U'] = h1['EMA_kc'] + mult * h1['ATR_kc']
    h1['KC_L'] = h1['EMA_kc'] - mult * h1['ATR_kc']
    h1['kc_dir'] = 'NEUTRAL'
    h1.loc[h1['Close'] > h1['KC_U'], 'kc_dir'] = 'BULL'
    h1.loc[h1['Close'] < h1['KC_L'], 'kc_dir'] = 'BEAR'
    return h1


def filter_trades_by_h1_kc(trades, h1_kc, mode='same'):
    """Filter L7 trades by H1 KC direction.
    mode='same': keep only when L7 dir matches H1 KC dir
    """
    kept = []; skipped = 0
    for t in trades:
        et = t.entry_time if hasattr(t, 'entry_time') else t['entry_time']
        td = t.direction if hasattr(t, 'direction') else (t.get('dir') if isinstance(t, dict) else getattr(t, 'dir', None))
        et_ts = pd.Timestamp(et)
        h1_mask = h1_kc.index <= et_ts
        if not h1_mask.any(): skipped += 1; continue
        kc_d = h1_kc.loc[h1_kc.index[h1_mask][-1], 'kc_dir']
        if mode == 'same':
            if (td == 'BUY' and kc_d == 'BULL') or (td == 'SELL' and kc_d == 'BEAR'):
                kept.append(t)
            else: skipped += 1
        else:
            kept.append(t)
    return kept, skipped


def stats_from_kept(kept):
    if not kept: return 0, 0, 0
    daily = {}
    for t in kept:
        exit_t = t.exit_time if hasattr(t, 'exit_time') else t['exit_time']
        pnl = t.pnl if hasattr(t, 'pnl') else t['pnl']
        d = pd.Timestamp(exit_t).date()
        daily.setdefault(d, 0); daily[d] += pnl
    da = np.array(list(daily.values()))
    sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0
    return sh, da.sum(), len(kept)


def main():
    t0 = time.time()
    out_path = OUT_DIR / "R32c_output.txt"
    out = open(out_path, 'w', encoding='utf-8', buffering=1)
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, out)

    print(f"# R32-C: K-Fold Validation for Multi-TF + Correlation Analysis")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    data = DataBundle.load_default()
    h1_kc = add_h1_kc_dir(data.h1_df, ema_period=20, mult=2.0)

    # Phase A-KF: K-Fold validation using data.slice()
    print("=" * 80)
    print("Phase A-KF: K-Fold for H1 KC(EMA20/M2.0) Same-Dir Filter")
    print("=" * 80)

    folds = [
        ("Fold1", "2015-01-01", "2017-01-01"),
        ("Fold2", "2017-01-01", "2019-01-01"),
        ("Fold3", "2019-01-01", "2021-01-01"),
        ("Fold4", "2021-01-01", "2023-01-01"),
        ("Fold5", "2023-01-01", "2025-01-01"),
        ("Fold6", "2025-01-01", "2026-04-01"),
    ]

    print(f"  {'Fold':>6} {'Base_Sh':>8} {'Filter_Sh':>10} {'Delta':>7} {'N_base':>7} {'N_filt':>7}")
    pass_count = 0
    for fname, start, end in folds:
        fold_data = data.slice(start, end)
        if len(fold_data.m15_df) < 1000:
            print(f"  {fname:>6} SKIP (too few bars)")
            continue
        base = run_variant(fold_data, f"base_{fname}", verbose=False, **L7_MH8)
        trades_base = base['_trades']
        sh_base = base['sharpe']

        kept, skipped = filter_trades_by_h1_kc(trades_base, h1_kc, mode='same')
        sh_filt, pnl_filt, n_filt = stats_from_kept(kept)

        delta = sh_filt - sh_base
        if sh_filt > 0: pass_count += 1
        print(f"  {fname:>6} {sh_base:>8.2f} {sh_filt:>10.2f} {delta:>+7.2f} "
              f"{len(trades_base):>7} {n_filt:>7}")

    print(f"\n  K-Fold pass: {pass_count}/6 (Sharpe > 0)")
    print(f"  (Need 6/6 for deployment)")

    # Also test H1 KC(EMA25/M1.2) — the variant that showed best Sharpe 12.34
    print(f"\n  --- H1 KC(EMA25/M1.2) K-Fold ---")
    h1_kc_25 = add_h1_kc_dir(data.h1_df, ema_period=25, mult=1.2)
    print(f"  {'Fold':>6} {'Base_Sh':>8} {'Filter_Sh':>10} {'Delta':>7} {'N_base':>7} {'N_filt':>7}")
    pass_count_25 = 0
    for fname, start, end in folds:
        fold_data = data.slice(start, end)
        if len(fold_data.m15_df) < 1000: continue
        base = run_variant(fold_data, f"base25_{fname}", verbose=False, **L7_MH8)
        trades_base = base['_trades']
        sh_base = base['sharpe']

        kept, _ = filter_trades_by_h1_kc(trades_base, h1_kc_25, mode='same')
        sh_filt, pnl_filt, n_filt = stats_from_kept(kept)
        delta = sh_filt - sh_base
        if sh_filt > 0: pass_count_25 += 1
        print(f"  {fname:>6} {sh_base:>8.2f} {sh_filt:>10.2f} {delta:>+7.2f} "
              f"{len(trades_base):>7} {n_filt:>7}")
    print(f"\n  K-Fold pass: {pass_count_25}/6")

    # Correlation analysis: L7 filtered vs L7 base daily PnL
    print(f"\n  --- Correlation: L7(base) vs L7(H1-filtered) daily PnL ---")
    base_full = run_variant(data, "L7MH8_corr", verbose=False, **L7_MH8)
    trades_full = base_full['_trades']
    kept_full, _ = filter_trades_by_h1_kc(trades_full, h1_kc, mode='same')

    daily_base = make_daily(trades_full)
    daily_filt = make_daily(kept_full)

    aligned = pd.DataFrame({'base': daily_base, 'filt': daily_filt}).fillna(0)
    corr = aligned['base'].corr(aligned['filt'])
    print(f"  Correlation: {corr:.4f}")
    print(f"  (High correlation expected since filtered is a subset)")

    # Impact analysis: what's the PnL of ONLY the skipped trades?
    kept_set = set()
    for t in kept_full:
        et = t.entry_time if hasattr(t, 'entry_time') else t['entry_time']
        kept_set.add(str(et))

    skipped_pnl = []
    for t in trades_full:
        et = t.entry_time if hasattr(t, 'entry_time') else t['entry_time']
        if str(et) not in kept_set:
            pnl = t.pnl if hasattr(t, 'pnl') else t['pnl']
            skipped_pnl.append(pnl)

    print(f"\n  Skipped trades analysis:")
    print(f"    Total skipped: {len(skipped_pnl)}")
    if skipped_pnl:
        sp = np.array(skipped_pnl)
        print(f"    Skipped avg PnL: ${sp.mean():.3f}")
        print(f"    Skipped total PnL: ${sp.sum():.0f}")
        print(f"    Skipped win rate: {sum(1 for p in sp if p > 0)/len(sp)*100:.1f}%")
        print(f"    Kept avg PnL: ${sum(t.pnl if hasattr(t,'pnl') else t['pnl'] for t in kept_full)/len(kept_full):.3f}")
        print(f"    Baseline avg PnL: ${sum(t.pnl if hasattr(t,'pnl') else t['pnl'] for t in trades_full)/len(trades_full):.3f}")

    elapsed = time.time() - t0
    print(f"\n# Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Elapsed: {elapsed/60:.1f} minutes")

    sys.stdout = old_stdout
    out.close()
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
