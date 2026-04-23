"""
R33: External Data — Cross-Asset Correlation Regime + GVZ Implied Volatility
=============================================================================
A: XAUUSD vs DXY/US10Y/SPX rolling correlation analysis + L7 performance by regime
B: GVZ (Gold Vol Index) as sizing/filtering signal + GVZ-RV spread
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS

OUT_DIR = Path("results/round33_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXT_DIR = Path("data/external")


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


def load_ext(name):
    """Load external data CSV (yfinance format with multi-level header)."""
    path = EXT_DIR / f"{name}_daily.csv"
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    df.columns = df.columns.get_level_values(0)
    df = df.iloc[1:]  # skip the Ticker row if present
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df.dropna(subset=['Close'])
    return df


def make_daily(trades):
    daily = {}
    for t in trades:
        exit_t = t.exit_time if hasattr(t, 'exit_time') else t['exit_time']
        pnl = t.pnl if hasattr(t, 'pnl') else t['pnl']
        d = pd.Timestamp(exit_t).date()
        daily.setdefault(d, 0); daily[d] += pnl
    return pd.Series(daily).sort_index()


def run_phase_A(data):
    """Cross-asset correlation regime analysis."""
    print("\n" + "=" * 80)
    print("Phase A: Cross-Asset Correlation Regime")
    print("=" * 80)

    # Load external data
    ext_data = {}
    for name in ['DXY', 'US10Y', 'SPX']:
        try:
            df = load_ext(name)
            ext_data[name] = df['Close']
            print(f"  {name}: {len(df)} bars, {df.index[0].date()} -> {df.index[-1].date()}")
        except Exception as e:
            print(f"  {name}: FAILED to load: {e}")

    # Get XAUUSD daily close from H1 data
    h1 = data.h1_df
    xau_daily = h1['Close'].resample('1D').last().dropna()
    xau_daily.index = xau_daily.index.tz_localize(None)
    print(f"  XAUUSD: {len(xau_daily)} daily bars")

    # A1: Rolling correlation analysis
    print(f"\n  --- A1: Rolling Correlation (60d) ---")
    for name, ext_close in ext_data.items():
        aligned = pd.DataFrame({'xau': xau_daily, name: ext_close}).dropna()
        if len(aligned) < 100:
            print(f"  {name}: too few aligned bars ({len(aligned)})")
            continue

        xau_ret = aligned['xau'].pct_change()
        ext_ret = aligned[name].pct_change()

        for window in [20, 60, 120]:
            corr = xau_ret.rolling(window).corr(ext_ret)
            print(f"  XAUUSD vs {name} ({window}d): mean={corr.mean():.4f}, "
                  f"std={corr.std():.4f}, min={corr.min():.4f}, max={corr.max():.4f}")

    # A2: L7 performance segmented by correlation regime
    print(f"\n  --- A2: L7 Performance by DXY Correlation Regime ---")
    base = run_variant(data, "L7MH8_corr", verbose=False, **L7_MH8)
    trades = base['_trades']
    print(f"  Baseline: N={len(trades)}, Sharpe={base['sharpe']:.2f}")

    if 'DXY' in ext_data:
        dxy_close = ext_data['DXY']
        aligned = pd.DataFrame({'xau': xau_daily, 'dxy': dxy_close}).dropna()
        xau_ret = aligned['xau'].pct_change()
        dxy_ret = aligned['dxy'].pct_change()
        corr_60 = xau_ret.rolling(60).corr(dxy_ret)

        # Classify each trade by the correlation at entry time
        for regime_name, low, high in [
            ("Strong neg (<-0.5)", -1.1, -0.5),
            ("Moderate neg (-0.5~-0.2)", -0.5, -0.2),
            ("Weak (-0.2~0.2)", -0.2, 0.2),
            ("Moderate pos (0.2~0.5)", 0.2, 0.5),
            ("Strong pos (>0.5)", 0.5, 1.1),
        ]:
            kept = []; skipped = 0
            for t in trades:
                et = t.entry_time if hasattr(t, 'entry_time') else t['entry_time']
                et_date = pd.Timestamp(et).tz_localize(None).date()
                et_ts = pd.Timestamp(et_date)
                mask = corr_60.index <= et_ts
                if not mask.any() or pd.isna(corr_60[mask].iloc[-1]):
                    skipped += 1; continue
                c = corr_60[mask].iloc[-1]
                if low <= c < high:
                    kept.append(t)
                else:
                    skipped += 1

            if kept:
                daily = {}
                for t in kept:
                    exit_t = t.exit_time if hasattr(t, 'exit_time') else t['exit_time']
                    pnl = t.pnl if hasattr(t, 'pnl') else t['pnl']
                    d = pd.Timestamp(exit_t).date()
                    daily.setdefault(d, 0); daily[d] += pnl
                da = np.array(list(daily.values()))
                sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0
                avg_pnl = np.mean([t.pnl if hasattr(t,'pnl') else t['pnl'] for t in kept])
            else:
                sh = 0; avg_pnl = 0

            print(f"  {regime_name:>30}: N={len(kept):>5}, Sharpe={sh:>6.2f}, "
                  f"AvgPnL=${avg_pnl:>6.3f}")

    # A3: Same analysis for US10Y
    print(f"\n  --- A3: L7 Performance by US10Y Correlation Regime ---")
    if 'US10Y' in ext_data:
        us10_close = ext_data['US10Y']
        aligned = pd.DataFrame({'xau': xau_daily, 'us10': us10_close}).dropna()
        xau_ret = aligned['xau'].pct_change()
        us10_ret = aligned['us10'].pct_change()
        corr_60_us10 = xau_ret.rolling(60).corr(us10_ret)

        for regime_name, low, high in [
            ("Strong neg (<-0.5)", -1.1, -0.5),
            ("Moderate neg (-0.5~-0.2)", -0.5, -0.2),
            ("Weak (-0.2~0.2)", -0.2, 0.2),
            ("Positive (>0.2)", 0.2, 1.1),
        ]:
            kept = []
            for t in trades:
                et = t.entry_time if hasattr(t, 'entry_time') else t['entry_time']
                et_date = pd.Timestamp(et).tz_localize(None).date()
                et_ts = pd.Timestamp(et_date)
                mask = corr_60_us10.index <= et_ts
                if not mask.any() or pd.isna(corr_60_us10[mask].iloc[-1]): continue
                c = corr_60_us10[mask].iloc[-1]
                if low <= c < high: kept.append(t)

            if kept:
                daily = {}
                for t in kept:
                    exit_t = t.exit_time if hasattr(t, 'exit_time') else t['exit_time']
                    pnl = t.pnl if hasattr(t, 'pnl') else t['pnl']
                    d = pd.Timestamp(exit_t).date()
                    daily.setdefault(d, 0); daily[d] += pnl
                da = np.array(list(daily.values()))
                sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0
                avg_pnl = np.mean([t.pnl if hasattr(t,'pnl') else t['pnl'] for t in kept])
            else:
                sh = 0; avg_pnl = 0

            print(f"  {regime_name:>30}: N={len(kept):>5}, Sharpe={sh:>6.2f}, "
                  f"AvgPnL=${avg_pnl:>6.3f}")

    # A4: Correlation breakdown as filter
    print(f"\n  --- A4: DXY Correlation Filter for L7 ---")
    if 'DXY' in ext_data:
        for thresh_name, keep_fn in [
            ("Keep when neg corr (<0)", lambda c: c < 0),
            ("Keep when strong neg (<-0.3)", lambda c: c < -0.3),
            ("Skip when pos corr (>0.2)", lambda c: c < 0.2),
        ]:
            kept = []; skipped = 0
            for t in trades:
                et = t.entry_time if hasattr(t, 'entry_time') else t['entry_time']
                et_date = pd.Timestamp(et).tz_localize(None).date()
                et_ts = pd.Timestamp(et_date)
                mask = corr_60.index <= et_ts
                if not mask.any() or pd.isna(corr_60[mask].iloc[-1]):
                    kept.append(t); continue
                c = corr_60[mask].iloc[-1]
                if keep_fn(c): kept.append(t)
                else: skipped += 1

            daily = {}
            for t in kept:
                exit_t = t.exit_time if hasattr(t, 'exit_time') else t['exit_time']
                pnl = t.pnl if hasattr(t, 'pnl') else t['pnl']
                d = pd.Timestamp(exit_t).date()
                daily.setdefault(d, 0); daily[d] += pnl
            da = np.array(list(daily.values())) if daily else np.array([0])
            sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0

            print(f"  {thresh_name:>35}: N={len(kept):>5}, Sharpe={sh:>6.2f}, "
                  f"PnL=${da.sum():>8.0f}, Skipped={skipped}")


def run_phase_B(data):
    """GVZ (Gold Volatility Index) analysis."""
    print("\n" + "=" * 80)
    print("Phase B: GVZ (Gold Implied Volatility) Analysis")
    print("=" * 80)

    try:
        gvz_df = load_ext('GVZ')
        gvz = gvz_df['Close']
        print(f"  GVZ: {len(gvz)} bars, {gvz.index[0].date()} -> {gvz.index[-1].date()}")
        print(f"  GVZ stats: mean={gvz.mean():.1f}, std={gvz.std():.1f}, "
              f"min={gvz.min():.1f}, max={gvz.max():.1f}")
    except Exception as e:
        print(f"  GVZ FAILED to load: {e}")
        return

    # Compute realized vol from H1 data
    h1 = data.h1_df
    xau_daily = h1['Close'].resample('1D').last().dropna()
    xau_daily.index = xau_daily.index.tz_localize(None)
    rv_20 = xau_daily.pct_change().rolling(20).std() * np.sqrt(252) * 100  # annualized %

    aligned = pd.DataFrame({'gvz': gvz, 'rv': rv_20}).dropna()
    print(f"  RV(20d) stats: mean={aligned['rv'].mean():.1f}, std={aligned['rv'].std():.1f}")
    print(f"  GVZ-RV spread: mean={( aligned['gvz'] - aligned['rv']).mean():.2f}")

    # B1: GVZ percentile analysis
    print(f"\n  --- B1: GVZ Percentile Distribution ---")
    for pct in [10, 25, 50, 75, 90]:
        val = gvz.quantile(pct/100)
        print(f"  P{pct}: {val:.1f}")

    # B2: L7 performance by GVZ level
    print(f"\n  --- B2: L7 Performance by GVZ Level ---")
    base = run_variant(data, "L7MH8_gvz", verbose=False, **L7_MH8)
    trades = base['_trades']

    gvz_pct20 = gvz.quantile(0.20)
    gvz_pct40 = gvz.quantile(0.40)
    gvz_pct60 = gvz.quantile(0.60)
    gvz_pct80 = gvz.quantile(0.80)

    for regime_name, low, high in [
        (f"Very low (<P20={gvz_pct20:.0f})", 0, gvz_pct20),
        (f"Low (P20-P40={gvz_pct20:.0f}-{gvz_pct40:.0f})", gvz_pct20, gvz_pct40),
        (f"Medium (P40-P60={gvz_pct40:.0f}-{gvz_pct60:.0f})", gvz_pct40, gvz_pct60),
        (f"High (P60-P80={gvz_pct60:.0f}-{gvz_pct80:.0f})", gvz_pct60, gvz_pct80),
        (f"Very high (>P80={gvz_pct80:.0f})", gvz_pct80, 999),
    ]:
        kept = []; skipped = 0
        for t in trades:
            et = t.entry_time if hasattr(t, 'entry_time') else t['entry_time']
            et_date = pd.Timestamp(et).tz_localize(None).date()
            et_ts = pd.Timestamp(et_date)
            mask = gvz.index <= et_ts
            if not mask.any() or pd.isna(gvz[mask].iloc[-1]):
                skipped += 1; continue
            g = gvz[mask].iloc[-1]
            if low <= g < high: kept.append(t)
            else: skipped += 1

        if kept:
            daily = {}
            for t in kept:
                exit_t = t.exit_time if hasattr(t, 'exit_time') else t['exit_time']
                pnl = t.pnl if hasattr(t, 'pnl') else t['pnl']
                d = pd.Timestamp(exit_t).date()
                daily.setdefault(d, 0); daily[d] += pnl
            da = np.array(list(daily.values()))
            sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0
            avg_pnl = np.mean([t.pnl if hasattr(t,'pnl') else t['pnl'] for t in kept])
            wr = sum(1 for t in kept if (t.pnl if hasattr(t,'pnl') else t['pnl']) > 0) / len(kept) * 100
        else:
            sh = 0; avg_pnl = 0; wr = 0

        print(f"  {regime_name:>40}: N={len(kept):>5}, Sharpe={sh:>6.2f}, "
              f"AvgPnL=${avg_pnl:>6.3f}, WR={wr:>5.1f}%")

    # B3: GVZ as sizing signal
    print(f"\n  --- B3: GVZ-Based Position Sizing ---")
    gvz_median = gvz.median()

    for sizing_name, size_fn in [
        ("GVZ<median → 1.5x, else 0.5x", lambda g: 1.5 if g < gvz_median else 0.5),
        ("GVZ<P25 → skip", lambda g: 0.0 if g < gvz.quantile(0.25) else 1.0),
        ("GVZ>P75 → 0.5x", lambda g: 0.5 if g > gvz.quantile(0.75) else 1.0),
        ("GVZ>P80 → skip", lambda g: 0.0 if g > gvz.quantile(0.80) else 1.0),
    ]:
        daily = {}
        n_kept = 0
        for t in trades:
            et = t.entry_time if hasattr(t, 'entry_time') else t['entry_time']
            pnl = t.pnl if hasattr(t, 'pnl') else t['pnl']
            et_date = pd.Timestamp(et).tz_localize(None).date()
            et_ts = pd.Timestamp(et_date)
            mask = gvz.index <= et_ts
            if not mask.any() or pd.isna(gvz[mask].iloc[-1]):
                mult = 1.0
            else:
                g = gvz[mask].iloc[-1]
                mult = size_fn(g)
            if mult > 0:
                n_kept += 1
                d = pd.Timestamp(t.exit_time if hasattr(t,'exit_time') else t['exit_time']).date()
                daily.setdefault(d, 0); daily[d] += pnl * mult

        da = np.array(list(daily.values())) if daily else np.array([0])
        sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0

        print(f"  {sizing_name:>35}: N={n_kept:>5}, Sharpe={sh:>6.2f}, PnL=${da.sum():>8.0f}")

    # B4: GVZ-RV spread (volatility risk premium)
    print(f"\n  --- B4: GVZ-RV Spread as Signal ---")
    vrp = aligned['gvz'] - aligned['rv']

    for regime_name, low, high in [
        ("VRP < 0 (RV > IV)", -999, 0),
        ("VRP 0-5", 0, 5),
        ("VRP 5-10", 5, 10),
        ("VRP > 10 (IV >> RV)", 10, 999),
    ]:
        kept = []
        for t in trades:
            et = t.entry_time if hasattr(t, 'entry_time') else t['entry_time']
            et_date = pd.Timestamp(et).tz_localize(None).date()
            et_ts = pd.Timestamp(et_date)
            mask = vrp.index <= et_ts
            if not mask.any() or pd.isna(vrp[mask].iloc[-1]): continue
            v = vrp[mask].iloc[-1]
            if low <= v < high: kept.append(t)

        if kept:
            daily = {}
            for t in kept:
                exit_t = t.exit_time if hasattr(t, 'exit_time') else t['exit_time']
                pnl = t.pnl if hasattr(t, 'pnl') else t['pnl']
                d = pd.Timestamp(exit_t).date()
                daily.setdefault(d, 0); daily[d] += pnl
            da = np.array(list(daily.values()))
            sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0
            avg_pnl = np.mean([t.pnl if hasattr(t,'pnl') else t['pnl'] for t in kept])
        else:
            sh = 0; avg_pnl = 0

        print(f"  {regime_name:>25}: N={len(kept):>5}, Sharpe={sh:>6.2f}, AvgPnL=${avg_pnl:>6.3f}")


def main():
    t0 = time.time()
    out_path = OUT_DIR / "R33_output.txt"
    out = open(out_path, 'w', encoding='utf-8', buffering=1)
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, out)

    print(f"# R33: External Data — Cross-Asset Correlation + GVZ")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    data = DataBundle.load_default()

    for name, fn in [("A", lambda: run_phase_A(data)),
                     ("B", lambda: run_phase_B(data))]:
        try:
            fn()
            print(f"\n# Phase {name} completed at {datetime.now().strftime('%H:%M:%S')}")
            out.flush()
        except Exception as e:
            print(f"\n# Phase {name} FAILED: {e}")
            import traceback; traceback.print_exc()
            out.flush()

    elapsed = time.time() - t0
    print(f"\n# Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Elapsed: {elapsed/60:.1f} minutes")

    sys.stdout = old_stdout
    out.close()
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
