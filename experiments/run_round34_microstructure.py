"""
R34: Order Flow / Microstructure Analysis (Tick Data)
======================================================
A: Tick-level bid-ask spread anomaly detection
B: Trade density (ticks per minute) as momentum confirmation
C: Large tick jumps (price jump direction bias)
D: Liquidity drought signals (spread widening)
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS

OUT_DIR = Path("results/round34_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TICK_DIR = Path("data/tick")
TICK_FILE = "xauusd_ticks_2026q1.csv"


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


def load_tick_data():
    """Load tick data CSV, keeping only essential columns."""
    path = TICK_DIR / TICK_FILE
    if not path.exists():
        print(f"  Tick data not found: {path}")
        return None
    print(f"  Loading tick data from {path}...", flush=True)
    df = pd.read_csv(path, usecols=['timestamp', 'ask', 'bid', 'spread', 'mid'],
                     dtype={'ask': 'float32', 'bid': 'float32',
                            'spread': 'float32', 'mid': 'float32'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    df = df.set_index('timestamp').sort_index()
    print(f"  Tick data: {len(df):,} ticks, {df.index[0]} -> {df.index[-1]}")
    print(f"  Spread: mean=${df['spread'].mean():.4f}, median=${df['spread'].median():.4f}")
    print(f"  Memory: {df.memory_usage(deep=True).sum()/1024/1024:.0f} MB")
    return df


def run_phase_A(tick_df):
    """Bid-ask spread anomaly detection."""
    print("\n" + "=" * 80)
    print("Phase A: Bid-Ask Spread Anomaly Detection")
    print("=" * 80)

    spread = tick_df['spread']
    spread_1m = spread.resample('1min').mean().dropna()
    spread_5m = spread.resample('5min').mean().dropna()
    spread_15m = spread.resample('15min').mean().dropna()

    print(f"\n  --- A1: Spread Distribution ---")
    for name, s in [("1min", spread_1m), ("5min", spread_5m), ("15min", spread_15m)]:
        print(f"  {name}: N={len(s)}, mean=${s.mean():.4f}, median=${s.median():.4f}, "
              f"P95=${s.quantile(0.95):.4f}, P99=${s.quantile(0.99):.4f}, max=${s.max():.4f}")

    # A2: Spread by hour of day
    print(f"\n  --- A2: Average Spread by Hour ---")
    spread_hourly = spread_1m.groupby(spread_1m.index.hour).mean()
    for hour in range(24):
        if hour in spread_hourly.index:
            print(f"  Hour {hour:>2}: ${spread_hourly[hour]:.4f}")

    # A3: Spread spikes (>3x median)
    median_sp = spread_1m.median()
    spikes = spread_1m[spread_1m > 3 * median_sp]
    print(f"\n  --- A3: Spread Spikes (>3x median=${3*median_sp:.4f}) ---")
    print(f"  Total spikes: {len(spikes)} ({100*len(spikes)/len(spread_1m):.2f}%)")
    if len(spikes) > 0:
        spike_hours = pd.Series(spikes.index.hour).value_counts().sort_index()
        print(f"  Spike distribution by hour:")
        for h, cnt in spike_hours.items():
            print(f"    Hour {h:>2}: {cnt} spikes")


def run_phase_B(tick_df):
    """Trade density analysis."""
    print("\n" + "=" * 80)
    print("Phase B: Trade Density (Ticks per Minute)")
    print("=" * 80)

    # Count ticks per minute
    tpm = tick_df.resample('1min').size()
    tpm = tpm[tpm > 0]  # exclude market-closed minutes

    print(f"\n  --- B1: Tick Density Distribution ---")
    print(f"  Ticks/min: mean={tpm.mean():.1f}, median={tpm.median():.0f}, "
          f"P95={tpm.quantile(0.95):.0f}, P99={tpm.quantile(0.99):.0f}, max={tpm.max():.0f}")

    # B2: Density by hour
    print(f"\n  --- B2: Average Ticks/Min by Hour ---")
    hourly_density = tpm.groupby(tpm.index.hour).mean()
    for hour in range(24):
        if hour in hourly_density.index:
            bars = "█" * int(hourly_density[hour] / 5)
            print(f"  Hour {hour:>2}: {hourly_density[hour]:>6.1f} tpm {bars}")

    # B3: High density bursts
    density_5m = tick_df.resample('5min').size()
    density_5m = density_5m[density_5m > 0]
    high_density = density_5m[density_5m > density_5m.quantile(0.95)]
    print(f"\n  --- B3: High Density Bursts (>P95={density_5m.quantile(0.95):.0f} ticks/5min) ---")
    print(f"  Burst count: {len(high_density)}")

    # B4: Price movement during high vs low density
    mid_5m = tick_df['mid'].resample('5min').agg(['first', 'last'])
    mid_5m['ret'] = (mid_5m['last'] - mid_5m['first']).abs()
    mid_5m['density'] = density_5m
    mid_5m = mid_5m.dropna()

    high_dens = mid_5m[mid_5m['density'] > mid_5m['density'].quantile(0.75)]
    low_dens = mid_5m[mid_5m['density'] <= mid_5m['density'].quantile(0.25)]

    print(f"\n  --- B4: Price Movement vs Density ---")
    print(f"  High density (>P75): avg |move| = ${high_dens['ret'].mean():.3f}")
    print(f"  Low density  (<P25): avg |move| = ${low_dens['ret'].mean():.3f}")
    print(f"  Ratio: {high_dens['ret'].mean() / max(low_dens['ret'].mean(), 0.001):.2f}x")


def run_phase_C(tick_df):
    """Large tick jump analysis."""
    print("\n" + "=" * 80)
    print("Phase C: Large Tick Jumps")
    print("=" * 80)

    mid = tick_df['mid']
    jumps = mid.diff()

    abs_jumps = jumps.abs()
    threshold = abs_jumps.quantile(0.999)

    large_jumps = jumps[abs_jumps > threshold]
    print(f"\n  --- C1: Jump Distribution ---")
    print(f"  Total ticks: {len(mid):,}")
    print(f"  Jump P99.9 threshold: ${threshold:.4f}")
    print(f"  Large jumps (>P99.9): {len(large_jumps):,}")

    # Direction bias
    up_jumps = large_jumps[large_jumps > 0]
    down_jumps = large_jumps[large_jumps < 0]
    print(f"\n  --- C2: Jump Direction Bias ---")
    print(f"  Up jumps: {len(up_jumps)} (avg ${up_jumps.mean():.3f})")
    print(f"  Down jumps: {len(down_jumps)} (avg ${down_jumps.mean():.3f})")
    print(f"  Up/Down ratio: {len(up_jumps)/max(len(down_jumps),1):.2f}")

    # What happens after a large jump? (next 1min, 5min, 15min return)
    print(f"\n  --- C3: Post-Jump Returns ---")
    jump_times = large_jumps.index
    for horizon, label in [(60, "1min"), (300, "5min"), (900, "15min")]:
        continuations = 0; reversals = 0
        for jt in jump_times[:min(1000, len(jump_times))]:
            future_mask = (mid.index > jt) & (mid.index <= jt + pd.Timedelta(seconds=horizon))
            if not future_mask.any(): continue
            future_mid = mid[future_mask]
            if len(future_mid) == 0: continue
            ret = future_mid.iloc[-1] - mid[jt]
            if (jumps[jt] > 0 and ret > 0) or (jumps[jt] < 0 and ret < 0):
                continuations += 1
            elif (jumps[jt] > 0 and ret < 0) or (jumps[jt] < 0 and ret > 0):
                reversals += 1
        total = continuations + reversals
        if total > 0:
            print(f"  {label}: continuation={continuations/total*100:.1f}%, "
                  f"reversal={reversals/total*100:.1f}% (n={total})")


def run_phase_D(tick_df, data):
    """Liquidity drought signals."""
    print("\n" + "=" * 80)
    print("Phase D: Liquidity Drought Signals")
    print("=" * 80)

    spread_15m = tick_df['spread'].resample('15min').mean()
    spread_15m = spread_15m.dropna()

    # Create a "liquidity drought" indicator: spread > 2x rolling median
    spread_median = spread_15m.rolling(48).median()  # 12-hour median
    drought = spread_15m > 2 * spread_median

    print(f"  --- D1: Drought Frequency ---")
    print(f"  Drought events (spread > 2x 12h median): {drought.sum()} "
          f"({100*drought.sum()/len(drought):.1f}%)")

    # D2: L7 performance during drought vs normal
    base = run_variant(data, "L7MH8_liq", verbose=False, **L7_MH8)
    trades = base['_trades']

    # Filter to tick data period only
    tick_start = tick_df.index[0]
    tick_end = tick_df.index[-1]

    kept_drought = []; kept_normal = []; skipped = 0
    for t in trades:
        et = t.entry_time if hasattr(t, 'entry_time') else t['entry_time']
        et_ts = pd.Timestamp(et)
        if et_ts.tzinfo:
            et_ts = et_ts.tz_localize(None)
        if et_ts < tick_start or et_ts > tick_end:
            skipped += 1; continue

        mask = spread_15m.index <= et_ts
        if not mask.any(): continue
        sp = spread_15m[mask].iloc[-1]
        med = spread_median[mask].iloc[-1] if mask.any() and not pd.isna(spread_median[mask].iloc[-1]) else spread_15m.median()

        if sp > 2 * med:
            kept_drought.append(t)
        else:
            kept_normal.append(t)

    print(f"\n  --- D2: L7 in Tick Data Period ---")
    print(f"  Total trades in period: {len(kept_drought)+len(kept_normal)} "
          f"(skipped {skipped} outside tick range)")

    for name, kept in [("Normal liquidity", kept_normal), ("Drought", kept_drought)]:
        if not kept: continue
        pnls = [t.pnl if hasattr(t,'pnl') else t['pnl'] for t in kept]
        daily = {}
        for t in kept:
            exit_t = t.exit_time if hasattr(t, 'exit_time') else t['exit_time']
            pnl = t.pnl if hasattr(t, 'pnl') else t['pnl']
            d = pd.Timestamp(exit_t).date()
            daily.setdefault(d, 0); daily[d] += pnl
        da = np.array(list(daily.values()))
        sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100

        print(f"  {name:>20}: N={len(kept):>5}, Sharpe={sh:>6.2f}, AvgPnL=${np.mean(pnls):>.3f}, WR={wr:.1f}%")


def main():
    t0 = time.time()
    out_path = OUT_DIR / "R34_output.txt"
    out = open(out_path, 'w', encoding='utf-8', buffering=1)
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, out)

    print(f"# R34: Order Flow / Microstructure Analysis")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    tick_df = load_tick_data()
    if tick_df is None:
        print("ABORTED: No tick data available")
        sys.stdout = old_stdout; out.close()
        return

    data = DataBundle.load_default()

    for name, fn in [("A", lambda: run_phase_A(tick_df)),
                     ("B", lambda: run_phase_B(tick_df)),
                     ("C", lambda: run_phase_C(tick_df)),
                     ("D", lambda: run_phase_D(tick_df, data))]:
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
