"""Profile the backtest engine on the fixed benchmark window.

Goal: find ACTUAL bottlenecks before guessing what to optimize.
"""
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine
from indicators import get_orb_strategy
import indicators as signals_mod


BASELINE_START = "2025-01-01"
BASELINE_END = "2025-06-30"

L8_BASE_KWARGS = {
    **LIVE_PARITY_KWARGS,
    'keltner_adx_threshold': 14,
    'regime_config': {
        'low':    {'trail_act': 0.22, 'trail_dist': 0.04},
        'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
        'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
    },
    'min_entry_gap_hours': 1.0,
    'keltner_max_hold_m15': 20,
    'spread_cost': 0.50,
    'kc_bw_filter_bars': 5,
}


def _reset_globals():
    get_orb_strategy().reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False


def main():
    lead_in = (pd.Timestamp(BASELINE_START, tz='UTC') - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    bundle = DataBundle.load_default(start=lead_in, end=BASELINE_END)
    m15 = bundle.m15_df[bundle.m15_df.index >= pd.Timestamp(BASELINE_START, tz='UTC')]
    h1 = bundle.h1_df[bundle.h1_df.index >= pd.Timestamp(BASELINE_START, tz='UTC')]
    print(f"  Window {BASELINE_START} -> {BASELINE_END}: M15={len(m15)} bars, H1={len(h1)} bars")

    _reset_globals()

    profiler = cProfile.Profile()
    print("\n  Running with cProfile ...")
    t0 = time.perf_counter()
    profiler.enable()
    engine = BacktestEngine(m15, h1, label="profile", **L8_BASE_KWARGS)
    trades = engine.run()
    profiler.disable()
    elapsed = time.perf_counter() - t0
    print(f"\n  Done. elapsed={elapsed:.2f}s  trades={len(trades)}")

    out_path = ROOT / "benchmarks" / "results" / "profile.prof"
    profiler.dump_stats(str(out_path))
    print(f"  Saved profile to {out_path}")

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(40)
    print("\n=== TOP 40 BY CUMULATIVE TIME ===")
    print(s.getvalue())

    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=s2).sort_stats('tottime')
    ps2.print_stats(40)
    print("\n=== TOP 40 BY TOTAL TIME (self) ===")
    print(s2.getvalue())


if __name__ == '__main__':
    main()
