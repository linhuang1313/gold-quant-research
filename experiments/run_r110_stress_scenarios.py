#!/usr/bin/env python3
"""
R110 — Historical Stress Scenario Analysis
=============================================
Tests portfolio performance during known market crises and synthetic shocks.

  Phase 1-5: COVID, Russia-Ukraine, Fed Pivot, Gold Breakout, Tariff Shock
  Phase 6: Synthetic overnight gap tests (5%/10%/15%)
  Phase 7: Cross-scenario vulnerability analysis
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import DataBundle, load_m15, load_h1_aligned, H1_CSV_PATH
from backtest.runner import run_variant, LIVE_PARITY_KWARGS

OUTPUT_DIR = Path("results/r110_stress_scenarios")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']

SCENARIOS = [
    ("COVID_Crash", "2020-02-20", "2020-04-15"),
    ("Russia_Ukraine", "2022-02-15", "2022-04-15"),
    ("Fed_Pivot_2023", "2023-10-01", "2024-01-15"),
    ("Gold_Breakout_2024", "2024-02-01", "2024-05-01"),
    ("Tariff_Shock_2025", "2025-01-15", "2025-04-15"),
]


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


def add_psar(df, af_start=0.01, af_max=0.05):
    df = df.copy(); n = len(df)
    psar = np.zeros(n); direction = np.ones(n)
    af = af_start; ep = df['High'].iloc[0]; psar[0] = df['Low'].iloc[0]
    for i in range(1, n):
        prev = psar[i-1]
        if direction[i-1] == 1:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], df['Low'].iloc[i-1], df['Low'].iloc[max(0, i-2)])
            if df['Low'].iloc[i] < psar[i]:
                direction[i] = -1; psar[i] = ep; ep = df['Low'].iloc[i]; af = af_start
            else:
                direction[i] = 1
                if df['High'].iloc[i] > ep:
                    ep = df['High'].iloc[i]; af = min(af+af_start, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], df['High'].iloc[i-1], df['High'].iloc[max(0, i-2)])
            if df['High'].iloc[i] > psar[i]:
                direction[i] = 1; psar[i] = ep; ep = df['High'].iloc[i]; af = af_start
            else:
                direction[i] = -1
                if df['Low'].iloc[i] < ep:
                    ep = df['Low'].iloc[i]; af = min(af+af_start, af_max)
    df['PSAR_dir'] = direction; df['ATR'] = compute_atr(df)
    return df


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


def _run_exit_with_cap(pos, i, h, lo_v, c, spread, lot, pv, times,
                       sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold,
                       maxloss_cap=0):
    held = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_h = (h - pos['entry'] - spread) * lot * pv
        pnl_l = (lo_v - pos['entry'] - spread) * lot * pv
        pnl_c = (c - pos['entry'] - spread) * lot * pv
    else:
        pnl_h = (pos['entry'] - lo_v - spread) * lot * pv
        pnl_l = (pos['entry'] - h - spread) * lot * pv
        pnl_c = (pos['entry'] - c - spread) * lot * pv
    tp_val = tp_atr * pos['atr'] * lot * pv
    sl_val = sl_atr * pos['atr'] * lot * pv
    if pnl_h >= tp_val:
        return _mk(pos, c, times[i], "TP", i, tp_val)
    if pnl_l <= -sl_val:
        return _mk(pos, c, times[i], "SL", i, -sl_val)
    if maxloss_cap > 0 and pnl_c < -maxloss_cap:
        return _mk(pos, c, times[i], "MaxLossCap", i, -maxloss_cap)
    ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
    if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
        ts_p = h - td
        if lo_v <= ts_p:
            return _mk(pos, c, times[i], "Trail", i, (ts_p - pos['entry'] - spread) * lot * pv)
    elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
        ts_p = lo_v + td
        if h >= ts_p:
            return _mk(pos, c, times[i], "Trail", i, (pos['entry'] - ts_p - spread) * lot * pv)
    if held >= max_hold:
        return _mk(pos, c, times[i], "Timeout", i, pnl_c)
    return None


# ═══════════════════════════════════════════════════════════════
# Strategy backtests (4 core)
# ═══════════════════════════════════════════════════════════════

def bt_psar(h1_df, spread, lot, maxloss_cap=0,
            sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04, max_hold=20):
    df = add_psar(h1_df).dropna(subset=['PSAR_dir', 'ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    pdir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pdir[i-1] == -1 and pdir[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif pdir[i-1] == 1 and pdir[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom(h1_df, spread, lot, maxloss_cap=0,
             fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
             trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    max_lb = max(fast, slow)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        if c[i-fast] > 0: s += 0.5 * np.sign(c[i]/c[i-fast] - 1.0)
        if c[i-slow] > 0: s += 0.5 * np.sign(c[i]/c[i-slow] - 1.0)
        score[i] = s
    trades = []; pos = None; last_exit = -999
    for i in range(max_lb+1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                pnl = (c[i]-pos['entry']-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                pnl = (pos['entry']-c[i]-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(score[i]) or np.isnan(score[i-1]): continue
        if score[i] > 0 and score[i-1] <= 0:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_sess_bo(h1_df, spread, lot, maxloss_cap=0,
               session_hour=12, lookback=4, sl_atr=4.5, tp_atr=4.0,
               trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(lookback, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if hours[i] != session_hour: continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_l8_max(data_bundle, spread, lot, maxloss_cap=35):
    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': maxloss_cap,
          'spread_cost': spread, 'initial_capital': 2000,
          'min_lot_size': lot, 'max_lot_size': lot}
    result = run_variant(data_bundle, "L8_MAX", verbose=False, **kw)
    raw_trades = result.get('_trades', [])
    trades = []
    for t in raw_trades:
        trades.append({
            'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0,
        })
    return trades


# ═══════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════

def trades_to_daily_series(trades):
    if not trades:
        return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


def sharpe(arr):
    if len(arr) < 10: return 0.0
    s = np.std(arr, ddof=1)
    return float(np.mean(arr) / s * np.sqrt(252)) if s > 0 else 0.0


def max_dd(arr):
    if len(arr) == 0: return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def build_portfolio_daily(strat_trades_dict, lots):
    all_daily = {}
    for name, trades in strat_trades_dict.items():
        lot = lots.get(name, UNIT_LOT)
        scale = lot / UNIT_LOT
        for t in trades:
            d = pd.Timestamp(t['exit_time']).date()
            all_daily[d] = all_daily.get(d, 0) + t['pnl'] * scale
    dates = sorted(all_daily.keys())
    return pd.Series([all_daily[d] for d in dates], index=pd.DatetimeIndex(dates))


# ═══════════════════════════════════════════════════════════════
# Run helpers
# ═══════════════════════════════════════════════════════════════

def run_all_strategies_h1(h1_df, data_bundle):
    """Run all 4 strategies on given H1 data, returns dict of trade lists."""
    strat_trades = {}
    strat_trades['PSAR'] = bt_psar(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['PSAR'])
    strat_trades['TSMOM'] = bt_tsmom(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['TSMOM'])
    strat_trades['SESS_BO'] = bt_sess_bo(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['SESS_BO'])
    l8_trades = bt_l8_max(data_bundle, SPREAD, UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])
    strat_trades['L8_MAX'] = l8_trades
    return strat_trades


def filter_trades_to_period(trades, start, end):
    """Keep only trades whose exit_time falls within [start, end)."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    def _ts(t):
        ts = pd.Timestamp(t['exit_time'])
        if ts.tz is not None:
            ts = ts.tz_localize(None)
        return ts
    return [t for t in trades if s <= _ts(t) < e]


def strat_summary(trades):
    """Single-strategy summary dict for a trade list."""
    n = len(trades)
    if n == 0:
        return {'n_trades': 0, 'pnl': 0.0, 'max_dd': 0.0, 'wr': 0.0, 'worst_trade': 0.0}
    pnls = [t['pnl'] for t in trades]
    daily = trades_to_daily_series(trades)
    wins = sum(1 for p in pnls if p > 0)
    return {
        'n_trades': n,
        'pnl': round(sum(pnls), 2),
        'max_dd': round(max_dd(daily.values), 2) if len(daily) > 0 else 0.0,
        'wr': round(wins / n * 100, 1),
        'worst_trade': round(min(pnls), 2),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80)
    print("  R110 — Historical Stress Scenario Analysis")
    print("=" * 80)

    print("\n  Loading data...")
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    data_bundle = DataBundle.load_custom()
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0].date()} ~ {h1_df.index[-1].date()})")

    # Run all strategies on full data (needed for L8_MAX filtering)
    print("\n  Running full-data backtests (for L8_MAX trade filtering)...")
    full_trades = run_all_strategies_h1(h1_df, data_bundle)
    for name in STRAT_ORDER:
        print(f"    {name}: {len(full_trades[name])} trades")

    results = {'experiment': 'R110 Historical Stress Scenario Analysis', 'scenarios': {}}

    # ═══════════════════════════════════════════════════════════
    # Phases 1-5: Historical Crisis Backtests
    # ═══════════════════════════════════════════════════════════
    scenario_pnl_matrix = {}  # strategy -> {scenario_name: pnl}
    scenario_gold_returns = {}

    for phase_idx, (sc_name, sc_start, sc_end) in enumerate(SCENARIOS, start=1):
        print(f"\n{'=' * 70}")
        print(f"  Phase {phase_idx}: {sc_name} ({sc_start} to {sc_end})")
        print(f"{'=' * 70}")

        h1_slice = h1_df[(h1_df.index >= sc_start) & (h1_df.index < sc_end)]
        if len(h1_slice) < 10:
            print(f"    SKIP: only {len(h1_slice)} bars in period")
            continue

        gold_start = h1_slice['Close'].iloc[0]
        gold_end = h1_slice['Close'].iloc[-1]
        gold_change_pct = (gold_end - gold_start) / gold_start * 100
        scenario_gold_returns[sc_name] = gold_change_pct

        print(f"    Gold: ${gold_start:.2f} -> ${gold_end:.2f} ({gold_change_pct:+.2f}%)")
        print(f"    H1 bars: {len(h1_slice)}")

        scenario_strat_trades = {}
        scenario_result = {
            'period': f"{sc_start} to {sc_end}",
            'gold_start': round(gold_start, 2),
            'gold_end': round(gold_end, 2),
            'gold_change_pct': round(gold_change_pct, 2),
            'h1_bars': len(h1_slice),
            'strategies': {},
        }

        print(f"\n    {'Strategy':<10} {'#Trades':>7} {'PnL':>10} {'MaxDD':>10} "
              f"{'WR':>6} {'WorstTrade':>12}")
        print(f"    {'-'*60}")

        for name in STRAT_ORDER:
            if name == 'L8_MAX':
                trades = filter_trades_to_period(full_trades['L8_MAX'], sc_start, sc_end)
            else:
                trades = filter_trades_to_period(full_trades[name], sc_start, sc_end)

            sm = strat_summary(trades)
            scenario_strat_trades[name] = trades
            scenario_result['strategies'][name] = sm

            if name not in scenario_pnl_matrix:
                scenario_pnl_matrix[name] = {}
            scenario_pnl_matrix[name][sc_name] = sm['pnl']

            print(f"    {name:<10} {sm['n_trades']:>7} ${sm['pnl']:>9.2f} "
                  f"${sm['max_dd']:>9.2f} {sm['wr']:>5.1f}% ${sm['worst_trade']:>11.2f}")

        # Portfolio with R89 lots
        port_daily = build_portfolio_daily(scenario_strat_trades, R89_LOTS)
        port_pnl = float(port_daily.sum()) if len(port_daily) > 0 else 0.0
        port_dd = max_dd(port_daily.values) if len(port_daily) > 0 else 0.0

        scenario_result['portfolio'] = {
            'pnl': round(port_pnl, 2),
            'max_dd': round(port_dd, 2),
        }

        # Weakest link: strategy with lowest PnL contribution (scaled by R89 lots)
        contributions = {}
        for name in STRAT_ORDER:
            scale = R89_LOTS[name] / UNIT_LOT
            contributions[name] = scenario_result['strategies'][name]['pnl'] * scale
        weakest = min(contributions, key=contributions.get)
        scenario_result['weakest_link'] = weakest
        scenario_result['contributions'] = {k: round(v, 2) for k, v in contributions.items()}

        print(f"\n    Portfolio (R89 lots): PnL=${port_pnl:.2f}, MaxDD=${port_dd:.2f}")
        print(f"    Contributions: {', '.join(f'{k}=${v:.2f}' for k,v in contributions.items())}")
        print(f"    Weakest link: {weakest} (${contributions[weakest]:.2f})")

        results['scenarios'][sc_name] = scenario_result

    # ═══════════════════════════════════════════════════════════
    # Phase 6: Synthetic Gap Tests
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"  Phase 6: Synthetic Overnight Gap Tests")
    print(f"{'=' * 70}")

    rng = np.random.default_rng(seed=42)
    gap_sizes = [0.05, 0.10, 0.15]
    n_random_dates = 10

    valid_mask = (h1_df.index >= '2020-01-01') & (h1_df.index < '2025-01-01')
    valid_indices = np.where(valid_mask.values)[0]
    margin = 100
    valid_indices = valid_indices[(valid_indices > margin) & (valid_indices < len(h1_df) - margin)]
    sample_indices = rng.choice(valid_indices, size=n_random_dates, replace=False)
    sample_indices.sort()

    gap_results = {}

    for gap_pct in gap_sizes:
        label = f"{int(gap_pct*100)}%"
        print(f"\n    Gap size: {label}")

        up_losses = []
        down_losses = []
        up_affected = []
        down_affected = []

        for trial, bar_idx in enumerate(sample_indices):
            gap_date = h1_df.index[bar_idx]

            for direction_label, sign in [("up", 1), ("down", -1)]:
                h1_mod = h1_df.copy()
                factor = 1 + sign * gap_pct
                h1_mod.iloc[bar_idx:, h1_mod.columns.get_indexer(['Open', 'High', 'Low', 'Close'])] *= factor

                mod_strat_trades = {}
                mod_strat_trades['PSAR'] = bt_psar(h1_mod, SPREAD, UNIT_LOT, maxloss_cap=CAPS['PSAR'])
                mod_strat_trades['TSMOM'] = bt_tsmom(h1_mod, SPREAD, UNIT_LOT, maxloss_cap=CAPS['TSMOM'])
                mod_strat_trades['SESS_BO'] = bt_sess_bo(h1_mod, SPREAD, UNIT_LOT, maxloss_cap=CAPS['SESS_BO'])
                mod_strat_trades['L8_MAX'] = full_trades['L8_MAX']

                # Count positions open at gap time
                affected = 0
                gap_loss = 0.0
                for name in ['PSAR', 'TSMOM', 'SESS_BO']:
                    orig_trades = full_trades[name]
                    for t in orig_trades:
                        et = pd.Timestamp(t['entry_time'])
                        xt = pd.Timestamp(t['exit_time'])
                        if et.tz is not None: et = et.tz_localize(None)
                        if xt.tz is not None: xt = xt.tz_localize(None)
                        gd = gap_date.tz_localize(None) if hasattr(gap_date, 'tz') and gap_date.tz is not None else gap_date
                        if et < gd <= xt:
                            affected += 1
                            gap_loss += t['pnl'] * sign * gap_pct * (R89_LOTS[name] / UNIT_LOT)

                if direction_label == "up":
                    up_affected.append(affected)
                    mod_port = build_portfolio_daily(mod_strat_trades, R89_LOTS)
                    orig_port = build_portfolio_daily(
                        {k: full_trades[k] for k in ['PSAR', 'TSMOM', 'SESS_BO', 'L8_MAX']},
                        R89_LOTS)
                    pnl_diff = float(mod_port.sum() - orig_port.sum())
                    up_losses.append(pnl_diff)
                else:
                    down_affected.append(affected)
                    mod_port = build_portfolio_daily(mod_strat_trades, R89_LOTS)
                    orig_port = build_portfolio_daily(
                        {k: full_trades[k] for k in ['PSAR', 'TSMOM', 'SESS_BO', 'L8_MAX']},
                        R89_LOTS)
                    pnl_diff = float(mod_port.sum() - orig_port.sum())
                    down_losses.append(pnl_diff)

        gap_results[label] = {
            'gap_pct': gap_pct,
            'n_trials': n_random_dates,
            'avg_affected_positions': round(float(np.mean(up_affected + down_affected)), 1),
            'up_gap': {
                'avg_pnl_impact': round(float(np.mean(up_losses)), 2),
                'worst_pnl_impact': round(float(min(up_losses)), 2),
                'best_pnl_impact': round(float(max(up_losses)), 2),
            },
            'down_gap': {
                'avg_pnl_impact': round(float(np.mean(down_losses)), 2),
                'worst_pnl_impact': round(float(min(down_losses)), 2),
                'best_pnl_impact': round(float(max(down_losses)), 2),
            },
        }

        print(f"      Avg affected positions: {np.mean(up_affected + down_affected):.1f}")
        print(f"      Up gap:   avg impact=${np.mean(up_losses):+.2f}, "
              f"worst=${min(up_losses):+.2f}, best=${max(up_losses):+.2f}")
        print(f"      Down gap: avg impact=${np.mean(down_losses):+.2f}, "
              f"worst=${min(down_losses):+.2f}, best=${max(down_losses):+.2f}")

    results['synthetic_gaps'] = gap_results

    # ═══════════════════════════════════════════════════════════
    # Phase 7: Cross-Scenario Vulnerability Analysis
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"  Phase 7: Cross-Scenario Vulnerability Analysis")
    print(f"{'=' * 70}")

    # PnL matrix: rows = strategies, cols = scenarios
    sc_names = [s[0] for s in SCENARIOS]

    print(f"\n    PnL Matrix (unit lot):")
    header = f"    {'Strategy':<10}" + "".join(f"{s:>16}" for s in sc_names)
    print(header)
    print(f"    {'-' * (10 + 16 * len(sc_names))}")

    vulnerability_scores = {}
    for name in STRAT_ORDER:
        row_vals = [scenario_pnl_matrix.get(name, {}).get(s, 0.0) for s in sc_names]
        row_str = f"    {name:<10}" + "".join(f"${v:>14.2f}" for v in row_vals)
        print(row_str)

        neg_count = sum(1 for v in row_vals if v < 0)
        avg_pnl = np.mean(row_vals) if row_vals else 0
        vulnerability_scores[name] = {
            'scenario_pnls': {s: round(v, 2) for s, v in zip(sc_names, row_vals)},
            'negative_scenarios': neg_count,
            'avg_crisis_pnl': round(float(avg_pnl), 2),
            'total_crisis_pnl': round(float(sum(row_vals)), 2),
        }

    # Consistently weak strategy
    weakest_overall = min(vulnerability_scores,
                          key=lambda k: vulnerability_scores[k]['total_crisis_pnl'])
    strongest_overall = max(vulnerability_scores,
                            key=lambda k: vulnerability_scores[k]['total_crisis_pnl'])

    print(f"\n    Weakest overall: {weakest_overall} "
          f"(total crisis PnL=${vulnerability_scores[weakest_overall]['total_crisis_pnl']:.2f})")
    print(f"    Strongest overall: {strongest_overall} "
          f"(total crisis PnL=${vulnerability_scores[strongest_overall]['total_crisis_pnl']:.2f})")

    # Crisis beta: correlation of strategy PnL with gold return during crisis
    print(f"\n    Crisis Beta (correlation with gold return):")
    crisis_betas = {}
    gold_rets = np.array([scenario_gold_returns.get(s, 0) for s in sc_names])

    for name in STRAT_ORDER:
        strat_pnls = np.array([scenario_pnl_matrix.get(name, {}).get(s, 0) for s in sc_names])
        if len(strat_pnls) >= 3 and np.std(strat_pnls) > 0 and np.std(gold_rets) > 0:
            beta = float(np.corrcoef(strat_pnls, gold_rets)[0, 1])
        else:
            beta = 0.0
        crisis_betas[name] = round(beta, 3)
        print(f"      {name:<10}: beta={beta:+.3f}")

    vulnerability_scores_out = {}
    for name in STRAT_ORDER:
        vulnerability_scores_out[name] = {
            **vulnerability_scores[name],
            'crisis_beta': crisis_betas[name],
        }

    # Recommendations
    print(f"\n    Recommendations:")
    recommendations = {}
    for name in STRAT_ORDER:
        vs = vulnerability_scores[name]
        beta = crisis_betas[name]
        neg = vs['negative_scenarios']
        total = vs['total_crisis_pnl']

        if neg >= 4:
            rec = "HIGH RISK: Loses money in most crises. Consider reducing allocation or adding hedging."
        elif neg >= 3:
            rec = "MODERATE RISK: Vulnerable in majority of crises. Monitor closely during volatility spikes."
        elif total < 0:
            rec = "SLIGHT RISK: Net negative during crises but not consistently. Acceptable with portfolio diversification."
        else:
            rec = "RESILIENT: Performs well during crises. Good portfolio hedge component."

        if abs(beta) > 0.7:
            rec += f" Strong {'positive' if beta > 0 else 'negative'} gold correlation (beta={beta:+.3f})."
        elif abs(beta) > 0.3:
            rec += f" Moderate {'positive' if beta > 0 else 'negative'} gold correlation (beta={beta:+.3f})."

        recommendations[name] = rec
        print(f"      {name}: {rec}")

    results['vulnerability'] = {
        'pnl_matrix': {name: vulnerability_scores[name]['scenario_pnls'] for name in STRAT_ORDER},
        'strategy_analysis': vulnerability_scores_out,
        'crisis_betas': crisis_betas,
        'gold_returns_pct': {s: round(v, 2) for s, v in scenario_gold_returns.items()},
        'weakest_strategy': weakest_overall,
        'strongest_strategy': strongest_overall,
        'recommendations': recommendations,
    }

    # ═══════════════════════════════════════════════════════════
    # Final summary and save
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n    Scenario Results:")
    for sc_name, sc_data in results['scenarios'].items():
        port = sc_data.get('portfolio', {})
        wl = sc_data.get('weakest_link', '?')
        print(f"      {sc_name:<25} Gold={sc_data['gold_change_pct']:+.1f}%, "
              f"Portfolio PnL=${port.get('pnl', 0):+.2f}, "
              f"MaxDD=${port.get('max_dd', 0):.2f}, Weakest={wl}")

    print(f"\n    Gap Test Summary:")
    for label, gd in gap_results.items():
        print(f"      {label} gap: up_impact=${gd['up_gap']['avg_pnl_impact']:+.2f}, "
              f"down_impact=${gd['down_gap']['avg_pnl_impact']:+.2f}")

    print(f"\n    Strategy Crisis Resilience Ranking:")
    ranked = sorted(STRAT_ORDER,
                    key=lambda s: vulnerability_scores[s]['total_crisis_pnl'], reverse=True)
    for i, name in enumerate(ranked, 1):
        vs = vulnerability_scores[name]
        print(f"      #{i} {name:<10}: total=${vs['total_crisis_pnl']:+.2f}, "
              f"neg_scenarios={vs['negative_scenarios']}/5, "
              f"beta={crisis_betas[name]:+.3f}")

    out_file = OUTPUT_DIR / "r110_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 80}")
    print(f"  R110 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Saved: {out_file}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
