#!/usr/bin/env python3
"""
R96 — Slippage & Execution Cost Analysis
==========================================
Tests how different spread assumptions affect strategy performance:
1. Fixed spreads: 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.70, 1.00
2. All 4 strategies at each spread level
3. Sharpe degradation per $0.10 additional spread (linear regression)
4. Spike spread model: L8_MAX with random +0.5 spread on 10% of trades
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r96_slippage")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}
SPREAD_LEVELS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.70, 1.00]


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs()
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


def _mk(pos, exit_price, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_price,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


def _run_exit_with_cap(pos, i, h, lo_v, c, spread, lot, pv, times,
                       sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold, maxloss_cap=0):
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
    if pnl_h >= tp_val: return _mk(pos, c, times[i], "TP", i, tp_val)
    if pnl_l <= -sl_val: return _mk(pos, c, times[i], "SL", i, -sl_val)
    if maxloss_cap > 0 and pnl_c < -maxloss_cap:
        return _mk(pos, c, times[i], "MaxLossCap", i, -maxloss_cap)
    ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
    if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
        ts_p = h - td
        if lo_v <= ts_p: return _mk(pos, c, times[i], "Trail", i, (ts_p - pos['entry'] - spread) * lot * pv)
    elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
        ts_p = lo_v + td
        if h >= ts_p: return _mk(pos, c, times[i], "Trail", i, (pos['entry'] - ts_p - spread) * lot * pv)
    if held >= max_hold: return _mk(pos, c, times[i], "Timeout", i, pnl_c)
    return None


# ═══════════════════════════════════════════════════════════════
# Strategy backtests
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


def bt_l8_max(data_bundle, spread, lot, maxloss_cap=0):
    from backtest.runner import run_variant, LIVE_PARITY_KWARGS
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
# Stats helpers
# ═══════════════════════════════════════════════════════════════

def trades_to_daily_series(trades):
    if not trades: return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        ts = pd.Timestamp(t['exit_time'])
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        d = ts.normalize()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))

def sharpe(daily_pnl):
    if len(daily_pnl) < 10: return 0.0
    arr = np.array(daily_pnl) if not isinstance(daily_pnl, np.ndarray) else daily_pnl
    if arr.std() == 0: return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(252))

def max_dd(daily_pnl):
    arr = np.array(daily_pnl) if not isinstance(daily_pnl, np.ndarray) else daily_pnl
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(dd.max()) if len(dd) > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("  R96 — Slippage & Execution Cost Analysis", flush=True)
    print(f"  Spread levels: {SPREAD_LEVELS}", flush=True)
    print("=" * 80, flush=True)

    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH, DataBundle

    print("\n  Loading data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    bundle = DataBundle.load_custom()
    print(f"  H1: {len(h1_df)} bars", flush=True)

    results = {}

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Fixed spread sweep — all 4 strategies
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 1: Fixed Spread Sweep", flush=True)
    print("=" * 70, flush=True)

    strat_names = ['PSAR', 'TSMOM', 'SESS_BO', 'L8_MAX']
    spread_results = {s: {} for s in strat_names}

    for sp in SPREAD_LEVELS:
        print(f"\n    Spread = ${sp:.2f}", flush=True)

        for sname in ['PSAR', 'TSMOM', 'SESS_BO']:
            fn = {'PSAR': bt_psar, 'TSMOM': bt_tsmom, 'SESS_BO': bt_sess_bo}[sname]
            trades = fn(h1_df, spread=sp, lot=UNIT_LOT, maxloss_cap=CAPS[sname])
            ds = trades_to_daily_series(trades)
            sh = sharpe(ds)
            dd = max_dd(ds)
            pnl = sum(t['pnl'] for t in trades)
            spread_results[sname][f"sp_{sp:.2f}"] = {
                'spread': sp, 'n_trades': len(trades),
                'sharpe': round(sh, 3), 'max_dd': round(dd, 2), 'pnl': round(pnl, 2),
            }
            print(f"      {sname:>8}: {len(trades)} trades, Sharpe={sh:.3f}, PnL=${pnl:.0f}, MaxDD=${dd:.0f}", flush=True)

        trades = bt_l8_max(bundle, spread=sp, lot=UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])
        ds = trades_to_daily_series(trades)
        sh = sharpe(ds)
        dd = max_dd(ds)
        pnl = sum(t['pnl'] for t in trades)
        spread_results['L8_MAX'][f"sp_{sp:.2f}"] = {
            'spread': sp, 'n_trades': len(trades),
            'sharpe': round(sh, 3), 'max_dd': round(dd, 2), 'pnl': round(pnl, 2),
        }
        print(f"      {'L8_MAX':>8}: {len(trades)} trades, Sharpe={sh:.3f}, PnL=${pnl:.0f}, MaxDD=${dd:.0f}", flush=True)

    results['spread_sweep'] = spread_results

    # ══════════════════════════════════════════════════════════════
    # Phase 2: Sharpe degradation per $0.10 spread (linear regression)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: Sharpe Degradation Analysis", flush=True)
    print("=" * 70, flush=True)

    degradation = {}
    for sname in strat_names:
        spreads_arr = np.array([v['spread'] for v in spread_results[sname].values()])
        sharpes_arr = np.array([v['sharpe'] for v in spread_results[sname].values()])

        if len(spreads_arr) >= 2:
            coeffs = np.polyfit(spreads_arr, sharpes_arr, 1)
            slope = coeffs[0]
            degradation_per_010 = slope * 0.10
            r_squared = 1 - np.sum((sharpes_arr - np.polyval(coeffs, spreads_arr))**2) / \
                        max(np.sum((sharpes_arr - sharpes_arr.mean())**2), 1e-10)
        else:
            degradation_per_010 = 0.0
            r_squared = 0.0

        degradation[sname] = {
            'slope_per_dollar': round(float(slope), 4) if len(spreads_arr) >= 2 else 0,
            'degradation_per_010': round(float(degradation_per_010), 4),
            'r_squared': round(float(r_squared), 4),
            'sharpe_at_030': round(float(np.polyval(coeffs, 0.30)), 3) if len(spreads_arr) >= 2 else 0,
            'breakeven_spread': round(float(-coeffs[1] / coeffs[0]), 2) if len(spreads_arr) >= 2 and coeffs[0] != 0 else None,
        }
        print(f"    {sname:>8}: Sharpe drop = {degradation_per_010:+.4f} per $0.10 spread "
              f"(R²={r_squared:.3f})", flush=True)
        if degradation[sname]['breakeven_spread'] is not None:
            print(f"             Breakeven spread: ${degradation[sname]['breakeven_spread']:.2f}", flush=True)

    results['degradation'] = degradation

    # ══════════════════════════════════════════════════════════════
    # Phase 3: Portfolio-level spread sensitivity at R89 lots
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3: Portfolio-Level Spread Sensitivity (R89 lots)", flush=True)
    print("=" * 70, flush=True)

    portfolio_by_spread = {}
    for sp in SPREAD_LEVELS:
        all_daily = {}
        for sname in ['PSAR', 'TSMOM', 'SESS_BO']:
            fn = {'PSAR': bt_psar, 'TSMOM': bt_tsmom, 'SESS_BO': bt_sess_bo}[sname]
            trades = fn(h1_df, spread=sp, lot=UNIT_LOT, maxloss_cap=CAPS[sname])
            all_daily[sname] = trades_to_daily_series(trades)

        l8_trades = bt_l8_max(bundle, spread=sp, lot=UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])
        all_daily['L8_MAX'] = trades_to_daily_series(l8_trades)

        all_dates = set()
        for ds in all_daily.values():
            all_dates.update(ds.index)
        all_dates = sorted(all_dates)
        idx = pd.DatetimeIndex(all_dates)

        portfolio = np.zeros(len(idx))
        for sname in strat_names:
            if sname not in all_daily:
                continue
            mult = R89_LOTS[sname] / UNIT_LOT
            aligned = all_daily[sname].reindex(idx, fill_value=0.0).values * mult
            portfolio += aligned

        sh = sharpe(portfolio)
        dd = max_dd(portfolio)
        pnl = float(np.sum(portfolio))
        portfolio_by_spread[f"sp_{sp:.2f}"] = {
            'spread': sp, 'sharpe': round(sh, 3), 'max_dd': round(dd, 2), 'pnl': round(pnl, 2),
        }
        print(f"    Spread=${sp:.2f}: Portfolio Sharpe={sh:.3f}, PnL=${pnl:.0f}, MaxDD=${dd:.0f}", flush=True)

    results['portfolio_spread'] = portfolio_by_spread

    # ══════════════════════════════════════════════════════════════
    # Phase 4: Spike spread model (L8_MAX only)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 4: Spike Spread Model (L8_MAX)", flush=True)
    print("=" * 70, flush=True)

    np.random.seed(42)
    base_trades = bt_l8_max(bundle, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])
    n_trades_l8 = len(base_trades)
    spike_extra = 0.50
    spike_pct = 0.10

    n_runs = 20
    spike_results = []
    for run_i in range(n_runs):
        spike_mask = np.random.random(n_trades_l8) < spike_pct
        modified_trades = []
        for j, t in enumerate(base_trades):
            mt = dict(t)
            if spike_mask[j]:
                mt['pnl'] = t['pnl'] - spike_extra * UNIT_LOT * PV
            modified_trades.append(mt)

        ds = trades_to_daily_series(modified_trades)
        spike_results.append({
            'sharpe': round(sharpe(ds), 3),
            'pnl': round(sum(mt['pnl'] for mt in modified_trades), 2),
            'max_dd': round(max_dd(ds), 2),
            'n_spiked': int(spike_mask.sum()),
        })

    base_ds = trades_to_daily_series(base_trades)
    base_sharpe = sharpe(base_ds)
    spike_sharpes = [r['sharpe'] for r in spike_results]
    avg_spike_sharpe = float(np.mean(spike_sharpes))

    spike_summary = {
        'base_spread': SPREAD,
        'spike_extra': spike_extra,
        'spike_pct': spike_pct,
        'n_runs': n_runs,
        'n_trades': n_trades_l8,
        'base_sharpe': round(base_sharpe, 3),
        'avg_spike_sharpe': round(avg_spike_sharpe, 3),
        'sharpe_impact': round(avg_spike_sharpe - base_sharpe, 4),
        'worst_spike_sharpe': round(float(np.min(spike_sharpes)), 3),
        'best_spike_sharpe': round(float(np.max(spike_sharpes)), 3),
        'runs': spike_results,
    }

    print(f"    Base L8_MAX Sharpe:   {base_sharpe:.3f}", flush=True)
    print(f"    Avg spike Sharpe:     {avg_spike_sharpe:.3f} (impact: {avg_spike_sharpe - base_sharpe:+.4f})", flush=True)
    print(f"    Worst spike Sharpe:   {np.min(spike_sharpes):.3f}", flush=True)
    print(f"    Spike config: +${spike_extra} on {spike_pct*100:.0f}% of {n_trades_l8} trades, {n_runs} MC runs", flush=True)

    results['spike_spread'] = spike_summary

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)

    print(f"\n{'='*80}", flush=True)
    print(f"  R96 — Summary", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"  Most spread-sensitive:  {max(degradation, key=lambda k: abs(degradation[k]['degradation_per_010']))}", flush=True)
    print(f"  Least spread-sensitive: {min(degradation, key=lambda k: abs(degradation[k]['degradation_per_010']))}", flush=True)
    for sname in strat_names:
        be = degradation[sname]['breakeven_spread']
        be_str = f"${be:.2f}" if be is not None else "N/A"
        print(f"    {sname:>8}: breakeven spread = {be_str}, "
              f"Sharpe@0.30 = {degradation[sname]['sharpe_at_030']:.3f}", flush=True)

    print(f"\n  R96 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"{'='*80}", flush=True)

    with open(OUTPUT_DIR / "r96_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r96_results.json", flush=True)


if __name__ == "__main__":
    main()
