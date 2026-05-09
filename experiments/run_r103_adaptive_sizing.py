#!/usr/bin/env python3
"""
R103 — Adaptive Position Sizing
=================================
Tests 5 dynamic lot-sizing methods vs fixed R89 lots:

  Phase 1: Vol-Target (scale lot by target_vol / realized_vol)
  Phase 2: Half-Kelly (rolling Kelly fraction from trailing trades)
  Phase 3: ATR-Scaled (lot inversely proportional to ATR)
  Phase 4: Regime-Conditional (different multipliers per macro regime)
  Phase 5: Composite best + K-Fold validation
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import DataBundle, load_m15, load_h1_aligned, H1_CSV_PATH
from backtest.runner import run_variant, LIVE_PARITY_KWARGS

OUTPUT_DIR = Path("results/r103_adaptive_sizing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01
CAPITAL = 5000

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]

t0 = time.time()

# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _mk(pos, exit_px, exit_time, reason, bar_i, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_px,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_i - pos['bar']}

def _run_exit_with_cap(pos, i, hi, lo, cl, spread, lot, pv, times,
                       sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap):
    atr = pos['atr']
    sl = atr * sl_atr; tp = atr * tp_atr; bars = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_now = (cl - pos['entry'] - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if lo <= pos['entry'] - sl:
            return _mk(pos, pos['entry'] - sl, times[i], "SL", i, -sl * lot * pv)
        if hi >= pos['entry'] + tp:
            return _mk(pos, pos['entry'] + tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry']); extreme = max(extreme, hi); pos['extreme'] = extreme
        if extreme - pos['entry'] >= atr * trail_act:
            trail_price = extreme - atr * trail_dist
            if cl <= trail_price:
                return _mk(pos, trail_price, times[i], "Trail", i, (trail_price - pos['entry'] - spread) * lot * pv)
        if bars >= max_hold:
            return _mk(pos, cl, times[i], "TimeExit", i, pnl_now)
    else:
        pnl_now = (pos['entry'] - cl - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if hi >= pos['entry'] + sl:
            return _mk(pos, pos['entry'] + sl, times[i], "SL", i, -sl * lot * pv)
        if lo <= pos['entry'] - tp:
            return _mk(pos, pos['entry'] - tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry']); extreme = min(extreme, lo); pos['extreme'] = extreme
        if pos['entry'] - extreme >= atr * trail_act:
            trail_price = extreme + atr * trail_dist
            if cl >= trail_price:
                return _mk(pos, trail_price, times[i], "Trail", i, (pos['entry'] - trail_price - spread) * lot * pv)
        if bars >= max_hold:
            return _mk(pos, cl, times[i], "TimeExit", i, pnl_now)
    return None


# ═══════════════════════════════════════════════════════════════
# Strategy backtests
# ═══════════════════════════════════════════════════════════════

def add_psar(df, af_step=0.01, af_max=0.05):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    n = len(df)
    psar = np.empty(n); psar[:] = np.nan
    af = af_step; rising = True; ep = h[0]; psar[0] = l[0]
    for i in range(1, n):
        prev = psar[i-1]
        if rising:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], l[i-1], l[max(0,i-2)])
            if l[i] < psar[i]:
                rising = False; psar[i] = ep; ep = l[i]; af = af_step
            else:
                if h[i] > ep: ep = h[i]; af = min(af + af_step, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], h[i-1], h[max(0,i-2)])
            if h[i] > psar[i]:
                rising = True; psar[i] = ep; ep = h[i]; af = af_step
            else:
                if l[i] < ep: ep = l[i]; af = min(af + af_step, af_max)
    df['PSAR'] = psar

def bt_psar(h1_df, spread, lot, maxloss_cap=0,
            sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04, max_hold=20):
    df = h1_df.copy(); add_psar(df); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR', 'PSAR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; psar = df['PSAR'].values; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if psar[i-1] > c[i-1] and psar[i] < c[i]:
            pos = {'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif psar[i-1] < c[i-1] and psar[i] > c[i]:
            pos = {'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
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
            pos = {'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
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
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif c[i] < ll:
            pos = {'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
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
    if not trades: return pd.Series(dtype=float)
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

def build_portfolio_daily(unit_dailies, lots):
    all_dates = set()
    for ds in unit_dailies.values():
        all_dates.update(ds.index)
    all_dates = sorted(all_dates)
    idx = pd.DatetimeIndex(all_dates)
    portfolio = np.zeros(len(idx))
    for name in STRAT_ORDER:
        if name not in unit_dailies: continue
        ds = unit_dailies[name]
        multiplier = lots[name] / UNIT_LOT
        aligned = ds.reindex(idx, fill_value=0.0).values * multiplier
        portfolio += aligned
    return portfolio

def portfolio_metrics(daily_arr):
    return {
        'sharpe': round(sharpe(daily_arr), 3),
        'pnl': round(float(np.sum(daily_arr)), 2),
        'max_dd': round(max_dd(daily_arr), 2),
    }


# ═══════════════════════════════════════════════════════════════
# Sizing methods
# ═══════════════════════════════════════════════════════════════

def apply_vol_target_scaling(trades, target_vol_ann, window=20):
    """Scale each trade's PnL by target_vol / realized_vol on entry date."""
    if not trades:
        return []
    daily_s = trades_to_daily_series(trades)
    if len(daily_s) < window + 1:
        return trades

    rolling_vol = daily_s.rolling(window).std() * np.sqrt(252)
    target_daily = target_vol_ann / np.sqrt(252)

    scaled = []
    for t in trades:
        entry_d = pd.Timestamp(t['entry_time']).date()
        idx = daily_s.index.searchsorted(pd.Timestamp(entry_d))
        idx = min(idx, len(rolling_vol) - 1)
        rv = rolling_vol.iloc[idx] if idx >= window else np.nan
        if np.isnan(rv) or rv < 1e-6:
            mult = 1.0
        else:
            mult = np.clip(target_daily / (rv / np.sqrt(252)), 0.5, 2.0)
        st = dict(t)
        st['pnl'] = t['pnl'] * mult
        scaled.append(st)
    return scaled


def apply_half_kelly_scaling(trades, trail_n=120):
    """Scale PnL by half-Kelly fraction from trailing N trades, clamped [0.5x, 2.0x]."""
    if not trades:
        return []
    scaled = []
    for idx, t in enumerate(trades):
        lookback = trades[max(0, idx - trail_n):idx]
        if len(lookback) < 20:
            mult = 1.0
        else:
            wins = [x['pnl'] for x in lookback if x['pnl'] > 0]
            losses = [x['pnl'] for x in lookback if x['pnl'] <= 0]
            w_rate = len(wins) / len(lookback) if lookback else 0.5
            avg_w = np.mean(wins) if wins else 1.0
            avg_l = abs(np.mean(losses)) if losses else 1.0
            r = avg_w / avg_l if avg_l > 0 else 1.0
            kelly = w_rate - (1 - w_rate) / r if r > 0 else 0.0
            mult = np.clip(kelly / 2.0, 0.5, 2.0)
        st = dict(t)
        st['pnl'] = t['pnl'] * mult
        scaled.append(st)
    return scaled


def apply_atr_scaling(trades, h1_df, base_lot):
    """Lot inversely proportional to ATR(14). Normalize so average = base_lot.
    Clamp multiplier to [0.5x, 2.0x]."""
    if not trades:
        return []
    atr_series = compute_atr(h1_df)
    mean_atr = atr_series.mean()
    if np.isnan(mean_atr) or mean_atr < 0.01:
        return trades

    scaled = []
    for t in trades:
        entry_ts = pd.Timestamp(t['entry_time'])
        idx = atr_series.index.searchsorted(entry_ts)
        idx = min(max(idx - 1, 0), len(atr_series) - 1)
        atr_val = atr_series.iloc[idx]
        if np.isnan(atr_val) or atr_val < 0.01:
            mult = 1.0
        else:
            mult = np.clip(mean_atr / atr_val, 0.5, 2.0)
        st = dict(t)
        st['pnl'] = t['pnl'] * mult
        scaled.append(st)
    return scaled


def load_regime_data():
    """Load VIX data and compute regime percentile ranks."""
    csv_path = Path("data/external/aligned_daily.csv")
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
    vix = df['VIX_Close'].dropna()
    if len(vix) < 100:
        return None
    pct_rank = vix.rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    regime = pd.Series('Med', index=pct_rank.index)
    regime[pct_rank <= 0.33] = 'Low'
    regime[pct_rank >= 0.67] = 'High'
    return regime


def apply_regime_scaling(trades, regime_series, mult_low, mult_med, mult_high):
    """Scale PnL by regime-conditional multiplier."""
    if not trades or regime_series is None:
        return trades
    regime_map = {'Low': mult_low, 'Med': mult_med, 'High': mult_high}
    scaled = []
    for t in trades:
        entry_d = pd.Timestamp(t['entry_time']).date()
        ts = pd.Timestamp(entry_d)
        idx = regime_series.index.searchsorted(ts)
        idx = min(max(idx - 1, 0), len(regime_series) - 1)
        reg = regime_series.iloc[idx]
        mult = np.clip(regime_map.get(reg, 1.0), 0.3, 2.0)
        st = dict(t)
        st['pnl'] = t['pnl'] * mult
        scaled.append(st)
    return scaled


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  R103 — Adaptive Position Sizing")
    print("=" * 80)

    print("\n  Loading data...")
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    bundle = DataBundle.load_custom()
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0].date()} ~ {h1_df.index[-1].date()})")

    # ─── Run base strategies at unit lot ─────────────────────────
    print("\n  Running 4 strategies at unit lot (0.01)...")
    base_trades = {}
    base_trades['PSAR'] = bt_psar(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['PSAR'])
    base_trades['TSMOM'] = bt_tsmom(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['TSMOM'])
    base_trades['SESS_BO'] = bt_sess_bo(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['SESS_BO'])
    base_trades['L8_MAX'] = bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])

    unit_dailies = {}
    for name in STRAT_ORDER:
        unit_dailies[name] = trades_to_daily_series(base_trades[name])
        n_t = len(base_trades[name])
        pnl = sum(t['pnl'] for t in base_trades[name])
        print(f"    {name:10s}: {n_t:5d} trades, unit PnL=${pnl:,.2f}")

    # Fixed-lot baseline
    fixed_daily = build_portfolio_daily(unit_dailies, R89_LOTS)
    fixed_m = portfolio_metrics(fixed_daily)
    print(f"\n  Fixed R89 Baseline: Sharpe={fixed_m['sharpe']}, "
          f"PnL=${fixed_m['pnl']:,.2f}, MaxDD=${fixed_m['max_dd']:,.2f}")

    results = {'experiment': 'R103 Adaptive Position Sizing',
               'fixed_baseline': fixed_m}

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Vol-Target Sizing
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 1: Vol-Target Sizing")
    print("=" * 60)

    vol_targets = [0.10, 0.15, 0.20]
    phase1 = {}
    for tv in vol_targets:
        scaled_dailies = {}
        for name in STRAT_ORDER:
            scaled_trades = apply_vol_target_scaling(base_trades[name], tv)
            multiplier = R89_LOTS[name] / UNIT_LOT
            ds = trades_to_daily_series(scaled_trades)
            scaled_dailies[name] = ds * multiplier
        all_dates = set()
        for ds in scaled_dailies.values():
            all_dates.update(ds.index)
        all_dates = sorted(all_dates)
        idx = pd.DatetimeIndex(all_dates)
        port = np.zeros(len(idx))
        for name in STRAT_ORDER:
            if name not in scaled_dailies: continue
            port += scaled_dailies[name].reindex(idx, fill_value=0.0).values
        m = portfolio_metrics(port)
        delta_sh = m['sharpe'] - fixed_m['sharpe']
        phase1[f"vol_{int(tv*100)}pct"] = m
        print(f"    Target={tv*100:.0f}%: Sharpe={m['sharpe']:6.3f} (Δ={delta_sh:+.3f}), "
              f"PnL=${m['pnl']:,.2f}, MaxDD=${m['max_dd']:,.2f}")

    results['phase1_vol_target'] = phase1
    best_p1_key = max(phase1, key=lambda k: phase1[k]['sharpe'])
    best_p1_sharpe = phase1[best_p1_key]['sharpe']
    print(f"  -> Best: {best_p1_key} (Sharpe={best_p1_sharpe:.3f})")

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Half-Kelly Sizing
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 2: Half-Kelly Sizing")
    print("=" * 60)

    kelly_windows = [60, 120, 240]
    phase2 = {}
    for kw in kelly_windows:
        scaled_dailies = {}
        for name in STRAT_ORDER:
            scaled_trades = apply_half_kelly_scaling(base_trades[name], trail_n=kw)
            multiplier = R89_LOTS[name] / UNIT_LOT
            ds = trades_to_daily_series(scaled_trades)
            scaled_dailies[name] = ds * multiplier
        all_dates = set()
        for ds in scaled_dailies.values():
            all_dates.update(ds.index)
        all_dates = sorted(all_dates)
        idx = pd.DatetimeIndex(all_dates)
        port = np.zeros(len(idx))
        for name in STRAT_ORDER:
            if name not in scaled_dailies: continue
            port += scaled_dailies[name].reindex(idx, fill_value=0.0).values
        m = portfolio_metrics(port)
        delta_sh = m['sharpe'] - fixed_m['sharpe']
        phase2[f"kelly_N{kw}"] = m
        print(f"    N={kw:3d}: Sharpe={m['sharpe']:6.3f} (Δ={delta_sh:+.3f}), "
              f"PnL=${m['pnl']:,.2f}, MaxDD=${m['max_dd']:,.2f}")

    results['phase2_half_kelly'] = phase2
    best_p2_key = max(phase2, key=lambda k: phase2[k]['sharpe'])
    best_p2_sharpe = phase2[best_p2_key]['sharpe']
    print(f"  -> Best: {best_p2_key} (Sharpe={best_p2_sharpe:.3f})")

    # ═══════════════════════════════════════════════════════════
    # Phase 3: ATR-Scaled Sizing
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 3: ATR-Scaled Sizing")
    print("=" * 60)

    phase3 = {}
    scaled_dailies = {}
    for name in STRAT_ORDER:
        scaled_trades = apply_atr_scaling(base_trades[name], h1_df, R89_LOTS[name])
        multiplier = R89_LOTS[name] / UNIT_LOT
        ds = trades_to_daily_series(scaled_trades)
        scaled_dailies[name] = ds * multiplier
    all_dates = set()
    for ds in scaled_dailies.values():
        all_dates.update(ds.index)
    all_dates = sorted(all_dates)
    idx = pd.DatetimeIndex(all_dates)
    port = np.zeros(len(idx))
    for name in STRAT_ORDER:
        if name not in scaled_dailies: continue
        port += scaled_dailies[name].reindex(idx, fill_value=0.0).values
    m = portfolio_metrics(port)
    delta_sh = m['sharpe'] - fixed_m['sharpe']
    phase3['atr_scaled'] = m
    print(f"    ATR-Scaled: Sharpe={m['sharpe']:6.3f} (Δ={delta_sh:+.3f}), "
          f"PnL=${m['pnl']:,.2f}, MaxDD=${m['max_dd']:,.2f}")

    results['phase3_atr_scaled'] = phase3
    best_p3_sharpe = m['sharpe']
    print(f"  -> ATR-Scaled Sharpe={best_p3_sharpe:.3f}")

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Regime-Conditional Sizing
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 4: Regime-Conditional Sizing")
    print("=" * 60)

    regime_series = load_regime_data()
    phase4 = {}
    if regime_series is None:
        print("  *** No VIX data found — using ATR percentile rank as regime proxy ***")
        atr_daily = h1_df['Close'].resample('D').last().dropna()
        atr_d = compute_atr(pd.DataFrame({
            'High': h1_df['High'].resample('D').max(),
            'Low': h1_df['Low'].resample('D').min(),
            'Close': atr_daily
        }).dropna())
        pct_rank = atr_d.rolling(252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        regime_series = pd.Series('Med', index=pct_rank.index)
        regime_series[pct_rank <= 0.33] = 'Low'
        regime_series[pct_rank >= 0.67] = 'High'

    regime_counts = regime_series.value_counts()
    print(f"  Regime distribution: {dict(regime_counts)}")

    low_grid = [0.8, 1.0, 1.2, 1.5]
    med_grid = [0.6, 0.8, 1.0, 1.2]
    high_grid = [0.3, 0.5, 0.7, 1.0]

    total_combos = len(low_grid) * len(med_grid) * len(high_grid)
    print(f"  Grid: {total_combos} combinations...")

    best_regime = None
    best_regime_sharpe = -999
    combo_count = 0
    for ml, mm, mh in product(low_grid, med_grid, high_grid):
        combo_count += 1
        scaled_dailies = {}
        for name in STRAT_ORDER:
            scaled_trades = apply_regime_scaling(base_trades[name], regime_series, ml, mm, mh)
            multiplier = R89_LOTS[name] / UNIT_LOT
            ds = trades_to_daily_series(scaled_trades)
            scaled_dailies[name] = ds * multiplier
        all_dates = set()
        for ds in scaled_dailies.values():
            all_dates.update(ds.index)
        all_dates = sorted(all_dates)
        idx_r = pd.DatetimeIndex(all_dates)
        port = np.zeros(len(idx_r))
        for name in STRAT_ORDER:
            if name not in scaled_dailies: continue
            port += scaled_dailies[name].reindex(idx_r, fill_value=0.0).values
        m = portfolio_metrics(port)
        key = f"L{ml}_M{mm}_H{mh}"
        phase4[key] = m
        if m['sharpe'] > best_regime_sharpe:
            best_regime_sharpe = m['sharpe']
            best_regime = {'low': ml, 'med': mm, 'high': mh, **m}
        if combo_count % 16 == 0:
            print(f"    {combo_count}/{total_combos}... best so far: Sharpe={best_regime_sharpe:.3f}")

    delta_sh = best_regime_sharpe - fixed_m['sharpe']
    print(f"\n  Best Regime: Low={best_regime['low']}, Med={best_regime['med']}, "
          f"High={best_regime['high']}")
    print(f"    Sharpe={best_regime_sharpe:6.3f} (Δ={delta_sh:+.3f}), "
          f"PnL=${best_regime['pnl']:,.2f}, MaxDD=${best_regime['max_dd']:,.2f}")

    results['phase4_regime'] = {'best': best_regime, 'grid_size': total_combos}

    # ═══════════════════════════════════════════════════════════
    # Phase 5: Composite Best + K-Fold Validation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 5: Composite Best + K-Fold Validation")
    print("=" * 60)

    # Determine best method from Phases 1-4
    candidates = []
    candidates.append(('vol_target', best_p1_key, best_p1_sharpe))
    candidates.append(('half_kelly', best_p2_key, best_p2_sharpe))
    candidates.append(('atr_scaled', 'atr_scaled', best_p3_sharpe))
    candidates.append(('regime', 'best_regime', best_regime_sharpe))

    candidates.sort(key=lambda x: x[2], reverse=True)
    winner_method = candidates[0][0]
    winner_label = candidates[0][1]
    winner_sharpe = candidates[0][2]

    print(f"\n  Phase ranking:")
    for rank, (method, label, sh) in enumerate(candidates):
        marker = " <-- WINNER" if rank == 0 else ""
        print(f"    #{rank+1}: {method:15s} ({label:20s}) Sharpe={sh:6.3f}{marker}")

    print(f"\n  K-Fold validation on winner: {winner_method}")

    kfold_fixed = []
    kfold_adaptive = []

    for fname, start, end in FOLDS:
        fold_h1 = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
        if len(fold_h1) < 100:
            kfold_fixed.append(0.0)
            kfold_adaptive.append(0.0)
            continue

        fold_trades = {}
        fold_trades['PSAR'] = bt_psar(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=CAPS['PSAR'])
        fold_trades['TSMOM'] = bt_tsmom(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=CAPS['TSMOM'])
        fold_trades['SESS_BO'] = bt_sess_bo(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=CAPS['SESS_BO'])
        fold_trades['L8_MAX'] = bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])
        fold_trades['L8_MAX'] = [t for t in fold_trades['L8_MAX']
                                 if start <= str(pd.Timestamp(t['exit_time']).date()) < end]

        # Fixed baseline for this fold
        fold_unit_dailies = {n: trades_to_daily_series(fold_trades[n]) for n in STRAT_ORDER}
        fold_fixed = build_portfolio_daily(fold_unit_dailies, R89_LOTS)
        kfold_fixed.append(sharpe(fold_fixed))

        # Adaptive for this fold — train on OTHER folds, apply on this fold
        train_trades = {}
        for tname, tstart, tend in FOLDS:
            if tname == fname:
                continue
            for name in STRAT_ORDER:
                if name not in train_trades:
                    train_trades[name] = []
                train_trades[name].extend([
                    t for t in base_trades[name]
                    if tstart <= str(pd.Timestamp(t['exit_time']).date()) < tend
                ])

        # Apply the winning method with parameters learned from training set
        adaptive_dailies = {}
        for name in STRAT_ORDER:
            if winner_method == 'vol_target':
                tv_pct = int(best_p1_key.split('_')[1].replace('pct', ''))
                tv = tv_pct / 100.0
                scaled = apply_vol_target_scaling(fold_trades[name], tv)
            elif winner_method == 'half_kelly':
                kw_n = int(best_p2_key.split('N')[1])
                # Use train trades to seed the lookback, then apply to fold
                seeded = train_trades.get(name, []) + fold_trades[name]
                n_train = len(train_trades.get(name, []))
                all_scaled = apply_half_kelly_scaling(seeded, trail_n=kw_n)
                scaled = all_scaled[n_train:]
            elif winner_method == 'atr_scaled':
                scaled = apply_atr_scaling(fold_trades[name], fold_h1, R89_LOTS[name])
            elif winner_method == 'regime':
                ml = best_regime['low']
                mm = best_regime['med']
                mh = best_regime['high']
                scaled = apply_regime_scaling(fold_trades[name], regime_series, ml, mm, mh)
            else:
                scaled = fold_trades[name]

            multiplier = R89_LOTS[name] / UNIT_LOT
            ds = trades_to_daily_series(scaled)
            adaptive_dailies[name] = ds * multiplier

        all_dates = set()
        for ds in adaptive_dailies.values():
            all_dates.update(ds.index)
        all_dates = sorted(all_dates)
        idx_f = pd.DatetimeIndex(all_dates)
        port = np.zeros(len(idx_f))
        for name in STRAT_ORDER:
            if name not in adaptive_dailies: continue
            port += adaptive_dailies[name].reindex(idx_f, fill_value=0.0).values
        kfold_adaptive.append(sharpe(port))

    print(f"\n  K-Fold Results:")
    print(f"  {'Fold':<8} {'Fixed':>10} {'Adaptive':>10} {'Delta':>10}")
    print(f"  {'-'*40}")
    for i, (fname, _, _) in enumerate(FOLDS):
        delta = kfold_adaptive[i] - kfold_fixed[i]
        print(f"  {fname:<8} {kfold_fixed[i]:10.3f} {kfold_adaptive[i]:10.3f} {delta:+10.3f}")

    mean_fixed = np.mean(kfold_fixed)
    mean_adaptive = np.mean(kfold_adaptive)
    adaptive_wins = sum(1 for a, f in zip(kfold_adaptive, kfold_fixed) if a > f)

    print(f"  {'-'*40}")
    print(f"  {'Mean':<8} {mean_fixed:10.3f} {mean_adaptive:10.3f} {mean_adaptive-mean_fixed:+10.3f}")
    print(f"\n  Adaptive wins: {adaptive_wins}/6 folds")

    kfold_pass = adaptive_wins >= 4

    results['phase5_kfold'] = {
        'winner_method': winner_method,
        'winner_label': winner_label,
        'fixed_folds': [round(s, 3) for s in kfold_fixed],
        'adaptive_folds': [round(s, 3) for s in kfold_adaptive],
        'fixed_mean': round(mean_fixed, 3),
        'adaptive_mean': round(mean_adaptive, 3),
        'adaptive_wins': adaptive_wins,
        'pass_4of6': kfold_pass,
    }

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    print("\n" + "=" * 80)
    print("  R103 SUMMARY — Adaptive Position Sizing")
    print("=" * 80)

    print(f"\n  Fixed R89 Baseline:  Sharpe={fixed_m['sharpe']:.3f}, "
          f"PnL=${fixed_m['pnl']:,.2f}, MaxDD=${fixed_m['max_dd']:,.2f}")

    print(f"\n  {'Method':<20} {'Sharpe':>8} {'Delta':>8} {'PnL':>12} {'MaxDD':>10}")
    print(f"  {'-'*60}")
    summary_rows = [
        ('Vol-Target', best_p1_key, phase1[best_p1_key]),
        ('Half-Kelly', best_p2_key, phase2[best_p2_key]),
        ('ATR-Scaled', 'atr_scaled', phase3['atr_scaled']),
        ('Regime-Cond', 'best', {'sharpe': best_regime_sharpe,
                                  'pnl': best_regime['pnl'],
                                  'max_dd': best_regime['max_dd']}),
    ]
    for label, key, m in summary_rows:
        d = m['sharpe'] - fixed_m['sharpe']
        print(f"  {label:<20} {m['sharpe']:8.3f} {d:+8.3f} ${m['pnl']:>10,.2f} ${m['max_dd']:>8,.2f}")

    if kfold_pass:
        print(f"\n  RECOMMENDATION: Use {winner_method} sizing "
              f"(wins {adaptive_wins}/6 folds, mean Sharpe {mean_adaptive:.3f} vs {mean_fixed:.3f})")
    else:
        print(f"\n  RECOMMENDATION: Keep fixed R89 lots "
              f"(adaptive wins only {adaptive_wins}/6 folds)")

    results['elapsed_s'] = round(elapsed, 1)
    results['recommendation'] = (
        f"Use {winner_method} sizing (wins {adaptive_wins}/6 folds)"
        if kfold_pass
        else "Keep fixed R89 lots"
    )

    out_file = OUTPUT_DIR / "r103_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Saved: {out_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
