#!/usr/bin/env python3
"""
R122 — COT Deep Analysis
==========================
Uses real CFTC COT data for gold futures positioning analysis.

Phase 1: Load COT data, compute features (z-scores, commercial ratio, OI momentum)
Phase 2: COT as standalone contrarian signal (grid over thresholds + hold periods)
Phase 3: COT as filter on existing 4 strategies
Phase 4: Walk-forward validation (2yr train / 6mo test)
Phase 5: K-Fold validation (5 folds)
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import run_variant, LIVE_PARITY_KWARGS, DataBundle

OUTPUT_DIR = Path("results/r122_cot_deep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01

FOLDS = [
    ("Fold1", "2015-01-01", "2017-03-01"),
    ("Fold2", "2017-03-01", "2019-06-01"),
    ("Fold3", "2019-06-01", "2021-09-01"),
    ("Fold4", "2021-09-01", "2023-12-01"),
    ("Fold5", "2023-12-01", "2027-01-01"),
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
    sl = atr * sl_atr
    tp = atr * tp_atr
    bars = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_now = (cl - pos['entry'] - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if lo <= pos['entry'] - sl:
            return _mk(pos, pos['entry'] - sl, times[i], "SL", i, -sl * lot * pv)
        if hi >= pos['entry'] + tp:
            return _mk(pos, pos['entry'] + tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry'])
        extreme = max(extreme, hi)
        pos['extreme'] = extreme
        act_dist = atr * trail_act
        if extreme - pos['entry'] >= act_dist:
            trail_price = extreme - atr * trail_dist
            if cl <= trail_price:
                pnl = (trail_price - pos['entry'] - spread) * lot * pv
                return _mk(pos, trail_price, times[i], "Trail", i, pnl)
        if bars >= max_hold:
            pnl = (cl - pos['entry'] - spread) * lot * pv
            return _mk(pos, cl, times[i], "TimeExit", i, pnl)
    else:
        pnl_now = (pos['entry'] - cl - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if hi >= pos['entry'] + sl:
            return _mk(pos, pos['entry'] + sl, times[i], "SL", i, -sl * lot * pv)
        if lo <= pos['entry'] - tp:
            return _mk(pos, pos['entry'] - tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry'])
        extreme = min(extreme, lo)
        pos['extreme'] = extreme
        act_dist = atr * trail_act
        if pos['entry'] - extreme >= act_dist:
            trail_price = extreme + atr * trail_dist
            if cl >= trail_price:
                pnl = (pos['entry'] - trail_price - spread) * lot * pv
                return _mk(pos, trail_price, times[i], "Trail", i, pnl)
        if bars >= max_hold:
            pnl = (pos['entry'] - cl - spread) * lot * pv
            return _mk(pos, cl, times[i], "TimeExit", i, pnl)
    return None


# ═══════════════════════════════════════════════════════════════
# Strategy backtests
# ═══════════════════════════════════════════════════════════════

def add_psar(df, af_step=0.01, af_max=0.05):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    n = len(df)
    psar = np.empty(n); psar[:] = np.nan
    af = af_step; rising = True
    ep = h[0]; psar[0] = l[0]
    for i in range(1, n):
        prev = psar[i-1]
        if rising:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], l[i-1], l[max(0, i-2)])
            if l[i] < psar[i]:
                rising = False; psar[i] = ep; ep = l[i]; af = af_step
            else:
                if h[i] > ep: ep = h[i]; af = min(af + af_step, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], h[i-1], h[max(0, i-2)])
            if h[i] > psar[i]:
                rising = True; psar[i] = ep; ep = h[i]; af = af_step
            else:
                if l[i] < ep: ep = l[i]; af = min(af + af_step, af_max)
    df['PSAR'] = psar


def bt_psar(h1_df, spread, lot, maxloss_cap=5,
            sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04,
            max_hold=20, af_step=0.01, af_max=0.05, min_atr=0.1):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    add_psar(df, af_step, af_max)
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
        if np.isnan(atr[i]) or atr[i] < min_atr: continue
        if psar[i-1] > c[i-1] and psar[i] < c[i]:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif psar[i-1] < c[i-1] and psar[i] > c[i]:
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


def bt_sess_bo(h1_df, spread, lot, maxloss_cap=35,
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
        ll_val = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll_val:
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
# Metrics
# ═══════════════════════════════════════════════════════════════

def sharpe(arr, ann=252):
    if len(arr) < 10: return 0.0
    s = np.std(arr, ddof=1)
    return float(np.mean(arr) / s * np.sqrt(ann)) if s > 0 else 0.0


def max_dd(arr):
    if len(arr) == 0: return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def trades_to_daily_series(trades):
    if not trades: return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


def metrics(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'max_dd': 0, 'wr': 0, 'avg_pnl': 0}
    daily = trades_to_daily_series(trades)
    wins = [t for t in trades if t['pnl'] > 0]
    return {
        'n': len(trades),
        'sharpe': round(sharpe(daily.values), 3) if len(daily) > 0 else 0,
        'pnl': round(sum(t['pnl'] for t in trades), 2),
        'max_dd': round(max_dd(daily.values), 2) if len(daily) > 0 else 0,
        'wr': round(len(wins) / len(trades) * 100, 1),
        'avg_pnl': round(sum(t['pnl'] for t in trades) / len(trades), 3),
    }


def metrics_simple(pnl_arr):
    if len(pnl_arr) < 5:
        return {'n': len(pnl_arr), 'sharpe': 0, 'pnl': 0, 'max_dd': 0, 'wr': 0, 'avg': 0}
    wins = (pnl_arr > 0).sum()
    return {
        'n': len(pnl_arr), 'sharpe': round(sharpe(pnl_arr), 3),
        'pnl': round(float(pnl_arr.sum()), 2), 'max_dd': round(max_dd(pnl_arr), 2),
        'wr': round(wins / len(pnl_arr) * 100, 1),
        'avg': round(float(pnl_arr.mean()), 4),
    }


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_h1():
    csv_candidates = sorted(DATA_DIR.glob("download/xauusd-h1-bid-*.csv"))
    if not csv_candidates:
        raise FileNotFoundError("No Dukascopy H1 CSV found in data/download/")
    csv_path = csv_candidates[-1]
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.set_index('time')[['open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[df['Close'] > 100]
    df['ATR14'] = compute_atr(df)
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['hour'] = df.index.hour
    df['date'] = df.index.date
    return df.dropna(subset=['ATR14'])


def load_cot():
    cot_path = DATA_DIR / "cot_gold_weekly.csv"
    if not cot_path.exists():
        print(f"  WARNING: {cot_path} not found", flush=True)
        return None

    cot = pd.read_csv(cot_path, index_col=0, parse_dates=True)
    if cot.index.tz is not None:
        cot.index = cot.index.tz_localize(None)

    print(f"  COT raw: {len(cot)} rows ({cot.index[0].date()} ~ {cot.index[-1].date()})", flush=True)
    print(f"  COT columns: {list(cot.columns)}", flush=True)
    return cot


def compute_cot_features(cot):
    """Compute extended COT features."""
    feat = cot.copy()

    if 'net_spec' in feat.columns:
        for window, suffix in [(4, '4w'), (13, '13w'), (26, '26w')]:
            rm = feat['net_spec'].rolling(window).mean()
            rs = feat['net_spec'].rolling(window).std()
            feat[f'net_spec_z{suffix}'] = (feat['net_spec'] - rm) / rs.replace(0, np.nan)

    if 'noncomm_long' in feat.columns and 'noncomm_short' in feat.columns:
        comm_long = feat.get('comm_long', pd.Series(0, index=feat.index))
        comm_short = feat.get('comm_short', pd.Series(0, index=feat.index))
        denom = comm_long + comm_short
        feat['commercial_ratio'] = comm_long / denom.replace(0, np.nan)
    else:
        feat['commercial_ratio'] = np.nan

    if 'open_interest' in feat.columns:
        feat['oi_momentum_4w'] = feat['open_interest'].pct_change(4)
    else:
        feat['oi_momentum_4w'] = np.nan

    if 'net_spec' in feat.columns:
        feat['net_spec_change_4w'] = feat['net_spec'].diff(4)
    else:
        feat['net_spec_change_4w'] = np.nan

    return feat


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80, flush=True)
    print("  R122 — COT Deep Analysis", flush=True)
    print("=" * 80, flush=True)

    # ════════════════════════════════════════════════════════════════
    # Phase 1: Load data and compute features
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 1: Load COT Data & Compute Features", flush=True)
    print("=" * 70, flush=True)

    h1 = load_h1()
    print(f"  H1 data: {len(h1)} bars ({h1.index[0]} ~ {h1.index[-1]})", flush=True)

    cot_raw = load_cot()
    if cot_raw is None:
        print("  FATAL: COT data not available, exiting.", flush=True)
        out_file = OUTPUT_DIR / "r122_results.json"
        with open(out_file, 'w') as f:
            json.dump({'error': 'COT data not found'}, f)
        return

    cot = compute_cot_features(cot_raw)
    z_cols = [c for c in cot.columns if 'net_spec_z' in c]
    print(f"  COT features computed: {list(cot.columns)}", flush=True)
    print(f"  Z-score columns: {z_cols}", flush=True)

    if 'net_spec_z26w' in cot.columns:
        valid_z = cot['net_spec_z26w'].dropna()
        print(f"  net_spec_z26w: {len(valid_z)} valid ({valid_z.index[0].date()} ~ {valid_z.index[-1].date()})", flush=True)
        print(f"    range: [{valid_z.min():.2f}, {valid_z.max():.2f}], current: {valid_z.iloc[-1]:.2f}", flush=True)

    h1_daily = h1['Close'].resample('D').last().dropna()
    cot_daily = cot.reindex(h1_daily.index, method='ffill')

    all_results = {
        'experiment': 'R122 COT Deep Analysis',
        'data_info': {
            'h1_bars': len(h1),
            'cot_reports': len(cot),
            'cot_period': f"{cot.index[0].date()} ~ {cot.index[-1].date()}",
        },
    }

    # ════════════════════════════════════════════════════════════════
    # Phase 2: COT as standalone contrarian signal
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: COT Standalone Contrarian Signal", flush=True)
    print("  BUY when net_spec_z26w < threshold (specs very short)", flush=True)
    print("  SELL when net_spec_z26w > threshold (specs very long)", flush=True)
    print("=" * 70, flush=True)

    z_col = 'net_spec_z26w' if 'net_spec_z26w' in cot_daily.columns else 'net_spec_z'
    if z_col not in cot_daily.columns:
        print(f"  WARNING: No z-score column available, skipping Phase 2", flush=True)
        phase2_grid = []
    else:
        merged_daily = pd.DataFrame({
            'close': h1_daily,
            'cot_z': cot_daily[z_col],
        }).dropna()

        close_arr = merged_daily['close'].values
        z_arr = merged_daily['cot_z'].values
        dates_idx = merged_daily.index
        n = len(merged_daily)

        buy_thresholds = [-2.0, -1.5, -1.0]
        sell_thresholds = [1.0, 1.5, 2.0]
        hold_periods = [5, 10, 20]

        phase2_grid = []
        print(f"\n  {'Config':<35s}  {'n':>5s}  {'Sharpe':>7s}  {'PnL':>10s}  {'WR':>6s}  {'MaxDD':>8s}  {'Avg':>8s}", flush=True)
        print("  " + "-" * 85, flush=True)

        for z_buy in buy_thresholds:
            for z_sell in sell_thresholds:
                for hold_d in hold_periods:
                    pnls = []
                    for i in range(n - hold_d):
                        if z_arr[i] < z_buy:
                            entry = close_arr[i] + SPREAD / 2
                            exit_px = close_arr[i + hold_d]
                            pnl = (exit_px - entry - SPREAD) * UNIT_LOT * PV
                            pnls.append(pnl)
                        elif z_arr[i] > z_sell:
                            entry = close_arr[i] - SPREAD / 2
                            exit_px = close_arr[i + hold_d]
                            pnl = (entry - exit_px - SPREAD) * UNIT_LOT * PV
                            pnls.append(pnl)

                    if len(pnls) < 10:
                        continue

                    pnl_arr = np.array(pnls)
                    m = metrics_simple(pnl_arr)
                    label = f"zB={z_buy}/zS={z_sell}/hold={hold_d}d"

                    phase2_grid.append({
                        'z_buy': z_buy, 'z_sell': z_sell, 'hold_days': hold_d,
                        **m,
                    })

                    if m['sharpe'] > 0.1:
                        print(f"  {label:<35s}  {m['n']:5d}  {m['sharpe']:7.3f}  ${m['pnl']:>9.0f}  "
                              f"{m['wr']:5.1f}%  ${m['max_dd']:>7.0f}  {m['avg']:8.4f}", flush=True)

        phase2_grid.sort(key=lambda x: x['sharpe'], reverse=True)
        print(f"\n  Top 10 by Sharpe:", flush=True)
        for i, g in enumerate(phase2_grid[:10]):
            print(f"    #{i+1}: zB={g['z_buy']}/zS={g['z_sell']}/hold={g['hold_days']}d "
                  f"-> Sharpe={g['sharpe']}, n={g['n']}, PnL=${g['pnl']:.0f}, WR={g['wr']:.1f}%", flush=True)

    all_results['phase2_standalone'] = phase2_grid[:20]

    # ════════════════════════════════════════════════════════════════
    # Phase 3: COT as filter on existing strategies
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3: COT as Filter on 4 Strategies", flush=True)
    print("=" * 70, flush=True)

    print("  Running baseline strategies...", flush=True)
    bundle = DataBundle.load_custom()
    psar_trades = bt_psar(h1, SPREAD, UNIT_LOT, maxloss_cap=5)
    tsmom_trades = bt_tsmom(h1, SPREAD, UNIT_LOT, maxloss_cap=0)
    sess_trades = bt_sess_bo(h1, SPREAD, UNIT_LOT, maxloss_cap=35)
    l8_trades = bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=35)

    strat_trades = {
        'PSAR': psar_trades, 'TSMOM': tsmom_trades,
        'SESS_BO': sess_trades, 'L8_MAX': l8_trades,
    }

    print(f"\n  Baseline:", flush=True)
    for name, trades in strat_trades.items():
        m = metrics(trades)
        print(f"    {name:10s}: {m['n']:5d} trades, Sharpe={m['sharpe']:6.3f}, "
              f"PnL=${m['pnl']:>8.0f}, WR={m['wr']:.1f}%", flush=True)

    phase3 = {}
    if z_col in cot_daily.columns:
        cot_z_daily = cot_daily[z_col]

        filter_configs = [
            ("BUY_if_z<0", lambda t, z: t['dir'] != 'BUY' or z < 0),
            ("BUY_if_z<-0.5", lambda t, z: t['dir'] != 'BUY' or z < -0.5),
            ("BUY_if_z<0.5", lambda t, z: t['dir'] != 'BUY' or z < 0.5),
            ("SELL_if_z>0", lambda t, z: t['dir'] != 'SELL' or z > 0),
            ("SELL_if_z>0.5", lambda t, z: t['dir'] != 'SELL' or z > 0.5),
            ("skip_extreme_neg_OI", lambda t, z: True),  # placeholder
        ]

        oi_daily = cot_daily.get('oi_momentum_4w', pd.Series(np.nan, index=cot_daily.index))

        for strat_name, trades in strat_trades.items():
            strat_phase3 = {}
            baseline_m = metrics(trades)

            for filter_label, filter_fn in filter_configs:
                filtered = []
                for t in trades:
                    entry_date = pd.Timestamp(t['entry_time']).normalize()
                    if entry_date.tzinfo is not None:
                        entry_date = entry_date.tz_localize(None)
                    z_val = cot_z_daily.get(entry_date, np.nan)
                    if pd.isna(z_val):
                        z_val = cot_z_daily.asof(entry_date) if hasattr(cot_z_daily, 'asof') else np.nan
                    if pd.isna(z_val):
                        filtered.append(t)
                        continue

                    if filter_label == "skip_extreme_neg_OI":
                        oi_val = oi_daily.get(entry_date, np.nan)
                        if pd.isna(oi_val) or oi_val > -0.10:
                            filtered.append(t)
                    else:
                        if filter_fn(t, z_val):
                            filtered.append(t)

                filt_m = metrics(filtered)
                delta = filt_m['sharpe'] - baseline_m['sharpe']
                strat_phase3[filter_label] = {
                    **filt_m,
                    'delta_sharpe': round(delta, 3),
                    'trades_removed': baseline_m['n'] - filt_m['n'],
                }

            phase3[strat_name] = strat_phase3

        print(f"\n  Filter impact (delta Sharpe from baseline):", flush=True)
        print(f"  {'Filter':<25s}", end="", flush=True)
        for name in strat_trades.keys():
            print(f"  {name:>10s}", end="")
        print(flush=True)
        print("  " + "-" * 70, flush=True)

        for filter_label, _ in filter_configs:
            print(f"  {filter_label:<25s}", end="", flush=True)
            for name in strat_trades.keys():
                delta = phase3.get(name, {}).get(filter_label, {}).get('delta_sharpe', 0)
                print(f"  {delta:>+10.3f}", end="")
            print(flush=True)

    all_results['phase3_filter'] = phase3

    # ════════════════════════════════════════════════════════════════
    # Phase 4: Walk-forward validation
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 4: Walk-Forward Validation (2yr train / 6mo test)", flush=True)
    print("=" * 70, flush=True)

    phase4 = []
    if z_col in cot_daily.columns and len(phase2_grid) > 0:
        merged_daily_wf = pd.DataFrame({
            'close': h1_daily,
            'cot_z': cot_daily[z_col],
        }).dropna()

        wf_starts = pd.date_range(start='2008-01-01', end='2025-06-01', freq='6ME')

        train_window_days = 252 * 2
        test_window_days = 126

        wf_results = []
        print(f"\n  {'Window':<25s}  {'OptZBuy':>8s}  {'OptZSell':>8s}  {'Hold':>5s}  "
              f"{'TrainS':>7s}  {'TestS':>7s}  {'TestN':>6s}", flush=True)
        print("  " + "-" * 75, flush=True)

        for wf_start in wf_starts:
            train_start = wf_start - pd.Timedelta(days=train_window_days)
            train_end = wf_start
            test_start = wf_start
            test_end = wf_start + pd.Timedelta(days=test_window_days)

            train_data = merged_daily_wf[(merged_daily_wf.index >= train_start) &
                                         (merged_daily_wf.index < train_end)]
            test_data = merged_daily_wf[(merged_daily_wf.index >= test_start) &
                                        (merged_daily_wf.index < test_end)]

            if len(train_data) < 200 or len(test_data) < 30:
                continue

            best_train_sharpe = -999
            best_params = None

            for z_buy in [-2.0, -1.5, -1.0]:
                for z_sell in [1.0, 1.5, 2.0]:
                    for hold_d in [5, 10, 20]:
                        c_arr = train_data['close'].values
                        z_arr = train_data['cot_z'].values
                        pnls = []
                        for j in range(len(c_arr) - hold_d):
                            if z_arr[j] < z_buy:
                                pnl = (c_arr[j + hold_d] - c_arr[j] - SPREAD * 2) * UNIT_LOT * PV
                                pnls.append(pnl)
                            elif z_arr[j] > z_sell:
                                pnl = (c_arr[j] - c_arr[j + hold_d] - SPREAD * 2) * UNIT_LOT * PV
                                pnls.append(pnl)
                        if len(pnls) < 10:
                            continue
                        tr_sharpe = sharpe(np.array(pnls))
                        if tr_sharpe > best_train_sharpe:
                            best_train_sharpe = tr_sharpe
                            best_params = (z_buy, z_sell, hold_d)

            if best_params is None:
                continue

            z_buy, z_sell, hold_d = best_params
            c_arr = test_data['close'].values
            z_arr = test_data['cot_z'].values
            pnls = []
            for j in range(len(c_arr) - hold_d):
                if z_arr[j] < z_buy:
                    pnl = (c_arr[j + hold_d] - c_arr[j] - SPREAD * 2) * UNIT_LOT * PV
                    pnls.append(pnl)
                elif z_arr[j] > z_sell:
                    pnl = (c_arr[j] - c_arr[j + hold_d] - SPREAD * 2) * UNIT_LOT * PV
                    pnls.append(pnl)

            test_sharpe = sharpe(np.array(pnls)) if len(pnls) >= 5 else 0.0
            window_label = f"{train_end.strftime('%Y-%m')} -> {test_end.strftime('%Y-%m')}"

            wf_results.append({
                'window': window_label,
                'opt_z_buy': z_buy, 'opt_z_sell': z_sell, 'opt_hold': hold_d,
                'train_sharpe': round(best_train_sharpe, 3),
                'test_sharpe': round(test_sharpe, 3),
                'test_n': len(pnls),
            })

            if len(pnls) >= 5:
                print(f"  {window_label:<25s}  {z_buy:>8.1f}  {z_sell:>8.1f}  {hold_d:>5d}  "
                      f"{best_train_sharpe:>7.3f}  {test_sharpe:>7.3f}  {len(pnls):>6d}", flush=True)

        if wf_results:
            test_sharpes = [r['test_sharpe'] for r in wf_results if r['test_n'] >= 5]
            pos = sum(1 for s in test_sharpes if s > 0)
            mean_test = np.mean(test_sharpes) if test_sharpes else 0
            print(f"\n  Walk-forward summary:", flush=True)
            print(f"    Windows tested: {len(wf_results)}", flush=True)
            print(f"    Positive test Sharpe: {pos}/{len(test_sharpes)}", flush=True)
            print(f"    Mean test Sharpe: {mean_test:.3f}", flush=True)

        phase4 = wf_results

    all_results['phase4_walkforward'] = phase4

    # ════════════════════════════════════════════════════════════════
    # Phase 5: K-Fold Validation (5 folds)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 5: K-Fold Validation (5 folds)", flush=True)
    print("=" * 70, flush=True)

    phase5 = {}

    # 5a: Standalone COT signal k-fold
    if z_col in cot_daily.columns and len(phase2_grid) >= 3:
        print("\n  A. Standalone COT signal (top 3 configs):", flush=True)
        merged_daily_kf = pd.DataFrame({
            'close': h1_daily,
            'cot_z': cot_daily[z_col],
        }).dropna()

        standalone_kf = {}
        for rank, g in enumerate(phase2_grid[:3]):
            z_buy = g['z_buy']
            z_sell = g['z_sell']
            hold_d = g['hold_days']
            label = f"zB={z_buy}/zS={z_sell}/H={hold_d}"

            fold_sharpes = []
            for fname, start, end in FOLDS:
                fold_data = merged_daily_kf[(merged_daily_kf.index >= start) &
                                            (merged_daily_kf.index < end)]
                if len(fold_data) < 60:
                    fold_sharpes.append(0.0)
                    continue

                c_arr = fold_data['close'].values
                z_arr = fold_data['cot_z'].values
                pnls = []
                for j in range(len(c_arr) - hold_d):
                    if z_arr[j] < z_buy:
                        pnl = (c_arr[j + hold_d] - c_arr[j] - SPREAD * 2) * UNIT_LOT * PV
                        pnls.append(pnl)
                    elif z_arr[j] > z_sell:
                        pnl = (c_arr[j] - c_arr[j + hold_d] - SPREAD * 2) * UNIT_LOT * PV
                        pnls.append(pnl)

                fold_sharpes.append(round(sharpe(np.array(pnls)) if len(pnls) >= 5 else 0.0, 3))

            pos = sum(1 for s in fold_sharpes if s > 0)
            mean_s = round(np.mean(fold_sharpes), 3)
            status = "PASS" if pos >= 3 else "FAIL"
            print(f"    #{rank+1} {label}: {fold_sharpes} -> {pos}/5 [{status}] mean={mean_s}", flush=True)

            standalone_kf[label] = {
                'params': {'z_buy': z_buy, 'z_sell': z_sell, 'hold_days': hold_d},
                'fold_sharpes': fold_sharpes,
                'positive': pos, 'mean': mean_s, 'pass': pos >= 3,
            }

        phase5['standalone'] = standalone_kf

    # 5b: Filter variant k-fold
    if z_col in cot_daily.columns and phase3:
        print("\n  B. COT filter on strategies (best filter per strategy):", flush=True)
        filter_kf = {}

        best_filter_per_strat = {}
        for sname, filters in phase3.items():
            best_f = max(filters.items(), key=lambda x: x[1].get('delta_sharpe', -999))
            best_filter_per_strat[sname] = best_f[0]

        cot_z_series = cot_daily[z_col]
        oi_series = cot_daily.get('oi_momentum_4w', pd.Series(np.nan, index=cot_daily.index))

        for sname in strat_trades.keys():
            best_filter = best_filter_per_strat.get(sname, "BUY_if_z<0")
            fold_sharpes_base = []
            fold_sharpes_filt = []

            for fname, start, end in FOLDS:
                fold_h1 = h1[(h1.index >= start) & (h1.index < end)]
                if len(fold_h1) < 200:
                    fold_sharpes_base.append(0.0)
                    fold_sharpes_filt.append(0.0)
                    continue

                if sname == 'PSAR':
                    fold_trades = bt_psar(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=5)
                elif sname == 'TSMOM':
                    fold_trades = bt_tsmom(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=0)
                elif sname == 'SESS_BO':
                    fold_trades = bt_sess_bo(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=35)
                else:
                    fold_trades = bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=35)

                base_m = metrics(fold_trades)
                fold_sharpes_base.append(base_m['sharpe'])

                filtered = []
                for t in fold_trades:
                    entry_date = pd.Timestamp(t['entry_time']).normalize()
                    if entry_date.tzinfo is not None:
                        entry_date = entry_date.tz_localize(None)
                    z_val = cot_z_series.asof(entry_date) if hasattr(cot_z_series, 'asof') else np.nan
                    if pd.isna(z_val):
                        filtered.append(t)
                        continue

                    if best_filter == "BUY_if_z<0":
                        if t['dir'] != 'BUY' or z_val < 0:
                            filtered.append(t)
                    elif best_filter == "BUY_if_z<-0.5":
                        if t['dir'] != 'BUY' or z_val < -0.5:
                            filtered.append(t)
                    elif best_filter == "BUY_if_z<0.5":
                        if t['dir'] != 'BUY' or z_val < 0.5:
                            filtered.append(t)
                    elif best_filter == "SELL_if_z>0":
                        if t['dir'] != 'SELL' or z_val > 0:
                            filtered.append(t)
                    elif best_filter == "SELL_if_z>0.5":
                        if t['dir'] != 'SELL' or z_val > 0.5:
                            filtered.append(t)
                    elif best_filter == "skip_extreme_neg_OI":
                        oi_val = oi_series.asof(entry_date) if hasattr(oi_series, 'asof') else np.nan
                        if pd.isna(oi_val) or oi_val > -0.10:
                            filtered.append(t)
                    else:
                        filtered.append(t)

                filt_m = metrics(filtered)
                fold_sharpes_filt.append(filt_m['sharpe'])

            wins = sum(1 for a, b in zip(fold_sharpes_filt, fold_sharpes_base) if a > b)
            print(f"    {sname:10s} filter={best_filter}: "
                  f"base={[f'{s:.2f}' for s in fold_sharpes_base]} "
                  f"filt={[f'{s:.2f}' for s in fold_sharpes_filt]} "
                  f"wins={wins}/5", flush=True)

            filter_kf[sname] = {
                'filter': best_filter,
                'fold_sharpes_base': [round(s, 3) for s in fold_sharpes_base],
                'fold_sharpes_filt': [round(s, 3) for s in fold_sharpes_filt],
                'mean_base': round(np.mean(fold_sharpes_base), 3),
                'mean_filt': round(np.mean(fold_sharpes_filt), 3),
                'wins': wins,
            }

        phase5['filter'] = filter_kf

    all_results['phase5_kfold'] = phase5

    # ════════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    all_results['elapsed_s'] = round(elapsed, 1)

    print("\n" + "=" * 80, flush=True)
    print("  R122 FINAL SUMMARY", flush=True)
    print("=" * 80, flush=True)

    if phase2_grid:
        top = phase2_grid[0]
        print(f"\n  Phase 2 — Best standalone COT signal:", flush=True)
        print(f"    zBuy={top['z_buy']}, zSell={top['z_sell']}, hold={top['hold_days']}d", flush=True)
        print(f"    Sharpe={top['sharpe']}, n={top['n']}, PnL=${top['pnl']:.0f}", flush=True)

    if phase3:
        print(f"\n  Phase 3 — Best COT filter impact (delta Sharpe):", flush=True)
        for sname, filters in phase3.items():
            best_f = max(filters.items(), key=lambda x: x[1].get('delta_sharpe', -999))
            print(f"    {sname:10s}: {best_f[0]} -> delta={best_f[1]['delta_sharpe']:+.3f}", flush=True)

    if phase4:
        test_sharpes = [r['test_sharpe'] for r in phase4 if r['test_n'] >= 5]
        pos = sum(1 for s in test_sharpes if s > 0)
        print(f"\n  Phase 4 — Walk-forward: {pos}/{len(test_sharpes)} positive test windows", flush=True)

    if 'standalone' in phase5:
        print(f"\n  Phase 5 — K-Fold standalone:", flush=True)
        for label, info in phase5['standalone'].items():
            status = "PASS" if info['pass'] else "FAIL"
            print(f"    {label}: {info['positive']}/5 [{status}] mean={info['mean']}", flush=True)

    if 'filter' in phase5:
        print(f"\n  Phase 5 — K-Fold filter:", flush=True)
        for sname, info in phase5['filter'].items():
            print(f"    {sname:10s}: wins={info['wins']}/5, "
                  f"mean_base={info['mean_base']:.3f}, mean_filt={info['mean_filt']:.3f}", flush=True)

    out_file = OUTPUT_DIR / "r122_results.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Saved: {out_file}", flush=True)
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)", flush=True)


if __name__ == '__main__':
    main()
