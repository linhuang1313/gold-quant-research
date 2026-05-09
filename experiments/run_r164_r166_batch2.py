#!/usr/bin/env python3
"""
R164–R166 — Batch 2: Friday Close, Directional CB, Adaptive Cap
================================================================
Three experiments targeting specific loss patterns:

R164: Friday Close Force-Exit
  - Filter out trades entered on Friday after UTC cutoff hours (16, 18, 20)
  - Addresses weekend gap losses (#6 and #7)

R165: Directional Consecutive Loss Filter
  - Pause only the losing direction after N consecutive losses in same direction
  - More granular than Circuit Breaker V1

R166: Gold-Price Adaptive Cap
  - MaxLoss cap scaled to gold price level (gold $1100→$3300 over 10 years)
  - Price-proportional cap and ATR-proportional cap variants
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS, load_m15, load_h1_aligned, H1_CSV_PATH

OUTPUT_DIR = Path("results/r164_r166_batch2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01
CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}

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
        pnl_c = (c - pos['entry'] - spread) * lot * pv
    else:
        pnl_c = (pos['entry'] - c - spread) * lot * pv
    tp_val = tp_atr * pos['atr'] * lot * pv
    sl_val = sl_atr * pos['atr'] * lot * pv
    if pos['dir'] == 'BUY':
        pnl_h = (h - pos['entry'] - spread) * lot * pv
        pnl_l = (lo_v - pos['entry'] - spread) * lot * pv
    else:
        pnl_h = (pos['entry'] - lo_v - spread) * lot * pv
        pnl_l = (pos['entry'] - h - spread) * lot * pv
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
# Strategy backtests
# ═══════════════════════════════════════════════════════════════

def bt_psar(h1_df, spread, lot, maxloss_cap=0):
    sl_atr=4.5; tp_atr=16.0; trail_act=0.20; trail_dist=0.04; max_hold=20
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


def bt_tsmom(h1_df, spread, lot, maxloss_cap=0):
    fast=480; slow=720; sl_atr=4.5; tp_atr=6.0
    trail_act=0.14; trail_dist=0.025; max_hold=20
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
          'keltner_max_hold_m15': 8,
          'min_lot_size': lot, 'max_lot_size': lot}
    result = run_variant(data_bundle, "L8_MAX", verbose=False, **kw)
    raw_trades = result.get('_trades', [])
    return [
        {'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
         'entry_time': t.entry_time, 'exit_time': t.exit_time,
         'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0}
        for t in raw_trades
    ]


# ═══════════════════════════════════════════════════════════════
# Stats helpers
# ═══════════════════════════════════════════════════════════════

def _normalize_ts(ts):
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    return t


def trades_to_daily(trades):
    if not trades:
        return np.array([0.0])
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).normalize()
        if d.tzinfo is not None:
            d = d.tz_localize(None)
        daily[d] = daily.get(d, 0) + t['pnl']
    if not daily:
        return np.array([0.0])
    return np.array([daily[k] for k in sorted(daily.keys())])


def sharpe(arr):
    if len(arr) < 10 or np.std(arr, ddof=1) == 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))


def max_dd(arr):
    if len(arr) < 2:
        return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def compute_stats(trades, label=""):
    n = len(trades)
    if n == 0:
        return {'label': label, 'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0,
                'max_dd': 0, 'avg_pnl': 0, 'worst_trade': 0}
    pnls = np.array([t['pnl'] for t in trades])
    daily = trades_to_daily(trades)
    return {
        'label': label, 'n': n,
        'sharpe': round(sharpe(daily), 3),
        'pnl': round(float(pnls.sum()), 2),
        'wr': round(float((pnls > 0).sum()) / n * 100, 1),
        'max_dd': round(max_dd(daily), 2),
        'avg_pnl': round(float(pnls.mean()), 4),
        'worst_trade': round(float(pnls.min()), 2),
    }


# ═══════════════════════════════════════════════════════════════
# Portfolio merge
# ═══════════════════════════════════════════════════════════════

def merge_portfolio_trades(strat_trades_dict, lot_scale=None):
    if lot_scale is None:
        lot_scale = R89_LOTS
    all_trades = []
    for strat_name, trades in strat_trades_dict.items():
        mult = lot_scale.get(strat_name, UNIT_LOT) / UNIT_LOT
        for t in trades:
            all_trades.append({
                'strategy': strat_name,
                'dir': t['dir'], 'entry': t['entry'], 'exit': t['exit'],
                'entry_time': _normalize_ts(t['entry_time']),
                'exit_time': _normalize_ts(t['exit_time']),
                'pnl': t['pnl'] * mult,
                'pnl_unit': t['pnl'],
                'reason': t['reason'],
            })
    all_trades.sort(key=lambda x: x['exit_time'])
    return all_trades


# ═══════════════════════════════════════════════════════════════
# Rule B / Circuit Breaker (shared)
# ═══════════════════════════════════════════════════════════════

def build_extreme_mask(h1_df, cusum_sigma=3.0, extreme_window=12):
    n = len(h1_df)
    extreme = np.zeros(n, dtype=bool)
    atr = compute_atr(h1_df).values
    atr_clean = np.nan_to_num(atr, nan=0.0)
    atr_mean = pd.Series(atr_clean).rolling(60, min_periods=20).mean().values
    atr_std = pd.Series(atr_clean).rolling(60, min_periods=20).std().values
    atr_std = np.maximum(atr_std, 1e-6)
    cusum_trigger = atr_clean > (atr_mean + cusum_sigma * atr_std)
    for i in range(n):
        if cusum_trigger[i]:
            end_i = min(i + extreme_window, n)
            extreme[i:end_i] = True
    return extreme


def apply_rule_b(trades, h1_df, extreme_mask, skip_bars=8):
    if not trades:
        return [], 0
    times_idx = h1_df.index
    n_bars = len(times_idx)
    idx_is_tz_aware = times_idx.tz is not None

    def _find_bar(ts):
        ts = pd.Timestamp(ts)
        if idx_is_tz_aware and ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        elif not idx_is_tz_aware and ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        idx = times_idx.searchsorted(ts)
        return min(idx, n_bars - 1) if idx < n_bars else n_bars - 1

    protected = []; skipped = 0
    for t in trades:
        entry_bar = _find_bar(t['entry_time'])
        skip_end = entry_bar
        for j in range(max(0, entry_bar - skip_bars), entry_bar):
            if j < n_bars and extreme_mask[j]:
                skip_end = j + skip_bars
                break
        if entry_bar < skip_end:
            skipped += 1
            continue
        protected.append(t)
    return protected, skipped


def apply_circuit_breaker(portfolio_trades, streak_thresh=3, pause_hours=1):
    taken = []; skipped = 0
    consec_losses = 0; pause_until = None
    for trade in portfolio_trades:
        entry_t = trade['entry_time']
        exit_t = trade['exit_time']
        if pause_until is not None and entry_t < pause_until:
            skipped += 1
            continue
        if pause_until is not None and entry_t >= pause_until:
            pause_until = None
            consec_losses = 0
        taken.append(trade)
        if trade['pnl'] < 0:
            consec_losses += 1
            if consec_losses >= streak_thresh:
                pause_until = exit_t + timedelta(hours=pause_hours)
        else:
            consec_losses = 0
    return taken, skipped


# ═══════════════════════════════════════════════════════════════
# Run all strategies
# ═══════════════════════════════════════════════════════════════

def run_all_strategies(h1_df, l8_bundle):
    strat_trades = {}
    strat_trades['L8_MAX'] = bt_l8_max(l8_bundle, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
    strat_trades['PSAR'] = bt_psar(h1_df, SPREAD, UNIT_LOT, CAPS['PSAR'])
    strat_trades['TSMOM'] = bt_tsmom(h1_df, SPREAD, UNIT_LOT, CAPS['TSMOM'])
    strat_trades['SESS_BO'] = bt_sess_bo(h1_df, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
    return strat_trades


# ═══════════════════════════════════════════════════════════════
# R164: Friday Close Force-Exit
# ═══════════════════════════════════════════════════════════════

def apply_friday_filter(portfolio_trades, cutoff_hour=18):
    """Remove trades entered on Friday after cutoff_hour UTC."""
    taken = []; skipped = 0
    for trade in portfolio_trades:
        entry_t = pd.Timestamp(trade['entry_time'])
        if entry_t.tzinfo is not None:
            entry_t = entry_t.tz_localize(None)
        if entry_t.weekday() == 4 and entry_t.hour >= cutoff_hour:
            skipped += 1
            continue
        taken.append(trade)
    return taken, skipped


def run_r164(h1_df, l8_bundle, portfolio_all):
    print("\n" + "=" * 80)
    print("  R164 — Friday Close Force-Exit")
    print("  Hypothesis: blocking Friday afternoon entries avoids weekend gap losses")
    print("=" * 80, flush=True)

    results = {}

    # Phase 1: Baseline
    baseline_stats = compute_stats(portfolio_all, "Baseline")
    results['baseline'] = baseline_stats
    print(f"\n    Baseline: N={baseline_stats['n']}, Sharpe={baseline_stats['sharpe']:.3f}, "
          f"PnL=${baseline_stats['pnl']:.1f}", flush=True)

    # Phase 2: Friday entry filter at different cutoff hours
    print(f"\n    Phase 2: Friday Entry Filter (remove trades entered after cutoff)")
    print(f"    {'Cutoff':>8} {'N':>6} {'Skip':>6} {'Sharpe':>8} {'PnL':>10} "
          f"{'MaxDD':>8} {'WR':>6} {'Worst':>8}")
    print("    " + "-" * 70)

    cutoff_hours = [16, 18, 20]
    phase2 = {}
    for ch in cutoff_hours:
        filtered, skipped = apply_friday_filter(portfolio_all, ch)
        st = compute_stats(filtered, f"FriCut_{ch}")
        st['skipped'] = skipped
        phase2[ch] = st
        print(f"    {ch:>8} {st['n']:>6} {skipped:>6} {st['sharpe']:>8.3f} "
              f"{st['pnl']:>10.1f} {st['max_dd']:>8.1f} {st['wr']:>5.1f}% "
              f"{st['worst_trade']:>8.2f}", flush=True)

    results['phase2_friday_filter'] = phase2

    # Phase 3: Analyze Friday trades specifically
    print(f"\n    Phase 3: Friday Afternoon Trades Analysis")
    friday_trades = []
    for t in portfolio_all:
        entry_t = pd.Timestamp(t['entry_time'])
        if entry_t.tzinfo is not None:
            entry_t = entry_t.tz_localize(None)
        if entry_t.weekday() == 4 and entry_t.hour >= 16:
            friday_trades.append(t)

    if friday_trades:
        fri_pnls = np.array([t['pnl'] for t in friday_trades])
        fri_stats = {
            'count': len(friday_trades),
            'total_pnl': round(float(fri_pnls.sum()), 2),
            'avg_pnl': round(float(fri_pnls.mean()), 4),
            'wr': round(float((fri_pnls > 0).sum()) / len(fri_pnls) * 100, 1),
            'worst': round(float(fri_pnls.min()), 2),
        }
        print(f"      Friday >=16h trades: N={fri_stats['count']}, "
              f"PnL=${fri_stats['total_pnl']:.1f}, WR={fri_stats['wr']:.1f}%, "
              f"Avg=${fri_stats['avg_pnl']:.4f}, Worst=${fri_stats['worst']:.2f}",
              flush=True)
    else:
        fri_stats = {'count': 0, 'total_pnl': 0, 'avg_pnl': 0, 'wr': 0, 'worst': 0}
        print("      No Friday afternoon trades found.", flush=True)

    results['phase3_friday_analysis'] = fri_stats

    # Phase 4: K-Fold on best Friday filter
    best_cutoff = max(phase2.items(), key=lambda x: x[1]['sharpe'])[0]
    print(f"\n    Phase 4: K-Fold on best cutoff={best_cutoff}h", flush=True)

    fold_sharpes_baseline = []
    fold_sharpes_filter = []

    for fold_name, start, end in FOLDS:
        fs = pd.Timestamp(start); fe = pd.Timestamp(end)
        fold_trades = [t for t in portfolio_all
                       if fs <= t['exit_time'] < fe]
        if not fold_trades:
            fold_sharpes_baseline.append(0.0)
            fold_sharpes_filter.append(0.0)
            continue

        daily_b = trades_to_daily(fold_trades)
        fold_sharpes_baseline.append(sharpe(daily_b))

        filtered, _ = apply_friday_filter(fold_trades, best_cutoff)
        daily_f = trades_to_daily(filtered)
        fold_sharpes_filter.append(sharpe(daily_f))

        print(f"      {fold_name}: Baseline={fold_sharpes_baseline[-1]:.3f}, "
              f"FriFilter={fold_sharpes_filter[-1]:.3f}", flush=True)

    wins = sum(1 for b, f in zip(fold_sharpes_baseline, fold_sharpes_filter) if f > b)
    results['phase4_kfold'] = {
        'best_cutoff': best_cutoff,
        'baseline_sharpes': [round(s, 3) for s in fold_sharpes_baseline],
        'filter_sharpes': [round(s, 3) for s in fold_sharpes_filter],
        'filter_wins': wins,
        'mean_baseline': round(float(np.mean(fold_sharpes_baseline)), 3),
        'mean_filter': round(float(np.mean(fold_sharpes_filter)), 3),
    }
    print(f"    Filter wins {wins}/6 folds. "
          f"Mean: Baseline={np.mean(fold_sharpes_baseline):.3f} vs "
          f"Filter={np.mean(fold_sharpes_filter):.3f}", flush=True)

    return results


# ═══════════════════════════════════════════════════════════════
# R165: Directional Consecutive Loss Filter
# ═══════════════════════════════════════════════════════════════

def apply_directional_cb(portfolio_trades, streak_thresh=2, pause_hours=1):
    """Pause only the losing direction after N consecutive losses in same direction."""
    taken = []; skipped = 0
    consec_buy_losses = 0; consec_sell_losses = 0
    buy_pause_until = None; sell_pause_until = None
    for trade in portfolio_trades:
        entry_t = trade['entry_time']
        exit_t = trade['exit_time']
        direction = trade['dir']
        if direction == 'BUY' and buy_pause_until and entry_t < buy_pause_until:
            skipped += 1; continue
        if direction == 'SELL' and sell_pause_until and entry_t < sell_pause_until:
            skipped += 1; continue
        if direction == 'BUY' and buy_pause_until and entry_t >= buy_pause_until:
            buy_pause_until = None; consec_buy_losses = 0
        if direction == 'SELL' and sell_pause_until and entry_t >= sell_pause_until:
            sell_pause_until = None; consec_sell_losses = 0
        taken.append(trade)
        if trade['pnl'] < 0:
            if direction == 'BUY':
                consec_buy_losses += 1
                if consec_buy_losses >= streak_thresh:
                    buy_pause_until = exit_t + timedelta(hours=pause_hours)
            else:
                consec_sell_losses += 1
                if consec_sell_losses >= streak_thresh:
                    sell_pause_until = exit_t + timedelta(hours=pause_hours)
        else:
            if direction == 'BUY':
                consec_buy_losses = 0
            else:
                consec_sell_losses = 0
    return taken, skipped


def run_r165(h1_df, l8_bundle, portfolio_all):
    print("\n" + "=" * 80)
    print("  R165 — Directional Consecutive Loss Filter")
    print("  Hypothesis: pausing only the losing direction is better than full CB")
    print("=" * 80, flush=True)

    results = {}

    # Baseline
    baseline_stats = compute_stats(portfolio_all, "Baseline")
    results['baseline'] = baseline_stats

    # CB V1 for comparison
    cb_trades, cb_skip = apply_circuit_breaker(portfolio_all, streak_thresh=3, pause_hours=1)
    cb_stats = compute_stats(cb_trades, "CB_V1_3x1h")
    cb_stats['skipped'] = cb_skip
    results['cb_v1'] = cb_stats
    print(f"\n    Baseline:  N={baseline_stats['n']}, Sharpe={baseline_stats['sharpe']:.3f}")
    print(f"    CB V1:     N={cb_stats['n']}, Sharpe={cb_stats['sharpe']:.3f}, "
          f"skipped={cb_skip}", flush=True)

    # Phase 1: Grid search
    print(f"\n    Phase 1: Directional CB Grid Search")
    print(f"    {'Streak':>7} {'Pause_h':>8} {'N':>6} {'Skip':>6} {'Sharpe':>8} "
          f"{'PnL':>10} {'MaxDD':>8} {'WR':>6}")
    print("    " + "-" * 70)

    streak_grid = [2, 3]
    pause_grid = [0.5, 1.0, 2.0]
    phase1 = {}

    for streak in streak_grid:
        for pause in pause_grid:
            filtered, skipped = apply_directional_cb(portfolio_all, streak, pause)
            st = compute_stats(filtered, f"DCB_s{streak}_p{pause}")
            st['skipped'] = skipped
            key = f"s{streak}_p{pause}"
            phase1[key] = st
            print(f"    {streak:>7} {pause:>8.1f} {st['n']:>6} {skipped:>6} "
                  f"{st['sharpe']:>8.3f} {st['pnl']:>10.1f} {st['max_dd']:>8.1f} "
                  f"{st['wr']:>5.1f}%", flush=True)

    results['phase1_grid'] = phase1

    # Find best
    best_key = max(phase1.items(), key=lambda x: x[1]['sharpe'])[0]
    best_streak = int(best_key.split('_')[0][1:])
    best_pause = float(best_key.split('_')[1][1:])
    results['best_params'] = {'streak': best_streak, 'pause_hours': best_pause}
    print(f"\n    Best: streak={best_streak}, pause={best_pause}h "
          f"(Sharpe={phase1[best_key]['sharpe']:.3f})", flush=True)

    # Phase 2: K-Fold on best
    print(f"\n    Phase 2: K-Fold validation", flush=True)

    fold_sharpes_baseline = []
    fold_sharpes_dcb = []
    fold_sharpes_cb = []

    for fold_name, start, end in FOLDS:
        fs = pd.Timestamp(start); fe = pd.Timestamp(end)
        fold_trades = [t for t in portfolio_all
                       if fs <= t['exit_time'] < fe]
        if not fold_trades:
            fold_sharpes_baseline.append(0.0)
            fold_sharpes_dcb.append(0.0)
            fold_sharpes_cb.append(0.0)
            continue

        daily_b = trades_to_daily(fold_trades)
        fold_sharpes_baseline.append(sharpe(daily_b))

        dcb_filtered, _ = apply_directional_cb(fold_trades, best_streak, best_pause)
        daily_dcb = trades_to_daily(dcb_filtered)
        fold_sharpes_dcb.append(sharpe(daily_dcb))

        cb_filtered, _ = apply_circuit_breaker(fold_trades, 3, 1)
        daily_cb = trades_to_daily(cb_filtered)
        fold_sharpes_cb.append(sharpe(daily_cb))

        print(f"      {fold_name}: Base={fold_sharpes_baseline[-1]:.3f}, "
              f"DCB={fold_sharpes_dcb[-1]:.3f}, CB={fold_sharpes_cb[-1]:.3f}", flush=True)

    dcb_wins_vs_base = sum(1 for b, d in zip(fold_sharpes_baseline, fold_sharpes_dcb) if d > b)
    dcb_wins_vs_cb = sum(1 for c, d in zip(fold_sharpes_cb, fold_sharpes_dcb) if d > c)

    results['phase2_kfold'] = {
        'baseline_sharpes': [round(s, 3) for s in fold_sharpes_baseline],
        'dcb_sharpes': [round(s, 3) for s in fold_sharpes_dcb],
        'cb_sharpes': [round(s, 3) for s in fold_sharpes_cb],
        'dcb_wins_vs_baseline': dcb_wins_vs_base,
        'dcb_wins_vs_cb': dcb_wins_vs_cb,
        'mean_baseline': round(float(np.mean(fold_sharpes_baseline)), 3),
        'mean_dcb': round(float(np.mean(fold_sharpes_dcb)), 3),
        'mean_cb': round(float(np.mean(fold_sharpes_cb)), 3),
    }
    print(f"\n    DCB wins vs Baseline: {dcb_wins_vs_base}/6")
    print(f"    DCB wins vs CB V1:    {dcb_wins_vs_cb}/6")
    print(f"    Mean: Base={np.mean(fold_sharpes_baseline):.3f}, "
          f"DCB={np.mean(fold_sharpes_dcb):.3f}, CB={np.mean(fold_sharpes_cb):.3f}",
          flush=True)

    return results


# ═══════════════════════════════════════════════════════════════
# R166: Gold-Price Adaptive Cap
# ═══════════════════════════════════════════════════════════════

def apply_price_proportional_cap(trades, base_cap, reference_price=2000):
    """Post-process: clip losing trades to price-proportional cap."""
    adjusted = []
    for t in trades:
        entry_price = t.get('entry', 0)
        if entry_price <= 0:
            adjusted.append(t)
            continue
        adj_cap = base_cap * (entry_price / reference_price)
        if t['pnl'] < -adj_cap:
            new_t = dict(t)
            new_t['pnl'] = -adj_cap
            new_t['reason'] = 'AdaptiveCap'
            adjusted.append(new_t)
        else:
            adjusted.append(t)
    return adjusted


def apply_atr_proportional_cap(trades, h1_df, multiplier=2.5, lot=UNIT_LOT):
    """Post-process: clip losing trades to ATR-proportional cap."""
    atr_series = compute_atr(h1_df)
    times_idx = h1_df.index
    idx_is_tz_aware = times_idx.tz is not None

    adjusted = []
    for t in trades:
        entry_ts = pd.Timestamp(t['entry_time'])
        if idx_is_tz_aware and entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize('UTC')
        elif not idx_is_tz_aware and entry_ts.tzinfo is not None:
            entry_ts = entry_ts.tz_localize(None)

        bar_idx = times_idx.searchsorted(entry_ts)
        if bar_idx >= len(times_idx):
            bar_idx = len(times_idx) - 1

        atr_val = atr_series.iloc[bar_idx] if bar_idx < len(atr_series) else 10.0
        if np.isnan(atr_val):
            atr_val = 10.0

        trade_lot = lot
        if 'strategy' in t:
            trade_lot = R89_LOTS.get(t['strategy'], lot)
        adj_cap = multiplier * atr_val * trade_lot * PV

        if t['pnl'] < -adj_cap:
            new_t = dict(t)
            new_t['pnl'] = -adj_cap
            new_t['reason'] = 'ATR_Cap'
            adjusted.append(new_t)
        else:
            adjusted.append(t)
    return adjusted


def run_r166(h1_df, l8_bundle, portfolio_all):
    print("\n" + "=" * 80)
    print("  R166 — Gold-Price Adaptive Cap")
    print("  Hypothesis: fixed $35 cap is too tight at low gold, too loose at high gold")
    print("=" * 80, flush=True)

    results = {}

    # Baseline
    baseline_stats = compute_stats(portfolio_all, "Baseline")
    results['baseline'] = baseline_stats
    print(f"\n    Baseline: N={baseline_stats['n']}, Sharpe={baseline_stats['sharpe']:.3f}, "
          f"PnL=${baseline_stats['pnl']:.1f}", flush=True)

    # Phase 1: Analyze cap vs price level
    print(f"\n    Phase 1: $35 cap impact analysis by price regime")
    price_regimes = [
        ("<$1400", 0, 1400),
        ("$1400-1800", 1400, 1800),
        ("$1800-2200", 1800, 2200),
        ("$2200-2600", 2200, 2600),
        ("$2600-3000", 2600, 3000),
        (">$3000", 3000, 99999),
    ]

    phase1 = []
    print(f"    {'Regime':<14} {'N':>6} {'N_capped':>8} {'Avg_pnl':>10} {'Cap_pips':>10}")
    print("    " + "-" * 55)

    for label, lo_p, hi_p in price_regimes:
        regime_trades = [t for t in portfolio_all
                         if lo_p <= t.get('entry', 0) < hi_p]
        if not regime_trades:
            continue
        n_capped = sum(1 for t in regime_trades if t['pnl'] <= -34.9)
        avg_pnl = np.mean([t['pnl'] for t in regime_trades])
        mid_price = (lo_p + min(hi_p, 3500)) / 2
        cap_pips = 35.0 / (UNIT_LOT * PV) if UNIT_LOT * PV > 0 else 0
        row = {'regime': label, 'n': len(regime_trades), 'n_capped': n_capped,
               'avg_pnl': round(float(avg_pnl), 4), 'cap_in_pips': round(cap_pips, 1),
               'mid_price': mid_price}
        phase1.append(row)
        print(f"    {label:<14} {len(regime_trades):>6} {n_capped:>8} "
              f"{avg_pnl:>10.4f} {cap_pips:>10.1f}", flush=True)

    results['phase1_regime_analysis'] = phase1

    # Phase 2: Price-proportional cap
    print(f"\n    Phase 2: Price-Proportional Cap (ref=$2000)")
    print(f"    {'Base_cap':>10} {'N':>6} {'Sharpe':>8} {'PnL':>10} "
          f"{'MaxDD':>8} {'WR':>6} {'Worst':>8}")
    print("    " + "-" * 65)

    base_caps = [20, 25, 30, 35]
    phase2 = {}
    for bc in base_caps:
        adjusted = apply_price_proportional_cap(portfolio_all, bc, reference_price=2000)
        st = compute_stats(adjusted, f"PriceCap_b{bc}")
        phase2[bc] = st
        print(f"    ${bc:>9} {st['n']:>6} {st['sharpe']:>8.3f} {st['pnl']:>10.1f} "
              f"{st['max_dd']:>8.1f} {st['wr']:>5.1f}% {st['worst_trade']:>8.2f}",
              flush=True)

    results['phase2_price_cap'] = phase2

    # Phase 3: ATR-proportional cap
    print(f"\n    Phase 3: ATR-Proportional Cap")
    print(f"    {'Mult':>10} {'N':>6} {'Sharpe':>8} {'PnL':>10} "
          f"{'MaxDD':>8} {'WR':>6} {'Worst':>8}")
    print("    " + "-" * 65)

    multipliers = [2.0, 2.5, 3.0, 3.5]
    phase3 = {}
    for mult in multipliers:
        adjusted = apply_atr_proportional_cap(portfolio_all, h1_df, mult)
        st = compute_stats(adjusted, f"ATR_Cap_x{mult}")
        phase3[mult] = st
        print(f"    x{mult:>9.1f} {st['n']:>6} {st['sharpe']:>8.3f} {st['pnl']:>10.1f} "
              f"{st['max_dd']:>8.1f} {st['wr']:>5.1f}% {st['worst_trade']:>8.2f}",
              flush=True)

    results['phase3_atr_cap'] = phase3

    # Phase 4: K-Fold on best
    best_price_cap = max(phase2.items(), key=lambda x: x[1]['sharpe'])[0]
    best_atr_mult = max(phase3.items(), key=lambda x: x[1]['sharpe'])[0]

    print(f"\n    Phase 4: K-Fold on best variants")
    print(f"    Best price-cap: base=${best_price_cap}")
    print(f"    Best ATR-cap:   x{best_atr_mult}", flush=True)

    fold_sharpes_baseline = []
    fold_sharpes_price = []
    fold_sharpes_atr = []

    for fold_name, start, end in FOLDS:
        fs = pd.Timestamp(start); fe = pd.Timestamp(end)
        fold_trades = [t for t in portfolio_all
                       if fs <= t['exit_time'] < fe]
        h1_fold = h1_df[(h1_df.index >= start) & (h1_df.index < end)]

        if not fold_trades:
            fold_sharpes_baseline.append(0.0)
            fold_sharpes_price.append(0.0)
            fold_sharpes_atr.append(0.0)
            continue

        daily_b = trades_to_daily(fold_trades)
        fold_sharpes_baseline.append(sharpe(daily_b))

        price_adj = apply_price_proportional_cap(fold_trades, best_price_cap)
        daily_p = trades_to_daily(price_adj)
        fold_sharpes_price.append(sharpe(daily_p))

        if len(h1_fold) > 20:
            atr_adj = apply_atr_proportional_cap(fold_trades, h1_fold, best_atr_mult)
        else:
            atr_adj = fold_trades
        daily_a = trades_to_daily(atr_adj)
        fold_sharpes_atr.append(sharpe(daily_a))

        print(f"      {fold_name}: Base={fold_sharpes_baseline[-1]:.3f}, "
              f"PriceCap={fold_sharpes_price[-1]:.3f}, "
              f"ATRCap={fold_sharpes_atr[-1]:.3f}", flush=True)

    price_wins = sum(1 for b, p in zip(fold_sharpes_baseline, fold_sharpes_price) if p > b)
    atr_wins = sum(1 for b, a in zip(fold_sharpes_baseline, fold_sharpes_atr) if a > b)

    results['phase4_kfold'] = {
        'best_price_cap': best_price_cap,
        'best_atr_mult': best_atr_mult,
        'baseline_sharpes': [round(s, 3) for s in fold_sharpes_baseline],
        'price_cap_sharpes': [round(s, 3) for s in fold_sharpes_price],
        'atr_cap_sharpes': [round(s, 3) for s in fold_sharpes_atr],
        'price_wins_vs_baseline': price_wins,
        'atr_wins_vs_baseline': atr_wins,
        'mean_baseline': round(float(np.mean(fold_sharpes_baseline)), 3),
        'mean_price': round(float(np.mean(fold_sharpes_price)), 3),
        'mean_atr': round(float(np.mean(fold_sharpes_atr)), 3),
    }
    print(f"\n    Price-cap wins vs Baseline: {price_wins}/6")
    print(f"    ATR-cap wins vs Baseline:   {atr_wins}/6", flush=True)

    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  R164–R166 Batch 2: Friday Close / Directional CB / Adaptive Cap")
    print("=" * 80, flush=True)

    # ─── Load data ───
    print("\n  Loading data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    l8_bundle = DataBundle.load_custom()
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})")
    print(f"  M15: {len(m15_raw)} bars", flush=True)

    # ─── Run all strategies once ───
    print("\n" + "=" * 70)
    print("  Running all 4 strategies (L8_MAX with MaxHold=8)...")
    print("=" * 70, flush=True)
    strat_trades = run_all_strategies(h1_df, l8_bundle)
    for s, tr in strat_trades.items():
        print(f"    {s}: {len(tr)} trades", flush=True)

    portfolio_all = merge_portfolio_trades(strat_trades)
    print(f"\n    Merged portfolio: {len(portfolio_all)} trades")
    if portfolio_all:
        print(f"    Range: {portfolio_all[0]['exit_time'].date()} ~ "
              f"{portfolio_all[-1]['exit_time'].date()}", flush=True)

    # ═════════════════════════════════════════════════════════════
    # R164: Friday Close Force-Exit
    # ═════════════════════════════════════════════════════════════
    r164_results = run_r164(h1_df, l8_bundle, portfolio_all)

    # ═════════════════════════════════════════════════════════════
    # R165: Directional Consecutive Loss Filter
    # ═════════════════════════════════════════════════════════════
    r165_results = run_r165(h1_df, l8_bundle, portfolio_all)

    # ═════════════════════════════════════════════════════════════
    # R166: Gold-Price Adaptive Cap
    # ═════════════════════════════════════════════════════════════
    r166_results = run_r166(h1_df, l8_bundle, portfolio_all)

    # ═════════════════════════════════════════════════════════════
    # Summary & Save
    # ═════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  R164–R166 BATCH COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}", flush=True)

    # Verdicts
    r164_best = r164_results.get('phase4_kfold', {})
    r164_verdict = ("DEPLOY" if r164_best.get('filter_wins', 0) >= 4
                    else "HOLD — Friday filter not robust across folds")

    r165_kf = r165_results.get('phase2_kfold', {})
    r165_verdict = ("DEPLOY" if r165_kf.get('dcb_wins_vs_baseline', 0) >= 4
                    else "HOLD — Directional CB not robust across folds")

    r166_kf = r166_results.get('phase4_kfold', {})
    r166_price_wins = r166_kf.get('price_wins_vs_baseline', 0)
    r166_atr_wins = r166_kf.get('atr_wins_vs_baseline', 0)
    if max(r166_price_wins, r166_atr_wins) >= 4:
        r166_verdict = "DEPLOY — Adaptive cap robust"
    else:
        r166_verdict = "HOLD — Adaptive cap not robust across folds"

    print(f"\n  R164 Verdict: {r164_verdict}")
    print(f"  R165 Verdict: {r165_verdict}")
    print(f"  R166 Verdict: {r166_verdict}", flush=True)

    output = {
        'experiment': 'R164-R166 Batch 2',
        'description': 'Friday Close / Directional CB / Adaptive Cap',
        'config': {
            'pv': PV, 'spread': SPREAD, 'unit_lot': UNIT_LOT,
            'r89_lots': R89_LOTS, 'caps': CAPS,
        },
        'r164_friday_close': r164_results,
        'r165_directional_cb': r165_results,
        'r166_adaptive_cap': r166_results,
        'verdicts': {
            'r164': r164_verdict,
            'r165': r165_verdict,
            'r166': r166_verdict,
        },
        'elapsed_s': round(elapsed, 1),
    }

    with open(OUTPUT_DIR / "r164_r166_results.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_DIR}/r164_r166_results.json", flush=True)


if __name__ == "__main__":
    main()
