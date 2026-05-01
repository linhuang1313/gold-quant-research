#!/usr/bin/env python3
"""
R83 — Advanced Research: Vol-Sizing, Multi-Period TSMOM, PBO Theory Test
=========================================================================
Three research directions inspired by BetaPlus trend-following literature:

  A. Vol-Normalized Position Sizing
     - Replace fixed lot with: lot = base_lot * (target_vol / realized_vol)
     - Compare Sharpe, DD, risk-adjusted metrics vs fixed lot
     - Professional CTA standard (Moskowitz et al. 2012, Baltas 2015)

  B. Multi-Period Trend Signal Fusion
     - Current TSMOM: single frequency (fast=480, slow=720)
     - Upgrade: blend 3 frequencies (short/medium/long)
     - Hurst, Ooi & Pedersen (2017): 1/3/12 month signals

  C. Theory-Grounded PBO Analysis
     - Run identical PBO on strategies WITH theory basis (PSAR, TSMOM)
       vs a pure data-mined "kitchen sink" strategy
     - Demonstrates that PBO alone cannot distinguish theory-based from noise

Estimated runtime: ~5-8 minutes.
"""
import sys, os, io, time, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r83_advanced_research")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.30
REALISTIC_SPREAD = 0.88
BASE_LOT = 0.03
PV = 100


def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


def _trades_to_daily(trades):
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    if not daily:
        return np.array([])
    return np.array([daily[k] for k in sorted(daily.keys())])


def _sharpe(arr):
    if len(arr) < 10 or arr.std() == 0:
        return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(252))


def _max_dd(arr):
    if len(arr) == 0:
        return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


# ═══════════════════════════════════════════════════════════════
# PART A: Volatility-Normalized Position Sizing
# ═══════════════════════════════════════════════════════════════

def add_psar(df, af_start=0.01, af_max=0.05):
    df = df.copy(); n = len(df)
    psar = np.zeros(n); direction = np.ones(n)
    af = af_start; ep = df['High'].iloc[0]; psar[0] = df['Low'].iloc[0]
    for i in range(1, n):
        prev = psar[i-1]
        if direction[i-1] == 1:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], df['Low'].iloc[i-1], df['Low'].iloc[max(0,i-2)])
            if df['Low'].iloc[i] < psar[i]:
                direction[i] = -1; psar[i] = ep; ep = df['Low'].iloc[i]; af = af_start
            else:
                direction[i] = 1
                if df['High'].iloc[i] > ep: ep = df['High'].iloc[i]; af = min(af+af_start, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], df['High'].iloc[i-1], df['High'].iloc[max(0,i-2)])
            if df['High'].iloc[i] > psar[i]:
                direction[i] = 1; psar[i] = ep; ep = df['High'].iloc[i]; af = af_start
            else:
                direction[i] = -1
                if df['Low'].iloc[i] < ep: ep = df['Low'].iloc[i]; af = min(af+af_start, af_max)
    df['PSAR_dir'] = direction; df['ATR'] = compute_atr(df)
    return df


def backtest_psar_volsized(h1_df, spread=SPREAD, base_lot=BASE_LOT,
                           target_vol_pct=0.01, vol_lookback=20,
                           sl_atr=4.5, tp_atr=16.0, trail_act_atr=0.20,
                           trail_dist_atr=0.04, max_hold=20,
                           use_vol_sizing=True):
    """PSAR with optional volatility-normalized lot sizing.

    target_vol_pct: target daily PnL volatility as fraction of notional
    vol_lookback: lookback period for realized vol estimation (ATR-based)
    """
    df = add_psar(h1_df).dropna(subset=['PSAR_dir', 'ATR'])
    c_arr = df['Close'].values; h_arr = df['High'].values; l_arr = df['Low'].values
    psar_dir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999

    # Precompute rolling ATR for vol sizing
    atr_ma = pd.Series(atr).rolling(vol_lookback).mean().values

    for i in range(1, n):
        c = c_arr[i]; h = h_arr[i]; lo = l_arr[i]; cur_atr = atr[i]
        if pos is not None:
            lot = pos['lot']
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (h - pos['entry'] - spread) * lot * PV
                pnl_l = (lo - pos['entry'] - spread) * lot * PV
                pnl_c = (c - pos['entry'] - spread) * lot * PV
            else:
                pnl_h = (pos['entry'] - lo - spread) * lot * PV
                pnl_l = (pos['entry'] - h - spread) * lot * PV
                pnl_c = (pos['entry'] - c - spread) * lot * PV
            tp_val = tp_atr * pos['atr'] * lot * PV
            sl_val = sl_atr * pos['atr'] * lot * PV
            exited = False
            if pnl_h >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); exited = True
            else:
                ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
                    ts_p = h - td
                    if lo <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos['entry'] - spread) * lot * PV)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo >= ad:
                    ts_p = lo + td
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * PV)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if exited: pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        if psar_dir[i-1] == -1 and psar_dir[i] == 1:
            lot = _calc_lot(use_vol_sizing, base_lot, target_vol_pct, cur_atr, atr_ma[i], c)
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i],
                   'atr': cur_atr, 'lot': lot}
        elif psar_dir[i-1] == 1 and psar_dir[i] == -1:
            lot = _calc_lot(use_vol_sizing, base_lot, target_vol_pct, cur_atr, atr_ma[i], c)
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i],
                   'atr': cur_atr, 'lot': lot}
    return trades


def _calc_lot(use_vol_sizing, base_lot, target_vol_pct, cur_atr, avg_atr, price):
    """Calculate lot size: fixed or vol-normalized."""
    if not use_vol_sizing or np.isnan(avg_atr) or avg_atr < 0.1:
        return base_lot
    # target_vol_pct of capital per ATR unit of risk
    # lot = target_risk / (ATR * point_value)
    # Use avg_atr for stability, clamp lot to [0.01, 0.10]
    target_dollar_risk = target_vol_pct * price * PV  # e.g. 1% of notional
    lot = target_dollar_risk / (avg_atr * PV)
    return max(0.01, min(0.10, round(lot, 2)))


def backtest_tsmom_volsized(h1_df, spread=SPREAD, base_lot=BASE_LOT,
                            target_vol_pct=0.01, vol_lookback=20,
                            fast_period=480, slow_period=720,
                            sl_atr=4.5, tp_atr=6.0, trail_act_atr=0.14,
                            trail_dist_atr=0.025, max_hold=20,
                            use_vol_sizing=True):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df['fast_ma'] = df['Close'].rolling(fast_period).mean()
    df['slow_ma'] = df['Close'].rolling(slow_period).mean()
    df = df.dropna(subset=['ATR', 'fast_ma', 'slow_ma'])
    c_arr = df['Close'].values; h_arr = df['High'].values; l_arr = df['Low'].values
    atr = df['ATR'].values; fast = df['fast_ma'].values; slow = df['slow_ma'].values
    times = df.index; n = len(df)
    atr_ma = pd.Series(atr).rolling(vol_lookback).mean().values
    trades = []; pos = None; last_exit = -999

    for i in range(1, n):
        c = c_arr[i]; h = h_arr[i]; lo = l_arr[i]; cur_atr = atr[i]
        if pos is not None:
            lot = pos['lot']; held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (h - pos['entry'] - spread) * lot * PV
                pnl_l = (lo - pos['entry'] - spread) * lot * PV
                pnl_c = (c - pos['entry'] - spread) * lot * PV
            else:
                pnl_h = (pos['entry'] - lo - spread) * lot * PV
                pnl_l = (pos['entry'] - h - spread) * lot * PV
                pnl_c = (pos['entry'] - c - spread) * lot * PV
            tp_val = tp_atr * pos['atr'] * lot * PV
            sl_val = sl_atr * pos['atr'] * lot * PV
            exited = False
            if pnl_h >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); exited = True
            else:
                ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
                    ts_p = h - td
                    if lo <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos['entry'] - spread) * lot * PV)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo >= ad:
                    ts_p = lo + td
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * PV)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if exited: pos = None; last_exit = i; continue
            if (pos['dir'] == 'BUY' and fast[i] < slow[i]) or \
               (pos['dir'] == 'SELL' and fast[i] > slow[i]):
                if pos['dir'] == 'BUY': pnl = (c - pos['entry'] - spread) * lot * PV
                else: pnl = (pos['entry'] - c - spread) * lot * PV
                trades.append(_mk(pos, c, times[i], "Reversal", i, pnl))
                pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        if fast[i] > slow[i] and fast[i-1] <= slow[i-1]:
            lot = _calc_lot(use_vol_sizing, base_lot, target_vol_pct, cur_atr, atr_ma[i], c)
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i],
                   'atr': cur_atr, 'lot': lot}
        elif fast[i] < slow[i] and fast[i-1] >= slow[i-1]:
            lot = _calc_lot(use_vol_sizing, base_lot, target_vol_pct, cur_atr, atr_ma[i], c)
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i],
                   'atr': cur_atr, 'lot': lot}
    return trades


def run_part_a(h1_df):
    """Compare fixed vs vol-sized for PSAR and TSMOM."""
    print("=" * 72)
    print("  PART A: Volatility-Normalized Position Sizing")
    print("  Formula: lot = target_risk / (ATR * PV)")
    print("  Target: 1% of notional per ATR unit")
    print("=" * 72, flush=True)

    results = {}
    for name, bt_fn in [("PSAR", backtest_psar_volsized), ("TSMOM", backtest_tsmom_volsized)]:
        print(f"\n  --- {name} ---")
        for mode, use_vs in [("Fixed_0.03", False), ("VolSized_1pct", True)]:
            trades = bt_fn(h1_df, SPREAD, BASE_LOT, use_vol_sizing=use_vs)
            daily = _trades_to_daily(trades)
            sh = _sharpe(daily)
            pnl = float(daily.sum()) if len(daily) > 0 else 0
            dd = _max_dd(daily)
            n_trades = len(trades)

            # Per-trade lot distribution
            lots = [t.get('lot', BASE_LOT) if 'lot' in t else BASE_LOT for t in trades]
            # Since we store lot in pos, not in trade output, approximate from PnL
            wr = sum(1 for t in trades if t['pnl'] > 0) / max(n_trades, 1) * 100

            # Calmar ratio = annualized return / max DD
            ann_ret = float(daily.mean() * 252) if len(daily) > 0 else 0
            calmar = ann_ret / dd if dd > 0 else 0

            # Realistic spread
            trades_real = bt_fn(h1_df, REALISTIC_SPREAD, BASE_LOT, use_vol_sizing=use_vs)
            daily_real = _trades_to_daily(trades_real)
            sh_real = _sharpe(daily_real)
            pnl_real = float(daily_real.sum()) if len(daily_real) > 0 else 0

            results[f"{name}_{mode}"] = {
                'sharpe': round(sh, 2), 'sharpe_real': round(sh_real, 2),
                'pnl': round(pnl, 2), 'pnl_real': round(pnl_real, 2),
                'max_dd': round(dd, 2), 'n_trades': n_trades,
                'win_rate': round(wr, 1), 'calmar': round(calmar, 4),
                'ann_return': round(ann_ret, 2),
            }
            print(f"    {mode:>20}: Sharpe={sh:.2f} (real={sh_real:.2f}) "
                  f"PnL=${pnl:,.0f} (real=${pnl_real:,.0f}) "
                  f"DD=${dd:,.0f} Trades={n_trades} Win={wr:.1f}% "
                  f"Calmar={calmar:.3f}", flush=True)

    # Vol-sizing at different target levels
    print(f"\n  --- Target Vol Sensitivity ---")
    for target in [0.005, 0.008, 0.01, 0.015, 0.02]:
        trades = backtest_psar_volsized(h1_df, SPREAD, BASE_LOT,
                                        target_vol_pct=target, use_vol_sizing=True)
        daily = _trades_to_daily(trades)
        sh = _sharpe(daily); pnl = float(daily.sum()) if len(daily) > 0 else 0
        dd = _max_dd(daily)
        results[f"PSAR_target_{target}"] = {
            'sharpe': round(sh, 2), 'pnl': round(pnl, 2),
            'max_dd': round(dd, 2), 'n_trades': len(trades),
        }
        print(f"    target={target:.3f}: Sharpe={sh:.2f} PnL=${pnl:,.0f} DD=${dd:,.0f}", flush=True)

    return results


# ═══════════════════════════════════════════════════════════════
# PART B: Multi-Period Trend Signal Fusion
# ═══════════════════════════════════════════════════════════════

def backtest_tsmom_multi(h1_df, spread=SPREAD, lot=BASE_LOT,
                         periods=None,
                         sl_atr=4.5, tp_atr=6.0, trail_act_atr=0.14,
                         trail_dist_atr=0.025, max_hold=20):
    """Multi-period TSMOM: blend signals from multiple frequency pairs.

    periods: list of (fast, slow) tuples. Signal = average of all.
    Entry when blended signal crosses 0.
    """
    if periods is None:
        periods = [(120, 240), (480, 720), (960, 1440)]  # ~1wk, ~1mo, ~3mo

    df = h1_df.copy()
    df['ATR'] = compute_atr(df)

    signals = []
    max_lookback = 0
    for fast, slow in periods:
        f_ma = df['Close'].rolling(fast).mean()
        s_ma = df['Close'].rolling(slow).mean()
        # Normalized signal: (fast - slow) / slow, range ~[-1, 1]
        sig = (f_ma - s_ma) / s_ma
        signals.append(sig)
        max_lookback = max(max_lookback, slow)

    df['blend_signal'] = sum(signals) / len(signals)
    df = df.dropna(subset=['ATR', 'blend_signal'])

    c_arr = df['Close'].values; h_arr = df['High'].values; l_arr = df['Low'].values
    atr = df['ATR'].values; sig = df['blend_signal'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999

    for i in range(1, n):
        c = c_arr[i]; h = h_arr[i]; lo = l_arr[i]; cur_atr = atr[i]
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (h - pos['entry'] - spread) * lot * PV
                pnl_l = (lo - pos['entry'] - spread) * lot * PV
                pnl_c = (c - pos['entry'] - spread) * lot * PV
            else:
                pnl_h = (pos['entry'] - lo - spread) * lot * PV
                pnl_l = (pos['entry'] - h - spread) * lot * PV
                pnl_c = (pos['entry'] - c - spread) * lot * PV
            tp_val = tp_atr * pos['atr'] * lot * PV
            sl_val = sl_atr * pos['atr'] * lot * PV
            exited = False
            if pnl_h >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); exited = True
            else:
                ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
                    ts_p = h - td
                    if lo <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos['entry'] - spread) * lot * PV)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo >= ad:
                    ts_p = lo + td
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * PV)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if exited: pos = None; last_exit = i; continue
            # Reversal: blended signal flips
            if (pos['dir'] == 'BUY' and sig[i] < 0) or (pos['dir'] == 'SELL' and sig[i] > 0):
                if pos['dir'] == 'BUY': pnl = (c - pos['entry'] - spread) * lot * PV
                else: pnl = (pos['entry'] - c - spread) * lot * PV
                trades.append(_mk(pos, c, times[i], "Reversal", i, pnl))
                pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        if sig[i] > 0 and sig[i-1] <= 0:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif sig[i] < 0 and sig[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades


def backtest_tsmom_single(h1_df, spread=SPREAD, lot=BASE_LOT,
                          fast_period=480, slow_period=720,
                          sl_atr=4.5, tp_atr=6.0, trail_act_atr=0.14,
                          trail_dist_atr=0.025, max_hold=20):
    """Original single-frequency TSMOM for comparison."""
    return backtest_tsmom_volsized(h1_df, spread, lot, use_vol_sizing=False,
                                   fast_period=fast_period, slow_period=slow_period,
                                   sl_atr=sl_atr, tp_atr=tp_atr,
                                   trail_act_atr=trail_act_atr, trail_dist_atr=trail_dist_atr,
                                   max_hold=max_hold)


def run_part_b(h1_df):
    """Compare single vs multi-period TSMOM."""
    print(f"\n\n{'=' * 72}")
    print("  PART B: Multi-Period Trend Signal Fusion")
    print("  Hurst, Ooi & Pedersen (2017) approach")
    print("=" * 72, flush=True)

    configs = {
        'Single (480/720)': {'periods': None, 'fn': 'single'},
        'Dual (240/480 + 480/720)': {'periods': [(240, 480), (480, 720)]},
        'Triple-Short (120/240 + 480/720 + 960/1440)': {'periods': [(120, 240), (480, 720), (960, 1440)]},
        'Triple-Medium (240/480 + 480/720 + 720/1440)': {'periods': [(240, 480), (480, 720), (720, 1440)]},
        'Triple-Long (480/720 + 720/1440 + 1440/2880)': {'periods': [(480, 720), (720, 1440), (1440, 2880)]},
    }

    results = {}
    for label, cfg in configs.items():
        if cfg.get('fn') == 'single':
            trades = backtest_tsmom_single(h1_df, SPREAD, BASE_LOT)
            trades_real = backtest_tsmom_single(h1_df, REALISTIC_SPREAD, BASE_LOT)
        else:
            trades = backtest_tsmom_multi(h1_df, SPREAD, BASE_LOT, periods=cfg['periods'])
            trades_real = backtest_tsmom_multi(h1_df, REALISTIC_SPREAD, BASE_LOT, periods=cfg['periods'])

        daily = _trades_to_daily(trades); daily_real = _trades_to_daily(trades_real)
        sh = _sharpe(daily); sh_real = _sharpe(daily_real)
        pnl = float(daily.sum()) if len(daily) > 0 else 0
        pnl_real = float(daily_real.sum()) if len(daily_real) > 0 else 0
        dd = _max_dd(daily); dd_real = _max_dd(daily_real)
        wr = sum(1 for t in trades if t['pnl'] > 0) / max(len(trades), 1) * 100

        # Correlation with single-period TSMOM
        corr_note = ""
        if cfg.get('fn') != 'single':
            base_trades = backtest_tsmom_single(h1_df, SPREAD, BASE_LOT)
            base_daily = _trades_to_daily(base_trades)
            # Align dates
            all_dates = sorted(set(
                [pd.Timestamp(t['exit_time']).date() for t in trades] +
                [pd.Timestamp(t['exit_time']).date() for t in base_trades]
            ))
            d1 = {}; d2 = {}
            for t in trades: d1[pd.Timestamp(t['exit_time']).date()] = d1.get(pd.Timestamp(t['exit_time']).date(), 0) + t['pnl']
            for t in base_trades: d2[pd.Timestamp(t['exit_time']).date()] = d2.get(pd.Timestamp(t['exit_time']).date(), 0) + t['pnl']
            a1 = np.array([d1.get(d, 0) for d in all_dates])
            a2 = np.array([d2.get(d, 0) for d in all_dates])
            if len(a1) > 10 and a1.std() > 0 and a2.std() > 0:
                corr = float(np.corrcoef(a1, a2)[0, 1])
                corr_note = f" Corr_w_single={corr:.3f}"

        results[label] = {
            'sharpe': round(sh, 2), 'sharpe_real': round(sh_real, 2),
            'pnl': round(pnl, 2), 'pnl_real': round(pnl_real, 2),
            'max_dd': round(dd, 2), 'max_dd_real': round(dd_real, 2),
            'n_trades': len(trades), 'win_rate': round(wr, 1),
        }
        print(f"    {label:>45}: Sharpe={sh:.2f} (real={sh_real:.2f}) "
              f"PnL=${pnl:,.0f} DD=${dd:,.0f} Trades={len(trades)}{corr_note}", flush=True)

    return results


# ═══════════════════════════════════════════════════════════════
# PART C: PBO & Theory Basis — Data-Mined vs Theory-Grounded
# ═══════════════════════════════════════════════════════════════

def backtest_kitchen_sink(h1_df, spread=SPREAD, lot=BASE_LOT,
                          rsi_period=14, rsi_threshold=50,
                          bb_period=20, bb_mult=2.0,
                          sl_atr=4.5, tp_atr=6.0, max_hold=20):
    """A deliberately over-parameterized 'kitchen sink' strategy.
    Combines RSI + Bollinger + hour-of-day filter + day-of-week filter.
    No theoretical basis — pure data mining.
    """
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    # Bollinger
    df['BB_mid'] = df['Close'].rolling(bb_period).mean()
    df['BB_std'] = df['Close'].rolling(bb_period).std()
    df['BB_upper'] = df['BB_mid'] + bb_mult * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - bb_mult * df['BB_std']
    df = df.dropna(subset=['ATR', 'RSI', 'BB_upper'])

    c_arr = df['Close'].values; h_arr = df['High'].values; l_arr = df['Low'].values
    atr = df['ATR'].values; rsi = df['RSI'].values
    bb_u = df['BB_upper'].values; bb_l = df['BB_lower'].values
    hours = df.index.hour; dows = df.index.dayofweek
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999

    for i in range(1, n):
        c = c_arr[i]; h = h_arr[i]; lo = l_arr[i]; cur_atr = atr[i]
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_c = (c - pos['entry'] - spread) * lot * PV
            else:
                pnl_c = (pos['entry'] - c - spread) * lot * PV
            tp_val = tp_atr * pos['atr'] * lot * PV
            sl_val = sl_atr * pos['atr'] * lot * PV
            if pnl_c >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); pos = None; last_exit = i; continue
            if pnl_c <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); pos = None; last_exit = i; continue
            if held >= max_hold:
                trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        # "Kitchen sink" entry: RSI + Bollinger + hour + day filters
        if hours[i] in (10, 14) and dows[i] in (1, 2, 3):
            if rsi[i] > rsi_threshold and c > bb_u[i]:
                pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
            elif rsi[i] < (100 - rsi_threshold) and c < bb_l[i]:
                pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades


def run_part_c(h1_df):
    """Compare PBO for theory-grounded vs data-mined strategies."""
    print(f"\n\n{'=' * 72}")
    print("  PART C: PBO — Theory-Grounded vs Data-Mined Strategy")
    print("  Hypothesis: PBO alone cannot capture theoretical validity")
    print("=" * 72, flush=True)

    from backtest.stats import compute_pbo

    strategies = {
        'PSAR (theory: PSAR indicator)': lambda df, sp, lt: backtest_psar_volsized(df, sp, lt, use_vol_sizing=False),
        'TSMOM (theory: time-series momentum)': lambda df, sp, lt: backtest_tsmom_single(df, sp, lt),
        'KitchenSink (no theory: RSI+BB+hour+day)': lambda df, sp, lt: backtest_kitchen_sink(df, sp, lt),
    }

    results = {}
    rng = np.random.RandomState(42)
    for label, bt_fn in strategies.items():
        trades = bt_fn(h1_df, SPREAD, BASE_LOT)
        daily = _trades_to_daily(trades)
        sh = _sharpe(daily)
        pnl = float(daily.sum()) if len(daily) > 0 else 0

        # Generate perturbation variants for PBO
        perturb_dailies = {'SELECTED': daily.tolist()}
        for j in range(100):
            def p(base, pct=0.25): return base * (1 + rng.uniform(-pct, pct))
            if 'PSAR' in label:
                pt = backtest_psar_volsized(h1_df, SPREAD, BASE_LOT, use_vol_sizing=False,
                                             sl_atr=p(4.5), tp_atr=p(16.0),
                                             trail_act_atr=p(0.20), trail_dist_atr=p(0.04),
                                             max_hold=max(5, int(p(20))))
            elif 'TSMOM' in label:
                pt = backtest_tsmom_single(h1_df, SPREAD, BASE_LOT,
                                           fast_period=max(100, int(p(480))),
                                           slow_period=max(200, int(p(720))),
                                           sl_atr=p(4.5), tp_atr=p(6.0))
            else:
                pt = backtest_kitchen_sink(h1_df, SPREAD, BASE_LOT,
                                            rsi_period=max(5, int(p(14))),
                                            rsi_threshold=p(50),
                                            bb_period=max(10, int(p(20))),
                                            bb_mult=p(2.0),
                                            sl_atr=p(4.5), tp_atr=p(6.0))
            pd_arr = _trades_to_daily(pt)
            if len(pd_arr) > 16:
                perturb_dailies[f"v{j}"] = pd_arr.tolist()

        pbo = compute_pbo(perturb_dailies, n_partitions=8)

        results[label] = {
            'sharpe': round(sh, 2), 'pnl': round(pnl, 2),
            'n_trades': len(trades),
            'pbo': round(pbo.get('pbo', 0), 4),
            'pbo_risk': pbo.get('overfit_risk', 'N/A'),
            'n_combos': pbo.get('n_combinations', 0),
        }
        print(f"    {label:>50}:", flush=True)
        print(f"      Sharpe={sh:.2f} PnL=${pnl:,.0f} Trades={len(trades)} "
              f"PBO={pbo.get('pbo', 0):.1%} ({pbo.get('overfit_risk', 'N/A')})", flush=True)

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 72)
    print("  R83 — Advanced Research Suite")
    print("  Inspired by BetaPlus Trend Following Literature")
    print("=" * 72, flush=True)

    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH
    print("\n  Loading H1 data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})\n")

    result_a = run_part_a(h1_df)
    result_b = run_part_b(h1_df)
    result_c = run_part_c(h1_df)

    elapsed = time.time() - t0
    print(f"\n\n{'=' * 72}")
    print(f"  R83 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 72}")

    combined = {
        'part_a_vol_sizing': result_a,
        'part_b_multi_period': result_b,
        'part_c_pbo_theory': result_c,
        'elapsed_s': round(elapsed, 1),
    }
    with open(OUTPUT_DIR / "r83_results.json", 'w') as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"  Results saved to {OUTPUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
