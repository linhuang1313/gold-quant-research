#!/usr/bin/env python3
"""
R84 + R85 — Combined: Vol-Sizing for $5000 Account + Triple-Medium TSMOM 8-Stage
=================================================================================
R84: Volatility-normalized position sizing calibrated for $5000 account
     - Fixed lot baseline: 0.03 (current live)
     - Vol-sizing formula: lot = risk_per_trade / (ATR * PV)
     - risk_per_trade = account * risk_pct (e.g. 1% of $5000 = $50)
     - Test different risk_pct: 0.5%, 0.75%, 1%, 1.5%, 2%
     - Apply to PSAR and TSMOM
     - Full per-year breakdown + drawdown analysis

R85: 8-stage validation for Triple-Medium TSMOM (240/480 + 480/720 + 720/1440)
     - Uses StrategyValidator pipeline
     - With dual PBO (perturb + CSCV)

Estimated runtime: ~15-25 minutes.
"""
import sys, os, io, time, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r84_r85")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.30
REALISTIC_SPREAD = 0.88
ACCOUNT = 5000
BASE_LOT = 0.03
PV = 100  # $100 per lot per point for XAUUSD


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


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar'],
            'lot': pos.get('lot', BASE_LOT)}


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


def _yearly_breakdown(trades):
    yearly = {}
    for t in trades:
        yr = pd.Timestamp(t['exit_time']).year
        yearly.setdefault(yr, []).append(t['pnl'])
    result = {}
    for yr in sorted(yearly.keys()):
        pnls = yearly[yr]
        arr = np.array(pnls)
        result[yr] = {
            'pnl': round(float(arr.sum()), 2),
            'trades': len(pnls),
            'win_rate': round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
            'avg_trade': round(float(arr.mean()), 2),
        }
    return result


def _calc_volsized_lot(account, risk_pct, cur_atr, avg_atr, min_lot=0.01, max_lot=0.10):
    """Calculate vol-normalized lot size for a given account and risk %.

    lot = risk_per_trade / (SL_width * PV)
    We use avg_atr as a proxy for SL_width since SL ~ 4.5 * ATR.
    So: lot = (account * risk_pct) / (4.5 * avg_atr * PV)
    Clamp to [min_lot, max_lot].
    """
    if np.isnan(avg_atr) or avg_atr < 0.1:
        return BASE_LOT
    sl_width = 4.5 * avg_atr  # typical SL in price terms
    risk_dollars = account * risk_pct
    lot = risk_dollars / (sl_width * PV)
    return max(min_lot, min(max_lot, round(lot, 2)))


# ═══════════════════════════════════════════════════════════════
# PSAR backtest with vol sizing
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
                if df['High'].iloc[i] > ep:
                    ep = df['High'].iloc[i]; af = min(af+af_start, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], df['High'].iloc[i-1], df['High'].iloc[max(0,i-2)])
            if df['High'].iloc[i] > psar[i]:
                direction[i] = 1; psar[i] = ep; ep = df['High'].iloc[i]; af = af_start
            else:
                direction[i] = -1
                if df['Low'].iloc[i] < ep:
                    ep = df['Low'].iloc[i]; af = min(af+af_start, af_max)
    df['PSAR_dir'] = direction
    df['ATR'] = compute_atr(df)
    return df


def backtest_psar(h1_df, spread=SPREAD, lot=BASE_LOT,
                  sl_atr=4.5, tp_atr=16.0, trail_act_atr=0.20,
                  trail_dist_atr=0.04, max_hold=20,
                  vol_sizing=False, risk_pct=0.01, account=ACCOUNT):
    df = add_psar(h1_df).dropna(subset=['PSAR_dir', 'ATR'])
    c_arr = df['Close'].values; h_arr = df['High'].values; l_arr = df['Low'].values
    psar_dir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    atr_ma = pd.Series(atr).rolling(20).mean().values
    trades = []; pos = None; last_exit = -999

    for i in range(1, n):
        c = c_arr[i]; h = h_arr[i]; lo = l_arr[i]; cur_atr = atr[i]
        if pos is not None:
            lt = pos['lot']; held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (h - pos['entry'] - spread) * lt * PV
                pnl_l = (lo - pos['entry'] - spread) * lt * PV
                pnl_c = (c - pos['entry'] - spread) * lt * PV
            else:
                pnl_h = (pos['entry'] - lo - spread) * lt * PV
                pnl_l = (pos['entry'] - h - spread) * lt * PV
                pnl_c = (pos['entry'] - c - spread) * lt * PV
            tp_val = tp_atr * pos['atr'] * lt * PV
            sl_val = sl_atr * pos['atr'] * lt * PV
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
                                          (ts_p - pos['entry'] - spread) * lt * PV)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo >= ad:
                    ts_p = lo + td
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lt * PV)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if exited:
                if vol_sizing:
                    account_val = account + sum(t['pnl'] for t in trades)
                pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        if psar_dir[i-1] == -1 and psar_dir[i] == 1:
            if vol_sizing:
                acct = account + sum(t['pnl'] for t in trades)
                lt = _calc_volsized_lot(acct, risk_pct, cur_atr, atr_ma[i])
            else:
                lt = lot
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i],
                   'atr': cur_atr, 'lot': lt}
        elif psar_dir[i-1] == 1 and psar_dir[i] == -1:
            if vol_sizing:
                acct = account + sum(t['pnl'] for t in trades)
                lt = _calc_volsized_lot(acct, risk_pct, cur_atr, atr_ma[i])
            else:
                lt = lot
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i],
                   'atr': cur_atr, 'lot': lt}
    return trades


# ═══════════════════════════════════════════════════════════════
# TSMOM backtest with vol sizing (single period)
# ═══════════════════════════════════════════════════════════════

def backtest_tsmom(h1_df, spread=SPREAD, lot=BASE_LOT,
                   fast_period=480, slow_period=720,
                   sl_atr=4.5, tp_atr=6.0, trail_act_atr=0.14,
                   trail_dist_atr=0.025, max_hold=20,
                   vol_sizing=False, risk_pct=0.01, account=ACCOUNT):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df['fast_ma'] = df['Close'].rolling(fast_period).mean()
    df['slow_ma'] = df['Close'].rolling(slow_period).mean()
    df = df.dropna(subset=['ATR', 'fast_ma', 'slow_ma'])
    c_arr = df['Close'].values; h_arr = df['High'].values; l_arr = df['Low'].values
    atr = df['ATR'].values; fast = df['fast_ma'].values; slow = df['slow_ma'].values
    times = df.index; n = len(df)
    atr_ma = pd.Series(atr).rolling(20).mean().values
    trades = []; pos = None; last_exit = -999

    for i in range(1, n):
        c = c_arr[i]; h = h_arr[i]; lo = l_arr[i]; cur_atr = atr[i]
        if pos is not None:
            lt = pos['lot']; held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (h - pos['entry'] - spread) * lt * PV
                pnl_l = (lo - pos['entry'] - spread) * lt * PV
                pnl_c = (c - pos['entry'] - spread) * lt * PV
            else:
                pnl_h = (pos['entry'] - lo - spread) * lt * PV
                pnl_l = (pos['entry'] - h - spread) * lt * PV
                pnl_c = (pos['entry'] - c - spread) * lt * PV
            tp_val = tp_atr * pos['atr'] * lt * PV
            sl_val = sl_atr * pos['atr'] * lt * PV
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
                                          (ts_p - pos['entry'] - spread) * lt * PV)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo >= ad:
                    ts_p = lo + td
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lt * PV)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if exited: pos = None; last_exit = i; continue
            # Reversal exit
            if (pos['dir'] == 'BUY' and fast[i] < slow[i]) or \
               (pos['dir'] == 'SELL' and fast[i] > slow[i]):
                if pos['dir'] == 'BUY': pnl = (c - pos['entry'] - spread) * lt * PV
                else: pnl = (pos['entry'] - c - spread) * lt * PV
                trades.append(_mk(pos, c, times[i], "Reversal", i, pnl))
                pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        if fast[i] > slow[i] and fast[i-1] <= slow[i-1]:
            if vol_sizing:
                acct = account + sum(t['pnl'] for t in trades)
                lt = _calc_volsized_lot(acct, risk_pct, cur_atr, atr_ma[i])
            else:
                lt = lot
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i],
                   'atr': cur_atr, 'lot': lt}
        elif fast[i] < slow[i] and fast[i-1] >= slow[i-1]:
            if vol_sizing:
                acct = account + sum(t['pnl'] for t in trades)
                lt = _calc_volsized_lot(acct, risk_pct, cur_atr, atr_ma[i])
            else:
                lt = lot
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i],
                   'atr': cur_atr, 'lot': lt}
    return trades


# ═══════════════════════════════════════════════════════════════
# Triple-Medium TSMOM
# ═══════════════════════════════════════════════════════════════

def backtest_tsmom_triple(h1_df, spread=SPREAD, lot=BASE_LOT,
                          periods=None,
                          sl_atr=4.5, tp_atr=6.0, trail_act_atr=0.14,
                          trail_dist_atr=0.025, max_hold=20):
    """Triple-period TSMOM: blended signal from 3 MA pairs."""
    if periods is None:
        periods = [(240, 480), (480, 720), (720, 1440)]

    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    signals = []
    for fast, slow in periods:
        f_ma = df['Close'].rolling(fast).mean()
        s_ma = df['Close'].rolling(slow).mean()
        sig = (f_ma - s_ma) / s_ma
        signals.append(sig)
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
            if (pos['dir'] == 'BUY' and sig[i] < 0) or (pos['dir'] == 'SELL' and sig[i] > 0):
                if pos['dir'] == 'BUY': pnl = (c - pos['entry'] - spread) * lot * PV
                else: pnl = (pos['entry'] - c - spread) * lot * PV
                trades.append(_mk(pos, c, times[i], "Reversal", i, pnl))
                pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        if sig[i] > 0 and sig[i-1] <= 0:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr, 'lot': lot}
        elif sig[i] < 0 and sig[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr, 'lot': lot}
    return trades


# ═══════════════════════════════════════════════════════════════
# R84: Vol-Sizing Study for $5000 Account
# ═══════════════════════════════════════════════════════════════

def run_r84(h1_df):
    print("=" * 72)
    print(f"  R84 — Vol-Sizing for ${ACCOUNT:,} Account")
    print(f"  Formula: lot = (account * risk%) / (4.5 * ATR * PV)")
    print(f"  Lot clamp: [0.01, 0.10]")
    print("=" * 72, flush=True)

    # Show what lot sizes the formula produces at different ATR / gold price levels
    print("\n  --- Lot Size Calculator Preview ---")
    for price_label, atr_val in [("Low vol (ATR=5)", 5), ("Normal (ATR=15)", 15),
                                  ("High vol (ATR=30)", 30), ("Crisis (ATR=50)", 50)]:
        for rp in [0.005, 0.01, 0.015, 0.02]:
            lt = _calc_volsized_lot(ACCOUNT, rp, atr_val, atr_val)
            risk_d = ACCOUNT * rp
            print(f"    {price_label}, risk={rp:.1%}: lot={lt:.2f} "
                  f"(risk=${risk_d:.0f}, SL_width={4.5*atr_val:.1f})", flush=True)

    results = {}
    for strat_name, bt_fn in [("PSAR", backtest_psar), ("TSMOM", backtest_tsmom)]:
        print(f"\n  === {strat_name} ===")

        # Fixed lot baseline
        label = f"{strat_name}_Fixed_0.03"
        trades = bt_fn(h1_df, SPREAD, BASE_LOT, vol_sizing=False)
        trades_real = bt_fn(h1_df, REALISTIC_SPREAD, BASE_LOT, vol_sizing=False)
        daily = _trades_to_daily(trades); daily_real = _trades_to_daily(trades_real)
        lots_used = [t.get('lot', BASE_LOT) for t in trades]
        results[label] = {
            'sharpe': round(_sharpe(daily), 2),
            'sharpe_real': round(_sharpe(daily_real), 2),
            'pnl': round(float(daily.sum()), 2) if len(daily) > 0 else 0,
            'pnl_real': round(float(daily_real.sum()), 2) if len(daily_real) > 0 else 0,
            'max_dd': round(_max_dd(daily), 2),
            'max_dd_real': round(_max_dd(daily_real), 2),
            'n_trades': len(trades),
            'win_rate': round(sum(1 for t in trades if t['pnl']>0)/max(len(trades),1)*100, 1),
            'avg_lot': round(float(np.mean(lots_used)), 3),
            'min_lot': round(float(np.min(lots_used)), 3) if lots_used else 0,
            'max_lot': round(float(np.max(lots_used)), 3) if lots_used else 0,
            'dd_pct_of_account': round(_max_dd(daily)/ACCOUNT*100, 1),
            'dd_pct_of_account_real': round(_max_dd(daily_real)/ACCOUNT*100, 1),
            'yearly': _yearly_breakdown(trades),
        }
        print(f"    {label}: Sharpe={results[label]['sharpe']:.2f} "
              f"PnL=${results[label]['pnl']:,.0f} "
              f"DD=${results[label]['max_dd']:,.0f} ({results[label]['dd_pct_of_account']:.1f}% of acct)"
              f" Lot=fixed {BASE_LOT}", flush=True)

        # Vol-sized at different risk levels
        for risk_pct in [0.005, 0.0075, 0.01, 0.015, 0.02]:
            label = f"{strat_name}_VolSize_{risk_pct:.1%}"
            trades = bt_fn(h1_df, SPREAD, BASE_LOT, vol_sizing=True,
                           risk_pct=risk_pct, account=ACCOUNT)
            trades_real = bt_fn(h1_df, REALISTIC_SPREAD, BASE_LOT, vol_sizing=True,
                                risk_pct=risk_pct, account=ACCOUNT)
            daily = _trades_to_daily(trades); daily_real = _trades_to_daily(trades_real)
            lots_used = [t.get('lot', BASE_LOT) for t in trades]
            results[label] = {
                'sharpe': round(_sharpe(daily), 2),
                'sharpe_real': round(_sharpe(daily_real), 2),
                'pnl': round(float(daily.sum()), 2) if len(daily) > 0 else 0,
                'pnl_real': round(float(daily_real.sum()), 2) if len(daily_real) > 0 else 0,
                'max_dd': round(_max_dd(daily), 2),
                'max_dd_real': round(_max_dd(daily_real), 2),
                'n_trades': len(trades),
                'win_rate': round(sum(1 for t in trades if t['pnl']>0)/max(len(trades),1)*100, 1),
                'avg_lot': round(float(np.mean(lots_used)), 3),
                'min_lot': round(float(np.min(lots_used)), 3) if lots_used else 0,
                'max_lot': round(float(np.max(lots_used)), 3) if lots_used else 0,
                'dd_pct_of_account': round(_max_dd(daily)/ACCOUNT*100, 1),
                'dd_pct_of_account_real': round(_max_dd(daily_real)/ACCOUNT*100, 1),
                'risk_pct': risk_pct,
                'yearly': _yearly_breakdown(trades),
            }
            print(f"    {label}: Sharpe={results[label]['sharpe']:.2f} "
                  f"PnL=${results[label]['pnl']:,.0f} "
                  f"DD=${results[label]['max_dd']:,.0f} ({results[label]['dd_pct_of_account']:.1f}% of acct)"
                  f" Lot avg={results[label]['avg_lot']:.3f} "
                  f"[{results[label]['min_lot']:.2f}-{results[label]['max_lot']:.2f}]", flush=True)

    with open(OUTPUT_DIR / "r84_volsize_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  R84 results saved.", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# R85: 8-Stage Validation for Triple-Medium TSMOM
# ═══════════════════════════════════════════════════════════════

def run_r85(h1_df):
    print(f"\n\n{'=' * 72}")
    print("  R85 — 8-Stage Validation: Triple-Medium TSMOM")
    print("  Periods: (240/480) + (480/720) + (720/1440)")
    print("=" * 72, flush=True)

    from backtest.validator import StrategyValidator, ValidatorConfig

    def bt_fn(df, spread, lot):
        return backtest_tsmom_triple(df, spread, lot,
                                     periods=[(240, 480), (480, 720), (720, 1440)])

    def base_fn(df, spread, lot):
        return backtest_tsmom(df, spread, lot, fast_period=480, slow_period=720)

    rng_seed = 42
    def perturb_fn(df, spread, lot, rng):
        def p(v, pct=0.20): return v * (1 + rng.uniform(-pct, pct))
        periods = [
            (max(100, int(p(240))), max(200, int(p(480)))),
            (max(200, int(p(480))), max(300, int(p(720)))),
            (max(400, int(p(720))), max(800, int(p(1440)))),
        ]
        return backtest_tsmom_triple(df, spread, lot, periods=periods,
                                      sl_atr=p(4.5), tp_atr=p(6.0),
                                      trail_act_atr=p(0.14), trail_dist_atr=p(0.025),
                                      max_hold=max(5, int(p(20))))

    def grid_fn(df, spread, lot):
        grid_sharpes = {}
        for sl in [3.0, 4.0, 4.5, 5.0, 6.0]:
            for tp in [4.0, 5.0, 6.0, 8.0]:
                for mh in [10, 15, 20, 30]:
                    label = f"sl{sl}_tp{tp}_mh{mh}"
                    trades = backtest_tsmom_triple(df, spread, lot,
                                                   sl_atr=sl, tp_atr=tp, max_hold=mh)
                    daily = _trades_to_daily(trades)
                    grid_sharpes[label] = _sharpe(daily)
        return grid_sharpes

    def grid_backtest_fn(df, spread, lot):
        """For CSCV PBO: return {label: trades_list}."""
        results = {}
        for sl in [3.0, 4.0, 4.5, 5.0, 6.0]:
            for tp in [4.0, 5.0, 6.0, 8.0]:
                for mh in [10, 15, 20, 30]:
                    label = f"sl{sl}_tp{tp}_mh{mh}"
                    trades = backtest_tsmom_triple(df, spread, lot,
                                                   sl_atr=sl, tp_atr=tp, max_hold=mh)
                    results[label] = trades
        return results

    # 5 SL x 4 TP x 4 MH = 80 combos
    config = ValidatorConfig(
        n_trials_tested=80,
        min_trades=30,
        min_sharpe=0.5,
        realistic_spread=REALISTIC_SPREAD,
        n_param_perturb=200,
        pbo_max_grid_combos=80,
    )

    validator = StrategyValidator(
        name="TSMOM_Triple_Medium",
        backtest_fn=bt_fn,
        spread=SPREAD,
        lot=BASE_LOT,
        config=config,
        output_dir=str(OUTPUT_DIR / "r85_triple_tsmom"),
        h1_df=h1_df,
        base_backtest_fn=base_fn,
        param_perturb_fn=perturb_fn,
        param_grid_fn=grid_fn,
        param_grid_backtest_fn=grid_backtest_fn,
    )

    results = validator.run_all(stop_on_fail=False)

    summary = {}
    for stage, r in sorted(results.items()):
        summary[f"stage{stage}"] = {
            'name': r.name, 'passed': r.passed,
            'sharpe': r.sharpe, 'verdict': r.verdict,
        }
    with open(OUTPUT_DIR / "r85_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  R85 results saved.", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 72)
    print("  R84 + R85 — Vol-Sizing & Triple TSMOM Validation")
    print("=" * 72, flush=True)

    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH
    print("\n  Loading H1 data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})\n")

    r84 = run_r84(h1_df)
    r85 = run_r85(h1_df)

    elapsed = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"  ALL DONE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
