#!/usr/bin/env python3
"""
R71 — Pure Out-of-Sample Validation (2024-2026 Holdout)
=========================================================
Train period: 2015-01-01 to 2023-12-31 (9 years)
Holdout period: 2024-01-01 to 2026-04-27 (2.3 years, untouched)

Uses the EXACT EA-deployed parameters. No re-optimization.
Measures true forward performance for all 4 strategies.

Estimated runtime: ~5 minutes.
"""
import sys, os, io, time, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r71_oos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.30
REALISTIC_SPREAD = 0.88
LOT = 0.03

HOLDOUT_START = "2024-01-01"

# Full-sample Sharpe baselines from R69/R70 for decay comparison
BASELINES = {
    "PSAR_EA":    {"sharpe": 4.96, "sharpe_real": 2.67},
    "SESS_BO_EA": {"sharpe": 6.48, "sharpe_real": 3.91},
    "TSMOM":      {"sharpe": 4.11, "sharpe_real": 1.92},
    "L8_MAX":     {"sharpe": 7.88, "sharpe_real": None},
}

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
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


def _trades_to_daily(trades):
    if not trades:
        return np.array([0.0])
    df = pd.DataFrame(trades)
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    daily = df.set_index('exit_time').resample('D')['pnl'].sum()
    return daily.values


def _sharpe(daily, ann=252):
    if len(daily) < 2:
        return 0.0
    m = np.mean(daily); s = np.std(daily, ddof=1)
    if s == 0: return 0.0
    return float(m / s * np.sqrt(ann))


def _compute_metrics(trades, label=""):
    """Compute comprehensive metrics from a trade list."""
    if not trades:
        return {"n_trades": 0, "total_pnl": 0, "sharpe": 0, "max_dd": 0,
                "win_rate": 0, "avg_pnl": 0, "monthly_pnl": {}}

    pnls = [t['pnl'] for t in trades]
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(dd.min())

    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / len(pnls) * 100

    daily = _trades_to_daily(trades)
    sharpe = _sharpe(daily)

    # Monthly breakdown
    df = pd.DataFrame(trades)
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    monthly = df.set_index('exit_time').resample('ME')['pnl'].sum()
    monthly_dict = {str(k.strftime('%Y-%m')): round(float(v), 2) for k, v in monthly.items()}

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        r = t.get('reason', 'unknown')
        reasons[r] = reasons.get(r, 0) + 1

    # Buy/Sell breakdown
    buy_pnl = sum(t['pnl'] for t in trades if t['dir'] == 'BUY')
    sell_pnl = sum(t['pnl'] for t in trades if t['dir'] == 'SELL')

    return {
        "n_trades": len(trades),
        "total_pnl": round(float(cum[-1]), 2),
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd, 2),
        "win_rate": round(win_rate, 1),
        "avg_pnl": round(float(np.mean(pnls)), 2),
        "buy_pnl": round(buy_pnl, 2),
        "sell_pnl": round(sell_pnl, 2),
        "exit_reasons": reasons,
        "monthly_pnl": monthly_dict,
    }


# ═══════════════════════════════════════════════════════════════
# PSAR — EA parameters
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


def backtest_psar(h1_df, spread=SPREAD, lot=LOT,
                  sl_atr=4.5, tp_atr=16.0, trail_act_atr=0.20,
                  trail_dist_atr=0.04, max_hold=20):
    df = add_psar(h1_df).dropna(subset=['PSAR_dir', 'ATR'])
    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    psar_dir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df); last_exit = -999
    for i in range(1, n):
        c = close[i]; h = high[i]; lo_v = low[i]; cur_atr = atr[i]
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (h - pos['entry'] - spread) * lot * 100
                pnl_l = (lo_v - pos['entry'] - spread) * lot * 100
                pnl_c = (c - pos['entry'] - spread) * lot * 100
            else:
                pnl_h = (pos['entry'] - lo_v - spread) * lot * 100
                pnl_l = (pos['entry'] - h - spread) * lot * 100
                pnl_c = (pos['entry'] - c - spread) * lot * 100
            tp_val = tp_atr * pos['atr'] * lot * 100
            sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); exited = True
            else:
                ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
                    ts_p = h - td
                    if lo_v <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos['entry'] - spread) * lot * 100)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
                    ts_p = lo_v + td
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * 100)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if exited: pos = None; last_exit = i; continue
        if pos is not None: continue
        if i - last_exit < 2: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        prev_d = psar_dir[i-1]; cur_d = psar_dir[i]
        if prev_d == -1 and cur_d == 1:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif prev_d == 1 and cur_d == -1:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades


# ═══════════════════════════════════════════════════════════════
# SESS_BO — EA parameters
# ═══════════════════════════════════════════════════════════════

def backtest_sess_bo(h1_df, spread=SPREAD, lot=LOT,
                     session="peak_12_14", lookback_bars=4,
                     sl_atr=4.5, tp_atr=4.0, trail_act_atr=0.14,
                     trail_dist_atr=0.025, max_hold=20):
    SESSION_DEFS = {"asian": (0,7), "london": (8,11), "ny_peak": (12,16),
                    "late": (17,23), "peak_12_14": (12,14)}
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour
    times = df.index; n = len(df); last_exit = -999
    sess_start, sess_end = SESSION_DEFS[session]
    for i in range(lookback_bars, n):
        c = close[i]; h = high[i]; lo_v = low[i]; cur_atr = atr[i]; cur_hour = hours[i]
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (h - pos['entry'] - spread) * lot * 100
                pnl_l = (lo_v - pos['entry'] - spread) * lot * 100
                pnl_c = (c - pos['entry'] - spread) * lot * 100
            else:
                pnl_h = (pos['entry'] - lo_v - spread) * lot * 100
                pnl_l = (pos['entry'] - h - spread) * lot * 100
                pnl_c = (pos['entry'] - c - spread) * lot * 100
            tp_val = tp_atr * pos['atr'] * lot * 100
            sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); exited = True
            else:
                act_dist = trail_act_atr * pos['atr']; trail_d = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= act_dist:
                    ts_p = h - trail_d
                    if lo_v <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos['entry'] - spread) * lot * 100)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= act_dist:
                    ts_p = lo_v + trail_d
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * 100)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if exited: pos = None; last_exit = i; continue
        if pos is not None: continue
        if i - last_exit < 2: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        if cur_hour != sess_start: continue
        if i > 0 and hours[i-1] == sess_start: continue
        range_high = max(high[i - lookback_bars:i])
        range_low  = min(low[i - lookback_bars:i])
        if c > range_high:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif c < range_low:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades


# ═══════════════════════════════════════════════════════════════
# TSMOM — EA parameters
# ═══════════════════════════════════════════════════════════════

def backtest_tsmom(h1_df, spread=SPREAD, lot=LOT,
                   fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
                   trail_act_atr=0.14, trail_dist_atr=0.025, max_hold=20):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(close)
    weights = [(fast, 0.5), (slow, 0.5)]
    max_lb = max(lb for lb, _ in weights)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        for lb, w in weights:
            if i >= lb:
                s += w * np.sign(close[i] / close[i - lb] - 1.0)
        score[i] = s
    trades = []; pos = None; last_exit = -999
    for i in range(max_lb + 1, n):
        c = close[i]; h = high[i]; lo_v = low[i]; cur_atr = atr[i]
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (h - pos['entry'] - spread) * lot * 100
                pnl_l = (lo_v - pos['entry'] - spread) * lot * 100
                pnl_c = (c - pos['entry'] - spread) * lot * 100
            else:
                pnl_h = (pos['entry'] - lo_v - spread) * lot * 100
                pnl_l = (pos['entry'] - h - spread) * lot * 100
                pnl_c = (pos['entry'] - c - spread) * lot * 100
            tp_val = tp_atr * pos['atr'] * lot * 100
            sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); exited = True
            else:
                ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
                    ts_p = h - td
                    if lo_v <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos['entry'] - spread) * lot * 100)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
                    ts_p = lo_v + td
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * 100)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if not exited and not np.isnan(score[i]):
                if pos['dir'] == 'BUY' and score[i] < 0:
                    trades.append(_mk(pos, c, times[i], "Reversal", i, pnl_c)); exited = True
                elif pos['dir'] == 'SELL' and score[i] > 0:
                    trades.append(_mk(pos, c, times[i], "Reversal", i, pnl_c)); exited = True
            if exited: pos = None; last_exit = i; continue
        if pos is not None or i - last_exit < 2: continue
        if np.isnan(score[i]) or np.isnan(cur_atr) or cur_atr < 0.1: continue
        if score[i] > 0 and score[i - 1] <= 0:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif score[i] < 0 and score[i - 1] >= 0:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades


# ═══════════════════════════════════════════════════════════════
# L8_MAX — BacktestEngine
# ═══════════════════════════════════════════════════════════════

_L8_DATA = None

def _load_l8_data():
    global _L8_DATA
    if _L8_DATA is None:
        from backtest.runner import DataBundle
        _L8_DATA = DataBundle.load_default()
    return _L8_DATA


def _run_l8_engine(data_bundle, spread, lot, maxloss_cap=37):
    from backtest.runner import run_variant, LIVE_PARITY_KWARGS
    import backtest.runner as runner_mod
    try:
        from indicators import signals as signals_mod
    except ImportError:
        signals_mod = None

    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': maxloss_cap,
          'spread_cost': spread, 'initial_capital': 2000,
          'min_lot_size': lot, 'max_lot_size': lot}

    if hasattr(runner_mod, 'get_orb_strategy'):
        runner_mod.get_orb_strategy().reset_daily()
    if signals_mod and hasattr(signals_mod, '_friday_close_price'):
        signals_mod._friday_close_price = None
        signals_mod._gap_traded_today = False

    result = run_variant(data_bundle, "L8_MAX", verbose=False, **kw)
    raw = result.get('_trades', [])
    trades = []
    for t in raw:
        pnl = t.pnl if hasattr(t, 'pnl') else t.get('pnl', 0)
        ext = t.exit_time if hasattr(t, 'exit_time') else t.get('exit_time', '')
        ent = t.entry_time if hasattr(t, 'entry_time') else t.get('entry_time', ext)
        d = t.direction if hasattr(t, 'direction') else t.get('dir', 'BUY')
        trades.append({'pnl': pnl, 'exit_time': ext, 'entry_time': ent, 'dir': d})
    return trades


def l8_backtest(h1_df, spread, lot):
    full_data = _load_l8_data()
    if len(h1_df) < len(full_data.h1_df):
        start = h1_df.index[0]
        end = h1_df.index[-1]
        data = full_data.slice(str(start), str(end + pd.Timedelta(hours=1)))
    else:
        data = full_data
    return _run_l8_engine(data, spread, lot)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0_total = time.time()
    print("=" * 72)
    print("  R71 — Pure Out-of-Sample Validation")
    print(f"  Train: 2015-01-01 ~ 2023-12-31 | Holdout: {HOLDOUT_START} ~ end")
    print("  Parameters: EA-deployed (NO re-optimization)")
    print("=" * 72, flush=True)

    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH
    print("\n  Loading H1 data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})")

    holdout_ts = pd.Timestamp(HOLDOUT_START, tz='UTC')
    h1_train = h1_df[h1_df.index < holdout_ts]
    h1_holdout = h1_df[h1_df.index >= holdout_ts]
    print(f"  Train:   {len(h1_train)} bars ({h1_train.index[0]} ~ {h1_train.index[-1]})")
    print(f"  Holdout: {len(h1_holdout)} bars ({h1_holdout.index[0]} ~ {h1_holdout.index[-1]})")
    print(flush=True)

    # H1 strategies
    h1_strategies = [
        ("PSAR_EA", backtest_psar),
        ("SESS_BO_EA", backtest_sess_bo),
        ("TSMOM", backtest_tsmom),
    ]

    all_results = {}

    for name, fn in h1_strategies:
        print(f"\n{'#' * 60}")
        print(f"  {name}")
        print(f"{'#' * 60}")

        t0 = time.time()

        # Full sample
        trades_full = fn(h1_df, SPREAD, LOT)
        trades_full_real = fn(h1_df, REALISTIC_SPREAD, LOT)

        # Train only
        trades_train = fn(h1_train, SPREAD, LOT)
        trades_train_real = fn(h1_train, REALISTIC_SPREAD, LOT)

        # Holdout only — IMPORTANT: feed full data so indicators warm up,
        # but only count trades that occur in holdout period
        trades_all = fn(h1_df, SPREAD, LOT)
        trades_holdout = [t for t in trades_all
                          if pd.Timestamp(t['exit_time']) >= holdout_ts]
        trades_all_real = fn(h1_df, REALISTIC_SPREAD, LOT)
        trades_holdout_real = [t for t in trades_all_real
                               if pd.Timestamp(t['exit_time']) >= holdout_ts]

        m_full = _compute_metrics(trades_full)
        m_full_real = _compute_metrics(trades_full_real)
        m_train = _compute_metrics(trades_train)
        m_holdout = _compute_metrics(trades_holdout)
        m_holdout_real = _compute_metrics(trades_holdout_real)

        elapsed = time.time() - t0

        baseline = BASELINES.get(name, {})
        decay_vs_full = 0.0
        if m_full['sharpe'] > 0:
            decay_vs_full = (m_full['sharpe'] - m_holdout['sharpe']) / m_full['sharpe'] * 100

        # Pass criteria
        pass_sharpe_pos = m_holdout['sharpe'] > 0
        pass_no_decay = m_holdout['sharpe'] >= m_full['sharpe'] * 0.5
        pass_real_profit = m_holdout_real['total_pnl'] > 0

        all_pass = pass_sharpe_pos and pass_no_decay and pass_real_profit
        verdict = "PASS" if all_pass else "FAIL"

        result = {
            "strategy": name,
            "full_sample": {"sharpe": m_full['sharpe'], "sharpe_real": m_full_real['sharpe'],
                            "pnl": m_full['total_pnl'], "n_trades": m_full['n_trades']},
            "train_2015_2023": {"sharpe": m_train['sharpe'], "pnl": m_train['total_pnl'],
                                "n_trades": m_train['n_trades']},
            "holdout_2024_2026": {
                "sharpe": m_holdout['sharpe'],
                "sharpe_real": m_holdout_real['sharpe'],
                "pnl": m_holdout['total_pnl'],
                "pnl_real": m_holdout_real['total_pnl'],
                "n_trades": m_holdout['n_trades'],
                "max_dd": m_holdout['max_dd'],
                "win_rate": m_holdout['win_rate'],
                "avg_pnl": m_holdout['avg_pnl'],
                "buy_pnl": m_holdout.get('buy_pnl', 0),
                "sell_pnl": m_holdout.get('sell_pnl', 0),
                "exit_reasons": m_holdout.get('exit_reasons', {}),
                "monthly_pnl": m_holdout.get('monthly_pnl', {}),
            },
            "decay_vs_full_pct": round(decay_vs_full, 1),
            "checks": {
                "holdout_sharpe_positive": pass_sharpe_pos,
                "holdout_sharpe_gt_50pct_full": pass_no_decay,
                "holdout_real_profitable": pass_real_profit,
            },
            "verdict": verdict,
            "elapsed_s": round(elapsed, 1),
        }
        all_results[name] = result

        print(f"\n  Full:    Sharpe={m_full['sharpe']:>6.2f} (real={m_full_real['sharpe']:.2f})  "
              f"PnL=${m_full['total_pnl']:>10,.0f}  trades={m_full['n_trades']}")
        print(f"  Train:   Sharpe={m_train['sharpe']:>6.2f}  "
              f"PnL=${m_train['total_pnl']:>10,.0f}  trades={m_train['n_trades']}")
        print(f"  Holdout: Sharpe={m_holdout['sharpe']:>6.2f} (real={m_holdout_real['sharpe']:.2f})  "
              f"PnL=${m_holdout['total_pnl']:>10,.0f}  trades={m_holdout['n_trades']}")
        print(f"  Decay vs full: {decay_vs_full:+.1f}%")
        print(f"  Holdout MaxDD: ${m_holdout['max_dd']:.0f}  WinRate: {m_holdout['win_rate']:.1f}%")
        print(f"  [{verdict}] ({elapsed:.1f}s)", flush=True)

        out_path = OUTPUT_DIR / f"{name}_oos.json"
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)

    # L8_MAX
    print(f"\n{'#' * 60}")
    print(f"  L8_MAX")
    print(f"{'#' * 60}")
    t0 = time.time()

    print("  Loading L8_MAX DataBundle...", flush=True)
    full_data = _load_l8_data()

    # Full
    trades_full = _run_l8_engine(full_data, SPREAD, LOT)
    trades_full_real = _run_l8_engine(full_data, REALISTIC_SPREAD, LOT)

    # Train slice
    train_data = full_data.slice("2015-01-01", "2023-12-31 23:59:59")
    trades_train = _run_l8_engine(train_data, SPREAD, LOT)

    # Holdout slice
    holdout_data = full_data.slice(HOLDOUT_START, "2027-01-01")
    trades_holdout = _run_l8_engine(holdout_data, SPREAD, LOT)
    trades_holdout_real = _run_l8_engine(holdout_data, REALISTIC_SPREAD, LOT)

    m_full = _compute_metrics(trades_full)
    m_full_real = _compute_metrics(trades_full_real)
    m_train = _compute_metrics(trades_train)
    m_holdout = _compute_metrics(trades_holdout)
    m_holdout_real = _compute_metrics(trades_holdout_real)

    elapsed = time.time() - t0

    decay_vs_full = 0.0
    if m_full['sharpe'] > 0:
        decay_vs_full = (m_full['sharpe'] - m_holdout['sharpe']) / m_full['sharpe'] * 100

    pass_sharpe_pos = m_holdout['sharpe'] > 0
    pass_no_decay = m_holdout['sharpe'] >= m_full['sharpe'] * 0.5
    pass_real_profit = m_holdout_real['total_pnl'] > 0

    all_pass = pass_sharpe_pos and pass_no_decay and pass_real_profit
    verdict = "PASS" if all_pass else "FAIL"

    result = {
        "strategy": "L8_MAX",
        "full_sample": {"sharpe": m_full['sharpe'], "sharpe_real": m_full_real['sharpe'],
                        "pnl": m_full['total_pnl'], "n_trades": m_full['n_trades']},
        "train_2015_2023": {"sharpe": m_train['sharpe'], "pnl": m_train['total_pnl'],
                            "n_trades": m_train['n_trades']},
        "holdout_2024_2026": {
            "sharpe": m_holdout['sharpe'],
            "sharpe_real": m_holdout_real['sharpe'],
            "pnl": m_holdout['total_pnl'],
            "pnl_real": m_holdout_real['total_pnl'],
            "n_trades": m_holdout['n_trades'],
            "max_dd": m_holdout['max_dd'],
            "win_rate": m_holdout['win_rate'],
            "avg_pnl": m_holdout['avg_pnl'],
            "buy_pnl": m_holdout.get('buy_pnl', 0),
            "sell_pnl": m_holdout.get('sell_pnl', 0),
            "exit_reasons": m_holdout.get('exit_reasons', {}),
            "monthly_pnl": m_holdout.get('monthly_pnl', {}),
        },
        "decay_vs_full_pct": round(decay_vs_full, 1),
        "checks": {
            "holdout_sharpe_positive": pass_sharpe_pos,
            "holdout_sharpe_gt_50pct_full": pass_no_decay,
            "holdout_real_profitable": pass_real_profit,
        },
        "verdict": verdict,
        "elapsed_s": round(elapsed, 1),
    }
    all_results["L8_MAX"] = result

    print(f"\n  Full:    Sharpe={m_full['sharpe']:>6.2f} (real={m_full_real['sharpe']:.2f})  "
          f"PnL=${m_full['total_pnl']:>10,.0f}  trades={m_full['n_trades']}")
    print(f"  Train:   Sharpe={m_train['sharpe']:>6.2f}  "
          f"PnL=${m_train['total_pnl']:>10,.0f}  trades={m_train['n_trades']}")
    print(f"  Holdout: Sharpe={m_holdout['sharpe']:>6.2f} (real={m_holdout_real['sharpe']:.2f})  "
          f"PnL=${m_holdout['total_pnl']:>10,.0f}  trades={m_holdout['n_trades']}")
    print(f"  Decay vs full: {decay_vs_full:+.1f}%")
    print(f"  Holdout MaxDD: ${m_holdout['max_dd']:.0f}  WinRate: {m_holdout['win_rate']:.1f}%")
    print(f"  [{verdict}] ({elapsed:.1f}s)", flush=True)

    out_path = OUTPUT_DIR / "L8_MAX_oos.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    # Final Summary
    total_elapsed = time.time() - t0_total
    print(f"\n\n{'=' * 72}")
    print(f"  R71 FINAL SUMMARY — Out-of-Sample Validation")
    print(f"  Holdout: {HOLDOUT_START} ~ end | Total time: {total_elapsed:.0f}s")
    print(f"{'=' * 72}")
    print(f"  {'Strategy':<14} {'Full Sh':>8} {'Train Sh':>9} {'Hold Sh':>8} "
          f"{'HoldReal':>9} {'Decay':>7} {'Hold PnL':>10} {'Verdict'}")
    print(f"  {'-' * 68}")
    for name, r in all_results.items():
        h = r['holdout_2024_2026']
        print(f"  {name:<14} {r['full_sample']['sharpe']:>8.2f} "
              f"{r['train_2015_2023']['sharpe']:>9.2f} "
              f"{h['sharpe']:>8.2f} {h['sharpe_real']:>9.2f} "
              f"{r['decay_vs_full_pct']:>+6.1f}% "
              f"${h['pnl']:>9,.0f} {r['verdict']}")
    print(f"{'=' * 72}", flush=True)

    summary_path = OUTPUT_DIR / "r71_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
