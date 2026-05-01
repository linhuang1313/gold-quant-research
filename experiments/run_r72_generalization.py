#!/usr/bin/env python3
"""
R72 — Multi-Asset Generalization + Multi-Split OOS
====================================================
Part A: Run PSAR/TSMOM/SESS_BO on XAGUSD (silver) with gold EA parameters.
Part B: Multi-split holdout on XAUUSD (2020+, 2022+) for regime robustness.

Estimated runtime: ~5 minutes.
"""
import sys, os, io, time, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r72_generalization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Gold params
GOLD_SPREAD = 0.30
GOLD_SPREAD_REAL = 0.88
# Silver params (XAGUSD: ~10x smaller price, proportional spread)
SILVER_SPREAD = 0.03
SILVER_SPREAD_REAL = 0.05
LOT = 0.03
# XAGUSD: 1 standard lot = 5000 oz, point_value = lot * 5000
# XAUUSD: 1 standard lot = 100 oz, point_value = lot * 100
GOLD_POINT_VALUE = 100
SILVER_POINT_VALUE = 5000

XAGUSD_H1_CANDIDATES = [
    Path("data/download/xagusd-h1-bid-2015-01-01-2026-05-01.csv"),
    Path("data/download/xagusd-h1-bid-2015-01-01-2026-04-27.csv"),
    Path("data/download/xagusd-h1-bid-2015-01-01-2026-04-10.csv"),
]

# ═══════════════════════════════════════════════════════════════
# Helpers
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
    if len(daily) < 2: return 0.0
    m = np.mean(daily); s = np.std(daily, ddof=1)
    if s == 0: return 0.0
    return float(m / s * np.sqrt(ann))


def _compute_metrics(trades):
    if not trades:
        return {"n_trades": 0, "total_pnl": 0, "sharpe": 0, "max_dd": 0,
                "win_rate": 0, "avg_pnl": 0}
    pnls = [t['pnl'] for t in trades]
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    wins = sum(1 for p in pnls if p > 0)
    daily = _trades_to_daily(trades)

    df = pd.DataFrame(trades)
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    monthly = df.set_index('exit_time').resample('ME')['pnl'].sum()
    monthly_dict = {str(k.strftime('%Y-%m')): round(float(v), 2) for k, v in monthly.items()}

    reasons = {}
    for t in trades:
        r = t.get('reason', 'unknown')
        reasons[r] = reasons.get(r, 0) + 1

    return {
        "n_trades": len(trades),
        "total_pnl": round(float(cum[-1]), 2),
        "sharpe": round(_sharpe(daily), 2),
        "max_dd": round(float(dd.min()), 2),
        "win_rate": round(wins / len(pnls) * 100, 1),
        "avg_pnl": round(float(np.mean(pnls)), 2),
        "exit_reasons": reasons,
        "monthly_pnl": monthly_dict,
    }


# ═══════════════════════════════════════════════════════════════
# Strategy backtests (parameterized by point_value)
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


def backtest_psar(h1_df, spread, lot, pv=GOLD_POINT_VALUE,
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
                pnl_h = (h - pos['entry'] - spread) * lot * pv
                pnl_l = (lo_v - pos['entry'] - spread) * lot * pv
                pnl_c = (c - pos['entry'] - spread) * lot * pv
            else:
                pnl_h = (pos['entry'] - lo_v - spread) * lot * pv
                pnl_l = (pos['entry'] - h - spread) * lot * pv
                pnl_c = (pos['entry'] - c - spread) * lot * pv
            tp_val = tp_atr * pos['atr'] * lot * pv
            sl_val = sl_atr * pos['atr'] * lot * pv
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
                                          (ts_p - pos['entry'] - spread) * lot * pv)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
                    ts_p = lo_v + td
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * pv)); exited = True
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


def backtest_sess_bo(h1_df, spread, lot, pv=GOLD_POINT_VALUE,
                     session="peak_12_14", lookback_bars=4,
                     sl_atr=4.5, tp_atr=4.0, trail_act_atr=0.14,
                     trail_dist_atr=0.025, max_hold=20):
    SESSION_DEFS = {"peak_12_14": (12,14)}
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
                pnl_h = (h - pos['entry'] - spread) * lot * pv
                pnl_l = (lo_v - pos['entry'] - spread) * lot * pv
                pnl_c = (c - pos['entry'] - spread) * lot * pv
            else:
                pnl_h = (pos['entry'] - lo_v - spread) * lot * pv
                pnl_l = (pos['entry'] - h - spread) * lot * pv
                pnl_c = (pos['entry'] - c - spread) * lot * pv
            tp_val = tp_atr * pos['atr'] * lot * pv
            sl_val = sl_atr * pos['atr'] * lot * pv
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
                                          (ts_p - pos['entry'] - spread) * lot * pv)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= act_dist:
                    ts_p = lo_v + trail_d
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * pv)); exited = True
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


def backtest_tsmom(h1_df, spread, lot, pv=GOLD_POINT_VALUE,
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
                pnl_h = (h - pos['entry'] - spread) * lot * pv
                pnl_l = (lo_v - pos['entry'] - spread) * lot * pv
                pnl_c = (c - pos['entry'] - spread) * lot * pv
            else:
                pnl_h = (pos['entry'] - lo_v - spread) * lot * pv
                pnl_l = (pos['entry'] - h - spread) * lot * pv
                pnl_c = (pos['entry'] - c - spread) * lot * pv
            tp_val = tp_atr * pos['atr'] * lot * pv
            sl_val = sl_atr * pos['atr'] * lot * pv
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
                                          (ts_p - pos['entry'] - spread) * lot * pv)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
                    ts_p = lo_v + td
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * pv)); exited = True
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
# Part A: XAGUSD generalization
# ═══════════════════════════════════════════════════════════════

def run_part_a(gold_h1):
    """Run all 3 strategies on XAGUSD and compare with XAUUSD."""
    print(f"\n{'=' * 72}")
    print(f"  PART A: Multi-Asset Generalization (XAGUSD Silver)")
    print(f"{'=' * 72}\n")

    # Load XAGUSD
    ag_path = None
    for p in XAGUSD_H1_CANDIDATES:
        if p.exists():
            ag_path = p; break

    if ag_path is None:
        print("  XAGUSD H1 data not found. Attempting download...", flush=True)
        try:
            ag_path = download_xagusd_h1()
        except Exception as e:
            print(f"  Download failed: {e}")
            print("  SKIPPING Part A — no XAGUSD data available.")
            return {"skipped": True, "reason": "no XAGUSD data"}

    from backtest.runner import load_csv
    ag_h1 = load_csv(str(ag_path))
    print(f"  XAGUSD H1: {len(ag_h1)} bars ({ag_h1.index[0]} ~ {ag_h1.index[-1]})")
    print(f"  XAGUSD price range: ${ag_h1['Close'].min():.2f} ~ ${ag_h1['Close'].max():.2f}")
    print(f"  XAGUSD H1 ATR(14) median: ${compute_atr(ag_h1).median():.4f}\n", flush=True)

    # ATR filter for silver — some strategies check atr < 0.1 which is too high for silver
    # Silver ATR is typically 0.2-0.5, so 0.1 filter still works but is tighter

    strategies = [
        ("PSAR", backtest_psar),
        ("SESS_BO", backtest_sess_bo),
        ("TSMOM", backtest_tsmom),
    ]

    results = {}
    for name, fn in strategies:
        print(f"  --- {name} ---")

        # Gold (reference)
        t_gold = fn(gold_h1, GOLD_SPREAD, LOT, pv=GOLD_POINT_VALUE)
        t_gold_r = fn(gold_h1, GOLD_SPREAD_REAL, LOT, pv=GOLD_POINT_VALUE)
        m_gold = _compute_metrics(t_gold)
        m_gold_r = _compute_metrics(t_gold_r)

        # Silver
        t_silver = fn(ag_h1, SILVER_SPREAD, LOT, pv=SILVER_POINT_VALUE)
        t_silver_r = fn(ag_h1, SILVER_SPREAD_REAL, LOT, pv=SILVER_POINT_VALUE)
        m_silver = _compute_metrics(t_silver)
        m_silver_r = _compute_metrics(t_silver_r)

        # Generalization ratio
        gen_ratio = 0.0
        if m_gold['sharpe'] > 0:
            gen_ratio = m_silver['sharpe'] / m_gold['sharpe'] * 100

        pass_positive = m_silver['sharpe'] > 0
        pass_ratio = gen_ratio >= 30
        pass_real = m_silver_r['total_pnl'] > 0
        verdict = "PASS" if (pass_positive and pass_ratio and pass_real) else "FAIL"

        results[name] = {
            "gold": {"sharpe": m_gold['sharpe'], "sharpe_real": m_gold_r['sharpe'],
                     "pnl": m_gold['total_pnl'], "n_trades": m_gold['n_trades']},
            "silver": {"sharpe": m_silver['sharpe'], "sharpe_real": m_silver_r['sharpe'],
                       "pnl": m_silver['total_pnl'], "pnl_real": m_silver_r['total_pnl'],
                       "n_trades": m_silver['n_trades'], "max_dd": m_silver['max_dd'],
                       "win_rate": m_silver['win_rate'],
                       "exit_reasons": m_silver.get('exit_reasons', {}),
                       "monthly_pnl": m_silver.get('monthly_pnl', {})},
            "generalization_pct": round(gen_ratio, 1),
            "checks": {"silver_sharpe_positive": pass_positive,
                       "gen_ratio_gte_30pct": pass_ratio,
                       "silver_real_profitable": pass_real},
            "verdict": verdict,
        }

        print(f"    XAUUSD: Sharpe={m_gold['sharpe']:.2f} (real={m_gold_r['sharpe']:.2f})  "
              f"PnL=${m_gold['total_pnl']:,.0f}  trades={m_gold['n_trades']}")
        print(f"    XAGUSD: Sharpe={m_silver['sharpe']:.2f} (real={m_silver_r['sharpe']:.2f})  "
              f"PnL=${m_silver['total_pnl']:,.0f}  trades={m_silver['n_trades']}")
        print(f"    Generalization: {gen_ratio:.1f}%  [{verdict}]\n", flush=True)

    out_path = OUTPUT_DIR / "part_a_multi_asset.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def download_xagusd_h1():
    """Download XAGUSD H1 data using Dukascopy."""
    import dukascopy_python as dk
    from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAG_USD
    from datetime import datetime

    dl_dir = Path("data/download")
    dl_dir.mkdir(parents=True, exist_ok=True)

    start = datetime(2015, 1, 1)
    end = datetime(2026, 5, 1)
    date_suffix = f"{start.strftime('%Y-%m-%d')}-{end.strftime('%Y-%m-%d')}"
    filepath = dl_dir / f"xagusd-h1-bid-{date_suffix}.csv"

    if filepath.exists():
        return filepath

    print(f"    Downloading XAGUSD H1 from Dukascopy...", flush=True)

    all_dfs = []
    chunk_start = start
    from datetime import timedelta
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=180), end)
        print(f"      {chunk_start.strftime('%Y-%m-%d')} -> {chunk_end.strftime('%Y-%m-%d')}...",
              end="", flush=True)
        try:
            df = dk.fetch(
                INSTRUMENT_FX_METALS_XAG_USD,
                dk.INTERVAL_HOUR_1,
                dk.OFFER_SIDE_BID,
                chunk_start, chunk_end,
                max_retries=5
            )
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                print(f" {len(df)} bars")
            else:
                print(" 0 bars")
        except Exception as e:
            print(f" ERROR: {e}")
        chunk_start = chunk_end
        import time as _time
        _time.sleep(1)

    if not all_dfs:
        raise RuntimeError("No XAGUSD data downloaded")

    combined = pd.concat(all_dfs)
    combined = combined[~combined.index.duplicated(keep='first')].sort_index()

    out = pd.DataFrame({
        'timestamp': (combined.index.astype('int64') // 10**6),
        'open': combined['open'], 'high': combined['high'],
        'low': combined['low'], 'close': combined['close'],
        'volume': combined['volume'],
    })
    out.to_csv(filepath, index=False)
    print(f"    Saved: {filepath} ({len(out)} bars)")
    return filepath


# ═══════════════════════════════════════════════════════════════
# Part B: Multi-split holdout
# ═══════════════════════════════════════════════════════════════

def run_part_b(gold_h1):
    """Run multi-split holdout tests on XAUUSD."""
    print(f"\n{'=' * 72}")
    print(f"  PART B: Multi-Split Holdout OOS (XAUUSD)")
    print(f"{'=' * 72}\n")

    splits = [
        {"name": "Split_A_2020", "holdout_start": "2020-01-01",
         "regime": "COVID crash, recovery, inflation, rate hikes"},
        {"name": "Split_B_2022", "holdout_start": "2022-01-01",
         "regime": "Rate hike cycle, 2022 correction, 2024-25 rally"},
        {"name": "Split_C_2024", "holdout_start": "2024-01-01",
         "regime": "Recent gold rally (R71 reference)"},
    ]

    strategies = [
        ("PSAR", lambda df, sp, lot: backtest_psar(df, sp, lot, pv=GOLD_POINT_VALUE)),
        ("SESS_BO", lambda df, sp, lot: backtest_sess_bo(df, sp, lot, pv=GOLD_POINT_VALUE)),
        ("TSMOM", lambda df, sp, lot: backtest_tsmom(df, sp, lot, pv=GOLD_POINT_VALUE)),
    ]

    results = {}

    for split in splits:
        split_name = split["name"]
        holdout_ts = pd.Timestamp(split["holdout_start"], tz='UTC')
        print(f"  ### {split_name} — Holdout from {split['holdout_start']}")
        print(f"      Regime: {split['regime']}\n")

        split_results = {}
        for strat_name, fn in strategies:
            # Run on full data, filter trades by holdout period
            trades_full = fn(gold_h1, GOLD_SPREAD, LOT)
            trades_holdout = [t for t in trades_full
                              if pd.Timestamp(t['exit_time']) >= holdout_ts]
            trades_full_real = fn(gold_h1, GOLD_SPREAD_REAL, LOT)
            trades_holdout_real = [t for t in trades_full_real
                                   if pd.Timestamp(t['exit_time']) >= holdout_ts]

            m_full = _compute_metrics(trades_full)
            m_hold = _compute_metrics(trades_holdout)
            m_hold_r = _compute_metrics(trades_holdout_real)

            decay = 0.0
            if m_full['sharpe'] > 0:
                decay = (m_full['sharpe'] - m_hold['sharpe']) / m_full['sharpe'] * 100

            pass_pos = m_hold['sharpe'] > 0
            pass_decay = abs(decay) < 70
            pass_real = m_hold_r['total_pnl'] > 0
            verdict = "PASS" if (pass_pos and pass_decay and pass_real) else "FAIL"

            split_results[strat_name] = {
                "full_sharpe": m_full['sharpe'],
                "holdout_sharpe": m_hold['sharpe'],
                "holdout_sharpe_real": m_hold_r['sharpe'],
                "holdout_pnl": m_hold['total_pnl'],
                "holdout_pnl_real": m_hold_r['total_pnl'],
                "holdout_trades": m_hold['n_trades'],
                "holdout_max_dd": m_hold['max_dd'],
                "holdout_win_rate": m_hold['win_rate'],
                "decay_pct": round(decay, 1),
                "verdict": verdict,
            }

            print(f"    {strat_name:<10} Full={m_full['sharpe']:.2f}  "
                  f"Hold={m_hold['sharpe']:.2f} (real={m_hold_r['sharpe']:.2f})  "
                  f"Decay={decay:+.1f}%  PnL=${m_hold['total_pnl']:,.0f}  "
                  f"[{verdict}]")

        results[split_name] = {
            "holdout_start": split["holdout_start"],
            "regime": split["regime"],
            "strategies": split_results,
        }
        print(flush=True)

    out_path = OUTPUT_DIR / "part_b_multi_split.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 72)
    print("  R72 — Multi-Asset Generalization + Multi-Split OOS")
    print("=" * 72, flush=True)

    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH
    print("\n  Loading XAUUSD H1 data...", flush=True)
    m15_raw = load_m15()
    gold_h1 = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"  XAUUSD H1: {len(gold_h1)} bars ({gold_h1.index[0]} ~ {gold_h1.index[-1]})\n")

    part_a = run_part_a(gold_h1)
    part_b = run_part_b(gold_h1)

    total = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"  R72 COMPLETE — {total:.0f}s ({total/60:.1f}min)")
    print(f"{'=' * 72}")

    # Summary table
    print(f"\n  Part A: Multi-Asset Generalization")
    if isinstance(part_a, dict) and not part_a.get('skipped'):
        print(f"  {'Strategy':<10} {'AU Sharpe':>10} {'AG Sharpe':>10} {'Gen%':>6} {'Verdict'}")
        for name, r in part_a.items():
            print(f"  {name:<10} {r['gold']['sharpe']:>10.2f} {r['silver']['sharpe']:>10.2f} "
                  f"{r['generalization_pct']:>5.1f}% {r['verdict']}")
    else:
        print("  SKIPPED (no XAGUSD data)")

    print(f"\n  Part B: Multi-Split Holdout")
    print(f"  {'Split':<16} {'PSAR':>8} {'SESS_BO':>8} {'TSMOM':>8}")
    for split_name, sr in part_b.items():
        vals = []
        for sn in ['PSAR', 'SESS_BO', 'TSMOM']:
            s = sr['strategies'].get(sn, {})
            vals.append(f"{s.get('holdout_sharpe', 0):.2f}")
        print(f"  {split_name:<16} {'  '.join(f'{v:>8}' for v in vals)}")

    print(f"{'=' * 72}", flush=True)

    combined = {"part_a": part_a, "part_b": part_b, "elapsed_s": round(total, 1)}
    with open(OUTPUT_DIR / "r72_summary.json", 'w') as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
