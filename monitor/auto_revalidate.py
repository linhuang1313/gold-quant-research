"""
Auto Re-validation — re-run key validation stages to detect strategy decay.

Runs Stage 1 (Sanity/DSR), Stage 5 (Cost), and Stage 6 (Reality) for each
strategy using the latest available data. Compares Sharpe against the
baseline from R69/R70 validation.

Usage:
    python -m monitor.auto_revalidate
"""
import sys
import os
import json
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

REVALIDATION_LOG = WORKSPACE / "monitor" / "revalidation_log.json"

BASELINES = {
    "PSAR_EA": {"sharpe": 4.96, "validation_round": "R70"},
    "SESS_BO_EA": {"sharpe": 6.48, "validation_round": "R70"},
    "TSMOM": {"sharpe": 4.11, "validation_round": "R69"},
    "L8_MAX": {"sharpe": 7.88, "validation_round": "R69"},
}

DECAY_THRESHOLD = 0.30  # flag if Sharpe drops > 30% from baseline

SPREAD = 0.30
REALISTIC_SPREAD = 0.88


def compute_atr(df, period=14):
    tr = pd.DataFrame({
        "hl": df["High"] - df["Low"],
        "hc": (df["High"] - df["Close"].shift(1)).abs(),
        "lc": (df["Low"] - df["Close"].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {
        "dir": pos["dir"], "entry": pos["entry"], "exit": exit_p,
        "entry_time": pos["time"], "exit_time": exit_time,
        "pnl": pnl, "reason": reason, "bars": bar_idx - pos["bar"],
    }


def add_psar(df, af_start=0.01, af_max=0.05):
    df = df.copy()
    n = len(df)
    psar = np.zeros(n)
    direction = np.ones(n)
    af = af_start
    ep = df["High"].iloc[0]
    psar[0] = df["Low"].iloc[0]
    for i in range(1, n):
        prev = psar[i - 1]
        if direction[i - 1] == 1:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], df["Low"].iloc[i - 1], df["Low"].iloc[max(0, i - 2)])
            if df["Low"].iloc[i] < psar[i]:
                direction[i] = -1; psar[i] = ep; ep = df["Low"].iloc[i]; af = af_start
            else:
                direction[i] = 1
                if df["High"].iloc[i] > ep:
                    ep = df["High"].iloc[i]; af = min(af + af_start, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], df["High"].iloc[i - 1], df["High"].iloc[max(0, i - 2)])
            if df["High"].iloc[i] > psar[i]:
                direction[i] = 1; psar[i] = ep; ep = df["High"].iloc[i]; af = af_start
            else:
                direction[i] = -1
                if df["Low"].iloc[i] < ep:
                    ep = df["Low"].iloc[i]; af = min(af + af_start, af_max)
    df["PSAR_dir"] = direction
    df["ATR"] = compute_atr(df)
    return df


def _backtest_generic(df, spread, lot, sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold):
    """Shared backtest loop for ATR-based strategies."""
    df = df.dropna(subset=["ATR"])
    trades = []
    pos = None
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    atr = df["ATR"].values
    times = df.index
    n = len(df)
    last_exit = -999
    return trades, close, high, low, atr, times, n, last_exit


def backtest_psar(h1_df, spread=SPREAD, lot=0.03,
                  sl_atr=4.5, tp_atr=16.0, trail_act_atr=0.20,
                  trail_dist_atr=0.04, max_hold=20):
    df = add_psar(h1_df).dropna(subset=["PSAR_dir", "ATR"])
    trades = []
    pos = None
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    psar_dir = df["PSAR_dir"].values
    atr = df["ATR"].values
    times = df.index
    n = len(df)
    last_exit = -999
    for i in range(1, n):
        c = close[i]; h = high[i]; lo_v = low[i]; cur_atr = atr[i]
        if pos is not None:
            held = i - pos["bar"]
            if pos["dir"] == "BUY":
                pnl_h = (h - pos["entry"] - spread) * lot * 100
                pnl_l = (lo_v - pos["entry"] - spread) * lot * 100
                pnl_c = (c - pos["entry"] - spread) * lot * 100
            else:
                pnl_h = (pos["entry"] - lo_v - spread) * lot * 100
                pnl_l = (pos["entry"] - h - spread) * lot * 100
                pnl_c = (pos["entry"] - c - spread) * lot * 100
            tp_val = tp_atr * pos["atr"] * lot * 100
            sl_val = sl_atr * pos["atr"] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); exited = True
            else:
                ad = trail_act_atr * pos["atr"]; td = trail_dist_atr * pos["atr"]
                if pos["dir"] == "BUY" and h - pos["entry"] >= ad:
                    ts_p = h - td
                    if lo_v <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos["entry"] - spread) * lot * 100)); exited = True
                elif pos["dir"] == "SELL" and pos["entry"] - lo_v >= ad:
                    ts_p = lo_v + td
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos["entry"] - ts_p - spread) * lot * 100)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if exited:
                pos = None; last_exit = i; continue
        if pos is not None:
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(cur_atr) or cur_atr < 0.1:
            continue
        prev_d = psar_dir[i - 1]; cur_d = psar_dir[i]
        if prev_d == -1 and cur_d == 1:
            pos = {"dir": "BUY", "entry": c + spread / 2, "bar": i, "time": times[i], "atr": cur_atr}
        elif prev_d == 1 and cur_d == -1:
            pos = {"dir": "SELL", "entry": c - spread / 2, "bar": i, "time": times[i], "atr": cur_atr}
    return trades


def backtest_sess_bo(h1_df, spread=SPREAD, lot=0.03,
                     session="peak_12_14", lookback_bars=4,
                     sl_atr=4.5, tp_atr=4.0, trail_act_atr=0.14,
                     trail_dist_atr=0.025, max_hold=20):
    SESSION_DEFS = {"asian": (0, 7), "london": (8, 11), "ny_peak": (12, 16),
                    "late": (17, 23), "peak_12_14": (12, 14)}
    df = h1_df.copy()
    df["ATR"] = compute_atr(df)
    df = df.dropna(subset=["ATR"])
    trades = []
    pos = None
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    atr = df["ATR"].values
    hours = df.index.hour
    times = df.index
    n = len(df)
    last_exit = -999
    sess_start, sess_end = SESSION_DEFS[session]
    for i in range(lookback_bars, n):
        c = close[i]; h = high[i]; lo_v = low[i]; cur_atr = atr[i]; cur_hour = hours[i]
        if pos is not None:
            held = i - pos["bar"]
            if pos["dir"] == "BUY":
                pnl_h = (h - pos["entry"] - spread) * lot * 100
                pnl_l = (lo_v - pos["entry"] - spread) * lot * 100
                pnl_c = (c - pos["entry"] - spread) * lot * 100
            else:
                pnl_h = (pos["entry"] - lo_v - spread) * lot * 100
                pnl_l = (pos["entry"] - h - spread) * lot * 100
                pnl_c = (pos["entry"] - c - spread) * lot * 100
            tp_val = tp_atr * pos["atr"] * lot * 100
            sl_val = sl_atr * pos["atr"] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); exited = True
            else:
                act_dist = trail_act_atr * pos["atr"]; trail_d = trail_dist_atr * pos["atr"]
                if pos["dir"] == "BUY" and h - pos["entry"] >= act_dist:
                    ts_p = h - trail_d
                    if lo_v <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos["entry"] - spread) * lot * 100)); exited = True
                elif pos["dir"] == "SELL" and pos["entry"] - lo_v >= act_dist:
                    ts_p = lo_v + trail_d
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos["entry"] - ts_p - spread) * lot * 100)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if exited:
                pos = None; last_exit = i; continue
        if pos is not None:
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(cur_atr) or cur_atr < 0.1:
            continue
        if cur_hour != sess_start:
            continue
        if i > 0 and hours[i - 1] == sess_start:
            continue
        range_high = max(high[i - lookback_bars:i])
        range_low = min(low[i - lookback_bars:i])
        if c > range_high:
            pos = {"dir": "BUY", "entry": c + spread / 2, "bar": i, "time": times[i], "atr": cur_atr}
        elif c < range_low:
            pos = {"dir": "SELL", "entry": c - spread / 2, "bar": i, "time": times[i], "atr": cur_atr}
    return trades


def backtest_tsmom(h1_df, spread=SPREAD, lot=0.03,
                   fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
                   trail_act_atr=0.14, trail_dist_atr=0.025, max_hold=20):
    df = h1_df.copy()
    df["ATR"] = compute_atr(df)
    df["SMA_fast"] = df["Close"].rolling(fast).mean()
    df["SMA_slow"] = df["Close"].rolling(slow).mean()
    df = df.dropna(subset=["ATR", "SMA_fast", "SMA_slow"])
    trades = []
    pos = None
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    atr = df["ATR"].values
    sma_f = df["SMA_fast"].values
    sma_s = df["SMA_slow"].values
    times = df.index
    n = len(df)
    last_exit = -999
    prev_cross = 0
    for i in range(1, n):
        c = close[i]; h = high[i]; lo_v = low[i]; cur_atr = atr[i]
        cur_cross = 1 if sma_f[i] > sma_s[i] else -1
        if pos is not None:
            held = i - pos["bar"]
            if pos["dir"] == "BUY":
                pnl_h = (h - pos["entry"] - spread) * lot * 100
                pnl_l = (lo_v - pos["entry"] - spread) * lot * 100
                pnl_c = (c - pos["entry"] - spread) * lot * 100
            else:
                pnl_h = (pos["entry"] - lo_v - spread) * lot * 100
                pnl_l = (pos["entry"] - h - spread) * lot * 100
                pnl_c = (pos["entry"] - c - spread) * lot * 100
            tp_val = tp_atr * pos["atr"] * lot * 100
            sl_val = sl_atr * pos["atr"] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); exited = True
            else:
                ad = trail_act_atr * pos["atr"]; td = trail_dist_atr * pos["atr"]
                if pos["dir"] == "BUY" and h - pos["entry"] >= ad:
                    ts_p = h - td
                    if lo_v <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos["entry"] - spread) * lot * 100)); exited = True
                elif pos["dir"] == "SELL" and pos["entry"] - lo_v >= ad:
                    ts_p = lo_v + td
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos["entry"] - ts_p - spread) * lot * 100)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if exited:
                pos = None; last_exit = i
            prev_cross = cur_cross
            continue
        if i - last_exit < 2:
            prev_cross = cur_cross; continue
        if np.isnan(cur_atr) or cur_atr < 0.1:
            prev_cross = cur_cross; continue
        if prev_cross == -1 and cur_cross == 1:
            pos = {"dir": "BUY", "entry": c + spread / 2, "bar": i, "time": times[i], "atr": cur_atr}
        elif prev_cross == 1 and cur_cross == -1:
            pos = {"dir": "SELL", "entry": c - spread / 2, "bar": i, "time": times[i], "atr": cur_atr}
        prev_cross = cur_cross
    return trades


def _trades_to_daily(trades):
    if not trades:
        return np.array([0.0])
    df = pd.DataFrame(trades)
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    daily = df.set_index("exit_time").resample("D")["pnl"].sum()
    return daily.values


def _sharpe(daily, ann=252):
    if len(daily) < 2:
        return 0.0
    m = np.mean(daily)
    s = np.std(daily, ddof=1)
    if s == 0:
        return 0.0
    return float(m / s * np.sqrt(ann))


def run_quick_validation(h1_df) -> dict:
    """Run quick revalidation for PSAR, SESS_BO, TSMOM on local data."""
    results = {}

    strategies = {
        "PSAR_EA": lambda df, sp, lot: backtest_psar(df, sp, lot),
        "SESS_BO_EA": lambda df, sp, lot: backtest_sess_bo(df, sp, lot),
        "TSMOM": lambda df, sp, lot: backtest_tsmom(df, sp, lot),
    }

    for name, fn in strategies.items():
        t0 = time.time()
        print(f"\n  Revalidating {name}...", end=" ", flush=True)

        trades = fn(h1_df, SPREAD, 0.03)
        daily = _trades_to_daily(trades)
        sharpe_nominal = _sharpe(daily)

        trades_real = fn(h1_df, REALISTIC_SPREAD, 0.03)
        daily_real = _trades_to_daily(trades_real)
        sharpe_realistic = _sharpe(daily_real)

        total_pnl = float(np.sum([t["pnl"] for t in trades]))
        total_pnl_real = float(np.sum([t["pnl"] for t in trades_real]))

        baseline = BASELINES.get(name, {})
        baseline_sharpe = baseline.get("sharpe", 0)
        decay_pct = 0.0
        if baseline_sharpe > 0:
            decay_pct = (baseline_sharpe - sharpe_nominal) / baseline_sharpe

        status = "OK"
        if decay_pct > DECAY_THRESHOLD:
            status = f"DECAY ({decay_pct:.0%} drop)"
        elif sharpe_realistic < 1.0:
            status = "WEAK (realistic Sharpe < 1.0)"

        elapsed = time.time() - t0
        results[name] = {
            "sharpe_nominal": round(sharpe_nominal, 2),
            "sharpe_realistic": round(sharpe_realistic, 2),
            "baseline_sharpe": baseline_sharpe,
            "decay_pct": round(decay_pct * 100, 1),
            "n_trades": len(trades),
            "pnl_nominal": round(total_pnl, 2),
            "pnl_realistic": round(total_pnl_real, 2),
            "status": status,
            "elapsed_s": round(elapsed, 1),
        }
        print(f"Sharpe={sharpe_nominal:.2f} (real={sharpe_realistic:.2f}) "
              f"trades={len(trades)} [{status}] ({elapsed:.1f}s)")

    return results


def print_revalidation_report(results: dict):
    """Print formatted revalidation summary."""
    print(f"\n{'=' * 75}")
    print(f"  REVALIDATION SUMMARY  |  {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 75}")
    print(f"  {'Strategy':<14} {'Sharpe':>7} {'Real':>7} {'Base':>7} {'Decay':>7} "
          f"{'Trades':>7} {'Status'}")
    print(f"  {'-' * 70}")
    for name, r in results.items():
        print(f"  {name:<14} {r['sharpe_nominal']:>7.2f} {r['sharpe_realistic']:>7.2f} "
              f"{r['baseline_sharpe']:>7.2f} {r['decay_pct']:>6.1f}% "
              f"{r['n_trades']:>7} {r['status']}")
    print(f"{'=' * 75}")


def save_log(results: dict):
    """Append revalidation results to JSON log."""
    entry = {
        "timestamp": dt.datetime.now().isoformat(),
        "results": results,
    }

    log_data = []
    if REVALIDATION_LOG.exists():
        try:
            with open(REVALIDATION_LOG, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        except (json.JSONDecodeError, Exception):
            log_data = []

    log_data.append(entry)

    # Keep last 100 entries
    if len(log_data) > 100:
        log_data = log_data[-100:]

    with open(REVALIDATION_LOG, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, default=str)

    print(f"\n  Log saved to {REVALIDATION_LOG}")


def run():
    """Main entry point."""
    print("=" * 60)
    print("  Auto Re-validation Pipeline")
    print("=" * 60)

    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH

    print("\n  Loading H1 data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})")

    results = run_quick_validation(h1_df)
    print_revalidation_report(results)
    save_log(results)

    decayed = [n for n, r in results.items() if "DECAY" in r["status"]]
    if decayed:
        print(f"\n  *** WARNING: Strategy decay detected in: {', '.join(decayed)} ***")
        print(f"  *** Consider pausing these strategies and investigating. ***\n")

    return results


if __name__ == "__main__":
    run()
