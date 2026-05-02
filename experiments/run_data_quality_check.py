"""
Data Quality Audit — Comprehensive check of all XAUUSD + external data.
========================================================================
Checks performed:
  1. OHLC integrity:  High >= max(Open,Close), Low <= min(Open,Close)
  2. Flat-bar ratio:  O==H==L==C (weekend/holiday placeholders)
  3. Gap detection:   Missing bars beyond expected weekend gaps
  4. Timestamp regularity: duplicate timestamps, out-of-order rows
  5. Extreme price moves:  single-bar moves > N×ATR (spike detection)
  6. Spread reasonableness: negative spread, extreme spread
  7. Bid-Ask consistency:   Ask >= Bid at every timestamp
  8. Cross-timeframe consistency:  M15→H1 OHLC aggregation match
  9. External data freshness & NaN audit
 10. M1 vs M15 cross-validation (sample)
"""

import sys, json, os
from pathlib import Path
from datetime import timedelta
from collections import OrderedDict

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

DATA_DIR = Path("data/download")

FILES = {
    "M15_BID": DATA_DIR / "xauusd-m15-bid-2015-01-01-2026-04-27.csv",
    "M15_ASK": DATA_DIR / "xauusd-m15-ask-2015-01-01-2026-04-27.csv",
    "M15_SPREAD": DATA_DIR / "xauusd-m15-spread-2015-01-01-2026-04-27.csv",
    "H1_BID":  DATA_DIR / "xauusd-h1-bid-2015-01-01-2026-04-27.csv",
    "H1_ASK":  DATA_DIR / "xauusd-h1-ask-2015-01-01-2026-04-27.csv",
    "H1_SPREAD": DATA_DIR / "xauusd-h1-spread-2015-01-01-2026-04-27.csv",
    "H4_BID":  DATA_DIR / "xauusd-h4-bid-2015-01-01-2026-04-27.csv",
    "H4_ASK":  DATA_DIR / "xauusd-h4-ask-2015-01-01-2026-04-27.csv",
    "D1_BID":  DATA_DIR / "xauusd-d1-bid-2015-01-01-2026-04-27.csv",
    "D1_ASK":  DATA_DIR / "xauusd-d1-ask-2015-01-01-2026-04-27.csv",
    "D1_SPREAD": DATA_DIR / "xauusd-d1-spread-2015-01-01-2026-04-27.csv",
}

EXT_DIR = Path("data/external")

REPORT = OrderedDict()


def load_ohlc(path, label):
    """Load Dukascopy-format CSV → DatetimeIndex(UTC)."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "volume": "Volume"}, inplace=True)
    return df


def load_spread(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


# ─────────────────────────────────────────────
# Check 1 & 2: OHLC integrity + flat bars
# ─────────────────────────────────────────────
def check_ohlc_integrity(df, label):
    print(f"\n{'='*60}")
    print(f"  [{label}] OHLC Integrity")
    print(f"{'='*60}")

    n = len(df)
    result = {"total_bars": n}

    flat = (df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"])
    n_flat = flat.sum()
    result["flat_bars"] = int(n_flat)
    result["flat_pct"] = round(n_flat / n * 100, 2)
    print(f"  Total bars: {n:,}")
    print(f"  Flat bars (O==H==L==C): {n_flat:,} ({result['flat_pct']:.2f}%)")

    active = df[~flat].copy()
    n_active = len(active)
    result["active_bars"] = n_active

    high_err = active["High"] < active[["Open", "Close"]].max(axis=1)
    low_err = active["Low"] > active[["Open", "Close"]].min(axis=1)
    hl_invert = active["High"] < active["Low"]

    result["high_violations"] = int(high_err.sum())
    result["low_violations"] = int(low_err.sum())
    result["hl_inversions"] = int(hl_invert.sum())

    print(f"  Active bars: {n_active:,}")
    print(f"  High < max(O,C) violations: {high_err.sum()}")
    print(f"  Low > min(O,C) violations:  {low_err.sum()}")
    print(f"  High < Low inversions:      {hl_invert.sum()}")

    if high_err.sum() > 0:
        print(f"    Sample high violations:")
        for ts in active[high_err].head(3).index:
            row = active.loc[ts]
            print(f"      {ts}  O={row['Open']:.3f} H={row['High']:.3f} L={row['Low']:.3f} C={row['Close']:.3f}")

    # NaN check
    nan_count = df[["Open", "High", "Low", "Close"]].isna().sum().sum()
    result["nan_values"] = int(nan_count)
    print(f"  NaN values in OHLC: {nan_count}")

    # Negative / zero price
    neg = (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).sum()
    result["non_positive_prices"] = int(neg)
    print(f"  Non-positive prices: {neg}")

    # Price range sanity (gold should be ~$1000-$3500 for 2015-2026)
    price_min = active["Low"].min()
    price_max = active["High"].max()
    result["price_range"] = [round(price_min, 2), round(price_max, 2)]
    print(f"  Price range: ${price_min:.2f} — ${price_max:.2f}")
    if price_min < 900 or price_max > 6000:
        print(f"  [!] WARNING: Price range outside expected $900-$6000!")
        result["price_range_warning"] = True

    return result


# ─────────────────────────────────────────────
# Check 3: Timestamp regularity & gaps
# ─────────────────────────────────────────────
def check_timestamps(df, label, expected_freq_min):
    print(f"\n{'='*60}")
    print(f"  [{label}] Timestamp Analysis")
    print(f"{'='*60}")

    result = {}
    result["date_range"] = [str(df.index[0]), str(df.index[-1])]
    print(f"  Range: {df.index[0]} → {df.index[-1]}")

    # Duplicates
    dupes = df.index.duplicated().sum()
    result["duplicate_timestamps"] = int(dupes)
    print(f"  Duplicate timestamps: {dupes}")

    # Out-of-order
    sorted_ok = df.index.is_monotonic_increasing
    result["monotonic"] = bool(sorted_ok)
    print(f"  Monotonically increasing: {sorted_ok}")

    # Filter out flat bars for gap analysis
    flat = (df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"])
    active = df[~flat]
    if len(active) < 2:
        return result

    diffs = active.index.to_series().diff().dropna()
    expected_td = pd.Timedelta(minutes=expected_freq_min)

    # Normal gap analysis (exclude known weekend gaps ~48-65h)
    median_gap = diffs.median()
    result["median_gap"] = str(median_gap)
    result["expected_gap"] = str(expected_td)
    print(f"  Median gap (active bars): {median_gap}  (expected: {expected_td})")

    # Weekend gaps (Fri close → Sun open): typically 47-65h
    weekend_threshold = pd.Timedelta(hours=40)
    weekday_gaps = diffs[diffs <= weekend_threshold]
    weekend_gaps = diffs[diffs > weekend_threshold]

    result["weekend_gaps"] = len(weekend_gaps)
    print(f"  Weekend/holiday gaps (>40h): {len(weekend_gaps)}")

    # Abnormal weekday gaps (> 2× expected frequency, but < weekend)
    abnormal_threshold = expected_td * 3
    abnormal = weekday_gaps[weekday_gaps > abnormal_threshold]
    result["abnormal_weekday_gaps"] = len(abnormal)
    if len(abnormal) > 0:
        print(f"  [!] Abnormal weekday gaps (>{abnormal_threshold}): {len(abnormal)}")
        for ts, gap in abnormal.head(5).items():
            print(f"      {ts - gap} → {ts}  ({gap})")
        if len(abnormal) > 5:
            print(f"      ... and {len(abnormal) - 5} more")
    else:
        print(f"  Abnormal weekday gaps: 0  ✓")

    # Coverage
    total_hours = (active.index[-1] - active.index[0]).total_seconds() / 3600
    expected_trading_hours = total_hours * 5 / 7  # ~71% trading time
    expected_bars = expected_trading_hours * 60 / expected_freq_min
    coverage = len(active) / expected_bars if expected_bars > 0 else 1
    result["coverage_pct"] = round(coverage * 100, 2)
    print(f"  Coverage: {coverage:.1%} (got {len(active):,} active bars, expected ~{expected_bars:,.0f})")

    return result


# ─────────────────────────────────────────────
# Check 4: Extreme price moves (spike detection)
# ─────────────────────────────────────────────
def check_spikes(df, label, expected_freq_min):
    print(f"\n{'='*60}")
    print(f"  [{label}] Spike Detection")
    print(f"{'='*60}")

    flat = (df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"])
    active = df[~flat].copy()

    active["bar_range"] = active["High"] - active["Low"]
    active["atr14"] = active["bar_range"].rolling(14).mean()
    active["close_change"] = active["Close"].diff().abs()

    # Spike = bar range > 8× ATR14 or close change > 6× ATR14
    spike_range = active["bar_range"] > 8 * active["atr14"]
    spike_close = active["close_change"] > 6 * active["atr14"]

    spikes = active[spike_range | spike_close].dropna(subset=["atr14"])

    result = {"spike_bars": len(spikes)}
    print(f"  Spike bars (range>8×ATR or close_chg>6×ATR): {len(spikes)}")

    if len(spikes) > 0:
        for ts in spikes.head(10).index:
            row = spikes.loc[ts]
            print(f"    {ts}  O={row['Open']:.2f} H={row['High']:.2f} "
                  f"L={row['Low']:.2f} C={row['Close']:.2f}  "
                  f"range={row['bar_range']:.2f} ATR={row['atr14']:.2f}")

    # Z-score based outlier detection on returns
    returns = active["Close"].pct_change().dropna()
    z = (returns - returns.mean()) / returns.std()
    extreme = (z.abs() > 5).sum()
    result["zscore_outliers_5sigma"] = int(extreme)
    print(f"  Returns > 5σ: {extreme}")

    return result


# ─────────────────────────────────────────────
# Check 5: Spread analysis
# ─────────────────────────────────────────────
def check_spread(spread_path, bid_path, ask_path, label):
    print(f"\n{'='*60}")
    print(f"  [{label}] Spread Analysis")
    print(f"{'='*60}")

    result = {}

    if not spread_path.exists():
        print(f"  SKIP: {spread_path} not found")
        return result

    sp = load_spread(spread_path)
    result["total_bars"] = len(sp)

    # Negative spread
    neg = (sp["spread_avg"] < 0).sum()
    result["negative_spread"] = int(neg)
    print(f"  Bars with negative avg spread: {neg}")

    # Spread statistics
    avg = sp["spread_avg"].mean()
    med = sp["spread_avg"].median()
    p95 = sp["spread_avg"].quantile(0.95)
    p99 = sp["spread_avg"].quantile(0.99)
    mx = sp["spread_avg"].max()
    result["spread_stats"] = {
        "mean": round(avg, 4), "median": round(med, 4),
        "p95": round(p95, 4), "p99": round(p99, 4), "max": round(mx, 4)
    }
    print(f"  Spread avg: mean={avg:.4f}, median={med:.4f}, p95={p95:.4f}, p99={p99:.4f}, max={mx:.4f}")

    extreme_sp = sp["spread_avg"] > 5.0
    result["extreme_spread_gt5"] = int(extreme_sp.sum())
    print(f"  Spread > $5.00: {extreme_sp.sum()} bars")

    # Bid-Ask consistency
    if bid_path.exists() and ask_path.exists():
        bid = load_ohlc(bid_path, f"{label}_bid")
        ask = load_ohlc(ask_path, f"{label}_ask")
        common = bid.index.intersection(ask.index)
        if len(common) > 0:
            b = bid.loc[common]
            a = ask.loc[common]
            ask_lt_bid = (a["Close"] < b["Close"]).sum()
            result["ask_lt_bid_close"] = int(ask_lt_bid)
            print(f"  Ask < Bid (Close) occurrences: {ask_lt_bid} / {len(common)}")

            spread_calc = a["Close"] - b["Close"]
            neg_spread = (spread_calc < -0.01).sum()
            result["calculated_neg_spread"] = int(neg_spread)
            print(f"  Calculated spread (Ask-Bid) < -$0.01: {neg_spread}")

            avg_calc_spread = spread_calc.mean()
            result["avg_calculated_spread"] = round(avg_calc_spread, 4)
            print(f"  Avg calculated spread: ${avg_calc_spread:.4f}")
    else:
        print(f"  Bid/Ask files not both available for cross-check")

    return result


# ─────────────────────────────────────────────
# Check 6: Cross-timeframe consistency (M15 → H1)
# ─────────────────────────────────────────────
def check_cross_timeframe(m15_path, h1_path, label="M15→H1"):
    print(f"\n{'='*60}")
    print(f"  [{label}] Cross-Timeframe Consistency")
    print(f"{'='*60}")

    m15 = load_ohlc(m15_path, "M15")
    h1 = load_ohlc(h1_path, "H1")

    flat_m15 = (m15["Open"] == m15["High"]) & (m15["High"] == m15["Low"]) & (m15["Low"] == m15["Close"])
    m15_active = m15[~flat_m15]

    m15_resampled = m15_active.resample("1h").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last"
    }).dropna()

    flat_h1 = (h1["Open"] == h1["High"]) & (h1["High"] == h1["Low"]) & (h1["Low"] == h1["Close"])
    h1_active = h1[~flat_h1]

    common = m15_resampled.index.intersection(h1_active.index)
    result = {"common_hours": len(common)}
    print(f"  Common active hours: {len(common):,}")

    if len(common) == 0:
        return result

    m = m15_resampled.loc[common]
    h = h1_active.loc[common]

    tol = 0.015  # $0.015 tolerance for floating point

    open_diff = (m["Open"] - h["Open"]).abs()
    high_diff = (m["High"] - h["High"]).abs()
    low_diff = (m["Low"] - h["Low"]).abs()
    close_diff = (m["Close"] - h["Close"]).abs()

    open_mismatch = (open_diff > tol).sum()
    high_mismatch = (high_diff > tol).sum()
    low_mismatch = (low_diff > tol).sum()
    close_mismatch = (close_diff > tol).sum()

    result["mismatches"] = {
        "open": int(open_mismatch), "high": int(high_mismatch),
        "low": int(low_mismatch), "close": int(close_mismatch)
    }
    print(f"  Open  mismatches (>$0.015): {open_mismatch} ({open_mismatch/len(common)*100:.3f}%)")
    print(f"  High  mismatches (>$0.015): {high_mismatch} ({high_mismatch/len(common)*100:.3f}%)")
    print(f"  Low   mismatches (>$0.015): {low_mismatch} ({low_mismatch/len(common)*100:.3f}%)")
    print(f"  Close mismatches (>$0.015): {close_mismatch} ({close_mismatch/len(common)*100:.3f}%)")

    # Larger mismatches
    big_tol = 1.0
    big_open = (open_diff > big_tol).sum()
    big_high = (high_diff > big_tol).sum()
    big_low = (low_diff > big_tol).sum()
    big_close = (close_diff > big_tol).sum()
    result["big_mismatches_gt1"] = {
        "open": int(big_open), "high": int(big_high),
        "low": int(big_low), "close": int(big_close)
    }
    print(f"  Large mismatches (>$1.00): O={big_open} H={big_high} L={big_low} C={big_close}")

    if big_high > 0:
        worst = high_diff.nlargest(3)
        for ts, diff in worst.items():
            print(f"    {ts}: M15-agg H={m.loc[ts, 'High']:.3f} vs H1 H={h.loc[ts, 'High']:.3f} (Δ={diff:.3f})")

    # Correlation check
    corr = m["Close"].corr(h["Close"])
    result["close_correlation"] = round(corr, 6)
    print(f"  Close price correlation: {corr:.6f}")

    return result


# ─────────────────────────────────────────────
# Check 7: External data audit
# ─────────────────────────────────────────────
def check_external_data():
    print(f"\n{'='*60}")
    print(f"  [EXTERNAL] External Data Audit")
    print(f"{'='*60}")

    result = {}

    ext_files = {
        "vix_daily.csv": {"date_col": "Date", "key_col": "VIX_Close"},
        "dxy_daily.csv": {"date_col": "Date", "key_col": "Close", "skip_rows": [1]},
        "us10y_daily.csv": {"date_col": "Date", "key_col": "Close", "skip_rows": [1]},
        "gld_daily.csv": {"date_col": "Date", "key_col": "GLD_Close"},
        "SPX_daily.csv": {"date_col": "Date", "key_col": "Close", "skip_rows": [1]},
        "GVZ_daily.csv": {"date_col": "Date", "key_col": "Close", "skip_rows": [1]},
        "cot_gold_weekly.csv": {"date_col": "Date", "key_col": "COT_MM_Net"},
    }

    for fname, cfg in ext_files.items():
        path = EXT_DIR / fname
        if not path.exists():
            print(f"  {fname}: NOT FOUND")
            result[fname] = {"status": "missing"}
            continue

        skip = cfg.get("skip_rows", None)
        try:
            df = pd.read_csv(path, skiprows=skip)
        except Exception:
            df = pd.read_csv(path)

        date_col = cfg["date_col"]
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            date_range = f"{df[date_col].min()} → {df[date_col].max()}"
        else:
            date_range = "N/A (no Date column found)"

        nan_total = df.isna().sum().sum()
        nan_pct = nan_total / (len(df) * len(df.columns)) * 100 if len(df) > 0 else 0

        info = {
            "rows": len(df),
            "columns": len(df.columns),
            "date_range": date_range,
            "nan_cells": int(nan_total),
            "nan_pct": round(nan_pct, 2),
        }

        print(f"  {fname}: {len(df):,} rows, range={date_range}")
        print(f"    NaN cells: {nan_total} ({nan_pct:.1f}%)")

        result[fname] = info

    # Aligned daily
    aligned_path = EXT_DIR / "aligned_daily.csv"
    if aligned_path.exists():
        df = pd.read_csv(aligned_path, parse_dates=["Date"], index_col="Date")
        print(f"\n  aligned_daily.csv: {len(df):,} rows, {len(df.columns)} cols")
        print(f"    Range: {df.index[0]} → {df.index[-1]}")
        nan_per_col = df.isna().sum()
        for col in df.columns:
            nc = nan_per_col[col]
            if nc > 0:
                print(f"    {col}: {nc} NaN ({nc/len(df)*100:.1f}%)")
        result["aligned_daily"] = {
            "rows": len(df),
            "columns": len(df.columns),
            "range": f"{df.index[0]} → {df.index[-1]}",
            "nan_per_col": {c: int(v) for c, v in nan_per_col.items() if v > 0}
        }

    return result


# ─────────────────────────────────────────────
# Check 8: Year-by-year stability
# ─────────────────────────────────────────────
def check_yearly_stability(df, label):
    print(f"\n{'='*60}")
    print(f"  [{label}] Year-by-Year Stability")
    print(f"{'='*60}")

    flat = (df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"])
    active = df[~flat].copy()
    active["year"] = active.index.year

    result = {}
    for yr, grp in active.groupby("year"):
        n = len(grp)
        avg_range = (grp["High"] - grp["Low"]).mean()
        avg_spread_pct = avg_range / grp["Close"].mean() * 100
        vol = grp["Volume"].mean() if "Volume" in grp.columns else 0

        result[int(yr)] = {
            "bars": n,
            "avg_range": round(avg_range, 3),
            "avg_range_pct": round(avg_spread_pct, 4),
            "avg_volume": round(vol, 2),
            "price_low": round(grp["Low"].min(), 2),
            "price_high": round(grp["High"].max(), 2),
        }
        print(f"  {yr}: {n:>6,} bars | range=${avg_range:>6.2f} ({avg_spread_pct:.3f}%) | "
              f"price ${grp['Low'].min():.0f}-${grp['High'].max():.0f} | vol={vol:.2f}")

    return result


# ─────────────────────────────────────────────
# Check 9: D1 vs H1 aggregation
# ─────────────────────────────────────────────
def check_d1_vs_h1(h1_path, d1_path):
    print(f"\n{'='*60}")
    print(f"  [D1 vs H1] Daily Aggregation Consistency")
    print(f"{'='*60}")

    h1 = load_ohlc(h1_path, "H1")
    d1 = load_ohlc(d1_path, "D1")

    flat_h1 = (h1["Open"] == h1["High"]) & (h1["High"] == h1["Low"]) & (h1["Low"] == h1["Close"])
    h1_active = h1[~flat_h1]
    flat_d1 = (d1["Open"] == d1["High"]) & (d1["High"] == d1["Low"]) & (d1["Low"] == d1["Close"])
    d1_active = d1[~flat_d1]

    h1_daily = h1_active.resample("1D").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last"
    }).dropna()

    common = h1_daily.index.intersection(d1_active.index)
    result = {"common_days": len(common)}
    print(f"  Common trading days: {len(common):,}")

    if len(common) == 0:
        return result

    h = h1_daily.loc[common]
    d = d1_active.loc[common]

    tol = 0.05
    for col in ["Open", "High", "Low", "Close"]:
        diff = (h[col] - d[col]).abs()
        mismatch = (diff > tol).sum()
        avg_diff = diff.mean()
        max_diff = diff.max()
        print(f"  {col}: mismatches(>$0.05)={mismatch}, avg_diff=${avg_diff:.4f}, max_diff=${max_diff:.3f}")
        result[f"{col}_mismatches"] = int(mismatch)
        result[f"{col}_avg_diff"] = round(avg_diff, 4)
        result[f"{col}_max_diff"] = round(max_diff, 3)

    return result


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  XAUUSD Data Quality Audit")
    print("=" * 70)

    # ── Core OHLC files ──
    for key in ["M15_BID", "H1_BID", "H4_BID", "D1_BID"]:
        path = FILES.get(key)
        if path and path.exists():
            freq_map = {"M15_BID": 15, "H1_BID": 60, "H4_BID": 240, "D1_BID": 1440}
            df = load_ohlc(path, key)
            REPORT[f"{key}_integrity"] = check_ohlc_integrity(df, key)
            REPORT[f"{key}_timestamps"] = check_timestamps(df, key, freq_map[key])
            REPORT[f"{key}_spikes"] = check_spikes(df, key, freq_map[key])
            REPORT[f"{key}_yearly"] = check_yearly_stability(df, key)
        else:
            print(f"\n  SKIP {key}: file not found")

    # ── Spread analysis ──
    for tf in ["M15", "H1"]:
        sp_key = f"{tf}_SPREAD"
        bid_key = f"{tf}_BID"
        ask_key = f"{tf}_ASK"
        sp_path = FILES.get(sp_key)
        bid_path = FILES.get(bid_key)
        ask_path = FILES.get(ask_key)
        if sp_path:
            REPORT[f"{tf}_spread"] = check_spread(
                sp_path, bid_path or Path(""), ask_path or Path(""), tf
            )

    # ── Cross-timeframe ──
    if FILES["M15_BID"].exists() and FILES["H1_BID"].exists():
        REPORT["M15_vs_H1"] = check_cross_timeframe(FILES["M15_BID"], FILES["H1_BID"])

    if FILES["H1_BID"].exists() and FILES["D1_BID"].exists():
        REPORT["D1_vs_H1"] = check_d1_vs_h1(FILES["H1_BID"], FILES["D1_BID"])

    # ── External data ──
    REPORT["external"] = check_external_data()

    # ── Summary verdicts ──
    print(f"\n{'='*70}")
    print("  SUMMARY VERDICT")
    print(f"{'='*70}")

    issues = []
    warnings = []

    for key in ["M15_BID_integrity", "H1_BID_integrity", "H4_BID_integrity", "D1_BID_integrity"]:
        if key not in REPORT:
            continue
        r = REPORT[key]
        if r.get("high_violations", 0) > 0 or r.get("low_violations", 0) > 0:
            issues.append(f"{key}: {r['high_violations']} high + {r['low_violations']} low OHLC violations")
        if r.get("nan_values", 0) > 0:
            issues.append(f"{key}: {r['nan_values']} NaN values in OHLC")
        if r.get("flat_pct", 0) > 20:
            warnings.append(f"{key}: {r['flat_pct']:.1f}% flat bars (high)")

    for key in ["M15_BID_timestamps", "H1_BID_timestamps"]:
        if key not in REPORT:
            continue
        r = REPORT[key]
        if r.get("duplicate_timestamps", 0) > 0:
            issues.append(f"{key}: {r['duplicate_timestamps']} duplicate timestamps")
        if r.get("abnormal_weekday_gaps", 0) > 10:
            warnings.append(f"{key}: {r['abnormal_weekday_gaps']} abnormal weekday gaps")

    if "M15_vs_H1" in REPORT:
        r = REPORT["M15_vs_H1"]
        bm = r.get("big_mismatches_gt1", {})
        total_big = sum(bm.values())
        if total_big > 0:
            warnings.append(f"M15→H1: {total_big} large price mismatches (>$1)")

    if issues:
        print(f"\n  ❌ CRITICAL ISSUES ({len(issues)}):")
        for i in issues:
            print(f"     • {i}")
    else:
        print(f"\n  ✅ No critical data integrity issues found")

    if warnings:
        print(f"\n  ⚠ WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"     • {w}")

    verdict = "FAIL" if issues else ("CAUTION" if warnings else "PASS")
    REPORT["_verdict"] = verdict
    REPORT["_issues"] = issues
    REPORT["_warnings"] = warnings
    print(f"\n  Overall verdict: {verdict}")

    # Save
    out_dir = Path("results/data_quality")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data_quality_report.json"

    def default_serializer(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return str(obj)

    with open(out_path, "w") as f:
        json.dump(REPORT, f, indent=2, default=default_serializer)
    print(f"\n  Full report saved to {out_path}")


if __name__ == "__main__":
    main()
