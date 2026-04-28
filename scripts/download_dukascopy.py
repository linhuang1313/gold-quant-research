"""Dukascopy 全量数据下载器
=============================
下载 XAUUSD / EURUSD / XAGUSD 全套历史数据 (M15, H1, H4, D1, Bid/Ask).
Spread 由 Ask - Bid 计算得出。

输出格式与现有 data/download/ 文件一致:
  timestamp (ms), open, high, low, close, volume

用法:
  python scripts/download_dukascopy.py             # 全量下载
  python scripts/download_dukascopy.py --xau-only   # 只更新 XAUUSD
  python scripts/download_dukascopy.py --quick       # 只更新 XAUUSD 增量 (4/10→今天)
"""
import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import dukascopy_python as dk
from dukascopy_python.instruments import (
    INSTRUMENT_FX_METALS_XAU_USD,
    INSTRUMENT_FX_MAJORS_EUR_USD,
    INSTRUMENT_FX_METALS_XAG_USD,
)

ROOT = Path(__file__).resolve().parent.parent
DL_DIR = ROOT / "data" / "download"
DL_DIR.mkdir(parents=True, exist_ok=True)

TODAY = datetime.utcnow().date()
TODAY_STR = TODAY.strftime("%Y-%m-%d")

INTERVALS = {
    "m15": dk.INTERVAL_MIN_15,
    "h1":  dk.INTERVAL_HOUR_1,
    "h4":  dk.INTERVAL_HOUR_4,
    "d1":  dk.INTERVAL_DAY_1,
}

INSTRUMENTS = {
    "xauusd": INSTRUMENT_FX_METALS_XAU_USD,
    "eurusd": INSTRUMENT_FX_MAJORS_EUR_USD,
    "xagusd": INSTRUMENT_FX_METALS_XAG_USD,
}

def to_ms_timestamp(dt_index: pd.DatetimeIndex) -> pd.Series:
    """Convert DatetimeIndex to millisecond timestamps (int64)."""
    return (dt_index.astype('int64') // 10**6)


def fetch_chunked(instrument, interval, offer_side, start: datetime, end: datetime,
                  chunk_months: int = 6, label: str = "") -> pd.DataFrame:
    """Fetch data in chunks to avoid timeout on large date ranges."""
    all_dfs = []
    chunk_start = start
    chunk_num = 0

    while chunk_start < end:
        chunk_end = min(
            chunk_start + timedelta(days=chunk_months * 30),
            end
        )
        chunk_num += 1
        print(f"    Chunk {chunk_num}: {chunk_start.strftime('%Y-%m-%d')} → "
              f"{chunk_end.strftime('%Y-%m-%d')} ...", end="", flush=True)

        retries = 0
        while retries < 3:
            try:
                df = dk.fetch(
                    instrument, interval, offer_side,
                    chunk_start, chunk_end,
                    max_retries=5
                )
                break
            except Exception as e:
                retries += 1
                print(f" RETRY {retries}/3 ({e})", end="", flush=True)
                time.sleep(5 * retries)
        else:
            print(f" FAILED after 3 retries, skipping chunk")
            chunk_start = chunk_end
            continue

        if df is not None and len(df) > 0:
            all_dfs.append(df)
            print(f" {len(df)} bars", flush=True)
        else:
            print(f" 0 bars", flush=True)

        chunk_start = chunk_end
        time.sleep(1)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs)
    combined = combined[~combined.index.duplicated(keep='first')]
    combined = combined.sort_index()
    return combined


def save_ohlcv(df: pd.DataFrame, filepath: Path):
    """Save DataFrame in the existing CSV format (ms timestamp, OHLCV)."""
    if df.empty:
        print(f"    SKIP (empty): {filepath.name}")
        return

    out = pd.DataFrame({
        'timestamp': to_ms_timestamp(df.index),
        'open': df['open'],
        'high': df['high'],
        'low': df['low'],
        'close': df['close'],
        'volume': df['volume'],
    })
    out.to_csv(filepath, index=False)
    print(f"    SAVED: {filepath.name} ({len(out)} rows, {filepath.stat().st_size/1024/1024:.1f} MB)")


def compute_spread(bid_path: Path, ask_path: Path, spread_path: Path):
    """Compute spread OHLCV from bid and ask files."""
    if not bid_path.exists() or not ask_path.exists():
        print(f"    SKIP spread (missing bid/ask): {spread_path.name}")
        return

    bid = pd.read_csv(bid_path)
    ask = pd.read_csv(ask_path)

    merged = bid.merge(ask, on='timestamp', suffixes=('_bid', '_ask'))

    spread = pd.DataFrame({
        'timestamp': merged['timestamp'],
        'open':  merged['open_ask']  - merged['open_bid'],
        'high':  merged['high_ask']  - merged['low_bid'],
        'low':   merged['low_ask']   - merged['high_bid'],
        'close': merged['close_ask'] - merged['close_bid'],
        'volume': 0,
    })
    spread.to_csv(spread_path, index=False)
    print(f"    SAVED spread: {spread_path.name} ({len(spread)} rows)")


def download_instrument(symbol: str, timeframes: list, start: datetime, end: datetime):
    """Download bid + ask + spread for one instrument across multiple timeframes."""
    instrument = INSTRUMENTS[symbol]
    date_suffix = f"{start.strftime('%Y-%m-%d')}-{end.strftime('%Y-%m-%d')}"

    for tf in timeframes:
        interval = INTERVALS[tf]
        print(f"\n  [{symbol.upper()} {tf.upper()}]")

        for side_name, side in [("bid", dk.OFFER_SIDE_BID), ("ask", dk.OFFER_SIDE_ASK)]:
            label = f"{symbol}-{tf}-{side_name}"
            filename = f"{symbol}-{tf}-{side_name}-{date_suffix}.csv"
            filepath = DL_DIR / filename

            if filepath.exists():
                existing = pd.read_csv(filepath)
                print(f"    EXISTS: {filename} ({len(existing)} rows) - SKIPPING")
                continue

            print(f"    Downloading {label}...")
            t0 = time.time()
            df = fetch_chunked(instrument, interval, side, start, end, label=label)
            dt = time.time() - t0
            print(f"    Fetched {len(df)} bars in {dt:.0f}s")
            save_ohlcv(df, filepath)

        bid_path = DL_DIR / f"{symbol}-{tf}-bid-{date_suffix}.csv"
        ask_path = DL_DIR / f"{symbol}-{tf}-ask-{date_suffix}.csv"
        spread_path = DL_DIR / f"{symbol}-{tf}-spread-{date_suffix}.csv"
        compute_spread(bid_path, ask_path, spread_path)


def update_xauusd_incremental():
    """Only download new data since last file (2026-04-10 → today)."""
    start = datetime(2026, 4, 10)
    end = datetime(TODAY.year, TODAY.month, TODAY.day)

    if start >= end:
        print("XAUUSD already up to date!")
        return

    print(f"\n{'='*60}")
    print(f"  XAUUSD Incremental Update: {start.date()} → {end.date()}")
    print(f"{'='*60}")

    instrument = INSTRUMENTS["xauusd"]
    date_suffix = f"{start.strftime('%Y-%m-%d')}-{end.strftime('%Y-%m-%d')}"

    for tf in ["m15", "h1"]:
        interval = INTERVALS[tf]
        for side_name, side in [("bid", dk.OFFER_SIDE_BID), ("ask", dk.OFFER_SIDE_ASK)]:
            filename = f"xauusd-{tf}-{side_name}-{date_suffix}.csv"
            filepath = DL_DIR / filename

            print(f"\n  Downloading xauusd-{tf}-{side_name}...")
            df = fetch_chunked(instrument, interval, side, start, end)
            save_ohlcv(df, filepath)

        bid_path = DL_DIR / f"xauusd-{tf}-bid-{date_suffix}.csv"
        ask_path = DL_DIR / f"xauusd-{tf}-ask-{date_suffix}.csv"
        spread_path = DL_DIR / f"xauusd-{tf}-spread-{date_suffix}.csv"
        compute_spread(bid_path, ask_path, spread_path)


def merge_xauusd():
    """Merge incremental XAUUSD data with existing full files."""
    date_suffix_new = f"2026-04-10-{TODAY_STR}"
    date_suffix_full = f"2015-01-01-{TODAY_STR}"
    date_suffix_old = "2015-01-01-2026-04-10"

    for tf in ["m15", "h1"]:
        for data_type in ["bid", "ask", "spread"]:
            old_file = DL_DIR / f"xauusd-{tf}-{data_type}-{date_suffix_old}.csv"
            new_file = DL_DIR / f"xauusd-{tf}-{data_type}-{date_suffix_new}.csv"
            merged_file = DL_DIR / f"xauusd-{tf}-{data_type}-{date_suffix_full}.csv"

            if not old_file.exists():
                print(f"  SKIP merge {tf}-{data_type}: old file not found")
                continue
            if not new_file.exists():
                print(f"  SKIP merge {tf}-{data_type}: new file not found")
                continue

            old_df = pd.read_csv(old_file)
            new_df = pd.read_csv(new_file)

            combined = pd.concat([old_df, new_df])
            combined = combined.drop_duplicates(subset='timestamp', keep='last')
            combined = combined.sort_values('timestamp')

            combined.to_csv(merged_file, index=False)
            print(f"  MERGED: {merged_file.name} — {len(old_df)} + {len(new_df)} "
                  f"→ {len(combined)} rows ({merged_file.stat().st_size/1024/1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Dukascopy data downloader")
    parser.add_argument("--quick", action="store_true",
                        help="Only update XAUUSD incrementally (4/10 → today)")
    parser.add_argument("--xau-only", action="store_true",
                        help="Only download XAUUSD (all timeframes)")
    parser.add_argument("--no-merge", action="store_true",
                        help="Skip merge step for XAUUSD")
    args = parser.parse_args()

    t_start = time.time()

    START = datetime(2015, 1, 1)
    END = datetime(TODAY.year, TODAY.month, TODAY.day)

    if args.quick:
        update_xauusd_incremental()
        if not args.no_merge:
            print(f"\n{'='*60}")
            print(f"  Merging XAUUSD files...")
            print(f"{'='*60}")
            merge_xauusd()
        print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
        return

    # ── Task 1: XAUUSD 增量更新 ──
    update_xauusd_incremental()

    # ── Task 2: XAUUSD D1 + H4 (new timeframes) ──
    print(f"\n{'='*60}")
    print(f"  XAUUSD D1 + H4 Full Download")
    print(f"{'='*60}")
    download_instrument("xauusd", ["d1", "h4"], START, END)

    if not args.xau_only:
        # ── Task 3: EURUSD full (M15 + H1, Bid/Ask/Spread) ──
        print(f"\n{'='*60}")
        print(f"  EURUSD Full Download (2015 → {TODAY_STR})")
        print(f"{'='*60}")
        download_instrument("eurusd", ["m15", "h1"], START, END)

        # ── Task 4: XAGUSD full (M15 + H1, Bid/Ask/Spread) ──
        print(f"\n{'='*60}")
        print(f"  XAGUSD Full Download (2015 → {TODAY_STR})")
        print(f"{'='*60}")
        download_instrument("xagusd", ["m15", "h1"], START, END)

    # ── Task 5: Merge XAUUSD ──
    if not args.no_merge:
        print(f"\n{'='*60}")
        print(f"  Merging XAUUSD files...")
        print(f"{'='*60}")
        merge_xauusd()

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  ALL DOWNLOADS COMPLETE")
    print(f"  Total time: {total/60:.1f} min ({total/3600:.1f} hours)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
