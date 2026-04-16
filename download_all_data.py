"""
Download XAUUSD M15 + H1 data (BID + ASK) from Dukascopy.
Updates existing BID data and downloads full ASK data for spread modeling.

Downloads:
1. M15 BID: 2026-03-25 -> 2026-04-10 (incremental update)
2. H1  BID: 2026-03-25 -> 2026-04-10 (incremental update)
3. M15 ASK: 2015-01-01 -> 2026-04-10 (full, for spread model)
4. H1  ASK: 2015-01-01 -> 2026-04-10 (full, for spread model)
"""
import subprocess
import sys
import os
import io
from pathlib import Path
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUT_DIR = Path("data/download")
TEMP_DIR = OUT_DIR / "temp_chunks"

EXISTING_M15_BID = OUT_DIR / "xauusd-m15-bid-2015-01-01-2026-03-25.csv"
EXISTING_H1_BID = OUT_DIR / "xauusd-h1-bid-2015-01-01-2026-03-25.csv"

FINAL_M15_BID = OUT_DIR / "xauusd-m15-bid-2015-01-01-2026-04-10.csv"
FINAL_H1_BID = OUT_DIR / "xauusd-h1-bid-2015-01-01-2026-04-10.csv"
FINAL_M15_ASK = OUT_DIR / "xauusd-m15-ask-2015-01-01-2026-04-10.csv"
FINAL_H1_ASK = OUT_DIR / "xauusd-h1-ask-2015-01-01-2026-04-10.csv"

MAX_RETRIES = 3
BATCH_PAUSE = 1500

ASK_CHUNKS_6M = [
    ("2015-01-01", "2015-07-01"),
    ("2015-07-01", "2016-01-01"),
    ("2016-01-01", "2016-07-01"),
    ("2016-07-01", "2017-01-01"),
    ("2017-01-01", "2017-07-01"),
    ("2017-07-01", "2018-01-01"),
    ("2018-01-01", "2018-07-01"),
    ("2018-07-01", "2019-01-01"),
    ("2019-01-01", "2019-07-01"),
    ("2019-07-01", "2020-01-01"),
    ("2020-01-01", "2020-07-01"),
    ("2020-07-01", "2021-01-01"),
    ("2021-01-01", "2021-07-01"),
    ("2021-07-01", "2022-01-01"),
    ("2022-01-01", "2022-07-01"),
    ("2022-07-01", "2023-01-01"),
    ("2023-01-01", "2023-07-01"),
    ("2023-07-01", "2024-01-01"),
    ("2024-01-01", "2024-07-01"),
    ("2024-07-01", "2025-01-01"),
    ("2025-01-01", "2025-07-01"),
    ("2025-07-01", "2026-04-10"),
]


def download_one(instrument, date_from, date_to, timeframe, price_type, out_dir):
    """Download a single chunk. Returns True on success."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        f'npx dukascopy-node -i {instrument} '
        f'-from {date_from} -to {date_to} '
        f'-t {timeframe} -p {price_type} '
        f'-f csv -v --flats '
        f'-r 3 --retry-pause 2000 '
        f'-bs 10 -bp {BATCH_PAUSE} '
        f'-dir "{out_dir}"'
    )
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"    [{date_from} -> {date_to}] {timeframe} {price_type} attempt {attempt}...",
              end=" ", flush=True)
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=600, shell=True,
                encoding='utf-8', errors='replace',
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            if result.returncode == 0 and "File saved" in stdout:
                print("OK")
                return True
            else:
                err = (stderr.strip() or stdout.strip())[-120:]
                print(f"FAIL ({err})")
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
    return False


def merge_csvs(chunk_dir, output_file, existing_file=None):
    """Merge chunk CSVs (optionally appending to existing data)."""
    frames = []

    if existing_file and existing_file.exists():
        df_existing = pd.read_csv(existing_file)
        frames.append(df_existing)
        print(f"  Loaded existing: {existing_file.name} ({len(df_existing)} rows)")

    csv_files = sorted(chunk_dir.glob("*.csv"))
    for f in csv_files:
        df = pd.read_csv(f)
        frames.append(df)
        print(f"  + chunk: {f.name} ({len(df)} rows)")

    if not frames:
        print("  ERROR: No data to merge!")
        return None

    merged = pd.concat(frames, ignore_index=True)
    merged.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
    merged.sort_values('timestamp', inplace=True)
    merged.reset_index(drop=True, inplace=True)

    merged.to_csv(output_file, index=False)

    ts_start = pd.to_datetime(merged['timestamp'].iloc[0], unit='ms', utc=True)
    ts_end = pd.to_datetime(merged['timestamp'].iloc[-1], unit='ms', utc=True)
    print(f"  Merged: {len(merged)} rows -> {output_file.name}")
    print(f"  Range: {ts_start} -> {ts_end}")
    return merged


def download_incremental_bid():
    """Download only the missing BID data (2026-03-25 -> 2026-04-10)."""
    print("\n" + "=" * 60)
    print("Phase 1: Incremental BID update (2026-03-25 -> 2026-04-10)")
    print("=" * 60)

    bid_temp = TEMP_DIR / "bid_update"

    for tf, existing, final in [
        ("m15", EXISTING_M15_BID, FINAL_M15_BID),
        ("h1", EXISTING_H1_BID, FINAL_H1_BID),
    ]:
        print(f"\n--- {tf.upper()} BID ---")
        chunk_dir = bid_temp / tf
        ok = download_one("xauusd", "2026-03-25", "2026-04-10", tf, "bid", chunk_dir)
        if ok:
            merge_csvs(chunk_dir, final, existing)
        else:
            print(f"  FAILED to download {tf} BID update!")


def download_full_ask():
    """Download full ASK data (2015-2026) for spread modeling."""
    print("\n" + "=" * 60)
    print("Phase 2: Full ASK data download (2015-01-01 -> 2026-04-10)")
    print("=" * 60)

    for tf, final in [("m15", FINAL_M15_ASK), ("h1", FINAL_H1_ASK)]:
        print(f"\n--- {tf.upper()} ASK ---")
        chunk_dir = TEMP_DIR / f"ask_{tf}"
        failed = []

        for i, (date_from, date_to) in enumerate(ASK_CHUNKS_6M):
            print(f"  Chunk {i+1}/{len(ASK_CHUNKS_6M)}:")
            if not download_one("xauusd", date_from, date_to, tf, "ask", chunk_dir):
                failed.append((date_from, date_to))

        if failed:
            print(f"\n  WARNING: {len(failed)} chunks failed for {tf} ASK:")
            for f, t in failed:
                print(f"    {f} -> {t}")
        else:
            print(f"\n  All {len(ASK_CHUNKS_6M)} chunks downloaded!")

        merge_csvs(chunk_dir, final)


def verify_and_build_spread():
    """Build spread = ASK - BID and verify quality."""
    print("\n" + "=" * 60)
    print("Phase 3: Build spread time series")
    print("=" * 60)

    for tf in ["m15", "h1"]:
        bid_file = FINAL_M15_BID if tf == "m15" else FINAL_H1_BID
        ask_file = FINAL_M15_ASK if tf == "m15" else FINAL_H1_ASK

        if not bid_file.exists() or not ask_file.exists():
            print(f"  {tf.upper()}: Missing BID or ASK file, skipping spread calc")
            continue

        bid = pd.read_csv(bid_file)
        ask = pd.read_csv(ask_file)

        bid.set_index('timestamp', inplace=True)
        ask.set_index('timestamp', inplace=True)

        common = bid.index.intersection(ask.index)
        print(f"\n  {tf.upper()} spread analysis:")
        print(f"    BID bars: {len(bid)}")
        print(f"    ASK bars: {len(ask)}")
        print(f"    Common timestamps: {len(common)}")

        if len(common) == 0:
            print("    No common timestamps!")
            continue

        spread_open = ask.loc[common, 'open'] - bid.loc[common, 'open']
        spread_close = ask.loc[common, 'close'] - bid.loc[common, 'close']

        spread_df = pd.DataFrame({
            'timestamp': common,
            'spread_open': spread_open.values,
            'spread_close': spread_close.values,
            'spread_avg': ((spread_open + spread_close) / 2).values,
        })

        spread_file = OUT_DIR / f"xauusd-{tf}-spread-2015-01-01-2026-04-10.csv"
        spread_df.to_csv(spread_file, index=False)

        print(f"    Spread stats (close-based):")
        print(f"      Mean:   ${spread_close.mean():.4f}")
        print(f"      Median: ${spread_close.median():.4f}")
        print(f"      Std:    ${spread_close.std():.4f}")
        print(f"      Min:    ${spread_close.min():.4f}")
        print(f"      Max:    ${spread_close.max():.4f}")
        print(f"      < $0:   {(spread_close < 0).sum()} ({(spread_close < 0).mean()*100:.2f}%)")
        print(f"      > $1:   {(spread_close > 1).sum()} ({(spread_close > 1).mean()*100:.2f}%)")
        print(f"      > $5:   {(spread_close > 5).sum()} ({(spread_close > 5).mean()*100:.2f}%)")

        ts = pd.to_datetime(common, unit='ms', utc=True)
        spread_close_series = pd.Series(spread_close.values, index=ts)

        hours = ts.hour
        print(f"\n    Spread by hour (selected):")
        for h in [0, 3, 6, 8, 10, 13, 15, 18, 21]:
            mask = hours == h
            if mask.sum() > 0:
                mean_sp = spread_close_series[mask].mean()
                print(f"      Hour {h:2d}: ${mean_sp:.4f}  (n={mask.sum()})")

        print(f"    Saved: {spread_file.name}")


def main():
    print("XAUUSD Data Download & Spread Model Builder")
    print("=" * 60)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    download_incremental_bid()
    download_full_ask()
    verify_and_build_spread()

    print("\n" + "=" * 60)
    print("All done! Summary:")
    print("=" * 60)
    for f in [FINAL_M15_BID, FINAL_H1_BID, FINAL_M15_ASK, FINAL_H1_ASK]:
        if f.exists():
            size_mb = f.stat().st_size / 1024 / 1024
            df = pd.read_csv(f, nrows=1)
            total = sum(1 for _ in open(f)) - 1
            print(f"  {f.name}: {total:,} rows, {size_mb:.1f} MB")
        else:
            print(f"  {f.name}: NOT FOUND")

    spread_m15 = OUT_DIR / "xauusd-m15-spread-2015-01-01-2026-04-10.csv"
    spread_h1 = OUT_DIR / "xauusd-h1-spread-2015-01-01-2026-04-10.csv"
    for f in [spread_m15, spread_h1]:
        if f.exists():
            print(f"  {f.name}: READY")


if __name__ == "__main__":
    main()
