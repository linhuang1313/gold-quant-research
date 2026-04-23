"""
Download XAUUSD M1 data from HistData.com (2015-2026)
Uses the `histdata` Python package.

API rules:
  - Past years (< current year): download per YEAR with month=None
  - Current year: download per MONTH

CSV format (semicolon-separated, no header):
  DateTime;Open;High;Low;Close;Volume
  20150101 170000;1184.130;1184.130;1183.630;1183.930;0

DateTime is EST (no DST).
"""
import os
import sys
import zipfile
from pathlib import Path

try:
    from histdata import download_hist_data as dl
    from histdata.api import Platform as P, TimeFrame as TF
except ImportError:
    print("Installing histdata package...")
    os.system(f"{sys.executable} -m pip install histdata")
    from histdata import download_hist_data as dl
    from histdata.api import Platform as P, TimeFrame as TF

import pandas as pd

OUT_DIR = Path(__file__).parent / "download"
RAW_DIR = Path(__file__).parent / "m1_raw"
FINAL_CSV = OUT_DIR / "xauusd-m1-bid-2015-01-01-2026-04-10.csv"

START_YEAR = 2015
CURRENT_YEAR = 2026
CURRENT_MAX_MONTH = 4  # April 2026


def download_all():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for year in range(START_YEAR, CURRENT_YEAR + 1):
        if year < CURRENT_YEAR:
            # Past years: download whole year at once, month=None
            csv_check = RAW_DIR / f"xauusd_{year}.csv"
            if csv_check.exists() and csv_check.stat().st_size > 10000:
                print(f"  {year} — already have CSV ({csv_check.stat().st_size/1024/1024:.1f} MB), skip")
                continue

            print(f"  Downloading {year} (full year)...", end=" ", flush=True)
            try:
                result = dl(
                    year=str(year),
                    month=None,
                    pair='xauusd',
                    platform=P.GENERIC_ASCII,
                    time_frame=TF.ONE_MINUTE,
                )
                if result is None:
                    print("NO DATA (None returned)")
                    continue

                zip_path = Path(result)
                if not zip_path.exists():
                    print(f"NO FILE at {result}")
                    continue

                with zipfile.ZipFile(str(zip_path), 'r') as zf:
                    csv_names = [n for n in zf.namelist() if n.endswith('.csv')]
                    if not csv_names:
                        print("NO CSV in zip")
                        continue
                    zf.extract(csv_names[0], str(RAW_DIR))
                    extracted = RAW_DIR / csv_names[0]
                    if extracted != csv_check:
                        extracted.rename(csv_check)

                zip_path.unlink(missing_ok=True)
                size_mb = csv_check.stat().st_size / 1024 / 1024
                print(f"OK ({size_mb:.1f} MB)")

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback; traceback.print_exc()
                continue
        else:
            # Current year: download by month
            for month in range(1, CURRENT_MAX_MONTH + 1):
                label = f"{year}/{month:02d}"
                csv_check = RAW_DIR / f"xauusd_{year}_{month:02d}.csv"
                if csv_check.exists() and csv_check.stat().st_size > 1000:
                    print(f"  {label} — already have CSV, skip")
                    continue

                print(f"  Downloading {label}...", end=" ", flush=True)
                try:
                    result = dl(
                        year=str(year),
                        month=str(month),
                        pair='xauusd',
                        platform=P.GENERIC_ASCII,
                        time_frame=TF.ONE_MINUTE,
                    )
                    if result is None:
                        print("NO DATA")
                        continue

                    zip_path = Path(result)
                    if not zip_path.exists():
                        print(f"NO FILE")
                        continue

                    with zipfile.ZipFile(str(zip_path), 'r') as zf:
                        csv_names = [n for n in zf.namelist() if n.endswith('.csv')]
                        if not csv_names:
                            print("NO CSV in zip")
                            continue
                        zf.extract(csv_names[0], str(RAW_DIR))
                        extracted = RAW_DIR / csv_names[0]
                        if extracted != csv_check:
                            extracted.rename(csv_check)

                    zip_path.unlink(missing_ok=True)
                    size_mb = csv_check.stat().st_size / 1024 / 1024
                    print(f"OK ({size_mb:.1f} MB)")

                except Exception as e:
                    print(f"ERROR: {e}")
                    continue


def merge_csvs():
    """Merge all CSVs into one sorted DataFrame and save."""
    print("\nMerging all CSVs...")
    all_files = sorted(RAW_DIR.glob("xauusd_*.csv"))
    if not all_files:
        print("No CSV files found!")
        return

    dfs = []
    for f in all_files:
        print(f"  Reading {f.name}...", end=" ", flush=True)
        try:
            df = pd.read_csv(
                f,
                sep=';',
                header=None,
                names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'],
            )
            print(f"{len(df):,} rows")
            dfs.append(df)
        except Exception as e:
            print(f"ERROR: {e}")

    if not dfs:
        print("No data!")
        return

    merged = pd.concat(dfs, ignore_index=True)
    merged['DateTime'] = merged['DateTime'].str.strip()
    merged['DateTime'] = pd.to_datetime(merged['DateTime'], format='%Y%m%d %H%M%S')
    merged = merged.sort_values('DateTime').drop_duplicates('DateTime')
    merged = merged.reset_index(drop=True)

    # EST → UTC (+5h)
    merged['DateTime'] = merged['DateTime'] + pd.Timedelta(hours=5)

    merged.rename(columns={'DateTime': 'Gmt time'}, inplace=True)
    merged['Gmt time'] = merged['Gmt time'].dt.strftime('%d.%m.%Y %H:%M:%S.000')

    FINAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(FINAL_CSV, index=False)

    n = len(merged)
    print(f"\nDone! {n:,} M1 bars saved to {FINAL_CSV}")
    print(f"  File size: {FINAL_CSV.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Date range: {merged['Gmt time'].iloc[0]} -> {merged['Gmt time'].iloc[-1]}")


if __name__ == '__main__':
    print("=" * 60)
    print("XAUUSD M1 Data Downloader (HistData.com)")
    print("=" * 60)
    print(f"Range: {START_YEAR} - {CURRENT_YEAR}")
    print(f"Raw dir: {RAW_DIR}")
    print(f"Output:  {FINAL_CSV}\n")

    download_all()
    merge_csvs()
