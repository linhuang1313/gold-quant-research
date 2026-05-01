"""
Auto Data Update — download latest XAUUSD data and rebuild M15/H1 CSVs.

Two modes:
  1. histdata: Download from HistData.com (current month) and rebuild
  2. server:   Pull latest CSVs from the compute server via SFTP

Usage:
    python -m monitor.update_data                  # default: histdata mode
    python -m monitor.update_data --mode server     # pull from server
"""
import os
import sys
import zipfile
import datetime as dt
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE / "data"
DOWNLOAD_DIR = DATA_DIR / "download"
RAW_DIR = DATA_DIR / "m1_raw"


def _current_h1_path() -> Path:
    """Find the latest H1 CSV in the download directory."""
    candidates = sorted(DOWNLOAD_DIR.glob("xauusd-h1-bid-*.csv"), reverse=True)
    if candidates:
        return candidates[0]
    return DOWNLOAD_DIR / "xauusd-h1-bid-2015-01-01-2026-04-27.csv"


def _current_m15_path() -> Path:
    candidates = sorted(DOWNLOAD_DIR.glob("xauusd-m15-bid-*.csv"), reverse=True)
    if candidates:
        return candidates[0]
    return DOWNLOAD_DIR / "xauusd-m15-bid-2015-01-01-2026-04-27.csv"


def get_data_info() -> dict:
    """Report current data coverage."""
    info = {}
    for label, finder in [("H1", _current_h1_path), ("M15", _current_m15_path)]:
        path = finder()
        if path.exists():
            df = pd.read_csv(path, nrows=5)
            df_tail = pd.read_csv(path, skiprows=max(0, sum(1 for _ in open(path)) - 5))
            info[label] = {
                "path": str(path),
                "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
                "exists": True,
            }
        else:
            info[label] = {"path": str(path), "exists": False}
    return info


def update_from_histdata():
    """Download latest months from HistData.com and rebuild M1 -> M15/H1."""
    try:
        from histdata import download_hist_data as dl
        from histdata.api import Platform as P, TimeFrame as TF
    except ImportError:
        print("Installing histdata package...")
        os.system(f"{sys.executable} -m pip install histdata")
        from histdata import download_hist_data as dl
        from histdata.api import Platform as P, TimeFrame as TF

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    now = dt.datetime.utcnow()
    year = now.year
    month = now.month

    months_to_check = []
    if month > 1:
        months_to_check.append((year, month - 1))
    months_to_check.append((year, month))

    downloaded_any = False
    for y, m in months_to_check:
        csv_check = RAW_DIR / f"xauusd_{y}_{m:02d}.csv"
        if m == month:
            pass  # always re-download current month
        elif csv_check.exists() and csv_check.stat().st_size > 10000:
            print(f"  {y}/{m:02d} — already have data, skip")
            continue

        print(f"  Downloading {y}/{m:02d}...", end=" ", flush=True)
        try:
            result = dl(
                year=str(y), month=str(m),
                pair="xauusd",
                platform=P.GENERIC_ASCII,
                time_frame=TF.ONE_MINUTE,
            )
            if result is None:
                print("NO DATA (month not available yet)")
                continue

            zip_path = Path(result)
            if not zip_path.exists():
                print("NO FILE")
                continue

            with zipfile.ZipFile(str(zip_path), "r") as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
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
            downloaded_any = True

        except Exception as e:
            print(f"ERROR: {e}")

    if not downloaded_any:
        print("  No new data downloaded.")
        return False

    print("\n  Rebuilding M15 and H1 from M1 data...")
    _rebuild_from_m1()
    return True


def _rebuild_from_m1():
    """Merge all M1 CSVs and resample to M15 and H1."""
    all_files = sorted(RAW_DIR.glob("xauusd_*.csv"))
    if not all_files:
        print("  No M1 CSV files found in", RAW_DIR)
        return

    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(
                f, sep=";", header=None,
                names=["DateTime", "Open", "High", "Low", "Close", "Volume"],
            )
            dfs.append(df)
        except Exception as e:
            print(f"  Error reading {f.name}: {e}")

    if not dfs:
        return

    merged = pd.concat(dfs, ignore_index=True)
    merged["DateTime"] = merged["DateTime"].str.strip()
    merged["DateTime"] = pd.to_datetime(merged["DateTime"], format="%Y%m%d %H%M%S")
    merged = merged.sort_values("DateTime").drop_duplicates("DateTime")

    # EST -> UTC (+5h)
    merged["DateTime"] = merged["DateTime"] + pd.Timedelta(hours=5)
    merged = merged.set_index("DateTime")

    start_str = merged.index[0].strftime("%Y-%m-%d")
    end_str = merged.index[-1].strftime("%Y-%m-%d")

    for tf_label, rule in [("m15", "15min"), ("h1", "1h")]:
        resampled = merged.resample(rule).agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna(subset=["Open"])

        # Convert to millisecond timestamp format to match existing CSVs
        out_df = pd.DataFrame({
            "timestamp": (resampled.index.astype("int64") // 10**6).values,
            "open": resampled["Open"].values,
            "high": resampled["High"].values,
            "low": resampled["Low"].values,
            "close": resampled["Close"].values,
            "volume": resampled["Volume"].values,
        })

        out_path = DOWNLOAD_DIR / f"xauusd-{tf_label}-bid-{start_str}-{end_str}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  {tf_label.upper()}: {len(resampled)} bars -> {out_path.name}")


def update_from_server():
    """Pull latest data CSVs from the compute server."""
    import paramiko

    HOST = "connect.westd.seetacloud.com"
    PORT = 41109
    USER = "root"
    PASS = "3sCdENtzYfse"
    REMOTE_DIR = "/root/gold-quant-research/data/download"

    print("  Connecting to server...", flush=True)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=90, banner_timeout=180)
    c.get_transport().set_keepalive(30)

    sftp = c.open_sftp()
    _, out, _ = c.exec_command(
        f"ls -1 {REMOTE_DIR}/xauusd-*-bid-*.csv 2>/dev/null", timeout=15
    )
    remote_files = out.read().decode("utf-8", errors="replace").strip().split("\n")
    remote_files = [f.strip() for f in remote_files if f.strip()]

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for rf in remote_files:
        fname = os.path.basename(rf)
        local_path = DOWNLOAD_DIR / fname
        remote_size = sftp.stat(rf).st_size
        local_size = local_path.stat().st_size if local_path.exists() else 0

        if local_size >= remote_size and local_size > 0:
            print(f"  {fname} — up to date, skip")
            continue

        print(f"  Downloading {fname} ({remote_size / 1024 / 1024:.1f} MB)...", end=" ", flush=True)
        sftp.get(rf, str(local_path))
        print("OK")
        downloaded += 1

    sftp.close()
    c.close()

    if downloaded > 0:
        print(f"\n  Downloaded {downloaded} file(s)")
    else:
        print("  All files up to date.")
    return downloaded > 0


def run(mode: str = "histdata"):
    """Main entry point."""
    print("=" * 60)
    print("  XAUUSD Data Updater")
    print("=" * 60)

    info = get_data_info()
    for label, d in info.items():
        if d.get("exists"):
            print(f"  Current {label}: {d['path']} ({d['size_mb']} MB)")
        else:
            print(f"  Current {label}: {d['path']} (NOT FOUND)")

    print(f"\n  Mode: {mode}\n")

    if mode == "histdata":
        updated = update_from_histdata()
    elif mode == "server":
        updated = update_from_server()
    else:
        print(f"  Unknown mode: {mode}")
        return False

    if updated:
        print("\n  Data updated. Re-run validation to check for strategy decay.")
    return updated


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="XAUUSD Data Updater")
    parser.add_argument("--mode", choices=["histdata", "server"], default="histdata")
    args = parser.parse_args()
    run(args.mode)
