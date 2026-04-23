"""
Download external data for R33: DXY, US10Y, SPX, GVZ from Yahoo Finance.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from datetime import datetime

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False
    print("yfinance not installed. Installing...")
    os.system(f"{sys.executable} -m pip install yfinance -q")
    import yfinance as yf

OUT_DIR = Path("data/external")
OUT_DIR.mkdir(parents=True, exist_ok=True)

tickers = {
    'DXY': 'DX-Y.NYB',
    'US10Y': '^TNX',
    'SPX': '^GSPC',
    'GVZ': '^GVZ',
}

start = "2015-01-01"
end = datetime.now().strftime("%Y-%m-%d")

print(f"Downloading external data: {start} to {end}\n")

for name, ticker in tickers.items():
    print(f"  [{name}] Downloading {ticker}...", end='', flush=True)
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df is not None and len(df) > 0:
            out_path = OUT_DIR / f"{name}_daily.csv"
            df.to_csv(out_path)
            print(f" OK ({len(df)} bars, {df.index[0].date()} -> {df.index[-1].date()})")
        else:
            print(f" EMPTY (no data returned)")
    except Exception as e:
        print(f" FAILED: {e}")

print(f"\nData saved to {OUT_DIR}/")
print("Files:")
for f in OUT_DIR.glob("*.csv"):
    print(f"  {f.name} ({f.stat().st_size/1024:.1f} KB)")
