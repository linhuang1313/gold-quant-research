"""
Gold Quant Research — External Data Hub
========================================
Downloads, processes, and aligns all external data sources for
gold trading signal quality scoring, regime detection, and factor analysis.

Data Sources (17 series across 6 categories):
  ── Volatility & Fear ──
    1. VIX         (CBOE daily, 1990+)        — equity fear gauge
    2. GVZ         (yfinance, 2015+)          — gold-specific volatility
    3. Credit Spread / HYG  (yfinance, 2007+) — credit fear gauge

  ── Rates & Inflation ──
    4. US 10Y Yield   (yfinance, 2003+)       — nominal long-term rate
    5. TIPS / Real Yield (FRED DFII10, 2003+) — real interest rate (gold's pricing anchor)
    6. Fed Funds Rate    (FRED DFF, 2000+)     — policy rate
    7. US 2Y Yield       (yfinance, 2003+)     — short-term rate expectations

  ── Currencies ──
    8. DXY         (yfinance, 2003+)          — dollar index
    9. USDJPY      (yfinance, 2003+)          — yen (safe haven peer)
   10. USDCNH      (yfinance, 2015+)          — yuan (China gold demand proxy)

  ── Equities & Risk ──
   11. SPX / S&P 500 (yfinance, 2003+)        — risk appetite
   12. GLD ETF       (yfinance, 2004+)         — gold ETF price + volume

  ── Commodities ──
   13. Crude Oil WTI  (yfinance, 2003+)        — inflation / geopolitical proxy
   14. Copper HG      (yfinance, 2003+)        — growth cycle indicator

  ── Positioning & Flows ──
   15. COT Managed Money (CFTC weekly, 2006+)  — institutional positioning
   16. GLD Holdings      (SPDR daily, 2004+)   — ETF flow proxy

  ── Macro / Liquidity ──
   17. US M2 Money Supply (FRED M2SL, monthly) — liquidity backdrop

All outputs: data/external/ as CSV with DatetimeIndex.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = os.path.join(os.path.dirname(__file__), "external")
os.makedirs(OUT_DIR, exist_ok=True)

START = "2006-01-01"
END = datetime.now().strftime("%Y-%m-%d")

DOWNLOADED = {}


# ═══════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════

def _save(df, filename, label):
    path = os.path.join(OUT_DIR, filename)
    df.to_csv(path)
    DOWNLOADED[label] = {"rows": len(df), "file": filename,
                         "range": f"{df.index[0]} ~ {df.index[-1]}"}
    print(f"  Saved {len(df):,} rows → {filename}")
    print(f"  Range: {df.index[0]} ~ {df.index[-1]}")
    return df


def _yf_download(ticker, prefix, filename, start=START, end=END):
    """Generic yfinance daily OHLCV downloader."""
    print(f"\n[{prefix}] Downloading {ticker} from yfinance...")
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df is None or len(df) == 0:
            print(f"  [FAIL] No data for {ticker}")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        rename_map = {col: f"{prefix}_{col}" for col in df.columns}
        df = df.rename(columns=rename_map)
        df.index.name = "Date"
        return _save(df, filename, prefix)
    except Exception as e:
        print(f"  [FAIL] {prefix}: {e}")
        return None


def _fred_download(series_id, label, filename, start=START):
    """Download from FRED (Federal Reserve Economic Data) via CSV API."""
    print(f"\n[{label}] Downloading {series_id} from FRED...")
    try:
        url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv"
               f"?id={series_id}&cosd={start}&coed={END}")
        df = pd.read_csv(url)
        # FRED columns: observation_date, <series_id>
        date_col = [c for c in df.columns if "date" in c.lower()]
        date_col = date_col[0] if date_col else df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col)
        df.index.name = "Date"
        df.columns = [f"{label}_{c}" if not c.startswith(label) else c for c in df.columns]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(how="all")
        return _save(df, filename, label)
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return None


# ═══════════════════════════════════════════════════════════
# 1. VIX — CBOE Volatility Index
# ═══════════════════════════════════════════════════════════

def download_vix():
    print(f"\n[VIX] Downloading from CBOE...")
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    try:
        df = pd.read_csv(url, parse_dates=["DATE"], index_col="DATE")
        df.index.name = "Date"
        df = df.rename(columns={"OPEN": "VIX_Open", "HIGH": "VIX_High",
                                "LOW": "VIX_Low", "CLOSE": "VIX_Close"})
        df = df.loc[START:]
        return _save(df, "vix_daily.csv", "VIX")
    except Exception as e:
        print(f"  [WARN] CBOE direct failed: {e}")
        print("  Trying yfinance ^VIX fallback...")
        df = yf.download("^VIX", start=START, end=END, progress=False)
        if df is not None and len(df) > 0:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={"Open": "VIX_Open", "High": "VIX_High",
                                    "Low": "VIX_Low", "Close": "VIX_Close"})
            df = df[["VIX_Open", "VIX_High", "VIX_Low", "VIX_Close"]]
            df.index.name = "Date"
            return _save(df, "vix_daily.csv", "VIX")
        return None


# ═══════════════════════════════════════════════════════════
# 2. GVZ — Gold Volatility Index
# ═══════════════════════════════════════════════════════════

def download_gvz():
    return _yf_download("^GVZ", "GVZ", "GVZ_daily.csv")


# ═══════════════════════════════════════════════════════════
# 3. Credit Spread proxy — HYG (High Yield Corporate Bond ETF)
# ═══════════════════════════════════════════════════════════

def download_hyg():
    return _yf_download("HYG", "HYG", "hyg_daily.csv")


# ═══════════════════════════════════════════════════════════
# 4. US 10Y Yield
# ═══════════════════════════════════════════════════════════

def download_us10y():
    return _yf_download("^TNX", "US10Y", "us10y_daily.csv")


# ═══════════════════════════════════════════════════════════
# 5. TIPS / Real Yield (FRED DFII10)
# ═══════════════════════════════════════════════════════════

def download_real_yield():
    return _fred_download("DFII10", "REAL_YIELD", "real_yield_daily.csv", start="2003-01-01")


# ═══════════════════════════════════════════════════════════
# 6. Fed Funds Effective Rate (FRED DFF)
# ═══════════════════════════════════════════════════════════

def download_fed_funds():
    return _fred_download("DFF", "FED_FUNDS", "fed_funds_daily.csv", start="2000-01-01")


# ═══════════════════════════════════════════════════════════
# 7. US 2Y Yield — short-term rate expectations
# ═══════════════════════════════════════════════════════════

def download_us2y():
    # Try ^IRX (13-week T-bill) for longer history, then try 2Y futures
    result = _yf_download("^IRX", "US2Y", "us2y_daily.csv")
    if result is None or len(result) < 1000:
        print("  Trying 2YY=F futures as fallback...")
        result = _yf_download("2YY=F", "US2Y", "us2y_daily.csv")
    return result


# ═══════════════════════════════════════════════════════════
# 8. DXY — Dollar Index
# ═══════════════════════════════════════════════════════════

def download_dxy():
    return _yf_download("DX-Y.NYB", "DXY", "dxy_daily.csv")


# ═══════════════════════════════════════════════════════════
# 9. USDJPY — Japanese Yen
# ═══════════════════════════════════════════════════════════

def download_usdjpy():
    return _yf_download("JPY=X", "USDJPY", "usdjpy_daily.csv")


# ═══════════════════════════════════════════════════════════
# 10. USDCNH — Chinese Yuan (offshore)
# ═══════════════════════════════════════════════════════════

def download_usdcnh():
    return _yf_download("CNY=X", "USDCNH", "usdcnh_daily.csv")


# ═══════════════════════════════════════════════════════════
# 11. S&P 500
# ═══════════════════════════════════════════════════════════

def download_spx():
    return _yf_download("^GSPC", "SPX", "SPX_daily.csv")


# ═══════════════════════════════════════════════════════════
# 12. GLD ETF (price + volume)
# ═══════════════════════════════════════════════════════════

def download_gld():
    return _yf_download("GLD", "GLD", "gld_daily.csv")


# ═══════════════════════════════════════════════════════════
# 13. Crude Oil — WTI
# ═══════════════════════════════════════════════════════════

def download_crude():
    return _yf_download("CL=F", "CRUDE", "crude_wti_daily.csv")


# ═══════════════════════════════════════════════════════════
# 14. Copper — HG Futures
# ═══════════════════════════════════════════════════════════

def download_copper():
    return _yf_download("HG=F", "COPPER", "copper_daily.csv")


# ═══════════════════════════════════════════════════════════
# 15. COT Managed Money — CFTC Gold Positioning
# ═══════════════════════════════════════════════════════════

def download_cot():
    print(f"\n[COT] Downloading from CFTC...")
    try:
        from cot_reports import cot_all
        df = cot_all(cot_report_type="disaggregated_fut")

        market_col = date_col = None
        for c in df.columns:
            if "market" in c.lower() and "exchange" in c.lower():
                market_col = c
            if "report_date" in c.lower() or "yyyy-mm-dd" in c.lower():
                date_col = c
        if market_col is None:
            market_col = df.columns[0]
        if date_col is None:
            date_col = df.columns[2]

        gold_mask = df[market_col].str.contains("GOLD", case=False, na=False)
        gold = df[gold_mask].copy()
        if len(gold) == 0:
            print("  [FAIL] No gold rows found")
            return None

        gold["Date"] = pd.to_datetime(gold[date_col])
        gold = gold.set_index("Date").sort_index()

        long_col = short_col = None
        for col in gold.columns:
            if "M_Money_Positions_Long_All" in col:
                long_col = col
            elif "M_Money_Positions_Short_All" in col:
                short_col = col

        if long_col is None or short_col is None:
            for col in gold.columns:
                cl = col.lower()
                if "m_money" in cl and "long" in cl and "all" in cl and "spread" not in cl:
                    long_col = long_col or col
                if "m_money" in cl and "short" in cl and "all" in cl and "spread" not in cl:
                    short_col = short_col or col

        if long_col is None or short_col is None:
            print(f"  [FAIL] Cannot find MM cols")
            return None

        result = pd.DataFrame({
            "COT_MM_Long": pd.to_numeric(gold[long_col], errors="coerce"),
            "COT_MM_Short": pd.to_numeric(gold[short_col], errors="coerce"),
        }, index=gold.index)
        result["COT_MM_Net"] = result["COT_MM_Long"] - result["COT_MM_Short"]
        result = result[~result.index.duplicated(keep="last")]
        return _save(result, "cot_gold_weekly.csv", "COT")

    except Exception as e:
        print(f"  [FAIL] COT: {e}")
        import traceback; traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════════
# 16. GLD Holdings (tonnes) — ETF flow proxy
# ═══════════════════════════════════════════════════════════

def download_gld_holdings():
    """Download GLD ETF holdings (shares outstanding as proxy for tonnes).
    
    Primary: FRED GLDNSO (GLD Net Assets) or shares outstanding.
    Fallback: Derive from GLD NAV and price.
    """
    print(f"\n[GLD_HOLDINGS] Downloading GLD holdings proxy...")

    # Method 1: Use GLD shares outstanding from yfinance info
    try:
        import yfinance as yf
        gld = yf.Ticker("GLD")
        hist = gld.history(start="2006-01-01", end=END)
        if hist is not None and len(hist) > 0:
            # GLD shares outstanding * price ≈ total fund value
            # Each GLD share ≈ 1/10 oz gold initially, but ratio changes
            # We use volume-weighted cumulative flow as proxy
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            result = pd.DataFrame({
                "GLD_Holdings_Price": hist["Close"].values,
                "GLD_Holdings_Volume": hist["Volume"].values,
            }, index=hist.index)
            result.index.name = "Date"
            result = result.dropna()
            # Derive cumulative flow proxy: sum of signed volume * price
            result["GLD_Flow_Proxy"] = (result["GLD_Holdings_Volume"] * 
                                         result["GLD_Holdings_Price"].pct_change().apply(
                                             lambda x: 1 if x > 0 else -1 if x < 0 else 0
                                         )).cumsum()
            return _save(result, "gld_holdings_daily.csv", "GLD_HOLDINGS")
    except Exception as e:
        print(f"  [WARN] yfinance method: {e}")

    # Method 2: FRED total known ETF holdings
    try:
        result = _fred_download("GLDNSO", "GLD_HOLD", "gld_holdings_daily.csv", start="2006-01-01")
        if result is not None:
            return result
    except Exception:
        pass

    print("  [INFO] GLD holdings proxy will be derived from volume in alignment step")
    return None


# ═══════════════════════════════════════════════════════════
# 17. US M2 Money Supply (FRED, monthly)
# ═══════════════════════════════════════════════════════════

def download_m2():
    return _fred_download("M2SL", "M2", "m2_monthly.csv", start="2000-01-01")


# ═══════════════════════════════════════════════════════════
# Alignment: Build master daily dataset
# ═══════════════════════════════════════════════════════════

def build_aligned_daily():
    """Merge all external data into a single aligned daily DataFrame with derived features."""
    print(f"\n{'='*70}")
    print("  Building Aligned Master Dataset")
    print(f"{'='*70}")

    # ── Load all available files ──
    load_specs = [
        ("vix_daily.csv",           "VIX",          None),
        ("dxy_daily.csv",           "DXY",          [1]),    # skip ticker row
        ("us10y_daily.csv",         "US10Y",        [1]),
        ("us2y_daily.csv",          "US2Y",         [1]),
        ("gld_daily.csv",           "GLD",          None),
        ("SPX_daily.csv",           "SPX",          [1]),
        ("GVZ_daily.csv",           "GVZ",          [1]),
        ("hyg_daily.csv",           "HYG",          [1]),
        ("usdjpy_daily.csv",        "USDJPY",       [1]),
        ("usdcnh_daily.csv",        "USDCNH",       [1]),
        ("crude_wti_daily.csv",     "CRUDE",        [1]),
        ("copper_daily.csv",        "COPPER",       [1]),
        ("real_yield_daily.csv",    "REAL_YIELD",   None),
        ("fed_funds_daily.csv",     "FED_FUNDS",    None),
        ("cot_gold_weekly.csv",     "COT",          None),
        ("gld_holdings_daily.csv",  "GLD_HOLD",     None),
        ("m2_monthly.csv",          "M2",           None),
    ]

    frames = {}
    for fname, key, skip_rows in load_specs:
        path = os.path.join(OUT_DIR, fname)
        if not os.path.exists(path):
            print(f"  [SKIP] {fname} not found")
            continue
        try:
            df = pd.read_csv(path, skiprows=skip_rows) if skip_rows else pd.read_csv(path)
            # Find date column
            date_col = None
            for c in df.columns:
                if c.lower() in ("date", "date.1"):
                    date_col = c
                    break
            if date_col is None:
                date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            df = df.set_index(date_col)
            df.index.name = "Date"
            df = df[~df.index.duplicated(keep="last")]
            frames[key] = df
            print(f"  Loaded {key}: {len(df):,} rows ({df.index[0].date()} → {df.index[-1].date()})")
        except Exception as e:
            print(f"  [WARN] Failed to load {fname}: {e}")

    if not frames:
        print("  [FAIL] No data to align")
        return None

    # ── Build daily index ──
    date_range = pd.bdate_range(start=START, end=END)
    aligned = pd.DataFrame(index=date_range)
    aligned.index.name = "Date"

    for key, df in frames.items():
        aligned = aligned.join(df, how="left")

    aligned = aligned.ffill()

    # ── Derived Features ──
    print("\n  Computing derived features...")

    # Volatility
    if "VIX_Close" in aligned.columns:
        aligned["VIX_SMA20"] = aligned["VIX_Close"].rolling(20).mean()
        aligned["VIX_Zscore"] = (
            (aligned["VIX_Close"] - aligned["VIX_Close"].rolling(60).mean())
            / aligned["VIX_Close"].rolling(60).std()
        )
        aligned["VIX_Term_Structure"] = np.nan  # placeholder if we add VIX futures later

    # Dollar dynamics
    if "DXY_Close" in aligned.columns:
        aligned["DXY_Mom5"] = aligned["DXY_Close"].pct_change(5)
        aligned["DXY_Mom20"] = aligned["DXY_Close"].pct_change(20)
        aligned["DXY_SMA50"] = aligned["DXY_Close"].rolling(50).mean()

    # Rates
    if "US10Y_Close" in aligned.columns:
        aligned["US10Y_Change5"] = aligned["US10Y_Close"].diff(5)
        aligned["US10Y_Change20"] = aligned["US10Y_Close"].diff(20)

    # Yield curve: 10Y - 2Y spread
    if "US10Y_Close" in aligned.columns and "US2Y_Close" in aligned.columns:
        aligned["YIELD_CURVE_10Y2Y"] = aligned["US10Y_Close"] - aligned["US2Y_Close"]

    # Real yield dynamics
    if "REAL_YIELD_DFII10" in aligned.columns:
        aligned["REAL_YIELD_Change5"] = aligned["REAL_YIELD_DFII10"].diff(5)
        aligned["REAL_YIELD_Change20"] = aligned["REAL_YIELD_DFII10"].diff(20)
        aligned["REAL_YIELD_SMA20"] = aligned["REAL_YIELD_DFII10"].rolling(20).mean()

    # GLD volume
    if "GLD_Volume" in aligned.columns:
        aligned["GLD_Vol_SMA20"] = aligned["GLD_Volume"].rolling(20).mean()
        aligned["GLD_Vol_Ratio"] = aligned["GLD_Volume"] / aligned["GLD_Vol_SMA20"]

    # COT features
    if "COT_MM_Net" in aligned.columns:
        aligned["COT_MM_Net_Zscore"] = (
            (aligned["COT_MM_Net"] - aligned["COT_MM_Net"].rolling(52).mean())
            / aligned["COT_MM_Net"].rolling(52).std()
        )
        aligned["COT_MM_Net_Pct"] = aligned["COT_MM_Net"].rolling(104).rank(pct=True)

    # Credit spread proxy: IEF-equivalent return - HYG return (higher = more stress)
    if "HYG_Close" in aligned.columns and "US10Y_Close" in aligned.columns:
        hyg_ret = aligned["HYG_Close"].pct_change()
        aligned["CREDIT_STRESS"] = -hyg_ret.rolling(5).mean() * 100  # higher = more stress

    # Copper/Gold ratio (growth vs safety)
    if "COPPER_Close" in aligned.columns and "GLD_Close" in aligned.columns:
        aligned["COPPER_GOLD_RATIO"] = aligned["COPPER_Close"] / aligned["GLD_Close"]
        aligned["CG_RATIO_Mom20"] = aligned["COPPER_GOLD_RATIO"].pct_change(20)

    # Oil dynamics
    if "CRUDE_Close" in aligned.columns:
        aligned["CRUDE_Mom5"] = aligned["CRUDE_Close"].pct_change(5)
        aligned["CRUDE_Mom20"] = aligned["CRUDE_Close"].pct_change(20)

    # Yen dynamics
    if "USDJPY_Close" in aligned.columns:
        aligned["USDJPY_Mom5"] = aligned["USDJPY_Close"].pct_change(5)

    # Yuan dynamics
    if "USDCNH_Close" in aligned.columns:
        aligned["USDCNH_Mom5"] = aligned["USDCNH_Close"].pct_change(5)
        aligned["USDCNH_Mom20"] = aligned["USDCNH_Close"].pct_change(20)

    # M2 liquidity growth — column may be M2_M2SL or M2SL depending on FRED format
    m2_col = "M2_M2SL" if "M2_M2SL" in aligned.columns else ("M2SL" if "M2SL" in aligned.columns else None)
    if m2_col:
        aligned["M2_YoY"] = aligned[m2_col].pct_change(252)  # ~1 year in bdays
        aligned["M2_Mom3M"] = aligned[m2_col].pct_change(63)  # ~3 months

    # GLD Holdings changes
    if "GLD_Holdings_Tonnes" in aligned.columns:
        aligned["GLD_Holdings_Change5"] = aligned["GLD_Holdings_Tonnes"].diff(5)
        aligned["GLD_Holdings_Change20"] = aligned["GLD_Holdings_Tonnes"].diff(20)

    # Risk-on/off composite: SPX return + HYG return - VIX change (simple proxy)
    risk_components = []
    if "SPX_Close" in aligned.columns:
        aligned["SPX_Mom5"] = aligned["SPX_Close"].pct_change(5)
        risk_components.append("SPX_Mom5")
    if "HYG_Close" in aligned.columns:
        aligned["HYG_Mom5"] = aligned["HYG_Close"].pct_change(5)
        risk_components.append("HYG_Mom5")
    if len(risk_components) >= 1:
        aligned["RISK_APPETITE"] = sum(aligned[c] for c in risk_components) / len(risk_components)
        # Z-score for regime detection
        aligned["RISK_APPETITE_Z"] = (
            (aligned["RISK_APPETITE"] - aligned["RISK_APPETITE"].rolling(60).mean())
            / aligned["RISK_APPETITE"].rolling(60).std()
        )

    # Drop rows before first valid data
    aligned = aligned.dropna(how="all")

    # Remove any columns that are entirely NaN
    all_nan_cols = [c for c in aligned.columns if aligned[c].isna().all()]
    if all_nan_cols:
        print(f"  Dropping all-NaN columns: {all_nan_cols}")
        aligned = aligned.drop(columns=all_nan_cols)

    # ── Save ──
    path = os.path.join(OUT_DIR, "aligned_daily.csv")
    aligned.to_csv(path)

    print(f"\n  {'─'*50}")
    print(f"  Master Dataset: {len(aligned):,} rows × {len(aligned.columns)} columns")
    print(f"  Range: {aligned.index[0].date()} → {aligned.index[-1].date()}")
    print(f"  {'─'*50}")

    # Category summary
    categories = {
        "Volatility":  [c for c in aligned.columns if any(x in c for x in ["VIX", "GVZ"])],
        "Rates":       [c for c in aligned.columns if any(x in c for x in ["US10Y", "US2Y", "REAL_YIELD", "FED_FUNDS", "YIELD_CURVE"])],
        "Currencies":  [c for c in aligned.columns if any(x in c for x in ["DXY", "USDJPY", "USDCNH"])],
        "Equities":    [c for c in aligned.columns if any(x in c for x in ["SPX", "GLD", "HYG"])],
        "Commodities": [c for c in aligned.columns if any(x in c for x in ["CRUDE", "COPPER", "CG_RATIO"])],
        "Positioning":  [c for c in aligned.columns if any(x in c for x in ["COT", "GLD_Hold"])],
        "Macro":       [c for c in aligned.columns if any(x in c for x in ["M2", "RISK_APPETITE"])],
    }

    for cat, cols in categories.items():
        if cols:
            nn_min = min(aligned[c].notna().sum() for c in cols)
            nn_max = max(aligned[c].notna().sum() for c in cols)
            print(f"  {cat:14s}: {len(cols):>3} series  (non-null: {nn_min:,}–{nn_max:,})")

    print(f"\n  Non-null counts per column:")
    for col in sorted(aligned.columns):
        nn = aligned[col].notna().sum()
        pct = nn / len(aligned) * 100
        print(f"    {col:40s}: {nn:>5,} ({pct:5.1f}%)")

    print(f"\n  Saved to {path}")
    return aligned


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  Gold Quant Research — External Data Hub")
    print(f"  Date range: {START} ~ {END}")
    print("=" * 70)

    t0 = time.time()

    # ── Volatility & Fear ──
    download_vix()
    download_gvz()
    download_hyg()

    # ── Rates & Inflation ──
    download_us10y()
    download_real_yield()
    download_fed_funds()
    download_us2y()

    # ── Currencies ──
    download_dxy()
    download_usdjpy()
    download_usdcnh()

    # ── Equities & Risk ──
    download_spx()
    download_gld()

    # ── Commodities ──
    download_crude()
    download_copper()

    # ── Positioning & Flows ──
    download_cot()
    download_gld_holdings()

    # ── Macro ──
    download_m2()

    # ── Align ──
    build_aligned_daily()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Download Summary ({elapsed:.0f}s)")
    print(f"{'='*70}")
    for label, info in DOWNLOADED.items():
        print(f"  {label:20s}: {info['rows']:>6,} rows  ({info['file']})")
    print(f"\n  Total sources: {len(DOWNLOADED)}")
    failed = set(["VIX","GVZ","HYG","US10Y","REAL_YIELD","FED_FUNDS","US2Y",
                   "DXY","USDJPY","USDCNH","SPX","GLD","CRUDE","COPPER",
                   "COT","GLD_HOLDINGS","M2"]) - set(DOWNLOADED.keys())
    if failed:
        print(f"  Failed: {', '.join(sorted(failed))}")
    print(f"{'='*70}")
