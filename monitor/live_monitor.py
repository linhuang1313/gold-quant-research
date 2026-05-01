"""
Live Trade Monitor — reads MT4 trade CSVs and checks stop criteria.

Expects CSV files written by TradeLogger.mqh in the MT4 Common Files folder:
  <EA_NAME>_trades.csv  with columns:
    ticket,symbol,type,open_time,close_time,open_price,close_price,
    lots,pnl,commission,swap,reason,magic,comment

Usage:
    python -m monitor.live_monitor                     # one-shot dashboard
    python -m monitor.live_monitor --loop --interval 15  # poll every 15 min
"""
import os
import json
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent

MT4_FILES_DIR = Path(
    os.environ.get(
        "MT4_FILES_DIR",
        r"C:\Program Files (x86)\EMXPRO MT4 Terminal\MQL4\Files",
    )
)

MT4_COMMON_DIR = Path(
    os.environ.get(
        "MT4_COMMON_DIR",
        str(Path(os.environ.get("APPDATA", "")) / "MetaQuotes" / "Terminal" / "Common" / "Files"),
    )
)

MT4_TERMINAL_DIR = Path(
    os.environ.get(
        "MT4_TERMINAL_DIR",
        str(Path(os.environ.get("APPDATA", ""))
            / "MetaQuotes" / "Terminal"
            / "35EEC3EFDB656AF6FC775F21FEAD053B" / "MQL4" / "Files"),
    )
)

STOP_CRITERIA: Dict[str, dict] = {
    "PSAR_H1": {
        "max_drawdown_live": 615.09,
        "max_monthly_loss_live": 353.74,
        "max_consecutive_loss_days": 4,
    },
    "SESS_BO": {
        "max_drawdown_live": 458.41,
        "max_monthly_loss_live": 145.26,
        "max_consecutive_loss_days": 6,
    },
    "TSMOM": {
        "max_drawdown_live": 241.20,
        "max_monthly_loss_live": 101.19,
        "max_consecutive_loss_days": 4,
    },
    "L8_MAX": {
        "max_drawdown_live": 325.02,
        "max_monthly_loss_live": 17.00,
        "max_consecutive_loss_days": 9,
    },
    "H4_KC": {
        "max_drawdown_live": 842.25,
        "max_monthly_loss_live": 248.54,
        "max_consecutive_loss_days": 6,
    },
    "D1_KC": {
        "max_drawdown_live": 842.25,
        "max_monthly_loss_live": 248.54,
        "max_consecutive_loss_days": 6,
    },
}

LOG_PATH = WORKSPACE / "monitor" / "monitor.log"


def _find_trade_csvs() -> List[Path]:
    """Search MT4 Files, Common Files, and Terminal data for *_trades.csv."""
    found = []
    for d in [MT4_FILES_DIR, MT4_COMMON_DIR, MT4_TERMINAL_DIR]:
        if d.exists():
            found.extend(d.glob("*_trades.csv"))
    return sorted(set(found))


def _parse_csv(path: Path) -> pd.DataFrame:
    """Read one EA trade CSV into a DataFrame."""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    for col in ["open_time", "close_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "pnl" not in df.columns:
        return pd.DataFrame()

    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0)
    return df


def load_all_trades() -> Dict[str, pd.DataFrame]:
    """Load trade records grouped by EA name."""
    csvs = _find_trade_csvs()
    result: Dict[str, pd.DataFrame] = {}
    for path in csvs:
        ea_name = path.stem.replace("_trades", "")
        df = _parse_csv(path)
        if not df.empty:
            result[ea_name] = df
    return result


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute live risk metrics from a DataFrame of closed trades."""
    if df.empty:
        return {
            "n_trades": 0, "total_pnl": 0, "current_dd": 0,
            "worst_month_pnl": 0, "max_consec_loss_days": 0,
        }

    pnl = df["pnl"].values
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    current_dd = float(dd[-1]) if len(dd) > 0 else 0.0
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    worst_month_pnl = 0.0
    if "close_time" in df.columns and df["close_time"].notna().any():
        monthly = df.set_index("close_time").resample("ME")["pnl"].sum()
        if len(monthly) > 0:
            worst_month_pnl = float(monthly.min())

    max_consec = 0
    if "close_time" in df.columns and df["close_time"].notna().any():
        daily = df.set_index("close_time").resample("D")["pnl"].sum()
        daily = daily[daily != 0]
        streak = 0
        for v in daily.values:
            if v < 0:
                streak += 1
                max_consec = max(max_consec, streak)
            else:
                streak = 0

    return {
        "n_trades": len(df),
        "total_pnl": float(cum[-1]) if len(cum) > 0 else 0.0,
        "current_dd": current_dd,
        "max_dd": max_dd,
        "worst_month_pnl": worst_month_pnl,
        "max_consec_loss_days": max_consec,
    }


def check_alerts(ea_name: str, metrics: dict) -> List[str]:
    """Check metrics against stop criteria. Returns list of alert strings."""
    alerts = []
    criteria = STOP_CRITERIA.get(ea_name)
    if criteria is None:
        return alerts

    if abs(metrics["max_dd"]) > criteria["max_drawdown_live"]:
        alerts.append(
            f"DRAWDOWN BREACH: ${abs(metrics['max_dd']):.0f} > limit ${criteria['max_drawdown_live']:.0f}"
        )

    if abs(metrics["worst_month_pnl"]) > criteria["max_monthly_loss_live"]:
        alerts.append(
            f"MONTHLY LOSS BREACH: ${abs(metrics['worst_month_pnl']):.0f} > limit ${criteria['max_monthly_loss_live']:.0f}"
        )

    if metrics["max_consec_loss_days"] > criteria["max_consecutive_loss_days"]:
        alerts.append(
            f"CONSEC LOSS BREACH: {metrics['max_consec_loss_days']} days > limit {criteria['max_consecutive_loss_days']}"
        )

    return alerts


def print_dashboard(all_trades: Dict[str, pd.DataFrame]) -> List[str]:
    """Print a formatted console dashboard and return alert lines."""
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    width = 100

    lines = []
    lines.append("=" * width)
    lines.append(f"  GOLD QUANT LIVE MONITOR  |  {now}")
    lines.append("=" * width)

    header = (
        f"{'Strategy':<12} | {'Trades':>6} | {'PnL':>10} | {'CurDD':>8} | "
        f"{'MaxDD':>8} | {'DD Lim':>8} | {'MonthPnL':>10} | {'M Lim':>8} | "
        f"{'CnsLs':>5} | {'C Lim':>5} | {'Status'}"
    )
    lines.append(header)
    lines.append("-" * width)

    all_alerts = []
    total_pnl = 0.0
    total_trades = 0

    ea_order = ["PSAR_H1", "SESS_BO", "TSMOM", "L8_MAX", "H4_KC", "D1_KC"]
    ea_names = [n for n in ea_order if n in all_trades]
    ea_names += [n for n in all_trades if n not in ea_names]

    for ea_name in ea_names:
        df = all_trades[ea_name]
        m = compute_metrics(df)
        alerts = check_alerts(ea_name, m)
        all_alerts.extend([(ea_name, a) for a in alerts])

        total_pnl += m["total_pnl"]
        total_trades += m["n_trades"]

        criteria = STOP_CRITERIA.get(ea_name, {})
        dd_lim = criteria.get("max_drawdown_live", 0)
        m_lim = criteria.get("max_monthly_loss_live", 0)
        c_lim = criteria.get("max_consecutive_loss_days", 0)

        status = "ALERT!" if alerts else "OK"

        line = (
            f"{ea_name:<12} | {m['n_trades']:>6} | "
            f"{'$' + format(m['total_pnl'], ',.0f'):>10} | "
            f"{'$' + format(m['current_dd'], ',.0f'):>8} | "
            f"{'$' + format(abs(m['max_dd']), ',.0f'):>8} | "
            f"{'$' + format(dd_lim, ',.0f'):>8} | "
            f"{'$' + format(m['worst_month_pnl'], ',.0f'):>10} | "
            f"{'$' + format(m_lim, ',.0f'):>8} | "
            f"{m['max_consec_loss_days']:>5} | "
            f"{c_lim:>5} | "
            f"{status}"
        )
        lines.append(line)

    lines.append("-" * width)
    lines.append(
        f"{'PORTFOLIO':<12} | {total_trades:>6} | "
        f"{'$' + format(total_pnl, ',.0f'):>10} | "
        f"{'':>8} | {'':>8} | {'':>8} | {'':>10} | {'':>8} | "
        f"{'':>5} | {'':>5} | "
        f"{'OK' if not all_alerts else 'ALERT!'}"
    )
    lines.append("=" * width)

    if all_alerts:
        lines.append("")
        lines.append("*** ALERTS ***")
        for ea_name, alert_msg in all_alerts:
            lines.append(f"  [{ea_name}] {alert_msg}")
        lines.append("")

    output = "\n".join(lines)
    print(output)
    return [f"[{n}] {a}" for n, a in all_alerts]


def log_to_file(alerts: List[str]):
    """Append a timestamped log entry."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now().isoformat()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        if alerts:
            for a in alerts:
                f.write(f"{now} ALERT {a}\n")
        else:
            f.write(f"{now} OK all strategies within limits\n")


def run_once():
    """Single check cycle."""
    all_trades = load_all_trades()
    if not all_trades:
        dirs_checked = [str(MT4_FILES_DIR), str(MT4_COMMON_DIR), str(MT4_TERMINAL_DIR)]
        print("No trade CSV files found. Searched:")
        for d in dirs_checked:
            exists = Path(d).exists()
            print(f"  {d} {'(exists)' if exists else '(not found)'}")
        print("\nEnsure TradeLogger.mqh is integrated and EAs have produced trades.")
        print("You can set MT4_FILES_DIR or MT4_COMMON_DIR env vars to override paths.")
        log_to_file(["No trade files found"])
        return

    alerts = print_dashboard(all_trades)
    log_to_file(alerts)


def run_loop(interval_minutes: int = 15):
    """Continuous monitoring loop."""
    import time
    print(f"Starting live monitor loop (interval={interval_minutes}min)")
    print(f"Press Ctrl+C to stop.\n")
    while True:
        try:
            run_once()
            print(f"\nNext check in {interval_minutes} minutes...\n")
            time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gold Quant Live Trade Monitor")
    parser.add_argument("--loop", action="store_true", help="Run in continuous loop")
    parser.add_argument("--interval", type=int, default=15, help="Minutes between checks (default: 15)")
    args = parser.parse_args()

    if args.loop:
        run_loop(args.interval)
    else:
        run_once()
