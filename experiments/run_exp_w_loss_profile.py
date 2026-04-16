#!/usr/bin/env python3
"""
EXP-W: Diagnostic loss profile — entry features, exits, streaks, sessions (baseline LIVE_PARITY + sp$0.30).
"""
from __future__ import annotations

import io
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

try:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
except Exception:
    pass

sys.path.insert(0, ".")

import research_config as config  # noqa: E402
from backtest.engine import BacktestEngine, TradeRecord  # noqa: E402
from backtest.runner import DataBundle, LIVE_PARITY_KWARGS, calc_stats, run_variant  # noqa: E402


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t


def h1_closed_bar_index(entry_time, h1_df: pd.DataFrame) -> int:
    et = _ts(entry_time)
    return int(h1_df.index.searchsorted(et, side="right")) - 2


def m15_bar_index(entry_time, m15_df: pd.DataFrame) -> int:
    et = _ts(entry_time)
    return int(m15_df.index.searchsorted(et, side="right")) - 1


def keltner_breakout_strength(trade: TradeRecord, h1_df: pd.DataFrame) -> float:
    idx = h1_closed_bar_index(trade.entry_time, h1_df)
    if idx < 0 or idx >= len(h1_df):
        return float("nan")
    row = h1_df.iloc[idx]
    atr = float(row.get("ATR", np.nan))
    if not np.isfinite(atr) or atr <= 0:
        return float("nan")
    ku = float(row.get("KC_upper", np.nan))
    kl = float(row.get("KC_lower", np.nan))
    if not np.isfinite(ku) or not np.isfinite(kl):
        return float("nan")
    ep = float(trade.entry_price)
    if trade.direction == "BUY":
        return (ep - ku) / atr
    if trade.direction == "SELL":
        return (kl - ep) / atr
    return float("nan")


def intraday_trend_score_at_entry(entry_time, h1_df: pd.DataFrame) -> float:
    """Same formula as BacktestEngine._calc_realtime_score / IntradayTrendMeter."""
    et = _ts(entry_time)
    idx = h1_closed_bar_index(entry_time, h1_df)
    if idx < 0:
        return float("nan")
    d = et.date()
    valid = [i for i in range(idx + 1) if h1_df.index[i].date() == d]
    if len(valid) < 2:
        return 0.5
    today_bars = h1_df.iloc[valid]
    return float(BacktestEngine._calc_realtime_score(today_bars))


def kc_position_h1(entry_time, h1_df: pd.DataFrame) -> float:
    idx = h1_closed_bar_index(entry_time, h1_df)
    if idx < 0 or idx >= len(h1_df):
        return float("nan")
    row = h1_df.iloc[idx]
    c = float(row["Close"])
    ku = float(row.get("KC_upper", np.nan))
    kl = float(row.get("KC_lower", np.nan))
    if not np.isfinite(ku) or not np.isfinite(kl) or abs(ku - kl) < 1e-9:
        return float("nan")
    return (c - kl) / (ku - kl)


def exit_reason_bucket(reason: str) -> str:
    s = str(reason)
    if s == "SL":
        return "SL"
    if s == "TP":
        return "TP"
    if s == "Trailing":
        return "Trailing"
    if "TimeDecay" in s or "timedecay" in s.lower():
        return "TimeDecayTP"
    if s.startswith("Timeout"):
        return "Timeout"
    return "Other"


def session_bucket_utc(hour: int) -> str:
    if 0 <= hour < 8:
        return "Asia_0_8"
    if 8 <= hour < 14:
        return "London_8_14"
    if 14 <= hour < 21:
        return "NY_14_21"
    return "Off_21_24"


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled <= 0 or not np.isfinite(pooled):
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def extract_trade_features(
    trade: TradeRecord, m15_df: pd.DataFrame, h1_df: pd.DataFrame
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    et = _ts(trade.entry_time)
    hi = h1_closed_bar_index(trade.entry_time, h1_df)
    mi = m15_bar_index(trade.entry_time, m15_df)

    if 0 <= hi < len(h1_df):
        hr = h1_df.iloc[hi]
        out["h1_ATR"] = float(hr.get("ATR", np.nan))
        out["h1_ADX"] = float(hr.get("ADX", np.nan))
        out["h1_atr_percentile"] = float(hr.get("atr_percentile", np.nan))
    else:
        out["h1_ATR"] = np.nan
        out["h1_ADX"] = np.nan
        out["h1_atr_percentile"] = np.nan

    if 0 <= mi < len(m15_df):
        mr = m15_df.iloc[mi]
        out["m15_RSI14"] = float(mr.get("RSI14", np.nan))
    else:
        out["m15_RSI14"] = np.nan

    out["trend_score"] = intraday_trend_score_at_entry(trade.entry_time, h1_df)
    out["kc_position_h1"] = kc_position_h1(trade.entry_time, h1_df)
    out["breakout_strength"] = (
        keltner_breakout_strength(trade, h1_df)
        if trade.strategy == "keltner"
        else np.nan
    )
    out["entry_hour_utc"] = int(et.hour)
    out["entry_dow"] = int(et.dayofweek)
    out["session"] = session_bucket_utc(int(et.hour))
    out["bars_held"] = int(trade.bars_held)
    out["exit_reason"] = str(trade.exit_reason)
    out["pnl"] = float(trade.pnl)
    out["strategy"] = trade.strategy
    out["direction"] = trade.direction
    return out


def feature_comparison_table(
    winners: pd.DataFrame, losers: pd.DataFrame, numeric_cols: List[str]
) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        a = winners[col].to_numpy(dtype=float)
        b = losers[col].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        row: Dict[str, Any] = {"feature": col}
        row["win_mean"] = np.mean(a) if len(a) else np.nan
        row["lose_mean"] = np.mean(b) if len(b) else np.nan
        row["win_p25_median_p75"] = (
            f"{np.percentile(a,25):.4g}/{np.percentile(a,50):.4g}/{np.percentile(a,75):.4g}"
            if len(a)
            else "-"
        )
        row["lose_p25_median_p75"] = (
            f"{np.percentile(b,25):.4g}/{np.percentile(b,50):.4g}/{np.percentile(b,75):.4g}"
            if len(b)
            else "-"
        )
        if len(a) > 1 and len(b) > 1:
            try:
                tt = stats.ttest_ind(a, b, equal_var=False)
                row["ttest_p"] = float(tt.pvalue)
            except Exception:
                row["ttest_p"] = np.nan
            try:
                mu = stats.mannwhitneyu(a, b, alternative="two-sided")
                row["mannwhitney_p"] = float(mu.pvalue)
            except Exception:
                row["mannwhitney_p"] = np.nan
        else:
            row["ttest_p"] = np.nan
            row["mannwhitney_p"] = np.nan
        row["cohens_d_win_minus_lose"] = cohens_d(
            winners[col].to_numpy(dtype=float), losers[col].to_numpy(dtype=float)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    print("\n" + "=" * 80)
    print("  EXP-W: LOSS PROFILE (diagnostic)")
    print("=" * 80)

    t0 = time.time()
    try:
        bundle = DataBundle.load_default()
    except Exception as e:
        print(f"[FATAL] Data load failed: {e}")
        return

    base_kw = {**LIVE_PARITY_KWARGS, "spread_cost": 0.30}
    try:
        baseline = run_variant(bundle, "LIVE_PARITY sp$0.30", **base_kw)
    except Exception as e:
        print(f"[FATAL] Baseline backtest failed: {e}")
        return

    trades: List[TradeRecord] = baseline.get("_trades") or []
    ec = baseline.get("_equity_curve") or []
    st = calc_stats(trades, ec)

    print(f"\n  Baseline: N={st['n']}  Sharpe={st['sharpe']:.3f}  PnL=${st['total_pnl']:.0f}  WR={st['win_rate']:.1f}%")

    rows: List[Dict[str, Any]] = []
    for t in trades:
        try:
            rows.append(extract_trade_features(t, bundle.m15_df, bundle.h1_df))
        except Exception as ex:
            rows.append({"error": str(ex), "pnl": t.pnl})

    feat_df = pd.DataFrame(rows)
    if "pnl" not in feat_df.columns:
        print("[FATAL] Feature extraction produced no pnl column")
        return

    winners = feat_df[feat_df["pnl"] > 0].copy()
    losers = feat_df[feat_df["pnl"] <= 0].copy()
    print(f"\n  Winners: {len(winners)}  |  Losers: {len(losers)}")

    numeric_cols = [
        "h1_ATR",
        "h1_atr_percentile",
        "m15_RSI14",
        "h1_ADX",
        "trend_score",
        "kc_position_h1",
        "breakout_strength",
        "entry_hour_utc",
        "entry_dow",
        "bars_held",
    ]
    numeric_cols = [c for c in numeric_cols if c in feat_df.columns]

    cmp_tbl = feature_comparison_table(winners, losers, numeric_cols)
    print("\n" + "=" * 80)
    print("  NUMERIC FEATURES: winners vs losers (mean, percentiles, tests)")
    print("=" * 80)
    print(cmp_tbl.to_string(index=False))

    # --- Loser exit reasons ---
    print("\n" + "=" * 80)
    print("  LOSER EXIT REASON BREAKDOWN (% of losers)")
    print("=" * 80)
    if len(losers):
        vc = losers["exit_reason"].value_counts(normalize=True) * 100.0
        er_df = vc.reset_index()
        er_df.columns = ["exit_reason", "pct"]
        print("  Raw reasons:")
        print(er_df.to_string(index=False))
        los2 = losers.copy()
        los2["exit_bucket"] = los2["exit_reason"].map(exit_reason_bucket)
        vc2 = los2["exit_bucket"].value_counts(normalize=True) * 100.0
        er2 = vc2.reset_index()
        er2.columns = ["exit_bucket", "pct"]
        print("\n  Grouped buckets (SL / TP / Trailing / TimeDecayTP / Timeout / Other):")
        print(er2.to_string(index=False))
    else:
        print("  (no losers)")

    # --- Losing streaks ---
    print("\n" + "=" * 80)
    print("  CONSECUTIVE LOSS STREAKS (by exit_time order)")
    print("=" * 80)
    tdf = pd.DataFrame(
        [
            {
                "exit_time": _ts(t.exit_time),
                "entry_time": _ts(t.entry_time),
                "pnl": t.pnl,
                "strategy": t.strategy,
            }
            for t in trades
        ]
    ).sort_values("exit_time")
    is_loss = (tdf["pnl"] <= 0).to_numpy()
    streak_lens = []
    i = 0
    while i < len(is_loss):
        if is_loss[i]:
            j = i
            while j < len(is_loss) and is_loss[j]:
                j += 1
            streak_lens.append(j - i)
            i = j
        else:
            i += 1
    if streak_lens:
        sl = pd.Series(streak_lens)
        print(f"  Streak count: {len(sl)}  max_len={int(sl.max())}  mean_len={sl.mean():.2f}")
        print(f"  Streak length distribution:\n{sl.value_counts().sort_index().to_string()}")
    else:
        print("  No losing streaks.")

    # Entry conditions for first trade in each streak of len >= 3
    print("\n" + "=" * 80)
    print("  ENTRIES IN 3+ LOSS STREAKS (first trade of each streak)")
    print("=" * 80)
    streak_first_idx: List[int] = []
    i = 0
    while i < len(is_loss):
        if is_loss[i]:
            j = i
            while j < len(is_loss) and is_loss[j]:
                j += 1
            L = j - i
            if L >= 3:
                streak_first_idx.append(i)
            i = j
        else:
            i += 1

    if streak_first_idx:
        first_trades = tdf.iloc[streak_first_idx]
        # map back to full TradeRecord by exit_time
        tr_map = {_ts(x.exit_time): x for x in trades}
        st_feat = []
        for et in first_trades["exit_time"]:
            tr = tr_map.get(et)
            if tr is None:
                continue
            try:
                st_feat.append(extract_trade_features(tr, bundle.m15_df, bundle.h1_df))
            except Exception:
                pass
        if st_feat:
            sdf = pd.DataFrame(st_feat)
            print(
                f"  N streak-starts={len(sdf)}  session counts:\n{sdf['session'].value_counts().to_string()}"
            )
            print(
                f"  Mean trend_score={sdf['trend_score'].mean():.3f}  "
                f"Mean h1_atr_pct={sdf['h1_atr_percentile'].mean():.3f}  "
                f"Mean ADX={sdf['h1_ADX'].mean():.1f}"
            )
    else:
        print("  No streaks of length >= 3.")

    # --- Top 20 losses ---
    print("\n" + "=" * 80)
    print("  TOP 20 LARGEST LOSSES (detail)")
    print("=" * 80)
    worst = sorted(trades, key=lambda x: x.pnl)[:20]
    for rank, t in enumerate(worst, 1):
        et = _ts(t.entry_time)
        hi = h1_closed_bar_index(t.entry_time, bundle.h1_df)
        atrp = h1_adx = rsi14 = float("nan")
        if 0 <= hi < len(bundle.h1_df):
            hr = bundle.h1_df.iloc[hi]
            atrp = float(hr.get("atr_percentile", np.nan))
            h1_adx = float(hr.get("ADX", np.nan))
        mi = m15_bar_index(t.entry_time, bundle.m15_df)
        if 0 <= mi < len(bundle.m15_df):
            rsi14 = float(bundle.m15_df.iloc[mi].get("RSI14", np.nan))
        print(
            f"  #{rank:02d}  pnl=${t.pnl:.2f}  {t.strategy} {t.direction}  "
            f"entry={et}  exit={_ts(t.exit_time)}  reason={t.exit_reason}  "
            f"bars={t.bars_held}  atr_pct={atrp:.2f} ADX={h1_adx:.1f} RSI14={rsi14:.1f}"
        )

    # --- Conditional loss analysis ---
    print("\n" + "=" * 80)
    print("  CONDITIONAL: LOSSES BY SESSION / DOW / ATR PERCENTILE BUCKET")
    print("=" * 80)
    if len(losers):
        los = losers.copy()
        total_l = len(los)
        for sess in ["Asia_0_8", "London_8_14", "NY_14_21", "Off_21_24"]:
            n = int((los["session"] == sess).sum())
            print(f"  Session {sess}: {n} losers ({100.0 * n / total_l:.1f}% of losers)")

        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        print("\n  Day of week (losers):")
        for d in range(7):
            n = int((los["entry_dow"] == d).sum())
            print(f"    {dow_names[d]}: {n} ({100.0 * n / total_l:.1f}%)")

        ap = los["h1_atr_percentile"].to_numpy(dtype=float)
        high = np.isfinite(ap) & (ap > 0.7)
        low = np.isfinite(ap) & (ap < 0.3)
        mid = np.isfinite(ap) & ~high & ~low
        print(
            f"\n  ATR pct buckets (losers): high>0.7: {int(high.sum())}  "
            f"low<0.3: {int(low.sum())}  normal: {int(mid.sum())}  missing: {int(np.isnan(ap).sum())}"
        )

    # --- Distinctive features summary ---
    print("\n" + "=" * 80)
    print("  SUMMARY — most distinguishing features (by |Cohen's d| & p-value)")
    print("=" * 80)
    ct = cmp_tbl.replace([np.inf, -np.inf], np.nan).dropna(subset=["cohens_d_win_minus_lose"])
    ct = ct.assign(
        abs_d=np.abs(ct["cohens_d_win_minus_lose"]),
        p_combined=ct[["ttest_p", "mannwhitney_p"]].min(axis=1),
    )
    ct = ct.sort_values(["abs_d", "p_combined"], ascending=[False, True])
    top = ct.head(5)
    for _, r in top.iterrows():
        print(
            f"  • {r['feature']}: Cohen's d={r['cohens_d_win_minus_lose']:.3f}  "
            f"win_mean={r['win_mean']:.4g} lose_mean={r['lose_mean']:.4g}  "
            f"min_p={r['p_combined']:.4g}"
        )

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
