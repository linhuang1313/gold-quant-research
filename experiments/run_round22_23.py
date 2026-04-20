"""
R22: EUR/USD Keltner expansion — full validation
R23: L7 + S1 + S3 portfolio analysis (Gold only, no engine dependency)
=====================================================================
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments'))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

OUT_DIR = Path("results/round22_23_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            try:
                f.write(data)
            except UnicodeEncodeError:
                f.write(data.encode('ascii', errors='replace').decode('ascii'))
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


# ═══════════════════════════════════════════════════════════════
# R22: EUR/USD
# ═══════════════════════════════════════════════════════════════

def download_eurusd_yfinance():
    """Download EUR/USD H1 data via yfinance (max 730 days per chunk)."""
    import yfinance as yf

    data_dir = Path("data/download")
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "eurusd-h1-yfinance-2019-2026.csv"

    if out_path.exists():
        df = pd.read_csv(str(out_path))
        print(f"  EUR/USD H1 already exists: {out_path} ({len(df)} bars)")
        return out_path

    print("  Downloading EUR/USD H1 from yfinance...")
    ticker = yf.Ticker("EURUSD=X")
    chunks = []
    periods = [
        ("2019-01-01", "2020-12-31"),
        ("2021-01-01", "2022-12-31"),
        ("2023-01-01", "2024-12-31"),
        ("2025-01-01", "2026-04-20"),
    ]
    for start, end in periods:
        print(f"    {start} -> {end}...", end='', flush=True)
        try:
            df = ticker.history(start=start, end=end, interval="1h")
            if len(df) > 0:
                chunks.append(df)
                print(f" {len(df)} bars")
            else:
                print(" 0 bars")
        except Exception as e:
            print(f" ERROR: {e}")
        time.sleep(1)

    if not chunks:
        print("  FATAL: No data downloaded!")
        return None

    combined = pd.concat(chunks)
    combined = combined[~combined.index.duplicated(keep='first')]
    combined.sort_index(inplace=True)

    out_df = pd.DataFrame({
        'timestamp': (combined.index.astype(np.int64) // 10**6).astype(int),
        'open': combined['Open'].values,
        'high': combined['High'].values,
        'low': combined['Low'].values,
        'close': combined['Close'].values,
        'volume': combined['Volume'].values if 'Volume' in combined.columns else 0,
    })
    out_df.to_csv(str(out_path), index=False)
    print(f"  Saved: {out_path} ({len(out_df)} bars, {combined.index[0]} -> {combined.index[-1]})")
    return out_path


def run_r22(out):
    """EUR/USD full validation using existing ForexBacktestEngine."""
    print("\n" + "=" * 80)
    print("R22: EUR/USD KELTNER EXPANSION — FULL VALIDATION")
    print("=" * 80)

    # Download data
    csv_path = download_eurusd_yfinance()
    if csv_path is None:
        print("  SKIPPING R22: no data")
        return None

    from run_eurusd_research import (
        load_forex_csv, prepare_forex_indicators, ForexBacktestEngine,
        calc_forex_stats, V3_REGIME
    )

    h1_raw = load_forex_csv(csv_path)
    print(f"  H1: {len(h1_raw)} bars, {h1_raw.index[0]} -> {h1_raw.index[-1]}")

    # ── Phase 1: Parameter Grid ──
    print(f"\n--- Phase 1: Parameter Grid ---")

    grid_results = []
    for kc_mult in [1.2, 1.5, 2.0]:
        h1_df = prepare_forex_indicators(h1_raw, kc_mult=kc_mult)
        for adx_th in [18, 22, 25]:
            for trail_a, trail_d in [(0.8, 0.25), (0.6, 0.15), (0.4, 0.10)]:
                for mh in [10, 15, 20]:
                    label = f"KC{kc_mult}_ADX{adx_th}_T{trail_a}/{trail_d}_MH{mh}"
                    engine = ForexBacktestEngine(
                        h1_df, label=label,
                        kc_adx_threshold=adx_th, sl_atr_mult=4.5, tp_atr_mult=8.0,
                        trail_activate_atr=trail_a, trail_distance_atr=trail_d,
                        regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=mh,
                    )
                    trades = engine.run()
                    stats = calc_forex_stats(trades, engine.equity_curve)
                    stats['label'] = label
                    stats['_trades'] = trades
                    stats['kc_mult'] = kc_mult
                    stats['adx_th'] = adx_th

                    if stats['n'] >= 20:
                        print(f"  {label}: N={stats['n']}, Sharpe={stats['sharpe']:.2f}, "
                              f"PnL=${stats['total_pnl']:.0f}, WR={stats['win_rate']:.1f}%")
                        grid_results.append(stats)

    if not grid_results:
        print("  No valid results!")
        return None

    # Top 10
    grid_results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Top 10:")
    for r in grid_results[:10]:
        print(f"    {r['label']}: Sharpe={r['sharpe']:.2f}, N={r['n']}, "
              f"PnL=${r['total_pnl']:.0f}, WR={r['win_rate']:.1f}%")

    best = grid_results[0]
    print(f"\n  BEST: {best['label']} Sharpe={best['sharpe']:.2f}")

    # ── Phase 2: Year-by-Year of best ──
    print(f"\n--- Phase 2: Year-by-Year ({best['label']}) ---")

    best_trades = best.get('_trades', [])
    yearly = {}
    for t in best_trades:
        y = t.exit_time.year
        yearly.setdefault(y, {'n': 0, 'pnl': 0, 'wins': 0})
        yearly[y]['n'] += 1
        yearly[y]['pnl'] += t.pnl
        if t.pnl > 0:
            yearly[y]['wins'] += 1

    positive_years = 0
    for y in sorted(yearly):
        v = yearly[y]
        wr = v['wins'] / v['n'] * 100 if v['n'] > 0 else 0
        marker = "+" if v['pnl'] > 0 else "-"
        if v['pnl'] > 0:
            positive_years += 1
        print(f"  {y}: N={v['n']}, PnL=${v['pnl']:.0f}, WR={wr:.1f}% {marker}")
    print(f"  Positive years: {positive_years}/{len(yearly)}")

    # ── Phase 3: K-Fold 6×2yr of best params ──
    print(f"\n--- Phase 3: K-Fold (best params) ---")

    # Extract params from best label
    best_kc_mult = best.get('kc_mult', 2.0)
    years = sorted(h1_raw.index.year.unique())
    folds = []
    fold_num = 1
    y = min(years)
    while y <= max(years) and fold_num <= 6:
        y_end = y + 2
        start = pd.Timestamp(f"{y}-01-01", tz='UTC')
        end = pd.Timestamp(f"{y_end}-01-01", tz='UTC')
        fold_df = h1_raw[(h1_raw.index >= start) & (h1_raw.index < end)]
        if len(fold_df) > 500:
            folds.append((f"Fold{fold_num}({y}-{y_end-1})", fold_df))
            fold_num += 1
        y = y_end

    kfold_results = []
    for fname, fdf in folds:
        fdf_prep = prepare_forex_indicators(fdf, kc_mult=best_kc_mult)
        engine = ForexBacktestEngine(
            fdf_prep, label=f"KF_{fname}",
            kc_adx_threshold=best.get('adx_th', 18),
            sl_atr_mult=4.5, tp_atr_mult=8.0,
            trail_activate_atr=0.8, trail_distance_atr=0.25,
            regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=15,
        )
        trades = engine.run()
        stats = calc_forex_stats(trades, engine.equity_curve)
        stats['fold'] = fname
        stats['label'] = fname
        kfold_results.append(stats)
        print(f"  {fname}: N={stats['n']}, Sharpe={stats['sharpe']:.2f}, "
              f"PnL=${stats['total_pnl']:.0f}, WR={stats['win_rate']:.1f}%")

    positive_folds = sum(1 for r in kfold_results if r['total_pnl'] > 0)
    kf_sharpes = [r['sharpe'] for r in kfold_results]
    print(f"\n  K-Fold: {positive_folds}/{len(kfold_results)} positive folds")
    if kf_sharpes:
        print(f"  Sharpe: mean={np.mean(kf_sharpes):.2f}, min={min(kf_sharpes):.2f}, max={max(kf_sharpes):.2f}")

    # ── Phase 4: Monte Carlo ──
    print(f"\n--- Phase 4: Monte Carlo (100 reshuffles) ---")
    if best_trades:
        pnl_list = [t.pnl for t in best_trades]
        mc_sharpes = []
        for _ in range(100):
            shuffled = np.random.permutation(pnl_list)
            equity = np.cumsum(shuffled)
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity).max()
            s = np.mean(shuffled) / np.std(shuffled, ddof=1) * np.sqrt(252) if np.std(shuffled, ddof=1) > 0 else 0
            mc_sharpes.append(s)

        mc_sharpes = np.array(mc_sharpes)
        pct_positive = (mc_sharpes > 0).mean() * 100
        print(f"  MC Sharpe: mean={mc_sharpes.mean():.2f}, P5={np.percentile(mc_sharpes, 5):.2f}, "
              f"P95={np.percentile(mc_sharpes, 95):.2f}")
        print(f"  Positive: {pct_positive:.0f}%")

    return {
        'best_label': best['label'],
        'best_sharpe': best['sharpe'],
        'best_pnl': best['total_pnl'],
        'kfold_positive': positive_folds,
        'kfold_total': len(kfold_results),
        'positive_years': positive_years,
        'total_years': len(yearly),
    }


# ═══════════════════════════════════════════════════════════════
# R23: L7 + S1 + S3 Portfolio (all Gold, no engine needed)
# ═══════════════════════════════════════════════════════════════

SPREAD_GOLD = 0.30

def _run_s1(h1_df):
    """S1 Squeeze Straddle: SqzB3, Trail 0.2/0.04, MH20."""
    atr = h1_df['ATR'].values
    close = h1_df['Close'].values
    high = h1_df['High'].values
    low = h1_df['Low'].values
    squeeze = h1_df['squeeze'].values if 'squeeze' in h1_df.columns else np.zeros(len(h1_df))
    times = h1_df.index

    trades = []
    squeeze_count = 0
    in_trade = False
    buy_pos = sell_pos = None

    for i in range(50, len(h1_df) - 1):
        cur_atr = atr[i]
        if cur_atr <= 0:
            continue

        if squeeze[i] == 1:
            squeeze_count += 1
        else:
            if squeeze_count >= 3 and not in_trade:
                ep = close[i]
                sl_d = 1.5 * cur_atr
                buy_pos = {'ep': ep, 'sl': ep - sl_d, 'trail': 0, 'ext': ep, 'bars': 0, 'closed': False, 'et': times[i]}
                sell_pos = {'ep': ep, 'sl': ep + sl_d, 'trail': 999999, 'ext': ep, 'bars': 0, 'closed': False, 'et': times[i]}
                in_trade = True
            squeeze_count = 0

        if not in_trade:
            continue

        for pos, d in [(buy_pos, 'BUY'), (sell_pos, 'SELL')]:
            if pos is None or pos['closed']:
                continue
            pos['bars'] += 1
            h, l, c, a = high[i], low[i], close[i], cur_atr
            xp = None

            if d == 'BUY':
                if l <= pos['sl']:
                    xp = pos['sl']
                else:
                    pos['ext'] = max(pos['ext'], h)
                    if h - pos['ep'] >= a * 0.2:
                        t = pos['ext'] - a * 0.04
                        pos['trail'] = max(pos['trail'], t)
                        if l <= pos['trail']:
                            xp = pos['trail']
                    if pos['bars'] >= 20 and xp is None:
                        xp = c
            else:
                if h >= pos['sl']:
                    xp = pos['sl']
                else:
                    pos['ext'] = min(pos['ext'], l)
                    if pos['ep'] - l >= a * 0.2:
                        t = pos['ext'] + a * 0.04
                        pos['trail'] = min(pos['trail'], t)
                        if h >= pos['trail']:
                            xp = pos['trail']
                    if pos['bars'] >= 20 and xp is None:
                        xp = c

            if xp is not None:
                pnl = (xp - pos['ep'] - SPREAD_GOLD) if d == 'BUY' else (pos['ep'] - xp - SPREAD_GOLD)
                trades.append({'exit_time': times[i], 'pnl': pnl, 'direction': d, 'entry_time': pos['et']})
                pos['closed'] = True

        if buy_pos and buy_pos['closed'] and sell_pos and sell_pos['closed']:
            in_trade = False
            buy_pos = sell_pos = None

    return trades


def _run_s3(h1_df):
    """S3 Overnight Hold: BUY at 21 UTC, close at 7 UTC."""
    trades = []
    dates_seen = set()
    hours = h1_df.index.hour
    dates = h1_df.index.date
    close_vals = h1_df['Close'].values

    for i in range(1, len(h1_df)):
        h = hours[i]
        d = dates[i]
        if h == 21 and d not in dates_seen:
            ep = close_vals[i]
            et = h1_df.index[i]
            for j in range(i + 1, min(i + 30, len(h1_df))):
                if hours[j] == 7:
                    xp = close_vals[j]
                    pnl = xp - ep - SPREAD_GOLD
                    trades.append({'exit_time': h1_df.index[j], 'pnl': pnl, 'entry_time': et})
                    dates_seen.add(d)
                    break
    return trades


def run_r23():
    """Portfolio analysis: L7 + S1 + S3."""
    print("\n" + "=" * 80)
    print("R23: L7 + S1 + S3 PORTFOLIO ANALYSIS")
    print("=" * 80)

    from backtest.runner import DataBundle
    data = DataBundle.load_default()
    h1_df = data.h1_df

    # S1
    print("\n--- Running S1 (Squeeze Straddle) ---")
    s1_trades = _run_s1(h1_df)
    print(f"  S1: {len(s1_trades)} trades, PnL=${sum(t['pnl'] for t in s1_trades):.0f}")

    # S3
    print("\n--- Running S3 (Overnight Hold) ---")
    s3_trades = _run_s3(h1_df)
    print(f"  S3: {len(s3_trades)} trades, PnL=${sum(t['pnl'] for t in s3_trades):.0f}")

    # L7 — use saved trades from run_round21_phase3 if available, otherwise run
    print("\n--- Running L7 Baseline ---")
    from backtest.runner import run_variant, LIVE_PARITY_KWARGS
    l7_kwargs = {**LIVE_PARITY_KWARGS}
    l7_kwargs['time_adaptive_trail'] = {'start': 2, 'decay': 0.75, 'floor': 0.003}
    l7_kwargs['min_entry_gap_hours'] = 1.0

    l7_stats = run_variant(data, "L7", **l7_kwargs)
    l7_trades = [{'exit_time': t.exit_time, 'pnl': t.pnl, 'entry_time': t.entry_time}
                 for t in l7_stats['_trades']]
    print(f"  L7: {len(l7_trades)} trades, PnL=${sum(t['pnl'] for t in l7_trades):.0f}")

    # ── Build daily PnL series ──
    def to_daily(trades, name):
        data = []
        for t in trades:
            d = pd.Timestamp(t['exit_time']).date()
            data.append({'date': d, 'pnl': t['pnl']})
        if not data:
            return pd.Series(dtype=float, name=name)
        df = pd.DataFrame(data)
        return df.groupby('date')['pnl'].sum().rename(name)

    l7_daily = to_daily(l7_trades, 'L7')
    s1_daily = to_daily(s1_trades, 'S1')
    s3_daily = to_daily(s3_trades, 'S3')

    all_dates = sorted(set(l7_daily.index) | set(s1_daily.index) | set(s3_daily.index))
    port = pd.DataFrame(index=all_dates)
    port['L7'] = l7_daily.reindex(all_dates).fillna(0)
    port['S1'] = s1_daily.reindex(all_dates).fillna(0)
    port['S3'] = s3_daily.reindex(all_dates).fillna(0)
    port['L7+S1'] = port['L7'] + port['S1']
    port['L7+S3'] = port['L7'] + port['S3']
    port['L7+S1+S3'] = port['L7'] + port['S1'] + port['S3']

    # ── Correlation ──
    print("\n--- Daily PnL Correlation ---")
    corr = port[['L7', 'S1', 'S3']].corr()
    print(corr.to_string())

    # ── Portfolio stats ──
    print("\n--- Portfolio Stats ---")
    print(f"  {'Portfolio':<14s} {'PnL':>10s} {'Sharpe':>8s} {'MaxDD':>8s} {'WinDays':>8s}")
    print(f"  {'-'*50}")
    for col in ['L7', 'S1', 'S3', 'L7+S1', 'L7+S3', 'L7+S1+S3']:
        s = port[col]
        total = s.sum()
        sharpe = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0
        eq = s.cumsum()
        peak = eq.cummax()
        dd = (peak - eq).max()
        win_days = (s > 0).mean()
        print(f"  {col:<14s} ${total:>9.0f} {sharpe:>7.2f} ${dd:>7.0f} {win_days:>7.1%}")

    # ── Year-by-year combined ──
    print("\n--- Year-by-Year: L7 + S1 + S3 ---")
    port['year'] = pd.to_datetime(port.index).year
    print(f"  {'Year':<6s} {'Total':>8s} {'L7':>8s} {'S1':>8s} {'S3':>8s}")
    print(f"  {'-'*40}")
    for y, grp in port.groupby('year'):
        t = grp['L7+S1+S3'].sum()
        l = grp['L7'].sum()
        s1 = grp['S1'].sum()
        s3 = grp['S3'].sum()
        print(f"  {y:<6d} ${t:>7.0f} ${l:>7.0f} ${s1:>7.0f} ${s3:>7.0f}")

    # ── Diversification ratio ──
    print("\n--- Diversification Benefit ---")
    l7_sharpe = port['L7'].mean() / port['L7'].std() * np.sqrt(252) if port['L7'].std() > 0 else 0
    combo_sharpe = port['L7+S1+S3'].mean() / port['L7+S1+S3'].std() * np.sqrt(252) if port['L7+S1+S3'].std() > 0 else 0
    l7_dd = (port['L7'].cumsum().cummax() - port['L7'].cumsum()).max()
    combo_dd = (port['L7+S1+S3'].cumsum().cummax() - port['L7+S1+S3'].cumsum()).max()

    print(f"  L7 alone:  Sharpe={l7_sharpe:.2f}, MaxDD=${l7_dd:.0f}")
    print(f"  L7+S1+S3:  Sharpe={combo_sharpe:.2f}, MaxDD=${combo_dd:.0f}")
    print(f"  Sharpe uplift: +{combo_sharpe - l7_sharpe:.2f}")
    print(f"  MaxDD change:  ${combo_dd - l7_dd:+.0f}")

    # ── Worst periods ──
    print("\n--- Worst Periods (L7+S1+S3) ---")
    port['combo_eq'] = port['L7+S1+S3'].cumsum()
    port['combo_peak'] = port['combo_eq'].cummax()
    port['combo_dd'] = port['combo_peak'] - port['combo_eq']

    # Rolling 30-day return
    port['roll30'] = port['L7+S1+S3'].rolling(30).sum()
    worst_30 = port['roll30'].idxmin()
    if worst_30 is not None:
        print(f"  Worst 30-day window ending {worst_30}: ${port.loc[worst_30, 'roll30']:.0f}")

    port['roll90'] = port['L7+S1+S3'].rolling(90).sum()
    worst_90 = port['roll90'].idxmin()
    if worst_90 is not None:
        print(f"  Worst 90-day window ending {worst_90}: ${port.loc[worst_90, 'roll90']:.0f}")


def main():
    t_start = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out_path = OUT_DIR / "R22_R23_output.txt"
    out = open(out_path, 'w', encoding='utf-8')

    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, out)

    print(f"# R22 + R23: EUR/USD Expansion & Portfolio Analysis")
    print(f"# Started: {ts}")

    # R22
    r22_summary = run_r22(out)

    # R23
    run_r23()

    elapsed = time.time() - t_start
    print(f"\n# Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Elapsed: {elapsed/60:.1f} minutes")

    sys.stdout = old_stdout
    out.close()
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
