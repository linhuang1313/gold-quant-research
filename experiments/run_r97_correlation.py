#!/usr/bin/env python3
"""
R97 — Strategy Correlation & Drawdown Decomposition
=====================================================
1. Run all 4 strategies at R89 lots to get trade lists
2. Convert to daily PnL series
3. Compute pairwise correlation, portfolio decomposition,
   top-10 worst days, conditional analysis, diversification ratio,
   and maximum drawdown overlap periods.
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r97_correlation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs()
    }).max(axis=1)
    return tr.rolling(period).mean()


def add_psar(df, af_start=0.01, af_max=0.05):
    df = df.copy(); n = len(df)
    psar = np.zeros(n); direction = np.ones(n)
    af = af_start; ep = df['High'].iloc[0]; psar[0] = df['Low'].iloc[0]
    for i in range(1, n):
        prev = psar[i-1]
        if direction[i-1] == 1:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], df['Low'].iloc[i-1], df['Low'].iloc[max(0, i-2)])
            if df['Low'].iloc[i] < psar[i]:
                direction[i] = -1; psar[i] = ep; ep = df['Low'].iloc[i]; af = af_start
            else:
                direction[i] = 1
                if df['High'].iloc[i] > ep:
                    ep = df['High'].iloc[i]; af = min(af+af_start, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], df['High'].iloc[i-1], df['High'].iloc[max(0, i-2)])
            if df['High'].iloc[i] > psar[i]:
                direction[i] = 1; psar[i] = ep; ep = df['High'].iloc[i]; af = af_start
            else:
                direction[i] = -1
                if df['Low'].iloc[i] < ep:
                    ep = df['Low'].iloc[i]; af = min(af+af_start, af_max)
    df['PSAR_dir'] = direction; df['ATR'] = compute_atr(df)
    return df


def _mk(pos, exit_price, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_price,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


def _run_exit_with_cap(pos, i, h, lo_v, c, spread, lot, pv, times,
                       sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold, maxloss_cap=0):
    held = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_h = (h - pos['entry'] - spread) * lot * pv
        pnl_l = (lo_v - pos['entry'] - spread) * lot * pv
        pnl_c = (c - pos['entry'] - spread) * lot * pv
    else:
        pnl_h = (pos['entry'] - lo_v - spread) * lot * pv
        pnl_l = (pos['entry'] - h - spread) * lot * pv
        pnl_c = (pos['entry'] - c - spread) * lot * pv
    tp_val = tp_atr * pos['atr'] * lot * pv
    sl_val = sl_atr * pos['atr'] * lot * pv
    if pnl_h >= tp_val: return _mk(pos, c, times[i], "TP", i, tp_val)
    if pnl_l <= -sl_val: return _mk(pos, c, times[i], "SL", i, -sl_val)
    if maxloss_cap > 0 and pnl_c < -maxloss_cap:
        return _mk(pos, c, times[i], "MaxLossCap", i, -maxloss_cap)
    ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
    if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
        ts_p = h - td
        if lo_v <= ts_p: return _mk(pos, c, times[i], "Trail", i, (ts_p - pos['entry'] - spread) * lot * pv)
    elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
        ts_p = lo_v + td
        if h >= ts_p: return _mk(pos, c, times[i], "Trail", i, (pos['entry'] - ts_p - spread) * lot * pv)
    if held >= max_hold: return _mk(pos, c, times[i], "Timeout", i, pnl_c)
    return None


# ═══════════════════════════════════════════════════════════════
# Strategy backtests
# ═══════════════════════════════════════════════════════════════

def bt_psar(h1_df, spread, lot, maxloss_cap=0,
            sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04, max_hold=20):
    df = add_psar(h1_df).dropna(subset=['PSAR_dir', 'ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    pdir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pdir[i-1] == -1 and pdir[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif pdir[i-1] == 1 and pdir[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom(h1_df, spread, lot, maxloss_cap=0,
             fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
             trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    max_lb = max(fast, slow)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        if c[i-fast] > 0: s += 0.5 * np.sign(c[i]/c[i-fast] - 1.0)
        if c[i-slow] > 0: s += 0.5 * np.sign(c[i]/c[i-slow] - 1.0)
        score[i] = s
    trades = []; pos = None; last_exit = -999
    for i in range(max_lb+1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                pnl = (c[i]-pos['entry']-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                pnl = (pos['entry']-c[i]-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(score[i]) or np.isnan(score[i-1]): continue
        if score[i] > 0 and score[i-1] <= 0:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_sess_bo(h1_df, spread, lot, maxloss_cap=0,
               session_hour=12, lookback=4, sl_atr=4.5, tp_atr=4.0,
               trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(lookback, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if hours[i] != session_hour: continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_l8_max(data_bundle, spread, lot, maxloss_cap=0):
    from backtest.runner import run_variant, LIVE_PARITY_KWARGS
    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': maxloss_cap,
          'spread_cost': spread, 'initial_capital': 2000,
          'min_lot_size': lot, 'max_lot_size': lot}
    result = run_variant(data_bundle, "L8_MAX", verbose=False, **kw)
    raw_trades = result.get('_trades', [])
    trades = []
    for t in raw_trades:
        trades.append({
            'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0,
        })
    return trades


# ═══════════════════════════════════════════════════════════════
# Stats helpers
# ═══════════════════════════════════════════════════════════════

def trades_to_daily_series(trades):
    if not trades: return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        ts = pd.Timestamp(t['exit_time'])
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        d = ts.normalize()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))

def sharpe(daily_pnl):
    if len(daily_pnl) < 10: return 0.0
    arr = np.array(daily_pnl) if not isinstance(daily_pnl, np.ndarray) else daily_pnl
    if arr.std() == 0: return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(252))

def max_dd(daily_pnl):
    arr = np.array(daily_pnl) if not isinstance(daily_pnl, np.ndarray) else daily_pnl
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(dd.max()) if len(dd) > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("  R97 — Strategy Correlation & Drawdown Decomposition", flush=True)
    print(f"  R89 lots: {R89_LOTS}", flush=True)
    print("=" * 80, flush=True)

    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH, DataBundle

    print("\n  Loading data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    bundle = DataBundle.load_custom()
    print(f"  H1: {len(h1_df)} bars", flush=True)

    results = {}

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Run all strategies at R89 lots, build daily PnL
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 1: Generate Daily PnL Series", flush=True)
    print("=" * 70, flush=True)

    strat_trades = {}
    strat_daily_unit = {}

    for sname in ['PSAR', 'TSMOM', 'SESS_BO']:
        fn = {'PSAR': bt_psar, 'TSMOM': bt_tsmom, 'SESS_BO': bt_sess_bo}[sname]
        trades = fn(h1_df, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=CAPS[sname])
        strat_trades[sname] = trades
        strat_daily_unit[sname] = trades_to_daily_series(trades)
        n_t = len(trades)
        pnl = sum(t['pnl'] for t in trades)
        print(f"    {sname:>8}: {n_t} trades, unit PnL=${pnl:.0f}", flush=True)

    l8_trades = bt_l8_max(bundle, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])
    strat_trades['L8_MAX'] = l8_trades
    strat_daily_unit['L8_MAX'] = trades_to_daily_series(l8_trades)
    print(f"    {'L8_MAX':>8}: {len(l8_trades)} trades, unit PnL=${sum(t['pnl'] for t in l8_trades):.0f}", flush=True)

    all_dates = set()
    for ds in strat_daily_unit.values():
        all_dates.update(ds.index)
    all_dates = sorted(all_dates)
    idx = pd.DatetimeIndex(all_dates)

    strat_daily_scaled = {}
    for sname in STRAT_ORDER:
        mult = R89_LOTS[sname] / UNIT_LOT
        strat_daily_scaled[sname] = strat_daily_unit[sname].reindex(idx, fill_value=0.0).values * mult

    portfolio_daily = np.zeros(len(idx))
    for sname in STRAT_ORDER:
        portfolio_daily += strat_daily_scaled[sname]

    # ══════════════════════════════════════════════════════════════
    # Phase 2: Pairwise Correlation Matrix
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: Pairwise Correlation Matrix", flush=True)
    print("=" * 70, flush=True)

    df_daily = pd.DataFrame({s: strat_daily_scaled[s] for s in STRAT_ORDER}, index=idx)
    corr_matrix = df_daily.corr()

    corr_dict = {}
    print(f"\n    {'':>10}", end="")
    for s in STRAT_ORDER:
        print(f" {s:>10}", end="")
    print()
    for s1 in STRAT_ORDER:
        print(f"    {s1:>10}", end="")
        corr_dict[s1] = {}
        for s2 in STRAT_ORDER:
            val = corr_matrix.loc[s1, s2]
            corr_dict[s1][s2] = round(float(val), 4)
            print(f" {val:>10.4f}", end="")
        print()

    results['correlation_matrix'] = corr_dict

    avg_corr = []
    for i, s1 in enumerate(STRAT_ORDER):
        for j, s2 in enumerate(STRAT_ORDER):
            if i < j:
                avg_corr.append(corr_matrix.loc[s1, s2])
    results['avg_pairwise_correlation'] = round(float(np.mean(avg_corr)), 4)
    print(f"\n    Average pairwise correlation: {np.mean(avg_corr):.4f}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Phase 3: Top-10 Worst Portfolio Days
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3: Top-10 Worst Portfolio Days", flush=True)
    print("=" * 70, flush=True)

    day_details = []
    for k in range(len(idx)):
        contrib = {s: float(strat_daily_scaled[s][k]) for s in STRAT_ORDER}
        day_details.append({
            'date': str(idx[k].date()),
            'portfolio_pnl': round(float(portfolio_daily[k]), 2),
            'contributions': {s: round(v, 2) for s, v in contrib.items()},
        })

    day_details.sort(key=lambda x: x['portfolio_pnl'])
    top10_worst = day_details[:10]

    print(f"\n    {'Date':>12} {'PortPnL':>10}", end="")
    for s in STRAT_ORDER:
        print(f" {s:>10}", end="")
    print()
    for d in top10_worst:
        print(f"    {d['date']:>12} ${d['portfolio_pnl']:>9.2f}", end="")
        for s in STRAT_ORDER:
            v = d['contributions'][s]
            print(f" ${v:>9.2f}", end="")
        print()

    results['top10_worst_days'] = top10_worst

    # ══════════════════════════════════════════════════════════════
    # Phase 4: Conditional Analysis
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 4: Conditional Analysis (losing day -> others)", flush=True)
    print("=" * 70, flush=True)

    conditional = {}
    for s1 in STRAT_ORDER:
        losing_mask = strat_daily_scaled[s1] < 0
        n_losing = int(losing_mask.sum())
        if n_losing == 0:
            conditional[s1] = {'n_losing_days': 0, 'others_avg': {}}
            continue
        others_avg = {}
        for s2 in STRAT_ORDER:
            if s2 == s1:
                continue
            avg_pnl_on_bad_days = float(np.mean(strat_daily_scaled[s2][losing_mask]))
            others_avg[s2] = round(avg_pnl_on_bad_days, 4)
        conditional[s1] = {
            'n_losing_days': n_losing,
            'avg_own_loss': round(float(np.mean(strat_daily_scaled[s1][losing_mask])), 4),
            'others_avg': others_avg,
        }
        print(f"    When {s1:>8} loses ({n_losing} days, avg=${conditional[s1]['avg_own_loss']:.2f}):", flush=True)
        for s2, avg in others_avg.items():
            sign = "+" if avg >= 0 else ""
            print(f"      {s2:>8} avg = {sign}${avg:.4f}", flush=True)

    results['conditional_analysis'] = conditional

    # ══════════════════════════════════════════════════════════════
    # Phase 5: Diversification Ratio
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 5: Diversification Ratio", flush=True)
    print("=" * 70, flush=True)

    port_sharpe = sharpe(portfolio_daily)
    weighted_sharpe_sum = 0.0
    total_weight = 0.0
    individual_sharpes = {}

    for sname in STRAT_ORDER:
        ind_sharpe = sharpe(strat_daily_scaled[sname])
        individual_sharpes[sname] = round(ind_sharpe, 3)
        weight = R89_LOTS[sname]
        weighted_sharpe_sum += weight * ind_sharpe
        total_weight += weight

    weighted_avg_sharpe = weighted_sharpe_sum / total_weight if total_weight > 0 else 0
    div_ratio = port_sharpe / weighted_avg_sharpe if weighted_avg_sharpe != 0 else 0

    print(f"    Portfolio Sharpe:          {port_sharpe:.3f}", flush=True)
    print(f"    Weighted avg ind. Sharpe:  {weighted_avg_sharpe:.3f}", flush=True)
    print(f"    Diversification ratio:     {div_ratio:.3f}", flush=True)
    for sname in STRAT_ORDER:
        print(f"      {sname:>8}: Sharpe={individual_sharpes[sname]:.3f} (lot={R89_LOTS[sname]})", flush=True)

    results['diversification'] = {
        'portfolio_sharpe': round(port_sharpe, 3),
        'weighted_avg_individual_sharpe': round(weighted_avg_sharpe, 3),
        'diversification_ratio': round(div_ratio, 3),
        'individual_sharpes': individual_sharpes,
        'lots': R89_LOTS,
    }

    # ══════════════════════════════════════════════════════════════
    # Phase 6: Maximum Drawdown Overlap
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 6: Drawdown Overlap Analysis", flush=True)
    print("=" * 70, flush=True)

    dd_flags = {}
    for sname in STRAT_ORDER:
        cum = np.cumsum(strat_daily_scaled[sname])
        peak = np.maximum.accumulate(cum)
        in_dd = (peak - cum) > 0
        dd_flags[sname] = in_dd

    overlap_counts = {}
    n_days = len(idx)
    for i, s1 in enumerate(STRAT_ORDER):
        for j, s2 in enumerate(STRAT_ORDER):
            if i >= j:
                continue
            both_dd = dd_flags[s1] & dd_flags[s2]
            overlap_counts[f"{s1}+{s2}"] = int(both_dd.sum())

    any_2_plus = np.zeros(n_days, dtype=bool)
    any_3_plus = np.zeros(n_days, dtype=bool)
    any_4 = np.zeros(n_days, dtype=bool)
    for k in range(n_days):
        n_in_dd = sum(dd_flags[s][k] for s in STRAT_ORDER)
        if n_in_dd >= 2: any_2_plus[k] = True
        if n_in_dd >= 3: any_3_plus[k] = True
        if n_in_dd >= 4: any_4[k] = True

    def count_periods(mask):
        periods = 0
        in_period = False
        for v in mask:
            if v and not in_period:
                periods += 1
                in_period = True
            elif not v:
                in_period = False
        return periods

    def longest_streak(mask):
        max_streak = 0; current = 0
        for v in mask:
            if v:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    dd_overlap = {
        'pairwise_overlap_days': overlap_counts,
        'days_2plus_in_dd': int(any_2_plus.sum()),
        'days_3plus_in_dd': int(any_3_plus.sum()),
        'days_all4_in_dd': int(any_4.sum()),
        'pct_2plus': round(float(any_2_plus.sum()) / n_days * 100, 1),
        'pct_3plus': round(float(any_3_plus.sum()) / n_days * 100, 1),
        'pct_all4': round(float(any_4.sum()) / n_days * 100, 1),
        'n_periods_2plus': count_periods(any_2_plus),
        'longest_streak_2plus': longest_streak(any_2_plus),
        'longest_streak_all4': longest_streak(any_4),
        'total_trading_days': n_days,
    }

    print(f"    Total trading days: {n_days}", flush=True)
    print(f"    Days with 2+ strats in DD: {dd_overlap['days_2plus_in_dd']} ({dd_overlap['pct_2plus']:.1f}%)", flush=True)
    print(f"    Days with 3+ strats in DD: {dd_overlap['days_3plus_in_dd']} ({dd_overlap['pct_3plus']:.1f}%)", flush=True)
    print(f"    Days with ALL 4 in DD:     {dd_overlap['days_all4_in_dd']} ({dd_overlap['pct_all4']:.1f}%)", flush=True)
    print(f"    Longest streak 2+ in DD:   {dd_overlap['longest_streak_2plus']} days", flush=True)
    print(f"    Longest streak ALL4 in DD: {dd_overlap['longest_streak_all4']} days", flush=True)
    print(f"\n    Pairwise overlap:", flush=True)
    for pair, count in sorted(overlap_counts.items()):
        print(f"      {pair:>20}: {count} days ({count/n_days*100:.1f}%)", flush=True)

    results['drawdown_overlap'] = dd_overlap

    # ══════════════════════════════════════════════════════════════
    # Phase 7: Portfolio summary stats
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 7: Portfolio Summary", flush=True)
    print("=" * 70, flush=True)

    port_pnl = float(np.sum(portfolio_daily))
    port_dd = max_dd(portfolio_daily)
    port_wr = float(np.sum(portfolio_daily > 0)) / n_days * 100

    results['portfolio_summary'] = {
        'sharpe': round(port_sharpe, 3),
        'total_pnl': round(port_pnl, 2),
        'max_dd': round(port_dd, 2),
        'daily_win_rate': round(port_wr, 1),
        'mean_daily_pnl': round(float(np.mean(portfolio_daily)), 2),
        'std_daily_pnl': round(float(np.std(portfolio_daily)), 2),
        'best_day': round(float(np.max(portfolio_daily)), 2),
        'worst_day': round(float(np.min(portfolio_daily)), 2),
    }

    print(f"    Sharpe:         {port_sharpe:.3f}", flush=True)
    print(f"    Total PnL:      ${port_pnl:.0f}", flush=True)
    print(f"    Max DD:         ${port_dd:.0f}", flush=True)
    print(f"    Daily WR:       {port_wr:.1f}%", flush=True)
    print(f"    Best day:       ${np.max(portfolio_daily):.2f}", flush=True)
    print(f"    Worst day:      ${np.min(portfolio_daily):.2f}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Save results
    # ══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)

    print(f"\n{'='*80}", flush=True)
    print(f"  R97 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"{'='*80}", flush=True)

    with open(OUTPUT_DIR / "r97_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r97_results.json", flush=True)


if __name__ == "__main__":
    main()
