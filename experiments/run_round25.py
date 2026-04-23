"""
R25: MaxHold Optimization + Dynamic Sizing + H4/Daily Keltner
==============================================================
Phase A: R24-B MaxHold Sweep 补完 (Timeout -$23K 是 L7 最大出血点)
Phase B: R24-C Dynamic Position Sizing
Phase C: H4/Daily Keltner — 全新时间尺度 alpha
"""

import sys, os, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from backtest.runner import (
    DataBundle, run_variant, run_kfold, LIVE_PARITY_KWARGS
)
from backtest.engine import BacktestEngine

OUT_DIR = Path("results/round25_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            try: f.write(data)
            except UnicodeEncodeError: f.write(data.encode('ascii', errors='replace').decode('ascii'))
            f.flush()
    def flush(self):
        for f in self.files: f.flush()


L7_KWARGS = {
    **LIVE_PARITY_KWARGS,
    'time_adaptive_trail': {'start': 2, 'decay': 0.75, 'floor': 0.003},
    'min_entry_gap_hours': 1.0,
}


def ps(stats, prefix=""):
    print(f"  {prefix}{stats.get('label','')}: N={stats['n']}, Sharpe={stats['sharpe']:.2f}, "
          f"PnL=${stats['total_pnl']:.0f}, WR={stats['win_rate']:.1f}%, "
          f"MaxDD=${stats['max_dd']:.0f}")


def kfs(results, label=""):
    sharpes = [r['sharpe'] for r in results]
    pos = sum(1 for s in sharpes if s > 0)
    print(f"\n  K-Fold [{label}]: {pos}/{len(results)} positive, "
          f"mean={np.mean(sharpes):.2f}, min={min(sharpes):.2f}, max={max(sharpes):.2f}")
    return pos


# ═══════════════════════════════════════════════════════════════
# Phase A: MaxHold Optimization
# ═══════════════════════════════════════════════════════════════

def run_phase_A(data):
    print("\n" + "=" * 80)
    print("Phase A: MaxHold Optimization (L7 Timeout = -$23K, 70% of losses)")
    print("=" * 80)

    # Baseline
    l7 = run_variant(data, "L7_MH20_baseline", **L7_KWARGS)
    ps(l7)

    # Exit profile
    trades = l7['_trades']
    exit_map = {}
    for t in trades:
        r = t.exit_reason
        exit_map.setdefault(r, {'n': 0, 'pnl': 0, 'wins': 0})
        exit_map[r]['n'] += 1; exit_map[r]['pnl'] += t.pnl
        if t.pnl > 0: exit_map[r]['wins'] += 1

    print(f"\n  Exit Profile:")
    for r in sorted(exit_map, key=lambda x: exit_map[x]['pnl'], reverse=True):
        v = exit_map[r]
        wr = v['wins']/v['n']*100 if v['n'] > 0 else 0
        print(f"    {r:<20s} N={v['n']:>6} PnL=${v['pnl']:>9.0f} WR={wr:.1f}%")

    # Fine MaxHold sweep
    print(f"\n  --- MaxHold Sweep (fine grid) ---")
    print(f"  {'MH':>4} {'N':>6} {'Sharpe':>8} {'PnL':>10} {'WR':>6} {'MaxDD':>8} {'TimeoutN':>9} {'TimeoutPnL':>11}")
    print(f"  {'-'*72}")

    sweep_results = []
    for mh in [8, 10, 12, 14, 15, 16, 18, 20, 22, 25, 30, 40, 50, 999]:
        kw = copy.deepcopy(L7_KWARGS)
        kw['keltner_max_hold_m15'] = mh
        s = run_variant(data, f"MH{mh}", verbose=False, **kw)
        trs = s['_trades']
        to = [t for t in trs if 'timeout' in t.exit_reason.lower() or t.exit_reason == 'Timeout']
        to_n = len(to); to_pnl = sum(t.pnl for t in to)
        print(f"  {mh:>4} {s['n']:>6} {s['sharpe']:>8.2f} ${s['total_pnl']:>9.0f} "
              f"{s['win_rate']:>5.1f}% ${s['max_dd']:>7.0f} {to_n:>9} ${to_pnl:>10.0f}")
        sweep_results.append((mh, s))

    # Identify best
    best_mh, best_s = max(sweep_results, key=lambda x: x[1]['sharpe'])
    print(f"\n  >>> Best MaxHold: MH={best_mh}, Sharpe={best_s['sharpe']:.2f}, PnL=${best_s['total_pnl']:.0f}")

    # K-Fold on top 3
    top3 = sorted(sweep_results, key=lambda x: x[1]['sharpe'], reverse=True)[:3]
    for mh, _ in top3:
        kw = copy.deepcopy(L7_KWARGS)
        kw['keltner_max_hold_m15'] = mh
        print(f"\n  --- K-Fold: MH={mh} ---")
        kf = run_kfold(data, kw, label_prefix=f"MH{mh}")
        for r in kf: ps(r, "  ")
        kfs(kf, f"MH{mh}")

    # Also K-Fold baseline MH=20 for comparison
    print(f"\n  --- K-Fold: MH=20 (baseline) ---")
    kf_base = run_kfold(data, L7_KWARGS, label_prefix="MH20")
    for r in kf_base: ps(r, "  ")
    kfs(kf_base, "MH20_baseline")


# ═══════════════════════════════════════════════════════════════
# Phase B: Dynamic Position Sizing
# ═══════════════════════════════════════════════════════════════

def run_phase_B(data):
    print("\n" + "=" * 80)
    print("Phase B: Dynamic Position Sizing (Post-hoc)")
    print("=" * 80)

    l7 = run_variant(data, "L7_sizing", **L7_KWARGS)
    trades = l7['_trades']
    pnl_list = [t.pnl for t in trades]

    daily = {}
    for t in trades:
        d = pd.Timestamp(t.exit_time).date()
        daily.setdefault(d, 0); daily[d] += t.pnl
    d_arr = np.array(list(daily.values()))
    base_sh = d_arr.mean() / d_arr.std() * np.sqrt(252) if d_arr.std() > 0 else 0
    base_eq = np.cumsum(pnl_list)
    base_dd = (np.maximum.accumulate(base_eq) - base_eq).max()

    print(f"\n  Baseline: N={len(trades)}, PnL=${sum(pnl_list):.0f}, Sharpe={base_sh:.2f}, MaxDD=${base_dd:.0f}")

    def eval_sizing(scaled_pnl, label):
        sp = np.array(scaled_pnl)
        eq = np.cumsum(sp); dd = (np.maximum.accumulate(eq) - eq).max()
        dp = {}
        for i, t in enumerate(trades):
            d = pd.Timestamp(t.exit_time).date()
            dp.setdefault(d, 0); dp[d] += sp[i]
        da = np.array(list(dp.values()))
        sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0
        delta = sh - base_sh
        print(f"  {label:<45s} PnL=${sp.sum():>8.0f} Sharpe={sh:>6.2f} ({delta:+.2f}) DD=${dd:>7.0f}")
        return sh

    # 1. Streak-based
    print(f"\n  --- Streak Sizing ---")
    for thr, wm, lm in [(3, 1.5, 0.5), (3, 2.0, 0.5), (3, 1.5, 0.75),
                         (5, 1.5, 0.5), (5, 2.0, 0.5), (2, 1.5, 0.5)]:
        sp = []; streak = 0
        for pnl in pnl_list:
            mult = wm if streak >= thr else (lm if streak <= -thr else 1.0)
            sp.append(pnl * mult)
            streak = (streak + 1 if streak > 0 else 1) if pnl > 0 else ((streak - 1 if streak < 0 else -1) if pnl < 0 else 0)
        eval_sizing(sp, f"Streak{thr} W{wm}x/L{lm}x")

    # 2. Regime-based (ATR percentile)
    print(f"\n  --- Regime Sizing ---")
    h1_df = data.h1_df
    for hi_t, hi_m, lo_t, lo_m in [(0.70, 1.5, 0.30, 0.5), (0.70, 1.5, 0.30, 0.75),
                                     (0.70, 2.0, 0.30, 0.5), (0.80, 1.5, 0.20, 0.5)]:
        sp = []
        for i, t in enumerate(trades):
            et = pd.Timestamp(t.entry_time)
            mask = h1_df.index <= et
            atr_pct = h1_df.loc[h1_df.index[mask][-1], 'atr_percentile'] if mask.any() and 'atr_percentile' in h1_df.columns else 0.5
            mult = hi_m if atr_pct > hi_t else (lo_m if atr_pct < lo_t else 1.0)
            sp.append(pnl_list[i] * mult)
        eval_sizing(sp, f"Regime H>{hi_t:.0%}={hi_m}x L<{lo_t:.0%}={lo_m}x")

    # 3. Equity curve
    print(f"\n  --- Equity Curve Mgmt ---")
    for lb, cut, red in [(30, 0, 0.5), (50, 0, 0.5), (50, -5, 0.5),
                          (100, 0, 0.5), (50, 0, 0.3), (50, 5, 0.5)]:
        sp = []; recent = []
        for pnl in pnl_list:
            recent.append(pnl)
            if len(recent) > lb: recent.pop(0)
            mult = red if len(recent) >= lb and np.mean(recent) < cut else 1.0
            sp.append(pnl * mult)
        eval_sizing(sp, f"EqCurve LB={lb} Cut=${cut} Red={red}x")

    # 4. Kelly
    print(f"\n  --- Kelly Criterion ---")
    wins = [p for p in pnl_list if p > 0]
    losses = [abs(p) for p in pnl_list if p < 0]
    if wins and losses:
        wr = len(wins)/len(pnl_list); aw = np.mean(wins); al = np.mean(losses)
        kf = wr - (1-wr)/(aw/al)
        print(f"  Full Kelly f={kf:.3f} (WR={wr:.3f}, AvgW=${aw:.2f}, AvgL=${al:.2f})")
        for frac in [0.25, 0.50, 0.75, 1.0]:
            f = kf * frac
            sp = [p * max(0.1, min(3.0, 1 + f)) for p in pnl_list]
            eval_sizing(sp, f"{frac:.0%} Kelly (f={f:.3f})")


# ═══════════════════════════════════════════════════════════════
# Phase C: H4/Daily Keltner (全新时间尺度)
# ═══════════════════════════════════════════════════════════════

def run_phase_C(data):
    print("\n" + "=" * 80)
    print("Phase C: H4/Daily Keltner — New Timeframe Alpha")
    print("=" * 80)

    h1_df = data.h1_df.copy()
    m15_df = data.m15_df.copy()

    # Build H4 from H1
    h4 = h1_df.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    # Build Daily from H1
    d1 = h1_df.resample('D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    print(f"  H1: {len(h1_df):,} bars")
    print(f"  H4: {len(h4):,} bars")
    print(f"  D1: {len(d1):,} bars")

    def add_kc(df, ema_period=20, atr_period=14, mult=1.5):
        df = df.copy()
        df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
        tr = pd.DataFrame({
            'hl': df['High'] - df['Low'],
            'hc': (df['High'] - df['Close'].shift(1)).abs(),
            'lc': (df['Low'] - df['Close'].shift(1)).abs(),
        }).max(axis=1)
        df['ATR'] = tr.rolling(atr_period).mean()
        df['KC_upper'] = df['EMA'] + mult * df['ATR']
        df['KC_lower'] = df['EMA'] - mult * df['ATR']
        df['ADX'] = compute_adx(df, atr_period)
        return df

    def compute_adx(df, period=14):
        high = df['High']; low = df['Low']; close = df['Close']
        plus_dm = high.diff(); minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        tr = pd.DataFrame({'hl': high-low, 'hc': (high-close.shift(1)).abs(), 'lc': (low-close.shift(1)).abs()}).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.rolling(period).mean()
        return adx

    def backtest_kc_simple(df, label, ema=20, atr_p=14, mult=1.5, adx_thresh=18,
                           sl_atr=3.5, tp_atr=8.0, trail_act_atr=0.28, trail_dist_atr=0.06,
                           max_hold=20, spread=0.30):
        """Simple KC backtest on any timeframe bar data."""
        df = add_kc(df, ema, atr_p, mult)
        df = df.dropna()

        trades = []; pos = None; equity = [2000.0]
        close = df['Close'].values; high = df['High'].values; low = df['Low'].values
        kc_up = df['KC_upper'].values; kc_lo = df['KC_lower'].values
        atr = df['ATR'].values; adx = df['ADX'].values
        times = df.index; n = len(df)

        lot = 0.03; cooldown = 0; last_exit = -999

        for i in range(1, n):
            c = close[i]; h = high[i]; lo_v = low[i]
            cur_atr = atr[i]; cur_adx = adx[i]

            if pos is not None:
                held = i - pos['bar']
                if pos['dir'] == 'BUY':
                    pnl_high = (h - pos['entry'] - spread) * lot * 100
                    pnl_low = (lo_v - pos['entry'] - spread) * lot * 100
                    pnl_cur = (c - pos['entry'] - spread) * lot * 100
                    trail_price = h
                else:
                    pnl_high = (pos['entry'] - lo_v - spread) * lot * 100
                    pnl_low = (pos['entry'] - h - spread) * lot * 100
                    pnl_cur = (pos['entry'] - c - spread) * lot * 100
                    trail_price = lo_v

                tp_val = tp_atr * pos['atr'] * lot * 100
                sl_val = sl_atr * pos['atr'] * lot * 100

                # TP
                if pnl_high >= tp_val:
                    _add_trade(trades, equity, pos, c, times[i], "TP", i, tp_val, lot)
                    pos = None; last_exit = i; continue
                # SL
                if pnl_low <= -sl_val:
                    _add_trade(trades, equity, pos, c, times[i], "SL", i, -sl_val, lot)
                    pos = None; last_exit = i; continue

                # Trailing
                act_dist = trail_act_atr * pos['atr']
                trail_d = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY':
                    if h - pos['entry'] >= act_dist:
                        trail_stop = h - trail_d
                        if lo_v <= trail_stop:
                            trail_pnl = (trail_stop - pos['entry'] - spread) * lot * 100
                            _add_trade(trades, equity, pos, c, times[i], "Trail", i, trail_pnl, lot)
                            pos = None; last_exit = i; continue
                        pos['best'] = max(pos.get('best', h), h)
                else:
                    if pos['entry'] - lo_v >= act_dist:
                        trail_stop = lo_v + trail_d
                        if h >= trail_stop:
                            trail_pnl = (pos['entry'] - trail_stop - spread) * lot * 100
                            _add_trade(trades, equity, pos, c, times[i], "Trail", i, trail_pnl, lot)
                            pos = None; last_exit = i; continue

                # Timeout
                if held >= max_hold:
                    _add_trade(trades, equity, pos, c, times[i], "Timeout", i, pnl_cur, lot)
                    pos = None; last_exit = i; continue

            # Entry
            if pos is not None: continue
            if i - last_exit < 2: continue
            if np.isnan(cur_adx) or cur_adx < adx_thresh: continue
            if np.isnan(cur_atr) or cur_atr < 0.1: continue

            prev_c = close[i-1]
            if prev_c > kc_up[i-1]:
                entry = c + spread/2
                pos = {'dir': 'BUY', 'entry': entry, 'bar': i, 'time': times[i],
                       'atr': cur_atr, 'best': entry}
            elif prev_c < kc_lo[i-1]:
                entry = c - spread/2
                pos = {'dir': 'SELL', 'entry': entry, 'bar': i, 'time': times[i],
                       'atr': cur_atr, 'best': entry}

        if not trades:
            print(f"  {label}: No trades")
            return {'label': label, 'n': 0, 'sharpe': 0, 'total_pnl': 0, 'win_rate': 0, 'max_dd': 0}

        pnls = [t['pnl'] for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        total = sum(pnls)
        eq = np.cumsum(pnls); dd = (np.maximum.accumulate(eq + 2000) - (eq + 2000)).max()

        daily = {}
        for t in trades:
            d = pd.Timestamp(t['exit_time']).date()
            daily.setdefault(d, 0); daily[d] += t['pnl']
        da = np.array(list(daily.values()))
        sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0

        wr = wins/len(trades)*100

        # Year breakdown
        yp = {}
        for t in trades:
            y = pd.Timestamp(t['exit_time']).year
            yp.setdefault(y, 0); yp[y] += t['pnl']

        print(f"  {label}: N={len(trades)}, Sharpe={sh:.2f}, PnL=${total:.0f}, "
              f"WR={wr:.1f}%, MaxDD=${dd:.0f}")
        years_str = " ".join(f"{y}:${p:+.0f}" for y, p in sorted(yp.items()))
        print(f"    Years: {years_str}")

        return {'label': label, 'n': len(trades), 'sharpe': sh, 'total_pnl': total,
                'win_rate': wr, 'max_dd': dd}

    def _add_trade(trades, equity, pos, exit_p, exit_time, reason, bar_idx, pnl, lot):
        trades.append({
            'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar'],
        })
        equity.append(equity[-1] + pnl)

    # ── H4 Keltner Grid ──
    print(f"\n  === H4 Keltner Channel ===")
    h4_results = []
    for ema in [20, 25, 30]:
        for mult in [1.0, 1.2, 1.5, 2.0]:
            for adx in [15, 18, 22]:
                lbl = f"H4_EMA{ema}_M{mult}_ADX{adx}"
                r = backtest_kc_simple(h4, lbl, ema=ema, mult=mult, adx_thresh=adx,
                                       max_hold=30, trail_act_atr=0.28, trail_dist_atr=0.06)
                h4_results.append(r)

    print(f"\n  H4 Top-5 by Sharpe:")
    h4_top = sorted([r for r in h4_results if r['n'] > 50], key=lambda x: x['sharpe'], reverse=True)[:5]
    for r in h4_top:
        print(f"    {r['label']}: Sharpe={r['sharpe']:.2f}, N={r['n']}, PnL=${r['total_pnl']:.0f}, WR={r['win_rate']:.1f}%")

    # ── Daily Keltner Grid ──
    print(f"\n  === Daily Keltner Channel ===")
    d1_results = []
    for ema in [10, 15, 20]:
        for mult in [1.0, 1.5, 2.0]:
            for adx in [15, 18, 22]:
                lbl = f"D1_EMA{ema}_M{mult}_ADX{adx}"
                r = backtest_kc_simple(d1, lbl, ema=ema, mult=mult, adx_thresh=adx,
                                       max_hold=15, trail_act_atr=0.40, trail_dist_atr=0.10)
                d1_results.append(r)

    print(f"\n  Daily Top-5 by Sharpe:")
    d1_top = sorted([r for r in d1_results if r['n'] > 30], key=lambda x: x['sharpe'], reverse=True)[:5]
    for r in d1_top:
        print(f"    {r['label']}: Sharpe={r['sharpe']:.2f}, N={r['n']}, PnL=${r['total_pnl']:.0f}, WR={r['win_rate']:.1f}%")

    # Correlation with L7 (if H4/D1 has positive results)
    if h4_top and h4_top[0]['sharpe'] > 0:
        print(f"\n  >>> H4 shows promise, consider correlation analysis with L7")
    if d1_top and d1_top[0]['sharpe'] > 0:
        print(f"\n  >>> Daily shows promise, consider correlation analysis with L7")


def main():
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out_path = OUT_DIR / "R25_output.txt"
    out = open(out_path, 'w', encoding='utf-8', buffering=1)
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, out)

    print(f"# R25: MaxHold + Dynamic Sizing + H4/Daily Keltner")
    print(f"# Started: {ts}\n")

    data = DataBundle.load_default()

    for phase_fn, name in [(run_phase_A, "A"), (run_phase_B, "B"), (run_phase_C, "C")]:
        try:
            phase_fn(data)
            out.flush()
            print(f"\n# Phase {name} completed at {datetime.now().strftime('%H:%M:%S')}")
            out.flush()
        except Exception as e:
            print(f"\n# Phase {name} FAILED: {e}")
            import traceback; traceback.print_exc()
            out.flush()

    elapsed = time.time() - t0
    print(f"\n# Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Elapsed: {elapsed/60:.1f} minutes")

    sys.stdout = old_stdout
    out.close()
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
