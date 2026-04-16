#!/usr/bin/env python3
"""
EXP36-41 批量串行执行
======================
共享一次数据加载 + 两次基线回测 (Current / Mega)，
然后依次执行 6 个实验的 post-hoc 分析。

预计总时间: ~15 分钟 (2x5min回测 + 6x<1min分析)
"""
import sys, os, time, gc
from datetime import datetime
from collections import defaultdict
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS
from backtest.stats import calc_stats

OUTPUT_FILE = "exp36_41_output.txt"


class TeeOutput:
    """同时写到文件和 stdout"""
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


tee = TeeOutput(OUTPUT_FILE)
sys.stdout = tee

print("=" * 70)
print("EXP36-41 BATCH RUN (SERIAL, SHARED DATA)")
print(f"Started: {datetime.now()}")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# SHARED: Load data + run 2 baselines
# ═══════════════════════════════════════════════════════════════

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

CURRENT = {**C12_KWARGS, "intraday_adaptive": True}
MEGA = {
    **C12_KWARGS, "intraday_adaptive": True,
    "trailing_activate_atr": 0.5, "trailing_distance_atr": 0.15,
    "regime_config": {
        'low': {'trail_act': 0.7, 'trail_dist': 0.25},
        'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
        'high': {'trail_act': 0.4, 'trail_dist': 0.10},
    },
}

print("\n--- Running Current baseline ---")
baseline_cur = run_variant(data, "Current", **CURRENT)
trades_cur = baseline_cur['_trades']

print("\n--- Running Mega baseline ---")
baseline_mega = run_variant(data, "Mega", **MEGA)
trades_mega = baseline_mega['_trades']

h1_df = data.h1_df.copy()

print(f"\n  Shared baselines ready:")
print(f"  Current: N={baseline_cur['n']:,} Sharpe={baseline_cur['sharpe']:.2f} PnL=${baseline_cur['total_pnl']:,.0f}")
print(f"  Mega:    N={baseline_mega['n']:,} Sharpe={baseline_mega['sharpe']:.2f} PnL=${baseline_mega['total_pnl']:,.0f}")


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_sharpe(trades, pnls):
    daily = defaultdict(float)
    for t, pnl in zip(trades, pnls):
        day = t.entry_time.strftime('%Y-%m-%d')
        daily[day] += pnl
    vals = list(daily.values())
    if len(vals) > 1 and np.std(vals) > 0:
        return np.mean(vals) / np.std(vals) * np.sqrt(252)
    return 0


def to_utc_ts(dt):
    """Safely convert any datetime to tz-aware UTC Timestamp."""
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        return ts.tz_localize('UTC')
    return ts


# ═══════════════════════════════════════════════════════════════
# EXP36: HOUR-OF-DAY SESSION ANALYSIS
# ═══════════════════════════════════════════════════════════════

def run_exp36():
    print("\n\n" + "=" * 70)
    print("EXP36: HOUR-OF-DAY SESSION ANALYSIS")
    print("=" * 70)
    t0 = time.time()

    SESSIONS = {
        'Asia (0-7)': range(0, 7),
        'London (7-13)': range(7, 13),
        'NY (13-17)': range(13, 17),
        'LDN-NY Overlap (13-16)': range(13, 16),
        'Late (17-21)': range(17, 21),
        'Night (21-24)': range(21, 24),
    }

    for label, trades, bstats in [("Current", trades_cur, baseline_cur),
                                   ("Mega", trades_mega, baseline_mega)]:
        by_hour = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0,
                                        'sl': 0, 'trail': 0, 'timeout': 0})
        for t in trades:
            h = t.entry_time.hour
            d = by_hour[h]
            d['n'] += 1
            d['pnl'] += t.pnl
            if t.pnl > 0:
                d['wins'] += 1
            reason = t.exit_reason.split(':')[0] if ':' in t.exit_reason else t.exit_reason
            if reason == 'SL': d['sl'] += 1
            elif 'railing' in reason: d['trail'] += 1
            elif reason in ('Timeout', 'time_stop'): d['timeout'] += 1

        print(f"\n  {label} Hourly Breakdown:")
        print(f"  {'Hour':>4} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6} {'SL':>4} {'Trail':>6} {'TO':>4}")
        print(f"  {'-'*55}")
        for h in range(24):
            d = by_hour[h]
            if d['n'] == 0: continue
            wr = 100 * d['wins'] / d['n']
            ppt = d['pnl'] / d['n']
            print(f"  {h:>4} {d['n']:>6} ${d['pnl']:>9,.0f} ${ppt:>6.2f} {wr:>5.1f}% {d['sl']:>4} {d['trail']:>6} {d['timeout']:>4}")

        # Session summary
        print(f"\n  {label} Session Summary:")
        print(f"  {'Session':<25} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
        print(f"  {'-'*58}")
        for sname, hours in SESSIONS.items():
            n = sum(by_hour[h]['n'] for h in hours)
            pnl = sum(by_hour[h]['pnl'] for h in hours)
            wins = sum(by_hour[h]['wins'] for h in hours)
            if n == 0: continue
            print(f"  {sname:<25} {n:>6} ${pnl:>9,.0f} ${pnl/n:>6.2f} {100*wins/n:>5.1f}%")

    # Skip-hour analysis
    hour_ppt = {h: by_hour[h]['pnl'] / by_hour[h]['n'] for h in range(24) if by_hour[h]['n'] > 20}
    sorted_hours = sorted(hour_ppt, key=hour_ppt.get) if hour_ppt else []

    filters = [
        ("Skip Asia (0-7)", lambda t: t.entry_time.hour not in range(0, 7)),
        ("Skip Late (17-21)", lambda t: t.entry_time.hour not in range(17, 21)),
        ("Only London+NY (7-17)", lambda t: 7 <= t.entry_time.hour < 17),
    ]
    if len(sorted_hours) >= 3:
        worst3 = sorted_hours[:3]
        filters.append((f"Skip worst 3h ({worst3})", lambda t, w=worst3: t.entry_time.hour not in w))

    for label, trades, bstats in [("Current", trades_cur, baseline_cur),
                                   ("Mega", trades_mega, baseline_mega)]:
        print(f"\n  {label} Skip-Hour Analysis:")
        print(f"  {'Filter':<35} {'N':>6} {'Sharpe':>8} {'Delta':>7} {'PnL':>10} {'$/t':>7}")
        print(f"  {'-'*75}")
        for fname, ffunc in filters:
            kept = [t for t in trades if ffunc(t)]
            if len(kept) < 50: continue
            equity = [0.0]
            for t in kept: equity.append(equity[-1] + t.pnl)
            stats = calc_stats(kept, equity)
            d = stats['sharpe'] - bstats['sharpe']
            ppt = stats['total_pnl'] / stats['n'] if stats['n'] > 0 else 0
            print(f"  {fname:<35} {stats['n']:>6} {stats['sharpe']:>8.2f} {d:>+7.2f} ${stats['total_pnl']:>9,.0f} ${ppt:>6.2f}")

    # Yearly session stability
    sessions_simple = {'Asia': range(0, 7), 'London': range(7, 13), 'NY': range(13, 17), 'Late': range(17, 22)}
    print(f"\n  Yearly Session $/trade (Current):")
    print(f"  {'Year':<6}", end="")
    for s in sessions_simple: print(f" {s:>12}", end="")
    print()
    for year in range(2015, 2027):
        start, end = f"{year}-01-01", f"{year+1}-01-01" if year < 2026 else "2026-04-01"
        yr = [t for t in trades_cur if start <= t.entry_time.strftime('%Y-%m-%d') < end]
        if not yr: continue
        print(f"  {year:<6}", end="")
        for sname, hours in sessions_simple.items():
            st = [t for t in yr if t.entry_time.hour in hours]
            if st: print(f"  ${sum(t.pnl for t in st)/len(st):>9.2f}", end="")
            else: print(f" {'--':>12}", end="")
        print()

    print(f"\n  EXP36 done in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════
# EXP37: PARTIAL PROFIT TAKING
# ═══════════════════════════════════════════════════════════════

def run_exp37():
    print("\n\n" + "=" * 70)
    print("EXP37: PARTIAL PROFIT TAKING ANALYSIS")
    print("=" * 70)
    t0 = time.time()

    # Reconstruct MFE per trade from H1 data
    def get_trade_mfe_and_sl(t):
        """Estimate MFE and SL distance from H1 price action during holding."""
        ts_entry = to_utc_ts(t.entry_time)
        ts_exit = to_utc_ts(t.exit_time)
        mask = (h1_df.index >= ts_entry) & (h1_df.index <= ts_exit)
        bars = h1_df.loc[mask]
        if len(bars) == 0:
            return 0, 0
        if t.direction == 'BUY':
            mfe = float(bars['High'].max()) - t.entry_price
            # SL distance ~ ATR * 4.5 (from C12_KWARGS)
            idx = h1_df.index.get_indexer([ts_entry], method='ffill')[0]
            atr = float(h1_df.iloc[max(0, idx)]['ATR']) if idx >= 0 else 10
            sl_dist = atr * 4.5
        else:
            mfe = t.entry_price - float(bars['Low'].min())
            idx = h1_df.index.get_indexer([ts_entry], method='ffill')[0]
            atr = float(h1_df.iloc[max(0, idx)]['ATR']) if idx >= 0 else 10
            sl_dist = atr * 4.5
        return max(mfe, 0), sl_dist

    def simulate_partial(trades, partial_at_pct=0.5, partial_ratio=0.5):
        adjusted_pnls = []
        n_partial = 0
        for t in trades:
            mfe, sl_dist = get_trade_mfe_and_sl(t)
            partial_level = sl_dist * partial_at_pct
            if mfe > partial_level and partial_level > 0:
                locked = partial_level * partial_ratio * t.lots * 100
                remaining = t.pnl * (1 - partial_ratio)
                adjusted_pnls.append(locked + remaining)
                n_partial += 1
            else:
                adjusted_pnls.append(t.pnl)
        return adjusted_pnls, n_partial

    configs = [
        ("No partial (baseline)", 0, 0),
        ("Partial @0.3xSL, close 30%", 0.3, 0.3),
        ("Partial @0.3xSL, close 50%", 0.3, 0.5),
        ("Partial @0.5xSL, close 30%", 0.5, 0.3),
        ("Partial @0.5xSL, close 50%", 0.5, 0.5),
        ("Partial @0.5xSL, close 70%", 0.5, 0.7),
        ("Partial @1.0xSL, close 50%", 1.0, 0.5),
        ("Partial @1.5xSL, close 50%", 1.5, 0.5),
        ("Partial @2.0xSL, close 50%", 2.0, 0.5),
    ]

    for label, trades in [("Current", trades_cur), ("Mega", trades_mega)]:
        base_pnls = [t.pnl for t in trades]
        base_sharpe = compute_sharpe(trades, base_pnls)
        base_total = sum(base_pnls)

        print(f"\n  {label}: N={len(trades):,} Sharpe={base_sharpe:.2f} PnL=${base_total:,.0f}")
        print(f"  {'Config':<35} {'N_partial':>10} {'PnL':>10} {'Sharpe':>8} {'Delta':>7} {'$/t':>7}")
        print(f"  {'-'*80}")
        for cname, at_pct, ratio in configs:
            if at_pct == 0:
                pnls, n_p = base_pnls, 0
            else:
                pnls, n_p = simulate_partial(trades, at_pct, ratio)
            total = sum(pnls)
            sharpe = compute_sharpe(trades, pnls)
            d = sharpe - base_sharpe
            ppt = total / len(pnls) if pnls else 0
            print(f"  {cname:<35} {n_p:>10} ${total:>9,.0f} {sharpe:>8.2f} {d:>+7.2f} ${ppt:>6.2f}")

    # MFE distribution
    print(f"\n  MFE Distribution:")
    for label, trades in [("Current", trades_cur), ("Mega", trades_mega)]:
        mfe_data = [(get_trade_mfe_and_sl(t)) for t in trades[:500]]
        mfe_sl = [m / s for m, s in mfe_data if s > 0 and m > 0]
        if not mfe_sl: continue
        print(f"  {label} (sample 500): MFE/SL mean={np.mean(mfe_sl):.2f} median={np.median(mfe_sl):.2f}")
        for thr in [0.3, 0.5, 1.0, 1.5, 2.0]:
            pct = 100 * sum(1 for r in mfe_sl if r >= thr) / len(mfe_sl)
            print(f"    MFE >= {thr:.1f}xSL: {pct:.1f}%")

    print(f"\n  EXP37 done in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════
# EXP38: RSI DIVERGENCE EARLY EXIT
# ═══════════════════════════════════════════════════════════════

def run_exp38():
    print("\n\n" + "=" * 70)
    print("EXP38: RSI DIVERGENCE EARLY EXIT ANALYSIS")
    print("=" * 70)
    t0 = time.time()

    # RSI(9)
    if 'RSI9' not in h1_df.columns:
        delta = h1_df['Close'].diff()
        gain = delta.clip(lower=0).rolling(9).mean()
        loss = (-delta.clip(upper=0)).rolling(9).mean()
        rs = gain / loss
        h1_df['RSI9'] = 100 - (100 / (1 + rs))

    def detect_divergence(h1_window, direction, lookback=5):
        if h1_window is None or len(h1_window) < lookback + 2:
            return False
        w = h1_window.iloc[-(lookback+2):]
        closes, rsis = w['Close'].values, w['RSI9'].values
        if any(np.isnan(rsis)): return False
        if direction == 'BUY':
            for i in range(2, len(closes)):
                if closes[i] > closes[i-2] and rsis[i] < rsis[i-2]: return True
        else:
            for i in range(2, len(closes)):
                if closes[i] < closes[i-2] and rsis[i] > rsis[i-2]: return True
        return False

    for label, trades in [("Current", trades_cur), ("Mega", trades_mega)]:
        keltner = [t for t in trades if t.strategy == 'keltner']
        div_w, div_l, nodiv_w, nodiv_l = 0, 0, 0, 0
        div_pnl, nodiv_pnl = 0, 0

        for t in keltner:
            et = getattr(t, 'exit_time', None)
            if et is None: continue
            mask = (h1_df.index >= to_utc_ts(t.entry_time)) & \
                   (h1_df.index <= to_utc_ts(et))
            hb = h1_df.loc[mask]
            if len(hb) < 3: continue

            found = False
            for i in range(3, len(hb)):
                if detect_divergence(hb.iloc[:i+1], t.direction):
                    found = True; break

            if found:
                div_pnl += t.pnl
                if t.pnl > 0: div_w += 1
                else: div_l += 1
            else:
                nodiv_pnl += t.pnl
                if t.pnl > 0: nodiv_w += 1
                else: nodiv_l += 1

        div_n = div_w + div_l
        nodiv_n = nodiv_w + nodiv_l
        total = div_n + nodiv_n
        if total == 0: continue

        print(f"\n  {label} RSI(9) Divergence ({len(keltner)} keltner trades):")
        print(f"    Divergence found: {div_n} ({100*div_n/total:.1f}%)")
        if div_n > 0:
            print(f"      WR: {100*div_w/div_n:.1f}%, $/t: ${div_pnl/div_n:.2f}, Total: ${div_pnl:,.0f}")
        print(f"    No divergence: {nodiv_n}")
        if nodiv_n > 0:
            print(f"      WR: {100*nodiv_w/nodiv_n:.1f}%, $/t: ${nodiv_pnl/nodiv_n:.2f}, Total: ${nodiv_pnl:,.0f}")
        if div_n > 0 and nodiv_n > 0:
            print(f"    Conclusion: Div $/t ${div_pnl/div_n:.2f} vs NoDIV ${nodiv_pnl/nodiv_n:.2f} "
                  f"=> diff ${div_pnl/div_n - nodiv_pnl/nodiv_n:+.2f}")

    print(f"\n  EXP38 done in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════
# EXP39: D1 DAILY TREND FILTER
# ═══════════════════════════════════════════════════════════════

def run_exp39():
    print("\n\n" + "=" * 70)
    print("EXP39: D1 DAILY TREND DIRECTION FILTER")
    print("=" * 70)
    t0 = time.time()

    # Build D1 from H1
    d1_df = h1_df.resample('1D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    for span in [20, 50, 100]:
        d1_df[f'EMA{span}'] = d1_df['Close'].ewm(span=span, adjust=False).mean()
    print(f"  D1 bars: {len(d1_df):,}")

    def get_d1_trend(entry_time, ema_span=50):
        ts = to_utc_ts(entry_time)
        prev = d1_df.loc[:ts]
        if len(prev) < 2: return 'NEUTRAL'
        row = prev.iloc[-2]
        ema = row.get(f'EMA{ema_span}', row['Close'])
        if pd.isna(ema): return 'NEUTRAL'
        if row['Close'] > ema * 1.001: return 'UP'
        if row['Close'] < ema * 0.999: return 'DOWN'
        return 'NEUTRAL'

    for label, trades, bstats in [("Current", trades_cur, baseline_cur),
                                   ("Mega", trades_mega, baseline_mega)]:
        for ema_span in [20, 50, 100]:
            aligned = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0})
            for t in trades:
                trend = get_d1_trend(t.entry_time, ema_span)
                is_aligned = (trend == 'UP' and t.direction == 'BUY') or \
                             (trend == 'DOWN' and t.direction == 'SELL')
                is_against = (trend == 'UP' and t.direction == 'SELL') or \
                             (trend == 'DOWN' and t.direction == 'BUY')
                key = 'Aligned' if is_aligned else ('Against' if is_against else 'Neutral')
                aligned[key]['n'] += 1
                aligned[key]['pnl'] += t.pnl
                if t.pnl > 0: aligned[key]['wins'] += 1

            print(f"\n  {label} D1 EMA{ema_span}:")
            print(f"  {'Type':<10} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
            print(f"  {'-'*42}")
            for key in ['Aligned', 'Against', 'Neutral']:
                d = aligned[key]
                if d['n'] == 0: continue
                print(f"  {key:<10} {d['n']:>6} ${d['pnl']:>9,.0f} ${d['pnl']/d['n']:>6.2f} {100*d['wins']/d['n']:>5.1f}%")

        # D1 filter Sharpe comparison
        filters_d1 = [
            ("No filter", lambda t: True),
            ("Block against D1 EMA50", lambda t: not (
                (get_d1_trend(t.entry_time, 50) == 'UP' and t.direction == 'SELL') or
                (get_d1_trend(t.entry_time, 50) == 'DOWN' and t.direction == 'BUY'))),
            ("Aligned only (drop neutral)", lambda t:
                (get_d1_trend(t.entry_time, 50) == 'UP' and t.direction == 'BUY') or
                (get_d1_trend(t.entry_time, 50) == 'DOWN' and t.direction == 'SELL')),
        ]
        print(f"\n  {label} D1 Filter Sharpe:")
        print(f"  {'Filter':<35} {'N':>6} {'Sharpe':>8} {'Delta':>7} {'PnL':>10}")
        print(f"  {'-'*68}")
        for fname, ffunc in filters_d1:
            kept = [t for t in trades if ffunc(t)]
            if len(kept) < 50: continue
            eq = [0.0]
            for t in kept: eq.append(eq[-1] + t.pnl)
            s = calc_stats(kept, eq)
            print(f"  {fname:<35} {s['n']:>6} {s['sharpe']:>8.2f} {s['sharpe']-bstats['sharpe']:>+7.2f} ${s['total_pnl']:>9,.0f}")

    # Yearly stability
    print(f"\n  Yearly Aligned vs Against $/trade (Current, D1 EMA50):")
    print(f"  {'Year':<6} {'Aligned_$/t':>12} {'Against_$/t':>12} {'Helps?':>8}")
    print(f"  {'-'*40}")
    for year in range(2015, 2027):
        start, end = f"{year}-01-01", f"{year+1}-01-01" if year < 2026 else "2026-04-01"
        yr = [t for t in trades_cur if start <= t.entry_time.strftime('%Y-%m-%d') < end]
        if not yr: continue
        al = [t for t in yr if (get_d1_trend(t.entry_time, 50) == 'UP' and t.direction == 'BUY') or
              (get_d1_trend(t.entry_time, 50) == 'DOWN' and t.direction == 'SELL')]
        ag = [t for t in yr if (get_d1_trend(t.entry_time, 50) == 'UP' and t.direction == 'SELL') or
              (get_d1_trend(t.entry_time, 50) == 'DOWN' and t.direction == 'BUY')]
        a_ppt = sum(t.pnl for t in al) / len(al) if al else 0
        ag_ppt = sum(t.pnl for t in ag) / len(ag) if ag else 0
        print(f"  {year:<6} ${a_ppt:>10.2f} ${ag_ppt:>10.2f} {'YES' if a_ppt > ag_ppt else 'NO':>8}")

    print(f"\n  EXP39 done in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════
# EXP40: CANDLE BODY RATIO FILTER
# ═══════════════════════════════════════════════════════════════

def run_exp40():
    print("\n\n" + "=" * 70)
    print("EXP40: CANDLE BODY RATIO FILTER")
    print("=" * 70)
    t0 = time.time()

    def get_body_ratio(entry_time):
        ts = to_utc_ts(entry_time)
        idx = h1_df.index.get_indexer([ts], method='ffill')[0]
        if idx < 0 or idx >= len(h1_df): return None
        bar = h1_df.iloc[idx]
        range_ = float(bar['High'] - bar['Low'])
        if range_ <= 0: return None
        return abs(float(bar['Close'] - bar['Open'])) / range_

    def get_wick_ratio(entry_time, direction):
        ts = to_utc_ts(entry_time)
        idx = h1_df.index.get_indexer([ts], method='ffill')[0]
        if idx < 0 or idx >= len(h1_df): return None
        bar = h1_df.iloc[idx]
        range_ = float(bar['High'] - bar['Low'])
        if range_ <= 0: return None
        if direction == 'BUY':
            wick = float(bar['High'] - max(bar['Close'], bar['Open']))
        else:
            wick = float(min(bar['Close'], bar['Open']) - bar['Low'])
        return wick / range_

    for label, trades, bstats in [("Current", trades_cur, baseline_cur),
                                   ("Mega", trades_mega, baseline_mega)]:
        keltner = [t for t in trades if t.strategy == 'keltner']
        rw = [get_body_ratio(t.entry_time) for t in keltner if get_body_ratio(t.entry_time) is not None and t.pnl > 0]
        rl = [get_body_ratio(t.entry_time) for t in keltner if get_body_ratio(t.entry_time) is not None and t.pnl <= 0]

        if rw and rl:
            print(f"\n  {label}: Win body ratio={np.mean(rw):.3f}, Lose body ratio={np.mean(rl):.3f}, diff={np.mean(rw)-np.mean(rl):+.4f}")

        # Body ratio buckets
        buckets = [(0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.7), (0.7, 1.01)]
        print(f"  {label} Body Ratio Buckets (Keltner only):")
        print(f"  {'Range':<12} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
        print(f"  {'-'*45}")
        for lo, hi in buckets:
            bt = [t for t in keltner if (get_body_ratio(t.entry_time) or 0.5) >= lo
                  and (get_body_ratio(t.entry_time) or 0.5) < hi]
            if not bt: continue
            pnl = sum(t.pnl for t in bt)
            wins = sum(1 for t in bt if t.pnl > 0)
            print(f"  [{lo:.1f}-{hi:.1f}) {len(bt):>6} ${pnl:>9,.0f} ${pnl/len(bt):>6.2f} {100*wins/len(bt):>5.1f}%")

        # Filter Sharpe comparison
        filters_body = [
            ("No filter", lambda t: True),
            ("Body > 0.3", lambda t: t.strategy != 'keltner' or (get_body_ratio(t.entry_time) or 0.5) > 0.3),
            ("Body > 0.4", lambda t: t.strategy != 'keltner' or (get_body_ratio(t.entry_time) or 0.5) > 0.4),
            ("Body > 0.5", lambda t: t.strategy != 'keltner' or (get_body_ratio(t.entry_time) or 0.5) > 0.5),
            ("Wick < 0.3", lambda t: t.strategy != 'keltner' or (get_wick_ratio(t.entry_time, t.direction) or 0) < 0.3),
        ]
        print(f"\n  {label} Body Ratio Filter Sharpe:")
        print(f"  {'Filter':<25} {'N':>6} {'Sharpe':>8} {'Delta':>7} {'PnL':>10}")
        print(f"  {'-'*58}")
        for fname, ffunc in filters_body:
            kept = [t for t in trades if ffunc(t)]
            if len(kept) < 50: continue
            eq = [0.0]
            for t in kept: eq.append(eq[-1] + t.pnl)
            s = calc_stats(kept, eq)
            print(f"  {fname:<25} {s['n']:>6} {s['sharpe']:>8.2f} {s['sharpe']-bstats['sharpe']:>+7.2f} ${s['total_pnl']:>9,.0f}")

    print(f"\n  EXP40 done in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════
# EXP41: ATR REGIME DYNAMIC POSITION SIZING
# ═══════════════════════════════════════════════════════════════

def run_exp41():
    print("\n\n" + "=" * 70)
    print("EXP41: ATR REGIME DYNAMIC POSITION SIZING (INVERSE VOLATILITY)")
    print("=" * 70)
    t0 = time.time()

    # Pre-compute ATR percentile cache
    atr_pct_cache = {}
    atr_series = h1_df['ATR'].dropna()
    for i in range(50, len(atr_series)):
        ts = atr_series.index[i]
        window = atr_series.iloc[i-50:i]
        atr_pct_cache[ts] = float((window < atr_series.iloc[i]).mean())

    def get_atr_pct(entry_time):
        ts = to_utc_ts(entry_time)
        idx = h1_df.index.get_indexer([ts], method='ffill')[0]
        if idx < 0: return 0.5
        return atr_pct_cache.get(h1_df.index[idx], 0.5)

    # Performance by ATR percentile
    buckets = [(0, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 1.01)]

    for label, trades in [("Current", trades_cur), ("Mega", trades_mega)]:
        print(f"\n  {label} by ATR Percentile:")
        print(f"  {'ATR Pct':<12} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
        print(f"  {'-'*45}")
        for lo, hi in buckets:
            bt = [t for t in trades if lo <= get_atr_pct(t.entry_time) < hi]
            if not bt: continue
            pnl = sum(t.pnl for t in bt)
            wins = sum(1 for t in bt if t.pnl > 0)
            print(f"  [{lo:.1f}-{hi:.1f}) {len(bt):>6} ${pnl:>9,.0f} ${pnl/len(bt):>6.2f} {100*wins/len(bt):>5.1f}%")

    # Inverse volatility sizing
    schemes = [
        ("Flat 1.0x (baseline)", {(0, 1.01): 1.0}),
        ("InvVol: L=1.5 N=1.0 H=0.6", {(0, 0.30): 1.5, (0.30, 0.70): 1.0, (0.70, 1.01): 0.6}),
        ("InvVol: L=1.3 N=1.0 H=0.7", {(0, 0.30): 1.3, (0.30, 0.70): 1.0, (0.70, 1.01): 0.7}),
        ("InvVol: L=2.0 N=1.0 H=0.5", {(0, 0.30): 2.0, (0.30, 0.70): 1.0, (0.70, 1.01): 0.5}),
        ("ProVol: L=0.7 N=1.0 H=1.3", {(0, 0.30): 0.7, (0.30, 0.70): 1.0, (0.70, 1.01): 1.3}),
        ("Smooth: 1.5-pct", {}),
    ]

    for label, trades in [("Current", trades_cur), ("Mega", trades_mega)]:
        base_pnls = [t.pnl for t in trades]
        base_sh = compute_sharpe(trades, base_pnls)
        base_total = sum(base_pnls)

        print(f"\n  {label} Sizing Schemes:")
        print(f"  {'Scheme':<35} {'PnL':>10} {'Sharpe':>8} {'Delta':>7}")
        print(f"  {'-'*62}")
        for sname, scheme in schemes:
            pnls = []
            for t in trades:
                pct = get_atr_pct(t.entry_time)
                if sname.startswith("Smooth"):
                    scale = max(0.5, min(2.0, 1.5 - pct))
                else:
                    scale = 1.0
                    for (lo, hi), s in scheme.items():
                        if lo <= pct < hi: scale = s; break
                pnls.append(t.pnl * scale)
            total = sum(pnls)
            sh = compute_sharpe(trades, pnls)
            print(f"  {sname:<35} ${total:>9,.0f} {sh:>8.2f} {sh-base_sh:>+7.2f}")

    # Yearly stability
    inv_scheme = {(0, 0.30): 1.5, (0.30, 0.70): 1.0, (0.70, 1.01): 0.6}
    print(f"\n  Yearly InvVol(1.5/1.0/0.6) vs Baseline (Current):")
    print(f"  {'Year':<6} {'Base_Sh':>8} {'InvVol_Sh':>10} {'Delta':>7}")
    print(f"  {'-'*34}")
    for year in range(2015, 2027):
        start, end = f"{year}-01-01", f"{year+1}-01-01" if year < 2026 else "2026-04-01"
        yr = [t for t in trades_cur if start <= t.entry_time.strftime('%Y-%m-%d') < end]
        if len(yr) < 20: continue
        bp = [t.pnl for t in yr]
        ip = [t.pnl * ([s for (lo, hi), s in inv_scheme.items() if lo <= get_atr_pct(t.entry_time) < hi] or [1.0])[0] for t in yr]
        bsh = compute_sharpe(yr, bp)
        ish = compute_sharpe(yr, ip)
        print(f"  {year:<6} {bsh:>8.2f} {ish:>10.2f} {ish-bsh:>+7.2f}")

    print(f"\n  EXP41 done in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════

experiments = [
    ("EXP36", run_exp36),
    ("EXP37", run_exp37),
    ("EXP38", run_exp38),
    ("EXP39", run_exp39),
    ("EXP40", run_exp40),
    ("EXP41", run_exp41),
]

for name, func in experiments:
    try:
        func()
    except Exception as e:
        print(f"\n  !!! {name} FAILED: {e}")
        import traceback
        traceback.print_exc()
    gc.collect()

total_elapsed = time.time() - t_total
print("\n\n" + "=" * 70)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 70)
print(f"  Total runtime: {total_elapsed/60:.1f} minutes")
print(f"  Current: Sharpe={baseline_cur['sharpe']:.2f} PnL=${baseline_cur['total_pnl']:,.0f}")
print(f"  Mega:    Sharpe={baseline_mega['sharpe']:.2f} PnL=${baseline_mega['total_pnl']:,.0f}")
print(f"  Output saved to: {OUTPUT_FILE}")
print(f"  Finished: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nDone! Results in {OUTPUT_FILE}")
