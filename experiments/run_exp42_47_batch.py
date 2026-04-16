#!/usr/bin/env python3
"""
EXP42-47 批量串行执行
======================
新一轮实验 — 聚焦于黄金定价核心驱动因素:
  EXP42: 宏观 Regime (DXY/VIX) 日级偏倚调节
  EXP43: 大波动日后的动量延续 vs 反转
  EXP44: 波动率聚集效应 → 动态 Trailing 参数
  EXP45: 连续方向偏倚（趋势日识别）
  EXP46: 整数关口效应
  EXP47: 隔夜 vs 日内收益不对称性

共享一次数据加载 + 两次基线回测。
"""
import sys, os, time, gc
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS
from backtest.stats import calc_stats

OUTPUT_FILE = "exp42_47_output.txt"


class TeeOutput:
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
print("EXP42-47 BATCH RUN — GOLD PRICING CORE FACTORS")
print(f"Started: {datetime.now()}")
print("=" * 70)

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
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        return ts.tz_localize('UTC')
    return ts


# ═══════════════════════════════════════════════════════════════
# Load macro data
# ═══════════════════════════════════════════════════════════════

macro_path = "data/macro_history.csv"
macro_df = pd.read_csv(macro_path, parse_dates=['date'], index_col='date')
print(f"\n  Macro data: {len(macro_df)} rows, {macro_df.index[0]} -> {macro_df.index[-1]}")
print(f"  Columns: {list(macro_df.columns)}")


# ═══════════════════════════════════════════════════════════════
# EXP42: MACRO REGIME DAILY BIAS
# ═══════════════════════════════════════════════════════════════

def run_exp42():
    print("\n\n" + "=" * 70)
    print("EXP42: MACRO REGIME (DXY/VIX) DAILY POSITION SIZING")
    print("=" * 70)
    t0 = time.time()

    def get_macro_on_date(date_str):
        """Get macro snapshot for a trading date."""
        dt = pd.Timestamp(date_str)
        prev = macro_df.loc[:dt]
        if len(prev) < 2:
            return {}
        return prev.iloc[-1].to_dict()

    # Part 1: Trade performance by VIX regime
    print("\n  PART 1: Trade performance by VIX level")
    vix_buckets = [(0, 15, 'Low'), (15, 20, 'Normal'), (20, 25, 'Elevated'),
                   (25, 35, 'High'), (35, 100, 'Panic')]

    for label, trades in [("Current", trades_cur), ("Mega", trades_mega)]:
        print(f"\n  {label} by VIX:")
        print(f"  {'VIX Range':<15} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
        print(f"  {'-'*48}")
        for lo, hi, name in vix_buckets:
            bt = []
            for t in trades:
                m = get_macro_on_date(t.entry_time.strftime('%Y-%m-%d'))
                vix = m.get('vix', None)
                if vix is not None and lo <= vix < hi:
                    bt.append(t)
            if not bt: continue
            pnl = sum(t.pnl for t in bt)
            wins = sum(1 for t in bt if t.pnl > 0)
            print(f"  {name} ({lo}-{hi}){'':<3} {len(bt):>6} ${pnl:>9,.0f} ${pnl/len(bt):>6.2f} {100*wins/len(bt):>5.1f}%")

    # Part 2: Trade performance by DXY trend
    print("\n\n  PART 2: Trade performance by DXY trend")
    for label, trades in [("Current", trades_cur), ("Mega", trades_mega)]:
        dxy_up, dxy_down, dxy_flat = [], [], []
        for t in trades:
            m = get_macro_on_date(t.entry_time.strftime('%Y-%m-%d'))
            dxy_chg = m.get('dxy_pct_change', None)
            if dxy_chg is None: continue
            if dxy_chg > 0.1: dxy_up.append(t)
            elif dxy_chg < -0.1: dxy_down.append(t)
            else: dxy_flat.append(t)

        print(f"\n  {label} by DXY daily change:")
        for name, bt in [("DXY Up (>0.1%)", dxy_up), ("DXY Flat", dxy_flat), ("DXY Down (<-0.1%)", dxy_down)]:
            if not bt: continue
            pnl = sum(t.pnl for t in bt)
            wins = sum(1 for t in bt if t.pnl > 0)
            buy_n = sum(1 for t in bt if t.direction == 'BUY')
            sell_n = sum(1 for t in bt if t.direction == 'SELL')
            buy_pnl = sum(t.pnl for t in bt if t.direction == 'BUY')
            sell_pnl = sum(t.pnl for t in bt if t.direction == 'SELL')
            print(f"  {name:<20} N={len(bt):>5} PnL=${pnl:>8,.0f} $/t=${pnl/len(bt):>5.2f} WR={100*wins/len(bt):.1f}% "
                  f"| BUY: {buy_n} ${buy_pnl:>7,.0f} | SELL: {sell_n} ${sell_pnl:>7,.0f}")

    # Part 3: Combined VIX+DXY regime for position sizing
    print("\n\n  PART 3: VIX+DXY combined regime sizing simulation")
    regimes = {
        'Gold Bull (DXY down + VIX up)': lambda m: m.get('dxy_pct_change', 0) and m['dxy_pct_change'] < -0.1 and m.get('vix', 15) > 20,
        'Headwind (DXY up + VIX down)': lambda m: m.get('dxy_pct_change', 0) and m['dxy_pct_change'] > 0.1 and m.get('vix', 15) < 18,
        'Goldilocks (DXY down + VIX down)': lambda m: m.get('dxy_pct_change', 0) and m['dxy_pct_change'] < -0.1 and m.get('vix', 15) < 18,
        'Haven (DXY up + VIX up)': lambda m: m.get('dxy_pct_change', 0) and m['dxy_pct_change'] > 0.1 and m.get('vix', 15) > 20,
    }

    for label, trades in [("Current", trades_cur), ("Mega", trades_mega)]:
        print(f"\n  {label} VIX+DXY Regime:")
        print(f"  {'Regime':<40} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
        print(f"  {'-'*72}")
        for rname, rfunc in regimes.items():
            bt = []
            for t in trades:
                m = get_macro_on_date(t.entry_time.strftime('%Y-%m-%d'))
                try:
                    if rfunc(m): bt.append(t)
                except (TypeError, KeyError): pass
            if not bt: continue
            pnl = sum(t.pnl for t in bt)
            wins = sum(1 for t in bt if t.pnl > 0)
            print(f"  {rname:<40} {len(bt):>6} ${pnl:>9,.0f} ${pnl/len(bt):>6.2f} {100*wins/len(bt):>5.1f}%")

    # Part 4: Simulate macro-driven sizing
    print("\n\n  PART 4: Macro-driven sizing simulation")
    sizing_rules = [
        ("Flat 1.0x", lambda m: 1.0),
        ("VIX<15: 1.3x, VIX>25: 0.7x", lambda m: 1.3 if m.get('vix', 18) < 15 else (0.7 if m.get('vix', 18) > 25 else 1.0)),
        ("DXY down: BUY 1.2x", lambda m: 1.2 if m.get('dxy_pct_change', 0) and m['dxy_pct_change'] < -0.1 else 1.0),
        ("Gold Bull: 1.5x, Headwind: 0.5x", lambda m: 1.5 if (m.get('dxy_pct_change', 0) and m['dxy_pct_change'] < -0.1 and m.get('vix', 15) > 20) else (0.5 if (m.get('dxy_pct_change', 0) and m['dxy_pct_change'] > 0.1 and m.get('vix', 15) < 18) else 1.0)),
    ]

    for label, trades in [("Current", trades_cur), ("Mega", trades_mega)]:
        base_pnls = [t.pnl for t in trades]
        base_sh = compute_sharpe(trades, base_pnls)
        print(f"\n  {label} Macro Sizing:")
        print(f"  {'Rule':<45} {'PnL':>10} {'Sharpe':>8} {'Delta':>7}")
        print(f"  {'-'*72}")
        for rname, rfunc in sizing_rules:
            pnls = []
            for t in trades:
                m = get_macro_on_date(t.entry_time.strftime('%Y-%m-%d'))
                try:
                    scale = rfunc(m)
                except (TypeError, KeyError):
                    scale = 1.0
                pnls.append(t.pnl * scale)
            total = sum(pnls)
            sh = compute_sharpe(trades, pnls)
            print(f"  {rname:<45} ${total:>9,.0f} {sh:>8.2f} {sh-base_sh:>+7.2f}")

    print(f"\n  EXP42 done in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════
# EXP43: POST LARGE MOVE MOMENTUM vs REVERSAL
# ═══════════════════════════════════════════════════════════════

def run_exp43():
    print("\n\n" + "=" * 70)
    print("EXP43: LARGE DAILY MOVE — NEXT DAY CONTINUATION vs REVERSAL")
    print("=" * 70)
    t0 = time.time()

    # Build daily returns
    d1 = h1_df.resample('1D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    d1['ret'] = d1['Close'].pct_change() * 100
    d1['range_pct'] = (d1['High'] - d1['Low']) / d1['Close'].shift(1) * 100
    d1['atr14'] = (d1['High'] - d1['Low']).rolling(14).mean()
    d1['range_ratio'] = (d1['High'] - d1['Low']) / d1['atr14']

    # Part 1: Next-day return after large moves
    print("\n  PART 1: Next-day gold return after large daily moves")
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5]

    print(f"  {'Move Type':<30} {'N':>5} {'Next_Ret%':>10} {'Next_WR%':>9} {'Cont%':>6}")
    print(f"  {'-'*65}")

    for thr in thresholds:
        # Big up days
        big_up = d1[d1['ret'] > thr]
        if len(big_up) > 5:
            next_rets = []
            for idx_pos in range(len(d1)):
                if d1.index[idx_pos] in big_up.index and idx_pos + 1 < len(d1):
                    next_rets.append(d1.iloc[idx_pos + 1]['ret'])
            if next_rets:
                cont = sum(1 for r in next_rets if r > 0) / len(next_rets) * 100
                print(f"  Up > {thr}%{'':<22} {len(next_rets):>5} {np.mean(next_rets):>+9.3f}% {cont:>8.1f}% {cont:>5.1f}%")

        # Big down days
        big_dn = d1[d1['ret'] < -thr]
        if len(big_dn) > 5:
            next_rets = []
            for idx_pos in range(len(d1)):
                if d1.index[idx_pos] in big_dn.index and idx_pos + 1 < len(d1):
                    next_rets.append(d1.iloc[idx_pos + 1]['ret'])
            if next_rets:
                cont = sum(1 for r in next_rets if r < 0) / len(next_rets) * 100
                print(f"  Down < -{thr}%{'':<20} {len(next_rets):>5} {np.mean(next_rets):>+9.3f}% {100-cont:>8.1f}% {cont:>5.1f}%")

    # Part 2: Trade performance after large moves
    print("\n\n  PART 2: Trade performance on day after large move")
    for label, trades in [("Current", trades_cur), ("Mega", trades_mega)]:
        categories = {'After big up (>1%)': [], 'After big down (<-1%)': [],
                      'After range>2xATR': [], 'Normal day': []}
        for t in trades:
            entry_date = t.entry_time.strftime('%Y-%m-%d')
            dt = pd.Timestamp(entry_date)
            prev_days = d1.loc[:dt]
            if len(prev_days) < 2: continue
            prev = prev_days.iloc[-2]
            if prev['ret'] > 1.0:
                categories['After big up (>1%)'].append(t)
            elif prev['ret'] < -1.0:
                categories['After big down (<-1%)'].append(t)
            elif prev.get('range_ratio', 0) > 2.0:
                categories['After range>2xATR'].append(t)
            else:
                categories['Normal day'].append(t)

        print(f"\n  {label}:")
        print(f"  {'Category':<25} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6} {'BUY_$/t':>8} {'SELL_$/t':>9}")
        print(f"  {'-'*75}")
        for cat, bt in categories.items():
            if not bt: continue
            pnl = sum(t.pnl for t in bt)
            wins = sum(1 for t in bt if t.pnl > 0)
            buy_t = [t for t in bt if t.direction == 'BUY']
            sell_t = [t for t in bt if t.direction == 'SELL']
            buy_ppt = sum(t.pnl for t in buy_t) / len(buy_t) if buy_t else 0
            sell_ppt = sum(t.pnl for t in sell_t) / len(sell_t) if sell_t else 0
            print(f"  {cat:<25} {len(bt):>6} ${pnl:>9,.0f} ${pnl/len(bt):>6.2f} {100*wins/len(bt):>5.1f}% ${buy_ppt:>7.2f} ${sell_ppt:>8.2f}")

    # Part 3: Simulate bias adjustment
    print("\n\n  PART 3: Simulate sizing after large moves")
    for label, trades in [("Current", trades_cur), ("Mega", trades_mega)]:
        base_pnls = [t.pnl for t in trades]
        base_sh = compute_sharpe(trades, base_pnls)

        rules = [
            ("Flat 1.0x", lambda t, prev: 1.0),
            ("After big up: SELL 1.3x, BUY 0.7x", lambda t, prev:
                (1.3 if t.direction == 'SELL' else 0.7) if prev['ret'] > 1.0 else 1.0),
            ("After big down: BUY 1.3x, SELL 0.7x", lambda t, prev:
                (1.3 if t.direction == 'BUY' else 0.7) if prev['ret'] < -1.0 else 1.0),
            ("After any big move: 0.7x (reduce)", lambda t, prev:
                0.7 if abs(prev['ret']) > 1.0 else 1.0),
            ("After any big move: 1.3x (momentum)", lambda t, prev:
                1.3 if abs(prev['ret']) > 1.0 else 1.0),
        ]

        print(f"\n  {label}:")
        print(f"  {'Rule':<45} {'PnL':>10} {'Sharpe':>8} {'Delta':>7}")
        print(f"  {'-'*72}")
        for rname, rfunc in rules:
            pnls = []
            for t in trades:
                entry_date = t.entry_time.strftime('%Y-%m-%d')
                dt = pd.Timestamp(entry_date)
                prev_days = d1.loc[:dt]
                if len(prev_days) < 2:
                    pnls.append(t.pnl)
                    continue
                prev = prev_days.iloc[-2]
                scale = rfunc(t, prev)
                pnls.append(t.pnl * scale)
            total = sum(pnls)
            sh = compute_sharpe(trades, pnls)
            print(f"  {rname:<45} ${total:>9,.0f} {sh:>8.2f} {sh-base_sh:>+7.2f}")

    print(f"\n  EXP43 done in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════
# EXP44: VOLATILITY CLUSTERING → ADAPTIVE TRAILING
# ═══════════════════════════════════════════════════════════════

def run_exp44():
    print("\n\n" + "=" * 70)
    print("EXP44: VOLATILITY CLUSTERING — ADAPTIVE TRAILING STOP")
    print("=" * 70)
    t0 = time.time()

    # Build daily ATR sequence
    d1 = h1_df.resample('1D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    d1['d_range'] = d1['High'] - d1['Low']
    d1['d_atr14'] = d1['d_range'].rolling(14).mean()
    d1['vol_change'] = d1['d_range'] / d1['d_atr14']

    # Part 1: Autocorrelation of daily range
    print("\n  PART 1: Daily range autocorrelation (is vol clustered?)")
    ranges = d1['d_range'].dropna().values
    for lag in [1, 2, 3, 5, 10]:
        if len(ranges) > lag:
            corr = np.corrcoef(ranges[:-lag], ranges[lag:])[0, 1]
            print(f"  Lag {lag:>2} days: autocorr = {corr:.4f}")

    # Part 2: Next day's vol prediction
    print("\n\n  PART 2: High-vol day followed by?")
    vol_changes = d1['vol_change'].dropna()
    for thr in [1.5, 2.0, 2.5]:
        high_vol_days = vol_changes[vol_changes > thr]
        next_day_vols = []
        for idx in high_vol_days.index:
            pos = d1.index.get_loc(idx)
            if pos + 1 < len(d1):
                next_day_vols.append(d1.iloc[pos + 1]['vol_change'])
        if next_day_vols:
            still_high = sum(1 for v in next_day_vols if v > 1.0)
            print(f"  After range>{thr:.1f}xATR: N={len(next_day_vols)}, "
                  f"next day avg vol={np.mean(next_day_vols):.2f}x, "
                  f"still >1x: {100*still_high/len(next_day_vols):.1f}%")

    # Part 3: Trade performance by previous day vol regime
    print("\n\n  PART 3: Trade performance by prev day's vol regime")
    for label, trades in [("Current", trades_cur), ("Mega", trades_mega)]:
        by_vol = {'Low (<0.7x)': [], 'Normal (0.7-1.3x)': [], 'High (1.3-2x)': [], 'Extreme (>2x)': []}
        for t in trades:
            entry_date = t.entry_time.strftime('%Y-%m-%d')
            dt = pd.Timestamp(entry_date)
            prev_days = d1.loc[:dt]
            if len(prev_days) < 2: continue
            vc = prev_days.iloc[-2].get('vol_change', 1.0)
            if pd.isna(vc): continue
            if vc < 0.7: by_vol['Low (<0.7x)'].append(t)
            elif vc < 1.3: by_vol['Normal (0.7-1.3x)'].append(t)
            elif vc < 2.0: by_vol['High (1.3-2x)'].append(t)
            else: by_vol['Extreme (>2x)'].append(t)

        print(f"\n  {label}:")
        print(f"  {'Prev Vol':<20} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
        print(f"  {'-'*52}")
        for cat, bt in by_vol.items():
            if not bt: continue
            pnl = sum(t.pnl for t in bt)
            wins = sum(1 for t in bt if t.pnl > 0)
            print(f"  {cat:<20} {len(bt):>6} ${pnl:>9,.0f} ${pnl/len(bt):>6.2f} {100*wins/len(bt):>5.1f}%")

    # Part 4: Simulate vol-adaptive sizing
    print("\n\n  PART 4: Vol-clustering adaptive sizing")
    for label, trades in [("Current", trades_cur), ("Mega", trades_mega)]:
        base_pnls = [t.pnl for t in trades]
        base_sh = compute_sharpe(trades, base_pnls)

        rules = [
            ("Flat 1.0x", lambda vc: 1.0),
            ("Vol expand: Low=0.7x High=1.3x", lambda vc: 0.7 if vc < 0.7 else (1.3 if vc > 1.3 else 1.0)),
            ("Vol contract: Low=1.3x High=0.7x", lambda vc: 1.3 if vc < 0.7 else (0.7 if vc > 1.3 else 1.0)),
            ("Extreme only: >2x reduce 0.5x", lambda vc: 0.5 if vc > 2.0 else 1.0),
            ("Momentum vol: High=1.5x (ride cluster)", lambda vc: 1.5 if vc > 1.5 else 1.0),
        ]

        print(f"\n  {label}:")
        print(f"  {'Rule':<45} {'PnL':>10} {'Sharpe':>8} {'Delta':>7}")
        print(f"  {'-'*72}")
        for rname, rfunc in rules:
            pnls = []
            for t in trades:
                entry_date = t.entry_time.strftime('%Y-%m-%d')
                dt = pd.Timestamp(entry_date)
                prev_days = d1.loc[:dt]
                if len(prev_days) < 2:
                    pnls.append(t.pnl)
                    continue
                vc = prev_days.iloc[-2].get('vol_change', 1.0)
                if pd.isna(vc): vc = 1.0
                pnls.append(t.pnl * rfunc(vc))
            sh = compute_sharpe(trades, pnls)
            print(f"  {rname:<45} ${sum(pnls):>9,.0f} {sh:>8.2f} {sh-base_sh:>+7.2f}")

    print(f"\n  EXP44 done in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════
# EXP45: CONSECUTIVE DIRECTION BIAS (TREND DAY ID)
# ═══════════════════════════════════════════════════════════════

def run_exp45():
    print("\n\n" + "=" * 70)
    print("EXP45: CONSECUTIVE DAILY DIRECTION — TREND DAY DETECTION")
    print("=" * 70)
    t0 = time.time()

    d1 = h1_df.resample('1D').agg({'Open': 'first', 'Close': 'last'}).dropna()
    d1['dir'] = np.where(d1['Close'] > d1['Open'], 1, -1)

    # Consecutive day streaks
    streak = [0] * len(d1)
    for i in range(1, len(d1)):
        if d1.iloc[i]['dir'] == d1.iloc[i-1]['dir']:
            streak[i] = streak[i-1] + (1 if d1.iloc[i]['dir'] > 0 else -1)
        else:
            streak[i] = d1.iloc[i]['dir']
    d1['streak'] = streak

    # Part 1: Next day continuation after N consecutive days
    print("\n  PART 1: Continuation probability after N same-direction days")
    print(f"  {'Streak':<15} {'N_occur':>8} {'Next_same%':>10} {'Next_ret':>10}")
    print(f"  {'-'*48}")

    for s_len in [1, 2, 3, 4, 5]:
        for direction in ['Up', 'Down']:
            sign = 1 if direction == 'Up' else -1
            matches = []
            for i in range(1, len(d1) - 1):
                if d1.iloc[i]['streak'] == sign * s_len:
                    next_same = 1 if d1.iloc[i+1]['dir'] == sign else 0
                    next_ret = (d1.iloc[i+1]['Close'] - d1.iloc[i+1]['Open']) / d1.iloc[i]['Close'] * 100
                    matches.append((next_same, next_ret))
            if len(matches) > 5:
                cont_rate = 100 * sum(m[0] for m in matches) / len(matches)
                avg_ret = np.mean([m[1] for m in matches])
                print(f"  {s_len}x {direction:<10} {len(matches):>8} {cont_rate:>9.1f}% {avg_ret:>+9.4f}%")

    # Part 2: Trade performance by streak context
    print("\n\n  PART 2: Trade $/t by entry day's streak context")
    for label, trades in [("Current", trades_cur), ("Mega", trades_mega)]:
        by_streak = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0})
        for t in trades:
            entry_date = t.entry_time.strftime('%Y-%m-%d')
            dt = pd.Timestamp(entry_date)
            match = d1.loc[:dt]
            if len(match) < 1: continue
            s = match.iloc[-1].get('streak', 0)
            if pd.isna(s): continue
            s = int(s)
            if s >= 3: bucket = 'Streak 3+ up'
            elif s <= -3: bucket = 'Streak 3+ down'
            elif s >= 1: bucket = 'Mild up (1-2)'
            elif s <= -1: bucket = 'Mild down (1-2)'
            else: bucket = 'Mixed'
            by_streak[bucket]['n'] += 1
            by_streak[bucket]['pnl'] += t.pnl
            if t.pnl > 0: by_streak[bucket]['wins'] += 1

        print(f"\n  {label}:")
        print(f"  {'Streak':<20} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
        print(f"  {'-'*52}")
        for cat in ['Streak 3+ up', 'Mild up (1-2)', 'Mixed', 'Mild down (1-2)', 'Streak 3+ down']:
            d = by_streak[cat]
            if d['n'] == 0: continue
            print(f"  {cat:<20} {d['n']:>6} ${d['pnl']:>9,.0f} ${d['pnl']/d['n']:>6.2f} {100*d['wins']/d['n']:>5.1f}%")

    print(f"\n  EXP45 done in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════
# EXP46: ROUND NUMBER / PSYCHOLOGICAL LEVEL EFFECT
# ═══════════════════════════════════════════════════════════════

def run_exp46():
    print("\n\n" + "=" * 70)
    print("EXP46: ROUND NUMBER / PSYCHOLOGICAL LEVEL EFFECT")
    print("=" * 70)
    t0 = time.time()

    def dist_to_round(price, multiple=50):
        """Distance to nearest round number as fraction of multiple."""
        remainder = price % multiple
        dist = min(remainder, multiple - remainder)
        return dist / multiple

    def nearest_round(price, multiple=50):
        return round(price / multiple) * multiple

    # Part 1: Trade outcome by proximity to $50 and $100 levels
    print("\n  PART 1: Trade outcome by proximity to round numbers")
    for mult in [50, 100]:
        for label, trades in [("Current", trades_cur), ("Mega", trades_mega)]:
            near, far = [], []
            for t in trades:
                d = dist_to_round(t.entry_price, mult)
                if d < 0.15:  # within 15% of round number
                    near.append(t)
                elif d > 0.35:
                    far.append(t)

            if not near or not far: continue
            near_pnl = sum(t.pnl for t in near) / len(near)
            far_pnl = sum(t.pnl for t in far) / len(far)
            near_wr = 100 * sum(1 for t in near if t.pnl > 0) / len(near)
            far_wr = 100 * sum(1 for t in far if t.pnl > 0) / len(far)
            print(f"  {label} ${mult} level: Near({len(near)}) $/t=${near_pnl:.2f} WR={near_wr:.1f}% | "
                  f"Far({len(far)}) $/t=${far_pnl:.2f} WR={far_wr:.1f}% | diff=${near_pnl-far_pnl:+.2f}")

    # Part 2: Direction bias near round numbers
    print("\n\n  PART 2: Direction bias — above vs below round number")
    for mult in [50, 100]:
        for label, trades in [("Current", trades_cur), ("Mega", trades_mega)]:
            above_buy, above_sell, below_buy, below_sell = [], [], [], []
            for t in trades:
                d = dist_to_round(t.entry_price, mult)
                if d > 0.3: continue
                remainder = t.entry_price % mult
                is_above = remainder < mult / 2
                if is_above:
                    if t.direction == 'BUY': above_buy.append(t)
                    else: above_sell.append(t)
                else:
                    if t.direction == 'BUY': below_buy.append(t)
                    else: below_sell.append(t)

            print(f"\n  {label} near ${mult}:")
            for name, bt in [("Above+BUY", above_buy), ("Above+SELL", above_sell),
                              ("Below+BUY", below_buy), ("Below+SELL", below_sell)]:
                if not bt: continue
                pnl = sum(t.pnl for t in bt)
                print(f"    {name:<15} N={len(bt):>5} $/t=${pnl/len(bt):>5.2f}")

    # Part 3: Price bounce/rejection at round numbers
    print("\n\n  PART 3: Bounce/Breakthrough at psychological levels")
    for mult in [50, 100]:
        # Count daily crosses
        d1 = h1_df.resample('1D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
        cross_up, cross_dn, reject_up, reject_dn = 0, 0, 0, 0
        for i in range(1, len(d1)):
            prev_close = d1.iloc[i-1]['Close']
            today = d1.iloc[i]
            prev_round = nearest_round(prev_close, mult)

            if today['Low'] <= prev_round <= today['High']:
                if today['Close'] > prev_round and prev_close < prev_round:
                    cross_up += 1
                elif today['Close'] < prev_round and prev_close > prev_round:
                    cross_dn += 1
                elif today['Close'] > prev_round and prev_close > prev_round:
                    reject_dn += 1  # tested below, bounced up
                elif today['Close'] < prev_round and prev_close < prev_round:
                    reject_up += 1  # tested above, rejected

        total = cross_up + cross_dn + reject_up + reject_dn
        if total > 0:
            print(f"  ${mult} level: Cross_Up={cross_up} ({100*cross_up/total:.1f}%), "
                  f"Cross_Dn={cross_dn} ({100*cross_dn/total:.1f}%), "
                  f"Reject_Up={reject_up}, Reject_Dn={reject_dn}")

    print(f"\n  EXP46 done in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════
# EXP47: OVERNIGHT vs INTRADAY RETURN ASYMMETRY
# ═══════════════════════════════════════════════════════════════

def run_exp47():
    print("\n\n" + "=" * 70)
    print("EXP47: OVERNIGHT vs INTRADAY RETURN ASYMMETRY")
    print("=" * 70)
    t0 = time.time()

    # Build overnight vs intraday returns
    d1 = h1_df.resample('1D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    d1['overnight_ret'] = (d1['Open'] - d1['Close'].shift(1)) / d1['Close'].shift(1) * 100
    d1['intraday_ret'] = (d1['Close'] - d1['Open']) / d1['Open'] * 100
    d1['total_ret'] = d1['overnight_ret'] + d1['intraday_ret']

    ov = d1['overnight_ret'].dropna()
    id_ret = d1['intraday_ret'].dropna()

    # Part 1: Basic statistics
    print("\n  PART 1: Overnight vs Intraday return statistics")
    print(f"  Overnight: mean={ov.mean():+.4f}%, std={ov.std():.4f}%, "
          f"cumulative={ov.sum():+.2f}%, positive={100*(ov>0).mean():.1f}%")
    print(f"  Intraday:  mean={id_ret.mean():+.4f}%, std={id_ret.std():.4f}%, "
          f"cumulative={id_ret.sum():+.2f}%, positive={100*(id_ret>0).mean():.1f}%")
    print(f"  Total:     cumulative={d1['total_ret'].dropna().sum():+.2f}%")

    # Part 2: Yearly breakdown
    print("\n\n  PART 2: Yearly overnight vs intraday cumulative return")
    print(f"  {'Year':<6} {'Overnight%':>11} {'Intraday%':>10} {'Total%':>8} {'OV_Sharpe':>10}")
    print(f"  {'-'*48}")
    for year in range(2015, 2027):
        yr = d1[str(year)]
        if len(yr) < 20: continue
        ov_yr = yr['overnight_ret'].dropna()
        id_yr = yr['intraday_ret'].dropna()
        ov_sh = ov_yr.mean() / ov_yr.std() * np.sqrt(252) if ov_yr.std() > 0 else 0
        print(f"  {year:<6} {ov_yr.sum():>+10.2f}% {id_yr.sum():>+9.2f}% {(ov_yr.sum()+id_yr.sum()):>+7.2f}% {ov_sh:>9.2f}")

    # Part 3: Trade outcome by overnight gap direction
    print("\n\n  PART 3: Trade performance by today's overnight gap")
    for label, trades in [("Current", trades_cur), ("Mega", trades_mega)]:
        gap_up, gap_dn, gap_flat = [], [], []
        for t in trades:
            entry_date = t.entry_time.strftime('%Y-%m-%d')
            dt = pd.Timestamp(entry_date)
            match = d1.loc[:dt]
            if dt not in d1.index and len(match) > 0:
                row = match.iloc[-1]
            elif dt in d1.index:
                row = d1.loc[dt]
            else:
                continue
            ov_r = row.get('overnight_ret', 0)
            if pd.isna(ov_r): continue
            if ov_r > 0.1: gap_up.append(t)
            elif ov_r < -0.1: gap_dn.append(t)
            else: gap_flat.append(t)

        print(f"\n  {label}:")
        for name, bt in [("Gap Up (>0.1%)", gap_up), ("Gap Flat", gap_flat), ("Gap Down (<-0.1%)", gap_dn)]:
            if not bt: continue
            pnl = sum(t.pnl for t in bt)
            wins = sum(1 for t in bt if t.pnl > 0)
            buy_pnl = sum(t.pnl for t in bt if t.direction == 'BUY')
            sell_pnl = sum(t.pnl for t in bt if t.direction == 'SELL')
            buy_n = sum(1 for t in bt if t.direction == 'BUY')
            sell_n = sum(1 for t in bt if t.direction == 'SELL')
            print(f"  {name:<20} N={len(bt):>5} $/t=${pnl/len(bt):>5.2f} WR={100*wins/len(bt):.1f}% "
                  f"| BUY({buy_n}) ${buy_pnl/max(buy_n,1):.2f} | SELL({sell_n}) ${sell_pnl/max(sell_n,1):.2f}")

    # Part 4: Overnight gap as signal
    print("\n\n  PART 4: Gap-follow strategy simulation")
    for label, trades in [("Current", trades_cur), ("Mega", trades_mega)]:
        base_pnls = [t.pnl for t in trades]
        base_sh = compute_sharpe(trades, base_pnls)

        rules = [
            ("Flat 1.0x", lambda t, ov_r: 1.0),
            ("Gap up: BUY 1.3x (momentum)", lambda t, ov_r:
                1.3 if ov_r > 0.1 and t.direction == 'BUY' else 1.0),
            ("Gap up: SELL 1.3x (reversal)", lambda t, ov_r:
                1.3 if ov_r > 0.1 and t.direction == 'SELL' else 1.0),
            ("Gap down: BUY 1.3x (reversal)", lambda t, ov_r:
                1.3 if ov_r < -0.1 and t.direction == 'BUY' else 1.0),
            ("Fade gap: opposite 1.3x", lambda t, ov_r:
                1.3 if (ov_r > 0.1 and t.direction == 'SELL') or (ov_r < -0.1 and t.direction == 'BUY') else 1.0),
        ]

        print(f"\n  {label}:")
        print(f"  {'Rule':<40} {'PnL':>10} {'Sharpe':>8} {'Delta':>7}")
        print(f"  {'-'*67}")
        for rname, rfunc in rules:
            pnls = []
            for t in trades:
                entry_date = t.entry_time.strftime('%Y-%m-%d')
                dt = pd.Timestamp(entry_date)
                match = d1.loc[:dt]
                if dt not in d1.index and len(match) > 0:
                    ov_r = match.iloc[-1].get('overnight_ret', 0)
                elif dt in d1.index:
                    ov_r = d1.loc[dt].get('overnight_ret', 0)
                else:
                    ov_r = 0
                if pd.isna(ov_r): ov_r = 0
                pnls.append(t.pnl * rfunc(t, ov_r))
            sh = compute_sharpe(trades, pnls)
            print(f"  {rname:<40} ${sum(pnls):>9,.0f} {sh:>8.2f} {sh-base_sh:>+7.2f}")

    print(f"\n  EXP47 done in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════

experiments = [
    ("EXP42", run_exp42),
    ("EXP43", run_exp43),
    ("EXP44", run_exp44),
    ("EXP45", run_exp45),
    ("EXP46", run_exp46),
    ("EXP47", run_exp47),
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
