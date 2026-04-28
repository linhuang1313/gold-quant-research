"""
R40: Market Regime Deep Study (~4 hours)
==========================================
黄金市场环境分析 + 策略在不同 regime 下的表现诊断

Phase 1: Regime 识别与标注 (D1 ATR + 价格趋势 + ADX)
Phase 2: L7 逐 regime 表现拆解 (震荡/趋势/牛/熊/高波动/低波动)
Phase 3: Regime 切换时的策略表现 (转折点分析)
Phase 4: Regime-Adaptive 参数测试 (不同 regime 用不同参数)
Phase 5: 2024-2026 近期 regime 诊断 (与历史对比)
Phase 6: Regime 检测器 K-Fold 验证
"""
import sys, os, time, multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.runner import DataBundle, run_variant, run_kfold, LIVE_PARITY_KWARGS
from backtest.stats import calc_stats, aggregate_daily_pnl
from backtest.engine import TradeRecord
import research_config as config

OUT_DIR = Path("results/round40_regime_study")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = min(mp.cpu_count(), 8)

L7_MH8 = {
    **LIVE_PARITY_KWARGS,
    'time_adaptive_trail': {'start': 2, 'decay': 0.75, 'floor': 0.003},
    'min_entry_gap_hours': 1.0,
    'keltner_max_hold_m15': 8,
}


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            try: f.write(data)
            except: pass
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def corrected_sharpe(trades):
    if not trades:
        return 0.0
    trade_daily = {}
    for t in trades:
        d = pd.Timestamp(t.exit_time).date()
        trade_daily[d] = trade_daily.get(d, 0) + t.pnl
    if not trade_daily:
        return 0.0
    start_date = min(trade_daily.keys())
    end_date = max(trade_daily.keys())
    all_dates = pd.bdate_range(start_date, end_date)
    full_daily = [trade_daily.get(d.date(), 0.0) for d in all_dates]
    arr = np.array(full_daily)
    if len(arr) < 2 or np.std(arr, ddof=1) <= 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))


def apply_max_loss_cap(trades, cap_usd):
    capped = []
    for t in trades:
        if t.pnl < -cap_usd:
            capped.append(TradeRecord(
                strategy=t.strategy, direction=t.direction,
                entry_price=t.entry_price, exit_price=t.exit_price,
                entry_time=t.entry_time, exit_time=t.exit_time,
                lots=t.lots, pnl=-cap_usd, exit_reason=t.exit_reason,
                bars_held=t.bars_held,
            ))
        else:
            capped.append(t)
    return capped


def _run_one(args):
    label, base_kwargs, spread = args
    data = DataBundle.load_default()
    kw = {**base_kwargs, 'spread_cost': spread}
    r = run_variant(data, label, verbose=False, **kw)
    csh = corrected_sharpe(r['_trades'])
    return {
        'label': label, 'spread': spread,
        'n': r['n'], 'total_pnl': r['total_pnl'], 'win_rate': r['win_rate'],
        'avg_win': r['avg_win'], 'avg_loss': r['avg_loss'], 'rr': r['rr'],
        'max_dd': r['max_dd'], 'corr_sharpe': csh, 'orig_sharpe': r['sharpe'],
        '_trades': r['_trades'],
    }


# ═════════════════════════════════════════════════════════════
# Phase 1: Regime 识别与标注
# ═════════════════════════════════════════════════════════════

def phase_1(h1_df, m15_df):
    print("\n" + "=" * 90)
    print("  PHASE 1: Market Regime Identification & Labeling")
    print("=" * 90)

    h1 = h1_df.copy()
    if h1.index.tz is not None:
        h1.index = h1.index.tz_localize(None)

    # D1 resample
    d1 = h1.resample('1D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    d1['ATR14'] = (d1['High'] - d1['Low']).rolling(14).mean()
    d1['ATR60_pct'] = d1['ATR14'].rolling(60).rank(pct=True) * 100
    d1['EMA50'] = d1['Close'].ewm(span=50).mean()
    d1['EMA200'] = d1['Close'].ewm(span=200).mean()
    d1['Return_20d'] = d1['Close'].pct_change(20) * 100
    d1['Return_60d'] = d1['Close'].pct_change(60) * 100

    # H1 ADX (use from h1_df if available)
    if 'ADX' in h1_df.columns:
        adx = h1_df['ADX'].copy()
        if adx.index.tz is not None:
            adx.index = adx.index.tz_localize(None)
        d1_adx = adx.resample('1D').mean().dropna()
    else:
        d1_adx = pd.Series(25.0, index=d1.index)

    # Regime labels
    def classify_regime(row, adx_val):
        atr_pct = row.get('ATR60_pct', 50)
        ret_20 = row.get('Return_20d', 0)
        ret_60 = row.get('Return_60d', 0)
        close = row['Close']
        ema50 = row.get('EMA50', close)
        ema200 = row.get('EMA200', close)

        # Volatility regime
        if atr_pct < 20:
            vol_regime = 'LowVol'
        elif atr_pct > 80:
            vol_regime = 'HighVol'
        else:
            vol_regime = 'NormVol'

        # Trend regime
        if close > ema50 > ema200 and ret_20 > 2:
            trend_regime = 'StrongBull'
        elif close > ema200 and ret_60 > 0:
            trend_regime = 'Bull'
        elif close < ema50 < ema200 and ret_20 < -2:
            trend_regime = 'StrongBear'
        elif close < ema200 and ret_60 < 0:
            trend_regime = 'Bear'
        else:
            trend_regime = 'Sideways'

        # ADX regime
        if adx_val > 30:
            adx_regime = 'Trending'
        elif adx_val < 20:
            adx_regime = 'Choppy'
        else:
            adx_regime = 'Moderate'

        return vol_regime, trend_regime, adx_regime

    regimes = []
    for date in d1.index:
        row = d1.loc[date]
        adx_val = d1_adx.get(date, 25)
        if pd.isna(adx_val):
            adx_val = 25
        vol, trend, adx_r = classify_regime(row, adx_val)
        regimes.append({
            'date': date, 'close': row['Close'],
            'atr14': row.get('ATR14', 0), 'atr_pct': row.get('ATR60_pct', 50),
            'ret_20d': row.get('Return_20d', 0), 'ret_60d': row.get('Return_60d', 0),
            'vol_regime': vol, 'trend_regime': trend, 'adx_regime': adx_r,
        })

    regime_df = pd.DataFrame(regimes).set_index('date')

    # Print regime distribution
    print("\n  --- Volatility Regime Distribution ---")
    for r in ['LowVol', 'NormVol', 'HighVol']:
        cnt = (regime_df['vol_regime'] == r).sum()
        pct = cnt / len(regime_df) * 100
        print(f"  {r:<12} {cnt:>6} days ({pct:>5.1f}%)")

    print("\n  --- Trend Regime Distribution ---")
    for r in ['StrongBull', 'Bull', 'Sideways', 'Bear', 'StrongBear']:
        cnt = (regime_df['trend_regime'] == r).sum()
        pct = cnt / len(regime_df) * 100
        print(f"  {r:<12} {cnt:>6} days ({pct:>5.1f}%)")

    print("\n  --- ADX Regime Distribution ---")
    for r in ['Choppy', 'Moderate', 'Trending']:
        cnt = (regime_df['adx_regime'] == r).sum()
        pct = cnt / len(regime_df) * 100
        print(f"  {r:<12} {cnt:>6} days ({pct:>5.1f}%)")

    # Yearly regime breakdown
    print("\n  --- Yearly Regime Breakdown ---")
    print(f"  {'Year':>5} {'LowVol':>7} {'NormVol':>8} {'HighVol':>8} | "
          f"{'Bull':>5} {'Side':>5} {'Bear':>5} | {'Price':>8}")
    print(f"  {'-'*5} {'-'*7} {'-'*8} {'-'*8}   {'-'*5} {'-'*5} {'-'*5}   {'-'*8}")
    for year in range(2015, 2027):
        yr = regime_df[regime_df.index.year == year]
        if len(yr) == 0: continue
        lv = (yr['vol_regime'] == 'LowVol').sum()
        nv = (yr['vol_regime'] == 'NormVol').sum()
        hv = (yr['vol_regime'] == 'HighVol').sum()
        bu = ((yr['trend_regime'] == 'Bull') | (yr['trend_regime'] == 'StrongBull')).sum()
        si = (yr['trend_regime'] == 'Sideways').sum()
        be = ((yr['trend_regime'] == 'Bear') | (yr['trend_regime'] == 'StrongBear')).sum()
        price = yr['close'].iloc[-1]
        print(f"  {year:>5} {lv:>7} {nv:>8} {hv:>8} | {bu:>5} {si:>5} {be:>5} | ${price:>7.0f}")

    return regime_df


# ═════════════════════════════════════════════════════════════
# Phase 2: L7 逐 Regime 表现拆解
# ═════════════════════════════════════════════════════════════

def phase_2(regime_df, trades):
    print("\n" + "=" * 90)
    print("  PHASE 2: L7 Performance by Market Regime")
    print("=" * 90)

    def get_trade_regime(t, regime_df):
        et = pd.Timestamp(t.entry_time)
        if et.tzinfo is not None:
            et = et.tz_localize(None)
        d = et.normalize()
        idx = regime_df.index.searchsorted(d, side='right') - 1
        if 0 <= idx < len(regime_df):
            row = regime_df.iloc[idx]
            return row['vol_regime'], row['trend_regime'], row['adx_regime']
        return 'NormVol', 'Sideways', 'Moderate'

    # Tag each trade with regime
    tagged = []
    for t in trades:
        vol, trend, adx_r = get_trade_regime(t, regime_df)
        tagged.append({
            'pnl': t.pnl, 'direction': t.direction,
            'vol': vol, 'trend': trend, 'adx': adx_r,
            'exit_reason': t.exit_reason, 'bars_held': t.bars_held,
        })
    df = pd.DataFrame(tagged)

    # By Volatility Regime
    print("\n  --- By Volatility Regime ---")
    print(f"  {'Regime':<12} {'N':>6} {'PnL':>10} {'WR%':>6} {'AvgPnL':>8} {'AvgWin':>8} {'AvgLoss':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    for r in ['LowVol', 'NormVol', 'HighVol']:
        sub = df[df['vol'] == r]
        if len(sub) == 0: continue
        n = len(sub)
        pnl = sub['pnl'].sum()
        wr = (sub['pnl'] > 0).mean() * 100
        avg = sub['pnl'].mean()
        avg_w = sub[sub['pnl'] > 0]['pnl'].mean() if (sub['pnl'] > 0).any() else 0
        avg_l = sub[sub['pnl'] <= 0]['pnl'].mean() if (sub['pnl'] <= 0).any() else 0
        print(f"  {r:<12} {n:>6} ${pnl:>9.0f} {wr:>5.1f}% ${avg:>7.2f} ${avg_w:>7.2f} ${avg_l:>7.2f}")

    # By Trend Regime
    print("\n  --- By Trend Regime ---")
    print(f"  {'Regime':<12} {'N':>6} {'PnL':>10} {'WR%':>6} {'AvgPnL':>8} {'BUY_PnL':>10} {'SELL_PnL':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*6} {'-'*8} {'-'*10} {'-'*10}")
    for r in ['StrongBull', 'Bull', 'Sideways', 'Bear', 'StrongBear']:
        sub = df[df['trend'] == r]
        if len(sub) == 0: continue
        n = len(sub)
        pnl = sub['pnl'].sum()
        wr = (sub['pnl'] > 0).mean() * 100
        avg = sub['pnl'].mean()
        buy_pnl = sub[sub['direction'] == 'BUY']['pnl'].sum()
        sell_pnl = sub[sub['direction'] == 'SELL']['pnl'].sum()
        print(f"  {r:<12} {n:>6} ${pnl:>9.0f} {wr:>5.1f}% ${avg:>7.2f} ${buy_pnl:>9.0f} ${sell_pnl:>9.0f}")

    # By ADX Regime
    print("\n  --- By ADX Regime ---")
    print(f"  {'Regime':<12} {'N':>6} {'PnL':>10} {'WR%':>6} {'AvgPnL':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*6} {'-'*8}")
    for r in ['Choppy', 'Moderate', 'Trending']:
        sub = df[df['adx'] == r]
        if len(sub) == 0: continue
        n = len(sub)
        pnl = sub['pnl'].sum()
        wr = (sub['pnl'] > 0).mean() * 100
        avg = sub['pnl'].mean()
        print(f"  {r:<12} {n:>6} ${pnl:>9.0f} {wr:>5.1f}% ${avg:>7.2f}")

    # By Exit Reason per Regime
    print("\n  --- Exit Reason by Vol Regime ---")
    for vol_r in ['LowVol', 'NormVol', 'HighVol']:
        sub = df[df['vol'] == vol_r]
        if len(sub) == 0: continue
        print(f"\n  [{vol_r}]")
        exit_groups = sub.groupby('exit_reason').agg(
            N=('pnl', 'count'), PnL=('pnl', 'sum'),
            WR=('pnl', lambda x: (x > 0).mean() * 100),
            AvgPnL=('pnl', 'mean'),
        )
        for er, row in exit_groups.iterrows():
            print(f"    {er:<20} N={int(row['N']):>5} PnL=${row['PnL']:>8.0f} "
                  f"WR={row['WR']:>5.1f}% Avg=${row['AvgPnL']:>6.2f}")

    # Combined: Vol × Trend
    print("\n  --- Vol × Trend Matrix (PnL) ---")
    print(f"  {'':>12} {'StrongBull':>12} {'Bull':>12} {'Sideways':>12} {'Bear':>12} {'StrongBear':>12}")
    for vol_r in ['LowVol', 'NormVol', 'HighVol']:
        row_str = f"  {vol_r:<12}"
        for trend_r in ['StrongBull', 'Bull', 'Sideways', 'Bear', 'StrongBear']:
            sub = df[(df['vol'] == vol_r) & (df['trend'] == trend_r)]
            if len(sub) == 0:
                row_str += f" {'---':>12}"
            else:
                row_str += f" ${sub['pnl'].sum():>10.0f}"
        print(row_str)

    print(f"\n  --- Vol × Trend Matrix (N trades) ---")
    print(f"  {'':>12} {'StrongBull':>12} {'Bull':>12} {'Sideways':>12} {'Bear':>12} {'StrongBear':>12}")
    for vol_r in ['LowVol', 'NormVol', 'HighVol']:
        row_str = f"  {vol_r:<12}"
        for trend_r in ['StrongBull', 'Bull', 'Sideways', 'Bear', 'StrongBear']:
            sub = df[(df['vol'] == vol_r) & (df['trend'] == trend_r)]
            row_str += f" {len(sub):>12}"
        print(row_str)

    return df


# ═════════════════════════════════════════════════════════════
# Phase 3: Regime 转换时的策略表现
# ═════════════════════════════════════════════════════════════

def phase_3(regime_df, trades):
    print("\n" + "=" * 90)
    print("  PHASE 3: Performance During Regime Transitions")
    print("=" * 90)

    # Detect regime transitions
    regime_df = regime_df.copy()
    regime_df['prev_trend'] = regime_df['trend_regime'].shift(1)
    regime_df['prev_vol'] = regime_df['vol_regime'].shift(1)
    regime_df['trend_change'] = regime_df['trend_regime'] != regime_df['prev_trend']
    regime_df['vol_change'] = regime_df['vol_regime'] != regime_df['prev_vol']

    transition_dates = regime_df[regime_df['trend_change']].index.tolist()
    print(f"\n  Total trend regime transitions: {len(transition_dates)}")

    # For each transition, look at trades in ±5 days window
    windows = [5, 10, 20]
    for w in windows:
        transition_trades = []
        non_transition_trades = []
        transition_zones = set()
        for td in transition_dates:
            for delta in range(-w, w+1):
                transition_zones.add(td + pd.Timedelta(days=delta))

        for t in trades:
            et = pd.Timestamp(t.entry_time)
            if et.tzinfo is not None:
                et = et.tz_localize(None)
            d = et.normalize()
            if d in transition_zones:
                transition_trades.append(t)
            else:
                non_transition_trades.append(t)

        t_pnl = [t.pnl for t in transition_trades]
        nt_pnl = [t.pnl for t in non_transition_trades]

        if t_pnl and nt_pnl:
            t_arr = np.array(t_pnl)
            nt_arr = np.array(nt_pnl)
            print(f"\n  --- Window ±{w} days around transitions ---")
            print(f"  Transition:     N={len(t_arr):>6}, PnL=${t_arr.sum():>9.0f}, "
                  f"WR={np.mean(t_arr>0)*100:>5.1f}%, Avg=${t_arr.mean():>6.2f}")
            print(f"  Non-transition: N={len(nt_arr):>6}, PnL=${nt_arr.sum():>9.0f}, "
                  f"WR={np.mean(nt_arr>0)*100:>5.1f}%, Avg=${nt_arr.mean():>6.2f}")

    # Specific transition types
    print("\n  --- Performance by Transition Type ---")
    print(f"  {'From→To':<25} {'N_trans':>8} {'N_trades':>8} {'PnL':>10} {'WR%':>6}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*6}")

    trans_types = defaultdict(list)
    for td in transition_dates:
        row = regime_df.loc[td]
        key = f"{row['prev_trend']}→{row['trend_regime']}"
        for t in trades:
            et = pd.Timestamp(t.entry_time)
            if et.tzinfo is not None:
                et = et.tz_localize(None)
            d = et.normalize()
            if abs((d - td).days) <= 5:
                trans_types[key].append(t.pnl)

    for key in sorted(trans_types.keys()):
        pnls = trans_types[key]
        if not pnls: continue
        arr = np.array(pnls)
        n_trans = sum(1 for td in transition_dates
                      if regime_df.loc[td, 'prev_trend'] + '→' + regime_df.loc[td, 'trend_regime'] == key)
        print(f"  {key:<25} {n_trans:>8} {len(arr):>8} ${arr.sum():>9.0f} "
              f"{np.mean(arr>0)*100:>5.1f}%")


# ═════════════════════════════════════════════════════════════
# Phase 4: Regime-Adaptive 参数测试
# ═════════════════════════════════════════════════════════════

def phase_4():
    print("\n" + "=" * 90)
    print("  PHASE 4: Regime-Adaptive Parameter Testing")
    print("=" * 90)

    # Test different trail parameters for different regimes
    configs = [
        ("P4_Baseline",        L7_MH8),
        ("P4_TightLow",        {**L7_MH8, 'regime_config': {
            'low': {'trail_act': 0.20, 'trail_dist': 0.04},
            'normal': {'trail_act': 0.28, 'trail_dist': 0.06},
            'high': {'trail_act': 0.12, 'trail_dist': 0.02},
        }}),
        ("P4_WideLow",         {**L7_MH8, 'regime_config': {
            'low': {'trail_act': 0.60, 'trail_dist': 0.15},
            'normal': {'trail_act': 0.28, 'trail_dist': 0.06},
            'high': {'trail_act': 0.12, 'trail_dist': 0.02},
        }}),
        ("P4_TightHigh",       {**L7_MH8, 'regime_config': {
            'low': {'trail_act': 0.40, 'trail_dist': 0.10},
            'normal': {'trail_act': 0.28, 'trail_dist': 0.06},
            'high': {'trail_act': 0.06, 'trail_dist': 0.01},
        }}),
        ("P4_WideHigh",        {**L7_MH8, 'regime_config': {
            'low': {'trail_act': 0.40, 'trail_dist': 0.10},
            'normal': {'trail_act': 0.28, 'trail_dist': 0.06},
            'high': {'trail_act': 0.20, 'trail_dist': 0.04},
        }}),
        ("P4_TightAll",        {**L7_MH8, 'regime_config': {
            'low': {'trail_act': 0.20, 'trail_dist': 0.04},
            'normal': {'trail_act': 0.14, 'trail_dist': 0.03},
            'high': {'trail_act': 0.06, 'trail_dist': 0.01},
        }}),
        ("P4_WideAll",         {**L7_MH8, 'regime_config': {
            'low': {'trail_act': 0.60, 'trail_dist': 0.15},
            'normal': {'trail_act': 0.40, 'trail_dist': 0.10},
            'high': {'trail_act': 0.20, 'trail_dist': 0.04},
        }}),
        # ADX threshold variations
        ("P4_ADX15",           {**L7_MH8, 'keltner_adx_threshold': 15}),
        ("P4_ADX20",           {**L7_MH8, 'keltner_adx_threshold': 20}),
        ("P4_ADX25",           {**L7_MH8, 'keltner_adx_threshold': 25}),
        # MaxHold variations
        ("P4_MH4",             {**L7_MH8, 'keltner_max_hold_m15': 4}),
        ("P4_MH12",            {**L7_MH8, 'keltner_max_hold_m15': 12}),
        ("P4_MH20",            {**L7_MH8, 'keltner_max_hold_m15': 20}),
    ]

    tasks = [(name, kw, 0.50) for name, kw in configs]

    t0 = time.time()
    with mp.Pool(MAX_WORKERS) as pool:
        results = pool.map(_run_one, tasks)
    print(f"  Done in {time.time()-t0:.0f}s")

    print(f"\n  {'Config':<20} {'N':>6} {'PnL':>10} {'CorrSh':>8} {'WR%':>6} {'MaxDD':>8} {'AvgPnL':>8}")
    print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*8} {'-'*6} {'-'*8} {'-'*8}")
    for r in results:
        avg = r['total_pnl'] / r['n'] if r['n'] > 0 else 0
        print(f"  {r['label']:<20} {r['n']:>6} ${r['total_pnl']:>9.0f} {r['corr_sharpe']:>8.2f} "
              f"{r['win_rate']:>5.1f}% ${r['max_dd']:>7.0f} ${avg:>7.2f}")


# ═════════════════════════════════════════════════════════════
# Phase 5: 2024-2026 近期 Regime 诊断
# ═════════════════════════════════════════════════════════════

def phase_5(regime_df, trades):
    print("\n" + "=" * 90)
    print("  PHASE 5: Recent Period (2024-2026) Regime Diagnosis")
    print("=" * 90)

    # Compare recent periods vs historical
    periods = {
        '2015-2017 (Low Vol)':   ('2015-01-01', '2017-12-31'),
        '2018-2019 (Choppy)':    ('2018-01-01', '2019-12-31'),
        '2020 (COVID Bull)':     ('2020-01-01', '2020-12-31'),
        '2021-2022 (Mixed)':     ('2021-01-01', '2022-12-31'),
        '2023 (Recovery)':       ('2023-01-01', '2023-12-31'),
        '2024 (Bull Run)':       ('2024-01-01', '2024-12-31'),
        '2025 Q1 (Current)':     ('2025-01-01', '2025-06-30'),
        '2025-10 to Now':        ('2025-10-01', '2026-12-31'),
    }

    print(f"\n  {'Period':<25} {'N':>6} {'PnL':>10} {'WR%':>6} {'AvgPnL':>8} "
          f"{'AvgWin':>8} {'AvgLoss':>8} {'RR':>5}")
    print(f"  {'-'*25} {'-'*6} {'-'*10} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")

    for pname, (start, end) in periods.items():
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        period_trades = []
        for t in trades:
            et = pd.Timestamp(t.entry_time)
            if et.tzinfo is not None:
                et = et.tz_localize(None)
            if start_dt <= et <= end_dt:
                period_trades.append(t)

        if not period_trades:
            print(f"  {pname:<25} {'no trades':>6}")
            continue

        n = len(period_trades)
        pnl = sum(t.pnl for t in period_trades)
        wins = [t.pnl for t in period_trades if t.pnl > 0]
        losses = [t.pnl for t in period_trades if t.pnl <= 0]
        wr = len(wins) / n * 100
        avg = pnl / n
        avg_w = np.mean(wins) if wins else 0
        avg_l = np.mean(losses) if losses else 0
        rr = abs(avg_w / avg_l) if avg_l != 0 else 999
        print(f"  {pname:<25} {n:>6} ${pnl:>9.0f} {wr:>5.1f}% ${avg:>7.2f} "
              f"${avg_w:>7.2f} ${avg_l:>7.2f} {rr:>4.2f}")

    # Monthly breakdown for 2024-2026
    print("\n  --- Monthly Breakdown (2024-2026) ---")
    print(f"  {'Month':>8} {'N':>5} {'PnL':>9} {'WR%':>6} {'AvgPnL':>8}")
    print(f"  {'-'*8} {'-'*5} {'-'*9} {'-'*6} {'-'*8}")

    for year in [2024, 2025, 2026]:
        for month in range(1, 13):
            start_dt = pd.Timestamp(f'{year}-{month:02d}-01')
            if month == 12:
                end_dt = pd.Timestamp(f'{year+1}-01-01')
            else:
                end_dt = pd.Timestamp(f'{year}-{month+1:02d}-01')

            period_trades = []
            for t in trades:
                et = pd.Timestamp(t.entry_time)
                if et.tzinfo is not None:
                    et = et.tz_localize(None)
                if start_dt <= et < end_dt:
                    period_trades.append(t)

            if not period_trades:
                continue

            n = len(period_trades)
            pnl = sum(t.pnl for t in period_trades)
            wr = sum(1 for t in period_trades if t.pnl > 0) / n * 100
            avg = pnl / n
            print(f"  {year}-{month:02d} {n:>5} ${pnl:>8.0f} {wr:>5.1f}% ${avg:>7.2f}")

    # Regime composition for 2024-2026 vs historical
    print("\n  --- Regime Composition: Recent vs Historical ---")
    recent = regime_df[regime_df.index >= '2024-01-01']
    historical = regime_df[regime_df.index < '2024-01-01']

    for regime_col in ['vol_regime', 'trend_regime', 'adx_regime']:
        print(f"\n  [{regime_col}]")
        all_vals = regime_df[regime_col].unique()
        print(f"  {'Regime':<15} {'Historical':>12} {'Recent':>12} {'Delta':>8}")
        for val in sorted(all_vals):
            h_pct = (historical[regime_col] == val).mean() * 100
            r_pct = (recent[regime_col] == val).mean() * 100
            delta = r_pct - h_pct
            print(f"  {val:<15} {h_pct:>11.1f}% {r_pct:>11.1f}% {delta:>+7.1f}%")


# ═════════════════════════════════════════════════════════════
# Phase 6: Regime Filter K-Fold
# ═════════════════════════════════════════════════════════════

def phase_6():
    print("\n" + "=" * 90)
    print("  PHASE 6: Regime-Aware K-Fold Validation")
    print("=" * 90)

    from backtest.runner import run_kfold

    def _run_kfold_one(args):
        label, base_kwargs, spread, cap = args
        data = DataBundle.load_default()
        kw = {**base_kwargs, 'spread_cost': spread}
        folds = run_kfold(data, kw, n_folds=6)
        results = []
        for f in folds:
            trades_f = f.get('_trades', [])
            if cap < 999 and trades_f:
                trades_f = apply_max_loss_cap(trades_f, cap)
            csh = corrected_sharpe(trades_f)
            pnl = sum(t.pnl for t in trades_f) if trades_f else f['total_pnl']
            results.append({
                'fold': f['label'], 'n': f['n'], 'orig_sharpe': f['sharpe'],
                'corr_sharpe': csh, 'pnl': pnl, 'win_rate': f['win_rate'],
            })
        return {'label': label, 'folds': results}

    configs = [
        ("KF_L7_Base",       L7_MH8,                                                   999),
        ("KF_TightHigh",     {**L7_MH8, 'regime_config': {
            'low': {'trail_act': 0.40, 'trail_dist': 0.10},
            'normal': {'trail_act': 0.28, 'trail_dist': 0.06},
            'high': {'trail_act': 0.06, 'trail_dist': 0.01},
        }},                                                                              999),
        ("KF_ADX20",         {**L7_MH8, 'keltner_adx_threshold': 20},                  999),
        ("KF_ADX20+Cap30",   {**L7_MH8, 'keltner_adx_threshold': 20},                   30),
        ("KF_KCBW5+ADX20",   {**L7_MH8, 'kc_bw_filter_bars': 5,
                               'keltner_adx_threshold': 20},                             999),
    ]

    tasks = [(c[0], c[1], 0.50, c[2]) for c in configs]

    t0 = time.time()
    with mp.Pool(MAX_WORKERS) as pool:
        results = pool.map(_run_kfold_one, tasks)
    print(f"  Done in {time.time()-t0:.0f}s")

    for kr in results:
        print(f"\n  [{kr['label']}]")
        print(f"  {'Fold':<8} {'N':>6} {'OrigSh':>8} {'CorrSh':>8} {'PnL':>10} {'WR%':>6}")
        sharpes = []
        for f in kr['folds']:
            sharpes.append(f['corr_sharpe'])
            print(f"  {f['fold']:<8} {f['n']:>6} {f['orig_sharpe']:>8.2f} {f['corr_sharpe']:>8.2f} "
                  f"${f['pnl']:>9.0f} {f['win_rate']:>5.1f}%")
        pos_count = sum(1 for s in sharpes if s > 0)
        all_pos = pos_count == 6
        print(f"  Mean={np.mean(sharpes):.2f}, Std={np.std(sharpes):.2f}, "
              f"Positive={pos_count}/6, PASS={'YES' if all_pos else 'NO'}")

    # Summary
    print(f"\n  --- K-Fold Summary ---")
    print(f"  {'Config':<25} {'Mean':>6} {'Std':>6} {'Min':>6} {'P/6':>4} {'PASS':>5}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*4} {'-'*5}")
    for kr in results:
        sharpes = [f['corr_sharpe'] for f in kr['folds']]
        pos = sum(1 for s in sharpes if s > 0)
        print(f"  {kr['label']:<25} {np.mean(sharpes):>6.2f} {np.std(sharpes):>6.2f} "
              f"{min(sharpes):>6.2f} {pos:>3}/6 {'YES' if pos==6 else 'NO':>5}")


# ═════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════

def main():
    out_path = OUT_DIR / "R40_regime_study_output.txt"
    f_out = open(out_path, 'w', encoding='utf-8')
    tee = Tee(sys.stdout, f_out)
    sys.stdout = tee

    print("=" * 90)
    print("  R40: MARKET REGIME DEEP STUDY")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  CPU cores: {mp.cpu_count()}, Workers: {MAX_WORKERS}")
    print("=" * 90)

    t0 = time.time()

    # Load data once for Phase 1/2/3/5
    data = DataBundle.load_default()
    print(f"\n  Data: M15={len(data.m15_df)}, H1={len(data.h1_df)}")

    # Run baseline to get trades
    print("\n  Running L7 baseline (spread=$0.50)...")
    r = run_variant(data, "L7_baseline", verbose=True, **L7_MH8, spread_cost=0.50)
    trades = r['_trades']
    print(f"  Baseline: N={r['n']}, Sharpe={r['sharpe']:.2f}, PnL=${r['total_pnl']:.0f}")

    # Phase 1: Regime identification
    regime_df = phase_1(data.h1_df, data.m15_df)
    print(f"\n  [Checkpoint] Phase 1 done, elapsed: {(time.time()-t0)/60:.1f} min")

    # Phase 2: L7 by regime
    trade_df = phase_2(regime_df, trades)
    print(f"\n  [Checkpoint] Phase 2 done, elapsed: {(time.time()-t0)/60:.1f} min")

    # Phase 3: Transition analysis
    phase_3(regime_df, trades)
    print(f"\n  [Checkpoint] Phase 3 done, elapsed: {(time.time()-t0)/60:.1f} min")

    # Phase 4: Regime-adaptive params (parallel)
    phase_4()
    print(f"\n  [Checkpoint] Phase 4 done, elapsed: {(time.time()-t0)/60:.1f} min")

    # Phase 5: Recent diagnosis
    phase_5(regime_df, trades)
    print(f"\n  [Checkpoint] Phase 5 done, elapsed: {(time.time()-t0)/60:.1f} min")

    # Phase 6: K-Fold
    phase_6()
    print(f"\n  [Checkpoint] Phase 6 done, elapsed: {(time.time()-t0)/60:.1f} min")

    total = time.time() - t0
    print(f"\n\n{'=' * 90}")
    print(f"  R40 REGIME STUDY COMPLETE")
    print(f"  Total runtime: {total/60:.1f} minutes ({total/3600:.1f} hours)")
    print(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Results: {out_path}")
    print(f"{'=' * 90}")

    sys.stdout = sys.__stdout__
    f_out.close()
    print(f"Done. Output: {out_path}")


if __name__ == "__main__":
    main()
