"""
R32: New Directions — First Batch
===================================
A: Multi-TF Confirmation (M15 signal + H1/H4 KC direction alignment)
B: Volatility Term Structure (ATR fast/slow ratio & its derivative)
C: Alternative Trend Indicators as independent strategies (SuperTrend, Ichimoku, PSAR)
D: ML Meta-Model (XGBoost to predict L7 trade quality, Walk-Forward)
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS

OUT_DIR = Path("results/round32_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            try: f.write(data)
            except: pass
            f.flush()
    def flush(self):
        for f in self.files: f.flush()


L7_MH8 = {
    **LIVE_PARITY_KWARGS,
    'time_adaptive_trail': {'start': 2, 'decay': 0.75, 'floor': 0.003},
    'min_entry_gap_hours': 1.0,
    'keltner_max_hold_m15': 8,
}


def compute_adx(df, period=14):
    high = df['High']; low = df['Low']; close = df['Close']
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.DataFrame({'hl': high-low, 'hc': (high-close.shift(1)).abs(),
                        'lc': (low-close.shift(1)).abs()}).max(axis=1)
    atr_s = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_s)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_s)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(period).mean()


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


def _at(trades, equity, pos, close, exit_time, reason, bar_idx, pnl):
    trades.append({'dir': pos['dir'], 'entry': pos['entry'],
                   'entry_time': pos['time'], 'exit_time': exit_time,
                   'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']})
    equity.append(equity[-1] + pnl)


def _stats_from_trades(trades, label):
    if not trades:
        return {'label': label, 'n': 0, 'sharpe': 0, 'total_pnl': 0, 'win_rate': 0, 'max_dd': 0,
                '_trades': [], '_daily_pnl': {}}
    pnls = [t['pnl'] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    eq = np.cumsum(pnls); dd = (np.maximum.accumulate(eq + 2000) - (eq + 2000)).max()
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily.setdefault(d, 0); daily[d] += t['pnl']
    da = np.array(list(daily.values()))
    sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0
    return {'label': label, 'n': len(trades), 'sharpe': sh, 'total_pnl': sum(pnls),
            'win_rate': wins/len(trades)*100, 'max_dd': dd, '_trades': trades, '_daily_pnl': daily}


def make_daily(trades):
    daily = {}
    for t in trades:
        exit_t = t.exit_time if hasattr(t, 'exit_time') else t['exit_time']
        pnl = t.pnl if hasattr(t, 'pnl') else t['pnl']
        d = pd.Timestamp(exit_t).date()
        daily.setdefault(d, 0); daily[d] += pnl
    return pd.Series(daily).sort_index()


def backtest_generic(df, label, signal_fn, sl_atr=3.5, tp_atr=8.0,
                     trail_act_atr=0.28, trail_dist_atr=0.06, max_hold=50,
                     spread=0.30, lot=0.03):
    """Generic backtester: signal_fn(df, i) -> 'BUY'/'SELL'/None."""
    df = df.dropna()
    trades = []; pos = None; equity = [2000.0]
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    atr = df['ATR'].values
    times = df.index; n = len(df); last_exit = -999

    for i in range(1, n):
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (high[i] - pos['entry'] - spread) * lot * 100
                pnl_l = (low[i] - pos['entry'] - spread) * lot * 100
                pnl_c = (close[i] - pos['entry'] - spread) * lot * 100
            else:
                pnl_h = (pos['entry'] - low[i] - spread) * lot * 100
                pnl_l = (pos['entry'] - high[i] - spread) * lot * 100
                pnl_c = (pos['entry'] - close[i] - spread) * lot * 100
            tp_val = tp_atr * pos['atr'] * lot * 100
            sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                _at(trades, equity, pos, close[i], times[i], "TP", i, tp_val); exited = True
            elif pnl_l <= -sl_val:
                _at(trades, equity, pos, close[i], times[i], "SL", i, -sl_val); exited = True
            else:
                ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and high[i] - pos['entry'] >= ad:
                    ts = high[i] - td
                    if low[i] <= ts:
                        _at(trades, equity, pos, close[i], times[i], "Trail", i,
                            (ts - pos['entry'] - spread) * lot * 100); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - low[i] >= ad:
                    ts = low[i] + td
                    if high[i] >= ts:
                        _at(trades, equity, pos, close[i], times[i], "Trail", i,
                            (pos['entry'] - ts - spread) * lot * 100); exited = True
                if not exited and held >= max_hold:
                    _at(trades, equity, pos, close[i], times[i], "Timeout", i, pnl_c); exited = True
            if exited: pos = None; last_exit = i; continue
        if pos is not None: continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        sig = signal_fn(df, i)
        if sig == 'BUY':
            pos = {'dir': 'BUY', 'entry': close[i] + spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif sig == 'SELL':
            pos = {'dir': 'SELL', 'entry': close[i] - spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}

    return _stats_from_trades(trades, label)


# ═══════════════════════════════════════════════════════════════
# Phase A: Multi-TF Confirmation
# ═══════════════════════════════════════════════════════════════

def run_phase_A(data):
    """Test L7(MH=8) with H1/H4 KC direction filter."""
    print("\n" + "=" * 80)
    print("Phase A: Multi-TF Confirmation (M15 + H1/H4 KC Direction)")
    print("=" * 80)

    base = run_variant(data, "L7MH8_base", verbose=False, **L7_MH8)
    trades_base = base['_trades']
    print(f"\n  Baseline: N={len(trades_base)}, Sharpe={base['sharpe']:.2f}, PnL=${base['total_pnl']:.0f}")

    h1_df = data.h1_df.copy()

    # Build H1 KC direction lookup
    for ema_p, mult in [(25, 1.2), (20, 2.0)]:
        h1_kc = add_kc(h1_df, ema_period=ema_p, mult=mult)
        h1_kc['kc_dir'] = 'NEUTRAL'
        h1_kc.loc[h1_kc['Close'] > h1_kc['KC_upper'], 'kc_dir'] = 'BULL'
        h1_kc.loc[h1_kc['Close'] < h1_kc['KC_lower'], 'kc_dir'] = 'BEAR'

        for filter_name, keep_fn in [
            ("Same dir only", lambda td, kd: (td == 'BUY' and kd == 'BULL') or (td == 'SELL' and kd == 'BEAR')),
            ("Opposite only", lambda td, kd: (td == 'BUY' and kd == 'BEAR') or (td == 'SELL' and kd == 'BULL')),
            ("Not neutral", lambda td, kd: kd != 'NEUTRAL'),
        ]:
            kept = []; skipped = 0
            for t in trades_base:
                et = t.entry_time if hasattr(t, 'entry_time') else t['entry_time']
                td = t.direction if hasattr(t, 'direction') else t['dir']
                et_ts = pd.Timestamp(et)
                h1_mask = h1_kc.index <= et_ts
                if not h1_mask.any():
                    skipped += 1; continue
                kc_d = h1_kc.loc[h1_kc.index[h1_mask][-1], 'kc_dir']
                if keep_fn(td, kc_d):
                    kept.append(t)
                else:
                    skipped += 1

            daily = {}
            for t in kept:
                exit_t = t.exit_time if hasattr(t, 'exit_time') else t['exit_time']
                pnl = t.pnl if hasattr(t, 'pnl') else t['pnl']
                d = pd.Timestamp(exit_t).date()
                daily.setdefault(d, 0); daily[d] += pnl
            da = np.array(list(daily.values())) if daily else np.array([0])
            sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0
            pnl_tot = da.sum()

            print(f"  H1 KC(EMA{ema_p}/M{mult}) {filter_name:>18}: "
                  f"N={len(kept):>5}, Sharpe={sh:>6.2f}, PnL=${pnl_tot:>8.0f}, "
                  f"Skipped={skipped}")

    # H4 KC direction
    h4_df = h1_df.resample('4h').agg({'Open':'first','High':'max','Low':'min',
                                       'Close':'last','Volume':'sum'}).dropna()
    h4_kc = add_kc(h4_df, ema_period=20, mult=2.0)
    h4_kc['kc_dir'] = 'NEUTRAL'
    h4_kc.loc[h4_kc['Close'] > h4_kc['KC_upper'], 'kc_dir'] = 'BULL'
    h4_kc.loc[h4_kc['Close'] < h4_kc['KC_lower'], 'kc_dir'] = 'BEAR'

    print(f"\n  --- H4 KC(EMA20/M2.0) ---")
    for filter_name, keep_fn in [
        ("Same dir only", lambda td, kd: (td == 'BUY' and kd == 'BULL') or (td == 'SELL' and kd == 'BEAR')),
        ("Opposite only", lambda td, kd: (td == 'BUY' and kd == 'BEAR') or (td == 'SELL' and kd == 'BULL')),
    ]:
        kept = []; skipped = 0
        for t in trades_base:
            et = t.entry_time if hasattr(t, 'entry_time') else t['entry_time']
            td = t.direction if hasattr(t, 'direction') else t['dir']
            et_ts = pd.Timestamp(et)
            h4_mask = h4_kc.index <= et_ts
            if not h4_mask.any(): skipped += 1; continue
            kc_d = h4_kc.loc[h4_kc.index[h4_mask][-1], 'kc_dir']
            if keep_fn(td, kc_d): kept.append(t)
            else: skipped += 1

        daily = {}
        for t in kept:
            exit_t = t.exit_time if hasattr(t, 'exit_time') else t['exit_time']
            pnl = t.pnl if hasattr(t, 'pnl') else t['pnl']
            d = pd.Timestamp(exit_t).date()
            daily.setdefault(d, 0); daily[d] += pnl
        da = np.array(list(daily.values())) if daily else np.array([0])
        sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0
        print(f"  H4 KC {filter_name:>18}: N={len(kept):>5}, Sharpe={sh:>6.2f}, "
              f"PnL=${da.sum():>8.0f}, Skipped={skipped}")


# ═══════════════════════════════════════════════════════════════
# Phase B: Volatility Term Structure
# ═══════════════════════════════════════════════════════════════

def run_phase_B(data):
    """ATR term structure and its derivative as signal."""
    print("\n" + "=" * 80)
    print("Phase B: Volatility Term Structure Strategy")
    print("=" * 80)

    h1 = data.h1_df.copy()
    tr = pd.DataFrame({
        'hl': h1['High'] - h1['Low'],
        'hc': (h1['High'] - h1['Close'].shift(1)).abs(),
        'lc': (h1['Low'] - h1['Close'].shift(1)).abs(),
    }).max(axis=1)

    # B1: IC analysis of ATR ratio and its derivative
    print(f"\n  --- B1: ATR Ratio IC Analysis ---")
    print(f"  {'Fast':>5} {'Slow':>5} {'IC_1h':>8} {'IC_4h':>8} {'IC_8h':>8} {'dRatio_IC_1h':>12} {'dRatio_IC_4h':>12}")

    for fast_p in [3, 5, 7, 10]:
        for slow_p in [30, 50, 70, 100]:
            atr_fast = tr.rolling(fast_p).mean()
            atr_slow = tr.rolling(slow_p).mean()
            ratio = (atr_fast / atr_slow).replace([np.inf, -np.inf], np.nan)
            d_ratio = ratio.diff()

            ret_1 = h1['Close'].pct_change(1).shift(-1)
            ret_4 = h1['Close'].pct_change(4).shift(-4)
            ret_8 = h1['Close'].pct_change(8).shift(-8)

            mask = ratio.notna() & ret_1.notna()
            ic1 = ratio[mask].corr(ret_1[mask]) if mask.sum() > 100 else 0
            ic4 = ratio[mask & ret_4.notna()].corr(ret_4[mask & ret_4.notna()]) if (mask & ret_4.notna()).sum() > 100 else 0
            ic8 = ratio[mask & ret_8.notna()].corr(ret_8[mask & ret_8.notna()]) if (mask & ret_8.notna()).sum() > 100 else 0

            mask_d = d_ratio.notna() & ret_1.notna()
            dic1 = d_ratio[mask_d].corr(ret_1[mask_d]) if mask_d.sum() > 100 else 0
            dic4 = d_ratio[mask_d & ret_4.notna()].corr(ret_4[mask_d & ret_4.notna()]) if (mask_d & ret_4.notna()).sum() > 100 else 0

            print(f"  {fast_p:>5} {slow_p:>5} {ic1:>+8.4f} {ic4:>+8.4f} {ic8:>+8.4f} "
                  f"{dic1:>+12.4f} {dic4:>+12.4f}")

    # B2: Vol expansion/contraction as L7 filter
    print(f"\n  --- B2: Vol Term Structure as L7 Filter ---")
    base = run_variant(data, "L7MH8_volB", verbose=False, **L7_MH8)
    trades_base = base['_trades']
    sh_base = base['sharpe']
    print(f"  Baseline: N={len(trades_base)}, Sharpe={sh_base:.2f}")

    atr_fast = tr.rolling(5).mean()
    atr_slow = tr.rolling(50).mean()
    vol_ratio = (atr_fast / atr_slow).replace([np.inf, -np.inf], np.nan)

    for thresh_name, keep_fn in [
        ("Expanding (>1.0)", lambda r: r > 1.0),
        ("Expanding (>1.2)", lambda r: r > 1.2),
        ("Contracting (<1.0)", lambda r: r < 1.0),
        ("Contracting (<0.8)", lambda r: r < 0.8),
        ("Neutral (0.8-1.2)", lambda r: 0.8 <= r <= 1.2),
    ]:
        kept = []; skipped = 0
        for t in trades_base:
            et = t.entry_time if hasattr(t, 'entry_time') else t['entry_time']
            et_ts = pd.Timestamp(et)
            mask = vol_ratio.index <= et_ts
            if not mask.any() or pd.isna(vol_ratio[mask].iloc[-1]):
                skipped += 1; continue
            r = vol_ratio[mask].iloc[-1]
            if keep_fn(r): kept.append(t)
            else: skipped += 1

        daily = {}
        for t in kept:
            exit_t = t.exit_time if hasattr(t, 'exit_time') else t['exit_time']
            pnl = t.pnl if hasattr(t, 'pnl') else t['pnl']
            d = pd.Timestamp(exit_t).date()
            daily.setdefault(d, 0); daily[d] += pnl
        da = np.array(list(daily.values())) if daily else np.array([0])
        sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0
        print(f"  {thresh_name:>25}: N={len(kept):>5}, Sharpe={sh:>6.2f}, "
              f"PnL=${da.sum():>8.0f}, delta={sh-sh_base:>+6.2f}")

    # B3: Vol term structure as independent strategy
    print(f"\n  --- B3: Vol Term Structure as Independent Strategy ---")
    h1c = add_kc(h1, ema_period=20, mult=2.0)
    h1c['vol_ratio'] = vol_ratio

    for fast_p, slow_p in [(5, 50), (7, 70), (10, 100)]:
        af = tr.rolling(fast_p).mean()
        aslow = tr.rolling(slow_p).mean()
        vr = (af / aslow).replace([np.inf, -np.inf], np.nan)
        h1c['vr'] = vr
        h1c['vr_prev'] = vr.shift(1)

        def signal_vol_expand(df, i):
            if i < 1: return None
            vr_now = df['vr'].iloc[i]; vr_prev = df['vr_prev'].iloc[i]
            if pd.isna(vr_now) or pd.isna(vr_prev): return None
            close_now = df['Close'].iloc[i]; close_prev = df['Close'].iloc[i-1]
            if vr_now > 1.0 and vr_prev <= 1.0:
                return 'BUY' if close_now > close_prev else 'SELL'
            return None

        r = backtest_generic(h1c, f"VolTS_{fast_p}/{slow_p}", signal_vol_expand,
                             sl_atr=3.5, tp_atr=8.0, max_hold=30)
        print(f"  ATR({fast_p}/{slow_p}) expand cross: N={r['n']:>5}, Sharpe={r['sharpe']:>6.2f}, "
              f"PnL=${r['total_pnl']:>8.0f}, WR={r['win_rate']:>5.1f}%")


# ═══════════════════════════════════════════════════════════════
# Phase C: Alternative Trend Indicators
# ═══════════════════════════════════════════════════════════════

def compute_supertrend(df, period=10, factor=3.0):
    """Compute SuperTrend indicator."""
    hl2 = (df['High'] + df['Low']) / 2
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    atr = tr.rolling(period).mean()

    upper_basic = hl2 + factor * atr
    lower_basic = hl2 - factor * atr

    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(period, len(df)):
        if i == period:
            upper_band.iloc[i] = upper_basic.iloc[i]
            lower_band.iloc[i] = lower_basic.iloc[i]
            direction.iloc[i] = -1 if df['Close'].iloc[i] > upper_band.iloc[i] else 1
        else:
            if lower_basic.iloc[i] > lower_band.iloc[i-1] or df['Close'].iloc[i-1] < lower_band.iloc[i-1]:
                lower_band.iloc[i] = lower_basic.iloc[i]
            else:
                lower_band.iloc[i] = lower_band.iloc[i-1]

            if upper_basic.iloc[i] < upper_band.iloc[i-1] or df['Close'].iloc[i-1] > upper_band.iloc[i-1]:
                upper_band.iloc[i] = upper_basic.iloc[i]
            else:
                upper_band.iloc[i] = upper_band.iloc[i-1]

            if direction.iloc[i-1] == 1:  # was bearish
                if df['Close'].iloc[i] > upper_band.iloc[i]:
                    direction.iloc[i] = -1  # flip to bullish
                else:
                    direction.iloc[i] = 1
            else:  # was bullish
                if df['Close'].iloc[i] < lower_band.iloc[i]:
                    direction.iloc[i] = 1  # flip to bearish
                else:
                    direction.iloc[i] = -1

        supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == -1 else upper_band.iloc[i]

    return supertrend, direction


def compute_ichimoku(df, tenkan=9, kijun=26, senkou_b=52):
    """Compute Ichimoku components."""
    tenkan_sen = (df['High'].rolling(tenkan).max() + df['Low'].rolling(tenkan).min()) / 2
    kijun_sen = (df['High'].rolling(kijun).max() + df['Low'].rolling(kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b_line = ((df['High'].rolling(senkou_b).max() + df['Low'].rolling(senkou_b).min()) / 2).shift(kijun)
    return tenkan_sen, kijun_sen, senkou_a, senkou_b_line


def compute_psar(df, af_start=0.02, af_step=0.02, af_max=0.20):
    """Compute Parabolic SAR."""
    high = df['High'].values; low = df['Low'].values; close = df['Close'].values
    n = len(close)
    psar = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)
    af = af_start
    ep = high[0]
    psar[0] = low[0]
    direction[0] = 1

    for i in range(1, n):
        if direction[i-1] == 1:  # bullish
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            psar[i] = min(psar[i], low[i-1], low[max(0,i-2)])
            if low[i] < psar[i]:
                direction[i] = -1; psar[i] = ep; ep = low[i]; af = af_start
            else:
                direction[i] = 1
                if high[i] > ep: ep = high[i]; af = min(af + af_step, af_max)
        else:  # bearish
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            psar[i] = max(psar[i], high[i-1], high[max(0,i-2)])
            if high[i] > psar[i]:
                direction[i] = 1; psar[i] = ep; ep = high[i]; af = af_start
            else:
                direction[i] = -1
                if low[i] < ep: ep = low[i]; af = min(af + af_step, af_max)

    return pd.Series(psar, index=df.index), pd.Series(direction, index=df.index)


def run_phase_C(h1_df):
    """Test SuperTrend, Ichimoku, PSAR as independent strategies."""
    print("\n" + "=" * 80)
    print("Phase C: Alternative Trend Indicators as Independent Strategies")
    print("=" * 80)

    h1 = h1_df.copy()
    tr = pd.DataFrame({
        'hl': h1['High'] - h1['Low'],
        'hc': (h1['High'] - h1['Close'].shift(1)).abs(),
        'lc': (h1['Low'] - h1['Close'].shift(1)).abs(),
    }).max(axis=1)
    h1['ATR'] = tr.rolling(14).mean()

    # C1: SuperTrend
    print(f"\n  --- C1: SuperTrend ---")
    print(f"  {'Period':>7} {'Factor':>7} {'N':>5} {'Sharpe':>7} {'PnL':>9} {'WR':>5}")
    for period in [7, 10, 14, 20]:
        for factor in [2.0, 2.5, 3.0, 4.0]:
            st, st_dir = compute_supertrend(h1, period=period, factor=factor)
            h1['st_dir'] = st_dir; h1['st_dir_prev'] = st_dir.shift(1)

            def signal_st(df, i):
                if pd.isna(df['st_dir'].iloc[i]) or pd.isna(df['st_dir_prev'].iloc[i]): return None
                if df['st_dir'].iloc[i] == -1 and df['st_dir_prev'].iloc[i] == 1: return 'BUY'
                if df['st_dir'].iloc[i] == 1 and df['st_dir_prev'].iloc[i] == -1: return 'SELL'
                return None

            r = backtest_generic(h1, f"ST_{period}_{factor}", signal_st,
                                 sl_atr=3.5, tp_atr=8.0, max_hold=50)
            print(f"  {period:>7} {factor:>7.1f} {r['n']:>5} {r['sharpe']:>7.2f} "
                  f"${r['total_pnl']:>8.0f} {r['win_rate']:>4.1f}%")

    # C2: Ichimoku
    print(f"\n  --- C2: Ichimoku Cloud ---")
    print(f"  {'Tenkan':>7} {'Kijun':>6} {'N':>5} {'Sharpe':>7} {'PnL':>9} {'WR':>5}")
    for tenkan, kijun in [(9, 26), (7, 22), (12, 30), (20, 52)]:
        ts, ks, sa, sb = compute_ichimoku(h1, tenkan=tenkan, kijun=kijun)
        h1['ts'] = ts; h1['ks'] = ks; h1['sa'] = sa; h1['sb'] = sb
        h1['ts_prev'] = ts.shift(1); h1['ks_prev'] = ks.shift(1)

        def signal_ichi(df, i):
            if any(pd.isna(df[c].iloc[i]) for c in ['ts','ks','sa','sb','ts_prev','ks_prev']): return None
            cloud_top = max(df['sa'].iloc[i], df['sb'].iloc[i])
            cloud_bot = min(df['sa'].iloc[i], df['sb'].iloc[i])
            c = df['Close'].iloc[i]
            if df['ts'].iloc[i] > df['ks'].iloc[i] and df['ts_prev'].iloc[i] <= df['ks_prev'].iloc[i] and c > cloud_top:
                return 'BUY'
            if df['ts'].iloc[i] < df['ks'].iloc[i] and df['ts_prev'].iloc[i] >= df['ks_prev'].iloc[i] and c < cloud_bot:
                return 'SELL'
            return None

        r = backtest_generic(h1, f"Ichi_{tenkan}_{kijun}", signal_ichi,
                             sl_atr=3.5, tp_atr=8.0, max_hold=50)
        print(f"  {tenkan:>7} {kijun:>6} {r['n']:>5} {r['sharpe']:>7.2f} "
              f"${r['total_pnl']:>8.0f} {r['win_rate']:>4.1f}%")

    # C3: Parabolic SAR
    print(f"\n  --- C3: Parabolic SAR ---")
    print(f"  {'AF_start':>9} {'AF_max':>7} {'N':>5} {'Sharpe':>7} {'PnL':>9} {'WR':>5}")
    for af_s in [0.01, 0.02, 0.03]:
        for af_m in [0.10, 0.20, 0.30]:
            psar, psar_dir = compute_psar(h1, af_start=af_s, af_max=af_m)
            h1['psar_dir'] = psar_dir; h1['psar_dir_prev'] = psar_dir.shift(1)

            def signal_psar(df, i):
                if pd.isna(df['psar_dir'].iloc[i]) or pd.isna(df['psar_dir_prev'].iloc[i]): return None
                if df['psar_dir'].iloc[i] == 1 and df['psar_dir_prev'].iloc[i] == -1: return 'BUY'
                if df['psar_dir'].iloc[i] == -1 and df['psar_dir_prev'].iloc[i] == 1: return 'SELL'
                return None

            r = backtest_generic(h1, f"PSAR_{af_s}_{af_m}", signal_psar,
                                 sl_atr=3.5, tp_atr=8.0, max_hold=50)
            print(f"  {af_s:>9.2f} {af_m:>7.2f} {r['n']:>5} {r['sharpe']:>7.2f} "
                  f"${r['total_pnl']:>8.0f} {r['win_rate']:>4.1f}%")

    # C4: Best from each — correlation with L7
    print(f"\n  --- C4: Summary of best from each indicator ---")
    print(f"  (Correlation with L7 to be computed if any Sharpe > 2.0)")


# ═══════════════════════════════════════════════════════════════
# Phase D: ML Meta-Model
# ═══════════════════════════════════════════════════════════════

def run_phase_D(data):
    """XGBoost meta-model to predict L7 trade quality (Walk-Forward)."""
    print("\n" + "=" * 80)
    print("Phase D: ML Meta-Model (XGBoost Walk-Forward)")
    print("=" * 80)

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        HAS_SKLEARN = True
    except ImportError:
        HAS_SKLEARN = False

    try:
        import xgboost as xgb
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False

    if not HAS_SKLEARN and not HAS_XGB:
        print("  WARNING: Neither sklearn nor xgboost available. Using simple heuristic instead.")

    # Get L7(MH=8) trades with features
    s = run_variant(data, "L7MH8_ml", verbose=False, **L7_MH8)
    trades = s['_trades']
    h1_df = data.h1_df

    print(f"  Total trades: {len(trades)}")

    # Build feature matrix
    features = []
    labels = []
    pnl_list = []
    for i, t in enumerate(trades):
        et = t.entry_time if hasattr(t, 'entry_time') else t['entry_time']
        pnl = t.pnl if hasattr(t, 'pnl') else t['pnl']
        et_ts = pd.Timestamp(et)
        h1_mask = h1_df.index <= et_ts
        if not h1_mask.any(): continue

        h1_row = h1_df[h1_mask].iloc[-1]
        atr = h1_row.get('ATR', h1_row.get('atr', np.nan))
        atr_pct = h1_row.get('atr_percentile', 0.5)
        adx_val = h1_row.get('ADX', h1_row.get('adx', np.nan))

        hour = et_ts.hour
        dow = et_ts.dayofweek

        # Recent trade stats
        recent_pnls = pnl_list[-10:] if len(pnl_list) >= 10 else pnl_list
        recent_mean = np.mean(recent_pnls) if recent_pnls else 0
        recent_wr = sum(1 for p in recent_pnls if p > 0) / len(recent_pnls) if recent_pnls else 0.5
        bars_since_loss = 0
        for j in range(len(pnl_list)-1, -1, -1):
            if pnl_list[j] < 0: break
            bars_since_loss += 1

        feat = {
            'atr': atr if not np.isnan(atr) else 0,
            'atr_pct': atr_pct,
            'adx': adx_val if not np.isnan(adx_val) else 0,
            'hour': hour,
            'dow': dow,
            'recent_mean_10': recent_mean,
            'recent_wr_10': recent_wr,
            'bars_since_loss': bars_since_loss,
            'trade_idx': i,
        }
        features.append(feat)
        labels.append(1 if pnl > 0 else 0)
        pnl_list.append(pnl)

    feat_df = pd.DataFrame(features)
    labels = np.array(labels)
    all_pnls = [t.pnl if hasattr(t, 'pnl') else t['pnl'] for t in trades]

    print(f"  Features built: {len(feat_df)} trades, {feat_df.shape[1]} features")
    print(f"  Win rate: {labels.mean()*100:.1f}%")

    # Walk-Forward: train on first N years, predict next
    wf_splits = [
        ("Train 2015-2019, Test 2019-2021", 0, 4*365*24, 4*365*24, 6*365*24),
        ("Train 2015-2021, Test 2021-2023", 0, 6*365*24, 6*365*24, 8*365*24),
        ("Train 2015-2023, Test 2023-2025", 0, 8*365*24, 8*365*24, 10*365*24),
        ("Train 2015-2025, Test 2025-2026", 0, 10*365*24, 10*365*24, 11*365*24),
    ]

    # Use year-based splits via trade_idx
    years = {}
    for i, t in enumerate(trades):
        et = t.entry_time if hasattr(t, 'entry_time') else t['entry_time']
        yr = pd.Timestamp(et).year
        years.setdefault(yr, [])
        years[yr].append(i)

    sorted_years = sorted(years.keys())
    print(f"  Years: {sorted_years}")

    wf_configs = [
        ("Train 15-18, Test 19-20", [2015,2016,2017,2018], [2019,2020]),
        ("Train 15-20, Test 21-22", [2015,2016,2017,2018,2019,2020], [2021,2022]),
        ("Train 15-22, Test 23-24", [2015,2016,2017,2018,2019,2020,2021,2022], [2023,2024]),
        ("Train 15-24, Test 25-26", [2015,2016,2017,2018,2019,2020,2021,2022,2023,2024], [2025,2026]),
    ]

    print(f"\n  --- D1: Walk-Forward ML Filter ---")
    print(f"  {'Split':>30} {'Base_Sh':>8} {'ML_Sh':>7} {'Delta':>7} {'Skip%':>6}")

    for name, train_yrs, test_yrs in wf_configs:
        train_idx = [i for yr in train_yrs if yr in years for i in years[yr]]
        test_idx = [i for yr in test_yrs if yr in years for i in years[yr]]
        if not train_idx or not test_idx: continue

        X_train = feat_df.iloc[train_idx].drop(columns=['trade_idx'])
        y_train = labels[train_idx]
        X_test = feat_df.iloc[test_idx].drop(columns=['trade_idx'])

        if HAS_XGB:
            model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                       use_label_encoder=False, eval_metric='logloss', verbosity=0)
        elif HAS_SKLEARN:
            model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
        else:
            # Fallback: use recent_wr_10 as heuristic
            model = None

        if model is not None:
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
        else:
            proba = X_test['recent_wr_10'].values

        # Test: skip trades where predicted win prob < threshold
        test_pnls = [all_pnls[i] for i in test_idx]
        test_trades_obj = [trades[i] for i in test_idx]

        base_daily = {}
        for j, idx in enumerate(test_idx):
            t = trades[idx]
            exit_t = t.exit_time if hasattr(t, 'exit_time') else t['exit_time']
            d = pd.Timestamp(exit_t).date()
            base_daily.setdefault(d, 0); base_daily[d] += test_pnls[j]
        da_b = np.array(list(base_daily.values()))
        sh_base = da_b.mean() / da_b.std() * np.sqrt(252) if len(da_b) > 1 and da_b.std() > 0 else 0

        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            ml_daily = {}
            skipped = 0
            for j, idx in enumerate(test_idx):
                if proba[j] < threshold:
                    skipped += 1; continue
                t = trades[idx]
                exit_t = t.exit_time if hasattr(t, 'exit_time') else t['exit_time']
                d = pd.Timestamp(exit_t).date()
                ml_daily.setdefault(d, 0); ml_daily[d] += test_pnls[j]

            if ml_daily:
                da_m = np.array(list(ml_daily.values()))
                sh_ml = da_m.mean() / da_m.std() * np.sqrt(252) if len(da_m) > 1 and da_m.std() > 0 else 0
            else:
                sh_ml = 0

            skip_pct = 100 * skipped / len(test_idx) if test_idx else 0
            if threshold == 0.5:
                print(f"  {name:>30} {sh_base:>8.2f} {sh_ml:>7.2f} {sh_ml-sh_base:>+7.2f} {skip_pct:>5.1f}%")
            else:
                print(f"  {'  thresh='+str(threshold):>30} {'':>8} {sh_ml:>7.2f} {sh_ml-sh_base:>+7.2f} {skip_pct:>5.1f}%")

    # D2: Feature importance
    if model is not None and hasattr(model, 'feature_importances_'):
        print(f"\n  --- D2: Feature Importance (last model) ---")
        cols = X_train.columns
        imp = model.feature_importances_
        for c, v in sorted(zip(cols, imp), key=lambda x: -x[1]):
            print(f"  {c:>20}: {v:.3f}")

    # D3: Compare ML filter vs EqCurve LB=10
    print(f"\n  --- D3: ML Filter vs EqCurve LB=10 Comparison ---")
    print(f"  (Use Walk-Forward test periods only)")

    from collections import defaultdict
    eq_daily = defaultdict(float)
    recent = []
    for i, pnl in enumerate(all_pnls):
        recent.append(pnl)
        if len(recent) > 10: recent.pop(0)
        skip = len(recent) >= 10 and np.mean(recent) < 0
        t = trades[i]
        yr = pd.Timestamp(t.entry_time if hasattr(t, 'entry_time') else t['entry_time']).year
        if yr in [2019,2020,2021,2022,2023,2024,2025,2026]:
            exit_t = t.exit_time if hasattr(t, 'exit_time') else t['exit_time']
            d = pd.Timestamp(exit_t).date()
            if not skip:
                eq_daily[d] += pnl

    if eq_daily:
        da_eq = np.array(list(eq_daily.values()))
        sh_eq = da_eq.mean() / da_eq.std() * np.sqrt(252) if len(da_eq) > 1 and da_eq.std() > 0 else 0
        print(f"  EqCurve LB=10 on test years: Sharpe={sh_eq:.2f}")


def main():
    t0 = time.time()
    out_path = OUT_DIR / "R32_output.txt"
    out = open(out_path, 'w', encoding='utf-8', buffering=1)
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, out)

    print(f"# R32: New Directions — First Batch")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()

    for name, fn in [("A", lambda: run_phase_A(data)),
                     ("B", lambda: run_phase_B(data)),
                     ("C", lambda: run_phase_C(h1_df)),
                     ("D", lambda: run_phase_D(data))]:
        try:
            fn()
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
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
