"""
R37B: Deep Validation of R37 Top Strategies
=============================================
A: Parameter Cliff Test (Donchian period, EMA period sensitivity)
B: Spread Robustness ($0.30 -> $1.50 gradient)
C: Yearly Breakdown (per-year PnL/Sharpe)
D: Cross-Correlation with L7 (M15 Keltner)
E: Walk-Forward rolling stability
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_FILE = "R37B_output.txt"

class Tee:
    def __init__(self, fname):
        self.file = open(fname, 'w', buffering=1)
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()

sys.stdout = Tee(OUTPUT_FILE)
print(f"R37B started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════
# Data Loading (reuse R37 framework)
# ═══════════════════════════════════════════════════════════════

from backtest.runner import load_csv, load_m15, load_h1_aligned, DataBundle, run_variant
from backtest.runner import prepare_indicators_custom, add_atr_percentile

M15_PATH = Path("data/download/xauusd-m15-bid-2015-01-01-2026-04-10.csv")
H1_PATH = Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv")

print("\nLoading data...")
m15_df_raw = load_m15(M15_PATH)
h1_df = load_h1_aligned(H1_PATH, m15_df_raw.index[0])

def resample_ohlcv(df, rule):
    return df.resample(rule).agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

h4_df = resample_ohlcv(h1_df, '4h')
d1_df = resample_ohlcv(h1_df, '1D')

print(f"  H4: {len(h4_df)} bars, D1: {len(d1_df)} bars")

def _calc_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _calc_adx(df, period):
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr = pd.DataFrame({
        'hl': high - low, 'hc': (high - close.shift(1)).abs(), 'lc': (low - close.shift(1)).abs()
    }).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(period).mean() / atr.replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.rolling(period).mean()

def add_indicators(df):
    df = df.copy()
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs()
    }).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['RSI14'] = _calc_rsi(df['Close'], 14)
    df['ADX'] = _calc_adx(df, 14)
    for span in [9, 20, 25, 50, 100, 200]:
        df[f'EMA{span}'] = df['Close'].ewm(span=span).mean()
    for n in [5, 10, 15, 20, 30, 40, 50, 60, 80]:
        df[f'Donchian_high_{n}'] = df['High'].rolling(n).max()
        df[f'Donchian_low_{n}'] = df['Low'].rolling(n).min()
    df['ROC5'] = df['Close'].pct_change(5) * 100
    return df

print("Adding indicators...")
h4_df = add_indicators(h4_df)
d1_df = add_indicators(d1_df)
print("Indicators ready.")

# ═══════════════════════════════════════════════════════════════
# Backtester (from R37)
# ═══════════════════════════════════════════════════════════════

def backtest(df, label, signal_fn, sl_atr=3.5, tp_atr=8.0,
             trail_act_atr=0.28, trail_dist_atr=0.06, max_hold=50,
             spread=0.30, lot=0.03, cooldown=2):
    df = df.dropna(subset=['ATR'])
    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    open_ = df['Open'].values; atr = df['ATR'].values
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
                _rec(trades, pos, close[i], times[i], "TP", i, tp_val); exited = True
            elif pnl_l <= -sl_val:
                _rec(trades, pos, close[i], times[i], "SL", i, -sl_val); exited = True
            else:
                ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and high[i] - pos['entry'] >= ad:
                    ts = high[i] - td
                    if low[i] <= ts:
                        _rec(trades, pos, close[i], times[i], "Trail", i,
                             (ts - pos['entry'] - spread) * lot * 100); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - low[i] >= ad:
                    ts = low[i] + td
                    if high[i] >= ts:
                        _rec(trades, pos, close[i], times[i], "Trail", i,
                             (pos['entry'] - ts - spread) * lot * 100); exited = True
                if not exited and held >= max_hold:
                    _rec(trades, pos, close[i], times[i], "Timeout", i, pnl_c); exited = True
            if exited: pos = None; last_exit = i; continue
        if pos is not None: continue
        if i - last_exit < cooldown: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        sig = signal_fn(df, i)
        if sig == 'BUY':
            pos = {'dir': 'BUY', 'entry': open_[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif sig == 'SELL':
            pos = {'dir': 'SELL', 'entry': open_[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return _stats(trades, label)

def _rec(trades, pos, exit_price, exit_time, reason, bar_i, pnl):
    trades.append({
        'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_price,
        'entry_time': pos['time'], 'exit_time': exit_time,
        'pnl': pnl, 'exit_reason': reason, 'bars_held': bar_i - pos['bar'],
        'entry_atr': pos['atr']
    })

def _stats(trades, label):
    if not trades:
        return {'label': label, 'n': 0, 'sharpe': 0, 'total_pnl': 0, 'win_rate': 0,
                'max_dd': 0, '_trades': [], '_daily_pnl': {}}
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
            'win_rate': wins / len(trades) * 100, 'max_dd': dd,
            '_trades': trades, '_daily_pnl': daily}

def pr(r):
    print(f"  {r['label']:>50}: N={r['n']:>5}, Sharpe={r['sharpe']:>7.2f}, "
          f"PnL=${r['total_pnl']:>8.0f}, WR={r['win_rate']:>5.1f}%, MaxDD=${r['max_dd']:>6.0f}")

def sharpe_from_daily(daily_dict):
    if not daily_dict: return 0
    da = np.array(list(daily_dict.values()))
    return da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0

# ═══════════════════════════════════════════════════════════════
# Phase A: Parameter Cliff Test
# ═══════════════════════════════════════════════════════════════

def run_phase_A():
    print("\n" + "=" * 80)
    print("Phase A: Parameter Cliff Test")
    print("=" * 80)

    # A1: Donchian period sweep on D1
    print("\n  --- A1: D1 Donchian Period Sweep (SL3.5/TP8.0/MH30) ---")
    for period in [5, 10, 15, 20, 30, 40, 50, 60, 80]:
        hcol = f'Donchian_high_{period}'; lcol = f'Donchian_low_{period}'
        if hcol not in d1_df.columns: continue
        def sig(d, i, _hc=hcol, _lc=lcol):
            if i < 2: return None
            ch = d[_hc].iloc[i-1]; cl = d[_lc].iloc[i-1]; c = d['Close'].iloc[i]
            if np.isnan(ch): return None
            if c > ch: return 'BUY'
            if c < cl: return 'SELL'
            return None
        r = backtest(d1_df, f"D1_Don{period}", sig, sl_atr=3.5, tp_atr=8.0, max_hold=30)
        pr(r)

    # A2: D1 EMA crossover fast/slow sweep
    print("\n  --- A2: D1 EMA Crossover Sweep (ADX25, SL4/TP10/MH50) ---")
    for fast in [5, 9, 12, 15, 20, 25]:
        for slow in [20, 25, 50, 100, 200]:
            if fast >= slow: continue
            fc = f'EMA{fast}'; sc = f'EMA{slow}'
            if fc not in d1_df.columns or sc not in d1_df.columns: continue
            def sig(d, i, _fc=fc, _sc=sc):
                if i < 2: return None
                fn = d[_fc].iloc[i]; fp = d[_fc].iloc[i-1]
                sn = d[_sc].iloc[i]; sp = d[_sc].iloc[i-1]
                if np.isnan(fn) or np.isnan(sn): return None
                adx = d['ADX'].iloc[i]
                if np.isnan(adx) or adx < 25: return None
                if fp <= sp and fn > sn: return 'BUY'
                if fp >= sp and fn < sn: return 'SELL'
                return None
            r = backtest(d1_df, f"D1_EMAx{fast}/{slow}_ADX25", sig, sl_atr=4.0, tp_atr=10.0, max_hold=50)
            if r['n'] > 5: pr(r)

    # A3: H4 ROC threshold sweep
    print("\n  --- A3: H4 ROC5 Threshold Sweep (EMA200, SL3/MH20) ---")
    for thresh in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        def sig(d, i, _t=thresh):
            if i < 2: return None
            roc = d['ROC5'].iloc[i]; c = d['Close'].iloc[i]; ema = d['EMA200'].iloc[i]
            if np.isnan(roc) or np.isnan(ema): return None
            if roc > _t and c > ema: return 'BUY'
            if roc < -_t and c < ema: return 'SELL'
            return None
        r = backtest(h4_df, f"H4_ROC5>{thresh}_EMA200", sig, sl_atr=3.0, tp_atr=6.0, max_hold=20)
        pr(r)

    # A4: D1+H4 Top-Down EMA trend period sweep
    print("\n  --- A4: Multi-TF EMA Trend Period Sweep (D1 trend + H4 ema_bounce) ---")
    d1_trend_data = d1_df[['Close', 'EMA9', 'EMA20', 'EMA25', 'EMA50', 'EMA100', 'EMA200']].copy()
    for ema_trend in [9, 20, 25, 50, 100, 200]:
        ec = f'EMA{ema_trend}'
        def sig(d, i, _ec=ec):
            if i < 2: return None
            c = d['Close'].iloc[i]; t = d.index[i]
            d1_mask = d1_trend_data.index <= t
            if not d1_mask.any(): return None
            d1_row = d1_trend_data.loc[d1_trend_data.index[d1_mask][-1]]
            if np.isnan(d1_row[_ec]): return None
            trend = 'BULL' if d1_row['Close'] > d1_row[_ec] else 'BEAR'
            ema20 = d['EMA20'].iloc[i]
            if np.isnan(ema20): return None
            prev_c = d['Close'].iloc[i-1]
            if trend == 'BULL' and prev_c < ema20 and c > ema20: return 'BUY'
            if trend == 'BEAR' and prev_c > ema20 and c < ema20: return 'SELL'
            return None
        r = backtest(h4_df, f"D1(EMA{ema_trend})+H4(bounce)", sig, sl_atr=4.0, tp_atr=10.0, max_hold=30)
        pr(r)

    # A5: Trail parameter sensitivity for D1 Donchian 50
    print("\n  --- A5: D1 Don50 Trail Parameter Sensitivity ---")
    def don50_sig(d, i):
        if i < 2: return None
        ch = d['Donchian_high_50'].iloc[i-1]; cl = d['Donchian_low_50'].iloc[i-1]
        c = d['Close'].iloc[i]
        if np.isnan(ch): return None
        if c > ch: return 'BUY'
        if c < cl: return 'SELL'
        return None
    for ta in [0.15, 0.20, 0.28, 0.40, 0.60, 1.0, 1.5, 2.0]:
        for td in [0.03, 0.06, 0.10, 0.15, 0.25, 0.40]:
            r = backtest(d1_df, f"D1_Don50_trail({ta}/{td})", don50_sig,
                         sl_atr=3.5, tp_atr=8.0, max_hold=30,
                         trail_act_atr=ta, trail_dist_atr=td)
            if r['n'] > 10:
                pr(r)


# ═══════════════════════════════════════════════════════════════
# Phase B: Spread Robustness
# ═══════════════════════════════════════════════════════════════

def run_phase_B():
    print("\n" + "=" * 80)
    print("Phase B: Spread Robustness")
    print("=" * 80)

    strategies = {
        'D1_Don50': lambda d, i: _don_sig(d, i, 50),
        'D1_EMAx9/25_ADX25': lambda d, i: _ema_cross_sig(d, i, 'EMA9', 'EMA25', 25),
        'D1+H4_TopDown': lambda d, i: _topdown_sig(d, i),
        'H4_ROC5_EMA200': lambda d, i: _roc_sig(d, i, 1.0),
    }

    for sp in [0.10, 0.30, 0.50, 0.80, 1.00, 1.50, 2.00, 3.00]:
        print(f"\n  --- Spread = ${sp:.2f} ---")
        for name, sig_fn in strategies.items():
            df = d1_df if 'D1' in name and 'H4' not in name else h4_df
            params = _get_params(name)
            r = backtest(df, f"{name}_sp{sp}", sig_fn, spread=sp, **params)
            if r['n'] > 0:
                pr(r)


def _don_sig(d, i, period):
    if i < 2: return None
    hc = f'Donchian_high_{period}'; lc = f'Donchian_low_{period}'
    ch = d[hc].iloc[i-1]; cl = d[lc].iloc[i-1]; c = d['Close'].iloc[i]
    if np.isnan(ch): return None
    if c > ch: return 'BUY'
    if c < cl: return 'SELL'
    return None

def _ema_cross_sig(d, i, fast_col, slow_col, adx_gate):
    if i < 2: return None
    fn = d[fast_col].iloc[i]; fp = d[fast_col].iloc[i-1]
    sn = d[slow_col].iloc[i]; sp = d[slow_col].iloc[i-1]
    if np.isnan(fn) or np.isnan(sn): return None
    if adx_gate > 0:
        adx = d['ADX'].iloc[i]
        if np.isnan(adx) or adx < adx_gate: return None
    if fp <= sp and fn > sn: return 'BUY'
    if fp >= sp and fn < sn: return 'SELL'
    return None

d1_trend_ref = d1_df[['Close', 'EMA50', 'EMA20']].copy()

def _topdown_sig(d, i):
    if i < 2: return None
    c = d['Close'].iloc[i]; t = d.index[i]
    mask = d1_trend_ref.index <= t
    if not mask.any(): return None
    d1r = d1_trend_ref.loc[d1_trend_ref.index[mask][-1]]
    trend = 'BULL' if d1r['Close'] > d1r['EMA50'] else 'BEAR'
    ema20 = d['EMA20'].iloc[i]
    if np.isnan(ema20): return None
    prev_c = d['Close'].iloc[i-1]
    if trend == 'BULL' and prev_c < ema20 and c > ema20: return 'BUY'
    if trend == 'BEAR' and prev_c > ema20 and c < ema20: return 'SELL'
    return None

def _roc_sig(d, i, thresh):
    if i < 2: return None
    roc = d['ROC5'].iloc[i]; c = d['Close'].iloc[i]; ema = d['EMA200'].iloc[i]
    if np.isnan(roc) or np.isnan(ema): return None
    if roc > thresh and c > ema: return 'BUY'
    if roc < -thresh and c < ema: return 'SELL'
    return None

def _get_params(name):
    if 'Don50' in name:
        return dict(sl_atr=3.5, tp_atr=8.0, max_hold=30, trail_act_atr=0.28, trail_dist_atr=0.06)
    elif 'EMAx' in name:
        return dict(sl_atr=4.0, tp_atr=10.0, max_hold=50, trail_act_atr=0.28, trail_dist_atr=0.06)
    elif 'TopDown' in name:
        return dict(sl_atr=4.0, tp_atr=10.0, max_hold=30, trail_act_atr=0.28, trail_dist_atr=0.06)
    else:
        return dict(sl_atr=3.0, tp_atr=6.0, max_hold=20, trail_act_atr=0.28, trail_dist_atr=0.06)


# ═══════════════════════════════════════════════════════════════
# Phase C: Yearly Breakdown
# ═══════════════════════════════════════════════════════════════

def run_phase_C():
    print("\n" + "=" * 80)
    print("Phase C: Yearly Breakdown")
    print("=" * 80)

    strats = [
        ('D1_Don50', d1_df, lambda d, i: _don_sig(d, i, 50), _get_params('Don50')),
        ('D1_EMAx9/25_ADX25', d1_df, lambda d, i: _ema_cross_sig(d, i, 'EMA9', 'EMA25', 25), _get_params('EMAx')),
        ('D1+H4_TopDown', h4_df, lambda d, i: _topdown_sig(d, i), _get_params('TopDown')),
        ('H4_ROC5_EMA200', h4_df, lambda d, i: _roc_sig(d, i, 1.0), _get_params('ROC')),
    ]

    for sname, df, sig_fn, params in strats:
        r = backtest(df, sname, sig_fn, **params)
        trades = r['_trades']
        if not trades: continue

        print(f"\n  --- {sname} (Full: Sharpe={r['sharpe']:.2f}, N={r['n']}) ---")
        print(f"  {'Year':>6} {'N':>5} {'PnL':>9} {'WR':>6} {'Sharpe':>7}")
        for year in range(2015, 2027):
            yr_trades = [t for t in trades
                         if pd.Timestamp(t['exit_time']).year == year]
            if not yr_trades: continue
            yr_pnl = sum(t['pnl'] for t in yr_trades)
            yr_wr = sum(1 for t in yr_trades if t['pnl'] > 0) / len(yr_trades) * 100
            yr_daily = {}
            for t in yr_trades:
                d = pd.Timestamp(t['exit_time']).date()
                yr_daily.setdefault(d, 0); yr_daily[d] += t['pnl']
            da = np.array(list(yr_daily.values()))
            yr_sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0
            print(f"  {year:>6} {len(yr_trades):>5} ${yr_pnl:>8.0f} {yr_wr:>5.1f}% {yr_sh:>7.2f}")


# ═══════════════════════════════════════════════════════════════
# Phase D: Cross-Correlation with L7
# ═══════════════════════════════════════════════════════════════

def run_phase_D():
    print("\n" + "=" * 80)
    print("Phase D: Cross-Correlation with L7 Trades")
    print("=" * 80)

    print("  Loading L7 data & running baseline...")
    data = DataBundle.load_default()

    L7_KWARGS = {
        'trailing_activate_atr': 0.28, 'trailing_distance_atr': 0.06,
        'sl_atr_mult': 3.5, 'tp_atr_mult': 8.0,
        'keltner_adx_threshold': 18, 'max_positions': 1,
        'intraday_adaptive': True, 'choppy_threshold': 0.50,
        'kc_only_threshold': 0.60, 'min_entry_gap_hours': 1.0,
        'regime_config': {
            'low_pct': 25, 'high_pct': 75,
            'low_trail': (0.10, 0.025), 'mid_trail': (0.20, 0.04),
            'high_trail': (0.28, 0.06),
        },
        'time_adaptive_trail': True, 'time_adaptive_trail_start': 2,
        'time_adaptive_trail_decay': 0.75, 'time_adaptive_trail_floor': 0.003,
        'keltner_max_hold_m15': 8,
        'spread_cost': 0.30,
    }
    l7_result = run_variant(data, "L7_MH8", verbose=False, **L7_KWARGS)
    l7_trades = l7_result['trades']
    l7_daily = {}
    for t in l7_trades:
        d = pd.Timestamp(t.exit_time).date()
        l7_daily.setdefault(d, 0); l7_daily[d] += t.pnl
    l7_daily_s = pd.Series(l7_daily).sort_index()
    l7_sh = sharpe_from_daily(l7_daily)
    print(f"  L7(MH=8): N={len(l7_trades)}, Sharpe={l7_sh:.2f}, PnL=${sum(t.pnl for t in l7_trades):.0f}")

    new_strats = [
        ('D1_Don50', d1_df, lambda d, i: _don_sig(d, i, 50), _get_params('Don50')),
        ('D1_EMAx9/25_ADX25', d1_df, lambda d, i: _ema_cross_sig(d, i, 'EMA9', 'EMA25', 25), _get_params('EMAx')),
        ('D1+H4_TopDown', h4_df, lambda d, i: _topdown_sig(d, i), _get_params('TopDown')),
        ('H4_ROC5_EMA200', h4_df, lambda d, i: _roc_sig(d, i, 1.0), _get_params('ROC')),
    ]

    print(f"\n  {'Strategy':>25} {'Corr':>7} {'ComboSh':>8} {'ComboPnL':>10} {'PortSh':>7}")
    for sname, df, sig_fn, params in new_strats:
        r = backtest(df, sname, sig_fn, **params)
        if not r['_daily_pnl']: continue
        new_daily = pd.Series(r['_daily_pnl']).sort_index()
        all_idx = sorted(set(l7_daily_s.index) | set(new_daily.index))
        corr_df = pd.DataFrame({
            'L7': l7_daily_s.reindex(all_idx, fill_value=0),
            sname: new_daily.reindex(all_idx, fill_value=0),
        })
        corr = corr_df.corr().iloc[0, 1]

        combo = corr_df['L7'] + corr_df[sname]
        combo_sh = combo.mean() / combo.std() * np.sqrt(252) if combo.std() > 0 else 0
        combo_pnl = combo.sum()

        for w in [0.5, 1.0, 1.5]:
            port = corr_df['L7'] + w * corr_df[sname]
            p_sh = port.mean() / port.std() * np.sqrt(252) if port.std() > 0 else 0
            if w == 1.0:
                print(f"  {sname:>25} {corr:>7.3f} {combo_sh:>8.2f} ${combo_pnl:>9.0f}")
            print(f"  {'':>25} w={w:.1f}: Sharpe={p_sh:.2f}, PnL=${port.sum():.0f}")


# ═══════════════════════════════════════════════════════════════
# Phase E: Rolling Walk-Forward
# ═══════════════════════════════════════════════════════════════

def run_phase_E():
    print("\n" + "=" * 80)
    print("Phase E: Rolling Walk-Forward (12-month windows)")
    print("=" * 80)

    strats = [
        ('D1_Don50', d1_df, lambda d, i: _don_sig(d, i, 50), _get_params('Don50')),
        ('D1+H4_TopDown', h4_df, lambda d, i: _topdown_sig(d, i), _get_params('TopDown')),
        ('H4_ROC5_EMA200', h4_df, lambda d, i: _roc_sig(d, i, 1.0), _get_params('ROC')),
    ]

    for sname, df, sig_fn, params in strats:
        r = backtest(df, sname, sig_fn, **params)
        trades = r['_trades']
        if not trades: continue

        print(f"\n  --- {sname} ---")
        daily = {}
        for t in trades:
            d = pd.Timestamp(t['exit_time']).date()
            daily.setdefault(d, 0); daily[d] += t['pnl']
        daily_s = pd.Series(daily).sort_index()

        windows = []
        for year in range(2015, 2026):
            for half in [0, 1]:
                start = pd.Timestamp(f"{year}-{1+half*6:02d}-01").date()
                end = pd.Timestamp(f"{year}-{7+half*6:02d}-01").date() if half == 0 else pd.Timestamp(f"{year+1}-01-01").date()
                w = daily_s[(daily_s.index >= start) & (daily_s.index < end)]
                if len(w) < 5: continue
                sh = w.mean() / w.std() * np.sqrt(252) if w.std() > 0 else 0
                windows.append({'period': f"{year}H{half+1}", 'sharpe': sh, 'pnl': w.sum(), 'days': len(w)})

        positive = sum(1 for w in windows if w['sharpe'] > 0)
        sharpes = [w['sharpe'] for w in windows]
        print(f"  Windows: {len(windows)}, Positive: {positive}/{len(windows)}")
        if sharpes:
            print(f"  Sharpe: min={min(sharpes):.2f}, max={max(sharpes):.2f}, "
                  f"mean={np.mean(sharpes):.2f}, median={np.median(sharpes):.2f}")
        for w in windows:
            status = 'PASS' if w['sharpe'] > 0 else 'FAIL'
            print(f"    {w['period']:>7}: Sharpe={w['sharpe']:>7.2f}, PnL=${w['pnl']:>7.0f}, Days={w['days']:>3} {status}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()

    run_phase_A()
    print(f"\n  Phase A done: {time.time()-t0:.0f}s")

    run_phase_B()
    print(f"\n  Phase B done: {time.time()-t0:.0f}s")

    run_phase_C()
    print(f"\n  Phase C done: {time.time()-t0:.0f}s")

    try:
        run_phase_D()
        print(f"\n  Phase D done: {time.time()-t0:.0f}s")
    except Exception as e:
        print(f"\n  Phase D error: {e}")
        import traceback; traceback.print_exc()

    run_phase_E()
    print(f"\n  Phase E done: {time.time()-t0:.0f}s")

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"R37B completed in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}")
