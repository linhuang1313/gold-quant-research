"""
Round 21: Four New Profit Dimensions
======================================
S1: Volatility Squeeze Straddle
S2: Event-Driven (NFP/FOMC/CPI)
S3: Overnight Return Anomaly
S4: Extreme Reversal / Liquidity Provision

Each strategy is independent of Keltner — they use raw OHLC data
with simple simulated execution (no engine dependency).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional

OUT_DIR = Path("results/round21_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.30


@dataclass
class SimTrade:
    entry_time: datetime
    direction: str
    entry_price: float
    sl: float
    tp: float
    strategy: str
    exit_time: Optional[datetime] = None
    exit_price: float = 0.0
    pnl: float = 0.0
    exit_reason: str = ""
    bars_held: int = 0


def calc_sharpe(pnls, periods_per_year=252*4):
    if len(pnls) < 2:
        return 0.0
    arr = np.array(pnls)
    if arr.std(ddof=1) == 0:
        return 0.0
    return arr.mean() / arr.std(ddof=1) * np.sqrt(periods_per_year)


def summarize_trades(trades: List[SimTrade], label: str):
    if not trades:
        print(f"  {label}: 0 trades")
        return {}
    pnls = [t.pnl for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    n = len(pnls)
    total = sum(pnls)
    avg = total / n
    wr = wins / n
    max_dd = 0
    peak = 0
    equity = 0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)

    sharpe = calc_sharpe(pnls, periods_per_year=n / 11 if n > 11 else 252)

    print(f"  {label}: N={n}, Sharpe={sharpe:.2f}, PnL=${total:.0f}, "
          f"WR={wr:.1%}, Avg$/t=${avg:.2f}, MaxDD=${max_dd:.0f}")

    by_year = {}
    for t in trades:
        y = t.entry_time.year
        by_year.setdefault(y, []).append(t.pnl)
    print(f"    Year-by-year PnL: ", end="")
    for y in sorted(by_year):
        yp = sum(by_year[y])
        print(f"{y}=${yp:.0f} ", end="")
    print()

    return {'label': label, 'n': n, 'sharpe': sharpe, 'total_pnl': total,
            'win_rate': wr, 'avg_pnl': avg, 'max_dd': max_dd}


# ═══════════════════════════════════════════════════════════════
# S1: Volatility Squeeze Straddle
# ═══════════════════════════════════════════════════════════════

def run_s1_squeeze_straddle(h1_df):
    """After N bars of Squeeze, open both BUY and SELL with tight trailing."""
    print("\n" + "=" * 70)
    print("S1: Volatility Squeeze Straddle")
    print("=" * 70)

    df = h1_df.copy()
    if 'squeeze' not in df.columns:
        bb_mid = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        df['squeeze'] = ((bb_upper < df['KC_upper']) & (bb_lower > df['KC_lower'])).astype(float)

    atr = df['ATR'].values
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    squeeze = df['squeeze'].values
    times = df.index

    all_results = {}

    for min_squeeze_bars in [3, 5, 8, 12]:
        for trail_act, trail_dist in [(0.15, 0.03), (0.20, 0.04), (0.30, 0.06)]:
            for max_hold in [8, 12, 20]:
                trades = []
                squeeze_count = 0
                in_trade = False
                buy_pos = None
                sell_pos = None

                for i in range(50, len(df) - 1):
                    cur_atr = atr[i]
                    if cur_atr <= 0:
                        continue

                    # Track squeeze duration
                    if squeeze[i] == 1:
                        squeeze_count += 1
                    else:
                        # Squeeze just released
                        if squeeze_count >= min_squeeze_bars and not in_trade:
                            entry_price = close[i]
                            sl_dist = 1.5 * cur_atr
                            # Open both directions
                            buy_pos = {
                                'entry_time': times[i], 'entry_price': entry_price,
                                'sl': entry_price - sl_dist,
                                'trail_stop': 0, 'extreme': entry_price,
                                'bars': 0, 'entry_atr': cur_atr
                            }
                            sell_pos = {
                                'entry_time': times[i], 'entry_price': entry_price,
                                'sl': entry_price + sl_dist,
                                'trail_stop': 999999, 'extreme': entry_price,
                                'bars': 0, 'entry_atr': cur_atr
                            }
                            in_trade = True
                        squeeze_count = 0

                    if not in_trade:
                        continue

                    # Process open positions
                    for pos, direction in [(buy_pos, 'BUY'), (sell_pos, 'SELL')]:
                        if pos is None or pos.get('closed'):
                            continue
                        pos['bars'] += 1
                        h, l, c = high[i], low[i], close[i]
                        a = cur_atr

                        exit_price = None
                        reason = ""

                        if direction == 'BUY':
                            # SL check
                            if l <= pos['sl']:
                                exit_price = pos['sl']
                                reason = "SL"
                            else:
                                # Trailing
                                pos['extreme'] = max(pos['extreme'], h)
                                float_profit = h - pos['entry_price']
                                if float_profit >= a * trail_act:
                                    trail = pos['extreme'] - a * trail_dist
                                    pos['trail_stop'] = max(pos.get('trail_stop', 0), trail)
                                    if l <= pos['trail_stop']:
                                        exit_price = pos['trail_stop']
                                        reason = "Trail"
                                # Timeout
                                if pos['bars'] >= max_hold and exit_price is None:
                                    exit_price = c
                                    reason = "Timeout"
                        else:  # SELL
                            if h >= pos['sl']:
                                exit_price = pos['sl']
                                reason = "SL"
                            else:
                                pos['extreme'] = min(pos['extreme'], l)
                                float_profit = pos['entry_price'] - l
                                if float_profit >= a * trail_act:
                                    trail = pos['extreme'] + a * trail_dist
                                    pos['trail_stop'] = min(pos.get('trail_stop', 999999), trail)
                                    if h >= pos['trail_stop']:
                                        exit_price = pos['trail_stop']
                                        reason = "Trail"
                                if pos['bars'] >= max_hold and exit_price is None:
                                    exit_price = c
                                    reason = "Timeout"

                        if exit_price is not None:
                            if direction == 'BUY':
                                pnl = exit_price - pos['entry_price'] - SPREAD
                            else:
                                pnl = pos['entry_price'] - exit_price - SPREAD
                            trades.append(SimTrade(
                                entry_time=pos['entry_time'], direction=direction,
                                entry_price=pos['entry_price'], sl=pos['sl'], tp=0,
                                strategy='s1_squeeze', exit_time=times[i],
                                exit_price=exit_price, pnl=pnl, exit_reason=reason,
                                bars_held=pos['bars']
                            ))
                            pos['closed'] = True

                    # Check if both closed
                    if (buy_pos and buy_pos.get('closed')) and (sell_pos and sell_pos.get('closed')):
                        in_trade = False
                        buy_pos = sell_pos = None

                label = f"SqzB{min_squeeze_bars}_T{trail_act}/{trail_dist}_MH{max_hold}"
                stats = summarize_trades(trades, label)
                if stats:
                    all_results[label] = stats

    # Find best
    if all_results:
        best = max(all_results.values(), key=lambda x: x.get('sharpe', 0))
        print(f"\n  Best S1: {best['label']} Sharpe={best['sharpe']:.2f}")
    return all_results


# ═══════════════════════════════════════════════════════════════
# S2: Event-Driven (simplified — use known dates)
# ═══════════════════════════════════════════════════════════════

def get_nfp_dates(start_year=2015, end_year=2026):
    """Generate approximate NFP dates (first Friday of each month)."""
    dates = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            d = datetime(y, m, 1)
            # Find first Friday
            while d.weekday() != 4:
                d += timedelta(days=1)
            dates.append(d)
    return dates


def get_fomc_dates_approx():
    """Known FOMC meeting dates 2015-2026 (announcement day, approximate)."""
    # 8 meetings per year, roughly every 6 weeks
    # Using mid-month pattern: Jan, Mar, May, Jun, Jul, Sep, Nov, Dec
    dates = []
    fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
    for y in range(2015, 2027):
        for m in fomc_months:
            # FOMC typically on Wednesday around 14th-20th
            d = datetime(y, m, 15)
            while d.weekday() != 2:
                d += timedelta(days=1)
            dates.append(d)
    return dates


def run_s2_event_driven(h1_df, m15_df):
    """Analyze price behavior around NFP/FOMC events."""
    print("\n" + "=" * 70)
    print("S2: Event-Driven Strategy (NFP/FOMC)")
    print("=" * 70)

    nfp_dates = get_nfp_dates()
    fomc_dates = get_fomc_dates_approx()

    for event_name, event_dates, event_hour_utc in [
        ("NFP", nfp_dates, 13),     # 8:30 ET = 13:30 UTC
        ("FOMC", fomc_dates, 19),    # 14:00 ET = 19:00 UTC
    ]:
        print(f"\n--- {event_name} Analysis ---")

        pre_moves = []   # 2h before event
        post_moves = []   # 2h after event
        straddle_results = []
        continuation_results = []

        for ed in event_dates:
            event_time = pd.Timestamp(ed.replace(hour=event_hour_utc), tz='UTC')

            # Find bars around event in H1
            pre_start = event_time - pd.Timedelta(hours=3)
            pre_end = event_time - pd.Timedelta(hours=1)
            post_start = event_time
            post_end = event_time + pd.Timedelta(hours=3)

            pre_bars = h1_df[(h1_df.index >= pre_start) & (h1_df.index < pre_end)]
            post_bars = h1_df[(h1_df.index >= post_start) & (h1_df.index < post_end)]

            if len(pre_bars) < 1 or len(post_bars) < 1:
                continue

            pre_atr = pre_bars['ATR'].mean()
            if pre_atr <= 0:
                continue

            # Pre-event move
            pre_move = (pre_bars['Close'].iloc[-1] - pre_bars['Open'].iloc[0])
            pre_moves.append(pre_move)

            # Post-event move (direction and magnitude)
            post_move = (post_bars['Close'].iloc[-1] - post_bars['Open'].iloc[0])
            post_abs = abs(post_move)
            post_moves.append(post_move)

            # Strategy A: Pre-event straddle (open both 1h before, close 2h after)
            entry_price = pre_bars['Close'].iloc[-1]
            post_high = post_bars['High'].max()
            post_low = post_bars['Low'].min()
            # BUY side: profit = post_high - entry - spread (or SL if hit first)
            buy_pnl = min(post_high - entry_price, 2 * pre_atr) - SPREAD
            # SELL side
            sell_pnl = min(entry_price - post_low, 2 * pre_atr) - SPREAD
            # SL for losing side = 1.0 * ATR
            if post_move > 0:  # market went up
                buy_pnl = min(post_high - entry_price, 3 * pre_atr) - SPREAD
                sell_pnl = max(-(1.0 * pre_atr), entry_price - post_high) - SPREAD
            else:  # market went down
                sell_pnl = min(entry_price - post_low, 3 * pre_atr) - SPREAD
                buy_pnl = max(-(1.0 * pre_atr), post_low - entry_price) - SPREAD

            straddle_pnl = buy_pnl + sell_pnl
            straddle_results.append({
                'date': ed, 'straddle_pnl': straddle_pnl,
                'post_move': post_move, 'post_abs': post_abs,
                'atr': pre_atr
            })

            # Strategy B: Post-event continuation
            if len(post_bars) >= 2:
                first_bar_move = post_bars['Close'].iloc[0] - post_bars['Open'].iloc[0]
                if abs(first_bar_move) > 0.5 * pre_atr:
                    # Follow the direction
                    direction = 'BUY' if first_bar_move > 0 else 'SELL'
                    cont_entry = post_bars['Close'].iloc[0]
                    cont_exit = post_bars['Close'].iloc[-1]
                    if direction == 'BUY':
                        cont_pnl = cont_exit - cont_entry - SPREAD
                    else:
                        cont_pnl = cont_entry - cont_exit - SPREAD
                    continuation_results.append({
                        'date': ed, 'direction': direction,
                        'pnl': cont_pnl, 'move': first_bar_move
                    })

        # Summarize
        if straddle_results:
            straddle_pnls = [r['straddle_pnl'] for r in straddle_results]
            n = len(straddle_pnls)
            total = sum(straddle_pnls)
            wins = sum(1 for p in straddle_pnls if p > 0)
            print(f"  Straddle: N={n}, PnL=${total:.0f}, WR={wins/n:.1%}, Avg=${total/n:.2f}")

            # By year
            by_year = {}
            for r in straddle_results:
                y = r['date'].year
                by_year.setdefault(y, []).append(r['straddle_pnl'])
            print(f"    Year: ", end="")
            for y in sorted(by_year):
                print(f"{y}=${sum(by_year[y]):.0f} ", end="")
            print()

        if continuation_results:
            cont_pnls = [r['pnl'] for r in continuation_results]
            n = len(cont_pnls)
            total = sum(cont_pnls)
            wins = sum(1 for p in cont_pnls if p > 0)
            print(f"  Continuation: N={n}, PnL=${total:.0f}, WR={wins/n:.1%}, Avg=${total/n:.2f}")

        # Post-event magnitude stats
        if post_moves:
            abs_moves = [abs(m) for m in post_moves]
            print(f"  Post-event |move|: mean=${np.mean(abs_moves):.1f}, "
                  f"median=${np.median(abs_moves):.1f}, max=${max(abs_moves):.1f}")


# ═══════════════════════════════════════════════════════════════
# S3: Overnight Return Anomaly
# ═══════════════════════════════════════════════════════════════

def run_s3_overnight(h1_df):
    """Test overnight vs intraday return patterns."""
    print("\n" + "=" * 70)
    print("S3: Overnight Return Anomaly")
    print("=" * 70)

    df = h1_df.copy()
    df['hour'] = df.index.hour
    df['date'] = df.index.date
    df['ret'] = df['Close'].pct_change()

    # Define sessions (UTC)
    # Asia: 0-7, London: 7-13, NY: 13-21, OffHours: 21-24
    sessions = {
        'Asia':    (0, 7),
        'London':  (7, 13),
        'NY':      (13, 21),
        'OffHours': (21, 24),
    }

    # Session returns
    print("\n--- Session Return Analysis ---")
    for sname, (start_h, end_h) in sessions.items():
        mask = (df['hour'] >= start_h) & (df['hour'] < end_h)
        session_bars = df[mask]
        if len(session_bars) < 100:
            continue

        # Group by date, sum returns within session
        daily_rets = session_bars.groupby('date')['ret'].sum()
        cum_ret = daily_rets.sum()
        avg_ret = daily_rets.mean()
        sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
        win_pct = (daily_rets > 0).mean()

        print(f"  {sname:10s}: N_days={len(daily_rets)}, CumRet={cum_ret:.4f} ({cum_ret*100:.2f}%), "
              f"Sharpe={sharpe:.2f}, WinDays={win_pct:.1%}, AvgRet={avg_ret*10000:.2f}bp")

    # Strategy: Buy at NY close (21 UTC), sell at London open (7 UTC)
    print("\n--- Overnight Hold Strategy ---")
    for entry_hour, exit_hour, label in [
        (21, 7, "NYclose_to_LDNopen"),
        (21, 0, "NYclose_to_AsiaOpen"),
        (0, 7, "AsiaOpen_to_LDNopen"),
        (13, 21, "LDN_to_NYclose"),
    ]:
        trades = []
        dates_seen = set()

        for i in range(1, len(df)):
            h = df['hour'].iloc[i]
            d = df['date'].iloc[i]

            if h == entry_hour and d not in dates_seen:
                entry_price = df['Close'].iloc[i]
                entry_time = df.index[i]
                entry_atr = df['ATR'].iloc[i] if df['ATR'].iloc[i] > 0 else 1.0

                # Find exit bar
                for j in range(i + 1, min(i + 30, len(df))):
                    if df['hour'].iloc[j] == exit_hour or (exit_hour < entry_hour and df['hour'].iloc[j] == exit_hour):
                        exit_price = df['Close'].iloc[j]
                        pnl_buy = exit_price - entry_price - SPREAD
                        pnl_sell = entry_price - exit_price - SPREAD

                        trades.append(SimTrade(
                            entry_time=entry_time, direction='BUY',
                            entry_price=entry_price, sl=0, tp=0,
                            strategy=f's3_{label}', exit_time=df.index[j],
                            exit_price=exit_price, pnl=pnl_buy,
                            exit_reason='SessionClose', bars_held=j - i
                        ))
                        dates_seen.add(d)
                        break

        if trades:
            pnls = [t.pnl for t in trades]
            n = len(pnls)
            total = sum(pnls)
            wins = sum(1 for p in pnls if p > 0)
            sharpe = np.mean(pnls) / np.std(pnls, ddof=1) * np.sqrt(252) if np.std(pnls, ddof=1) > 0 else 0
            print(f"  {label:25s}: N={n}, PnL=${total:.0f}, WR={wins/n:.1%}, "
                  f"Sharpe={sharpe:.2f}, Avg=${total/n:.2f}")

            # Year by year
            by_year = {}
            for t in trades:
                y = t.entry_time.year
                by_year.setdefault(y, []).append(t.pnl)
            positive_years = sum(1 for y in by_year if sum(by_year[y]) > 0)
            print(f"    Positive years: {positive_years}/{len(by_year)}")
            print(f"    Year PnL: ", end="")
            for y in sorted(by_year):
                print(f"{y}=${sum(by_year[y]):.0f} ", end="")
            print()


# ═══════════════════════════════════════════════════════════════
# S4: Extreme Reversal / Liquidity Provision
# ═══════════════════════════════════════════════════════════════

def run_s4_extreme_reversal(h1_df, m15_df):
    """Enter contrarian at extreme price displacements."""
    print("\n" + "=" * 70)
    print("S4: Extreme Reversal / Liquidity Provision")
    print("=" * 70)

    df = m15_df.copy()
    if 'ATR' not in df.columns or df['ATR'].isna().all():
        df['ATR'] = df['High'].rolling(14).max() - df['Low'].rolling(14).min()

    # EMA for mean reference
    df['ema50'] = df['Close'].ewm(span=50).mean()
    df['ema100'] = df['Close'].ewm(span=100).mean()
    df['atr_m15'] = (df['High'] - df['Low']).rolling(14).mean()
    df['atr_sma'] = df['atr_m15'].rolling(50).mean()

    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    ema = df['ema100'].values
    atr = df['atr_m15'].values
    atr_sma = df['atr_sma'].values
    times = df.index

    all_results = {}

    for deviation_mult in [2.5, 3.0, 4.0]:
        for sl_mult in [1.0, 1.5, 2.0]:
            for tp_mult in [2.0, 3.0, 4.0]:
                for max_hold in [8, 16, 24]:
                    trades = []
                    cooldown = 0

                    for i in range(120, len(df) - 1):
                        if cooldown > 0:
                            cooldown -= 1
                            continue
                        if atr[i] <= 0 or atr_sma[i] <= 0:
                            continue

                        deviation = (close[i] - ema[i]) / atr[i]
                        atr_spike = atr[i] / atr_sma[i]

                        # Only trade during volatility spikes
                        if atr_spike < 1.5:
                            continue

                        entry_price = close[i]
                        sl_dist = sl_mult * atr[i]
                        tp_dist = tp_mult * atr[i]

                        direction = None
                        if deviation < -deviation_mult:
                            direction = 'BUY'
                            sl = entry_price - sl_dist
                            tp = entry_price + tp_dist
                        elif deviation > deviation_mult:
                            direction = 'SELL'
                            sl = entry_price + sl_dist
                            tp = entry_price - tp_dist

                        if direction is None:
                            continue

                        # Simulate trade
                        exit_price = None
                        reason = ""
                        bars = 0
                        for j in range(i + 1, min(i + max_hold + 1, len(df))):
                            bars += 1
                            h_j, l_j, c_j = high[j], low[j], close[j]

                            if direction == 'BUY':
                                if l_j <= sl:
                                    exit_price = sl
                                    reason = "SL"
                                    break
                                if h_j >= tp:
                                    exit_price = tp
                                    reason = "TP"
                                    break
                            else:
                                if h_j >= sl:
                                    exit_price = sl
                                    reason = "SL"
                                    break
                                if l_j <= tp:
                                    exit_price = tp
                                    reason = "TP"
                                    break

                        if exit_price is None:
                            exit_price = close[min(i + max_hold, len(df) - 1)]
                            reason = "Timeout"

                        if direction == 'BUY':
                            pnl = exit_price - entry_price - SPREAD
                        else:
                            pnl = entry_price - exit_price - SPREAD

                        trades.append(SimTrade(
                            entry_time=times[i], direction=direction,
                            entry_price=entry_price, sl=sl, tp=tp,
                            strategy='s4_reversal', exit_time=times[min(i + bars, len(df) - 1)],
                            exit_price=exit_price, pnl=pnl, exit_reason=reason,
                            bars_held=bars
                        ))
                        cooldown = 4  # 1 hour cooldown

                    label = f"Dev{deviation_mult}_SL{sl_mult}_TP{tp_mult}_MH{max_hold}"
                    if trades and len(trades) >= 20:
                        stats = summarize_trades(trades, label)
                        if stats:
                            all_results[label] = stats

    if all_results:
        best = max(all_results.values(), key=lambda x: x.get('sharpe', 0))
        print(f"\n  Best S4: {best['label']} Sharpe={best['sharpe']:.2f}")

        # Show top 5
        sorted_results = sorted(all_results.values(), key=lambda x: x.get('sharpe', 0), reverse=True)
        print("\n  Top 5 configurations:")
        for r in sorted_results[:5]:
            print(f"    {r['label']}: N={r['n']}, Sharpe={r['sharpe']:.2f}, "
                  f"PnL=${r['total_pnl']:.0f}, WR={r['win_rate']:.1%}")

    return all_results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out_path = OUT_DIR / "R21_full_output.txt"
    out = open(out_path, 'w', encoding='utf-8')

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

    import sys as _sys
    old_stdout = _sys.stdout
    _sys.stdout = Tee(old_stdout, out)

    print(f"# R21: Four New Profit Dimensions")
    print(f"# Started: {ts}")

    # Load data
    from backtest.runner import DataBundle
    data = DataBundle.load_default()
    h1_df = data.h1_df
    m15_df = data.m15_df
    print(f"# H1: {len(h1_df)} bars, M15: {len(m15_df)} bars")

    # S1
    run_s1_squeeze_straddle(h1_df)

    # S2
    run_s2_event_driven(h1_df, m15_df)

    # S3
    run_s3_overnight(h1_df)

    # S4
    run_s4_extreme_reversal(h1_df, m15_df)

    elapsed = time.time() - t_start
    print(f"\n# Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Elapsed: {elapsed/60:.1f} minutes")

    _sys.stdout = old_stdout
    out.close()
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
