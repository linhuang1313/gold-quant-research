"""
Scalper v3 — 模拟截图策略的真正出场逻辑
========================================
核心改进: 去掉固定SL, 用自然出场:
  - TP: 利润达标就跑 ($1.5-2.5)
  - 亏损出场: 价格回到/穿过均值 → 平仓 (小亏)
  - 时间出场: 超过 N bars 未盈利 → 按市价平仓
  - 绝对止损: 只防极端行情 (SL=$8-10)

反弹 Scalper 的入场 + 自然出场 = 更接近截图策略
"""
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.engine import TradeRecord
from backtest.stats import aggregate_daily_pnl


def load_m1(path, start=None, end=None):
    df = pd.read_csv(path)
    df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.000')
    df = df.set_index('Gmt time').sort_index()
    if start: df = df[df.index >= start]
    if end: df = df[df.index <= end]
    return df


@dataclass
class Pos:
    direction: str
    entry_price: float
    entry_time: pd.Timestamp
    lots: float
    idx: int
    sma_at_entry: float


class SmartScalper:
    """
    Smart Scalper — 模拟截图策略出场逻辑。

    入场: 快速移动后逆向 (反弹)
    出场:
      1. TP: 利润达标 → 立即平仓
      2. 均值回归: 价格穿回 SMA → 平仓 (小赚/小亏)
      3. 时间: 超过 max_hold → 按市价平
      4. 硬止损: 防黑天鹅
    """

    def __init__(
        self,
        # 入场
        sma_period: int = 20,
        move_bars: int = 10,
        move_threshold: float = 3.0,
        # 出场
        tp: float = 2.0,
        hard_sl: float = 10.0,
        max_hold: int = 20,
        # 手数
        spread: float = 0.30,
        cooldown: int = 3,
        session_hours: tuple = tuple(range(0, 13)),
        max_per_day: int = 20,
        scale_after: int = 3,
        lots: tuple = (0.01, 0.02, 0.03),
    ):
        self.sma_period = sma_period
        self.move_bars = move_bars
        self.move_threshold = move_threshold
        self.tp = tp
        self.hard_sl = hard_sl
        self.max_hold = max_hold
        self.spread = spread
        self.cooldown = cooldown
        self.session_hours = set(session_hours)
        self.max_per_day = max_per_day
        self.scale_after = scale_after
        self.lots = lots

    def run(self, df):
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        times = df.index
        n = len(df)

        sma = np.full(n, np.nan)
        for i in range(self.sma_period, n):
            sma[i] = np.mean(close[i - self.sma_period:i])

        warmup = max(self.sma_period, self.move_bars) + 2
        trades = []
        equity = [2000.0]
        pos = None
        last_close = -999
        consec_wins = 0
        lot_idx = 0
        daily_cnt = {}
        pct = max(1, n // 20)

        for i in range(warmup, n):
            if i % pct == 0:
                print(f"    {i*100//n}%...", end=" ", flush=True)

            ts = times[i]; c = close[i]; h = high[i]; lo = low[i]
            day = str(ts.date())
            cur_sma = sma[i]
            if np.isnan(cur_sma): continue

            # ── Exit ──
            if pos is not None:
                p = pos
                lm = p.lots / 0.01
                if p.direction == 'BUY':
                    pnl_best = (h - p.entry_price - self.spread) * p.lots * 100
                    pnl_worst = (lo - p.entry_price - self.spread) * p.lots * 100
                    pnl_c = (c - p.entry_price - self.spread) * p.lots * 100
                else:
                    pnl_best = (p.entry_price - lo - self.spread) * p.lots * 100
                    pnl_worst = (p.entry_price - h - self.spread) * p.lots * 100
                    pnl_c = (p.entry_price - c - self.spread) * p.lots * 100

                tp_lvl = self.tp * lm
                sl_lvl = self.hard_sl * lm
                held = i - p.idx

                # 1. TP hit
                if pnl_best >= tp_lvl:
                    self._add(trades, equity, pos, c, ts, "TP", i, tp_lvl)
                    pos = None; last_close = i
                    consec_wins += 1
                    if consec_wins >= self.scale_after:
                        lot_idx = min(lot_idx + 1, len(self.lots) - 1)
                    continue

                # 2. Hard SL (black swan protection)
                if pnl_worst <= -sl_lvl:
                    self._add(trades, equity, pos, c, ts, "HardSL", i, -sl_lvl)
                    pos = None; last_close = i
                    consec_wins = 0; lot_idx = max(0, lot_idx - 1)
                    continue

                # 3. Mean-reversion exit: price crosses back to SMA
                if held >= 3:  # give at least 3 bars
                    if p.direction == 'BUY' and c >= cur_sma:
                        self._add(trades, equity, pos, c, ts, "MR_Exit", i, pnl_c)
                        pos = None; last_close = i
                        if pnl_c > 0: consec_wins += 1
                        else: consec_wins = 0; lot_idx = max(0, lot_idx - 1)
                        continue
                    elif p.direction == 'SELL' and c <= cur_sma:
                        self._add(trades, equity, pos, c, ts, "MR_Exit", i, pnl_c)
                        pos = None; last_close = i
                        if pnl_c > 0: consec_wins += 1
                        else: consec_wins = 0; lot_idx = max(0, lot_idx - 1)
                        continue

                # 4. Timeout
                if held >= self.max_hold:
                    self._add(trades, equity, pos, c, ts, "Timeout", i, pnl_c)
                    pos = None; last_close = i
                    if pnl_c > 0: consec_wins += 1
                    else: consec_wins = 0; lot_idx = max(0, lot_idx - 1)
                    continue

            # ── Entry ──
            if pos is not None: continue
            if i - last_close < self.cooldown: continue
            if ts.hour not in self.session_hours: continue
            if daily_cnt.get(day, 0) >= self.max_per_day: continue

            move = close[i] - close[i - self.move_bars]
            direction = None
            if move > self.move_threshold:
                direction = 'SELL'
            elif move < -self.move_threshold:
                direction = 'BUY'
            if direction is None: continue

            entry_px = c + self.spread / 2 if direction == 'BUY' else c - self.spread / 2
            pos = Pos(direction, entry_px, ts, self.lots[lot_idx], i, cur_sma)
            daily_cnt[day] = daily_cnt.get(day, 0) + 1

        if pos:
            pnl = self._pnl(pos, close[-1])
            self._add(trades, equity, pos, close[-1], times[-1], "EOD", n-1, pnl)

        print(" done!", flush=True)
        return trades, equity

    def _pnl(self, p, ep):
        if p.direction == 'BUY':
            return (ep - p.entry_price - self.spread) * p.lots * 100
        return (p.entry_price - ep - self.spread) * p.lots * 100

    def _add(self, trades, equity, pos, ep, ts, reason, idx, pnl):
        trades.append(TradeRecord(
            strategy="SmartScalp", direction=pos.direction,
            entry_price=pos.entry_price, exit_price=ep,
            entry_time=pos.entry_time, exit_time=ts,
            lots=pos.lots, pnl=pnl, exit_reason=reason,
            bars_held=idx - pos.idx,
        ))
        equity.append(equity[-1] + pnl)


class TrendMicroScalper:
    """
    趋势微结构 Scalper — 顺中期趋势,抢微回调后的继续。

    入场: 中期趋势明确 (SMA 斜率) + 价格短暂回调至 SMA 附近
    方向: 顺趋势
    TP: 小 ($1.5-2.5)
    出场: TP 或价格远离趋势方向 (反转)
    """

    def __init__(
        self,
        trend_sma: int = 50,
        min_slope: float = 0.05,
        pullback_sma: int = 10,
        pullback_dist: float = 1.0,
        tp: float = 2.0,
        hard_sl: float = 8.0,
        max_hold: int = 30,
        spread: float = 0.30,
        cooldown: int = 3,
        session_hours: tuple = tuple(range(0, 13)),
        max_per_day: int = 20,
        scale_after: int = 3,
        lots: tuple = (0.01, 0.02, 0.03),
    ):
        self.trend_sma = trend_sma
        self.min_slope = min_slope
        self.pullback_sma = pullback_sma
        self.pullback_dist = pullback_dist
        self.tp = tp
        self.hard_sl = hard_sl
        self.max_hold = max_hold
        self.spread = spread
        self.cooldown = cooldown
        self.session_hours = set(session_hours)
        self.max_per_day = max_per_day
        self.scale_after = scale_after
        self.lots = lots

    def run(self, df):
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        times = df.index
        n = len(df)

        sma_t = np.full(n, np.nan)
        sma_p = np.full(n, np.nan)
        for i in range(self.trend_sma, n):
            sma_t[i] = np.mean(close[i - self.trend_sma:i])
        for i in range(self.pullback_sma, n):
            sma_p[i] = np.mean(close[i - self.pullback_sma:i])

        warmup = self.trend_sma + 2
        trades = []
        equity = [2000.0]
        pos = None
        last_close = -999
        consec_wins = 0
        lot_idx = 0
        daily_cnt = {}
        pct = max(1, n // 20)

        for i in range(warmup, n):
            if i % pct == 0:
                print(f"    {i*100//n}%...", end=" ", flush=True)

            ts = times[i]; c = close[i]; h = high[i]; lo = low[i]
            day = str(ts.date())
            st = sma_t[i]; sp = sma_p[i]
            if np.isnan(st) or np.isnan(sp): continue

            slope = sma_t[i] - sma_t[i - 5] if i >= 5 and not np.isnan(sma_t[i-5]) else 0

            if pos is not None:
                p = pos
                lm = p.lots / 0.01
                if p.direction == 'BUY':
                    pnl_best = (h - p.entry_price - self.spread) * p.lots * 100
                    pnl_worst = (lo - p.entry_price - self.spread) * p.lots * 100
                    pnl_c = (c - p.entry_price - self.spread) * p.lots * 100
                else:
                    pnl_best = (p.entry_price - lo - self.spread) * p.lots * 100
                    pnl_worst = (p.entry_price - h - self.spread) * p.lots * 100
                    pnl_c = (p.entry_price - c - self.spread) * p.lots * 100

                tp_lvl = self.tp * lm
                sl_lvl = self.hard_sl * lm
                held = i - p.idx

                if pnl_best >= tp_lvl:
                    self._add(trades, equity, pos, c, ts, "TP", i, tp_lvl)
                    pos = None; last_close = i
                    consec_wins += 1
                    if consec_wins >= self.scale_after:
                        lot_idx = min(lot_idx + 1, len(self.lots) - 1)
                    continue
                if pnl_worst <= -sl_lvl:
                    self._add(trades, equity, pos, c, ts, "HardSL", i, -sl_lvl)
                    pos = None; last_close = i
                    consec_wins = 0; lot_idx = max(0, lot_idx - 1)
                    continue

                # Trend reversal exit
                if p.direction == 'BUY' and slope < -self.min_slope:
                    self._add(trades, equity, pos, c, ts, "TrendRev", i, pnl_c)
                    pos = None; last_close = i
                    if pnl_c > 0: consec_wins += 1
                    else: consec_wins = 0; lot_idx = max(0, lot_idx - 1)
                    continue
                elif p.direction == 'SELL' and slope > self.min_slope:
                    self._add(trades, equity, pos, c, ts, "TrendRev", i, pnl_c)
                    pos = None; last_close = i
                    if pnl_c > 0: consec_wins += 1
                    else: consec_wins = 0; lot_idx = max(0, lot_idx - 1)
                    continue

                if held >= self.max_hold:
                    self._add(trades, equity, pos, c, ts, "Timeout", i, pnl_c)
                    pos = None; last_close = i
                    if pnl_c > 0: consec_wins += 1
                    else: consec_wins = 0; lot_idx = max(0, lot_idx - 1)
                    continue

            if pos is not None: continue
            if i - last_close < self.cooldown: continue
            if ts.hour not in self.session_hours: continue
            if daily_cnt.get(day, 0) >= self.max_per_day: continue

            # Trend + pullback entry
            direction = None
            if slope > self.min_slope and c < sp + self.pullback_dist and c > sp - self.pullback_dist * 3:
                direction = 'BUY'
            elif slope < -self.min_slope and c > sp - self.pullback_dist and c < sp + self.pullback_dist * 3:
                direction = 'SELL'
            if direction is None: continue

            entry_px = c + self.spread / 2 if direction == 'BUY' else c - self.spread / 2
            pos = Pos(direction, entry_px, ts, self.lots[lot_idx], i, st)
            daily_cnt[day] = daily_cnt.get(day, 0) + 1

        if pos:
            pnl = self._pnl(pos, close[-1])
            self._add(trades, equity, pos, close[-1], times[-1], "EOD", n-1, pnl)

        print(" done!", flush=True)
        return trades, equity

    def _pnl(self, p, ep):
        if p.direction == 'BUY':
            return (ep - p.entry_price - self.spread) * p.lots * 100
        return (p.entry_price - ep - self.spread) * p.lots * 100

    def _add(self, trades, equity, pos, ep, ts, reason, idx, pnl):
        trades.append(TradeRecord(
            strategy="TrendScalp", direction=pos.direction,
            entry_price=pos.entry_price, exit_price=ep,
            entry_time=pos.entry_time, exit_time=ts,
            lots=pos.lots, pnl=pnl, exit_reason=reason,
            bars_held=idx - pos.idx,
        ))
        equity.append(equity[-1] + pnl)


def report(trades, equity, label):
    if not trades:
        print(f"  {label}: No trades"); return {}
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total = sum(pnls)
    wr = len(wins) / len(pnls) * 100
    avg_w = np.mean(wins) if wins else 0
    avg_l = abs(np.mean(losses)) if losses else 0
    daily = aggregate_daily_pnl(trades)
    sh = 0
    if len(daily) > 1 and np.std(daily, ddof=1) > 0:
        sh = np.mean(daily) / np.std(daily, ddof=1) * np.sqrt(252)
    pk = equity[0]; dd = 0
    for e in equity:
        if e > pk: pk = e
        if pk - e > dd: dd = pk - e

    by_reason = {}
    for t in trades:
        r = t.exit_reason
        if r not in by_reason: by_reason[r] = {'n': 0, 'pnl': 0}
        by_reason[r]['n'] += 1; by_reason[r]['pnl'] += t.pnl

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades: {len(trades):,}")
    print(f"  Total PnL: ${total:,.2f}")
    print(f"  Sharpe: {sh:.2f}")
    print(f"  Win Rate: {wr:.1f}%")
    rr = avg_w / avg_l if avg_l > 0 else 0
    print(f"  Avg Win: ${avg_w:.2f} | Avg Loss: ${avg_l:.2f} | RR: {rr:.2f}")
    print(f"  Max DD: ${dd:,.2f}")
    print(f"  Avg bars held: {np.mean([t.bars_held for t in trades]):.1f}")
    print(f"  Exit reasons:")
    for r, v in sorted(by_reason.items(), key=lambda x: -abs(x[1]['pnl'])):
        print(f"    {r:>10}: N={v['n']:>6}, PnL=${v['pnl']:>10,.2f}")

    year_pnl = {}
    for t in trades:
        y = pd.Timestamp(t.exit_time).year
        if y not in year_pnl: year_pnl[y] = [0, 0.0]
        year_pnl[y][0] += 1; year_pnl[y][1] += t.pnl
    print(f"  Year-by-Year:")
    for y in sorted(year_pnl.keys()):
        ny, p = year_pnl[y]
        print(f"    {y}: N={ny:>5}, PnL=${p:>10,.2f}")

    return {'n': len(trades), 'pnl': total, 'sharpe': sh, 'wr': wr, 'dd': dd, 'rr': rr}


def main():
    t0 = time.time()
    print("# Scalper v3 — Smart Exit + Multiple Entry Modes")
    print(f"# {pd.Timestamp.now()}\n")

    m1_path = "data/download/xauusd-m1-bid-2015-01-01-2026-04-10.csv"
    df = load_m1(m1_path, start="2024-01-01")
    print(f"  Data: {len(df):,} M1 bars\n")

    all_results = []

    # ═══════════════════════════════════════════════════════════
    # A. Smart Scalper (反弹入场 + 均值回归出场)
    # ═══════════════════════════════════════════════════════════
    print("=" * 60)
    print("A. Smart Scalper (Bounce + MR Exit)")
    print("=" * 60)

    smart_configs = [
        dict(sma_period=10, move_bars=5, move_threshold=2.0, tp=1.5, hard_sl=10, max_hold=15,
             cooldown=3, label="Smart: SMA10, Move5/$2, TP$1.5"),
        dict(sma_period=10, move_bars=5, move_threshold=3.0, tp=1.5, hard_sl=10, max_hold=15,
             cooldown=3, label="Smart: SMA10, Move5/$3, TP$1.5"),
        dict(sma_period=20, move_bars=10, move_threshold=3.0, tp=2.0, hard_sl=10, max_hold=20,
             cooldown=3, label="Smart: SMA20, Move10/$3, TP$2"),
        dict(sma_period=20, move_bars=10, move_threshold=4.0, tp=2.0, hard_sl=10, max_hold=20,
             cooldown=3, label="Smart: SMA20, Move10/$4, TP$2"),
        dict(sma_period=20, move_bars=10, move_threshold=5.0, tp=2.5, hard_sl=12, max_hold=25,
             cooldown=5, label="Smart: SMA20, Move10/$5, TP$2.5"),
        dict(sma_period=30, move_bars=15, move_threshold=5.0, tp=2.0, hard_sl=10, max_hold=20,
             cooldown=5, label="Smart: SMA30, Move15/$5, TP$2"),
        dict(sma_period=20, move_bars=10, move_threshold=3.0, tp=1.5, hard_sl=8, max_hold=15,
             cooldown=3, label="Smart: SMA20, Move10/$3, TP$1.5"),
        dict(sma_period=20, move_bars=5, move_threshold=2.5, tp=1.5, hard_sl=10, max_hold=15,
             cooldown=3, label="Smart: SMA20, Move5/$2.5, TP$1.5"),
    ]

    for cfg in smart_configs:
        lbl = cfg.pop('label')
        eng = SmartScalper(**cfg, spread=0.30,
                           session_hours=tuple(range(0, 13)), max_per_day=20)
        tr, eq = eng.run(df)
        res = report(tr, eq, lbl)
        all_results.append((lbl, res))

    # ═══════════════════════════════════════════════════════════
    # B. Trend Micro Scalper (顺趋势抢回调)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("B. Trend Micro Scalper (Trend + Pullback)")
    print("=" * 60)

    trend_configs = [
        dict(trend_sma=50, min_slope=0.05, pullback_sma=10, pullback_dist=1.0,
             tp=2.0, hard_sl=8, max_hold=30, label="Trend: SMA50, Slope>0.05, PB$1"),
        dict(trend_sma=50, min_slope=0.1, pullback_sma=10, pullback_dist=1.5,
             tp=2.0, hard_sl=8, max_hold=30, label="Trend: SMA50, Slope>0.1, PB$1.5"),
        dict(trend_sma=30, min_slope=0.05, pullback_sma=10, pullback_dist=1.0,
             tp=1.5, hard_sl=8, max_hold=20, label="Trend: SMA30, Slope>0.05, TP$1.5"),
        dict(trend_sma=100, min_slope=0.05, pullback_sma=20, pullback_dist=2.0,
             tp=2.5, hard_sl=10, max_hold=40, label="Trend: SMA100, PB$2, TP$2.5"),
        dict(trend_sma=50, min_slope=0.02, pullback_sma=10, pullback_dist=0.5,
             tp=1.5, hard_sl=8, max_hold=20, label="Trend: SMA50, Slope>0.02, PB$0.5"),
    ]

    for cfg in trend_configs:
        lbl = cfg.pop('label')
        eng = TrendMicroScalper(**cfg, spread=0.30, cooldown=3,
                                 session_hours=tuple(range(0, 13)), max_per_day=20)
        tr, eq = eng.run(df)
        res = report(tr, eq, lbl)
        all_results.append((lbl, res))

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY — All ranked by Sharpe")
    print("=" * 60)

    all_results.sort(key=lambda x: x[1].get('sharpe', -999), reverse=True)

    print(f"{'Strategy':<50} {'N':>6} {'PnL':>10} {'Sharpe':>7} {'WR':>6} {'RR':>5} {'DD':>8}")
    print("-" * 95)
    for lbl, res in all_results:
        if res:
            print(f"{lbl:<50} {res['n']:>6,} ${res['pnl']:>9,.2f} {res['sharpe']:>7.2f} {res['wr']:>5.1f}% {res.get('rr',0):>5.2f} ${res['dd']:>7,.2f}")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
