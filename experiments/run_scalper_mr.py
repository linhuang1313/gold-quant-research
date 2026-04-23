"""
Mean-Reversion Scalper on M1 — 均值回归高频策略回测
===================================================
核心洞察: 截图策略的82%胜率不来自方向预测,
而是来自M1级别的均值回归微结构:
  - 价格在短期内总会波动 $1-3
  - 只要TP够小($1.5-2.5), 大部分时间都能命中
  - 入场时机才是关键, 而不是方向预测

新策略逻辑:
  1. 计算短期价格偏离 (price vs 短期均值)
  2. 当价格偏离均值过远时, 逆向入场 (均值回归)
  3. TP很小 (快速获利), SL相对较大 (给回归留空间)
  4. 手数根据连续盈利递增
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


class MeanRevScalper:
    """
    均值回归 Scalper.

    入场逻辑:
    - 计算 close 相对于 lookback 期均值的偏离 (z-score)
    - 偏离超过阈值 → 逆向入场 (均值回归方向)
    - 可选: 要求 bar 方向与入场方向一致 (确认反转开始)

    出场:
    - TP: 固定美元
    - SL: 固定美元
    - Timeout: 最大持仓 bars
    """

    def __init__(
        self,
        lookback: int = 20,
        entry_z: float = 1.5,
        tp: float = 2.0,
        sl: float = 5.0,
        spread: float = 0.30,
        cooldown: int = 2,
        session_hours: tuple = tuple(range(0, 13)),
        max_per_day: int = 20,
        max_hold: int = 30,
        scale_after: int = 3,
        lots: tuple = (0.01, 0.02, 0.03),
        confirm_reversal: bool = False,
    ):
        self.lookback = lookback
        self.entry_z = entry_z
        self.tp = tp
        self.sl = sl
        self.spread = spread
        self.cooldown = cooldown
        self.session_hours = set(session_hours)
        self.max_per_day = max_per_day
        self.max_hold = max_hold
        self.scale_after = scale_after
        self.lots = lots
        self.confirm_reversal = confirm_reversal

    def run(self, df: pd.DataFrame) -> Tuple[List[TradeRecord], List[float]]:
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        opn = df['Open'].values
        times = df.index
        n = len(df)

        # Rolling mean and std
        sma = np.full(n, np.nan)
        std = np.full(n, np.nan)
        for i in range(self.lookback, n):
            window = close[i - self.lookback:i]
            sma[i] = np.mean(window)
            std[i] = np.std(window, ddof=1)

        trades = []
        equity = [2000.0]
        pos: Optional[Pos] = None
        last_close = -999
        consec_wins = 0
        lot_idx = 0
        daily_cnt: Dict[str, int] = {}
        pct = max(1, n // 20)

        for i in range(self.lookback + 1, n):
            if i % pct == 0:
                print(f"    {i*100//n}%...", end=" ", flush=True)

            ts = times[i]
            c = close[i]
            h = high[i]
            lo = low[i]
            o = opn[i]
            day = str(ts.date())

            if np.isnan(sma[i]) or np.isnan(std[i]) or std[i] < 0.1:
                continue

            z = (c - sma[i]) / std[i]

            # ── Exit ──
            if pos is not None:
                p = pos
                lm = p.lots / 0.01
                if p.direction == 'BUY':
                    pnl_hi = (h - p.entry_price - self.spread) * p.lots * 100
                    pnl_lo = (lo - p.entry_price - self.spread) * p.lots * 100
                    pnl_c = (c - p.entry_price - self.spread) * p.lots * 100
                else:
                    pnl_hi = (p.entry_price - lo - self.spread) * p.lots * 100
                    pnl_lo = (p.entry_price - h - self.spread) * p.lots * 100
                    pnl_c = (p.entry_price - c - self.spread) * p.lots * 100

                tp_lvl = self.tp * lm
                sl_lvl = self.sl * lm
                held = i - p.idx

                if pnl_hi >= tp_lvl:
                    self._close_pos(trades, equity, pos, c, ts, "TP", i, tp_lvl)
                    pos = None; last_close = i
                    consec_wins += 1
                    if consec_wins >= self.scale_after:
                        lot_idx = min(lot_idx + 1, len(self.lots) - 1)
                    continue

                if pnl_lo <= -sl_lvl:
                    self._close_pos(trades, equity, pos, c, ts, "SL", i, -sl_lvl)
                    pos = None; last_close = i
                    consec_wins = 0; lot_idx = max(0, lot_idx - 1)
                    continue

                if held >= self.max_hold:
                    self._close_pos(trades, equity, pos, c, ts, "Timeout", i, pnl_c)
                    pos = None; last_close = i
                    if pnl_c > 0: consec_wins += 1
                    else: consec_wins = 0; lot_idx = max(0, lot_idx - 1)
                    continue

            # ── Entry ──
            if pos is not None:
                continue
            if i - last_close < self.cooldown:
                continue
            if ts.hour not in self.session_hours:
                continue
            if daily_cnt.get(day, 0) >= self.max_per_day:
                continue

            direction = None
            if z > self.entry_z:
                direction = 'SELL'  # 价格偏高 → 做空 (回归均值)
            elif z < -self.entry_z:
                direction = 'BUY'  # 价格偏低 → 做多

            if direction is None:
                continue

            if self.confirm_reversal:
                bar_dir = c - o
                if direction == 'BUY' and bar_dir < 0:
                    continue
                if direction == 'SELL' and bar_dir > 0:
                    continue

            entry_px = c + self.spread / 2 if direction == 'BUY' else c - self.spread / 2
            pos = Pos(direction, entry_px, ts, self.lots[lot_idx], i)
            daily_cnt[day] = daily_cnt.get(day, 0) + 1

        if pos is not None:
            pnl = self._pnl(pos, close[-1])
            self._close_pos(trades, equity, pos, close[-1], times[-1], "EOD", n-1, pnl)

        print(" done!", flush=True)
        return trades, equity

    def _pnl(self, p, exit_px):
        if p.direction == 'BUY':
            return (exit_px - p.entry_price - self.spread) * p.lots * 100
        return (p.entry_price - exit_px - self.spread) * p.lots * 100

    def _close_pos(self, trades, equity, pos, exit_px, ts, reason, idx, pnl):
        trades.append(TradeRecord(
            strategy="MRScalp", direction=pos.direction,
            entry_price=pos.entry_price, exit_price=exit_px,
            entry_time=pos.entry_time, exit_time=ts,
            lots=pos.lots, pnl=pnl, exit_reason=reason,
            bars_held=idx - pos.idx,
        ))
        equity.append(equity[-1] + pnl)


class MomentumMicroScalper:
    """
    动量微结构 Scalper — 顺短期动量入场，抢小波段。

    入场: 短期动量 (过去 N bars 的 close 变化) 超过阈值
    方向: 顺动量方向
    TP: 极小 (抢到就跑)
    SL: 较大 (给方向继续的空间)
    """

    def __init__(
        self,
        mom_bars: int = 5,
        mom_threshold: float = 1.0,
        tp: float = 1.5,
        sl: float = 4.0,
        spread: float = 0.30,
        cooldown: int = 3,
        session_hours: tuple = tuple(range(0, 13)),
        max_per_day: int = 20,
        max_hold: int = 20,
        scale_after: int = 3,
        lots: tuple = (0.01, 0.02, 0.03),
    ):
        self.mom_bars = mom_bars
        self.mom_threshold = mom_threshold
        self.tp = tp
        self.sl = sl
        self.spread = spread
        self.cooldown = cooldown
        self.session_hours = set(session_hours)
        self.max_per_day = max_per_day
        self.max_hold = max_hold
        self.scale_after = scale_after
        self.lots = lots

    def run(self, df):
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        times = df.index
        n = len(df)

        trades = []
        equity = [2000.0]
        pos = None
        last_close = -999
        consec_wins = 0
        lot_idx = 0
        daily_cnt = {}
        pct = max(1, n // 20)

        for i in range(self.mom_bars + 1, n):
            if i % pct == 0:
                print(f"    {i*100//n}%...", end=" ", flush=True)

            ts = times[i]
            c = close[i]; h = high[i]; lo = low[i]
            day = str(ts.date())

            mom = close[i] - close[i - self.mom_bars]

            if pos is not None:
                p = pos
                lm = p.lots / 0.01
                if p.direction == 'BUY':
                    pnl_hi = (h - p.entry_price - self.spread) * p.lots * 100
                    pnl_lo = (lo - p.entry_price - self.spread) * p.lots * 100
                    pnl_c = (c - p.entry_price - self.spread) * p.lots * 100
                else:
                    pnl_hi = (p.entry_price - lo - self.spread) * p.lots * 100
                    pnl_lo = (p.entry_price - h - self.spread) * p.lots * 100
                    pnl_c = (p.entry_price - c - self.spread) * p.lots * 100

                tp_lvl = self.tp * lm
                sl_lvl = self.sl * lm
                held = i - p.idx

                if pnl_hi >= tp_lvl:
                    trades.append(self._tr(pos, c, ts, "TP", i, tp_lvl))
                    equity.append(equity[-1] + tp_lvl)
                    pos = None; last_close = i
                    consec_wins += 1
                    if consec_wins >= self.scale_after:
                        lot_idx = min(lot_idx + 1, len(self.lots) - 1)
                    continue
                if pnl_lo <= -sl_lvl:
                    trades.append(self._tr(pos, c, ts, "SL", i, -sl_lvl))
                    equity.append(equity[-1] - sl_lvl)
                    pos = None; last_close = i
                    consec_wins = 0; lot_idx = max(0, lot_idx - 1)
                    continue
                if held >= self.max_hold:
                    trades.append(self._tr(pos, c, ts, "Timeout", i, pnl_c))
                    equity.append(equity[-1] + pnl_c)
                    pos = None; last_close = i
                    if pnl_c > 0: consec_wins += 1
                    else: consec_wins = 0; lot_idx = max(0, lot_idx - 1)
                    continue

            if pos is not None: continue
            if i - last_close < self.cooldown: continue
            if ts.hour not in self.session_hours: continue
            if daily_cnt.get(day, 0) >= self.max_per_day: continue

            direction = None
            if mom > self.mom_threshold:
                direction = 'BUY'
            elif mom < -self.mom_threshold:
                direction = 'SELL'
            if direction is None: continue

            entry_px = c + self.spread / 2 if direction == 'BUY' else c - self.spread / 2
            pos = Pos(direction, entry_px, ts, self.lots[lot_idx], i)
            daily_cnt[day] = daily_cnt.get(day, 0) + 1

        if pos:
            pnl = (close[-1] - pos.entry_price - self.spread) * pos.lots * 100 if pos.direction == 'BUY' \
                else (pos.entry_price - close[-1] - self.spread) * pos.lots * 100
            trades.append(self._tr(pos, close[-1], times[-1], "EOD", n-1, pnl))
            equity.append(equity[-1] + pnl)

        print(" done!", flush=True)
        return trades, equity

    def _tr(self, pos, exit_px, ts, reason, idx, pnl):
        return TradeRecord(
            strategy="MomScalp", direction=pos.direction,
            entry_price=pos.entry_price, exit_price=exit_px,
            entry_time=pos.entry_time, exit_time=ts,
            lots=pos.lots, pnl=pnl, exit_reason=reason,
            bars_held=idx - pos.idx,
        )


class BounceScalper:
    """
    反弹 Scalper — 在快速移动后抢反弹。

    入场: 过去 N bars 单方向移动超过阈值 → 逆向入场
    核心: 快速移动后的短暂回调几乎是确定性的
    TP: 很小 (只抢回调的一小部分)
    SL: 较大
    """

    def __init__(
        self,
        move_bars: int = 10,
        move_threshold: float = 3.0,
        tp: float = 1.5,
        sl: float = 5.0,
        spread: float = 0.30,
        cooldown: int = 5,
        session_hours: tuple = tuple(range(0, 13)),
        max_per_day: int = 15,
        max_hold: int = 15,
        scale_after: int = 3,
        lots: tuple = (0.01, 0.02, 0.03),
    ):
        self.move_bars = move_bars
        self.move_threshold = move_threshold
        self.tp = tp
        self.sl = sl
        self.spread = spread
        self.cooldown = cooldown
        self.session_hours = set(session_hours)
        self.max_per_day = max_per_day
        self.max_hold = max_hold
        self.scale_after = scale_after
        self.lots = lots

    def run(self, df):
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        times = df.index
        n = len(df)

        trades = []
        equity = [2000.0]
        pos = None
        last_close = -999
        consec_wins = 0
        lot_idx = 0
        daily_cnt = {}
        pct = max(1, n // 20)

        for i in range(self.move_bars + 1, n):
            if i % pct == 0:
                print(f"    {i*100//n}%...", end=" ", flush=True)

            ts = times[i]
            c = close[i]; h = high[i]; lo = low[i]
            day = str(ts.date())

            move = close[i] - close[i - self.move_bars]

            if pos is not None:
                p = pos
                lm = p.lots / 0.01
                if p.direction == 'BUY':
                    pnl_hi = (h - p.entry_price - self.spread) * p.lots * 100
                    pnl_lo = (lo - p.entry_price - self.spread) * p.lots * 100
                    pnl_c = (c - p.entry_price - self.spread) * p.lots * 100
                else:
                    pnl_hi = (p.entry_price - lo - self.spread) * p.lots * 100
                    pnl_lo = (p.entry_price - h - self.spread) * p.lots * 100
                    pnl_c = (p.entry_price - c - self.spread) * p.lots * 100

                tp_lvl = self.tp * lm
                sl_lvl = self.sl * lm
                held = i - p.idx

                if pnl_hi >= tp_lvl:
                    trades.append(self._tr(pos, c, ts, "TP", i, tp_lvl))
                    equity.append(equity[-1] + tp_lvl)
                    pos = None; last_close = i
                    consec_wins += 1
                    if consec_wins >= self.scale_after:
                        lot_idx = min(lot_idx + 1, len(self.lots) - 1)
                    continue
                if pnl_lo <= -sl_lvl:
                    trades.append(self._tr(pos, c, ts, "SL", i, -sl_lvl))
                    equity.append(equity[-1] - sl_lvl)
                    pos = None; last_close = i
                    consec_wins = 0; lot_idx = max(0, lot_idx - 1)
                    continue
                if held >= self.max_hold:
                    trades.append(self._tr(pos, c, ts, "Timeout", i, pnl_c))
                    equity.append(equity[-1] + pnl_c)
                    pos = None; last_close = i
                    if pnl_c > 0: consec_wins += 1
                    else: consec_wins = 0; lot_idx = max(0, lot_idx - 1)
                    continue

            if pos is not None: continue
            if i - last_close < self.cooldown: continue
            if ts.hour not in self.session_hours: continue
            if daily_cnt.get(day, 0) >= self.max_per_day: continue

            # 逆向入场: 快速上涨后做空, 快速下跌后做多
            direction = None
            if move > self.move_threshold:
                direction = 'SELL'
            elif move < -self.move_threshold:
                direction = 'BUY'
            if direction is None: continue

            entry_px = c + self.spread / 2 if direction == 'BUY' else c - self.spread / 2
            pos = Pos(direction, entry_px, ts, self.lots[lot_idx], i)
            daily_cnt[day] = daily_cnt.get(day, 0) + 1

        if pos:
            pnl = (close[-1] - pos.entry_price - self.spread) * pos.lots * 100 if pos.direction == 'BUY' \
                else (pos.entry_price - close[-1] - self.spread) * pos.lots * 100
            trades.append(self._tr(pos, close[-1], times[-1], "EOD", n-1, pnl))
            equity.append(equity[-1] + pnl)

        print(" done!", flush=True)
        return trades, equity

    def _tr(self, pos, exit_px, ts, reason, idx, pnl):
        return TradeRecord(
            strategy="Bounce", direction=pos.direction,
            entry_price=pos.entry_price, exit_price=exit_px,
            entry_time=pos.entry_time, exit_time=ts,
            lots=pos.lots, pnl=pnl, exit_reason=reason,
            bars_held=idx - pos.idx,
        )


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
    if avg_l > 0:
        print(f"  Avg Win: ${avg_w:.2f} | Avg Loss: ${avg_l:.2f} | RR: {avg_w/avg_l:.2f}")
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

    return {'n': len(trades), 'pnl': total, 'sharpe': sh, 'wr': wr, 'dd': dd}


def main():
    t0 = time.time()
    print("# Mean-Reversion / Momentum / Bounce Scalper — M1 Backtest")
    print(f"# {pd.Timestamp.now()}\n")

    m1_path = "data/download/xauusd-m1-bid-2015-01-01-2026-04-10.csv"
    df = load_m1(m1_path, start="2024-01-01")
    print(f"  Data: {len(df):,} bars\n")

    # ═══════════════════════════════════════════════════════════
    # A. Mean-Reversion Scalper 参数扫描
    # ═══════════════════════════════════════════════════════════
    print("=" * 60)
    print("A. Mean-Reversion Scalper")
    print("=" * 60)

    mr_configs = [
        dict(lookback=10, entry_z=1.5, tp=1.5, sl=5.0, max_hold=20, label="MR: LB=10, Z=1.5, TP=$1.5, SL=$5"),
        dict(lookback=10, entry_z=2.0, tp=1.5, sl=5.0, max_hold=20, label="MR: LB=10, Z=2.0, TP=$1.5, SL=$5"),
        dict(lookback=20, entry_z=1.5, tp=2.0, sl=5.0, max_hold=30, label="MR: LB=20, Z=1.5, TP=$2, SL=$5"),
        dict(lookback=20, entry_z=2.0, tp=2.0, sl=6.0, max_hold=30, label="MR: LB=20, Z=2.0, TP=$2, SL=$6"),
        dict(lookback=30, entry_z=1.5, tp=2.0, sl=6.0, max_hold=40, label="MR: LB=30, Z=1.5, TP=$2, SL=$6"),
        dict(lookback=30, entry_z=2.0, tp=2.5, sl=6.0, max_hold=40, label="MR: LB=30, Z=2.0, TP=$2.5, SL=$6"),
        dict(lookback=20, entry_z=1.5, tp=1.5, sl=4.0, max_hold=20, label="MR: LB=20, Z=1.5, TP=$1.5, SL=$4"),
        dict(lookback=20, entry_z=1.0, tp=1.5, sl=5.0, max_hold=20, label="MR: LB=20, Z=1.0, TP=$1.5, SL=$5"),
    ]

    mr_results = []
    for cfg in mr_configs:
        lbl = cfg.pop('label')
        eng = MeanRevScalper(**cfg, spread=0.30, cooldown=2,
                             session_hours=tuple(range(0, 13)), max_per_day=20)
        tr, eq = eng.run(df)
        res = report(tr, eq, lbl)
        mr_results.append((lbl, res))

    # ═══════════════════════════════════════════════════════════
    # B. Momentum Micro Scalper
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("B. Momentum Micro Scalper")
    print("=" * 60)

    mom_configs = [
        dict(mom_bars=3, mom_threshold=1.0, tp=1.5, sl=4.0, max_hold=15, label="Mom: N=3, T=$1, TP=$1.5, SL=$4"),
        dict(mom_bars=5, mom_threshold=1.5, tp=1.5, sl=4.0, max_hold=20, label="Mom: N=5, T=$1.5, TP=$1.5, SL=$4"),
        dict(mom_bars=5, mom_threshold=2.0, tp=2.0, sl=5.0, max_hold=20, label="Mom: N=5, T=$2, TP=$2, SL=$5"),
        dict(mom_bars=10, mom_threshold=2.0, tp=2.0, sl=5.0, max_hold=30, label="Mom: N=10, T=$2, TP=$2, SL=$5"),
        dict(mom_bars=10, mom_threshold=3.0, tp=2.5, sl=5.0, max_hold=30, label="Mom: N=10, T=$3, TP=$2.5, SL=$5"),
    ]

    mom_results = []
    for cfg in mom_configs:
        lbl = cfg.pop('label')
        eng = MomentumMicroScalper(**cfg, spread=0.30, cooldown=3,
                                    session_hours=tuple(range(0, 13)), max_per_day=20)
        tr, eq = eng.run(df)
        res = report(tr, eq, lbl)
        mom_results.append((lbl, res))

    # ═══════════════════════════════════════════════════════════
    # C. Bounce Scalper (快速移动后抢反弹)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("C. Bounce Scalper (抢反弹)")
    print("=" * 60)

    bounce_configs = [
        dict(move_bars=5, move_threshold=2.0, tp=1.5, sl=5.0, max_hold=10, label="Bounce: N=5, T=$2, TP=$1.5, SL=$5"),
        dict(move_bars=5, move_threshold=3.0, tp=1.5, sl=5.0, max_hold=10, label="Bounce: N=5, T=$3, TP=$1.5, SL=$5"),
        dict(move_bars=10, move_threshold=3.0, tp=2.0, sl=5.0, max_hold=15, label="Bounce: N=10, T=$3, TP=$2, SL=$5"),
        dict(move_bars=10, move_threshold=4.0, tp=2.0, sl=6.0, max_hold=15, label="Bounce: N=10, T=$4, TP=$2, SL=$6"),
        dict(move_bars=10, move_threshold=5.0, tp=2.5, sl=6.0, max_hold=20, label="Bounce: N=10, T=$5, TP=$2.5, SL=$6"),
        dict(move_bars=15, move_threshold=5.0, tp=2.0, sl=5.0, max_hold=15, label="Bounce: N=15, T=$5, TP=$2, SL=$5"),
        dict(move_bars=20, move_threshold=5.0, tp=2.0, sl=6.0, max_hold=20, label="Bounce: N=20, T=$5, TP=$2, SL=$6"),
    ]

    bounce_results = []
    for cfg in bounce_configs:
        lbl = cfg.pop('label')
        eng = BounceScalper(**cfg, spread=0.30, cooldown=5,
                            session_hours=tuple(range(0, 13)), max_per_day=15)
        tr, eq = eng.run(df)
        res = report(tr, eq, lbl)
        bounce_results.append((lbl, res))

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY — All strategies ranked by Sharpe")
    print("=" * 60)

    all_res = mr_results + mom_results + bounce_results
    all_res.sort(key=lambda x: x[1].get('sharpe', -999), reverse=True)

    print(f"{'Strategy':<55} {'Trades':>7} {'PnL':>10} {'Sharpe':>7} {'WR':>6} {'DD':>8}")
    print("-" * 100)
    for lbl, res in all_res:
        if res:
            print(f"{lbl:<55} {res['n']:>7,} ${res['pnl']:>9,.2f} {res['sharpe']:>7.2f} {res['wr']:>5.1f}% ${res['dd']:>7,.2f}")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s")
    print("Done!")


if __name__ == "__main__":
    main()
