# -*- coding: utf-8 -*-
"""
Experiment U: Keltner channel mean-reversion exit (KC mid / band) before trailing.
Monkey-patches BacktestEngine._check_exits (restored after each run).
"""
import sys
import io
import time
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

import research_config as config
from indicators import check_exit_signal
from backtest.engine import BacktestEngine
from backtest.runner import DataBundle, LIVE_PARITY_KWARGS, run_variant, run_kfold

SPREAD = 0.30
MIN_BARS_GRID = [2, 4, 8, 12, 16]

# Keep reference to original implementation
_ORIGINAL_CHECK_EXITS = BacktestEngine._check_exits


def section(title: str):
    print('\n' + '=' * 72)
    print(f'  {title}')
    print('=' * 72)


def make_patched_check_exits(min_bars: int, mode: str):
    """
    mode: 'mid' -> kc_mid_revert; 'band' -> kc_band_revert (lower/upper).
    Inserted after SL/TP, before Keltner trailing (matches engine ordering intent).
    """
    if mode not in ('mid', 'band'):
        raise ValueError(mode)

    def _check_exits(self, m15_window, h1_window, bar, bar_time):
        high = float(bar['High'])
        low = float(bar['Low'])
        close = float(bar['Close'])

        if self._regime_config and h1_window is not None and len(h1_window) > 0:
            atr_pct = self._get_atr_percentile(h1_window)
            regime = 'low' if atr_pct < 0.30 else ('high' if atr_pct > 0.70 else 'normal')
            rc = self._regime_config.get(regime, {})
            self._trail_act = rc.get('trail_act', self._trail_act_base)
            self._trail_dist = rc.get('trail_dist', self._trail_dist_base)
            self._sl_atr_mult = rc.get('sl', self._sl_atr_mult_base)

        for pos in list(self.positions):
            pos.bars_held += 1
            reason = None
            exit_price = close

            if pos.direction == 'BUY':
                if low <= pos.sl_price:
                    reason = 'SL'
                    exit_price = pos.sl_price
                elif high >= pos.tp_price:
                    reason = 'TP'
                    exit_price = pos.tp_price
            else:
                if high >= pos.sl_price:
                    reason = 'SL'
                    exit_price = pos.sl_price
                elif low <= pos.tp_price:
                    reason = 'TP'
                    exit_price = pos.tp_price

            # ── 1b. KC reversion (before trailing) ─────────────────
            if (
                not reason
                and pos.strategy == 'keltner'
                and pos.bars_held >= min_bars
                and h1_window is not None
                and len(h1_window) > 0
            ):
                h1_last = h1_window.iloc[-1]
                h1_close = float(h1_last['Close'])
                fill = float(bar['Close'])
                if mode == 'mid':
                    kc_m = h1_last.get('KC_mid', None)
                    if kc_m is not None and not pd.isna(kc_m):
                        km = float(kc_m)
                        if pos.direction == 'BUY' and h1_close < km:
                            reason = 'kc_mid_revert'
                            exit_price = fill
                        elif pos.direction == 'SELL' and h1_close > km:
                            reason = 'kc_mid_revert'
                            exit_price = fill
                else:
                    kc_u = h1_last.get('KC_upper', None)
                    kc_l = h1_last.get('KC_lower', None)
                    if kc_u is not None and kc_l is not None and not pd.isna(kc_u) and not pd.isna(kc_l):
                        ku, kl = float(kc_u), float(kc_l)
                        if pos.direction == 'BUY' and h1_close < kl:
                            reason = 'kc_band_revert'
                            exit_price = fill
                        elif pos.direction == 'SELL' and h1_close > ku:
                            reason = 'kc_band_revert'
                            exit_price = fill

            if not reason and pos.strategy == 'keltner' and config.TRAILING_STOP_ENABLED:
                act_atr = self._trail_act or config.TRAILING_ACTIVATE_ATR
                dist_atr = self._trail_dist or config.TRAILING_DISTANCE_ATR
                atr = self._get_h1_atr(h1_window)
                if atr > 0:
                    if (
                        self._atr_spike_protection
                        and pos.entry_atr > 0
                        and atr > pos.entry_atr * self._atr_spike_threshold
                    ):
                        dist_atr = dist_atr * self._atr_spike_trail_mult
                        self.atr_spike_tighten_count += 1

                    if pos.direction == 'BUY':
                        float_profit = high - pos.entry_price
                        pos.extreme_price = max(pos.extreme_price, high)
                    else:
                        float_profit = pos.entry_price - low
                        pos.extreme_price = min(pos.extreme_price, low) if pos.extreme_price > 0 else low

                    if float_profit >= atr * act_atr:
                        trail_distance = atr * dist_atr
                        if pos.direction == 'BUY':
                            new_trail = pos.extreme_price - trail_distance
                            pos.trailing_stop_price = max(pos.trailing_stop_price, new_trail)
                            if low <= pos.trailing_stop_price:
                                reason = 'Trailing'
                                exit_price = pos.trailing_stop_price
                        else:
                            new_trail = pos.extreme_price + trail_distance
                            if pos.trailing_stop_price <= 0:
                                pos.trailing_stop_price = new_trail
                            else:
                                pos.trailing_stop_price = min(pos.trailing_stop_price, new_trail)
                            if high >= pos.trailing_stop_price:
                                reason = 'Trailing'
                                exit_price = pos.trailing_stop_price

            if not reason and pos.strategy == 'keltner':
                pass
            elif not reason and pos.strategy in ('m15_rsi', 'm5_rsi'):
                if pos.bars_held > 1:
                    exit_sig = check_exit_signal(m15_window, pos.strategy, pos.direction)
                    if exit_sig:
                        reason = exit_sig
                        exit_price = close
            elif not reason and pos.strategy not in ('keltner',):
                if h1_window is not None and len(h1_window) > 2:
                    exit_sig = check_exit_signal(h1_window, pos.strategy, pos.direction)
                    if exit_sig:
                        reason = exit_sig
                        exit_price = close

            if (
                not reason
                and self._time_decay_tp
                and pos.strategy == 'keltner'
                and pos.bars_held >= self._td_start_bars
            ):
                trailing_activated = (pos.trailing_stop_price > 0) if pos.direction == 'BUY' else (
                    pos.trailing_stop_price > 0
                )
                if not trailing_activated:
                    atr_td = self._get_h1_atr(h1_window) if h1_window is not None else 0
                    if atr_td > 0:
                        decay_bars = pos.bars_held - self._td_start_bars
                        min_profit_atr = max(0.0, self._td_atr_start - decay_bars * self._td_atr_step_per_bar)
                        min_profit = atr_td * min_profit_atr
                        if pos.direction == 'BUY':
                            float_pnl = close - pos.entry_price
                        else:
                            float_pnl = pos.entry_price - close
                        if float_pnl >= min_profit and float_pnl > 0:
                            reason = 'TimeDecayTP'
                            exit_price = close
                            self.time_decay_tp_count += 1

            if not reason:
                if pos.strategy == 'm15_rsi':
                    max_hold = self._rsi_max_hold_m15 if self._rsi_max_hold_m15 > 0 else 15
                elif pos.strategy == 'orb' and self._orb_max_hold_m15 > 0:
                    max_hold = self._orb_max_hold_m15
                elif pos.strategy == 'keltner' and self._keltner_max_hold_m15 > 0:
                    max_hold = self._keltner_max_hold_m15
                else:
                    max_hold_h1 = config.STRATEGIES.get(pos.strategy, {}).get('max_hold_bars', 15)
                    max_hold = max_hold_h1 * 4
                if pos.bars_held >= max_hold:
                    reason = f'Timeout:{pos.bars_held}>={max_hold}'
                    exit_price = close

            if reason:
                self._close_position(pos, exit_price, bar_time, reason)

    return _check_exits


def exit_reason_breakdown(trades):
    c = Counter()
    for t in trades:
        r = t.exit_reason
        if r.startswith('Timeout'):
            r = 'Timeout'
        c[r] += 1
    total = sum(c.values()) or 1
    rows = []
    for reason, cnt in sorted(c.items(), key=lambda x: -x[1]):
        rows.append((reason, cnt, 100.0 * cnt / total))
    return rows


def format_exit_breakdown_line(trades) -> str:
    parts = []
    for reason, cnt, pct in exit_reason_breakdown(trades):
        parts.append(f'{reason}={cnt}({pct:.0f}%)')
    return ' | '.join(parts)


def print_breakdown_block(label: str, trades):
    print(f'\n  [{label}] exit_reason × count (pct)')
    for reason, cnt, pct in exit_reason_breakdown(trades):
        print(f'    {reason:<22} {cnt:5d}  {pct:5.1f}%')


def run_with_patch(data: DataBundle, label: str, min_bars: int, mode: str, base_kw: dict):
    BacktestEngine._check_exits = make_patched_check_exits(min_bars, mode)
    try:
        return run_variant(data, label, verbose=True, **base_kw)
    finally:
        BacktestEngine._check_exits = _ORIGINAL_CHECK_EXITS


def main():
    t0 = time.time()
    section('Load data')
    data = DataBundle.load_default()

    base_kw = {**LIVE_PARITY_KWARGS, 'spread_cost': SPREAD, 'spread_model': 'fixed'}

    section('Baseline — LIVE_PARITY (standard _check_exits)')
    baseline = run_variant(data, 'baseline_LIVE_PARITY', **base_kw)
    print_breakdown_block('baseline (all strategies)', baseline['_trades'])

    section('KC mid reversion exit — sweep min_bars {}'.format(MIN_BARS_GRID))
    mid_results = []
    for mb in MIN_BARS_GRID:
        label = f'kc_mid_min{mb}'
        st = run_with_patch(data, label, mb, 'mid', base_kw)
        mid_results.append(st)
        print(
            f"    {label}: Sharpe={st['sharpe']:.3f} PnL=${st['total_pnl']:.0f} "
            f"n={st['n']} Keltner n={st['keltner_n']}"
        )
        print(f"       exits: {format_exit_breakdown_line(st['_trades'])}")

    section('KC band reversion (lower/upper) — sweep min_bars {}'.format(MIN_BARS_GRID))
    band_results = []
    for mb in MIN_BARS_GRID:
        label = f'kc_band_min{mb}'
        st = run_with_patch(data, label, mb, 'band', base_kw)
        band_results.append(st)
        print(
            f"    {label}: Sharpe={st['sharpe']:.3f} PnL=${st['total_pnl']:.0f} "
            f"n={st['n']} Keltner n={st['keltner_n']}"
        )
        print(f"       exits: {format_exit_breakdown_line(st['_trades'])}")

    section('SUMMARY — all variants vs baseline')
    all_rows = [('baseline', baseline)] + [(r['label'], r) for r in mid_results + band_results]
    print(f"\n  {'label':<22} {'Sharpe':>8} {'PnL':>12} {'trades':>7} {'keltner_n':>10}")
    print('  ' + '-' * 65)
    base_sh = baseline['sharpe']
    best_alt = None
    best_sh = base_sh
    for name, st in all_rows:
        mark = ''
        if name != 'baseline' and st['sharpe'] > base_sh:
            mark = ' *beats baseline*'
            if st['sharpe'] > best_sh:
                best_sh = st['sharpe']
                best_alt = st
        print(
            f"  {name:<22} {st['sharpe']:8.3f} {st['total_pnl']:12.0f} "
            f"{st['n']:7d} {st['keltner_n']:10d}{mark}"
        )

    section('EXIT REASON BREAKDOWN — full table (all strategies)')
    for name, st in all_rows:
        print(f'\n  --- {name} ---')
        print_breakdown_block(name, st['_trades'])

    section('K-Fold (6-fold) if a variant beats baseline Sharpe')
    if best_alt is None:
        print('  No variant with Sharpe > baseline on full sample — skipping K-fold.')
    else:
        st = best_alt
        name = st['label']
        print(f'  Best beating variant: {name}  Sharpe={st["sharpe"]:.3f} vs baseline {base_sh:.3f}')
        if 'kc_mid_' in name:
            mode = 'mid'
            mb = int(name.split('min')[1])
        else:
            mode = 'band'
            mb = int(name.split('min')[1])

        BacktestEngine._check_exits = make_patched_check_exits(mb, mode)
        try:
            kf = run_kfold(
                data,
                base_kw,
                n_folds=6,
                label_prefix=f'{name}_',
            )
        finally:
            BacktestEngine._check_exits = _ORIGINAL_CHECK_EXITS

        print(f'\n  K-fold mean Sharpe={np.mean([x["sharpe"] for x in kf]):.3f} '
              f'mean PnL=${np.mean([x["total_pnl"] for x in kf]):.0f}')
        for x in kf:
            print(
                f"    {x.get('fold', x['label'])}: Sharpe={x['sharpe']:.3f} "
                f"PnL=${x['total_pnl']:.0f} n={x['n']}"
            )

    print(f"\n  Finished in {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
