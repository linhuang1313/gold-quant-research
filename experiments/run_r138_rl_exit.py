#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R138 — Reinforcement Learning Exit Optimization (PPO)
======================================================
Train an RL agent to decide optimal exit timing for PSAR and TSMOM trades.

State (8 features):
  [bars_held, unrealized_pnl_norm, atr_ratio, close_vs_sma20,
   close_vs_sma50, hour, dow, distance_from_extreme]
Actions: 0=HOLD, 1=EXIT
Reward: at EXIT: realized PnL / entry_atr (normalized)
        per-step penalty: -0.001

Phases:
  1. Load H1 data, run PSAR & TSMOM to generate entry signals
  2. Build RL environment (one episode = one trade)
  3. Train PPO (PyTorch) or tabular Q-learning (fallback)
  4. Walk-Forward: train 2015-2021, test 2021-2026
  5. Compare: RL exit vs fixed-rule exit
  6. Per-strategy results
  7. K-Fold validation

Install: pip install torch  OR  runs with numpy-only Q-learning
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import load_csv

OUTPUT_DIR = Path("results/r138_rl_exit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01
MAX_HOLD = 30
STEP_PENALTY = -0.001
H1_CSV = Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-27.csv")

t0 = time.time()

# ═══════════════════════════════════════════════════════════════
# PyTorch availability
# ═══════════════════════════════════════════════════════════════
HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    HAS_TORCH = True
    print("[INFO] PyTorch available — using PPO agent")
except ImportError:
    print("[INFO] PyTorch not available — falling back to tabular Q-learning")


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_psar(df, af_step=0.01, af_max=0.05):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    n = len(df)
    psar = np.empty(n); psar[:] = np.nan
    af = af_step; rising = True; ep = h[0]; psar[0] = l[0]
    for i in range(1, n):
        prev = psar[i-1]
        if rising:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], l[i-1], l[max(0, i-2)])
            if l[i] < psar[i]:
                rising = False; psar[i] = ep; ep = l[i]; af = af_step
            else:
                if h[i] > ep: ep = h[i]; af = min(af + af_step, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], h[i-1], h[max(0, i-2)])
            if h[i] > psar[i]:
                rising = True; psar[i] = ep; ep = h[i]; af = af_step
            else:
                if l[i] < ep: ep = l[i]; af = min(af + af_step, af_max)
    df['PSAR'] = psar


def _mk(pos, exit_px, exit_time, reason, bar_i, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_px,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_i - pos['bar']}


def _run_fixed_exit(pos, i, hi, lo, cl, spread, lot, pv, times,
                    sl_atr=4.5, tp_atr=6.0, trail_act=0.14, trail_dist=0.025, max_hold=20):
    atr = pos['atr']
    sl = atr * sl_atr; tp = atr * tp_atr; bars = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_now = (cl - pos['entry'] - spread) * lot * pv
        if lo <= pos['entry'] - sl:
            return _mk(pos, pos['entry'] - sl, times[i], "SL", i, -sl * lot * pv)
        if hi >= pos['entry'] + tp:
            return _mk(pos, pos['entry'] + tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry']); extreme = max(extreme, hi); pos['extreme'] = extreme
        if extreme - pos['entry'] >= atr * trail_act:
            trail_price = extreme - atr * trail_dist
            if cl <= trail_price:
                return _mk(pos, trail_price, times[i], "Trail", i, (trail_price - pos['entry'] - spread) * lot * pv)
        if bars >= max_hold:
            return _mk(pos, cl, times[i], "TimeExit", i, pnl_now)
    else:
        pnl_now = (pos['entry'] - cl - spread) * lot * pv
        if hi >= pos['entry'] + sl:
            return _mk(pos, pos['entry'] + sl, times[i], "SL", i, -sl * lot * pv)
        if lo <= pos['entry'] - tp:
            return _mk(pos, pos['entry'] - tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry']); extreme = min(extreme, lo); pos['extreme'] = extreme
        if pos['entry'] - extreme >= atr * trail_act:
            trail_price = extreme + atr * trail_dist
            if cl >= trail_price:
                return _mk(pos, trail_price, times[i], "Trail", i, (pos['entry'] - trail_price - spread) * lot * pv)
        if bars >= max_hold:
            return _mk(pos, cl, times[i], "TimeExit", i, pnl_now)
    return None


def _trades_to_daily(trades):
    if not trades:
        return np.array([0.0])
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    if not daily:
        return np.array([0.0])
    return np.array([daily[k] for k in sorted(daily.keys())])


def _sharpe(arr):
    if len(arr) < 10 or np.std(arr, ddof=1) == 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))


def _max_dd(arr):
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 1 else 0.0


def _compute_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0, 'avg_bars': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    bars = [t['bars'] for t in trades]
    n = len(trades)
    return {
        'n': n,
        'sharpe': round(_sharpe(daily), 3),
        'pnl': round(sum(pnls), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'max_dd': round(_max_dd(daily), 2),
        'avg_bars': round(np.mean(bars), 1),
    }


# ═══════════════════════════════════════════════════════════════
# Phase 1: Load data & generate entry signals
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("R138 — RL Exit Optimization (PPO / Q-Learning)")
print("=" * 70)

print("\n[Phase 1] Loading H1 data & generating entry signals...")

h1 = load_csv(str(H1_CSV))
h1['ATR'] = compute_atr(h1, 14)
h1['SMA20'] = h1['Close'].rolling(20).mean()
h1['SMA50'] = h1['Close'].rolling(50).mean()
h1 = h1.dropna(subset=['ATR', 'SMA20', 'SMA50'])
print(f"  H1 bars: {len(h1)} ({h1.index[0].date()} to {h1.index[-1].date()})")


def generate_psar_entries(df):
    """Generate PSAR entry signals (bar index, direction, atr at entry)."""
    add_psar(df)
    c = df['Close'].values; psar = df['PSAR'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    entries = []
    for i in range(1, n):
        if np.isnan(atr[i]) or atr[i] < 0.1 or np.isnan(psar[i]):
            continue
        if psar[i-1] > c[i-1] and psar[i] < c[i]:
            entries.append({'bar': i, 'dir': 'BUY', 'entry_price': c[i] + SPREAD/2,
                           'atr': atr[i], 'time': times[i]})
        elif psar[i-1] < c[i-1] and psar[i] > c[i]:
            entries.append({'bar': i, 'dir': 'SELL', 'entry_price': c[i] - SPREAD/2,
                           'atr': atr[i], 'time': times[i]})
    return entries


def generate_tsmom_entries(df, fast=480, slow=720):
    """Generate TSMOM entry signals."""
    c = df['Close'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    max_lb = max(fast, slow)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        if c[i-fast] > 0: s += 0.5 * np.sign(c[i]/c[i-fast] - 1.0)
        if c[i-slow] > 0: s += 0.5 * np.sign(c[i]/c[i-slow] - 1.0)
        score[i] = s
    entries = []
    for i in range(max_lb+1, n):
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
        if np.isnan(score[i]) or np.isnan(score[i-1]):
            continue
        if score[i] > 0 and score[i-1] <= 0:
            entries.append({'bar': i, 'dir': 'BUY', 'entry_price': c[i] + SPREAD/2,
                           'atr': atr[i], 'time': times[i]})
        elif score[i] < 0 and score[i-1] >= 0:
            entries.append({'bar': i, 'dir': 'SELL', 'entry_price': c[i] - SPREAD/2,
                           'atr': atr[i], 'time': times[i]})
    return entries


h1_work = h1.copy()
psar_entries = generate_psar_entries(h1_work)
tsmom_entries = generate_tsmom_entries(h1_work)
print(f"  PSAR entries: {len(psar_entries)}")
print(f"  TSMOM entries: {len(tsmom_entries)}")


# ═══════════════════════════════════════════════════════════════
# Phase 2: RL Environment
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 2] Building RL environment...")


class TradeExitEnv:
    """RL environment for exit decision on a single trade.

    State: [bars_held_norm, unrealized_pnl_norm, atr_ratio, close_vs_sma20,
            close_vs_sma50, hour_norm, dow_norm, distance_from_extreme_norm]
    """

    def __init__(self, h1_data, entry_info, max_hold=MAX_HOLD):
        self.h1 = h1_data
        self.entry = entry_info
        self.max_hold = max_hold
        self.c = h1_data['Close'].values
        self.hi = h1_data['High'].values
        self.lo = h1_data['Low'].values
        self.atr = h1_data['ATR'].values
        self.sma20 = h1_data['SMA20'].values
        self.sma50 = h1_data['SMA50'].values
        self.hours = h1_data.index.hour
        self.dows = h1_data.index.dayofweek
        self.times = h1_data.index
        self.n = len(h1_data)
        self.reset()

    def reset(self):
        self.current_bar = self.entry['bar']
        self.bars_held = 0
        self.extreme = self.entry['entry_price']
        self.done = False
        return self._get_state()

    def _get_state(self):
        i = self.current_bar
        if i >= self.n:
            return np.zeros(8)
        entry_atr = self.entry['atr']
        if self.entry['dir'] == 'BUY':
            unrealized = (self.c[i] - self.entry['entry_price'] - SPREAD) / entry_atr
            self.extreme = max(self.extreme, self.hi[i])
            dist_extreme = (self.extreme - self.c[i]) / entry_atr
        else:
            unrealized = (self.entry['entry_price'] - self.c[i] - SPREAD) / entry_atr
            self.extreme = min(self.extreme, self.lo[i])
            dist_extreme = (self.c[i] - self.extreme) / entry_atr

        atr_ratio = self.atr[i] / entry_atr if entry_atr > 0 else 1.0
        close_vs_sma20 = (self.c[i] - self.sma20[i]) / entry_atr if entry_atr > 0 else 0.0
        close_vs_sma50 = (self.c[i] - self.sma50[i]) / entry_atr if entry_atr > 0 else 0.0
        hour_norm = self.hours[i] / 23.0
        dow_norm = self.dows[i] / 4.0

        return np.array([
            self.bars_held / self.max_hold,
            np.clip(unrealized, -5, 5),
            np.clip(atr_ratio, 0.5, 2.0),
            np.clip(close_vs_sma20, -3, 3),
            np.clip(close_vs_sma50, -3, 3),
            hour_norm,
            dow_norm,
            np.clip(dist_extreme, 0, 5),
        ], dtype=np.float32)

    def step(self, action):
        """action: 0=HOLD, 1=EXIT. Returns (next_state, reward, done)."""
        if self.done:
            return np.zeros(8), 0.0, True

        i = self.current_bar
        entry_atr = self.entry['atr']

        if action == 1:  # EXIT
            if self.entry['dir'] == 'BUY':
                pnl = (self.c[i] - self.entry['entry_price'] - SPREAD)
            else:
                pnl = (self.entry['entry_price'] - self.c[i] - SPREAD)
            reward = pnl / entry_atr if entry_atr > 0 else 0.0
            self.done = True
            return np.zeros(8), reward, True

        # HOLD
        self.bars_held += 1
        self.current_bar += 1

        if self.current_bar >= self.n or self.bars_held >= self.max_hold:
            # Forced exit
            exit_bar = min(self.current_bar, self.n - 1)
            if self.entry['dir'] == 'BUY':
                pnl = (self.c[exit_bar] - self.entry['entry_price'] - SPREAD)
            else:
                pnl = (self.entry['entry_price'] - self.c[exit_bar] - SPREAD)
            reward = pnl / entry_atr if entry_atr > 0 else 0.0
            self.done = True
            return np.zeros(8), reward, True

        reward = STEP_PENALTY
        return self._get_state(), reward, False


# ═══════════════════════════════════════════════════════════════
# Phase 3: PPO Agent (or Q-Learning fallback)
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 3] Agent definition...")

if HAS_TORCH:
    class ActorCritic(nn.Module):
        def __init__(self, state_dim=8, hidden1=64, hidden2=32):
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden1), nn.ReLU(),
                nn.Linear(hidden1, hidden2), nn.ReLU(),
                nn.Linear(hidden2, 2), nn.Softmax(dim=-1),
            )
            self.critic = nn.Sequential(
                nn.Linear(state_dim, hidden1), nn.ReLU(),
                nn.Linear(hidden1, hidden2), nn.ReLU(),
                nn.Linear(hidden2, 1),
            )

        def forward(self, x):
            return self.actor(x), self.critic(x)

        def act(self, state):
            state_t = torch.FloatTensor(state).unsqueeze(0)
            probs, value = self.forward(state_t)
            dist = Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action), value.squeeze()

    class PPOAgent:
        def __init__(self, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
            self.model = ActorCritic()
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.gamma = gamma
            self.eps_clip = eps_clip
            self.k_epochs = k_epochs

        def train_on_episodes(self, episodes, batch_size=64):
            """Train PPO on collected episodes."""
            all_states, all_actions, all_rewards, all_log_probs, all_values = [], [], [], [], []
            for ep in episodes:
                states, actions, rewards, log_probs, values = ep
                # Compute discounted returns
                returns = []
                R = 0
                for r in reversed(rewards):
                    R = r + self.gamma * R
                    returns.insert(0, R)
                all_states.extend(states)
                all_actions.extend(actions)
                all_rewards.extend(returns)
                all_log_probs.extend(log_probs)
                all_values.extend(values)

            if len(all_states) < 10:
                return 0.0

            states_t = torch.FloatTensor(np.array(all_states))
            actions_t = torch.LongTensor(all_actions)
            returns_t = torch.FloatTensor(all_rewards)
            old_log_probs_t = torch.stack(all_log_probs).detach()
            old_values_t = torch.stack(all_values).detach()

            advantages = returns_t - old_values_t.squeeze()
            if advantages.std() > 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            total_loss = 0.0
            for _ in range(self.k_epochs):
                probs, values = self.model(states_t)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(actions_t)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_t)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values.squeeze(), returns_t)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

            return total_loss / self.k_epochs

        def select_action(self, state, greedy=False):
            """Select action: 0=HOLD, 1=EXIT."""
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                probs, _ = self.model(state_t)
            if greedy:
                return probs.argmax(dim=-1).item()
            dist = Categorical(probs)
            return dist.sample().item()

        def collect_episode(self, env):
            """Collect one episode (trade) for training."""
            state = env.reset()
            states, actions, rewards, log_probs, values = [], [], [], [], []
            done = False
            while not done:
                action, lp, val = self.model.act(state)
                next_state, reward, done = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(lp)
                values.append(val)
                state = next_state
            return states, actions, rewards, log_probs, values

    print("  PPO: Actor(64,32) + Critic(64,32), eps_clip=0.2, gamma=0.99")

else:
    class QLearningAgent:
        """Tabular Q-learning with discretized state space."""

        def __init__(self, n_bins=5, lr=0.1, gamma=0.99, epsilon=0.3):
            self.n_bins = n_bins
            self.lr = lr
            self.gamma = gamma
            self.epsilon = epsilon
            self.q_table = {}
            self.bins = [
                np.linspace(0, 1, n_bins + 1)[1:-1],       # bars_held_norm
                np.linspace(-3, 3, n_bins + 1)[1:-1],      # unrealized_pnl
                np.linspace(0.5, 2, n_bins + 1)[1:-1],     # atr_ratio
                np.linspace(-2, 2, n_bins + 1)[1:-1],      # close_vs_sma20
                np.linspace(-2, 2, n_bins + 1)[1:-1],      # close_vs_sma50
                np.linspace(0, 1, n_bins + 1)[1:-1],       # hour_norm
                np.linspace(0, 1, n_bins + 1)[1:-1],       # dow_norm
                np.linspace(0, 3, n_bins + 1)[1:-1],       # dist_extreme
            ]

        def _discretize(self, state):
            disc = tuple(int(np.digitize(state[i], self.bins[i])) for i in range(8))
            return disc

        def get_q(self, state_disc, action):
            return self.q_table.get((state_disc, action), 0.0)

        def select_action(self, state, greedy=False):
            state_disc = self._discretize(state)
            if not greedy and np.random.rand() < self.epsilon:
                return np.random.randint(2)
            q0 = self.get_q(state_disc, 0)
            q1 = self.get_q(state_disc, 1)
            return 1 if q1 > q0 else 0

        def update(self, state, action, reward, next_state, done):
            state_disc = self._discretize(state)
            next_disc = self._discretize(next_state)
            q_current = self.get_q(state_disc, action)
            if done:
                q_target = reward
            else:
                q_target = reward + self.gamma * max(self.get_q(next_disc, 0), self.get_q(next_disc, 1))
            self.q_table[(state_disc, action)] = q_current + self.lr * (q_target - q_current)

        def train_on_episode(self, env):
            """Run one episode and update Q-table."""
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
            return total_reward

    print("  Q-Learning: n_bins=5, lr=0.1, gamma=0.99, epsilon=0.3")


# ═══════════════════════════════════════════════════════════════
# Phase 4: Walk-Forward training & testing
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 4] Walk-Forward (train 2015-2021, test 2021-2026)...")

split_date = pd.Timestamp('2021-01-01', tz='UTC')
train_mask = h1.index < split_date
test_mask = h1.index >= split_date

h1_arr_idx = {t: i for i, t in enumerate(h1.index)}


def get_entries_in_period(entries, mask):
    """Filter entries to those within the time mask."""
    filtered = []
    for e in entries:
        if e['bar'] < len(mask) and mask.iloc[e['bar']] if hasattr(mask, 'iloc') else mask[e['bar']]:
            filtered.append(e)
    return filtered


def filter_entries_by_time(entries, start_time, end_time):
    """Filter entries by timestamp range."""
    return [e for e in entries if start_time <= e['time'] < end_time]


def run_fixed_exit_strategy(entries, h1_data, sl_atr=4.5, tp_atr=6.0,
                            trail_act=0.14, trail_dist=0.025, max_hold=20):
    """Run fixed-rule exit on entries."""
    c = h1_data['Close'].values; hi = h1_data['High'].values; lo = h1_data['Low'].values
    times = h1_data.index; n = len(h1_data)
    trades = []
    for entry in entries:
        pos = {'dir': entry['dir'], 'entry': entry['entry_price'],
               'bar': entry['bar'], 'time': entry['time'], 'atr': entry['atr']}
        for i in range(entry['bar'] + 1, min(entry['bar'] + max_hold + 1, n)):
            result = _run_fixed_exit(pos, i, hi[i], lo[i], c[i], SPREAD, UNIT_LOT, PV, times,
                                     sl_atr, tp_atr, trail_act, trail_dist, max_hold)
            if result:
                trades.append(result)
                break
        else:
            if entry['bar'] + max_hold < n:
                exit_i = entry['bar'] + max_hold
                if pos['dir'] == 'BUY':
                    pnl = (c[exit_i] - pos['entry'] - SPREAD) * UNIT_LOT * PV
                else:
                    pnl = (pos['entry'] - c[exit_i] - SPREAD) * UNIT_LOT * PV
                trades.append(_mk(pos, c[exit_i], times[exit_i], "TimeExit", exit_i, pnl))
    return trades


def run_rl_exit_strategy(agent, entries, h1_data, greedy=True):
    """Run RL-based exit on entries."""
    c = h1_data['Close'].values; times = h1_data.index; n = len(h1_data)
    trades = []
    for entry in entries:
        env = TradeExitEnv(h1_data, entry)
        state = env.reset()
        done = False
        exit_bar = entry['bar']
        while not done:
            action = agent.select_action(state, greedy=greedy)
            next_state, reward, done = env.step(action)
            exit_bar = env.current_bar
            state = next_state

        actual_exit = min(exit_bar, n - 1)
        if entry['dir'] == 'BUY':
            pnl = (c[actual_exit] - entry['entry_price'] - SPREAD) * UNIT_LOT * PV
        else:
            pnl = (entry['entry_price'] - c[actual_exit] - SPREAD) * UNIT_LOT * PV
        trades.append({
            'dir': entry['dir'], 'entry': entry['entry_price'], 'exit': c[actual_exit],
            'entry_time': entry['time'], 'exit_time': times[actual_exit],
            'pnl': pnl, 'reason': 'RL_Exit',
            'bars': actual_exit - entry['bar'],
        })
    return trades


# Train RL agent
train_start = h1.index[0]
psar_train = filter_entries_by_time(psar_entries, train_start, split_date)
tsmom_train = filter_entries_by_time(tsmom_entries, train_start, split_date)
psar_test = filter_entries_by_time(psar_entries, split_date, h1.index[-1])
tsmom_test = filter_entries_by_time(tsmom_entries, split_date, h1.index[-1])

print(f"  PSAR train entries: {len(psar_train)}, test: {len(psar_test)}")
print(f"  TSMOM train entries: {len(tsmom_train)}, test: {len(tsmom_test)}")

all_train_entries = psar_train + tsmom_train
print(f"  Training agent on {len(all_train_entries)} episodes...")

if HAS_TORCH:
    agent = PPOAgent(lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4)
    n_epochs = 10
    for epoch in range(n_epochs):
        np.random.shuffle(all_train_entries)
        episodes = []
        for entry in all_train_entries[:500]:  # cap per epoch for speed
            env = TradeExitEnv(h1, entry)
            ep_data = agent.collect_episode(env)
            episodes.append(ep_data)
            if len(episodes) >= 64:
                loss = agent.train_on_episodes(episodes)
                episodes = []
        if episodes:
            loss = agent.train_on_episodes(episodes)
        if (epoch + 1) % 2 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs} done, loss={loss:.4f}")
else:
    agent = QLearningAgent(n_bins=5, lr=0.1, gamma=0.99, epsilon=0.3)
    n_epochs = 5
    for epoch in range(n_epochs):
        np.random.shuffle(all_train_entries)
        total_r = 0
        for entry in all_train_entries:
            env = TradeExitEnv(h1, entry)
            r = agent.train_on_episode(env)
            total_r += r
        avg_r = total_r / max(len(all_train_entries), 1)
        if (epoch + 1) % 2 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}, avg_reward={avg_r:.4f}")
    agent.epsilon = 0.0  # greedy at test time


# ═══════════════════════════════════════════════════════════════
# Phase 5: Compare RL exit vs fixed-rule exit
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 5] Comparing RL exit vs fixed-rule exit (test period)...")

# PSAR
psar_fixed_trades = run_fixed_exit_strategy(psar_test, h1)
psar_rl_trades = run_rl_exit_strategy(agent, psar_test, h1)
psar_fixed_stats = _compute_stats(psar_fixed_trades)
psar_rl_stats = _compute_stats(psar_rl_trades)

print(f"\n  PSAR Fixed Exit: {psar_fixed_stats}")
print(f"  PSAR RL Exit:    {psar_rl_stats}")

# TSMOM
tsmom_fixed_trades = run_fixed_exit_strategy(tsmom_test, h1)
tsmom_rl_trades = run_rl_exit_strategy(agent, tsmom_test, h1)
tsmom_fixed_stats = _compute_stats(tsmom_fixed_trades)
tsmom_rl_stats = _compute_stats(tsmom_rl_trades)

print(f"\n  TSMOM Fixed Exit: {tsmom_fixed_stats}")
print(f"  TSMOM RL Exit:    {tsmom_rl_stats}")


# ═══════════════════════════════════════════════════════════════
# Phase 6: Combined analysis
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 6] Combined portfolio analysis...")

all_fixed_trades = psar_fixed_trades + tsmom_fixed_trades
all_rl_trades = psar_rl_trades + tsmom_rl_trades
combined_fixed_stats = _compute_stats(all_fixed_trades)
combined_rl_stats = _compute_stats(all_rl_trades)

print(f"  Combined Fixed: {combined_fixed_stats}")
print(f"  Combined RL:    {combined_rl_stats}")

# Improvement
pnl_improvement = combined_rl_stats['pnl'] - combined_fixed_stats['pnl']
sharpe_improvement = combined_rl_stats['sharpe'] - combined_fixed_stats['sharpe']
print(f"\n  PnL improvement: ${pnl_improvement:.2f}")
print(f"  Sharpe improvement: {sharpe_improvement:.3f}")


# ═══════════════════════════════════════════════════════════════
# Phase 7: K-Fold validation
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 7] K-Fold validation (5 folds)...")

n_folds = 5
all_entries_sorted = sorted(psar_entries + tsmom_entries, key=lambda x: x['time'])
fold_size = len(all_entries_sorted) // n_folds
kfold_results = []

for fold in range(n_folds):
    test_start_idx = fold * fold_size
    test_end_idx = min((fold + 1) * fold_size, len(all_entries_sorted))
    train_entries_k = all_entries_sorted[:test_start_idx] + all_entries_sorted[test_end_idx:]
    test_entries_k = all_entries_sorted[test_start_idx:test_end_idx]

    if len(train_entries_k) < 50 or len(test_entries_k) < 20:
        continue

    # Train fold agent
    if HAS_TORCH:
        agent_k = PPOAgent(lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4)
        for ep_i in range(3):
            np.random.shuffle(train_entries_k)
            episodes = []
            for entry in train_entries_k[:300]:
                env = TradeExitEnv(h1, entry)
                ep_data = agent_k.collect_episode(env)
                episodes.append(ep_data)
                if len(episodes) >= 64:
                    agent_k.train_on_episodes(episodes)
                    episodes = []
            if episodes:
                agent_k.train_on_episodes(episodes)
    else:
        agent_k = QLearningAgent(n_bins=5, lr=0.1, gamma=0.99, epsilon=0.3)
        for ep_i in range(3):
            np.random.shuffle(train_entries_k)
            for entry in train_entries_k:
                env = TradeExitEnv(h1, entry)
                agent_k.train_on_episode(env)
        agent_k.epsilon = 0.0

    # Evaluate fold
    rl_trades_k = run_rl_exit_strategy(agent_k, test_entries_k, h1)
    fixed_trades_k = run_fixed_exit_strategy(test_entries_k, h1)
    rl_stats_k = _compute_stats(rl_trades_k)
    fixed_stats_k = _compute_stats(fixed_trades_k)

    fold_result = {
        'fold': fold,
        'rl': rl_stats_k,
        'fixed': fixed_stats_k,
        'pnl_delta': round(rl_stats_k['pnl'] - fixed_stats_k['pnl'], 2),
        'sharpe_delta': round(rl_stats_k['sharpe'] - fixed_stats_k['sharpe'], 3),
    }
    kfold_results.append(fold_result)
    print(f"  Fold {fold}: RL Sharpe={rl_stats_k['sharpe']:.3f} vs Fixed={fixed_stats_k['sharpe']:.3f}  "
          f"delta_pnl=${fold_result['pnl_delta']:.2f}")

if kfold_results:
    mean_pnl_delta = np.mean([r['pnl_delta'] for r in kfold_results])
    mean_sharpe_delta = np.mean([r['sharpe_delta'] for r in kfold_results])
    print(f"\n  K-Fold Mean: PnL delta=${mean_pnl_delta:.2f}, Sharpe delta={mean_sharpe_delta:.3f}")
else:
    mean_pnl_delta = 0.0
    mean_sharpe_delta = 0.0


# ═══════════════════════════════════════════════════════════════
# Save results
# ═══════════════════════════════════════════════════════════════
elapsed = time.time() - t0
print(f"\n{'=' * 70}")
print(f"R138 complete in {elapsed:.1f}s")

results = {
    'experiment': 'R138_RL_Exit',
    'agent_type': 'PPO (PyTorch)' if HAS_TORCH else 'Q-Learning (tabular)',
    'state_features': ['bars_held_norm', 'unrealized_pnl_norm', 'atr_ratio',
                       'close_vs_sma20', 'close_vs_sma50', 'hour_norm',
                       'dow_norm', 'distance_from_extreme'],
    'max_hold': MAX_HOLD,
    'step_penalty': STEP_PENALTY,
    'psar': {
        'fixed': psar_fixed_stats,
        'rl': psar_rl_stats,
        'pnl_delta': round(psar_rl_stats['pnl'] - psar_fixed_stats['pnl'], 2),
    },
    'tsmom': {
        'fixed': tsmom_fixed_stats,
        'rl': tsmom_rl_stats,
        'pnl_delta': round(tsmom_rl_stats['pnl'] - tsmom_fixed_stats['pnl'], 2),
    },
    'combined': {
        'fixed': combined_fixed_stats,
        'rl': combined_rl_stats,
        'pnl_improvement': round(pnl_improvement, 2),
        'sharpe_improvement': round(sharpe_improvement, 3),
    },
    'kfold': {
        'folds': kfold_results,
        'mean_pnl_delta': round(mean_pnl_delta, 2),
        'mean_sharpe_delta': round(mean_sharpe_delta, 3),
    },
    'elapsed_s': round(elapsed, 1),
}

with open(OUTPUT_DIR / "r138_results.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR / 'r138_results.json'}")
print("=" * 70)
