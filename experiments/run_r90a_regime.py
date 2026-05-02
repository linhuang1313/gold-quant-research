#!/usr/bin/env python3
"""
R90a — Macro Regime Detection (Phase A)
========================================
Three independent regime detection methods applied to gold + macro data:

Method 1: Rule-Based scoring of macro factors
Method 2: K-Means / GMM clustering on standardized features
Method 3: Hidden Markov Model (HMM) on returns + macro features

Each method is validated with per-regime stats, cross-regime ANOVA,
regime persistence analysis, and 70/30 out-of-sample tests.
"""
import sys, os, io, time, json, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

OUTPUT_DIR = Path("results/r90_external_data/r90a_regime")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

H1_CSV = Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv")
ALIGNED_CSV = Path("data/external/aligned_daily.csv")

TRAIN_FRAC = 0.70


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_h1_gold():
    """Load H1 gold data and build daily OHLC + returns."""
    print("  Loading H1 gold data...", flush=True)

    # Try the exact filename first, then glob for any matching file
    csv_path = H1_CSV
    if not csv_path.exists():
        import glob
        candidates = glob.glob("data/download/xauusd-h1-bid-*.csv")
        if candidates:
            csv_path = Path(sorted(candidates)[-1])
        else:
            raise FileNotFoundError("No xauusd H1 CSV found in data/download/")

    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('datetime').sort_index()
    df.columns = [c.capitalize() for c in df.columns]

    # Resample to daily OHLC
    daily = df['Close'].resample('1D').agg(
        Open='first', High='max', Low='min', Close='last'
    ).dropna()

    # Use the original OHLC columns for proper high/low
    daily_ohlc = pd.DataFrame({
        'Open': df['Open'].resample('1D').first(),
        'High': df['High'].resample('1D').max(),
        'Low': df['Low'].resample('1D').min(),
        'Close': df['Close'].resample('1D').last(),
    }).dropna()

    daily_ohlc['gold_return'] = daily_ohlc['Close'].pct_change()

    # ATR for HMM features
    tr = pd.DataFrame({
        'hl': daily_ohlc['High'] - daily_ohlc['Low'],
        'hc': (daily_ohlc['High'] - daily_ohlc['Close'].shift(1)).abs(),
        'lc': (daily_ohlc['Low'] - daily_ohlc['Close'].shift(1)).abs(),
    }).max(axis=1)
    daily_ohlc['ATR14'] = tr.rolling(14).mean()
    daily_ohlc['ATR_change'] = daily_ohlc['ATR14'].pct_change(5)

    print(f"    H1 bars: {len(df):,}  ->  Daily bars: {len(daily_ohlc):,}", flush=True)
    print(f"    Range: {daily_ohlc.index[0].date()} ~ {daily_ohlc.index[-1].date()}", flush=True)
    return daily_ohlc


def load_aligned_daily():
    """Load aligned external macro data."""
    print("  Loading aligned macro data...", flush=True)
    df = pd.read_csv(ALIGNED_CSV, parse_dates=['Date'], index_col='Date')
    df.index = pd.to_datetime(df.index, utc=True)
    print(f"    Macro columns: {len(df.columns)}  Rows: {len(df):,}", flush=True)
    print(f"    Range: {df.index[0].date()} ~ {df.index[-1].date()}", flush=True)
    return df


def merge_data(gold_daily, macro_df):
    """Merge gold daily returns with macro data on date."""
    merged = gold_daily.join(macro_df, how='inner')
    merged = merged.dropna(subset=['gold_return'])
    print(f"    Merged rows: {len(merged):,} "
          f"({merged.index[0].date()} ~ {merged.index[-1].date()})", flush=True)
    return merged


# ═══════════════════════════════════════════════════════════════
# Validation helpers
# ═══════════════════════════════════════════════════════════════

def regime_stats(returns, regime_labels, method_name):
    """Per-regime statistics: count, mean, std, t-test vs zero."""
    from scipy import stats as sp_stats
    results = {}
    unique = sorted(regime_labels.dropna().unique())

    for regime in unique:
        mask = regime_labels == regime
        r = returns[mask].dropna()
        n = len(r)
        if n < 2:
            results[str(regime)] = {'n': n, 'mean': 0, 'std': 0, 't_stat': 0, 'p_value': 1.0}
            continue
        mean_r = float(r.mean())
        std_r = float(r.std(ddof=1))
        t_stat, p_val = sp_stats.ttest_1samp(r, 0)
        results[str(regime)] = {
            'n': int(n),
            'mean_daily_return': round(mean_r * 1e4, 4),  # in bps
            'std_daily': round(std_r * 1e4, 4),
            'annualized_return_pct': round(mean_r * 252 * 100, 2),
            't_stat': round(float(t_stat), 3),
            'p_value': round(float(p_val), 4),
        }
    return results


def cross_regime_anova(returns, regime_labels):
    """ANOVA F-test across regimes for return separation."""
    from scipy import stats as sp_stats
    unique = sorted(regime_labels.dropna().unique())
    if len(unique) < 2:
        return {'f_stat': 0, 'p_value': 1.0, 'n_regimes': len(unique)}

    groups = []
    for regime in unique:
        mask = regime_labels == regime
        r = returns[mask].dropna()
        if len(r) > 1:
            groups.append(r.values)

    if len(groups) < 2:
        return {'f_stat': 0, 'p_value': 1.0, 'n_regimes': len(unique)}

    f_stat, p_val = sp_stats.f_oneway(*groups)
    return {
        'f_stat': round(float(f_stat), 3),
        'p_value': round(float(p_val), 6),
        'n_regimes': len(unique),
        'significant_5pct': bool(p_val < 0.05),
    }


def regime_persistence(regime_labels):
    """Mean duration (consecutive days) in each regime state."""
    labels = regime_labels.dropna()
    if len(labels) == 0:
        return {}

    durations = {}
    current = labels.iloc[0]
    run_len = 1

    for i in range(1, len(labels)):
        if labels.iloc[i] == current:
            run_len += 1
        else:
            durations.setdefault(str(current), []).append(run_len)
            current = labels.iloc[i]
            run_len = 1
    durations.setdefault(str(current), []).append(run_len)

    result = {}
    for regime, runs in durations.items():
        result[regime] = {
            'mean_duration_days': round(float(np.mean(runs)), 1),
            'median_duration_days': round(float(np.median(runs)), 1),
            'max_duration_days': int(np.max(runs)),
            'n_transitions': len(runs),
        }
    return result


def out_of_sample_test(returns, regime_labels, train_frac=TRAIN_FRAC):
    """Check if regime-return relationship holds OOS."""
    n = len(returns)
    split = int(n * train_frac)

    train_ret = returns.iloc[:split]
    train_lbl = regime_labels.iloc[:split]
    test_ret = returns.iloc[split:]
    test_lbl = regime_labels.iloc[split:]

    # Compute mean return per regime in-sample
    is_means = {}
    for regime in sorted(train_lbl.dropna().unique()):
        mask = train_lbl == regime
        is_means[str(regime)] = float(train_ret[mask].mean())

    # Compute mean return per regime out-of-sample
    oos_means = {}
    for regime in sorted(test_lbl.dropna().unique()):
        mask = test_lbl == regime
        r = test_ret[mask].dropna()
        oos_means[str(regime)] = float(r.mean()) if len(r) > 0 else 0.0

    # Check rank preservation: do regimes rank the same way?
    common = set(is_means.keys()) & set(oos_means.keys())
    if len(common) < 2:
        rank_preserved = None
    else:
        is_rank = sorted(common, key=lambda k: is_means[k], reverse=True)
        oos_rank = sorted(common, key=lambda k: oos_means[k], reverse=True)
        rank_preserved = is_rank == oos_rank

    # Direction preservation: bullish stays positive, bearish stays negative
    direction_match = 0
    total_compared = 0
    for regime in common:
        total_compared += 1
        if (is_means[regime] > 0) == (oos_means[regime] > 0):
            direction_match += 1

    return {
        'train_size': split,
        'test_size': n - split,
        'in_sample_means_bps': {k: round(v * 1e4, 2) for k, v in is_means.items()},
        'oos_means_bps': {k: round(v * 1e4, 2) for k, v in oos_means.items()},
        'rank_preserved': rank_preserved,
        'direction_match': f"{direction_match}/{total_compared}",
    }


def full_validation(returns, regime_labels, method_name):
    """Run all four validations for a regime method."""
    v = {}
    v['per_regime'] = regime_stats(returns, regime_labels, method_name)
    v['cross_regime_anova'] = cross_regime_anova(returns, regime_labels)
    v['persistence'] = regime_persistence(regime_labels)
    v['out_of_sample'] = out_of_sample_test(returns, regime_labels)
    return v


# ═══════════════════════════════════════════════════════════════
# Method 1: Rule-Based Regime
# ═══════════════════════════════════════════════════════════════

def method_rule_based(merged):
    """
    Score-based regime detection using macro factors.
    Each factor contributes +1 (bullish) or -1 (bearish).
    Threshold: sum >= +2 = Bullish, sum <= -2 = Bearish, else Neutral.
    """
    t0 = time.time()
    print("\n  ── Method 1: Rule-Based Regime ──", flush=True)

    scores = pd.Series(0.0, index=merged.index)

    # 1. Real Yield falling = bullish for gold
    ry_chg = merged.get('REAL_YIELD_Change20')
    if ry_chg is not None:
        scores += np.where(ry_chg < 0, 1, np.where(ry_chg > 0, -1, 0))

    # 2. DXY weakening = bullish for gold
    dxy_mom = merged.get('DXY_Mom20')
    if dxy_mom is not None:
        scores += np.where(dxy_mom < 0, 1, np.where(dxy_mom > 0, -1, 0))

    # 3. High VIX (fear) = bullish for gold (safe haven)
    vix_z = merged.get('VIX_Zscore')
    if vix_z is not None:
        scores += np.where(vix_z > 1.0, 1, np.where(vix_z < -1.0, -1, 0))

    # 4. Real yield level: negative = bullish
    ry_level = merged.get('REAL_YIELD_DFII10')
    if ry_level is not None:
        scores += np.where(ry_level < 0, 1, np.where(ry_level > 1.0, -1, 0))

    # 5. Fed funds direction: loosening = bullish
    fed = merged.get('FED_FUNDS_DFF')
    if fed is not None:
        fed_diff = fed.diff(20)
        scores += np.where(fed_diff < -0.1, 1, np.where(fed_diff > 0.1, -1, 0))

    # 6. Credit stress high = bullish (flight to safety)
    credit = merged.get('CREDIT_STRESS')
    if credit is not None:
        scores += np.where(credit > 0.5, 1, np.where(credit < -0.5, -1, 0))

    # 7. M2 growth = bullish (monetary expansion)
    m2_yoy = merged.get('M2_YoY')
    if m2_yoy is not None:
        scores += np.where(m2_yoy > 5, 1, np.where(m2_yoy < 2, -1, 0))

    # Classify
    regime = pd.Series('Neutral', index=merged.index)
    regime[scores >= 2] = 'Bullish'
    regime[scores <= -2] = 'Bearish'

    counts = regime.value_counts()
    elapsed = time.time() - t0
    print(f"    Regime distribution:", flush=True)
    for r_name in ['Bullish', 'Neutral', 'Bearish']:
        c = counts.get(r_name, 0)
        pct = c / len(regime) * 100
        print(f"      {r_name:>8}: {c:>5} days ({pct:>5.1f}%)", flush=True)
    print(f"    Elapsed: {elapsed:.1f}s", flush=True)

    validation = full_validation(merged['gold_return'], regime, "rule_based")
    return regime, scores, validation, elapsed


# ═══════════════════════════════════════════════════════════════
# Method 2: K-Means / GMM Clustering
# ═══════════════════════════════════════════════════════════════

def method_clustering(merged):
    """
    Unsupervised clustering on standardized macro features.
    Tests K-Means k=3,4,5 and GMM n=3,4,5. Picks best by silhouette score.
    Labels regimes by mean gold return per cluster.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score

    t0 = time.time()
    print("\n  ── Method 2: K-Means / GMM Clustering ──", flush=True)

    feature_cols = [
        'REAL_YIELD_DFII10', 'DXY_Mom20', 'VIX_Zscore', 'CREDIT_STRESS',
        'COPPER_GOLD_RATIO', 'YIELD_CURVE_10Y2Y', 'CRUDE_Mom20',
        'M2_YoY', 'US10Y_Change20', 'USDCNH_Mom20',
    ]

    available = [c for c in feature_cols if c in merged.columns]
    print(f"    Features available: {len(available)}/{len(feature_cols)}", flush=True)
    if len(available) < 3:
        raise ValueError(f"Too few features available: {available}")

    X_raw = merged[available].copy()
    valid_mask = X_raw.notna().all(axis=1)
    X_valid = X_raw[valid_mask]
    returns_valid = merged.loc[valid_mask, 'gold_return']
    print(f"    Valid rows after dropna: {len(X_valid):,} / {len(merged):,}", flush=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    best_score = -1
    best_labels = None
    best_model = None
    best_desc = ""
    all_scores = []

    # K-Means
    for k in [3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        all_scores.append({'method': f'KMeans_k{k}', 'silhouette': round(sil, 4), 'k': k})
        print(f"    KMeans k={k}: silhouette={sil:.4f}", flush=True)
        if sil > best_score:
            best_score = sil
            best_labels = labels
            best_model = km
            best_desc = f"KMeans_k{k}"

    # GMM
    for n in [3, 4, 5]:
        gmm = GaussianMixture(n_components=n, random_state=42, covariance_type='full',
                              max_iter=200, n_init=3)
        gmm.fit(X_scaled)
        labels = gmm.predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        all_scores.append({'method': f'GMM_n{n}', 'silhouette': round(sil, 4), 'k': n})
        print(f"    GMM   n={n}: silhouette={sil:.4f}", flush=True)
        if sil > best_score:
            best_score = sil
            best_labels = labels
            best_model = gmm
            best_desc = f"GMM_n{n}"

    print(f"    >>> Best: {best_desc} (silhouette={best_score:.4f})", flush=True)

    # Label regimes by mean gold return
    label_series = pd.Series(best_labels, index=X_valid.index)
    cluster_means = {}
    for c in sorted(np.unique(best_labels)):
        cluster_means[c] = float(returns_valid[label_series == c].mean())

    sorted_clusters = sorted(cluster_means.keys(), key=lambda c: cluster_means[c])
    label_map = {}
    n_clusters = len(sorted_clusters)
    if n_clusters == 3:
        label_map[sorted_clusters[0]] = 'Bearish'
        label_map[sorted_clusters[1]] = 'Neutral'
        label_map[sorted_clusters[2]] = 'Bullish'
    elif n_clusters == 4:
        label_map[sorted_clusters[0]] = 'Strong_Bearish'
        label_map[sorted_clusters[1]] = 'Bearish'
        label_map[sorted_clusters[2]] = 'Bullish'
        label_map[sorted_clusters[3]] = 'Strong_Bullish'
    elif n_clusters == 5:
        label_map[sorted_clusters[0]] = 'Strong_Bearish'
        label_map[sorted_clusters[1]] = 'Bearish'
        label_map[sorted_clusters[2]] = 'Neutral'
        label_map[sorted_clusters[3]] = 'Bullish'
        label_map[sorted_clusters[4]] = 'Strong_Bullish'

    regime_labels = label_series.map(label_map)

    # Expand to full index (NaN where features were missing)
    full_regime = pd.Series(np.nan, index=merged.index, dtype=object)
    full_regime[valid_mask] = regime_labels.values

    counts = regime_labels.value_counts()
    print(f"    Regime distribution:", flush=True)
    for r_name, cnt in counts.items():
        pct = cnt / len(regime_labels) * 100
        print(f"      {r_name:>15}: {cnt:>5} days ({pct:>5.1f}%)", flush=True)

    elapsed = time.time() - t0
    print(f"    Elapsed: {elapsed:.1f}s", flush=True)

    validation = full_validation(merged['gold_return'], full_regime, "clustering")
    validation['all_scores'] = all_scores
    validation['best_model'] = best_desc
    validation['best_silhouette'] = round(best_score, 4)
    validation['cluster_mean_returns_bps'] = {
        label_map[c]: round(cluster_means[c] * 1e4, 2) for c in sorted_clusters
    }

    model_bundle = {
        'model': best_model,
        'scaler': scaler,
        'feature_cols': available,
        'label_map': label_map,
        'desc': best_desc,
    }

    return full_regime, model_bundle, validation, elapsed


# ═══════════════════════════════════════════════════════════════
# Method 3: Hidden Markov Model (HMM)
# ═══════════════════════════════════════════════════════════════

def method_hmm(merged):
    """
    Gaussian HMM on gold returns + macro features.
    Tests n_components=2,3,4, picks best by BIC.
    Labels states by mean gold return.
    """
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    print("\n  ── Method 3: Hidden Markov Model ──", flush=True)

    feature_cols = ['gold_return', 'ATR_change', 'REAL_YIELD_Change5',
                    'DXY_Mom5', 'VIX_Zscore']

    # Use what's available
    avail_extra = [c for c in feature_cols[1:] if c in merged.columns]
    use_cols = ['gold_return'] + avail_extra
    print(f"    HMM features: {use_cols}", flush=True)

    X_raw = merged[use_cols].copy()
    # ATR_change comes from gold_daily, rest from macro
    valid_mask = X_raw.notna().all(axis=1)
    X_valid = X_raw[valid_mask]
    returns_valid = merged.loc[valid_mask, 'gold_return']
    print(f"    Valid rows: {len(X_valid):,}", flush=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    best_bic = np.inf
    best_model = None
    best_labels = None
    best_n = 0
    all_bics = []

    for n_comp in [2, 3, 4]:
        try:
            model = GaussianHMM(
                n_components=n_comp, covariance_type='full',
                n_iter=200, random_state=42, tol=1e-4,
            )
            model.fit(X_scaled)
            log_likelihood = model.score(X_scaled)
            n_params = n_comp * (n_comp - 1) + n_comp * len(use_cols) + \
                       n_comp * len(use_cols) * (len(use_cols) + 1) // 2
            bic = -2 * log_likelihood * len(X_scaled) + n_params * np.log(len(X_scaled))

            labels = model.predict(X_scaled)
            all_bics.append({
                'n_components': n_comp,
                'bic': round(float(bic), 1),
                'log_likelihood': round(float(log_likelihood), 4),
            })
            print(f"    HMM n={n_comp}: BIC={bic:,.0f}  LogL={log_likelihood:.4f}", flush=True)

            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_labels = labels
                best_n = n_comp
        except Exception as e:
            print(f"    HMM n={n_comp}: FAILED — {e}", flush=True)
            all_bics.append({'n_components': n_comp, 'error': str(e)})

    if best_model is None:
        raise ValueError("All HMM fits failed")

    print(f"    >>> Best: HMM n={best_n} (BIC={best_bic:,.0f})", flush=True)

    # Label states by mean gold return
    label_series = pd.Series(best_labels, index=X_valid.index)
    state_means = {}
    for s in range(best_n):
        state_means[s] = float(returns_valid[label_series == s].mean())

    sorted_states = sorted(state_means.keys(), key=lambda s: state_means[s])
    label_map = {}
    if best_n == 2:
        label_map[sorted_states[0]] = 'Bearish'
        label_map[sorted_states[1]] = 'Bullish'
    elif best_n == 3:
        label_map[sorted_states[0]] = 'Bearish'
        label_map[sorted_states[1]] = 'Neutral'
        label_map[sorted_states[2]] = 'Bullish'
    elif best_n == 4:
        label_map[sorted_states[0]] = 'Strong_Bearish'
        label_map[sorted_states[1]] = 'Bearish'
        label_map[sorted_states[2]] = 'Bullish'
        label_map[sorted_states[3]] = 'Strong_Bullish'

    regime_labels = label_series.map(label_map)

    full_regime = pd.Series(np.nan, index=merged.index, dtype=object)
    full_regime[valid_mask] = regime_labels.values

    counts = regime_labels.value_counts()
    print(f"    Regime distribution:", flush=True)
    for r_name, cnt in counts.items():
        pct = cnt / len(regime_labels) * 100
        print(f"      {r_name:>15}: {cnt:>5} days ({pct:>5.1f}%)", flush=True)

    # Transition matrix
    trans = best_model.transmat_
    print(f"    Transition matrix:", flush=True)
    state_names = [label_map[s] for s in range(best_n)]
    header = "      " + "".join(f"{sn:>15}" for sn in state_names)
    print(header, flush=True)
    for i, sn in enumerate(state_names):
        row = f"      {sn:>15}"
        for j in range(best_n):
            row += f"{trans[i, j]:>15.3f}"
        print(row, flush=True)

    elapsed = time.time() - t0
    print(f"    Elapsed: {elapsed:.1f}s", flush=True)

    validation = full_validation(merged['gold_return'], full_regime, "hmm")
    validation['all_bics'] = all_bics
    validation['best_n_components'] = best_n
    validation['best_bic'] = round(float(best_bic), 1)
    validation['state_mean_returns_bps'] = {
        label_map[s]: round(state_means[s] * 1e4, 2) for s in sorted_states
    }
    validation['transition_matrix'] = {
        state_names[i]: {state_names[j]: round(float(trans[i, j]), 4)
                         for j in range(best_n)}
        for i in range(best_n)
    }

    model_bundle = {
        'model': best_model,
        'scaler': scaler,
        'feature_cols': use_cols,
        'label_map': label_map,
        'desc': f'HMM_n{best_n}',
    }

    return full_regime, model_bundle, validation, elapsed


# ═══════════════════════════════════════════════════════════════
# Summary printer
# ═══════════════════════════════════════════════════════════════

def print_validation_summary(name, val):
    """Print a compact validation summary for one method."""
    print(f"\n    [{name}] Per-Regime Stats:", flush=True)
    print(f"    {'Regime':>15} {'N':>6} {'Mean(bps)':>10} {'Std(bps)':>10} "
          f"{'Ann.Ret%':>9} {'t-stat':>8} {'p-val':>8}", flush=True)
    print(f"    {'-'*15} {'-'*6} {'-'*10} {'-'*10} {'-'*9} {'-'*8} {'-'*8}", flush=True)
    for regime, stats in val['per_regime'].items():
        print(f"    {regime:>15} {stats['n']:>6} "
              f"{stats.get('mean_daily_return', 0):>10.2f} "
              f"{stats.get('std_daily', 0):>10.2f} "
              f"{stats.get('annualized_return_pct', 0):>9.2f} "
              f"{stats.get('t_stat', 0):>8.3f} "
              f"{stats.get('p_value', 1):>8.4f}", flush=True)

    anova = val['cross_regime_anova']
    sig = "YES" if anova.get('significant_5pct') else "NO"
    print(f"    ANOVA: F={anova['f_stat']:.3f}, p={anova['p_value']:.6f} "
          f"(significant at 5%: {sig})", flush=True)

    print(f"    Persistence:", flush=True)
    for regime, p in val['persistence'].items():
        print(f"      {regime:>15}: mean={p['mean_duration_days']:.1f}d, "
              f"median={p['median_duration_days']:.1f}d, "
              f"max={p['max_duration_days']}d, "
              f"transitions={p['n_transitions']}", flush=True)

    oos = val['out_of_sample']
    print(f"    OOS test (train={oos['train_size']}, test={oos['test_size']}):", flush=True)
    print(f"      IS means(bps):  {oos['in_sample_means_bps']}", flush=True)
    print(f"      OOS means(bps): {oos['oos_means_bps']}", flush=True)
    print(f"      Rank preserved: {oos['rank_preserved']}  "
          f"Direction match: {oos['direction_match']}", flush=True)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0_total = time.time()
    print("=" * 80)
    print("  R90a — Macro Regime Detection (Phase A)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80, flush=True)

    # ── Load data ──
    print(f"\n{'='*80}")
    print(f"  Data Loading")
    print(f"{'='*80}\n", flush=True)

    gold_daily = load_h1_gold()
    macro_df = load_aligned_daily()
    merged = merge_data(gold_daily, macro_df)

    all_results = {}
    regime_columns = {}
    best_models = {}
    timings = {}

    # ── Method 1: Rule-Based ──
    try:
        rule_regime, rule_scores, rule_val, rule_time = method_rule_based(merged)
        regime_columns['rule_regime'] = rule_regime
        all_results['method1_rule_based'] = rule_val
        timings['rule_based'] = round(rule_time, 1)
        print_validation_summary("Rule-Based", rule_val)
    except Exception as e:
        print(f"\n  [ERROR] Method 1 (Rule-Based) failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        all_results['method1_rule_based'] = {'error': str(e)}

    # ── Method 2: Clustering ──
    try:
        cluster_regime, cluster_bundle, cluster_val, cluster_time = method_clustering(merged)
        regime_columns['cluster_regime'] = cluster_regime
        best_models['clustering'] = cluster_bundle
        all_results['method2_clustering'] = cluster_val
        timings['clustering'] = round(cluster_time, 1)
        print_validation_summary("Clustering", cluster_val)
    except Exception as e:
        print(f"\n  [ERROR] Method 2 (Clustering) failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        all_results['method2_clustering'] = {'error': str(e)}

    # ── Method 3: HMM ──
    try:
        hmm_regime, hmm_bundle, hmm_val, hmm_time = method_hmm(merged)
        regime_columns['hmm_regime'] = hmm_regime
        best_models['hmm'] = hmm_bundle
        all_results['method3_hmm'] = hmm_val
        timings['hmm'] = round(hmm_time, 1)
        print_validation_summary("HMM", hmm_val)
    except ImportError:
        print(f"\n  [SKIP] Method 3 (HMM): hmmlearn not installed", flush=True)
        all_results['method3_hmm'] = {'error': 'hmmlearn not installed'}
    except Exception as e:
        print(f"\n  [ERROR] Method 3 (HMM) failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        all_results['method3_hmm'] = {'error': str(e)}

    # ── Cross-method agreement ──
    print(f"\n{'='*80}")
    print(f"  Cross-Method Agreement")
    print(f"{'='*80}\n", flush=True)

    if len(regime_columns) >= 2:
        methods = list(regime_columns.keys())
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                m1, m2 = methods[i], methods[j]
                s1 = regime_columns[m1].dropna()
                s2 = regime_columns[m2].dropna()
                common_idx = s1.index.intersection(s2.index)
                if len(common_idx) > 0:
                    # Simplified agreement: map to Bullish/Neutral/Bearish
                    def simplify(s):
                        return s.map(lambda x: 'Bullish' if 'Bullish' in str(x) or 'bullish' in str(x).lower()
                                     else ('Bearish' if 'Bearish' in str(x) or 'bearish' in str(x).lower()
                                           else 'Neutral'))
                    s1_simple = simplify(s1.loc[common_idx])
                    s2_simple = simplify(s2.loc[common_idx])
                    agree = (s1_simple == s2_simple).sum()
                    pct = agree / len(common_idx) * 100
                    print(f"  {m1} vs {m2}: {agree:,}/{len(common_idx):,} agree ({pct:.1f}%)", flush=True)
    else:
        print("  Not enough methods completed for comparison.", flush=True)

    # ── Save outputs ──
    print(f"\n{'='*80}")
    print(f"  Saving Outputs")
    print(f"{'='*80}\n", flush=True)

    # 1. Regime labels CSV
    labels_df = pd.DataFrame({'Date': merged.index, 'gold_return': merged['gold_return'].values})
    labels_df = labels_df.set_index('Date')
    for col_name, col_data in regime_columns.items():
        labels_df[col_name] = col_data.values

    labels_df.to_csv(OUTPUT_DIR / "regime_labels.csv")
    print(f"  Saved: {OUTPUT_DIR}/regime_labels.csv ({len(labels_df)} rows)", flush=True)

    # 2. Results JSON
    elapsed_total = time.time() - t0_total
    output = {
        'config': {
            'h1_csv': str(H1_CSV),
            'aligned_csv': str(ALIGNED_CSV),
            'train_fraction': TRAIN_FRAC,
            'merged_rows': len(merged),
            'date_range': f"{merged.index[0].date()} ~ {merged.index[-1].date()}",
        },
        'timings': timings,
        'methods': all_results,
        'methods_completed': [k for k, v in all_results.items() if 'error' not in v],
        'elapsed_total_s': round(elapsed_total, 1),
    }

    with open(OUTPUT_DIR / "r90a_results.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r90a_results.json", flush=True)

    # 3. Best model pickle
    if best_models:
        with open(OUTPUT_DIR / "regime_model.pkl", 'wb') as f:
            pickle.dump(best_models, f)
        print(f"  Saved: {OUTPUT_DIR}/regime_model.pkl ({list(best_models.keys())})", flush=True)

    # ── Final summary ──
    print(f"\n{'='*80}")
    print(f"  R90a FINAL SUMMARY")
    print(f"{'='*80}\n")

    for method_key, method_val in all_results.items():
        if 'error' in method_val:
            print(f"  {method_key}: FAILED ({method_val['error']})", flush=True)
            continue
        anova = method_val.get('cross_regime_anova', {})
        oos = method_val.get('out_of_sample', {})
        persist = method_val.get('persistence', {})
        n_regimes = anova.get('n_regimes', '?')
        sig = "YES" if anova.get('significant_5pct') else "NO"
        rank = oos.get('rank_preserved', '?')
        dir_match = oos.get('direction_match', '?')

        # Average persistence
        if persist:
            avg_dur = np.mean([p['mean_duration_days'] for p in persist.values()])
        else:
            avg_dur = 0

        print(f"  {method_key}:", flush=True)
        print(f"    Regimes: {n_regimes}  |  ANOVA sig: {sig} (F={anova.get('f_stat', 0):.2f})", flush=True)
        print(f"    OOS rank preserved: {rank}  |  Direction match: {dir_match}", flush=True)
        print(f"    Avg regime duration: {avg_dur:.1f} days", flush=True)

        # Best/worst regime returns
        per_regime = method_val.get('per_regime', {})
        if per_regime:
            best_r = max(per_regime.items(), key=lambda x: x[1].get('annualized_return_pct', 0))
            worst_r = min(per_regime.items(), key=lambda x: x[1].get('annualized_return_pct', 0))
            print(f"    Best regime: {best_r[0]} ({best_r[1].get('annualized_return_pct', 0):.1f}% ann.)", flush=True)
            print(f"    Worst regime: {worst_r[0]} ({worst_r[1].get('annualized_return_pct', 0):.1f}% ann.)", flush=True)

    print(f"\n  Timing breakdown:")
    for method, t in timings.items():
        print(f"    {method:>15}: {t:.1f}s", flush=True)

    print(f"\n{'='*80}")
    print(f"  R90a COMPLETE — {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
