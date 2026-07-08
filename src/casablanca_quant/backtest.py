"""Walk-forward factor backtest with Casablanca-grade frictions and rigor.

Engine rules (no lookahead, no fantasy fills):
- signals at close of t are traded at close of t+1 (1-day execution lag);
- long-only (no short selling at the CSE), equal-weight top quantile;
- liquidity filter: a stock must have traded on >= 60% of recent days and
  carry a minimum dirham volume to be eligible;
- transaction costs charged on turnover (default 100 bps per side is realistic
  for Moroccan retail brokerage + spread on thin names).

Rigor (the anti-self-deception stack, same discipline as a real fund):
- baseline: equal-weight portfolio of all eligible stocks (the honest "market");
- bootstrap CI on the Sharpe of active returns vs baseline;
- White's Reality Check across the whole factor family (data-snooping guard);
- Deflated Sharpe Ratio (Bailey & Lopez de Prado): the probability that the
  best Sharpe is real given how many strategies were tried.
"""
from __future__ import annotations

import json
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

from .ingest import load_prices, to_panels
from .paths import BACKTEST_JSON, STRATEGY_RETURNS_CSV, ensure_dirs
from .signals import FACTOR_LIBRARY, TRADING_DAYS


DEFAULT_COST_BPS = 100.0  # per side, conservative for CSE retail
MIN_TRADED_SHARE = 0.60
MIN_DIRHAM_VOLUME = 50_000.0


def eligibility_mask(panels: dict, window: int = 63) -> pd.DataFrame:
    traded_share = panels["traded"].rolling(window, min_periods=20).mean()
    dirham_volume = (panels["volume"] * panels["close"]).rolling(window, min_periods=20).median()
    eligible = traded_share >= MIN_TRADED_SHARE
    if not panels["volume"].isna().all().all():
        eligible &= dirham_volume >= MIN_DIRHAM_VOLUME
    return eligible.fillna(False)


def _weights_from_scores(scores: pd.DataFrame, eligible: pd.DataFrame, rebalance_days: int, top_quantile: float) -> pd.DataFrame:
    """Equal weights on the top quantile of eligible stocks, held between rebalances."""
    masked = scores.where(eligible)
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    rebalance_dates = scores.index[::rebalance_days]
    current = pd.Series(0.0, index=scores.columns)
    for date in scores.index:
        if date in rebalance_dates:
            row = masked.loc[date].dropna()
            if len(row) >= 5:
                n_top = max(3, int(np.ceil(len(row) * top_quantile)))
                top = row.nlargest(n_top).index
                current = pd.Series(0.0, index=scores.columns)
                current[top] = 1.0 / n_top
        weights.loc[date] = current
    return weights


def _net_strategy_returns(weights: pd.DataFrame, returns: pd.DataFrame, cost_bps: float) -> pd.Series:
    # Trade at close of t+1: today's target weights earn from t+1 to t+2.
    lagged = weights.shift(1).fillna(0.0)
    gross = (lagged * returns.fillna(0.0)).sum(axis=1)
    turnover = lagged.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * cost_bps / 10_000.0
    return gross - costs


def _performance(returns: pd.Series) -> dict:
    returns = returns.dropna()
    if len(returns) < 40:
        return {"error": "not enough data"}
    ann_return = float(returns.mean() * TRADING_DAYS)
    ann_vol = float(returns.std(ddof=1) * sqrt(TRADING_DAYS))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    equity = returns.cumsum()
    drawdown = equity - equity.cummax()
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((returns > 0).mean()),
        "days": int(len(returns)),
    }


def _bootstrap_sharpe_diff(active: np.ndarray, n_bootstrap: int = 2000, seed: int = 7) -> dict:
    active = active[~np.isnan(active)]
    if len(active) < 40 or active.std() == 0:
        return {"mean_ann": 0.0, "ci05": 0.0, "ci95": 0.0, "p_le_0": 1.0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(active, size=(n_bootstrap, len(active)), replace=True).mean(axis=1) * TRADING_DAYS
    return {
        "mean_ann": float(active.mean() * TRADING_DAYS),
        "ci05": float(np.quantile(samples, 0.05)),
        "ci95": float(np.quantile(samples, 0.95)),
        "p_le_0": float(np.mean(samples <= 0)),
    }


def deflated_sharpe(returns: pd.Series, n_trials: int) -> dict:
    """Bailey & Lopez de Prado: is the observed Sharpe real after N trials?"""
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < 60 or r.std() == 0:
        return {"dsr": 0.0}
    sr_daily = r.mean() / r.std(ddof=1)
    gamma3 = float(skew(r))
    gamma4 = float(kurtosis(r, fisher=False))
    variance_sr = (1 - gamma3 * sr_daily + (gamma4 - 1) / 4 * sr_daily**2) / (n - 1)
    std_sr = sqrt(max(variance_sr, 1e-12))
    euler = 0.5772156649
    max_z = (1 - euler) * norm.ppf(1 - 1.0 / n_trials) + euler * norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr_threshold = std_sr * max_z
    dsr = float(norm.cdf((sr_daily - sr_threshold) / std_sr))
    return {
        "daily_sharpe": float(sr_daily),
        "annualized_sharpe": float(sr_daily * sqrt(TRADING_DAYS)),
        "threshold_from_n_trials": float(sr_threshold * sqrt(TRADING_DAYS)),
        "n_trials": int(n_trials),
        "dsr": dsr,
        "interpretation": "DSR = probabilite que le Sharpe soit reel apres correction du nombre d'essais; >0.95 = solide.",
    }


def _reality_check(active_matrix: pd.DataFrame, seed: int = 7, block: int = 10, n_bootstrap: int = 1000) -> dict:
    """White's Reality Check on the family of factor active returns vs baseline."""
    data = active_matrix.dropna(how="all").fillna(0.0)
    names = list(data.columns)
    diffs = data.to_numpy()
    n = len(diffs)
    if n < 60:
        return {"skipped": True}
    means = diffs.mean(axis=0)
    statistic = float(sqrt(n) * means.max())
    rng = np.random.default_rng(seed + 99)
    n_blocks = int(np.ceil(n / block))
    stats = np.empty(n_bootstrap)
    for start in range(0, n_bootstrap, 100):
        size = min(100, n_bootstrap - start)
        starts = rng.integers(0, n, size=(size, n_blocks))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
        idx = idx.reshape(size, -1)[:, :n]
        boot_means = diffs[idx].mean(axis=1)
        stats[start : start + size] = sqrt(n) * (boot_means - means).max(axis=1)
    p_value = float((np.sum(stats >= statistic) + 1) / (n_bootstrap + 1))
    return {
        "statistic": statistic,
        "p_value": p_value,
        "best_factor": names[int(means.argmax())],
        "factors_in_family": len(names),
        "interpretation": (
            "p_value = probabilite qu'un 'meilleur facteur' apparent sorte du pur bruit apres avoir "
            "essaye toute la famille. < 0.05 = le meilleur facteur est probablement reel."
        ),
    }


def run_backtest(
    prices: pd.DataFrame | None = None,
    cost_bps: float = DEFAULT_COST_BPS,
    rebalance_days: int = 5,
    top_quantile: float = 0.20,
    warmup: int = 130,
    seed: int = 7,
) -> dict:
    ensure_dirs()
    if prices is None:
        prices = load_prices()
    panels = to_panels(prices)
    returns = panels["returns"]
    eligible = eligibility_mask(panels)

    # Honest baseline: equal-weight everything eligible, same frictions.
    baseline_scores = pd.DataFrame(1.0, index=returns.index, columns=returns.columns)
    baseline_weights = _weights_from_scores(baseline_scores, eligible, rebalance_days, top_quantile=1.0)
    baseline_returns = _net_strategy_returns(baseline_weights, returns, cost_bps).iloc[warmup:]

    results = {}
    daily_returns = {"baseline_equal_weight": baseline_returns}
    active_matrix = {}
    for name, factor in FACTOR_LIBRARY.items():
        scores = factor(panels)
        weights = _weights_from_scores(scores, eligible, rebalance_days, top_quantile)
        net = _net_strategy_returns(weights, returns, cost_bps).iloc[warmup:]
        active = (net - baseline_returns).to_numpy()
        perf = _performance(net)
        perf["active_vs_baseline"] = _bootstrap_sharpe_diff(active, seed=seed)
        perf["avg_daily_turnover"] = float(weights.shift(1).diff().abs().sum(axis=1).iloc[warmup:].mean())
        results[name] = perf
        daily_returns[name] = net
        active_matrix[name] = net - baseline_returns

    daily = pd.DataFrame(daily_returns)
    daily.to_csv(STRATEGY_RETURNS_CSV)

    reality = _reality_check(pd.DataFrame(active_matrix), seed=seed)
    ranking = sorted(results.items(), key=lambda item: item[1].get("sharpe", -9), reverse=True)
    best_name = ranking[0][0]
    deflated = deflated_sharpe(daily[best_name], n_trials=len(FACTOR_LIBRARY))

    # edge_guard equivalent: deploy a factor only if it clears every gate.
    best = results[best_name]
    gates = {
        "sharpe_positive": best.get("sharpe", 0) > 0,
        "beats_baseline_ci": best.get("active_vs_baseline", {}).get("ci05", -1) > 0,
        "reality_check_significant": reality.get("p_value", 1.0) < 0.05,
        "deflated_sharpe_solid": deflated.get("dsr", 0.0) > 0.90,
    }
    deployable = all(gates.values())

    summary = {
        "baseline_equal_weight": _performance(baseline_returns),
        "factors": results,
        "ranking": [name for name, _ in ranking],
        "best_factor": best_name,
        "reality_check": reality,
        "deflated_sharpe_best": deflated,
        "deployment_gates": gates,
        "production_verdict": (
            f"DEPLOYER {best_name}" if deployable
            else "RESTER SUR LE BASELINE: aucun facteur ne passe toutes les portes statistiques."
        ),
        "frictions": {"cost_bps_per_side": cost_bps, "rebalance_days": rebalance_days, "top_quantile": top_quantile},
    }
    BACKTEST_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
