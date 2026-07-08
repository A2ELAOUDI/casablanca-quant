"""Production portfolio construction for the CSE.

Takes the backtest's deployment verdict and builds TODAY's portfolio:
- if a factor cleared every statistical gate, tilt toward it;
- otherwise hold the honest baseline (equal-weight liquid universe);
- position sizes are volatility-scaled, capped per name, liquidity-capped
  (never hold more than a few days of median volume).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .backtest import DEFAULT_COST_BPS, eligibility_mask
from .ingest import load_prices, to_panels
from .paths import BACKTEST_JSON, CURRENT_PORTFOLIO_CSV, PORTFOLIO_JSON, ensure_dirs
from .signals import FACTOR_LIBRARY


MAX_WEIGHT = 0.12
ADV_DAYS_CAP = 3.0  # max position = 3 days of median dirham volume


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """Hierarchical Risk Parity (Lopez de Prado) on a Ledoit-Wolf shrunk covariance.

    Robust to small samples where Markowitz explodes: cluster stocks by
    correlation distance, then split risk budget recursively between clusters
    by inverse cluster variance.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    from sklearn.covariance import LedoitWolf

    clean = returns.dropna(axis=1, thresh=int(len(returns) * 0.6)).fillna(0.0)
    if clean.shape[1] == 0:
        return pd.Series(dtype=float)
    if clean.shape[1] == 1:
        return pd.Series([1.0], index=clean.columns)

    lw = LedoitWolf().fit(clean.to_numpy())
    cov = pd.DataFrame(lw.covariance_, index=clean.columns, columns=clean.columns)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr.to_numpy()), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="single")
    ordered = [clean.columns[idx] for idx in leaves_list(link)]

    weights = pd.Series(1.0, index=ordered)
    stack = [ordered]
    while stack:
        cluster = stack.pop()
        if len(cluster) <= 1:
            continue
        split = len(cluster) // 2
        left, right = cluster[:split], cluster[split:]

        def cluster_variance(items: list) -> float:
            sub = cov.loc[items, items].to_numpy()
            inverse = 1.0 / np.clip(np.diag(sub), 1e-10, None)
            inverse /= inverse.sum()
            return float(inverse @ sub @ inverse)

        var_left, var_right = cluster_variance(left), cluster_variance(right)
        alloc_left = 1.0 - var_left / (var_left + var_right) if (var_left + var_right) > 0 else 0.5
        weights[left] *= alloc_left
        weights[right] *= 1.0 - alloc_left
        stack.extend([left, right])
    return weights / weights.sum()


def build_portfolio(
    prices: pd.DataFrame | None = None,
    capital_mad: float = 100_000.0,
    top_quantile: float = 0.20,
) -> pd.DataFrame:
    ensure_dirs()
    if prices is None:
        prices = load_prices()
    panels = to_panels(prices)
    eligible = eligibility_mask(panels).iloc[-1]

    backtest = json.loads(BACKTEST_JSON.read_text(encoding="utf-8")) if BACKTEST_JSON.exists() else {}
    gates = backtest.get("deployment_gates", {})
    factor_name = backtest.get("best_factor") if gates and all(gates.values()) else None

    if factor_name:
        scores = FACTOR_LIBRARY[factor_name](panels).iloc[-1]
        mode = f"factor:{factor_name}"
    else:
        scores = pd.Series(1.0, index=eligible.index)
        mode = "baseline_equal_weight (aucun facteur ne passe les portes statistiques)"

    scores = scores.where(eligible).dropna()
    n_top = max(5, int(np.ceil(len(scores) * top_quantile))) if factor_name else len(scores)
    chosen = scores.nlargest(n_top)

    recent_returns = panels["returns"][chosen.index].tail(180)
    weights = hrp_weights(recent_returns)
    if weights.empty or not np.isfinite(weights.to_numpy()).all():
        vol = panels["returns"].rolling(63).std().iloc[-1]
        inverse_vol = (1.0 / vol[chosen.index].clip(lower=0.004)).fillna(0.0)
        weights = inverse_vol / inverse_vol.sum()
    weights = weights.clip(upper=MAX_WEIGHT)
    weights = weights / weights.sum()

    dirham_volume = (panels["volume"] * panels["close"]).rolling(63).median().iloc[-1]
    close = panels["close"].iloc[-1]
    rows = []
    for ticker, weight in weights.sort_values(ascending=False).items():
        target_mad = float(weight * capital_mad)
        adv_cap = float(dirham_volume.get(ticker, np.nan) * ADV_DAYS_CAP)
        if np.isfinite(adv_cap):
            target_mad = min(target_mad, adv_cap)
        price = float(close.get(ticker, np.nan))
        rows.append(
            {
                "ticker": ticker,
                "weight_pct": round(100 * target_mad / capital_mad, 2),
                "target_mad": round(target_mad, 0),
                "last_price": round(price, 2) if np.isfinite(price) else None,
                "shares": int(target_mad / price) if np.isfinite(price) and price > 0 else None,
            }
        )
    portfolio = pd.DataFrame(rows)
    portfolio = portfolio[portfolio["target_mad"] > 0]
    portfolio.to_csv(CURRENT_PORTFOLIO_CSV, index=False)

    report = {
        "as_of": str(panels["close"].index[-1].date()),
        "mode": mode,
        "capital_mad": capital_mad,
        "positions": int(len(portfolio)),
        "invested_pct": float(portfolio["weight_pct"].sum()),
        "estimated_entry_cost_pct": float(DEFAULT_COST_BPS / 100.0),
        "note": (
            "Volatilite-scaled, plafonne a 12% par ligne et 3 jours de volume median par position. "
            "Le mode baseline signifie: pas de signal valide, on tient le marche liquide a moindre cout."
        ),
    }
    PORTFOLIO_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return portfolio
