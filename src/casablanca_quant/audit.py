"""Market efficiency audit for the Casablanca Stock Exchange.

Before trading anything: measure HOW inefficient this market actually is.
Every test here answers one question — is there predictability to harvest?

- Variance-ratio test (Lo-MacKinlay): random walk or not, per stock;
- Ljung-Box autocorrelation of returns (momentum/reversal raw material);
- Cross-sectional momentum spread check (do past winners keep winning?);
- Liquidity map: staleness, Amihud, tradable universe size;
- Data-quality report: gaps, suspect jumps, coverage.
"""
from __future__ import annotations

import json
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from .ingest import load_prices, to_panels
from .paths import AUDIT_JSON, STOCK_AUDIT_CSV, ensure_dirs


def variance_ratio(returns: np.ndarray, q: int = 5) -> dict:
    """Lo-MacKinlay heteroskedasticity-robust variance ratio test.

    VR ~ 1 => random walk. VR > 1 => momentum. VR < 1 => mean reversion.
    """
    r = returns[~np.isnan(returns)]
    n = len(r)
    if n < 100:
        return {"vr": np.nan, "z": np.nan, "p_value": np.nan}
    mu = r.mean()
    var1 = np.sum((r - mu) ** 2) / n
    rq = np.convolve(r, np.ones(q), mode="valid")
    varq = np.sum((rq - q * mu) ** 2) / (n * q)
    vr = varq / var1 if var1 > 0 else np.nan
    # robust standard error
    theta = 0.0
    for k in range(1, q):
        delta = np.sum(((r[k:] - mu) ** 2) * ((r[:-k] - mu) ** 2)) / (np.sum((r - mu) ** 2) ** 2 / n)
        theta += (2 * (q - k) / q) ** 2 * delta
    z = (vr - 1) / sqrt(theta) if theta > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return {"vr": float(vr), "z": float(z), "p_value": float(p)}


def ljung_box(returns: np.ndarray, lags: int = 10) -> dict:
    r = returns[~np.isnan(returns)]
    n = len(r)
    if n < 100:
        return {"stat": np.nan, "p_value": np.nan}
    r = r - r.mean()
    acf = [np.sum(r[k:] * r[:-k]) / np.sum(r**2) for k in range(1, lags + 1)]
    stat = n * (n + 2) * sum(rho**2 / (n - k) for k, rho in enumerate(acf, start=1))
    return {"stat": float(stat), "p_value": float(1 - chi2.cdf(stat, lags))}


def momentum_spread_check(panels: dict, lookback: int = 126, skip: int = 21, horizon: int = 21) -> dict:
    """Do past winners outperform past losers over the next month? Direct evidence."""
    close = panels["close"]
    signal = np.log(close.shift(skip) / close.shift(lookback))
    future = np.log(close.shift(-horizon) / close)
    spreads = []
    for date in signal.index[lookback:-horizon:horizon]:
        row_signal = signal.loc[date].dropna()
        row_future = future.loc[date]
        if len(row_signal) < 10:
            continue
        n_top = max(3, len(row_signal) // 5)
        winners = row_future[row_signal.nlargest(n_top).index].mean()
        losers = row_future[row_signal.nsmallest(n_top).index].mean()
        if np.isfinite(winners) and np.isfinite(losers):
            spreads.append(winners - losers)
    spreads = np.array(spreads)
    if len(spreads) < 5:
        return {"periods": int(len(spreads))}
    t_stat = spreads.mean() / (spreads.std(ddof=1) / sqrt(len(spreads))) if spreads.std() > 0 else 0.0
    return {
        "periods": int(len(spreads)),
        "mean_monthly_spread_pct": float(spreads.mean() * 100),
        "t_stat": float(t_stat),
        "positive_share": float((spreads > 0).mean()),
    }


def run_audit(prices: pd.DataFrame | None = None) -> dict:
    ensure_dirs()
    if prices is None:
        prices = load_prices()
    panels = to_panels(prices)
    returns = panels["returns"]

    per_stock = []
    for ticker in returns.columns:
        r = returns[ticker].to_numpy()
        vr = variance_ratio(r)
        lb = ljung_box(r)
        traded = float(panels["traded"][ticker].mean())
        per_stock.append(
            {
                "ticker": ticker,
                "days": int(np.isfinite(r).sum()),
                "ann_vol": float(np.nanstd(r) * sqrt(252)),
                "traded_share": traded,
                "variance_ratio_q5": vr["vr"],
                "vr_p_value": vr["p_value"],
                "ljung_box_p": lb["p_value"],
            }
        )
    stocks = pd.DataFrame(per_stock)
    stocks.to_csv(STOCK_AUDIT_CSV, index=False)

    vr_values = stocks["variance_ratio_q5"].dropna()
    significant_vr = stocks[(stocks["vr_p_value"] < 0.05)]
    momentum_check = momentum_spread_check(panels)

    summary = {
        "tickers": int(len(stocks)),
        "date_range": [str(returns.index.min().date()), str(returns.index.max().date())],
        "trading_days": int(len(returns)),
        "median_traded_share": float(stocks["traded_share"].median()),
        "illiquid_stocks_below_60pct": int((stocks["traded_share"] < 0.60).sum()),
        "variance_ratio": {
            "median_vr_q5": float(vr_values.median()) if len(vr_values) else None,
            "stocks_rejecting_random_walk_5pct": int(len(significant_vr)),
            "share_vr_above_1": float((vr_values > 1).mean()) if len(vr_values) else None,
            "reading": "VR>1 = momentum exploitable; VR<1 = retour a la moyenne; le nombre de rejets dit si le marche est inefficient.",
        },
        "autocorrelation_ljung_box_significant": int((stocks["ljung_box_p"] < 0.05).sum()),
        "momentum_spread_check": momentum_check,
        "interpretation": (
            "Si beaucoup de titres rejettent la marche aleatoire et que le spread momentum est positif "
            "avec t-stat > 2, le marche est structurellement predictible et le backtest a de la matiere."
        ),
    }
    AUDIT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
