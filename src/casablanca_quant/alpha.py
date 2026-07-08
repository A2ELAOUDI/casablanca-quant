"""Alpha engine: ensemble learning with fund-grade validation discipline.

Techniques implemented, borrowed from institutional quant practice:
- a feature factory over every (date, stock) pair, strictly point-in-time;
- an ensemble of learners (ridge + gradient boosting) rather than one model;
- purged walk-forward validation with an embargo (Lopez de Prado): the train
  set never contains targets that overlap the test window — the main source
  of fake backtests;
- forecasts shrunk by out-of-sample skill (information coefficient): if the
  models have no proven skill, predictions collapse to the market average and
  say so — no false confidence;
- multi-horizon outputs (1M/3M/6M/12M) with residual-based prediction intervals;
- market regime detection (Gaussian mixture on index returns);
- a quarterly top-10 evaluation matching a buy-and-hold-per-quarter cadence.
"""
from __future__ import annotations

import json
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .ingest import load_prices, to_panels
from .paths import ALPHA_JSON, FORECASTS_CSV, TOP10_CSV, ensure_dirs


HORIZONS = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}
FEATURES = [
    "mom_21",
    "mom_63",
    "mom_126",
    "reversal_5",
    "vol_63",
    "vol_ratio",
    "dist_52w_high",
    "amihud",
    "volume_shock",
    "beta_63",
    "market_mom_63",
    "market_vol_21",
]
REFIT_EVERY = 42  # retrain cadence (2 months)
MIN_TRAIN = 150


def _market_series(returns: pd.DataFrame) -> pd.Series:
    return returns.mean(axis=1)


def feature_matrix(panels: dict) -> pd.DataFrame:
    """Point-in-time features per (date, ticker). Everything uses data <= t."""
    close, returns, volume = panels["close"], panels["returns"], panels["volume"]
    market = _market_series(returns)

    def stack(frame: pd.DataFrame, name: str) -> pd.Series:
        out = frame.stack()
        out.name = name
        return out

    def mom(lookback: int, skip: int = 0) -> pd.DataFrame:
        return np.log(close.shift(skip) / close.shift(lookback))

    vol63 = returns.rolling(63).std()
    features = [
        stack(mom(21), "mom_21"),
        stack(mom(63), "mom_63"),
        stack(mom(126, 21), "mom_126"),
        stack(-returns.rolling(5).sum(), "reversal_5"),
        stack(vol63, "vol_63"),
        stack(returns.rolling(21).std() / vol63, "vol_ratio"),
        stack(close / close.rolling(252, min_periods=120).max(), "dist_52w_high"),
        stack(
            np.log1p((returns.abs() / (volume * close).replace(0, np.nan)).rolling(63, min_periods=20).mean() * 1e6),
            "amihud",
        ),
        stack(np.log(volume.replace(0, np.nan).rolling(5).mean() / volume.replace(0, np.nan).rolling(63, min_periods=20).mean()), "volume_shock"),
    ]
    beta = returns.rolling(63).cov(market).div(market.rolling(63).var(), axis=0)
    features.append(stack(beta, "beta_63"))
    market_mom = pd.DataFrame({col: market.rolling(63).sum() for col in close.columns}, index=close.index)
    market_vol = pd.DataFrame({col: market.rolling(21).std() for col in close.columns}, index=close.index)
    features.append(stack(market_mom, "market_mom_63"))
    features.append(stack(market_vol, "market_vol_21"))

    matrix = pd.concat(features, axis=1)
    matrix.index.names = ["date", "ticker"]
    return matrix


def forward_returns(panels: dict, horizon: int) -> pd.Series:
    close = panels["close"]
    target = np.log(close.shift(-horizon) / close).stack()
    target.name = f"fwd_{horizon}"
    return target


def _fit_ensemble(x_train: np.ndarray, y_train: np.ndarray):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    ridge.fit(x_train, y_train)
    boost = HistGradientBoostingRegressor(max_depth=3, max_iter=120, learning_rate=0.05, l2_regularization=1.0, random_state=0)
    boost.fit(x_train, y_train)
    return ridge, boost


def _predict_ensemble(models, x: np.ndarray) -> np.ndarray:
    ridge, boost = models
    return 0.5 * ridge.predict(x) + 0.5 * boost.predict(x)


def purged_walk_forward(matrix: pd.DataFrame, target: pd.Series, horizon: int) -> pd.DataFrame:
    """Expanding-window walk-forward with an embargo of `horizon` days.

    At refit date T we train on samples whose TARGET window ends before T
    (dates <= T - horizon), then predict dates [T, T + REFIT_EVERY). No target
    overlap between train and test — the purge that kills lookahead.
    """
    data = matrix.join(target, how="inner").dropna()
    dates = data.index.get_level_values("date")
    unique_dates = dates.unique().sort_values()
    if len(unique_dates) < MIN_TRAIN + horizon + 10:
        return pd.DataFrame(columns=["prediction", "actual"])

    predictions = []
    refit_points = unique_dates[MIN_TRAIN + horizon :: REFIT_EVERY]
    for refit_date in refit_points:
        train_mask = dates <= (refit_date - pd.Timedelta(days=int(horizon * 1.6)))
        window_end_idx = min(len(unique_dates) - 1, unique_dates.get_loc(refit_date) + REFIT_EVERY - 1)
        window_end = unique_dates[window_end_idx]
        test_mask = (dates >= refit_date) & (dates <= window_end)
        if train_mask.sum() < 200 or test_mask.sum() == 0:
            continue
        x_train = data.loc[train_mask, FEATURES].to_numpy(dtype=float)
        y_train = data.loc[train_mask, target.name].to_numpy(dtype=float)
        models = _fit_ensemble(x_train, y_train)
        x_test = data.loc[test_mask, FEATURES].to_numpy(dtype=float)
        preds = _predict_ensemble(models, x_test)
        chunk = pd.DataFrame(
            {"prediction": preds, "actual": data.loc[test_mask, target.name].to_numpy(dtype=float)},
            index=data.index[test_mask],
        )
        predictions.append(chunk)
    if not predictions:
        return pd.DataFrame(columns=["prediction", "actual"])
    return pd.concat(predictions)


def information_coefficient(oos: pd.DataFrame) -> dict:
    """Out-of-sample skill: rank correlation prediction vs realized."""
    if len(oos) < 50:
        return {"ic": 0.0, "t_stat": 0.0, "n": int(len(oos))}
    by_date = []
    for _, group in oos.groupby(level="date"):
        if len(group) >= 8:
            rho = spearmanr(group["prediction"], group["actual"]).statistic
            if np.isfinite(rho):
                by_date.append(rho)
    if len(by_date) >= 20:
        ics = np.array(by_date)
        t_stat = ics.mean() / (ics.std(ddof=1) / sqrt(len(ics))) if ics.std() > 0 else 0.0
        return {"ic": float(ics.mean()), "t_stat": float(t_stat), "n": int(len(ics)), "mode": "cross_sectional"}
    rho = spearmanr(oos["prediction"], oos["actual"]).statistic
    n_eff = max(2.0, len(oos) / 21.0)  # overlapping targets: crude effective n
    t_stat = float(rho) * sqrt(n_eff - 2) / sqrt(max(1e-9, 1 - rho**2)) if np.isfinite(rho) else 0.0
    return {"ic": float(rho) if np.isfinite(rho) else 0.0, "t_stat": float(t_stat), "n": int(len(oos)), "mode": "pooled_time_series"}


def _skill_shrinkage(ic: float) -> float:
    """Forecast weight as a function of proven OOS skill. IC<=0 -> zero trust."""
    return float(np.clip(ic * 5.0, 0.0, 0.5))


def market_regime(panels: dict) -> dict:
    from sklearn.mixture import GaussianMixture

    market = _market_series(panels["returns"]).dropna()
    if len(market) < 120:
        return {"regime": "inconnu"}
    x = market.to_numpy().reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=0).fit(x)
    vols = np.sqrt(gm.covariances_.ravel())
    calm_state = int(np.argmin(vols))
    probs = gm.predict_proba(market.tail(21).to_numpy().reshape(-1, 1)).mean(axis=0)
    calm_prob = float(probs[calm_state])
    return {
        "regime": "CALME" if calm_prob > 0.5 else "STRESS",
        "calm_probability": calm_prob,
        "calm_ann_vol": float(vols[calm_state] * sqrt(252)),
        "stress_ann_vol": float(vols[1 - calm_state] * sqrt(252)),
        "note": "En regime STRESS, reduire l'exposition: les correlations montent et le momentum casse.",
    }


def quarterly_top10_backtest(matrix: pd.DataFrame, panels: dict, oos_63: pd.DataFrame) -> dict:
    """The user's real cadence: every quarter, buy the top-10 forecast, hold 63 days."""
    if oos_63.empty:
        return {"skipped": True}
    market_fwd = forward_returns(panels, 63)
    results = []
    dates = oos_63.index.get_level_values("date").unique().sort_values()
    for date in dates[::63]:
        day = oos_63.xs(date, level="date")
        if len(day) < 12:
            continue
        top = day.nlargest(10, "prediction")["actual"].mean()
        universe = day["actual"].mean()
        results.append(top - universe)
    if len(results) < 3:
        return {"skipped": True, "reason": "univers trop petit ou historique trop court"}
    excess = np.array(results)
    t_stat = excess.mean() / (excess.std(ddof=1) / sqrt(len(excess))) if excess.std() > 0 else 0.0
    return {
        "quarters": int(len(excess)),
        "mean_quarterly_excess_pct": float(excess.mean() * 100),
        "t_stat": float(t_stat),
        "positive_share": float((excess > 0).mean()),
    }


def run_alpha(prices: pd.DataFrame | None = None) -> dict:
    ensure_dirs()
    if prices is None:
        prices = load_prices()
    panels = to_panels(prices)
    matrix = feature_matrix(panels)
    market = _market_series(panels["returns"])

    horizon_reports = {}
    forecast_rows = []
    oos_63 = pd.DataFrame()
    latest_date = panels["close"].index[-1]
    latest_features = matrix.xs(latest_date, level="date", drop_level=True).dropna(subset=["mom_63"])

    for label, horizon in HORIZONS.items():
        target = forward_returns(panels, horizon)
        oos = purged_walk_forward(matrix, target, horizon)
        skill = information_coefficient(oos)
        shrink = _skill_shrinkage(skill["ic"])
        residual_q = (
            (oos["actual"] - oos["prediction"]).quantile([0.05, 0.95]).to_dict() if len(oos) > 60 else {0.05: np.nan, 0.95: np.nan}
        )
        market_base = float(market.tail(252).mean() * horizon)

        if len(oos) and len(latest_features):
            data = matrix.join(target, how="inner").dropna()
            x_all = data[FEATURES].to_numpy(dtype=float)
            y_all = data[target.name].to_numpy(dtype=float)
            models = _fit_ensemble(x_all, y_all)
            raw = _predict_ensemble(models, latest_features[FEATURES].fillna(0.0).to_numpy(dtype=float))
        else:
            raw = np.full(len(latest_features), market_base)

        for ticker, raw_forecast in zip(latest_features.index, raw):
            blended = market_base + shrink * (float(raw_forecast) - market_base)
            forecast_rows.append(
                {
                    "ticker": ticker,
                    "horizon": label,
                    "expected_return_pct": round(100 * (np.exp(blended) - 1), 2),
                    "raw_model_return_pct": round(100 * (np.exp(float(raw_forecast)) - 1), 2),
                    "low_5pct": round(100 * (np.exp(blended + residual_q.get(0.05, np.nan)) - 1), 2) if np.isfinite(residual_q.get(0.05, np.nan)) else None,
                    "high_95pct": round(100 * (np.exp(blended + residual_q.get(0.95, np.nan)) - 1), 2) if np.isfinite(residual_q.get(0.95, np.nan)) else None,
                    "skill_weight": shrink,
                }
            )
        horizon_reports[label] = {"oos_samples": int(len(oos)), "skill": skill, "forecast_shrinkage": shrink}
        if horizon == 63:
            oos_63 = oos

    forecasts = pd.DataFrame(forecast_rows)
    forecasts.to_csv(FORECASTS_CSV, index=False)

    # Top-10 at the user's quarterly horizon, with reasons from feature ranks.
    top10 = pd.DataFrame()
    quarter = forecasts[forecasts["horizon"] == "3M"].sort_values("expected_return_pct", ascending=False)
    if len(quarter):
        reasons = []
        ranks = latest_features[FEATURES].rank(pct=True)
        friendly = {
            "mom_63": "momentum 3 mois fort",
            "mom_126": "momentum 6 mois fort",
            "mom_21": "momentum 1 mois fort",
            "dist_52w_high": "proche de son plus-haut 52 semaines",
            "reversal_5": "rebond apres baisse recente",
            "vol_63": "volatilite maitrisee",
            "amihud": "prime d'illiquidite a capter",
            "volume_shock": "regain de volume anormal",
        }
        for ticker in quarter.head(10)["ticker"]:
            if ticker in ranks.index:
                row = ranks.loc[ticker]
                top_feats = row.sort_values(ascending=False).head(3)
                labels = [friendly.get(name, name) for name, value in top_feats.items() if value > 0.6]
                reasons.append("; ".join(labels) if labels else "profil equilibre")
            else:
                reasons.append("")
        top10 = quarter.head(10).copy()
        top10["pourquoi"] = reasons
        top10.to_csv(TOP10_CSV, index=False)

    regime = market_regime(panels)
    quarterly = quarterly_top10_backtest(matrix, panels, oos_63)

    best_skill = max((rep["skill"]["ic"] for rep in horizon_reports.values()), default=0.0)
    summary = {
        "as_of": str(latest_date.date()),
        "universe": int(panels["close"].shape[1]),
        "horizons": horizon_reports,
        "market_regime": regime,
        "quarterly_top10_backtest": quarterly,
        "top10_3m": top10.to_dict(orient="records") if len(top10) else [],
        "honesty_note": (
            "Chaque prevision est un melange: rendement moyen du marche + (signal modele x poids de "
            "competence). Le poids vient de l'IC hors echantillon: si les modeles n'ont pas prouve de "
            f"skill (meilleur IC actuel: {best_skill:.3f}), les previsions collent au marche et le disent."
        ),
    }
    ALPHA_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
