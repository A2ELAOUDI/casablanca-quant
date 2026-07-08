"""Factor library for the Casablanca Stock Exchange.

Every signal is a function panels -> score DataFrame (dates x tickers) where
the score at date t uses ONLY information available at the close of t. The
backtest engine applies an extra 1-day lag before trading, so there is zero
lookahead by construction.

The families implemented are the ones with the strongest evidence in frontier
markets (thin arbitrage, retail-dominated flow):
- cross-sectional momentum (12-1 style, adapted to a 2-year history);
- short-term reversal (retail overreaction);
- low-volatility premium;
- 52-week-high anchoring;
- Amihud illiquidity premium;
- abnormal volume (information arrival);
- time-series trend (moving-average crossover);
- a z-scored composite of the above.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def _zscore_cross_section(scores: pd.DataFrame) -> pd.DataFrame:
    mean = scores.mean(axis=1)
    std = scores.std(axis=1).replace(0, np.nan)
    return scores.sub(mean, axis=0).div(std, axis=0)


def momentum(panels: dict, lookback: int = 126, skip: int = 21) -> pd.DataFrame:
    """Classic cross-sectional momentum: past ~6 months, skipping the last month."""
    close = panels["close"]
    return np.log(close.shift(skip) / close.shift(lookback))


def short_term_reversal(panels: dict, window: int = 5) -> pd.DataFrame:
    """Retail overreaction: recent losers bounce, recent winners fade (score = -recent return)."""
    returns = panels["returns"]
    return -returns.rolling(window).sum()


def low_volatility(panels: dict, window: int = 63) -> pd.DataFrame:
    """Low-vol anomaly: score = -realized volatility (calmer stocks earn more per unit risk)."""
    returns = panels["returns"]
    return -returns.rolling(window).std()


def high_52w(panels: dict, window: int = 252) -> pd.DataFrame:
    """Proximity to the 52-week high: anchoring makes near-high stocks drift further."""
    close = panels["close"]
    rolling_max = close.rolling(window, min_periods=60).max()
    return close / rolling_max


def amihud_illiquidity(panels: dict, window: int = 63) -> pd.DataFrame:
    """Amihud (2002) illiquidity premium: |return| per unit of dirham volume.

    Illiquid stocks must pay a return premium. Score = +illiquidity (harvest
    the premium), but the backtest liquidity filter keeps positions tradable.
    """
    returns = panels["returns"].abs()
    dirham_volume = (panels["volume"] * panels["close"]).replace(0, np.nan)
    daily = returns / dirham_volume
    return np.log1p(daily.rolling(window, min_periods=20).mean() * 1e6)


def abnormal_volume(panels: dict, short: int = 5, long: int = 63) -> pd.DataFrame:
    """Volume shock: unusual activity precedes drift in thin markets."""
    volume = panels["volume"].replace(0, np.nan)
    ratio = volume.rolling(short).mean() / volume.rolling(long, min_periods=20).mean()
    return np.log(ratio)


def trend_ma(panels: dict, fast: int = 20, slow: int = 100) -> pd.DataFrame:
    """Time-series trend: fast MA above slow MA, scaled by distance."""
    close = panels["close"]
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow, min_periods=60).mean()
    return (fast_ma / slow_ma) - 1.0


def composite(panels: dict) -> pd.DataFrame:
    """Equal-risk composite: mean of cross-sectional z-scores of the core factors."""
    parts = [
        _zscore_cross_section(momentum(panels)),
        _zscore_cross_section(short_term_reversal(panels)),
        _zscore_cross_section(low_volatility(panels)),
        _zscore_cross_section(high_52w(panels)),
        _zscore_cross_section(trend_ma(panels)),
    ]
    stacked = pd.concat(parts).groupby(level=0).mean()
    return stacked


FACTOR_LIBRARY = {
    "momentum_126_21": partial(momentum, lookback=126, skip=21),
    "momentum_63_10": partial(momentum, lookback=63, skip=10),
    "reversal_5d": partial(short_term_reversal, window=5),
    "reversal_21d": partial(short_term_reversal, window=21),
    "low_volatility": low_volatility,
    "high_52w": high_52w,
    "amihud_illiquidity": amihud_illiquidity,
    "abnormal_volume": abnormal_volume,
    "trend_ma_20_100": trend_ma,
    "composite": composite,
}
