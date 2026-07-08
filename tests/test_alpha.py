import numpy as np
import pandas as pd

from casablanca_quant.alpha import _skill_shrinkage, feature_matrix, forward_returns
from casablanca_quant.ingest import generate_synthetic_dataset, to_panels
from casablanca_quant.portfolio import hrp_weights


def test_forward_returns_are_strictly_future() -> None:
    prices = generate_synthetic_dataset(n_stocks=6, n_days=120, seed=11)
    panels = to_panels(prices)
    close = panels["close"]
    target = forward_returns(panels, 21)
    date = close.index[50]
    ticker = close.columns[0]
    expected = np.log(close.iloc[71][ticker] / close.iloc[50][ticker])
    assert abs(target.loc[(date, ticker)] - expected) < 1e-12


def test_features_have_no_future_information() -> None:
    prices = generate_synthetic_dataset(n_stocks=6, n_days=200, seed=12)
    panels = to_panels(prices)
    full = feature_matrix(panels)
    truncated_panels = to_panels(prices[prices["date"] <= prices["date"].sort_values().unique()[150]])
    truncated = feature_matrix(truncated_panels)
    date = truncated_panels["close"].index[120]
    for ticker in truncated_panels["close"].columns[:3]:
        a = full.loc[(date, ticker)].dropna()
        b = truncated.loc[(date, ticker)][a.index]
        assert np.allclose(a.values.astype(float), b.values.astype(float), atol=1e-10)


def test_skill_shrinkage_refuses_negative_ic() -> None:
    assert _skill_shrinkage(-0.2) == 0.0
    assert _skill_shrinkage(0.0) == 0.0
    assert 0.0 < _skill_shrinkage(0.05) <= 0.5


def test_hrp_weights_valid() -> None:
    rng = np.random.default_rng(5)
    returns = pd.DataFrame(rng.normal(0, 0.01, (200, 8)), columns=[f"S{i}" for i in range(8)])
    weights = hrp_weights(returns)
    assert abs(float(weights.sum()) - 1.0) < 1e-9
    assert float(weights.min()) > 0
