import numpy as np
import pandas as pd
import pytest

from casablanca_quant.backtest import _net_strategy_returns, _weights_from_scores, deflated_sharpe, eligibility_mask
from casablanca_quant.audit import ljung_box, variance_ratio
from casablanca_quant.ingest import _long_from_frame, _long_from_wide, generate_synthetic_dataset, to_panels
from casablanca_quant.signals import FACTOR_LIBRARY


@pytest.fixture(scope="module")
def panels():
    prices = generate_synthetic_dataset(n_stocks=20, n_days=320, seed=3)
    return to_panels(prices)


def test_ingest_long_format_french_columns() -> None:
    frame = pd.DataFrame(
        {
            "Date de cotation": ["02/01/2025", "03/01/2025"],
            "Libellé": ["ATTIJARIWAFA", "ATTIJARIWAFA"],
            "Cours de clôture": ["512,00", "515,50"],
            "Nombre de titres": ["1 200", "900"],
        }
    )
    panel = _long_from_frame(frame)
    assert panel is not None and len(panel) == 2
    assert panel["close"].iloc[0] == 512.0
    assert panel["volume"].iloc[1] == 900.0


def test_ingest_wide_format() -> None:
    frame = pd.DataFrame(
        {
            "Date": ["02/01/2025", "03/01/2025", "06/01/2025"],
            "IAM": ["101,5", "102,0", "101,0"],
            "BCP": ["270", "272", "271"],
        }
    )
    panel = _long_from_wide(frame)
    assert panel is not None
    assert set(panel["ticker"].unique()) == {"IAM", "BCP"}
    assert len(panel) == 6


def test_all_factors_produce_scores_without_lookahead_shape(panels) -> None:
    for name, factor in FACTOR_LIBRARY.items():
        scores = factor(panels)
        assert scores.shape == panels["close"].shape, name
        assert not np.isinf(scores.to_numpy(dtype=float)).any(), name


def test_weights_long_only_and_sum_to_one(panels) -> None:
    scores = FACTOR_LIBRARY["momentum_126_21"](panels)
    eligible = eligibility_mask(panels)
    weights = _weights_from_scores(scores, eligible, rebalance_days=5, top_quantile=0.2)
    assert float(weights.min().min()) >= 0.0
    sums = weights.sum(axis=1)
    active_days = sums[sums > 0]
    assert np.allclose(active_days, 1.0, atol=1e-9)


def test_costs_reduce_returns(panels) -> None:
    scores = FACTOR_LIBRARY["reversal_5d"](panels)
    eligible = eligibility_mask(panels)
    weights = _weights_from_scores(scores, eligible, rebalance_days=5, top_quantile=0.2)
    gross = _net_strategy_returns(weights, panels["returns"], cost_bps=0.0).sum()
    net = _net_strategy_returns(weights, panels["returns"], cost_bps=100.0).sum()
    assert net < gross


def test_variance_ratio_near_one_on_white_noise() -> None:
    rng = np.random.default_rng(1)
    result = variance_ratio(rng.normal(0, 0.01, 2000))
    assert 0.9 < result["vr"] < 1.1
    assert result["p_value"] > 0.01


def test_ljung_box_detects_autocorrelation() -> None:
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.01, 1500)
    ar = np.zeros_like(noise)
    for t in range(1, len(noise)):
        ar[t] = 0.35 * ar[t - 1] + noise[t]
    assert ljung_box(ar)["p_value"] < 0.001
    assert ljung_box(noise)["p_value"] > 0.05


def test_deflated_sharpe_punishes_many_trials() -> None:
    rng = np.random.default_rng(4)
    lucky = pd.Series(rng.normal(0.0004, 0.01, 400))
    few = deflated_sharpe(lucky, n_trials=1)["dsr"]
    many = deflated_sharpe(lucky, n_trials=100)["dsr"]
    assert many < few
