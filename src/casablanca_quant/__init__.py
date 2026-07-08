"""Casablanca Quant Lab — recherche quantitative sur la Bourse de Casablanca."""

from .alpha import run_alpha
from .audit import run_audit
from .backtest import run_backtest
from .ingest import build_dataset, generate_synthetic_dataset, load_prices, to_panels
from .portfolio import build_portfolio, hrp_weights

__version__ = "0.1.0"
__all__ = [
    "build_dataset",
    "build_portfolio",
    "generate_synthetic_dataset",
    "hrp_weights",
    "load_prices",
    "run_alpha",
    "run_audit",
    "run_backtest",
    "to_panels",
]
