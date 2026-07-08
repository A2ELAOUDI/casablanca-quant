from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

PRICES_DATASET = PROCESSED_DIR / "prices.csv"
AUDIT_JSON = REPORTS_DIR / "market_audit.json"
BACKTEST_JSON = REPORTS_DIR / "factor_backtest.json"
PORTFOLIO_JSON = REPORTS_DIR / "portfolio_report.json"
INGEST_JSON = REPORTS_DIR / "ingest_metadata.json"
ALPHA_JSON = REPORTS_DIR / "alpha_report.json"

FORECASTS_CSV = PROCESSED_DIR / "forecasts.csv"
TOP10_CSV = PROCESSED_DIR / "top10.csv"
CURRENT_PORTFOLIO_CSV = PROCESSED_DIR / "current_portfolio.csv"
STOCK_AUDIT_CSV = PROCESSED_DIR / "stock_audit.csv"
STRATEGY_RETURNS_CSV = PROCESSED_DIR / "strategy_daily_returns.csv"


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
