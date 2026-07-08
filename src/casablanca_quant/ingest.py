"""Flexible ingestion for Bourse de Casablanca price data.

Drop any Excel/CSV files into data/raw/ and run `ingest`. The loader handles:
- long format (one row per date per stock) and wide format (one column per stock);
- French/English column names (date, valeur/ticker/instrument, cours/close, volume...);
- multiple files and multiple Excel sheets, merged and deduplicated.

Output: data/processed/prices.csv — a clean long panel:
date, ticker, close, volume (optional open/high/low kept when present).

A synthetic generator calibrated to CSE stylized facts (thin trading, price
limits, momentum) validates the whole pipeline before real data arrives.
"""
from __future__ import annotations

import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import INGEST_JSON, PRICES_DATASET, PROCESSED_DIR, RAW_DIR, ensure_dirs


DATE_CANDIDATES = ["date", "seance", "session", "jour", "date_seance", "trade_date", "date de cotation"]
TICKER_CANDIDATES = ["ticker", "valeur", "instrument", "societe", "société", "libelle", "libellé", "stock", "code", "symbole", "symbol", "isin", "nom"]
CLOSE_CANDIDATES = ["cours ajuste", "adjusted", "cours de cloture", "cours_cloture", "cloture", "clôture", "close", "dernier cours", "dernier", "cours", "prix", "px_last"]
OPEN_CANDIDATES = ["open", "ouverture", "cours_ouverture", "premier"]
HIGH_CANDIDATES = ["high", "plus_haut", "plus haut", "haut", "max"]
LOW_CANDIDATES = ["low", "plus_bas", "plus bas", "bas", "min"]
VOLUME_CANDIDATES = ["nombre de titres", "titres echanges", "titres_echanges", "quantite", "quantité", "volume_titres", "qte", "titres", "volume", "volume echange"]


def _normalize_label(label: object) -> str:
    text = str(label).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.replace("_", " ").split())


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {_normalize_label(col): col for col in columns}
    for candidate in candidates:
        key = _normalize_label(candidate)
        if key in normalized:
            return normalized[key]
    for candidate in candidates:
        key = _normalize_label(candidate)
        for norm, original in normalized.items():
            if key and key in norm:
                return original
    return None


def _parse_number(value: object) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().replace(" ", "").replace(" ", "")
    if not text or text in {"-", "--", "nd", "n/d", "na"}:
        return None
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _read_any(path: Path) -> list[pd.DataFrame]:
    if path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        sheets = pd.read_excel(path, sheet_name=None, dtype=object)
        frames = []
        for name, frame in sheets.items():
            frame = frame.dropna(how="all").dropna(axis=1, how="all")
            if not frame.empty:
                frame.attrs["source"] = f"{path.name}::{name}"
                frames.append(frame)
        return frames
    for sep in (";", ",", "\t"):
        try:
            frame = pd.read_csv(path, sep=sep, dtype=object, engine="python")
            if frame.shape[1] >= 2:
                frame.attrs["source"] = path.name
                return [frame.dropna(how="all")]
        except Exception:
            continue
    return []


def _long_from_frame(frame: pd.DataFrame) -> pd.DataFrame | None:
    """Try to interpret a raw frame as a long price panel."""
    columns = [str(col) for col in frame.columns]
    date_col = _find_column(columns, DATE_CANDIDATES)
    close_col = _find_column(columns, CLOSE_CANDIDATES)
    ticker_col = _find_column(columns, TICKER_CANDIDATES)
    if date_col is None or close_col is None:
        return None

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(frame[date_col], dayfirst=True, errors="coerce")
    out["close"] = frame[close_col].map(_parse_number)
    if ticker_col is not None:
        out["ticker"] = frame[ticker_col].astype(str).str.strip().str.upper()
    else:
        source = frame.attrs.get("source", "UNKNOWN")
        out["ticker"] = str(source).split("::")[0].rsplit(".", 1)[0].upper()

    for target, candidates in (("open", OPEN_CANDIDATES), ("high", HIGH_CANDIDATES), ("low", LOW_CANDIDATES), ("volume", VOLUME_CANDIDATES)):
        col = _find_column(columns, candidates)
        if col is not None and col not in {date_col, close_col, ticker_col}:
            out[target] = frame[col].map(_parse_number)

    out = out.dropna(subset=["date", "close"])
    out = out[out["close"] > 0]
    return out if len(out) else None


def _long_from_wide(frame: pd.DataFrame) -> pd.DataFrame | None:
    """Interpret a wide frame: first column = date, one column per ticker."""
    columns = [str(col) for col in frame.columns]
    date_col = _find_column(columns, DATE_CANDIDATES) or columns[0]
    dates = pd.to_datetime(frame[date_col], dayfirst=True, errors="coerce")
    if dates.notna().mean() < 0.7:
        return None
    value_cols = [col for col in frame.columns if str(col) != str(date_col)]
    if len(value_cols) < 2:
        return None
    rows = []
    for col in value_cols:
        closes = frame[col].map(_parse_number)
        sub = pd.DataFrame({"date": dates, "ticker": _normalize_label(col).upper(), "close": closes})
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True).dropna(subset=["date", "close"])
    out = out[out["close"] > 0]
    return out if len(out) else None


def build_dataset(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    ensure_dirs()
    files = sorted([p for p in raw_dir.glob("*") if p.suffix.lower() in {".xlsx", ".xls", ".xlsm", ".csv", ".txt"}])
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier dans {raw_dir}. Depose tes Excel/CSV de la Bourse de Casablanca dans data/raw/ "
            f"ou lance `demo` pour generer des donnees synthetiques."
        )
    panels = []
    sources = []
    for path in files:
        for frame in _read_any(path):
            panel = _long_from_frame(frame)
            mode = "long"
            if panel is None:
                panel = _long_from_wide(frame)
                mode = "wide"
            if panel is None:
                sources.append({"source": frame.attrs.get("source", path.name), "status": "SKIPPED: colonnes non reconnues"})
                continue
            panels.append(panel)
            sources.append({"source": frame.attrs.get("source", path.name), "status": f"OK ({mode})", "rows": int(len(panel))})
    if not panels:
        raise RuntimeError("Aucun panneau de prix reconnu dans data/raw/. Envoie-moi un extrait du fichier.")

    prices = pd.concat(panels, ignore_index=True)
    prices = prices.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"], keep="last")
    prices.to_csv(PRICES_DATASET, index=False)

    metadata = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": sources,
        "tickers": int(prices["ticker"].nunique()),
        "rows": int(len(prices)),
        "first_date": str(prices["date"].min().date()),
        "last_date": str(prices["date"].max().date()),
    }
    INGEST_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return prices


def generate_synthetic_dataset(n_stocks: int = 40, n_days: int = 500, seed: int = 7) -> pd.DataFrame:
    """Synthetic CSE-like panel to validate the pipeline end-to-end.

    Injects known structure so the factory has something real to find:
    momentum autocorrelation, a low-vol premium, thin trading (stale prices),
    and daily price limits — the stylized facts of Casablanca.
    """
    ensure_dirs()
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-07-01", periods=n_days)
    tickers = [f"CSE{idx:02d}" for idx in range(1, n_stocks + 1)]

    market = rng.normal(0.0003, 0.007, n_days)
    rows = []
    for ticker_idx, ticker in enumerate(tickers):
        beta = rng.uniform(0.4, 1.4)
        vol = rng.uniform(0.008, 0.028)
        drift = 0.0004 - 0.008 * vol  # low-vol premium baked in
        momentum_strength = 0.06
        stale_prob = rng.uniform(0.05, 0.45)  # thin trading
        idio = rng.normal(0, vol, n_days)
        returns = np.zeros(n_days)
        for t in range(n_days):
            trend = momentum_strength * np.mean(returns[max(0, t - 60) : t]) if t > 10 else 0.0
            returns[t] = drift + beta * market[t] + idio[t] + trend
            returns[t] = float(np.clip(returns[t], -0.06, 0.06))  # CSE price limits
        price = 100.0 * np.exp(np.cumsum(returns))
        stale = rng.random(n_days) < stale_prob
        for t in range(1, n_days):
            if stale[t]:
                price[t] = price[t - 1]
        base_volume = rng.uniform(2e3, 8e4)
        volumes = np.where(stale, 0.0, base_volume * np.exp(rng.normal(0, 0.8, n_days)))
        for t, date in enumerate(dates):
            rows.append({"date": date, "ticker": ticker, "close": round(float(price[t]), 2), "volume": round(float(volumes[t]), 0)})

    prices = pd.DataFrame(rows)
    prices.to_csv(PRICES_DATASET, index=False)
    INGEST_JSON.write_text(
        json.dumps({"built_at_utc": datetime.now(timezone.utc).isoformat(), "mode": "SYNTHETIC DEMO", "tickers": n_stocks, "rows": len(prices)}, indent=2),
        encoding="utf-8",
    )
    return prices


def load_prices(path: Path = PRICES_DATASET) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("Pas de dataset. Lance `ingest` (donnees reelles) ou `demo` (synthetique).")
    prices = pd.read_csv(path)
    prices["date"] = pd.to_datetime(prices["date"])
    return prices


def to_panels(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Long frame -> wide panels (dates x tickers): close, returns, volume, traded."""
    close = prices.pivot_table(index="date", columns="ticker", values="close").sort_index()
    close = close.ffill(limit=10)
    returns = np.log(close / close.shift(1))
    volume = (
        prices.pivot_table(index="date", columns="ticker", values="volume").sort_index().reindex(close.index)
        if "volume" in prices.columns
        else pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    )
    traded = returns.abs() > 1e-12
    return {"close": close, "returns": returns, "volume": volume, "traded": traded}
