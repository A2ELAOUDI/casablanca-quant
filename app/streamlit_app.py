from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Streamlit keeps library modules cached between reruns; purge so edits in src/ load fresh.
for _name in [name for name in list(sys.modules) if name.startswith("casablanca_quant")]:
    del sys.modules[_name]

from casablanca_quant.alpha import run_alpha  # noqa: E402
from casablanca_quant.audit import run_audit  # noqa: E402
from casablanca_quant.backtest import run_backtest  # noqa: E402
from casablanca_quant.ingest import build_dataset, load_prices, to_panels  # noqa: E402
from casablanca_quant.paths import (  # noqa: E402
    ALPHA_JSON,
    AUDIT_JSON,
    BACKTEST_JSON,
    PORTFOLIO_JSON,
    PRICES_DATASET,
    PROCESSED_DIR,
)
from casablanca_quant.portfolio import build_portfolio  # noqa: E402


st.set_page_config(page_title="Casablanca Quant Lab", layout="wide")

st.markdown(
    """
    <style>
    :root { --bg:#f6f5f1; --panel:#ffffff; --ink:#1c1e21; --muted:#687076; --line:#d9d6cd; --accent:#0f7b63; --accent-2:#a13d2d; }
    .stApp { background: var(--bg); color: var(--ink); }
    .block-container { padding-top: 1.4rem; max-width: 1480px; }
    div[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }
    div[data-testid="stMetricLabel"] { color: var(--muted); font-size: 0.86rem; }
    div[data-testid="stMetricValue"] { color: var(--ink); font-size: 1.5rem; }
    .stButton > button { border-radius: 6px; border: 1px solid var(--line); font-weight: 600; }
    .stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); }
    section[data-testid="stSidebar"] { background: #eeeae0; border-right: 1px solid var(--line); }
    .verdict { background: var(--panel); border: 1px solid var(--line); border-left: 4px solid var(--accent); border-radius: 8px; padding: 12px 16px; margin: 8px 0; }
    .verdict.warn { border-left-color: var(--accent-2); }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


st.title("Casablanca Quant Lab")
st.caption("Bourse de Casablanca — audit d'efficience, facteurs quant, ensemble ML purge, previsions multi-horizons, portefeuille HRP.")

with st.sidebar:
    st.header("Pipeline")
    capital = st.number_input("Capital (MAD)", min_value=1_000.0, value=100_000.0, step=10_000.0)
    cost_bps = st.slider("Couts par cote (bps)", 20, 200, 100, step=10)
    if st.button("Tout analyser", type="primary", use_container_width=True):
        with st.spinner("Ingestion..."):
            try:
                build_dataset()
            except Exception as error:
                st.warning(f"Ingestion: {error}")
        prices = load_prices()
        with st.spinner("Audit d'efficience..."):
            run_audit(prices)
        with st.spinner("Backtest des facteurs (patience)..."):
            run_backtest(prices, cost_bps=float(cost_bps))
        with st.spinner("Moteur alpha: ensemble ML + previsions..."):
            run_alpha(prices)
        with st.spinner("Portefeuille HRP..."):
            build_portfolio(prices, capital_mad=float(capital))
        st.success("Analyse complete terminee.")
        st.rerun()
    if st.button("Ingestion seule", use_container_width=True):
        with st.spinner("Lecture de data/raw/ ..."):
            prices = build_dataset()
        st.success(f"{prices['ticker'].nunique()} valeurs, {len(prices)} lignes.")
        st.rerun()
    st.divider()
    st.caption("Depose tes fichiers Excel de la Bourse de Casablanca dans data/raw/ puis clique Tout analyser.")

if not PRICES_DATASET.exists():
    st.warning("Aucune donnee. Depose tes Excel dans data/raw/ puis clique 'Tout analyser' (ou lance `demo` en CLI).")
    st.stop()

prices = load_prices()
panels = to_panels(prices)
close = panels["close"]
audit = read_json(AUDIT_JSON)
backtest = read_json(BACKTEST_JSON)
alpha = read_json(ALPHA_JSON)
portfolio_report = read_json(PORTFOLIO_JSON)
top10 = read_csv(PROCESSED_DIR / "top10.csv")
forecasts = read_csv(PROCESSED_DIR / "forecasts.csv")
current_portfolio = read_csv(PROCESSED_DIR / "current_portfolio.csv")

tab_market, tab_top10, tab_forecast, tab_models, tab_audit, tab_portfolio = st.tabs(
    ["Marche", "Top 10", "Previsions", "Modeles", "Audit", "Portefeuille"]
)

with tab_market:
    tickers = sorted(close.columns.tolist())
    c1, c2 = st.columns([1, 3])
    ticker = c1.selectbox("Valeur", tickers)
    series = close[ticker].dropna()
    volume = panels["volume"][ticker].reindex(series.index)

    last = float(series.iloc[-1])
    perf_1m = 100 * (last / float(series.iloc[-22]) - 1) if len(series) > 22 else np.nan
    perf_3m = 100 * (last / float(series.iloc[-64]) - 1) if len(series) > 64 else np.nan
    perf_1y = 100 * (last / float(series.iloc[-253]) - 1) if len(series) > 253 else np.nan
    vol_ann = 100 * float(np.log(series / series.shift(1)).std() * np.sqrt(252))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Dernier cours", f"{last:,.2f} MAD".replace(",", " "))
    m2.metric("1 mois", f"{perf_1m:+.1f} %" if np.isfinite(perf_1m) else "-")
    m3.metric("3 mois", f"{perf_3m:+.1f} %" if np.isfinite(perf_3m) else "-")
    m4.metric("1 an", f"{perf_1y:+.1f} %" if np.isfinite(perf_1y) else "-")
    m5.metric("Volatilite ann.", f"{vol_ann:.0f} %")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name="Cours ajuste", line={"color": "#0f7b63", "width": 2}))
    fig.add_trace(go.Scatter(x=series.index, y=series.rolling(50).mean(), name="MM50", line={"color": "#a13d2d", "width": 1, "dash": "dot"}))
    fig.update_layout(title=f"{ticker} — cours ajuste", height=420, margin={"t": 40, "b": 10})
    st.plotly_chart(fig, use_container_width=True)
    if volume.notna().any():
        st.plotly_chart(px.bar(x=volume.index, y=volume.values, title="Volume (titres)", height=200), use_container_width=True)

with tab_top10:
    st.subheader("Top 10 du trimestre (horizon 3 mois)")
    if top10.empty:
        st.info("Clique 'Tout analyser' pour generer le top 10.")
    else:
        regime = alpha.get("market_regime", {})
        skill_3m = alpha.get("horizons", {}).get("3M", {}).get("skill", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Regime de marche", regime.get("regime", "-"))
        c2.metric("Skill OOS 3M (IC)", f"{skill_3m.get('ic', 0):.3f}")
        c3.metric("Confiance modele", f"{100 * alpha.get('horizons', {}).get('3M', {}).get('forecast_shrinkage', 0):.0f} %")
        display = top10.rename(
            columns={"ticker": "Valeur", "expected_return_pct": "Prevision 3M %", "low_5pct": "Pire cas 5%", "high_95pct": "Meilleur cas 95%", "pourquoi": "Pourquoi"}
        )[["Valeur", "Prevision 3M %", "Pire cas 5%", "Meilleur cas 95%", "Pourquoi"]]
        st.dataframe(display, use_container_width=True, hide_index=True)
        quarterly = alpha.get("quarterly_top10_backtest", {})
        if not quarterly.get("skipped"):
            st.markdown(
                f'<div class="verdict"><b>Backtest trimestriel du top 10:</b> exces moyen '
                f'{quarterly.get("mean_quarterly_excess_pct", 0):.2f} %/trimestre vs univers, t-stat {quarterly.get("t_stat", 0):.2f} '
                f'sur {quarterly.get("quarters", 0)} trimestres.</div>',
                unsafe_allow_html=True,
            )
        st.caption(alpha.get("honesty_note", ""))

with tab_forecast:
    st.subheader("Previsions 1 mois a 1 an")
    if forecasts.empty:
        st.info("Clique 'Tout analyser' pour generer les previsions.")
    else:
        ticker_f = st.selectbox("Valeur ", sorted(forecasts["ticker"].unique()))
        sub = forecasts[forecasts["ticker"] == ticker_f].copy()
        order = {"1M": 0, "3M": 1, "6M": 2, "12M": 3}
        sub["order"] = sub["horizon"].map(order)
        sub = sub.sort_values("order")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=sub["horizon"], y=sub["expected_return_pct"], name="Prevision", marker_color="#0f7b63"))
        if sub["low_5pct"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=sub["horizon"], y=sub["high_95pct"], mode="markers", name="Meilleur cas (95%)", marker={"symbol": "triangle-up", "size": 11, "color": "#687076"}
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=sub["horizon"], y=sub["low_5pct"], mode="markers", name="Pire cas (5%)", marker={"symbol": "triangle-down", "size": 11, "color": "#a13d2d"}
                )
            )
        fig.update_layout(title=f"{ticker_f} — rendement attendu par horizon (%)", height=420, yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            sub[["horizon", "expected_return_pct", "raw_model_return_pct", "low_5pct", "high_95pct", "skill_weight"]],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "skill_weight = confiance accordee au modele selon son IC hors echantillon. "
            "0 = la prevision colle au rendement moyen du marche (aucun skill prouve a cet horizon)."
        )

with tab_models:
    st.subheader("Modeles quant")
    if backtest:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Meilleur facteur", backtest.get("best_factor", "-"))
        reality = backtest.get("reality_check", {})
        c2.metric("Reality Check p", f"{reality.get('p_value', 1):.3f}" if "p_value" in reality else "-")
        c3.metric("Sharpe deflate", f"{backtest.get('deflated_sharpe_best', {}).get('dsr', 0):.2f}")
        gates = backtest.get("deployment_gates", {})
        c4.metric("Portes passees", f"{sum(gates.values())}/{len(gates)}" if gates else "-")
        verdict = backtest.get("production_verdict", "")
        css = "verdict" if verdict.startswith("DEPLOYER") else "verdict warn"
        st.markdown(f'<div class="{css}"><b>Verdict de production:</b> {verdict}</div>', unsafe_allow_html=True)

        rows = []
        for name, stats in backtest.get("factors", {}).items():
            rows.append(
                {
                    "facteur": name,
                    "rendement_ann_%": round(100 * stats.get("ann_return", 0), 1),
                    "sharpe": round(stats.get("sharpe", 0), 2),
                    "max_drawdown_%": round(100 * stats.get("max_drawdown", 0), 1),
                    "actif_vs_baseline_%": round(100 * stats.get("active_vs_baseline", {}).get("mean_ann", 0), 1),
                    "p(actif<=0)": round(stats.get("active_vs_baseline", {}).get("p_le_0", 1), 2),
                }
            )
        table = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(table, x="facteur", y="sharpe", title="Sharpe net de couts par facteur"), use_container_width=True)
    else:
        st.info("Clique 'Tout analyser'.")
    if alpha:
        st.subheader("Moteur alpha (ensemble ML purge)")
        rows = []
        for label, report in alpha.get("horizons", {}).items():
            skill = report.get("skill", {})
            rows.append(
                {
                    "horizon": label,
                    "echantillons_OOS": report.get("oos_samples", 0),
                    "IC": round(skill.get("ic", 0), 3),
                    "t_stat": round(skill.get("t_stat", 0), 2),
                    "confiance_%": round(100 * report.get("forecast_shrinkage", 0)),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_audit:
    st.subheader("Le marche marocain est-il inefficient ?")
    if audit:
        vr = audit.get("variance_ratio", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Valeurs", audit.get("tickers", 0))
        c2.metric("VR median (q=5)", f"{vr.get('median_vr_q5', 0):.3f}" if vr.get("median_vr_q5") else "-")
        c3.metric("Rejets marche aleatoire", vr.get("stocks_rejecting_random_walk_5pct", 0))
        c4.metric("Autocorrelation (LB 5%)", audit.get("autocorrelation_ljung_box_significant", 0))
        momentum = audit.get("momentum_spread_check", {})
        if momentum.get("periods"):
            st.markdown(
                f'<div class="verdict"><b>Momentum spread:</b> {momentum.get("mean_monthly_spread_pct", 0):.2f} %/mois '
                f'(t-stat {momentum.get("t_stat", 0):.2f}, {momentum.get("periods")} periodes). '
                "VR>1 et t-stat>2 = matiere premiere pour les facteurs.</div>",
                unsafe_allow_html=True,
            )
        stock_audit = read_csv(PROCESSED_DIR / "stock_audit.csv")
        if not stock_audit.empty:
            st.dataframe(stock_audit.sort_values("variance_ratio_q5", ascending=False), use_container_width=True, hide_index=True)
        st.caption(audit.get("interpretation", ""))
    else:
        st.info("Clique 'Tout analyser'.")

with tab_portfolio:
    st.subheader("Portefeuille du jour (HRP + Ledoit-Wolf)")
    if current_portfolio.empty:
        st.info("Clique 'Tout analyser'.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Mode", portfolio_report.get("mode", "-"))
        c2.metric("Positions", portfolio_report.get("positions", len(current_portfolio)))
        c3.metric("Investi", f"{portfolio_report.get('invested_pct', 0):.1f} %")
        st.dataframe(current_portfolio, use_container_width=True, hide_index=True)
        st.plotly_chart(
            px.pie(current_portfolio, names="ticker", values="weight_pct", title="Allocation", hole=0.45),
            use_container_width=True,
        )
        st.download_button(
            "Telecharger le portefeuille (CSV)",
            (PROCESSED_DIR / "current_portfolio.csv").read_bytes(),
            file_name="portefeuille_cse.csv",
            mime="text/csv",
        )
        st.caption(portfolio_report.get("note", ""))
