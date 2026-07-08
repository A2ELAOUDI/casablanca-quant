from __future__ import annotations

import argparse
import json

from .alpha import run_alpha
from .audit import run_audit
from .backtest import run_backtest
from .ingest import build_dataset, generate_synthetic_dataset, load_prices
from .portfolio import build_portfolio


def _print_json(data: dict) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="casablanca-quant")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Lire les Excel/CSV de data/raw/ et construire prices.csv")

    demo = sub.add_parser("demo", help="Generer un dataset synthetique type CSE pour valider le pipeline")
    demo.add_argument("--stocks", type=int, default=40)
    demo.add_argument("--days", type=int, default=500)
    demo.add_argument("--seed", type=int, default=7)

    sub.add_parser("audit", help="Audit d'efficience du marche (variance ratio, autocorrelation, liquidite)")

    backtest = sub.add_parser("backtest", help="Backtest walk-forward de la bibliotheque de facteurs")
    backtest.add_argument("--cost-bps", type=float, default=100.0)
    backtest.add_argument("--rebalance-days", type=int, default=5)
    backtest.add_argument("--top-quantile", type=float, default=0.20)

    sub.add_parser("alpha", help="Moteur alpha: ensemble ML purge, previsions 1M-12M, regime, top-10 trimestriel")

    portfolio = sub.add_parser("portfolio", help="Construire le portefeuille du jour")
    portfolio.add_argument("--capital", type=float, default=100_000.0)

    all_cmd = sub.add_parser("all", help="Pipeline complet: ingest -> audit -> backtest -> portfolio")
    all_cmd.add_argument("--cost-bps", type=float, default=100.0)
    all_cmd.add_argument("--capital", type=float, default=100_000.0)

    args = parser.parse_args(argv)

    if args.command == "ingest":
        prices = build_dataset()
        print(f"OK: {prices['ticker'].nunique()} valeurs, {len(prices)} lignes, {prices['date'].min().date()} -> {prices['date'].max().date()}")
        return 0

    if args.command == "demo":
        prices = generate_synthetic_dataset(n_stocks=args.stocks, n_days=args.days, seed=args.seed)
        print(f"Dataset synthetique: {prices['ticker'].nunique()} valeurs, {len(prices)} lignes.")
        return 0

    if args.command == "audit":
        _print_json(run_audit(load_prices()))
        return 0

    if args.command == "backtest":
        _print_json(
            run_backtest(
                load_prices(),
                cost_bps=args.cost_bps,
                rebalance_days=args.rebalance_days,
                top_quantile=args.top_quantile,
            )
        )
        return 0

    if args.command == "alpha":
        _print_json(run_alpha(load_prices()))
        return 0

    if args.command == "portfolio":
        portfolio = build_portfolio(load_prices(), capital_mad=args.capital)
        print(portfolio.to_string(index=False))
        return 0

    if args.command == "all":
        prices = build_dataset()
        print(f"Ingest OK: {prices['ticker'].nunique()} valeurs.")
        audit = run_audit(prices)
        print(f"Audit OK: {audit['variance_ratio']['stocks_rejecting_random_walk_5pct']} titres rejettent la marche aleatoire.")
        backtest_summary = run_backtest(prices, cost_bps=args.cost_bps)
        print(f"Backtest OK: meilleur facteur {backtest_summary['best_factor']} | verdict: {backtest_summary['production_verdict']}")
        alpha_summary = run_alpha(prices)
        print(f"Alpha OK: regime {alpha_summary['market_regime'].get('regime')} | univers {alpha_summary['universe']}.")
        portfolio = build_portfolio(prices, capital_mad=args.capital)
        print("\nPortefeuille du jour:")
        print(portfolio.to_string(index=False))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
