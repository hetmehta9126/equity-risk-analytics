"""Run the whole pipeline: load prices, compute metrics, draw charts, write
the report.

    python main.py

Settings such as the tickers, the lookback period and the risk-free rate
live in src/equity_risk/config.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# make the package importable when running this file directly from the repo
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd

from equity_risk import config, data, metrics, plots, report, returns


def main() -> int:
    pd.set_option("display.width", 120)

    print("=" * 70)
    print("EQUITY RISK ANALYTICS")
    print("=" * 70)
    print(f"Stocks     : {', '.join(config.STOCKS)}")
    print(f"Benchmark  : {config.BENCHMARK} (NIFTY 50)")
    print(f"Lookback   : about {config.LOOKBACK_YEARS} years of daily data")
    print(f"Risk-free  : {config.RISK_FREE_RATE * 100:.2f}% per year (ASSUMPTION)")
    print("-" * 70)

    # 1. Data ---------------------------------------------------------------
    print("\n[1/5] Loading prices")
    price_data = data.load_price_data()
    prices = price_data.prices

    # 2. Returns ------------------------------------------------------------
    print("\n[2/5] Computing returns")
    simple = returns.simple_returns(prices)
    log = returns.log_returns(prices)
    cumulative = returns.cumulative_returns(simple)
    print(f"  simple returns : {simple.shape[0]} days x {simple.shape[1]} assets")
    print(f"  log returns    : {log.shape[0]} days x {log.shape[1]} assets")

    # 3. Metrics ------------------------------------------------------------
    print("\n[3/5] Computing risk metrics")
    summary = metrics.summarize(simple)
    corr = metrics.correlation_matrix(simple)
    rolling_vol = metrics.rolling_volatility(simple).dropna(how="all")
    drawdowns = metrics.drawdown_series(simple)

    display = summary[
        [
            "annualized_return",
            "annualized_volatility",
            "sharpe",
            "sortino",
            "max_drawdown",
            "var_95",
            "beta",
            "alpha",
        ]
    ]
    print(display.round(4).to_string())

    # 4. Charts -------------------------------------------------------------
    print("\n[4/5] Drawing charts")
    chart_paths = plots.make_all_charts(
        cumulative=cumulative,
        rolling_vol=rolling_vol,
        drawdowns=drawdowns,
        corr=corr,
        daily_returns=simple,
    )
    for name, path in chart_paths.items():
        print(f"  {name:22s} -> {path}")

    # 5. Report -------------------------------------------------------------
    print("\n[5/5] Writing report")
    report_path = report.build_report(
        price_data=price_data,
        summary=summary,
        corr=corr,
        chart_paths=chart_paths,
    )
    print(f"  report -> {report_path}")

    print("\nDone.")
    if not price_data.is_live:
        print(
            "NOTE: the synthetic test sample was used, so these numbers are a "
            "code check and not real market risk."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
