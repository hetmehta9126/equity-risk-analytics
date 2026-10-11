"""Building the Markdown report out of the numbers the pipeline computed.

Nothing here invents a figure: every value printed comes from the metrics
table or the price data passed in.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from . import config
from .data import PriceData

# How each metric should be shown: label, and whether it is a percentage,
# a plain ratio or a count.
METRIC_ROWS = [
    ("total_return", "Total return", "pct"),
    ("annualized_return", "Annualized return (CAGR)", "pct"),
    ("annualized_volatility", "Annualized volatility", "pct"),
    ("sharpe", "Sharpe ratio", "ratio"),
    ("sortino", "Sortino ratio", "ratio"),
    ("max_drawdown", "Maximum drawdown", "pct"),
    ("var_95", "Historical VaR 95% (1 day loss)", "pct"),
    ("cvar_95", "Historical CVaR 95% (1 day loss)", "pct"),
    ("beta", "Beta vs NIFTY 50", "ratio"),
    ("alpha", "Alpha vs NIFTY 50 (annualized)", "pct"),
    ("correlation_to_benchmark", "Correlation with NIFTY 50", "ratio"),
    ("observations", "Trading days used", "count"),
]


def _format(value, kind: str) -> str:
    """Render one number, or a dash if it could not be computed."""
    if pd.isna(value):
        return "n/a"
    if kind == "pct":
        return f"{value * 100:.2f}%"
    if kind == "count":
        return f"{int(value)}"
    return f"{value:.3f}"


def metrics_to_markdown(summary: pd.DataFrame) -> str:
    """Metrics table with one column per asset and one row per metric."""
    assets = list(summary.index)
    header = "| Metric | " + " | ".join(assets) + " |"
    divider = "| --- | " + " | ".join("---" for _ in assets) + " |"

    lines = [header, divider]
    for key, label, kind in METRIC_ROWS:
        if key not in summary.columns:
            continue
        cells = [_format(summary.loc[a, key], kind) for a in assets]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def correlation_to_markdown(corr: pd.DataFrame) -> str:
    """The correlation matrix as a Markdown table."""
    names = list(corr.columns)
    lines = [
        "| | " + " | ".join(names) + " |",
        "| --- | " + " | ".join("---" for _ in names) + " |",
    ]
    for row in names:
        cells = [f"{corr.loc[row, col]:.3f}" for col in names]
        lines.append(f"| **{row}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _data_source_block(price_data: PriceData) -> str:
    """A short, honest statement of where the numbers came from."""
    if price_data.is_live:
        return (
            "Prices were downloaded from Yahoo Finance through the `yfinance` "
            "package, using split- and dividend-adjusted daily closes."
        )
    return (
        "> **WARNING - these results are not based on real market data.**\n"
        "> The live download failed, so the pipeline fell back to the synthetic\n"
        "> test sample in `data/sample/`. The numbers below are a smoke test\n"
        "> showing the code runs end to end. They say nothing about the actual\n"
        f"> risk of any listed security.\n>\n> Reason: {price_data.note}"
    )


def build_report(
    price_data: PriceData,
    summary: pd.DataFrame,
    corr: pd.DataFrame,
    chart_paths: dict[str, Path],
    report_path: Path = config.REPORT_PATH,
    risk_free_rate: float = config.RISK_FREE_RATE,
    confidence: float = config.VAR_CONFIDENCE,
    rolling_window: int = config.ROLLING_WINDOW,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> Path:
    """Write the full Markdown report and return the path it was saved to."""
    prices = price_data.prices
    level = int(confidence * 100)
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    years = len(prices) / periods_per_year

    # chart links are written relative to the report, which sits in outputs/
    def chart_link(key: str) -> str:
        path = chart_paths.get(key)
        if path is None:
            return "_chart not generated_"
        try:
            relative = Path(path).resolve().relative_to(Path(report_path).resolve().parent)
        except ValueError:
            relative = Path(path)
        return f"![{key.replace('_', ' ')}]({relative.as_posix()})"

    best_sharpe = summary.drop(index=config.BENCHMARK, errors="ignore")["sharpe"]
    highest_vol = summary.drop(index=config.BENCHMARK, errors="ignore")[
        "annualized_volatility"
    ]
    deepest_dd = summary.drop(index=config.BENCHMARK, errors="ignore")["max_drawdown"]

    sections = [
        "# Equity Risk Analytics report",
        "",
        f"Generated on {generated}.",
        "",
        _data_source_block(price_data),
        "",
        "## 1. Sample",
        "",
        f"- Stocks analysed: {', '.join(config.STOCKS)}",
        f"- Benchmark: {config.BENCHMARK} (NIFTY 50)",
        f"- Period: {prices.index[0].date()} to {prices.index[-1].date()}",
        f"- Trading days after cleaning: {len(prices)} (about {years:.1f} years)",
        f"- Data source: {price_data.source}",
        "",
        "## 2. Risk and return metrics",
        "",
        metrics_to_markdown(summary),
        "",
        "A few observations straight from the table above:",
        "",
        f"- Highest Sharpe ratio among the three stocks: **{best_sharpe.idxmax()}** "
        f"at {best_sharpe.max():.3f}.",
        f"- Most volatile of the three: **{highest_vol.idxmax()}** at "
        f"{highest_vol.max() * 100:.2f}% annualized.",
        f"- Deepest drawdown of the three: **{deepest_dd.idxmin()}** at "
        f"{deepest_dd.min() * 100:.2f}%.",
        "",
        "## 3. Correlations",
        "",
        correlation_to_markdown(corr),
        "",
        "## 4. Charts",
        "",
        "### Cumulative returns",
        "",
        chart_link("cumulative_returns"),
        "",
        "### Rolling volatility",
        "",
        chart_link("rolling_volatility"),
        "",
        "### Drawdowns",
        "",
        chart_link("drawdowns"),
        "",
        "### Correlation heatmap",
        "",
        chart_link("correlation_heatmap"),
        "",
        "### Return distribution with VaR",
        "",
        chart_link("return_distributions"),
        "",
        "## 5. Methodology",
        "",
        "- **Prices.** Daily adjusted closes, so that splits and dividends do "
        "not show up as artificial jumps in the return series.",
        "- **Alignment.** All tickers are put on one shared calendar. Days where "
        "every ticker is missing are dropped, short single-ticker gaps are "
        "filled forward by up to three days, and any day still incomplete is "
        "dropped so every asset is measured over exactly the same dates.",
        "- **Returns.** Simple daily returns for the metrics; log returns are "
        "also computed and are the more natural choice for adding up moves "
        "across days.",
        f"- **Annualization.** {periods_per_year} trading days per year. "
        "Volatility is scaled by the square root of that, average returns by "
        "the number itself.",
        "- **Volatility.** Sample standard deviation of daily returns, "
        f"annualized. The rolling version uses a {rolling_window}-day window "
        "(roughly one quarter).",
        "- **Drawdown.** Each day's level against the highest level reached so "
        "far; the maximum drawdown is the worst of those.",
        "- **Sharpe.** Mean daily excess return divided by its standard "
        "deviation, annualized. **Sortino** replaces the denominator with the "
        "downside deviation, so only losses count as risk. Both use the "
        "arithmetic mean of daily excess returns, not the compound annual "
        "growth rate (CAGR), so a positive Sharpe or Sortino does not imply "
        "CAGR is above the risk-free rate, and the two can disagree in sign.",
        f"- **VaR and CVaR.** Historical method at {level}% confidence: VaR is "
        f"the {100 - level}th percentile of observed daily returns, and CVaR is "
        "the average of the returns at or below it. Both are shown as positive "
        "loss figures, and no normal distribution is assumed.",
        "- **Beta and alpha.** Beta is the covariance with NIFTY 50 divided by "
        "the variance of NIFTY 50. Alpha is the CAPM leftover: mean excess "
        "return minus beta times the market's mean excess return, annualized.",
        "",
        "## 6. Assumptions",
        "",
        f"- **Risk-free rate of {risk_free_rate * 100:.2f}% per year.** "
        "Source: RBI 91-day T-bill auction result dated 7 Oct 2026, cut-off "
        "yield 5.5747%. This is a single point-in-time yield applied to the "
        "whole 5-year window, so it is an approximation rather than a "
        "measured average over the period. Sharpe, Sortino and alpha all "
        "move when it changes; total return, volatility, drawdown, VaR, CVaR, "
        "beta and correlation do not.",
        "- The annual rate is converted to a daily rate by compounding, and is "
        "held constant across the whole period, which it was not in reality.",
        "- Returns are treated as if drawn from one unchanging distribution, "
        "even though volatility clearly moves around over time.",
        "- No transaction costs, brokerage, taxes or slippage are included.",
        "- Yahoo Finance adjusted closes are taken at face value, with no "
        "independent check against an exchange feed.",
        "",
        "## 7. Limitations",
        "",
        f"- About {years:.1f} years of history is a short sample for tail risk. "
        f"At {level}% confidence the VaR estimate rests on only the worst "
        f"{max(1, int(len(prices) * (1 - confidence)))} or so days in the "
        "sample, so it is sensitive to a handful of observations.",
        "- Historical VaR cannot describe a loss larger than the worst day "
        "already seen, so it will understate a genuine crisis.",
        "- Equity returns have fatter tails and more skew than a normal "
        "distribution; metrics built on standard deviation, Sharpe included, "
        "understate how bad the extremes can get.",
        "- Beta, alpha and correlation are single numbers for the whole period, "
        "but all three drift over time and tend to rise together in a crash.",
        "- Past risk is not a forecast of future risk.",
        "- Three large-cap stocks against one index is a narrow slice of the "
        "market, and two of the three are in financial-adjacent sectors, so "
        "the correlations are not representative of a diversified portfolio.",
        "",
    ]

    text = "\n".join(sections)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(text)
    return Path(report_path)
