"""Risk and performance metrics.

Conventions used throughout this module:

* `returns` are daily SIMPLE returns, as produced by returns.simple_returns.
* Anything "annualized" uses TRADING_DAYS_PER_YEAR (252) from config.
* VaR and CVaR are reported as POSITIVE loss magnitudes, so 0.021 means
  "a loss of 2.1%". Drawdowns stay negative, because a drawdown is more
  natural to read as "down 34%".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .returns import wealth_index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def daily_risk_free_rate(
    annual_rate: float = config.RISK_FREE_RATE,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> float:
    """Convert an annual risk-free rate into a per-day rate.

    Compounded rather than simply divided by 252, so that holding the daily
    rate for a year gets back exactly the annual rate.
    """
    return (1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0


# ---------------------------------------------------------------------------
# Return and volatility
# ---------------------------------------------------------------------------
def annualized_return(
    returns: pd.Series, periods_per_year: int = config.TRADING_DAYS_PER_YEAR
) -> float:
    """Compound annual growth rate implied by the sample."""
    growth = float((1.0 + returns).prod())
    years = len(returns) / periods_per_year
    if years <= 0:
        return float("nan")
    return growth ** (1.0 / years) - 1.0


def annualized_volatility(
    returns: pd.Series, periods_per_year: int = config.TRADING_DAYS_PER_YEAR
) -> float:
    """Standard deviation of daily returns scaled up by sqrt(252).

    Uses the sample standard deviation (ddof=1), which is what pandas does
    by default.
    """
    return float(returns.std()) * np.sqrt(periods_per_year)


def rolling_volatility(
    returns: pd.DataFrame | pd.Series,
    window: int = config.ROLLING_WINDOW,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame | pd.Series:
    """Annualized volatility measured over a moving window of days."""
    return returns.rolling(window).std() * np.sqrt(periods_per_year)


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------
def drawdown_series(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Percentage below the running peak, day by day.

    Zero means the series is at a new high; -0.2 means it is 20% below its
    best level so far.
    """
    wealth = wealth_index(returns)
    peak = wealth.cummax()
    return wealth / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough fall in the sample, as a negative number."""
    return float(drawdown_series(returns).min())


def max_drawdown_date(returns: pd.Series) -> pd.Timestamp:
    """The day the worst drawdown bottomed out."""
    return drawdown_series(returns).idxmin()


# ---------------------------------------------------------------------------
# Risk-adjusted ratios
# ---------------------------------------------------------------------------
def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = config.RISK_FREE_RATE,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized excess return per unit of total volatility.

    Sharpe = mean(excess daily return) / sd(excess daily return) * sqrt(252)
    """
    excess = returns - daily_risk_free_rate(risk_free_rate, periods_per_year)
    sd = float(excess.std())
    if sd == 0:
        return float("nan")
    return float(excess.mean()) / sd * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = config.RISK_FREE_RATE,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> float:
    """Like Sharpe, but only counts downside moves as risk.

    The denominator is the downside deviation: the root mean square of the
    excess returns that fell below zero, with days above zero entered as 0
    rather than dropped. Upside volatility is therefore not penalised.
    """
    excess = returns - daily_risk_free_rate(risk_free_rate, periods_per_year)
    downside = excess.clip(upper=0.0)
    downside_dev = float(np.sqrt((downside**2).mean()))
    if downside_dev == 0:
        return float("nan")
    return float(excess.mean()) / downside_dev * np.sqrt(periods_per_year)


# ---------------------------------------------------------------------------
# Tail risk
# ---------------------------------------------------------------------------
def historical_var(
    returns: pd.Series, confidence: float = config.VAR_CONFIDENCE
) -> float:
    """Historical Value at Risk for one day, as a positive loss.

    At 95% confidence this is the 5th percentile of the observed daily
    returns, sign-flipped: on the worst 5% of days the loss was at least
    this large. No distribution is assumed; the empirical sample is used
    as-is.
    """
    cutoff = float(returns.quantile(1.0 - confidence))
    return -cutoff


def historical_cvar(
    returns: pd.Series, confidence: float = config.VAR_CONFIDENCE
) -> float:
    """Conditional VaR: the average loss on the days that breach the VaR.

    Also called expected shortfall. Returned as a positive loss, and always
    at least as large as the VaR.
    """
    cutoff = float(returns.quantile(1.0 - confidence))
    tail = returns[returns <= cutoff]
    if tail.empty:
        return float("nan")
    return -float(tail.mean())


# ---------------------------------------------------------------------------
# Market-relative metrics
# ---------------------------------------------------------------------------
def beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Sensitivity to the benchmark: cov(asset, market) / var(market).

    A beta of 1.2 means the stock has historically moved about 1.2% for
    every 1% move in the index.
    """
    asset, bench = _align(asset_returns, benchmark_returns)
    market_var = float(bench.var())
    if market_var == 0:
        return float("nan")
    return float(asset.cov(bench)) / market_var


def alpha(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = config.RISK_FREE_RATE,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized return left over after paying for market exposure.

    From the CAPM relation, using average daily excess returns:
        alpha_daily = mean(asset - rf) - beta * mean(market - rf)
    then scaled to a year by multiplying by 252.
    """
    asset, bench = _align(asset_returns, benchmark_returns)
    rf_daily = daily_risk_free_rate(risk_free_rate, periods_per_year)
    b = beta(asset, bench)
    daily_alpha = float((asset - rf_daily).mean()) - b * float((bench - rf_daily).mean())
    return daily_alpha * periods_per_year


def correlation(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Pearson correlation of daily returns with the benchmark."""
    asset, bench = _align(asset_returns, benchmark_returns)
    return float(asset.corr(bench))


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation of daily returns between every pair of assets."""
    return returns.corr()


def _align(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Keep only the dates both series share, so pairs line up."""
    joined = pd.concat([left, right], axis=1, join="inner").dropna()
    return joined.iloc[:, 0], joined.iloc[:, 1]


# ---------------------------------------------------------------------------
# Everything at once
# ---------------------------------------------------------------------------
def summarize(
    returns: pd.DataFrame,
    benchmark: str = config.BENCHMARK,
    risk_free_rate: float = config.RISK_FREE_RATE,
    confidence: float = config.VAR_CONFIDENCE,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Build one table of metrics, one row per asset.

    The benchmark gets a row too; its beta is 1 and its alpha 0 by
    construction, which is a handy sanity check on the other rows.
    """
    bench_returns = returns[benchmark]
    rows = {}
    for name in returns.columns:
        r = returns[name].dropna()
        rows[name] = {
            "total_return": float((1.0 + r).prod() - 1.0),
            "annualized_return": annualized_return(r, periods_per_year),
            "annualized_volatility": annualized_volatility(r, periods_per_year),
            "sharpe": sharpe_ratio(r, risk_free_rate, periods_per_year),
            "sortino": sortino_ratio(r, risk_free_rate, periods_per_year),
            "max_drawdown": max_drawdown(r),
            f"var_{int(confidence * 100)}": historical_var(r, confidence),
            f"cvar_{int(confidence * 100)}": historical_cvar(r, confidence),
            "beta": beta(r, bench_returns),
            "alpha": alpha(r, bench_returns, risk_free_rate, periods_per_year),
            "correlation_to_benchmark": correlation(r, bench_returns),
            "observations": int(len(r)),
        }
    return pd.DataFrame(rows).T
