"""Turning prices into returns.

Every function takes a price DataFrame (or Series) indexed by date and
returns the same shape, with the first row dropped where a return cannot
be defined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Daily simple (percentage) returns: r_t = P_t / P_{t-1} - 1."""
    return prices.pct_change().dropna(how="all")


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Daily log returns: r_t = ln(P_t / P_{t-1}).

    Log returns add up over time, which makes them convenient for
    aggregating across days.
    """
    return np.log(prices / prices.shift(1)).dropna(how="all")


def cumulative_returns(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Cumulative growth from simple returns, as a fraction of the start.

    The series starts near 0 on the first day and 0.25 means "up 25%
    since the beginning".
    """
    return (1.0 + returns).cumprod() - 1.0


def wealth_index(
    returns: pd.DataFrame | pd.Series, initial: float = 1.0
) -> pd.DataFrame | pd.Series:
    """Value of `initial` invested at the start and held throughout."""
    return initial * (1.0 + returns).cumprod()


def total_return(returns: pd.DataFrame | pd.Series) -> pd.Series | float:
    """Overall return across the whole sample."""
    return (1.0 + returns).prod() - 1.0
