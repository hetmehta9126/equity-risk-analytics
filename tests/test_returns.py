"""Tests for the return calculations.

The examples use prices that are easy to check with a calculator: 100 ->
110 -> 99 is +10% then -10%, which compounds to -1% overall.
"""

import numpy as np
import pandas as pd
import pytest

from equity_risk import data, returns


@pytest.fixture
def toy_prices():
    """Three days of prices for one stock: 100, 110, 99."""
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    return pd.DataFrame({"AAA": [100.0, 110.0, 99.0]}, index=index)


def test_simple_returns_values(toy_prices):
    result = returns.simple_returns(toy_prices)["AAA"]
    # 110/100 - 1 = +0.10, then 99/110 - 1 = -0.10
    assert result.tolist() == pytest.approx([0.10, -0.10])


def test_simple_returns_drops_only_the_first_day(toy_prices):
    result = returns.simple_returns(toy_prices)
    assert len(result) == len(toy_prices) - 1
    assert result.index[0] == toy_prices.index[1]


def test_log_returns_values(toy_prices):
    result = returns.log_returns(toy_prices)["AAA"]
    expected = [np.log(1.10), np.log(0.90)]
    assert result.tolist() == pytest.approx(expected)


def test_log_returns_match_simple_returns(toy_prices):
    """log(1 + simple) should equal the log return, day by day."""
    simple = returns.simple_returns(toy_prices)["AAA"]
    log = returns.log_returns(toy_prices)["AAA"]
    assert np.log(1.0 + simple).tolist() == pytest.approx(log.tolist())


def test_log_returns_add_up_over_the_sample(toy_prices):
    """Log returns sum to the log of total growth: 99/100 = 0.99."""
    log = returns.log_returns(toy_prices)["AAA"]
    assert log.sum() == pytest.approx(np.log(0.99))


def test_cumulative_returns_values(toy_prices):
    simple = returns.simple_returns(toy_prices)
    result = returns.cumulative_returns(simple)["AAA"]
    # up 10%, then 1.10 * 0.90 = 0.99, i.e. down 1% from the start
    assert result.tolist() == pytest.approx([0.10, -0.01])


def test_wealth_index_values(toy_prices):
    simple = returns.simple_returns(toy_prices)
    result = returns.wealth_index(simple, initial=1000.0)["AAA"]
    assert result.tolist() == pytest.approx([1100.0, 990.0])


def test_total_return(toy_prices):
    simple = returns.simple_returns(toy_prices)
    assert float(returns.total_return(simple)["AAA"]) == pytest.approx(-0.01)


def test_cumulative_return_matches_the_price_change(toy_prices):
    """The last cumulative return must equal the start-to-end price move."""
    simple = returns.simple_returns(toy_prices)
    cumulative = returns.cumulative_returns(simple)["AAA"]
    price_change = toy_prices["AAA"].iloc[-1] / toy_prices["AAA"].iloc[0] - 1.0
    assert float(cumulative.iloc[-1]) == pytest.approx(price_change)


def test_flat_prices_give_zero_returns():
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    flat = pd.DataFrame({"AAA": [50.0, 50.0, 50.0]}, index=index)
    assert returns.simple_returns(flat)["AAA"].tolist() == pytest.approx([0.0, 0.0])
    assert returns.log_returns(flat)["AAA"].tolist() == pytest.approx([0.0, 0.0])


# ---------------------------------------------------------------------------
# The same functions run against the synthetic sample file
# ---------------------------------------------------------------------------
def test_returns_on_the_sample_file():
    """Shapes and relationships hold on the test sample in data/sample/."""
    prices = data.load_sample_prices()
    simple = returns.simple_returns(prices)

    assert len(simple) == len(prices) - 1
    assert list(simple.columns) == list(prices.columns)
    assert not simple.isna().any().any()

    # consistency between the two return definitions on real-shaped data
    log = returns.log_returns(prices)
    assert np.allclose(np.log(1.0 + simple.to_numpy()), log.to_numpy())
