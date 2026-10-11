"""Tests for the risk metrics.

Each example is small enough to check by hand. Where a closed form exists
the expected value is written out as the formula rather than as a decimal,
so the test shows the arithmetic it is checking.
"""

import numpy as np
import pandas as pd
import pytest

from equity_risk import config, data, metrics, returns


def series(values):
    """Daily returns on consecutive business days, as a pandas Series."""
    index = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.Series(values, index=index, dtype="float64")


# ---------------------------------------------------------------------------
# Risk-free rate conversion
# ---------------------------------------------------------------------------
def test_daily_risk_free_rate_compounds_back_to_the_annual_rate():
    daily = metrics.daily_risk_free_rate(0.065, 252)
    assert (1.0 + daily) ** 252 == pytest.approx(1.065)


def test_zero_risk_free_rate_gives_zero_daily_rate():
    assert metrics.daily_risk_free_rate(0.0, 252) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------
def test_annualized_volatility():
    """Four returns of +/-1% around a mean of zero.

    Every deviation is 0.01, so the sample variance is
    (4 * 0.01^2) / (4 - 1) and the daily standard deviation is
    0.01 * sqrt(4/3). Annualizing multiplies by sqrt(252).
    """
    r = series([0.01, -0.01, 0.01, -0.01])
    expected = 0.01 * np.sqrt(4 / 3) * np.sqrt(252)
    assert metrics.annualized_volatility(r) == pytest.approx(expected)


def test_constant_returns_have_zero_volatility():
    assert metrics.annualized_volatility(series([0.005] * 10)) == pytest.approx(0.0)


def test_rolling_volatility_window_and_alignment():
    r = series([0.01, -0.01, 0.01, -0.01, 0.02])
    rolling = metrics.rolling_volatility(r, window=4)

    # the first three days cannot fill a 4-day window
    assert rolling.iloc[:3].isna().all()
    # the fourth day is the same window as the test above
    assert rolling.iloc[3] == pytest.approx(0.01 * np.sqrt(4 / 3) * np.sqrt(252))


def test_annualized_return_of_a_constant_daily_gain():
    """Exactly one year of +0.1% a day compounds to 1.001^252 - 1."""
    r = series([0.001] * 252)
    assert metrics.annualized_return(r) == pytest.approx(1.001**252 - 1.0)


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------
def test_drawdown_series():
    """+10%, -50%, +20% gives wealth 1.10, 0.55, 0.66 against a 1.10 peak."""
    r = series([0.10, -0.50, 0.20])
    dd = metrics.drawdown_series(r)
    assert dd.tolist() == pytest.approx([0.0, -0.50, -0.40])


def test_max_drawdown_is_the_worst_point():
    r = series([0.10, -0.50, 0.20])
    assert metrics.max_drawdown(r) == pytest.approx(-0.50)


def test_max_drawdown_date():
    r = series([0.10, -0.50, 0.20])
    assert metrics.max_drawdown_date(r) == r.index[1]


def test_rising_series_never_draws_down():
    r = series([0.01, 0.02, 0.03])
    assert metrics.max_drawdown(r) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Sharpe and Sortino
# ---------------------------------------------------------------------------
def test_sharpe_ratio_with_no_risk_free_rate():
    """Returns of 2%, 0%, 2%, 0%: mean 1%, daily sd 0.01 * sqrt(4/3).

    Sharpe = 0.01 / (0.01 * sqrt(4/3)) * sqrt(252) = sqrt(0.75 * 252)
           = sqrt(189).
    """
    r = series([0.02, 0.0, 0.02, 0.0])
    assert metrics.sharpe_ratio(r, risk_free_rate=0.0) == pytest.approx(np.sqrt(189.0))


def test_sharpe_ratio_falls_when_the_risk_free_rate_rises():
    r = series([0.02, 0.0, 0.02, 0.0])
    low = metrics.sharpe_ratio(r, risk_free_rate=0.0)
    high = metrics.sharpe_ratio(r, risk_free_rate=0.10)
    assert high < low


def test_sharpe_ratio_is_nan_without_any_variation():
    assert np.isnan(metrics.sharpe_ratio(series([0.001] * 5), risk_free_rate=0.0))


def test_sortino_ratio_with_no_risk_free_rate():
    """Returns of 2%, -1%, 2%, -1%: mean 0.5%.

    Downside deviation counts the two losses and enters the gains as zero:
    sqrt((0 + 0.01^2 + 0 + 0.01^2) / 4) = 0.01 / sqrt(2).
    Sortino = 0.005 / (0.01 / sqrt(2)) * sqrt(252) = sqrt(0.5 * 252)
            = sqrt(126).
    """
    r = series([0.02, -0.01, 0.02, -0.01])
    assert metrics.sortino_ratio(r, risk_free_rate=0.0) == pytest.approx(
        np.sqrt(126.0)
    )


def test_sortino_is_higher_than_sharpe_when_losses_are_the_smaller_moves():
    """Big gains and small losses are punished by Sharpe but not by Sortino."""
    r = series([0.05, -0.01, 0.05, -0.01])
    sharpe = metrics.sharpe_ratio(r, risk_free_rate=0.0)
    sortino = metrics.sortino_ratio(r, risk_free_rate=0.0)
    assert sortino > sharpe


def test_sortino_is_nan_when_nothing_ever_falls():
    r = series([0.01, 0.02, 0.03])
    assert np.isnan(metrics.sortino_ratio(r, risk_free_rate=0.0))


# ---------------------------------------------------------------------------
# VaR and CVaR
# ---------------------------------------------------------------------------
@pytest.fixture
def twenty_returns():
    """Twenty returns in even 1% steps from -10% up to +9%."""
    return series([round(-0.10 + 0.01 * i, 4) for i in range(20)])


def test_historical_var_95(twenty_returns):
    """The 5th percentile of 20 sorted points sits at position 0.05 * 19 = 0.95,
    i.e. 95% of the way from -10% to -9%, which is -9.05%. VaR is reported
    as the positive loss, 9.05%.
    """
    assert metrics.historical_var(twenty_returns, 0.95) == pytest.approx(0.0905)


def test_historical_cvar_95(twenty_returns):
    """Only -10% is at or below the -9.05% cutoff, so the average loss
    beyond VaR is 10%."""
    assert metrics.historical_cvar(twenty_returns, 0.95) == pytest.approx(0.10)


def test_cvar_is_never_smaller_than_var(twenty_returns):
    var = metrics.historical_var(twenty_returns, 0.95)
    cvar = metrics.historical_cvar(twenty_returns, 0.95)
    assert cvar >= var


def test_var_grows_with_confidence(twenty_returns):
    assert metrics.historical_var(twenty_returns, 0.99) >= metrics.historical_var(
        twenty_returns, 0.90
    )


# ---------------------------------------------------------------------------
# Beta, alpha and correlation
# ---------------------------------------------------------------------------
@pytest.fixture
def benchmark():
    return series([0.01, -0.01, 0.02, -0.02])


def test_beta_of_exactly_double_the_market(benchmark):
    """An asset that always moves twice the index has a beta of 2."""
    asset = benchmark * 2.0
    assert metrics.beta(asset, benchmark) == pytest.approx(2.0)


def test_beta_of_the_market_against_itself_is_one(benchmark):
    assert metrics.beta(benchmark, benchmark) == pytest.approx(1.0)


def test_beta_of_an_opposite_asset_is_negative(benchmark):
    assert metrics.beta(-benchmark, benchmark) == pytest.approx(-1.0)


def test_correlation_of_a_scaled_copy_is_one(benchmark):
    assert metrics.correlation(benchmark * 3.0, benchmark) == pytest.approx(1.0)


def test_correlation_of_an_opposite_asset_is_minus_one(benchmark):
    assert metrics.correlation(-benchmark, benchmark) == pytest.approx(-1.0)


def test_alpha_is_zero_for_a_pure_multiple_of_the_market(benchmark):
    """Doubling the index earns no excess return once beta is paid for."""
    asset = benchmark * 2.0
    assert metrics.alpha(asset, benchmark, risk_free_rate=0.0) == pytest.approx(0.0)


def test_alpha_picks_up_a_constant_daily_edge(benchmark):
    """Beating the market by a flat 0.1% a day is 0.1% * 252 of alpha.

    The constant does not move with the index, so beta stays at 2 and the
    whole edge lands in alpha.
    """
    asset = benchmark * 2.0 + 0.001
    assert metrics.beta(asset, benchmark) == pytest.approx(2.0)
    assert metrics.alpha(asset, benchmark, risk_free_rate=0.0) == pytest.approx(
        0.001 * 252
    )


def test_align_handles_mismatched_dates(benchmark):
    """Only the overlapping dates are used for a pairwise metric."""
    asset = (benchmark * 2.0).iloc[1:]
    assert metrics.beta(asset, benchmark) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# The summary table, run on the synthetic sample file
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_summary():
    prices = data.load_sample_prices()
    simple = returns.simple_returns(prices)
    return metrics.summarize(simple)


def test_summary_has_a_row_for_every_asset(sample_summary):
    assert set(sample_summary.index) == set(config.ALL_TICKERS)


def test_summary_has_no_missing_numbers(sample_summary):
    assert not sample_summary.isna().any().any()


def test_benchmark_row_is_self_consistent(sample_summary):
    """The index measured against itself must give beta 1, alpha 0, corr 1."""
    row = sample_summary.loc[config.BENCHMARK]
    assert row["beta"] == pytest.approx(1.0)
    assert row["alpha"] == pytest.approx(0.0, abs=1e-12)
    assert row["correlation_to_benchmark"] == pytest.approx(1.0)


def test_summary_values_are_in_sensible_ranges(sample_summary):
    for name in sample_summary.index:
        row = sample_summary.loc[name]
        assert row["annualized_volatility"] > 0
        assert -1.0 <= row["max_drawdown"] <= 0.0
        assert -1.0 <= row["correlation_to_benchmark"] <= 1.0
        assert row["cvar_95"] >= row["var_95"]


def test_correlation_matrix_shape_and_diagonal():
    prices = data.load_sample_prices()
    corr = metrics.correlation_matrix(returns.simple_returns(prices))
    assert corr.shape == (len(config.ALL_TICKERS), len(config.ALL_TICKERS))
    assert np.allclose(np.diag(corr.to_numpy()), 1.0)
    # correlation is symmetric
    assert np.allclose(corr.to_numpy(), corr.to_numpy().T)
