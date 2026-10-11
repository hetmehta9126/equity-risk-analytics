"""Configuration section for the whole project.

Everything that a user might want to change lives here so that the
analysis modules stay free of hard-coded numbers.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------
# The three stocks I analyse, and the index I use as the market benchmark.
STOCKS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
BENCHMARK = "^NSEI"  # NIFTY 50

ALL_TICKERS = STOCKS + [BENCHMARK]

# ---------------------------------------------------------------------------
# Sample period
# ---------------------------------------------------------------------------
# Roughly five years of daily data, measured back from the day the script runs.
LOOKBACK_YEARS = 5

# ---------------------------------------------------------------------------
# Risk-free rate
# ---------------------------------------------------------------------------
# Annual risk-free rate used in the Sharpe, Sortino and alpha calculations.
# Source: RBI 91-day T-bill auction result dated 7 Oct 2026, cut-off yield
# 5.5747%. This is a single point-in-time yield applied to the whole 5-year
# window, so it is an approximation, not a measured average over the period.
RISK_FREE_RATE = 0.0557  # 5.57% per year

# ---------------------------------------------------------------------------
# Convention constants
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252  # standard count of NSE trading days in a year
ROLLING_WINDOW = 63          # about one quarter of trading days
VAR_CONFIDENCE = 0.95        # confidence level for VaR and CVaR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
REPORT_PATH = OUTPUT_DIR / "risk_report.md"

# Small synthetic file used by the tests and as an offline fallback.
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample" / "sample_prices_TEST_ONLY.csv"
