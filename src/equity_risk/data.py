"""Downloading, cleaning and validating the daily price history.

The rest of the project assumes one tidy DataFrame: a DatetimeIndex of
trading days, one column per ticker, holding adjusted closing prices.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config


class DataError(Exception):
    """Raised when the price data is unusable and nothing can be salvaged."""


@dataclass
class PriceData:
    """Prices plus a note of where they came from.

    Keeping the source on the object means the report can state honestly
    whether it was built from live data or from the synthetic sample.
    """

    prices: pd.DataFrame
    source: str          # "yfinance" or "sample"
    is_live: bool
    note: str = ""

    @property
    def start(self) -> pd.Timestamp:
        return self.prices.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.prices.index[-1]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_prices(
    tickers: list[str] | None = None,
    years: int = config.LOOKBACK_YEARS,
    end: dt.date | None = None,
) -> pd.DataFrame:
    """Download adjusted daily closing prices from Yahoo Finance.

    `auto_adjust=True` makes yfinance return prices already adjusted for
    splits and dividends, so the "Close" column it gives back is the
    adjusted close. Raises DataError if nothing usable comes back.
    """
    import yfinance as yf  # imported here so the tests never need the network

    tickers = list(tickers or config.ALL_TICKERS)
    end = end or dt.date.today()
    start = end - dt.timedelta(days=round(365.25 * years))

    raw = yf.download(
        tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="column",
        # One ticker at a time. Downloading in parallel makes the workers
        # fight over yfinance's small sqlite timezone cache, and a ticker
        # that loses that race comes back as an empty column.
        threads=False,
    )

    if raw is None or len(raw) == 0:
        raise DataError("yfinance returned an empty result for " + ", ".join(tickers))

    prices = _extract_close(raw, tickers)

    absent = [t for t in tickers if t not in prices.columns]
    if absent:
        raise DataError("no price column came back for: " + ", ".join(absent))

    # A ticker can come back as a column of nothing but NaN, which is a
    # failed download wearing the right shape. Catch it here rather than
    # letting the cleaning step quietly drop every row.
    blank = [t for t in tickers if prices[t].isna().all()]
    if blank:
        raise DataError("only empty prices came back for: " + ", ".join(blank))

    return prices[tickers]


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Pull the adjusted close out of whatever shape yfinance returned.

    With several tickers the columns are a MultiIndex of (field, ticker);
    with a single ticker they are just the field names.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        field = "Close" if "Close" in raw.columns.get_level_values(0) else "Adj Close"
        close = raw[field].copy()
    else:
        field = "Close" if "Close" in raw.columns else "Adj Close"
        close = raw[[field]].copy()
        close.columns = tickers[:1]
    return close


# ---------------------------------------------------------------------------
# Sample fallback
# ---------------------------------------------------------------------------
def load_sample_prices(path: Path | None = None) -> pd.DataFrame:
    """Read the synthetic sample CSV shipped in data/sample/.

    The file starts with '#' comment lines saying it is test data, so those
    are skipped on the way in.
    """
    path = Path(path or config.SAMPLE_DATA_PATH)
    if not path.exists():
        raise DataError(f"sample price file is missing: {path}")

    prices = pd.read_csv(path, comment="#", index_col="Date", parse_dates=["Date"])
    return prices.sort_index()


# ---------------------------------------------------------------------------
# Cleaning and validation
# ---------------------------------------------------------------------------
def missing_value_report(prices: pd.DataFrame) -> pd.DataFrame:
    """Count missing prices per ticker, before any cleaning."""
    counts = prices.isna().sum()
    return pd.DataFrame(
        {
            "missing_days": counts,
            "missing_pct": (counts / len(prices) * 100).round(2),
        }
    )


def clean_prices(prices: pd.DataFrame, max_fill_days: int = 3) -> pd.DataFrame:
    """Align the tickers onto one calendar and fill small gaps.

    Steps, in order:
      1. sort by date and drop duplicate dates (keeping the last one);
      2. drop days where every ticker is missing (holidays, non-trading days);
      3. carry the last price forward over short gaps, which is the usual
         way to patch a single missing print for one stock;
      4. drop any day that still has a gap, so all tickers share one index.
    """
    out = prices.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.dropna(how="all")
    out = out.ffill(limit=max_fill_days)
    out = out.dropna(how="any")
    return out


def validate_prices(prices: pd.DataFrame, min_rows: int = 100) -> None:
    """Check the cleaned frame is actually usable. Raises DataError if not."""
    if prices.empty:
        raise DataError("price frame is empty after cleaning")
    if len(prices) < min_rows:
        raise DataError(
            f"only {len(prices)} rows after cleaning, need at least {min_rows}"
        )
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise DataError("price index is not a DatetimeIndex")
    if not prices.index.is_monotonic_increasing:
        raise DataError("price index is not sorted by date")
    if prices.index.has_duplicates:
        raise DataError("price index has duplicate dates")
    if prices.isna().any().any():
        raise DataError("missing values survived cleaning")

    non_positive = [c for c in prices.columns if (prices[c] <= 0).any()]
    if non_positive:
        raise DataError("non-positive prices found in: " + ", ".join(non_positive))


# ---------------------------------------------------------------------------
# One call that does the whole job
# ---------------------------------------------------------------------------
def load_price_data(
    tickers: list[str] | None = None,
    years: int = config.LOOKBACK_YEARS,
    allow_sample_fallback: bool = True,
    verbose: bool = True,
) -> PriceData:
    """Fetch, clean and validate prices, falling back to the sample file.

    If the download fails for any reason the failure is printed in full and,
    unless `allow_sample_fallback` is False, the synthetic sample file is
    loaded instead. The returned PriceData says which of the two happened.
    """
    tickers = list(tickers or config.ALL_TICKERS)

    try:
        raw = download_prices(tickers, years=years)
        source, is_live, note = "yfinance", True, ""
        if verbose:
            print(f"Downloaded {len(raw)} rows from Yahoo Finance.")
    except Exception as exc:  # network errors, API changes, empty results
        message = f"{type(exc).__name__}: {exc}"
        print("DOWNLOAD FAILED -- could not get live prices from Yahoo Finance.")
        print(f"  reason: {message}")
        if not allow_sample_fallback:
            raise
        print("  falling back to the SYNTHETIC TEST SAMPLE in data/sample/.")
        print("  Results below are a smoke test of the code, not real market risk.")
        raw = load_sample_prices()
        raw = raw[[c for c in tickers if c in raw.columns]]
        source, is_live = "sample", False
        note = f"Live download failed ({message}); synthetic sample used instead."

    if verbose:
        gaps = missing_value_report(raw)
        total_missing = int(gaps["missing_days"].sum())
        print(f"Missing values before cleaning: {total_missing}")
        if total_missing:
            print(gaps[gaps["missing_days"] > 0].to_string())

    prices = clean_prices(raw)
    validate_prices(prices)

    if verbose:
        print(
            f"Clean data: {len(prices)} trading days, "
            f"{prices.index[0].date()} to {prices.index[-1].date()}, "
            f"{len(prices.columns)} tickers."
        )

    return PriceData(prices=prices, source=source, is_live=is_live, note=note)
