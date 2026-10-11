# Test sample data

`sample_prices_TEST_ONLY.csv` is **synthetic data**, not live market data.

The prices come from a seeded random walk. They exist for two reasons only:

1. the unit tests need a small, fixed price file that never changes, and
2. the pipeline needs something to fall back on when the yfinance download
   fails (for example, with no internet connection).

The numbers do not represent the real price history of RELIANCE, TCS,
HDFCBANK or the NIFTY 50. Any report generated from this file is a smoke
test of the code, not a statement about the market, and the report says so
at the top when the fallback is used.
