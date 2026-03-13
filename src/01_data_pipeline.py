"""
01_data_pipeline.py
===================
Phase 1 – Data Acquisition & Pre-processing for BTC/ETH Pairs Trading.

Fetches hourly OHLCV and funding-rate data for BTC/USDT:USDT and
ETH/USDT:USDT perpetual swaps on Binance, cleans and aligns them, then
saves a single Parquet file ready for downstream analysis.

Outputs
-------
data/btc_eth_1h.parquet   – cleaned, aligned DataFrame
data/price_overview.png   – log-price chart for quick visual inspection
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ccxt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXCHANGE_ID: str = "binanceusdm"          # Binance USD-M Futures (perpetuals)
SYMBOLS: dict[str, str] = {
    "btc": "BTC/USDT:USDT",
    "eth": "ETH/USDT:USDT",
}
TIMEFRAME: str = "1h"
SINCE_DT: datetime = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
UNTIL_DT: datetime = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
SINCE_MS: int = int(SINCE_DT.timestamp() * 1000)
UNTIL_MS: int = int(UNTIL_DT.timestamp() * 1000)

MAX_FORWARD_FILL_HOURS: int = 3           # max consecutive hours to forward-fill
REQUEST_DELAY_SEC: float = 0.5           # polite delay between API calls
OHLCV_LIMIT: int = 1000                  # candles per request (Binance max)
FUNDING_LIMIT: int = 1000                 # funding records per request

DATA_DIR: Path = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PARQUET: Path = DATA_DIR / "btc_eth_1h.parquet"
OUTPUT_CHART: Path = DATA_DIR / "price_overview.png"


# ---------------------------------------------------------------------------
# Exchange initialisation
# ---------------------------------------------------------------------------

def build_exchange() -> ccxt.Exchange:
    """
    Instantiate and load the Binance USD-M Futures exchange.

    Returns
    -------
    ccxt.Exchange
        Authenticated-free exchange object (read-only public endpoints).
    """
    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    exchange.load_markets()
    logger.info("Exchange '%s' initialised.", exchange.id)
    return exchange


# ---------------------------------------------------------------------------
# OHLCV fetcher
# ---------------------------------------------------------------------------

def fetch_ohlcv_paginated(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str = TIMEFRAME,
    since_ms: int = SINCE_MS,
    until_ms: int = UNTIL_MS,
    limit: int = OHLCV_LIMIT,
    delay: float = REQUEST_DELAY_SEC,
) -> pd.DataFrame:
    """
    Fetch complete OHLCV history for *symbol* using pagination.

    Parameters
    ----------
    exchange : ccxt.Exchange
        Initialised ccxt exchange object.
    symbol : str
        CCXT market symbol, e.g. ``'BTC/USDT:USDT'``.
    timeframe : str
        Candle period string, e.g. ``'1h'``.
    since_ms : int
        Start of the window, milliseconds UTC.
    until_ms : int
        End of the window (inclusive), milliseconds UTC.
    limit : int
        Number of candles per API request.
    delay : float
        Sleep time between requests (seconds).

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp`` (UTC datetime index), ``close``, ``volume``.
    """
    all_candles: list[list] = []
    cursor: int = since_ms

    logger.info("Fetching OHLCV for %s …", symbol)

    while cursor <= until_ms:
        try:
            candles = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=cursor, limit=limit
            )
        except ccxt.NetworkError as exc:
            logger.warning("Network error – retrying in 5s. %s", exc)
            time.sleep(5)
            continue
        except ccxt.ExchangeError as exc:
            logger.error("Exchange error: %s", exc)
            raise

        if not candles:
            break

        # Filter out any candles beyond our end date
        candles = [c for c in candles if c[0] <= until_ms]
        all_candles.extend(candles)

        last_ts: int = candles[-1][0]
        logger.debug("  Fetched up to %s (%d records so far)",
                     pd.Timestamp(last_ts, unit="ms", tz="UTC"), len(all_candles))

        if last_ts >= until_ms or len(candles) < limit:
            break

        # Advance cursor by one candle period to avoid overlap
        cursor = last_ts + exchange.parse_timeframe(timeframe) * 1000
        time.sleep(delay)

    if not all_candles:
        raise ValueError(f"No OHLCV data returned for {symbol}.")

    df = pd.DataFrame(all_candles, columns=["ts_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = (
        df[["timestamp", "close", "volume"]]
        .drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    logger.info("  → %d candles fetched for %s.", len(df), symbol)
    return df


# ---------------------------------------------------------------------------
# Funding rate fetcher
# ---------------------------------------------------------------------------

def fetch_funding_rates_paginated(
    exchange: ccxt.Exchange,
    symbol: str,
    since_ms: int = SINCE_MS,
    until_ms: int = UNTIL_MS,
    limit: int = FUNDING_LIMIT,
    delay: float = REQUEST_DELAY_SEC,
) -> pd.Series:
    """
    Fetch historical funding rates for *symbol* and return as a Series
    indexed by UTC timestamp.

    Binance settles funding every 8 hours (00:00, 08:00, 16:00 UTC).
    The returned Series contains NaN at all non-settlement hours; the
    caller is responsible for forward-filling onto the 1H grid.

    Parameters
    ----------
    exchange : ccxt.Exchange
        Initialised ccxt exchange object.
    symbol : str
        CCXT market symbol.
    since_ms : int
        Start of the window, milliseconds UTC.
    until_ms : int
        End of the window (inclusive), milliseconds UTC.
    limit : int
        Records per API request (max 1000 for Binance).
    delay : float
        Sleep time between requests (seconds).

    Returns
    -------
    pd.Series
        Name ``funding_rate``, UTC-indexed timestamps.
    """
    all_rates: list[dict] = []
    cursor: int = since_ms

    logger.info("Fetching funding rates for %s …", symbol)

    while cursor <= until_ms:
        try:
            rates = exchange.fetch_funding_rate_history(
                symbol, since=cursor, limit=limit
            )
        except ccxt.NetworkError as exc:
            logger.warning("Network error – retrying in 5s. %s", exc)
            time.sleep(5)
            continue
        except ccxt.ExchangeError as exc:
            logger.error("Exchange error: %s", exc)
            raise

        if not rates:
            break

        rates = [r for r in rates if r["timestamp"] <= until_ms]
        all_rates.extend(rates)

        last_ts: int = rates[-1]["timestamp"]
        logger.debug("  Funding rates up to %s (%d records so far)",
                     pd.Timestamp(last_ts, unit="ms", tz="UTC"), len(all_rates))

        if last_ts >= until_ms or len(rates) < limit:
            break

        cursor = last_ts + 1
        time.sleep(delay)

    if not all_rates:
        logger.warning("No funding rate data returned for %s. Returning empty series.", symbol)
        return pd.Series(name="funding_rate", dtype=float)

    df = pd.DataFrame(all_rates)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    series = (
        df[["timestamp", "fundingRate"]]
        .drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .set_index("timestamp")["fundingRate"]
        .rename("funding_rate")
    )
    logger.info("  → %d funding rate records for %s.", len(series), symbol)
    return series


# ---------------------------------------------------------------------------
# Data cleaning helpers
# ---------------------------------------------------------------------------

def clean_ohlcv(
    df: pd.DataFrame,
    asset: str,
    max_fill: int = MAX_FORWARD_FILL_HOURS,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Fill short OHLCV gaps and drop rows belonging to long gaps.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame with a UTC datetime index.
    asset : str
        Human-readable label (``'BTC'`` / ``'ETH'``) used in log messages.
    max_fill : int
        Maximum consecutive missing hours eligible for forward-fill.

    Returns
    -------
    (pd.DataFrame, list[str])
        Cleaned DataFrame and a list of warning/info log strings.
    """
    log_entries: list[str] = []

    # Build a regular 1H grid for the *observed* date range
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="1h", tz="UTC")
    df = df.reindex(full_idx)

    # Identify NaN runs
    is_nan = df["close"].isna()
    # Group consecutive NaN blocks
    nan_groups = is_nan.ne(is_nan.shift()).cumsum()[is_nan]

    rows_to_drop: pd.DatetimeIndex = pd.DatetimeIndex([])
    for group_id, group_idx in nan_groups.groupby(nan_groups):
        run_len = len(group_idx)
        start_ts = group_idx.index[0]
        end_ts = group_idx.index[-1]

        if run_len > max_fill:
            msg = (
                f"[WARNING] {asset}: dropping {run_len}-hour gap "
                f"({start_ts} → {end_ts}) – exceeds {max_fill}-hour limit."
            )
            logger.warning(msg)
            log_entries.append(msg)
            rows_to_drop = rows_to_drop.append(group_idx.index)
        else:
            msg = (
                f"[INFO]    {asset}: forward-filling {run_len}-hour gap "
                f"({start_ts} → {end_ts})."
            )
            logger.info(msg)
            log_entries.append(msg)

    df = df.drop(index=rows_to_drop)
    df = df.ffill()          # forward-fill remaining short gaps

    return df, log_entries


def align_funding_to_1h(
    ohlcv_index: pd.DatetimeIndex,
    funding: pd.Series,
) -> pd.Series:
    """
    Align funding rates (8H settlements) to a 1H OHLCV index.

    Strategy:
      1. Reindex onto the OHLCV index.
      2. Forward-fill so each hour carries the most recent settlement rate.
      3. Fill any remaining NaN (before the first settlement) with 0.

    Parameters
    ----------
    ohlcv_index : pd.DatetimeIndex
        Target hourly index.
    funding : pd.Series
        Funding-rate Series (sparse, settlement timestamps only).

    Returns
    -------
    pd.Series
        Dense hourly Series aligned to *ohlcv_index*.
    """
    aligned = funding.reindex(ohlcv_index.union(funding.index)).sort_index()
    aligned = aligned.ffill().reindex(ohlcv_index).fillna(0.0)
    return aligned


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> pd.DataFrame:
    """
    Orchestrate the full data pipeline.

    Steps
    -----
    1. Fetch OHLCV for BTC and ETH.
    2. Clean each series (gap handling).
    3. Fetch and align funding rates.
    4. Inner-join on UTC timestamp.
    5. Add log-price columns.
    6. Save Parquet and print summary.
    7. Plot and save log-price chart.

    Returns
    -------
    pd.DataFrame
        The final, cleaned, aligned DataFrame.
    """
    exchange = build_exchange()
    cleaning_log: list[str] = []

    # ------------------------------------------------------------------
    # 1. Fetch OHLCV
    # ------------------------------------------------------------------
    raw: dict[str, pd.DataFrame] = {}
    for name, symbol in SYMBOLS.items():
        raw[name] = fetch_ohlcv_paginated(exchange, symbol)

    # ------------------------------------------------------------------
    # 2. Clean OHLCV
    # ------------------------------------------------------------------
    clean: dict[str, pd.DataFrame] = {}
    for name, df in raw.items():
        cleaned_df, log_entries = clean_ohlcv(df, asset=name.upper())
        clean[name] = cleaned_df
        cleaning_log.extend(log_entries)

    # ------------------------------------------------------------------
    # 3. Fetch funding rates & align
    # ------------------------------------------------------------------
    funding: dict[str, pd.Series] = {}
    for name, symbol in SYMBOLS.items():
        fr_raw = fetch_funding_rates_paginated(exchange, symbol)
        funding[name] = fr_raw

    # ------------------------------------------------------------------
    # 4. Inner join on UTC timestamp
    # ------------------------------------------------------------------
    btc = clean["btc"].rename(columns={"close": "btc_close", "volume": "btc_volume"})
    eth = clean["eth"].rename(columns={"close": "eth_close", "volume": "eth_volume"})

    merged = btc.join(eth, how="inner")

    # Align funding onto merged index
    merged["btc_funding_rate"] = align_funding_to_1h(merged.index, funding["btc"])
    merged["eth_funding_rate"] = align_funding_to_1h(merged.index, funding["eth"])

    # ------------------------------------------------------------------
    # 5. Log prices
    # ------------------------------------------------------------------
    merged["btc_log_price"] = np.log(merged["btc_close"])
    merged["eth_log_price"] = np.log(merged["eth_close"])

    # Drop any all-NaN rows that survived (safety net)
    before = len(merged)
    merged = merged.dropna(subset=["btc_close", "eth_close"])
    dropped = before - len(merged)
    if dropped:
        logger.warning("Dropped %d residual NaN rows after join.", dropped)

    # ------------------------------------------------------------------
    # 6. Save & print summary
    # ------------------------------------------------------------------
    merged.to_parquet(OUTPUT_PARQUET)
    logger.info("Saved → %s", OUTPUT_PARQUET)

    _print_summary(merged, cleaning_log, dropped)

    # ------------------------------------------------------------------
    # 7. Plot
    # ------------------------------------------------------------------
    plot_log_prices(merged)

    return merged


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(
    df: pd.DataFrame,
    cleaning_log: list[str],
    residual_drops: int,
) -> None:
    """
    Print a formatted pipeline summary to stdout.

    Parameters
    ----------
    df : pd.DataFrame
        Final cleaned DataFrame.
    cleaning_log : list[str]
        Messages collected during the cleaning step.
    residual_drops : int
        Rows dropped after the final inner-join NaN check.
    """
    sep = "=" * 70
    print(f"\n{sep}")
    print("  DATA PIPELINE SUMMARY")
    print(sep)
    print(f"  Total rows        : {len(df):,}")
    print(f"  Start (UTC)       : {df.index.min()}")
    print(f"  End   (UTC)       : {df.index.max()}")
    print(f"  Columns           : {list(df.columns)}")
    print(f"  Remaining NaNs    : {df.isna().sum().to_dict()}")
    print(f"  Residual drops    : {residual_drops}")
    print(f"\n  --- Gap / Fill Log ({len(cleaning_log)} entries) ---")
    if cleaning_log:
        for entry in cleaning_log:
            print(f"  {entry}")
    else:
        print("  (no gaps detected)")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_log_prices(df: pd.DataFrame, output_path: Path = OUTPUT_CHART) -> None:
    """
    Plot BTC and ETH log-prices on a shared axis and save to disk.

    The chart uses a dark background with a twin-y-axis layout so that
    two assets at different price levels are both clearly visible.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``btc_log_price`` and ``eth_log_price`` columns with a
        UTC DatetimeIndex.
    output_path : Path
        Destination file (PNG).
    """
    plt.style.use("dark_background")

    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax2 = ax1.twinx()

    color_btc = "#F7931A"   # Bitcoin orange
    color_eth = "#627EEA"   # Ethereum blue

    ax1.plot(df.index, df["btc_log_price"], color=color_btc,
             linewidth=0.9, label="BTC log-price", alpha=0.95)
    ax2.plot(df.index, df["eth_log_price"], color=color_eth,
             linewidth=0.9, label="ETH log-price", alpha=0.95)

    # Axes labels
    ax1.set_ylabel("BTC log-price (ln USD)", color=color_btc, fontsize=11)
    ax2.set_ylabel("ETH log-price (ln USD)", color=color_eth, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_btc)
    ax2.tick_params(axis="y", labelcolor=color_eth)

    # X-axis formatting
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=35, ha="right")

    # Title & legend
    ax1.set_title(
        "BTC / ETH Log-Price (1H, Binance Perpetual Swaps)\n"
        f"{df.index.min().date()} → {df.index.max().date()}",
        fontsize=14, pad=14,
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", fontsize=10, framealpha=0.3)

    ax1.grid(alpha=0.15, linestyle="--")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved → %s", output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
