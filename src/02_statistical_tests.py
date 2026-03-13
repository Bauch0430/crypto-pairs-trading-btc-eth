"""
02_statistical_tests.py
========================
Phase 2 – Rolling Statistical Validation for BTC/ETH Pairs Trading.

Computes, for every hour t:
  • rolling_beta  : OLS hedge ratio estimated on [t-W, t-1] (NO look-ahead)
  • spread        : btc_log_price_t - rolling_beta_t × eth_log_price_t
  • adf_pvalue    : ADF test on the spread within [t-W, t-1]
  • half_life     : OU mean-reversion speed estimated on [t-W, t-1]

Design choices to avoid look-ahead bias
-----------------------------------------
  • RollingOLS is run with the full window, then ALL params are shifted
    forward by one row.  The beta used at time t therefore comes from
    a regression that only saw data up to t-1.
  • ADF and half-life are computed on the spread series lagged in the same
    way – i.e., the window fed into the test ends at t-1.

Outputs
-------
data/statistical_features.parquet
data/rolling_beta_plot.png
data/spread_and_pvalue.png
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & parameters
# ---------------------------------------------------------------------------
DATA_DIR: Path = Path(__file__).parent.parent / "data"
INPUT_PARQUET: Path = DATA_DIR / "btc_eth_1h.parquet"
OUTPUT_PARQUET: Path = DATA_DIR / "statistical_features.parquet"
BETA_PLOT: Path = DATA_DIR / "rolling_beta_plot.png"
SPREAD_PLOT: Path = DATA_DIR / "spread_and_pvalue.png"

WINDOW: int = 720          # 30 days × 24 h/day
ADF_MAX_LAG: int = 10      # max lags for ADF (BIC selection)
PVALUE_THRESHOLD: float = 0.05


# ---------------------------------------------------------------------------
# 1. Rolling OLS – hedge ratio β
# ---------------------------------------------------------------------------

def compute_rolling_beta(
    Y: pd.Series,
    X: pd.Series,
    window: int = WINDOW,
) -> pd.Series:
    """
    Estimate a rolling OLS hedge ratio β with strict look-ahead prevention.

    At time t we regress Y ~ X using observations in [t-window, t-1].
    Practically, statsmodels RollingOLS uses [t-window+1, t]; we then
    shift the resulting params forward by one period so that the beta
    labelled at t was estimated from data ending at t-1.

    Parameters
    ----------
    Y : pd.Series
        Dependent variable (btc_log_price).
    X : pd.Series
        Independent variable (eth_log_price).
    window : int
        Number of hourly observations in each rolling window.

    Returns
    -------
    pd.Series
        ``rolling_beta`` aligned to the original index; the first
        ``window`` values are NaN (warm-up period).
    """
    logger.info("Computing rolling OLS (window=%d) …", window)

    X_const = sm.add_constant(X, prepend=False)   # [X, 1]
    model = RollingOLS(endog=Y, exog=X_const, window=window, min_nobs=window)
    rres = model.fit(method="lstsq", reset=int(window / 2))

    # params columns: [X_col_name, 'const']
    beta_raw: pd.Series = rres.params.iloc[:, 0]   # slope on X

    # ── CRITICAL: shift forward by 1 to eliminate look-ahead bias ──────────
    # After shift, beta[t] = regression result from data [t-window, t-1]
    beta_shifted: pd.Series = beta_raw.shift(1)
    beta_shifted.name = "rolling_beta"

    valid = beta_shifted.notna().sum()
    logger.info("  → rolling_beta computed. Valid rows: %d / %d", valid, len(beta_shifted))
    return beta_shifted


# ---------------------------------------------------------------------------
# 2. Spread
# ---------------------------------------------------------------------------

def compute_spread(
    Y: pd.Series,
    X: pd.Series,
    beta: pd.Series,
) -> pd.Series:
    """
    Compute the cointegration spread at each time step.

    Spread_t = Y_t - β_t × X_t

    Because β_t was estimated on [t-W, t-1], this is an out-of-sample
    spread – no look-ahead.

    Parameters
    ----------
    Y : pd.Series
        Dependent log-price series (btc_log_price).
    X : pd.Series
        Independent log-price series (eth_log_price).
    beta : pd.Series
        Rolling hedge ratio aligned to the same index.

    Returns
    -------
    pd.Series
        Spread series named ``spread``.
    """
    spread = Y - beta * X
    spread.name = "spread"
    return spread


# ---------------------------------------------------------------------------
# 3. Rolling ADF & Half-life helpers (numpy for speed)
# ---------------------------------------------------------------------------

def _ols_1d(y: np.ndarray, x: np.ndarray) -> float:
    """
    Compute the OLS coefficient of x in the regression y ~ x (no intercept).

    Fast path: closed-form  β = (xᵀx)⁻¹ xᵀy.

    Parameters
    ----------
    y : np.ndarray  shape (n,)
    x : np.ndarray  shape (n,)

    Returns
    -------
    float
        Scalar OLS coefficient.
    """
    denom = float(x @ x)
    if denom == 0.0:
        return np.nan
    return float(x @ y) / denom


def _adf_pvalue(series: np.ndarray, max_lag: int = ADF_MAX_LAG) -> float:
    """
    Run the Augmented Dickey-Fuller test on a 1-D array and return the p-value.

    Parameters
    ----------
    series : np.ndarray
        The time series to test (spread within one window).
    max_lag : int
        Maximum number of lags; BIC is used for automatic selection.

    Returns
    -------
    float
        ADF p-value; returns NaN if the test fails.
    """
    try:
        result = adfuller(series, maxlag=max_lag, autolag="BIC", regression="c")
        return float(result[1])
    except Exception:
        return np.nan


def _half_life(series: np.ndarray) -> float:
    """
    Estimate the Ornstein-Uhlenbeck mean-reversion half-life in hours.

    Method: AR(1) on the spread increments.
        ΔS_t = α + λ·S_{t-1} + ε_t

    Half-life  = −ln(2) / ln(1 + λ̂)
               ≈ −ln(2) / λ̂  when |λ̂| is small.

    Parameters
    ----------
    series : np.ndarray
        Spread values within one window.

    Returns
    -------
    float
        Half-life in hours; returns NaN if estimation fails or diverges.
    """
    if len(series) < 3:
        return np.nan

    s_lag = series[:-1]
    delta_s = series[1:] - s_lag

    # OLS: δS = α + λ·S_{t-1}
    X = np.column_stack([np.ones(len(s_lag)), s_lag])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, delta_s, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan

    lam = float(coeffs[1])   # mean-reversion speed

    # If λ ≥ 0, the process is not mean-reverting → return NaN
    if lam >= 0.0:
        return np.nan

    hl = -np.log(2.0) / np.log(1.0 + lam)
    # Clip unreasonably long half-lives to reduce noise in the chart
    if hl <= 0 or hl > 10_000:
        return np.nan
    return float(hl)


# ---------------------------------------------------------------------------
# 4. Vectorised rolling ADF + half-life loop
# ---------------------------------------------------------------------------

def compute_rolling_adf_and_halflife(
    spread: pd.Series,
    window: int = WINDOW,
) -> tuple[pd.Series, pd.Series]:
    """
    Walk forward through the spread series and compute, for each t, the
    ADF p-value and OU half-life estimated on the window [t-window, t-1].

    Look-ahead is prevented because the window fed to each test ends at
    t-1 (i.e., index ``i-1`` in the underlying ndarray when evaluating
    at position ``i``).

    Parameters
    ----------
    spread : pd.Series
        The spread series with a UTC DatetimeIndex.
    window : int
        Look-back window size (number of observations).

    Returns
    -------
    (pd.Series, pd.Series)
        ``adf_pvalue``  – p-value from ADF test at each t
        ``half_life``   – OU half-life (hours) at each t
    """
    n = len(spread)
    arr = spread.to_numpy(dtype=float)
    idx = spread.index

    adf_pvals = np.full(n, np.nan)
    half_lives = np.full(n, np.nan)

    logger.info("Running rolling ADF & half-life (window=%d, n=%d) …", window, n)

    # Log progress every 10 %
    checkpoint = max(1, n // 10)

    for i in range(window + 1, n):    # +1 so window [i-window-1 … i-1] never touches i
        win_slice = arr[i - window - 1 : i - 1]   # exactly `window` obs ending at t-1

        if np.isnan(win_slice).any():
            continue

        adf_pvals[i] = _adf_pvalue(win_slice)
        half_lives[i] = _half_life(win_slice)

        if i % checkpoint == 0:
            pct = 100 * i / n
            logger.info("  … %.0f%% complete (%d / %d rows)", pct, i, n)

    logger.info("  → ADF & half-life loop complete.")

    return (
        pd.Series(adf_pvals, index=idx, name="adf_pvalue"),
        pd.Series(half_lives, index=idx, name="half_life"),
    )


# ---------------------------------------------------------------------------
# 5. Summary printer
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    """
    Print a formatted statistical summary of the feature DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``rolling_beta``, ``spread``, ``adf_pvalue``,
        ``half_life`` columns.
    """
    stat_rows = df.dropna(subset=["rolling_beta", "spread", "adf_pvalue", "half_life"])
    pct_cointegrated = (stat_rows["adf_pvalue"] < PVALUE_THRESHOLD).mean() * 100

    sep = "=" * 70
    print(f"\n{sep}")
    print("  STATISTICAL FEATURES SUMMARY")
    print(sep)
    print(f"  Total rows            : {len(df):,}")
    print(f"  Valid (post warm-up)  : {len(stat_rows):,}")
    print(f"  Warm-up rows (NaN)    : {len(df) - len(stat_rows):,}")
    print(f"\n  --- Rolling Beta (β) ---")
    print(f"  Mean                  : {stat_rows['rolling_beta'].mean():.4f}")
    print(f"  Std                   : {stat_rows['rolling_beta'].std():.4f}")
    print(f"  Min / Max             : {stat_rows['rolling_beta'].min():.4f} / {stat_rows['rolling_beta'].max():.4f}")
    print(f"\n  --- Spread ---")
    print(f"  Mean                  : {stat_rows['spread'].mean():.4f}")
    print(f"  Std (σ)               : {stat_rows['spread'].std():.4f}")
    print(f"\n  --- ADF p-value ---")
    print(f"  Mean                  : {stat_rows['adf_pvalue'].mean():.4f}")
    print(f"  % hours p < {PVALUE_THRESHOLD:.2f}      : {pct_cointegrated:.1f}%")
    print(f"\n  --- Half-life (OU, hours) ---")
    print(f"  Median                : {stat_rows['half_life'].median():.1f}")
    print(f"  Mean                  : {stat_rows['half_life'].mean():.1f}")
    print(f"  10th / 90th pct       : {stat_rows['half_life'].quantile(0.1):.1f} / "
          f"{stat_rows['half_life'].quantile(0.9):.1f}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# 6. Visualisations
# ---------------------------------------------------------------------------

def plot_rolling_beta(df: pd.DataFrame, output_path: Path = BETA_PLOT) -> None:
    """
    Plot the rolling OLS hedge ratio (β) over time.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``rolling_beta`` column with a UTC DatetimeIndex.
    output_path : Path
        Destination PNG file.
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 5))

    color_beta = "#00E5FF"

    series = df["rolling_beta"].dropna()
    ax.plot(series.index, series.values, color=color_beta, linewidth=0.8,
            alpha=0.9, label=f"Rolling β (W={WINDOW}h)")

    # Horizontal reference at β = 1
    ax.axhline(1.0, color="#FF6B6B", linewidth=0.9, linestyle="--",
               alpha=0.7, label="β = 1 (equal weight)")

    # Shaded ±1 std band
    mu = series.mean()
    sd = series.std()
    ax.axhspan(mu - sd, mu + sd, alpha=0.08, color=color_beta,
               label=f"μ ± 1σ  ({mu:.3f} ± {sd:.3f})")
    ax.axhline(mu, color=color_beta, linewidth=0.6, linestyle=":", alpha=0.5)

    ax.set_ylabel("Hedge Ratio β  (BTC ~ β·ETH + α)", fontsize=11)
    ax.set_title(
        f"Rolling OLS Hedge Ratio — BTC/ETH (Window = {WINDOW} hours / 30 days)",
        fontsize=14, pad=12,
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=35, ha="right")
    ax.legend(fontsize=10, framealpha=0.3, loc="upper right")
    ax.grid(alpha=0.12, linestyle="--")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved → %s", output_path)


def plot_spread_and_pvalue(df: pd.DataFrame, output_path: Path = SPREAD_PLOT) -> None:
    """
    Two-panel plot: (top) spread time series with ±2σ bands,
    (bottom) rolling ADF p-value with 5 % significance line.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``spread`` and ``adf_pvalue`` columns.
    output_path : Path
        Destination PNG file.
    """
    plt.style.use("dark_background")
    fig, (ax_spread, ax_pval) = plt.subplots(
        2, 1, figsize=(16, 9), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
    )

    # ── Top panel: Spread ──────────────────────────────────────────────────
    color_spread = "#F7CF47"
    spread = df["spread"].dropna()
    mu_s = spread.mean()
    sd_s = spread.std()

    ax_spread.plot(spread.index, spread.values, color=color_spread,
                   linewidth=0.7, alpha=0.9, label="Spread")
    ax_spread.axhline(mu_s, color="white", linewidth=0.8, linestyle="--",
                      alpha=0.5, label=f"Mean ({mu_s:.3f})")
    ax_spread.fill_between(spread.index, mu_s - 2 * sd_s, mu_s + 2 * sd_s,
                           alpha=0.10, color=color_spread, label="±2σ band")
    ax_spread.axhline(mu_s + 2 * sd_s, color=color_spread, linewidth=0.6,
                      linestyle=":", alpha=0.45)
    ax_spread.axhline(mu_s - 2 * sd_s, color=color_spread, linewidth=0.6,
                      linestyle=":", alpha=0.45)

    ax_spread.set_ylabel("Spread  (btc_log − β·eth_log)", fontsize=11)
    ax_spread.set_title(
        "BTC/ETH Cointegration Spread & Rolling ADF p-value  "
        f"(Window = {WINDOW} h)",
        fontsize=14, pad=12,
    )
    ax_spread.legend(fontsize=10, framealpha=0.3, loc="upper right")
    ax_spread.grid(alpha=0.12, linestyle="--")

    # ── Bottom panel: ADF p-value ──────────────────────────────────────────
    color_pval_low = "#4CAF50"    # green when cointegrated
    color_pval_high = "#F44336"   # red when not cointegrated

    pval = df["adf_pvalue"].dropna()

    # Colour-code by significance
    ax_pval.fill_between(pval.index, pval.values, PVALUE_THRESHOLD,
                         where=(pval.values < PVALUE_THRESHOLD),
                         interpolate=True, alpha=0.35, color=color_pval_low,
                         label=f"p < {PVALUE_THRESHOLD} (cointegrated)")
    ax_pval.fill_between(pval.index, pval.values, PVALUE_THRESHOLD,
                         where=(pval.values >= PVALUE_THRESHOLD),
                         interpolate=True, alpha=0.35, color=color_pval_high,
                         label=f"p ≥ {PVALUE_THRESHOLD} (not cointegrated)")
    ax_pval.plot(pval.index, pval.values, color="white", linewidth=0.5,
                 alpha=0.6)
    ax_pval.axhline(PVALUE_THRESHOLD, color="#FFD700", linewidth=1.2,
                    linestyle="--", label=f"p = {PVALUE_THRESHOLD} threshold")

    ax_pval.set_ylabel("ADF p-value", fontsize=11)
    ax_pval.set_ylim(-0.02, 1.05)
    ax_pval.set_xlabel("Date (UTC)", fontsize=11)
    ax_pval.legend(fontsize=10, framealpha=0.3, loc="upper right")
    ax_pval.grid(alpha=0.12, linestyle="--")

    # Shared x-axis formatting
    ax_pval.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_pval.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=35, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved → %s", output_path)


# ---------------------------------------------------------------------------
# 7. Main orchestrator
# ---------------------------------------------------------------------------

def run_statistical_tests() -> pd.DataFrame:
    """
    Full Phase-2 pipeline: load data → rolling OLS → spread → ADF →
    half-life → save & plot.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame with ``rolling_beta``, ``spread``,
        ``adf_pvalue``, ``half_life`` appended.
    """
    # ── Load ──────────────────────────────────────────────────────────────
    logger.info("Loading %s …", INPUT_PARQUET)
    df = pd.read_parquet(INPUT_PARQUET)
    logger.info("  → %d rows loaded.", len(df))

    Y: pd.Series = df["btc_log_price"]
    X: pd.Series = df["eth_log_price"]

    # ── Rolling β ─────────────────────────────────────────────────────────
    df["rolling_beta"] = compute_rolling_beta(Y, X, window=WINDOW)

    # ── Spread ────────────────────────────────────────────────────────────
    df["spread"] = compute_spread(Y, X, df["rolling_beta"])

    # ── ADF & half-life ───────────────────────────────────────────────────
    # Feed the *spread* series so ADF uses the residual already corrected
    # by the time-varying β (which itself had no look-ahead).
    df["adf_pvalue"], df["half_life"] = compute_rolling_adf_and_halflife(
        df["spread"], window=WINDOW
    )

    # ── Save ──────────────────────────────────────────────────────────────
    df.to_parquet(OUTPUT_PARQUET)
    logger.info("Saved → %s", OUTPUT_PARQUET)

    # ── Summary ───────────────────────────────────────────────────────────
    _print_summary(df)

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_rolling_beta(df)
    plot_spread_and_pvalue(df)

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_statistical_tests()
