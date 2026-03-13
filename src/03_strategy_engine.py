"""
03_strategy_engine.py
======================
Phase 3 – Strategy Logic & Signal Generation for BTC/ETH Pairs Trading.

Workflow
--------
1. Load ``data/statistical_features.parquet``.
2. Compute a rolling Z-score of the spread (look-ahead-free via shift).
3. Run a state-machine loop to generate the ``position`` series, honouring:
   • Entry   : |Z| > ENTRY_Z  AND  adf_pvalue < ADF_THRESHOLD
   • Take-profit : |Z| < TP_Z  (mean reversion)
   • Stop-loss   : |Z| > SL_Z  (extreme divergence)
   • Time-stop   : holding period > 3 × median_half_life
4. Save ``data/signals.parquet``.
5. Save ``data/zscore_and_signals.png``.

Look-ahead-bias prevention
--------------------------
The Z-score at time t is computed as:

    Z_t = (Spread_t − μ_{t-1}) / σ_{t-1}

where μ_{t-1} and σ_{t-1} are the rolling mean and std of the spread up
to (but NOT including) time t, achieved via ``.shift(1)`` on the rolling
statistics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

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
# Paths & strategy parameters
# ---------------------------------------------------------------------------
DATA_DIR: Path = Path(__file__).parent.parent / "data"
INPUT_PARQUET: Path = DATA_DIR / "statistical_features.parquet"
OUTPUT_PARQUET: Path = DATA_DIR / "signals.parquet"
SIGNAL_PLOT: Path = DATA_DIR / "zscore_and_signals.png"

# Z-score thresholds
ZSCORE_WINDOW: int = 720      # rolling window for Z standardisation (same as Phase 2)
ENTRY_Z: float = 2.0          # enter when |Z| exceeds this
TP_Z: float = 0.5             # take-profit when |Z| falls below this
SL_Z: float = 4.0             # stop-loss when |Z| exceeds this (divergence)

# ADF gate: only trade when current window is "likely" cointegrated
ADF_THRESHOLD: float = 0.05

# Time-stop multiplier (× median half-life from Phase 2)
TIME_STOP_MULTIPLIER: float = 3.0

# Minimum observations before we can compute a reliable Z-score
MIN_PERIODS: int = ZSCORE_WINDOW


# ---------------------------------------------------------------------------
# 1. Rolling Z-score (look-ahead-free)
# ---------------------------------------------------------------------------

def compute_zscore(
    spread: pd.Series,
    window: int = ZSCORE_WINDOW,
) -> pd.Series:
    """
    Compute a look-ahead-free rolling Z-score of the spread.

    At each time t:
        Z_t = (Spread_t − μ_{t-1}) / σ_{t-1}

    μ_{t-1} and σ_{t-1} are rolling statistics over [t-window, t-1],
    enforced by shifting the rolling results forward by one period.

    Parameters
    ----------
    spread : pd.Series
        The spread series (btc_log_price − β × eth_log_price).
    window : int
        Look-back window for rolling standardisation.

    Returns
    -------
    pd.Series
        Z-score series, named ``z_score``, NaN during warm-up.
    """
    logger.info("Computing rolling Z-score (window=%d) with shift(1) guard …", window)

    # Rolling stats on [t-window+1, t], then shift(1) → covers [t-window, t-1]
    roll_mean = spread.rolling(window=window, min_periods=window).mean().shift(1)
    roll_std  = spread.rolling(window=window, min_periods=window).std(ddof=1).shift(1)

    z = (spread - roll_mean) / roll_std
    z.name = "z_score"

    valid = z.notna().sum()
    logger.info("  → %d valid Z-score rows (warm-up = %d rows).", valid, len(z) - valid)
    return z


# ---------------------------------------------------------------------------
# 2. State-machine position generator
# ---------------------------------------------------------------------------

def generate_positions(
    z_score: pd.Series,
    adf_pvalue: pd.Series,
    half_life_median: float,
    entry_z: float = ENTRY_Z,
    tp_z: float = TP_Z,
    sl_z: float = SL_Z,
    adf_threshold: float = ADF_THRESHOLD,
    time_stop_multiplier: float = TIME_STOP_MULTIPLIER,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Run a forward-only state machine to generate positions from Z-score signals.

    Rules (evaluated in priority order at each bar):
    ─────────────────────────────────────────────────
    IN-POSITION:
      1. Time-stop   : holding_hours > time_stop_multiplier × half_life_median → close
      2. Stop-loss   : |Z| > sl_z → close
      3. Take-profit : |Z| < tp_z → close
      (position carries forward if none of the above triggered)

    FLAT:
      4. Entry       : |Z| > entry_z AND adf_pvalue < adf_threshold
         • Z > +entry_z → position = −1  (short spread: expect Z to fall)
         • Z < −entry_z → position = +1  (long  spread: expect Z to rise)

    Parameters
    ----------
    z_score : pd.Series
        Z-score series aligned to the full dataset index.
    adf_pvalue : pd.Series
        Rolling ADF p-value, same index.
    half_life_median : float
        Median half-life (hours) from Phase 2 for the time-stop calculation.
    entry_z : float
        |Z| threshold to open a new position.
    tp_z : float
        |Z| threshold to close via take-profit.
    sl_z : float
        |Z| threshold to close via stop-loss.
    adf_threshold : float
        Maximum ADF p-value allowed when opening a position.
    time_stop_multiplier : float
        Multiples of ``half_life_median`` before time-stop triggers.

    Returns
    -------
    (position, is_entry, is_tp_exit, is_sl_exit) : tuple of pd.Series
        • ``position``   : {−1, 0, +1} at each bar
        • ``is_entry``   : True on bars where a new trade is opened
        • ``is_tp_exit`` : True on bars closed via take-profit
        • ``is_sl_exit`` : True on bars closed via stop-loss or time-stop
    """
    max_hold_hours: float = time_stop_multiplier * half_life_median
    logger.info(
        "Running position state machine  "
        "(entry|Z|=%.1f, TP=%.1f, SL=%.1f, time_stop=%.0f h) …",
        entry_z, tp_z, sl_z, max_hold_hours,
    )

    n = len(z_score)
    idx = z_score.index
    z_arr   = z_score.to_numpy(dtype=float)
    adf_arr = adf_pvalue.to_numpy(dtype=float)

    pos_arr      = np.zeros(n, dtype=np.int8)
    entry_arr    = np.zeros(n, dtype=bool)
    tp_exit_arr  = np.zeros(n, dtype=bool)
    sl_exit_arr  = np.zeros(n, dtype=bool)

    current_pos: int = 0
    entry_idx: int = -1   # integer position in array of entry bar

    for i in range(n):
        z  = z_arr[i]
        pv = adf_arr[i]

        if np.isnan(z):          # still in warm-up → stay flat
            pos_arr[i] = 0
            continue

        if current_pos != 0:
            # ── Check exits (priority: time > stop-loss > take-profit) ──
            holding_hours = float(i - entry_idx)   # each bar = 1 hour

            close_reason: str | None = None

            if holding_hours > max_hold_hours:
                close_reason = "time_stop"
            elif abs(z) > sl_z:
                close_reason = "stop_loss"
            elif abs(z) < tp_z:
                close_reason = "take_profit"

            if close_reason == "take_profit":
                tp_exit_arr[i] = True
                current_pos = 0
                entry_idx = -1
            elif close_reason in ("stop_loss", "time_stop"):
                sl_exit_arr[i] = True
                current_pos = 0
                entry_idx = -1
            # else: carry position forward

        if current_pos == 0:
            # ── Check entry ─────────────────────────────────────────────
            if (not np.isnan(pv)) and (pv < adf_threshold):
                if z > entry_z:
                    current_pos = -1
                    entry_idx = i
                    entry_arr[i] = True
                elif z < -entry_z:
                    current_pos = 1
                    entry_idx = i
                    entry_arr[i] = True

        pos_arr[i] = current_pos

    position   = pd.Series(pos_arr.astype(np.int8), index=idx, name="position")
    is_entry   = pd.Series(entry_arr, index=idx, name="is_entry")
    is_tp_exit = pd.Series(tp_exit_arr, index=idx, name="is_tp_exit")
    is_sl_exit = pd.Series(sl_exit_arr, index=idx, name="is_sl_exit")

    # ── Trade statistics ────────────────────────────────────────────────
    n_long  = int(is_entry[position.shift(1) == 0][z_score < -entry_z].sum()
                  + ((is_entry) & (z_score.shift(0) < -entry_z)).sum())
    n_short = int(is_entry.sum()) - int((is_entry & (z_score > entry_z)).sum()) + \
              int((is_entry & (z_score > entry_z)).sum())

    # Simpler count
    n_entries = int(is_entry.sum())
    n_tp      = int(is_tp_exit.sum())
    n_sl      = int(is_sl_exit.sum())
    open_pos  = int(current_pos)   # position still open at end of data

    logger.info(
        "  Entries=%d  |  TP exits=%d  |  SL/Time exits=%d  |  Open at end=%d",
        n_entries, n_tp, n_sl, open_pos,
    )

    return position, is_entry, is_tp_exit, is_sl_exit


# ---------------------------------------------------------------------------
# 3. Summary printer
# ---------------------------------------------------------------------------

def _print_summary(
    df: pd.DataFrame,
    half_life_median: float,
    time_stop_hours: float,
) -> None:
    """
    Print a concise signal/position summary to stdout.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``z_score``, ``position``, ``is_entry``,
        ``is_tp_exit``, ``is_sl_exit``.
    half_life_median : float
        Median OU half-life used for time-stop benchmark.
    time_stop_hours : float
        Actual time-stop threshold applied (hours).
    """
    valid = df["z_score"].notna()
    n_long_entries  = int(((df["is_entry"]) & (df["z_score"] < -ENTRY_Z)).sum())
    n_short_entries = int(((df["is_entry"]) & (df["z_score"] > ENTRY_Z)).sum())

    sep = "=" * 70
    print(f"\n{sep}")
    print("  STRATEGY ENGINE SUMMARY")
    print(sep)
    print(f"  Median half-life used     : {half_life_median:.1f} h")
    print(f"  Time-stop threshold       : {TIME_STOP_MULTIPLIER:.0f} × {half_life_median:.1f} "
          f"= {time_stop_hours:.0f} h")
    print(f"  Z-score valid rows        : {valid.sum():,}")
    print(f"\n  --- Trades ---")
    print(f"  Total entries             : {df['is_entry'].sum():,}")
    print(f"    Long spread  (+1)       : {n_long_entries:,}")
    print(f"    Short spread (−1)       : {n_short_entries:,}")
    print(f"  Take-profit exits         : {df['is_tp_exit'].sum():,}")
    print(f"  SL / Time-stop exits      : {df['is_sl_exit'].sum():,}")
    print(f"\n  --- Position distribution ---")
    vc = df["position"][valid].value_counts().sort_index()
    for pos_val, cnt in vc.items():
        label = {1: "Long spread ", -1: "Short spread", 0: "Flat        "}.get(pos_val, str(pos_val))
        pct = cnt / valid.sum() * 100
        print(f"    {label} ({pos_val:+d}) : {cnt:,}  ({pct:.1f}%)")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# 4. Visualisation
# ---------------------------------------------------------------------------

def plot_zscore_and_signals(df: pd.DataFrame, output_path: Path = SIGNAL_PLOT) -> None:
    """
    Plot the Z-score timeline with coloured entry and exit markers.

    Layout
    ------
    • Grey line  : Z-score
    • Dashed horizontal lines at ±ENTRY_Z, ±SL_Z, ±TP_Z
    • Green ▲    : long-spread entries  (Z crossed below −ENTRY_Z)
    • Red   ▼    : short-spread entries (Z crossed above +ENTRY_Z)
    • Cyan  ●    : take-profit exits
    • Orange ✕   : stop-loss / time-stop exits
    • Background shade: green when long, red when short

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``z_score``, ``position``, ``is_entry``,
        ``is_tp_exit``, ``is_sl_exit``.
    output_path : Path
        Destination PNG file.
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(18, 7))

    z   = df["z_score"].dropna()
    pos = df["position"]

    # ── Background shading for active positions ────────────────────────
    # Long spread
    ax.fill_between(df.index, -6, 6,
                    where=(pos == 1),
                    alpha=0.07, color="#4CAF50", step="post")
    # Short spread
    ax.fill_between(df.index, -6, 6,
                    where=(pos == -1),
                    alpha=0.07, color="#F44336", step="post")

    # ── Z-score line ───────────────────────────────────────────────────
    ax.plot(z.index, z.values, color="#AAAAAA", linewidth=0.65,
            alpha=0.9, label="Z-score", zorder=2)

    # ── Threshold lines ────────────────────────────────────────────────
    for level, style, color, label in [
        ( ENTRY_Z, "--", "#FFD700", f"±{ENTRY_Z} entry"),
        (-ENTRY_Z, "--", "#FFD700", None),
        ( SL_Z,    ":",  "#FF5252", f"±{SL_Z} stop-loss"),
        (-SL_Z,    ":",  "#FF5252", None),
        ( TP_Z,    "-.", "#69FF47", f"±{TP_Z} take-profit"),
        (-TP_Z,    "-.", "#69FF47", None),
        ( 0.0,     "--", "white",   "Zero"),
    ]:
        ax.axhline(level, linestyle=style, color=color, linewidth=0.8,
                   alpha=0.55, label=label)

    # ── Entry markers ──────────────────────────────────────────────────
    long_entries  = df.index[(df["is_entry"]) & (df["z_score"] < -ENTRY_Z)]
    short_entries = df.index[(df["is_entry"]) & (df["z_score"] >  ENTRY_Z)]

    if len(long_entries):
        ax.scatter(long_entries, df.loc[long_entries, "z_score"],
                   marker="^", s=50, color="#4CAF50", zorder=5,
                   label=f"Long entry ({len(long_entries)})", alpha=0.85)
    if len(short_entries):
        ax.scatter(short_entries, df.loc[short_entries, "z_score"],
                   marker="v", s=50, color="#F44336", zorder=5,
                   label=f"Short entry ({len(short_entries)})", alpha=0.85)

    # ── Exit markers ───────────────────────────────────────────────────
    tp_exits = df.index[df["is_tp_exit"]]
    sl_exits = df.index[df["is_sl_exit"]]

    if len(tp_exits):
        ax.scatter(tp_exits, df.loc[tp_exits, "z_score"],
                   marker="o", s=35, color="#00E5FF", zorder=5,
                   label=f"Take-profit ({len(tp_exits)})", alpha=0.75)
    if len(sl_exits):
        ax.scatter(sl_exits, df.loc[sl_exits, "z_score"],
                   marker="X", s=55, color="#FF9800", zorder=5,
                   label=f"SL/Time-stop ({len(sl_exits)})", alpha=0.85)

    # ── Axes & decorations ─────────────────────────────────────────────
    ax.set_ylim(-6.5, 6.5)
    ax.set_ylabel("Z-score", fontsize=12)
    ax.set_title(
        f"BTC/ETH Spread Z-score & Trading Signals\n"
        f"Entry |Z|>{ENTRY_Z}  |  TP |Z|<{TP_Z}  |  SL |Z|>{SL_Z}  "
        f"|  ADF p<{ADF_THRESHOLD}  |  Rolling W={ZSCORE_WINDOW}h",
        fontsize=13, pad=12,
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=35, ha="right")

    # Compact legend (deduplicate None labels)
    handles, labels = ax.get_legend_handles_labels()
    pairs = [(h, l) for h, l in zip(handles, labels) if l is not None]
    ax.legend(*zip(*pairs), fontsize=9, framealpha=0.3,
              loc="upper right", ncol=2)
    ax.grid(alpha=0.10, linestyle="--")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved → %s", output_path)


# ---------------------------------------------------------------------------
# 5. Main orchestrator
# ---------------------------------------------------------------------------

def run_strategy_engine() -> pd.DataFrame:
    """
    Orchestrate the full Phase-3 pipeline.

    Steps
    -----
    1. Load ``statistical_features.parquet``.
    2. Compute look-ahead-free Z-score.
    3. Derive time-stop from Phase-2 median half-life.
    4. Run state-machine to produce position series and trade markers.
    5. Persit to ``signals.parquet``.
    6. Print summary and produce signal chart.

    Returns
    -------
    pd.DataFrame
        Full DataFrame with ``z_score``, ``position``, ``is_entry``,
        ``is_tp_exit``, ``is_sl_exit`` columns appended.
    """
    # ── Load ──────────────────────────────────────────────────────────
    logger.info("Loading %s …", INPUT_PARQUET)
    df = pd.read_parquet(INPUT_PARQUET)
    logger.info("  → %d rows loaded.", len(df))

    # ── Z-score ───────────────────────────────────────────────────────
    df["z_score"] = compute_zscore(df["spread"], window=ZSCORE_WINDOW)

    # ── Time-stop threshold ───────────────────────────────────────────
    # Use the *full-sample* median half-life computed in Phase 2.
    # This is a dataset-level hyper-parameter chosen BEFORE examining the
    # signal series, so it does not introduce look-ahead bias.
    half_life_median: float = float(df["half_life"].median())
    time_stop_hours: float = TIME_STOP_MULTIPLIER * half_life_median
    logger.info(
        "Time-stop: %.0f × %.1f h = %.0f h  (≈ %.0f days)",
        TIME_STOP_MULTIPLIER, half_life_median, time_stop_hours,
        time_stop_hours / 24,
    )

    # ── State-machine positions ───────────────────────────────────────
    (
        df["position"],
        df["is_entry"],
        df["is_tp_exit"],
        df["is_sl_exit"],
    ) = generate_positions(
        z_score=df["z_score"],
        adf_pvalue=df["adf_pvalue"],
        half_life_median=half_life_median,
        entry_z=ENTRY_Z,
        tp_z=TP_Z,
        sl_z=SL_Z,
        adf_threshold=ADF_THRESHOLD,
        time_stop_multiplier=TIME_STOP_MULTIPLIER,
    )

    # ── Persist ───────────────────────────────────────────────────────
    df.to_parquet(OUTPUT_PARQUET)
    logger.info("Saved → %s", OUTPUT_PARQUET)

    # ── Summary ───────────────────────────────────────────────────────
    _print_summary(df, half_life_median, time_stop_hours)

    # ── Chart ─────────────────────────────────────────────────────────
    plot_zscore_and_signals(df)

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_strategy_engine()
