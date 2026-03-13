"""
04_backtester.py
=================
Phase 4 – Realistic Backtesting & Performance Analysis for BTC/ETH Pairs Trading.

Strategy: Dollar-neutral pairs trade on the BTC/ETH spread.

Position sizing
---------------
Each trade deploys a fixed ``LEG_NOTIONAL`` USD per leg:
  • Long  spread (+1): Long  BTC ($LEG_NOTIONAL) + Short ETH ($LEG_NOTIONAL)
  • Short spread (–1): Short BTC ($LEG_NOTIONAL) + Long  ETH ($LEG_NOTIONAL)

Total gross exposure per trade ≈ 2 × LEG_NOTIONAL.

Cost model
----------
• Open  fees : FEE_RATE × LEG_NOTIONAL × 2 legs  (on entry bar)
• Close fees : FEE_RATE × LEG_NOTIONAL × 2 legs  (on exit bar)
• Funding    : applied only at 00:00, 08:00, 16:00 UTC (actual Binance settlement)
               rate × LEG_NOTIONAL, sign depends on leg direction

PnL is tracked hour-by-hour using mark-to-market price changes.
Look-ahead is NOT possible here: we enter/exit at the bar's own close,
which is AFTER the signal generated from the PREVIOUS bar's Z-score.

Outputs
-------
data/tearsheet.png  – 3-panel professional tearsheet
Terminal            – formatted metrics table
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
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
# Paths & parameters
# ---------------------------------------------------------------------------
DATA_DIR: Path = Path(__file__).parent.parent / "data"
INPUT_PARQUET: Path = DATA_DIR / "signals.parquet"
TEARSHEET_PATH: Path = DATA_DIR / "tearsheet.png"

INITIAL_CAPITAL: float = 10_000.0   # USD
LEG_NOTIONAL: float    = 5_000.0    # USD per leg (dollar-neutral: BTC leg = ETH leg)
FEE_RATE: float        = 0.0006     # 0.06% per leg per trade event (open or close)
RISK_FREE_RATE: float  = 0.0        # annualised, used for Sharpe calculation
FUNDING_HOURS: frozenset[int] = frozenset([0, 8, 16])   # UTC hours of settlement


# ---------------------------------------------------------------------------
# Data class to record each trade's attribution
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Container for one round-trip trade's PnL attribution."""
    trade_id: int
    direction: int        # +1 long spread, -1 short spread
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    exit_type: str        # 'take_profit' | 'stop_loss' | 'open_at_end'
    gross_pnl: float = 0.0
    total_fees: float = 0.0
    total_funding: float = 0.0

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.total_fees + self.total_funding

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0.0


# ---------------------------------------------------------------------------
# 1. Core simulation loop
# ---------------------------------------------------------------------------

def run_simulation(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[TradeRecord]]:
    """
    Simulate the strategy hour-by-hour, tracking equity mark-to-market.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``btc_close``, ``eth_close``,
        ``btc_funding_rate``, ``eth_funding_rate``,
        ``position``, ``is_entry``, ``is_tp_exit``, ``is_sl_exit``.

    Returns
    -------
    (net_equity, gross_equity, trade_records) : tuple
        ``net_equity``   – hourly equity curve including all costs (USD)
        ``gross_equity`` – hourly equity curve with fees stripped (USD)
        ``trade_records`` – one ``TradeRecord`` per completed/open trade
    """
    n = len(df)

    # Pre-extract numpy arrays for speed
    btc_close  = df["btc_close"].to_numpy(dtype=float)
    eth_close  = df["eth_close"].to_numpy(dtype=float)
    btc_fr     = df["btc_funding_rate"].to_numpy(dtype=float)
    eth_fr     = df["eth_funding_rate"].to_numpy(dtype=float)
    pos        = df["position"].to_numpy(dtype=np.int8)
    is_entry   = df["is_entry"].to_numpy(dtype=bool)
    is_tp      = df["is_tp_exit"].to_numpy(dtype=bool)
    is_sl      = df["is_sl_exit"].to_numpy(dtype=bool)
    idx        = df.index

    net_equity   = np.full(n, np.nan)
    gross_equity = np.full(n, np.nan)
    net_equity[0]   = INITIAL_CAPITAL
    gross_equity[0] = INITIAL_CAPITAL

    # Position state
    btc_units: float = 0.0
    eth_units: float = 0.0
    current_trade: Optional[TradeRecord] = None
    trade_records: list[TradeRecord] = []
    trade_counter: int = 0

    for i in range(1, n):
        net_i   = net_equity[i - 1]
        gross_i = gross_equity[i - 1]
        hour    = idx[i].hour

        # ── (A) Mark-to-market for any position held at the PREVIOUS bar ──
        # We held pos[i-1] from close[i-1] to close[i].
        prev_pos = pos[i - 1]
        if prev_pos == 1:          # long BTC, short ETH
            m2m = ((btc_close[i] - btc_close[i - 1]) * btc_units
                   - (eth_close[i] - eth_close[i - 1]) * eth_units)
            net_i   += m2m
            gross_i += m2m
            if current_trade is not None:
                current_trade.gross_pnl += m2m

        elif prev_pos == -1:       # short BTC, long ETH
            m2m = (-(btc_close[i] - btc_close[i - 1]) * btc_units
                   + (eth_close[i] - eth_close[i - 1]) * eth_units)
            net_i   += m2m
            gross_i += m2m
            if current_trade is not None:
                current_trade.gross_pnl += m2m

        # ── (B) Funding at settlement hours ───────────────────────────────
        # Apply when we ARE in a position at bar i (covers the settlement at
        # the open of hour i, which is after the previous bar's close).
        curr_pos = pos[i]
        if curr_pos != 0 and hour in FUNDING_HOURS:
            # Long spread (+1): Long BTC (pay BTC funding), Short ETH (receive ETH funding)
            # Short spread (−1): Short BTC (receive BTC funding), Long ETH (pay ETH funding)
            # Binance convention: positive funding → longs PAY shorts.
            sign = float(curr_pos)   # +1 or -1
            btc_fund = -sign * btc_fr[i] * LEG_NOTIONAL
            eth_fund = +sign * eth_fr[i] * LEG_NOTIONAL
            funding  = btc_fund + eth_fund
            net_i   += funding
            gross_i += funding        # funding is a real cash flow, kept in gross too
            if current_trade is not None:
                current_trade.total_funding += funding

        # ── (C) Entry fees ────────────────────────────────────────────────
        if is_entry[i]:
            open_fee = FEE_RATE * LEG_NOTIONAL * 2   # both legs
            net_i   -= open_fee
            # gross curve does NOT deduct fees
            btc_units = LEG_NOTIONAL / btc_close[i]
            eth_units = LEG_NOTIONAL / eth_close[i]
            trade_counter += 1
            current_trade = TradeRecord(
                trade_id=trade_counter,
                direction=int(curr_pos),
                entry_time=idx[i],
                exit_time=None,
                exit_type="open_at_end",
                total_fees=open_fee,
            )
            logger.debug("  Trade %d opened at %s  dir=%+d", trade_counter, idx[i], curr_pos)

        # ── (D) Exit fees ──────────────────────────────────────────────────
        if is_tp[i] or is_sl[i]:
            close_fee = FEE_RATE * LEG_NOTIONAL * 2
            net_i    -= close_fee
            if current_trade is not None:
                current_trade.total_fees  += close_fee
                current_trade.exit_time    = idx[i]
                current_trade.exit_type    = "take_profit" if is_tp[i] else "stop_loss"
                trade_records.append(current_trade)
                logger.debug(
                    "  Trade %d closed at %s  net_pnl=%.2f",
                    current_trade.trade_id, idx[i], current_trade.net_pnl,
                )
                current_trade = None
            btc_units = 0.0
            eth_units = 0.0

        net_equity[i]   = net_i
        gross_equity[i] = gross_i

    # Handle position still open at the end
    if current_trade is not None:
        current_trade.exit_time = idx[-1]
        current_trade.exit_type = "open_at_end"
        trade_records.append(current_trade)
        logger.info("  One trade still open at end of data – marked as 'open_at_end'.")

    logger.info("Simulation complete: %d round-trips recorded.", len(trade_records))
    return net_equity, gross_equity, trade_records


# ---------------------------------------------------------------------------
# 2. Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    net_equity: np.ndarray,
    df: pd.DataFrame,
    trade_records: list[TradeRecord],
) -> dict[str, float | int | str]:
    """
    Compute strategy-level performance metrics from the equity curve.

    Parameters
    ----------
    net_equity : np.ndarray
        Hourly net equity curve in USD.
    df : pd.DataFrame
        Original DataFrame providing the datetime index.
    trade_records : list[TradeRecord]
        Per-trade attribution records.

    Returns
    -------
    dict
        Mapping of metric names to values.
    """
    equity = pd.Series(net_equity, index=df.index)
    equity_clean = equity.dropna()

    # Hourly returns
    hourly_returns = equity_clean.pct_change().dropna()

    # CAGR
    n_hours = len(equity_clean)
    years   = n_hours / (365.25 * 24)
    cagr    = (equity_clean.iloc[-1] / equity_clean.iloc[0]) ** (1 / years) - 1

    # Max drawdown
    running_max = equity_clean.cummax()
    drawdown    = (equity_clean - running_max) / running_max
    max_dd      = float(drawdown.min())

    # Sharpe ratio (annualised, risk-free = 0)
    ann_factor = np.sqrt(365.25 * 24)
    sharpe = (float(hourly_returns.mean()) * ann_factor * (365.25 * 24)
              / (float(hourly_returns.std()) * ann_factor + 1e-12))
    # Simpler and standard:
    sharpe = float(hourly_returns.mean()) / (float(hourly_returns.std()) + 1e-12) * ann_factor

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    # Trade stats
    closed_trades  = [t for t in trade_records if t.exit_type != "open_at_end"]
    total_trades   = len(closed_trades)
    winning_trades = sum(1 for t in closed_trades if t.is_winner)
    win_rate       = winning_trades / total_trades if total_trades > 0 else 0.0
    avg_net_pnl    = np.mean([t.net_pnl for t in closed_trades]) if closed_trades else 0.0
    avg_gross_pnl  = np.mean([t.gross_pnl for t in closed_trades]) if closed_trades else 0.0
    avg_hold_hours = (
        np.mean([(t.exit_time - t.entry_time).total_seconds() / 3600
                 for t in closed_trades])
        if closed_trades else 0.0
    )

    # PnL attribution totals
    total_gross   = sum(t.gross_pnl  for t in trade_records)
    total_fees    = sum(t.total_fees for t in trade_records)
    total_funding = sum(t.total_funding for t in trade_records)
    total_net     = total_gross - total_fees + total_funding

    return {
        "Initial Capital (USD)"    : INITIAL_CAPITAL,
        "Final Equity (USD)"       : float(equity_clean.iloc[-1]),
        "Total Net PnL (USD)"      : float(equity_clean.iloc[-1]) - INITIAL_CAPITAL,
        "CAGR (%)"                 : cagr * 100,
        "Max Drawdown (%)"         : max_dd * 100,
        "Sharpe Ratio"             : sharpe,
        "Calmar Ratio"             : calmar,
        "Total Closed Trades"      : total_trades,
        "Win Rate (%)"             : win_rate * 100,
        "Avg Net PnL / Trade (USD)": avg_net_pnl,
        "Avg Gross PnL / Trade"    : avg_gross_pnl,
        "Avg Holding (hours)"      : avg_hold_hours,
        "--- Attribution ---"      : "---",
        "Gross PnL (USD)"          : total_gross,
        "Total Fees (USD)"         : -total_fees,
        "Funding & Carry (USD)"    : total_funding,
        "Net PnL (USD)"            : total_net,
    }


# ---------------------------------------------------------------------------
# 3. Print metrics table
# ---------------------------------------------------------------------------

def print_metrics(metrics: dict) -> None:
    """
    Print a formatted performance metrics table to stdout.

    Parameters
    ----------
    metrics : dict
        Output of ``compute_metrics``.
    """
    sep = "=" * 70
    print(f"\n{sep}")
    print("  BACKTEST PERFORMANCE REPORT")
    print(sep)
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:<36}: {val:>12.4f}")
        elif isinstance(val, int):
            print(f"  {key:<36}: {val:>12d}")
        else:
            print(f"  {key}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# 4. Tearsheet
# ---------------------------------------------------------------------------

def plot_tearsheet(
    net_equity: np.ndarray,
    gross_equity: np.ndarray,
    df: pd.DataFrame,
    trade_records: list[TradeRecord],
    metrics: dict,
    output_path: Path = TEARSHEET_PATH,
) -> None:
    """
    Generate a 3-panel professional tearsheet and save to disk.

    Panels (top → bottom)
    ----------------------
    1. Equity Curve   – Gross vs Net with entry/exit annotations
    2. Drawdown Chart – Net underwater plot
    3. PnL Attribution – Per-trade grouped bar chart

    Parameters
    ----------
    net_equity : np.ndarray
        Hourly net equity (USD).
    gross_equity : np.ndarray
        Hourly gross equity (fees excluded) (USD).
    df : pd.DataFrame
        Full signals DataFrame providing the datetime index.
    trade_records : list[TradeRecord]
        Per-trade attribution container.
    metrics : dict
        Output of ``compute_metrics`` for annotation.
    output_path : Path
        Destination PNG.
    """
    plt.style.use("dark_background")

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0D0D0D")

    gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[3, 1.5, 2.5],
        hspace=0.42,
    )
    ax_eq  = fig.add_subplot(gs[0])
    ax_dd  = fig.add_subplot(gs[1], sharex=ax_eq)
    ax_att = fig.add_subplot(gs[2])

    idx   = df.index
    eq_s  = pd.Series(net_equity,   index=idx).dropna()
    gr_s  = pd.Series(gross_equity, index=idx).dropna()
    common_idx = eq_s.index.intersection(gr_s.index)
    eq_s  = eq_s.loc[common_idx]
    gr_s  = gr_s.loc[common_idx]

    # ── Panel 1: Equity Curve ─────────────────────────────────────────────
    ax_eq.plot(eq_s.index, eq_s.values, color="#00E5FF", linewidth=1.1,
               label="Net equity (w/ fees & funding)", zorder=3)
    ax_eq.plot(gr_s.index, gr_s.values, color="#F7CF47", linewidth=0.8,
               linestyle="--", alpha=0.8, label="Gross equity (no fees)", zorder=2)
    ax_eq.axhline(INITIAL_CAPITAL, color="white", linewidth=0.6,
                  linestyle=":", alpha=0.4, label=f"Initial capital ${INITIAL_CAPITAL:,.0f}")

    # Shade between gross and net to highlight cost drag
    ax_eq.fill_between(common_idx,
                       gr_s.values, eq_s.values,
                       where=(gr_s.values > eq_s.values),
                       alpha=0.18, color="#F44336", label="Fee / funding drag")
    ax_eq.fill_between(common_idx,
                       gr_s.values, eq_s.values,
                       where=(gr_s.values <= eq_s.values),
                       alpha=0.18, color="#4CAF50")

    # Annotate entry/exit on equity curve
    for tr in trade_records:
        if tr.entry_time in eq_s.index:
            ax_eq.axvline(tr.entry_time, color="#4CAF50", linewidth=0.6,
                          alpha=0.5, linestyle="--")
        if tr.exit_time and tr.exit_time in eq_s.index:
            color = "#00E5FF" if tr.exit_type == "take_profit" else "#FF9800"
            ax_eq.axvline(tr.exit_time, color=color, linewidth=0.6,
                          alpha=0.5, linestyle=":")

    cagr_val  = metrics.get("CAGR (%)", 0)
    sharpe_val = metrics.get("Sharpe Ratio", 0)
    maxdd_val = metrics.get("Max Drawdown (%)", 0)
    final_eq  = metrics.get("Final Equity (USD)", INITIAL_CAPITAL)
    ax_eq.set_title(
        f"BTC/ETH Pairs Trading — Equity Curve\n"
        f"CAGR: {cagr_val:.2f}%  |  Sharpe: {sharpe_val:.3f}  |  "
        f"Max DD: {maxdd_val:.2f}%  |  Final: ${final_eq:,.2f}",
        fontsize=13, pad=10, color="white",
    )
    ax_eq.set_ylabel("Portfolio Value (USD)", fontsize=11)
    ax_eq.legend(fontsize=9, framealpha=0.25, loc="upper left")
    ax_eq.grid(alpha=0.10, linestyle="--")
    ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────
    running_peak = eq_s.cummax()
    drawdown     = (eq_s - running_peak) / running_peak * 100

    ax_dd.fill_between(drawdown.index, drawdown.values, 0,
                       color="#F44336", alpha=0.65, label="Net drawdown")
    ax_dd.plot(drawdown.index, drawdown.values, color="#FF5252",
               linewidth=0.7, alpha=0.9)
    ax_dd.axhline(0, color="white", linewidth=0.4, alpha=0.3)
    ax_dd.set_ylabel("Drawdown (%)", fontsize=10)
    ax_dd.set_title("Underwater Equity (Drawdown)", fontsize=11, pad=6)
    ax_dd.legend(fontsize=9, framealpha=0.25, loc="lower left")
    ax_dd.grid(alpha=0.10, linestyle="--")
    ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    # Shared x-axis formatting
    for ax in [ax_eq, ax_dd]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax_eq.xaxis.get_majorticklabels(), visible=False)
    fig.autofmt_xdate(rotation=30, ha="right")

    # ── Panel 3: PnL Attribution per Trade ────────────────────────────────
    closed = [t for t in trade_records if t.exit_type != "open_at_end"]
    if closed:
        trade_ids    = [f"T{t.trade_id}\n({'+' if t.direction > 0 else '−'}Spread)" for t in closed]
        gross_vals   = [t.gross_pnl for t in closed]
        fee_vals     = [-t.total_fees for t in closed]           # shown as negative
        funding_vals = [t.total_funding for t in closed]
        net_vals     = [t.net_pnl for t in closed]

        x = np.arange(len(closed))
        w = 0.18

        bars_g = ax_att.bar(x - 1.5 * w, gross_vals,   width=w, color="#F7CF47",
                            alpha=0.85, label="Gross PnL")
        bars_f = ax_att.bar(x - 0.5 * w, fee_vals,     width=w, color="#F44336",
                            alpha=0.85, label="Fees (−)")
        bars_u = ax_att.bar(x + 0.5 * w, funding_vals, width=w, color="#9C27B0",
                            alpha=0.85, label="Funding ± carry")
        bars_n = ax_att.bar(x + 1.5 * w, net_vals,     width=w,
                            color=["#4CAF50" if v > 0 else "#F44336" for v in net_vals],
                            alpha=0.95, label="Net PnL")

        # Value labels
        for bar_grp in [bars_g, bars_f, bars_u, bars_n]:
            for bar in bar_grp:
                h = bar.get_height()
                if abs(h) > 1:
                    ax_att.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + (2 if h >= 0 else -8),
                        f"${h:.0f}",
                        ha="center", va="bottom" if h >= 0 else "top",
                        fontsize=7, color="white", alpha=0.85,
                    )

        ax_att.axhline(0, color="white", linewidth=0.5, alpha=0.4)
        ax_att.set_xticks(x)
        ax_att.set_xticklabels(trade_ids, fontsize=9)
        ax_att.set_ylabel("PnL (USD)", fontsize=10)
        ax_att.set_title(
            "PnL Attribution per Trade  "
            f"(LEG={LEG_NOTIONAL:,.0f} USD/leg  |  Fee={FEE_RATE*100:.2f}%/leg/event)",
            fontsize=11, pad=6,
        )
        ax_att.legend(fontsize=9, framealpha=0.25, loc="upper right")
        ax_att.grid(alpha=0.08, linestyle="--", axis="y")
        ax_att.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    else:
        ax_att.text(0.5, 0.5, "No closed trades to display",
                    ha="center", va="center", color="white", fontsize=14,
                    transform=ax_att.transAxes)

    fig.suptitle(
        "BTC/ETH Cointegration Pairs Trading — Full Tearsheet\n"
        f"Period: {idx[0].date()} → {idx[-1].date()}  |  "
        f"Capital: ${INITIAL_CAPITAL:,.0f}  |  Legs: ${LEG_NOTIONAL:,.0f}/side",
        fontsize=14, y=0.995, color="white",
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Tearsheet saved → %s", output_path)


# ---------------------------------------------------------------------------
# 5. Trade-level detail table
# ---------------------------------------------------------------------------

def print_trade_log(trade_records: list[TradeRecord]) -> None:
    """
    Print a full per-trade log table to stdout.

    Parameters
    ----------
    trade_records : list[TradeRecord]
        Output of ``run_simulation``.
    """
    sep = "=" * 90
    print(f"\n{sep}")
    print("  TRADE-LEVEL LOG")
    print(sep)
    header = (
        f"  {'ID':>3}  {'Dir':>6}  {'Entry':>20}  {'Exit':>20}  "
        f"{'Type':>12}  {'Gross':>8}  {'Fees':>8}  {'Fund':>8}  {'Net':>8}"
    )
    print(header)
    print("-" * 90)
    for t in trade_records:
        dir_str  = "+Spread" if t.direction == 1 else "−Spread"
        exit_str = str(t.exit_time)[:19] if t.exit_time else "N/A"
        print(
            f"  {t.trade_id:>3}  {dir_str:>7}  "
            f"{str(t.entry_time)[:19]:>20}  {exit_str:>20}  "
            f"{t.exit_type:>12}  "
            f"{t.gross_pnl:>8.2f}  {-t.total_fees:>8.2f}  "
            f"{t.total_funding:>8.4f}  {t.net_pnl:>8.2f}"
        )
    print(sep + "\n")


# ---------------------------------------------------------------------------
# 6. Main orchestrator
# ---------------------------------------------------------------------------

def run_backtest() -> None:
    """
    Orchestrate the full Phase-4 backtest pipeline.

    Steps
    -----
    1. Load signals parquet.
    2. Run hour-by-hour simulation.
    3. Compute performance metrics.
    4. Print metrics + trade log.
    5. Generate and save tearsheet.
    """
    # ── Load ──────────────────────────────────────────────────────────────
    logger.info("Loading %s …", INPUT_PARQUET)
    df = pd.read_parquet(INPUT_PARQUET)
    logger.info("  → %d rows loaded.", len(df))

    # ── Simulation ────────────────────────────────────────────────────────
    logger.info(
        "Running simulation  [capital=$%,.0f  leg=$%,.0f  fee=%.2f%%] …",
        INITIAL_CAPITAL, LEG_NOTIONAL, FEE_RATE * 100,
    )
    net_equity, gross_equity, trade_records = run_simulation(df)

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics = compute_metrics(net_equity, df, trade_records)
    print_metrics(metrics)
    print_trade_log(trade_records)

    # ── Tearsheet ─────────────────────────────────────────────────────────
    plot_tearsheet(net_equity, gross_equity, df, trade_records, metrics)

    logger.info("Phase 4 complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_backtest()
