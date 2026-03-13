"""
04b_backtester_beta.py
=======================
Phase 4b – Beta-Neutral Backtest (control experiment).

Identical to ``04_backtester.py`` in structure and cost model, but uses
**Beta-Neutral** position sizing instead of Dollar-Neutral:

    BTC leg notional : fixed  $BTC_NOTIONAL  (anchor leg)
    ETH leg notional : dynamic $BTC_NOTIONAL × rolling_beta_at_entry

This ensures the two legs are economically matched through the rolling hedge
ratio β, and the spread approximately dollar-hedges the cointegration
relationship at its current estimated loading.

Key differences vs 04_backtester.py
-------------------------------------
• ``eth_notional`` is computed at entry time as ``BTC_NOTIONAL × β_entry``.
• Fees scale with actual notional per leg (not hardcoded $5k each).
• Funding settlement scales with actual leg notionals.
• Output: ``data/tearsheet_beta_neutral.png``  (original tearsheet untouched).

The original ``data/tearsheet.png`` and ``04_backtester.py`` are never
modified or referenced here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
TEARSHEET_PATH: Path = DATA_DIR / "tearsheet_beta_neutral.png"     # NEW path

INITIAL_CAPITAL: float = 10_000.0
BTC_NOTIONAL: float    = 5_000.0    # anchor leg; ETH scales with β
FEE_RATE: float        = 0.0006     # 0.06% per leg per trade event
RISK_FREE_RATE: float  = 0.0
FUNDING_HOURS: frozenset[int] = frozenset([0, 8, 16])


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Per-trade PnL attribution container."""
    trade_id: int
    direction: int
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    exit_type: str
    beta_at_entry: float = 0.0
    eth_notional_at_entry: float = 0.0
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
# 1. Core simulation loop (Beta-Neutral)
# ---------------------------------------------------------------------------

def run_simulation(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[TradeRecord]]:
    """
    Hour-by-hour mark-to-market simulation with Beta-Neutral position sizing.

    At each entry, the ETH notional is set to:
        eth_notional = BTC_NOTIONAL × rolling_beta[entry_bar]

    All fees and funding settlements use per-leg actual notionals.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``btc_close``, ``eth_close``, ``btc_funding_rate``,
        ``eth_funding_rate``, ``rolling_beta``, ``position``,
        ``is_entry``, ``is_tp_exit``, ``is_sl_exit``.

    Returns
    -------
    (net_equity, gross_equity, trade_records)
    """
    n = len(df)

    btc_close  = df["btc_close"].to_numpy(dtype=float)
    eth_close  = df["eth_close"].to_numpy(dtype=float)
    btc_fr     = df["btc_funding_rate"].to_numpy(dtype=float)
    eth_fr     = df["eth_funding_rate"].to_numpy(dtype=float)
    rolling_b  = df["rolling_beta"].to_numpy(dtype=float)
    pos        = df["position"].to_numpy(dtype=np.int8)
    is_entry   = df["is_entry"].to_numpy(dtype=bool)
    is_tp      = df["is_tp_exit"].to_numpy(dtype=bool)
    is_sl      = df["is_sl_exit"].to_numpy(dtype=bool)
    idx        = df.index

    net_equity   = np.full(n, np.nan)
    gross_equity = np.full(n, np.nan)
    net_equity[0]   = INITIAL_CAPITAL
    gross_equity[0] = INITIAL_CAPITAL

    # Live trade state
    btc_units: float         = 0.0
    eth_units: float         = 0.0
    btc_notional_live: float = 0.0
    eth_notional_live: float = 0.0
    current_trade: Optional[TradeRecord] = None
    trade_records: list[TradeRecord] = []
    trade_counter: int = 0

    for i in range(1, n):
        net_i   = net_equity[i - 1]
        gross_i = gross_equity[i - 1]
        hour    = idx[i].hour

        # ── (A) M2M for position held at previous bar ─────────────────────
        prev_pos = pos[i - 1]
        if prev_pos == 1:           # long BTC, short ETH
            m2m = ((btc_close[i] - btc_close[i - 1]) * btc_units
                   - (eth_close[i] - eth_close[i - 1]) * eth_units)
            net_i   += m2m
            gross_i += m2m
            if current_trade is not None:
                current_trade.gross_pnl += m2m

        elif prev_pos == -1:        # short BTC, long ETH
            m2m = (-(btc_close[i] - btc_close[i - 1]) * btc_units
                   + (eth_close[i] - eth_close[i - 1]) * eth_units)
            net_i   += m2m
            gross_i += m2m
            if current_trade is not None:
                current_trade.gross_pnl += m2m

        # ── (B) Funding at settlement hours ───────────────────────────────
        curr_pos = pos[i]
        if curr_pos != 0 and hour in FUNDING_HOURS:
            sign = float(curr_pos)
            # Long spread (+1): pay BTC funding, receive ETH funding
            btc_fund = -sign * btc_fr[i] * btc_notional_live
            eth_fund = +sign * eth_fr[i] * eth_notional_live
            funding  = btc_fund + eth_fund
            net_i   += funding
            gross_i += funding
            if current_trade is not None:
                current_trade.total_funding += funding

        # ── (C) Entry: compute Beta-Neutral sizes ─────────────────────────
        if is_entry[i]:
            beta_entry = float(rolling_b[i])
            if np.isnan(beta_entry) or beta_entry <= 0:
                # Safety: fall back to dollar-neutral if β invalid
                logger.warning("  Invalid β=%.4f at entry bar %d; using β=1.0", beta_entry, i)
                beta_entry = 1.0

            btc_notional_live = BTC_NOTIONAL
            eth_notional_live = BTC_NOTIONAL * beta_entry   # ← key change

            btc_units = btc_notional_live / btc_close[i]
            eth_units = eth_notional_live / eth_close[i]

            open_fee = FEE_RATE * (btc_notional_live + eth_notional_live)
            net_i   -= open_fee

            trade_counter += 1
            current_trade = TradeRecord(
                trade_id=trade_counter,
                direction=int(curr_pos),
                entry_time=idx[i],
                exit_time=None,
                exit_type="open_at_end",
                beta_at_entry=beta_entry,
                eth_notional_at_entry=eth_notional_live,
                total_fees=open_fee,
            )
            logger.debug(
                "  Trade %d: dir=%+d  β=%.4f  BTC_N=$%.0f  ETH_N=$%.0f",
                trade_counter, curr_pos, beta_entry,
                btc_notional_live, eth_notional_live,
            )

        # ── (D) Exit ──────────────────────────────────────────────────────
        if is_tp[i] or is_sl[i]:
            close_fee = FEE_RATE * (btc_notional_live + eth_notional_live)
            net_i    -= close_fee
            if current_trade is not None:
                current_trade.total_fees  += close_fee
                current_trade.exit_time    = idx[i]
                current_trade.exit_type    = "take_profit" if is_tp[i] else "stop_loss"
                trade_records.append(current_trade)
                logger.debug(
                    "  Trade %d closed: net_pnl=%.2f",
                    current_trade.trade_id, current_trade.net_pnl,
                )
                current_trade = None
            btc_units         = 0.0
            eth_units         = 0.0
            btc_notional_live = 0.0
            eth_notional_live = 0.0

        net_equity[i]   = net_i
        gross_equity[i] = gross_i

    # Handle still-open position at end of data
    if current_trade is not None:
        current_trade.exit_time = idx[-1]
        current_trade.exit_type = "open_at_end"
        trade_records.append(current_trade)

    logger.info("Simulation complete: %d round-trips recorded.", len(trade_records))
    return net_equity, gross_equity, trade_records


# ---------------------------------------------------------------------------
# 2. Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    net_equity: np.ndarray,
    df: pd.DataFrame,
    trade_records: list[TradeRecord],
) -> dict:
    """Compute strategy-level performance metrics."""
    equity = pd.Series(net_equity, index=df.index).dropna()
    hourly_returns = equity.pct_change().dropna()

    n_hours = len(equity)
    years   = n_hours / (365.25 * 24)
    cagr    = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1

    running_max = equity.cummax()
    drawdown    = (equity - running_max) / running_max
    max_dd      = float(drawdown.min())

    ann_factor = np.sqrt(365.25 * 24)
    sharpe = float(hourly_returns.mean()) / (float(hourly_returns.std()) + 1e-12) * ann_factor

    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    closed = [t for t in trade_records if t.exit_type != "open_at_end"]
    total_trades   = len(closed)
    win_rate       = sum(1 for t in closed if t.is_winner) / max(total_trades, 1)
    avg_net_pnl    = np.mean([t.net_pnl for t in closed]) if closed else 0.0
    avg_hold_hours = (
        np.mean([(t.exit_time - t.entry_time).total_seconds() / 3600 for t in closed])
        if closed else 0.0
    )

    total_gross   = sum(t.gross_pnl  for t in trade_records)
    total_fees    = sum(t.total_fees for t in trade_records)
    total_funding = sum(t.total_funding for t in trade_records)

    return {
        "Initial Capital (USD)"     : INITIAL_CAPITAL,
        "Final Equity (USD)"        : float(equity.iloc[-1]),
        "Total Net PnL (USD)"       : float(equity.iloc[-1]) - INITIAL_CAPITAL,
        "CAGR (%)"                  : cagr * 100,
        "Max Drawdown (%)"          : max_dd * 100,
        "Sharpe Ratio"              : sharpe,
        "Calmar Ratio"              : calmar,
        "Total Closed Trades"       : total_trades,
        "Win Rate (%)"              : win_rate * 100,
        "Avg Net PnL / Trade (USD)" : avg_net_pnl,
        "Avg Holding (hours)"       : avg_hold_hours,
        "--- Attribution ---"       : "---",
        "Gross PnL (USD)"           : total_gross,
        "Total Fees (USD)"          : -total_fees,
        "Funding & Carry (USD)"     : total_funding,
        "Net PnL (USD)"             : total_gross - total_fees + total_funding,
    }


# ---------------------------------------------------------------------------
# 3. Print helpers
# ---------------------------------------------------------------------------

def print_metrics(metrics: dict, label: str = "BETA-NEUTRAL") -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  BACKTEST PERFORMANCE REPORT  [{label}]")
    print(sep)
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:<36}: {val:>12.4f}")
        elif isinstance(val, int):
            print(f"  {key:<36}: {val:>12d}")
        else:
            print(f"  {key}")
    print(sep + "\n")


def print_trade_log(trade_records: list[TradeRecord]) -> None:
    sep = "=" * 105
    print(f"\n{sep}")
    print("  TRADE-LEVEL LOG  [Beta-Neutral]")
    print(sep)
    header = (
        f"  {'ID':>3}  {'Dir':>7}  {'β':>6}  {'ETH_N':>8}  "
        f"{'Entry':>20}  {'Exit':>20}  {'Type':>12}  "
        f"{'Gross':>8}  {'Fees':>8}  {'Fund':>8}  {'Net':>8}"
    )
    print(header)
    print("-" * 105)
    for t in trade_records:
        dir_str  = "+Spread" if t.direction == 1 else "−Spread"
        exit_str = str(t.exit_time)[:19] if t.exit_time else "N/A"
        print(
            f"  {t.trade_id:>3}  {dir_str:>7}  "
            f"{t.beta_at_entry:>6.3f}  ${t.eth_notional_at_entry:>7.0f}  "
            f"{str(t.entry_time)[:19]:>20}  {exit_str:>20}  "
            f"{t.exit_type:>12}  "
            f"{t.gross_pnl:>8.2f}  {-t.total_fees:>8.2f}  "
            f"{t.total_funding:>8.4f}  {t.net_pnl:>8.2f}"
        )
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
    """3-panel tearsheet: equity curve / drawdown / per-trade attribution."""
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0D0D0D")

    gs = gridspec.GridSpec(3, 1, figure=fig,
                           height_ratios=[3, 1.5, 2.5], hspace=0.42)
    ax_eq  = fig.add_subplot(gs[0])
    ax_dd  = fig.add_subplot(gs[1], sharex=ax_eq)
    ax_att = fig.add_subplot(gs[2])

    idx   = df.index
    eq_s  = pd.Series(net_equity,   index=idx).dropna()
    gr_s  = pd.Series(gross_equity, index=idx).dropna()
    common = eq_s.index.intersection(gr_s.index)
    eq_s, gr_s = eq_s.loc[common], gr_s.loc[common]

    # ── Panel 1: Equity curve ─────────────────────────────────────────────
    ax_eq.plot(eq_s.index, eq_s.values, color="#00E5FF", linewidth=1.1,
               label="Net equity (w/ fees & funding)", zorder=3)
    ax_eq.plot(gr_s.index, gr_s.values, color="#F7CF47", linewidth=0.8,
               linestyle="--", alpha=0.8, label="Gross equity (no fees)", zorder=2)
    ax_eq.axhline(INITIAL_CAPITAL, color="white", linewidth=0.6, linestyle=":",
                  alpha=0.4, label=f"Initial capital ${INITIAL_CAPITAL:,.0f}")
    ax_eq.fill_between(common, gr_s.values, eq_s.values,
                       where=(gr_s.values > eq_s.values),
                       alpha=0.18, color="#F44336", label="Fee/funding drag")
    ax_eq.fill_between(common, gr_s.values, eq_s.values,
                       where=(gr_s.values <= eq_s.values),
                       alpha=0.18, color="#4CAF50")

    for tr in trade_records:
        if tr.entry_time in eq_s.index:
            ax_eq.axvline(tr.entry_time, color="#4CAF50", lw=0.6, alpha=0.5, ls="--")
        if tr.exit_time and tr.exit_time in eq_s.index:
            c = "#00E5FF" if tr.exit_type == "take_profit" else "#FF9800"
            ax_eq.axvline(tr.exit_time, color=c, lw=0.6, alpha=0.5, ls=":")

    ax_eq.set_title(
        f"BTC/ETH Pairs Trading — Beta-Neutral Equity Curve\n"
        f"CAGR: {metrics['CAGR (%)']:.2f}%  |  Sharpe: {metrics['Sharpe Ratio']:.3f}  |  "
        f"Max DD: {metrics['Max Drawdown (%)']:.2f}%  |  "
        f"Final: ${metrics['Final Equity (USD)']:,.2f}",
        fontsize=13, pad=10, color="white",
    )
    ax_eq.set_ylabel("Portfolio Value (USD)", fontsize=11)
    ax_eq.legend(fontsize=9, framealpha=0.25, loc="upper left")
    ax_eq.grid(alpha=0.10, linestyle="--")
    ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────
    peak = eq_s.cummax()
    dd   = (eq_s - peak) / peak * 100
    ax_dd.fill_between(dd.index, dd.values, 0, color="#F44336", alpha=0.65)
    ax_dd.plot(dd.index, dd.values, color="#FF5252", linewidth=0.7, alpha=0.9)
    ax_dd.axhline(0, color="white", linewidth=0.4, alpha=0.3)
    ax_dd.set_ylabel("Drawdown (%)", fontsize=10)
    ax_dd.set_title("Underwater Equity (Drawdown)", fontsize=11, pad=6)
    ax_dd.grid(alpha=0.10, linestyle="--")
    ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    for ax in [ax_eq, ax_dd]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax_eq.xaxis.get_majorticklabels(), visible=False)
    fig.autofmt_xdate(rotation=30, ha="right")

    # ── Panel 3: PnL Attribution per trade ────────────────────────────────
    closed = [t for t in trade_records if t.exit_type != "open_at_end"]
    if closed:
        trade_ids    = [
            f"T{t.trade_id}\n({'+' if t.direction > 0 else '−'}Spread)\nβ={t.beta_at_entry:.2f}"
            for t in closed
        ]
        gross_vals   = [t.gross_pnl for t in closed]
        fee_vals     = [-t.total_fees for t in closed]
        funding_vals = [t.total_funding for t in closed]
        net_vals     = [t.net_pnl for t in closed]

        x = np.arange(len(closed))
        w = 0.18
        bars_g = ax_att.bar(x - 1.5 * w, gross_vals,   width=w, color="#F7CF47", alpha=0.85, label="Gross PnL")
        bars_f = ax_att.bar(x - 0.5 * w, fee_vals,     width=w, color="#F44336", alpha=0.85, label="Fees (−)")
        bars_u = ax_att.bar(x + 0.5 * w, funding_vals, width=w, color="#9C27B0", alpha=0.85, label="Funding ± carry")
        bars_n = ax_att.bar(x + 1.5 * w, net_vals, width=w,
                            color=["#4CAF50" if v > 0 else "#F44336" for v in net_vals],
                            alpha=0.95, label="Net PnL")

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
        ax_att.set_xticklabels(trade_ids, fontsize=8)
        ax_att.set_ylabel("PnL (USD)", fontsize=10)
        ax_att.set_title(
            f"PnL Attribution per Trade  [Beta-Neutral]  "
            f"(BTC leg fixed=$5,000  |  ETH leg = $5,000 × β  |  Fee={FEE_RATE*100:.2f}%/leg)",
            fontsize=11, pad=6,
        )
        ax_att.legend(fontsize=9, framealpha=0.25, loc="upper right")
        ax_att.grid(alpha=0.08, linestyle="--", axis="y")
        ax_att.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    fig.suptitle(
        "BTC/ETH Cointegration Pairs Trading — Beta-Neutral Tearsheet\n"
        f"Period: {idx[0].date()} → {idx[-1].date()}  |  "
        f"BTC anchor: ${BTC_NOTIONAL:,.0f}  |  ETH notional: $BTC × rolling β",
        fontsize=14, y=0.995, color="white",
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Tearsheet saved → %s", output_path)


# ---------------------------------------------------------------------------
# 5. Main orchestrator
# ---------------------------------------------------------------------------

def run_backtest() -> None:
    """Full Phase-4b beta-neutral backtest pipeline."""
    logger.info("Loading %s …", INPUT_PARQUET)
    df = pd.read_parquet(INPUT_PARQUET)
    logger.info("  → %d rows loaded.", len(df))

    logger.info(
        "Running BETA-NEUTRAL simulation  "
        "[capital=$%,.0f  BTC_leg=$%,.0f  fee=%.2f%%] …",
        INITIAL_CAPITAL, BTC_NOTIONAL, FEE_RATE * 100,
    )
    net_equity, gross_equity, trade_records = run_simulation(df)
    metrics = compute_metrics(net_equity, df, trade_records)

    print_metrics(metrics)
    print_trade_log(trade_records)
    plot_tearsheet(net_equity, gross_equity, df, trade_records, metrics)

    logger.info("Phase 4b (Beta-Neutral) complete.")


if __name__ == "__main__":
    run_backtest()
