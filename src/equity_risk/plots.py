"""Charts for the report.

All of them save a PNG into outputs/charts/ and return the path. Matplotlib
is used with the non-interactive "Agg" backend so the script runs the same
way from a plain terminal.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from . import config, metrics

# ---------------------------------------------------------------------------
# Shared look
# ---------------------------------------------------------------------------
# Four fixed colours, chosen to stay distinguishable for colour-blind readers
# and when printed in greyscale. Each ticker keeps the same colour in every
# chart, so a series can be recognised across the whole report.
SERIES_COLORS = ["#0072B2", "#009E73", "#D55E00", "#8C5AA8"]
GRID_COLOR = "#D8D8D4"
INK = "#2A2A28"
MUTED_INK = "#6B6B66"
SURFACE = "#FCFCFB"

LINE_WIDTH = 1.6
FIG_DPI = 150


def color_map(columns) -> dict[str, str]:
    """Fix one colour per ticker, in the order given by the config.

    Colours are tied to the asset rather than to its position in the chart,
    so dropping a series never recolours the others.
    """
    order = [t for t in config.ALL_TICKERS if t in columns]
    order += [c for c in columns if c not in order]
    return {name: SERIES_COLORS[i % len(SERIES_COLORS)] for i, name in enumerate(order)}


def _style_axes(ax, ylabel: str = "") -> None:
    """Push the chart furniture into the background so the data stands out."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=MUTED_INK, labelsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED_INK, fontsize=10)


def _new_figure(width: float = 10.0, height: float = 5.5):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _title(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=16 if subtitle else 10)
    if subtitle:
        ax.text(
            0.0,
            1.015,
            subtitle,
            transform=ax.transAxes,
            color=MUTED_INK,
            fontsize=9.5,
            va="bottom",
        )


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, facecolor=SURFACE)
    plt.close(fig)
    return path


def _legend_headroom(ax, fraction: float = 0.16) -> None:
    """Lift the top of the y-axis so the legend does not sit on the data."""
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + (high - low) * fraction)


def _as_percent(ax) -> None:
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%")
    )


# ---------------------------------------------------------------------------
# 1. Cumulative returns
# ---------------------------------------------------------------------------
def plot_cumulative_returns(
    cumulative: pd.DataFrame, out_dir: Path = config.CHART_DIR
) -> Path:
    """Growth of each asset over the sample, starting from zero."""
    colors = color_map(cumulative.columns)
    fig, ax = _new_figure()

    for name in cumulative.columns:
        ax.plot(
            cumulative.index,
            cumulative[name],
            color=colors[name],
            linewidth=LINE_WIDTH,
            label=name,
        )
        # label the end of each line as well as the legend, so identity is
        # never carried by colour alone
        last = cumulative[name].iloc[-1]
        ax.annotate(
            f" {last * 100:.0f}%",
            xy=(cumulative.index[-1], last),
            color=colors[name],
            fontsize=9,
            va="center",
        )

    ax.axhline(0.0, color=MUTED_INK, linewidth=0.8, alpha=0.6)
    _style_axes(ax, "Cumulative return")
    _as_percent(ax)
    _title(ax, "Cumulative returns", "Daily adjusted prices, reinvested, start = 0%")
    _legend_headroom(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncols=4, loc="upper left")
    ax.margins(x=0.08)
    return _save(fig, Path(out_dir) / "cumulative_returns.png")


# ---------------------------------------------------------------------------
# 2. Rolling volatility
# ---------------------------------------------------------------------------
def plot_rolling_volatility(
    rolling_vol: pd.DataFrame,
    window: int = config.ROLLING_WINDOW,
    out_dir: Path = config.CHART_DIR,
) -> Path:
    """How volatile each asset was over a moving window."""
    colors = color_map(rolling_vol.columns)
    fig, ax = _new_figure()

    for name in rolling_vol.columns:
        ax.plot(
            rolling_vol.index,
            rolling_vol[name],
            color=colors[name],
            linewidth=LINE_WIDTH,
            label=name,
        )

    _style_axes(ax, "Annualized volatility")
    _as_percent(ax)
    _title(
        ax,
        "Rolling volatility",
        f"Standard deviation over {window} trading days, annualized",
    )
    _legend_headroom(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncols=4, loc="upper left")
    return _save(fig, Path(out_dir) / "rolling_volatility.png")


# ---------------------------------------------------------------------------
# 3. Drawdowns
# ---------------------------------------------------------------------------
def plot_drawdowns(drawdowns: pd.DataFrame, out_dir: Path = config.CHART_DIR) -> Path:
    """One panel per asset, since filled drawdowns overlap badly."""
    colors = color_map(drawdowns.columns)
    names = list(drawdowns.columns)
    rows = int(np.ceil(len(names) / 2))

    fig, axes = plt.subplots(rows, 2, figsize=(10.0, 2.6 * rows), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes).ravel()

    worst = float(drawdowns.min().min())
    for ax, name in zip(axes, names):
        series = drawdowns[name]
        ax.fill_between(
            series.index, series, 0.0, color=colors[name], alpha=0.25, linewidth=0
        )
        ax.plot(series.index, series, color=colors[name], linewidth=1.2)

        trough = series.idxmin()
        ax.plot([trough], [series.min()], marker="o", markersize=5, color=colors[name])
        # put the label on whichever side of the trough has room, so a
        # late drawdown does not run off the right edge
        late = series.index.get_loc(trough) > len(series) * 0.7
        ax.annotate(
            f"{series.min() * 100:.1f}%",
            xy=(trough, series.min()),
            xytext=(-6 if late else 6, 6),
            textcoords="offset points",
            ha="right" if late else "left",
            color=INK,
            fontsize=9,
        )

        _style_axes(ax)
        _as_percent(ax)
        ax.set_ylim(worst * 1.18, 0.02)
        ax.set_title(name, color=INK, fontsize=11, loc="left")

    for ax in axes[len(names):]:
        ax.set_visible(False)

    fig.suptitle(
        "Drawdowns from running peak",
        color=INK,
        fontsize=13,
        x=0.012,
        ha="left",
        y=0.995,
    )
    return _save(fig, Path(out_dir) / "drawdowns.png")


# ---------------------------------------------------------------------------
# 4. Correlation heatmap
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(
    corr: pd.DataFrame, out_dir: Path = config.CHART_DIR
) -> Path:
    """Correlation of daily returns between every pair of assets.

    Correlation runs from -1 to +1 around a natural midpoint of zero, so the
    colour scale is a two-sided one fixed to that full range: a neutral tone
    means uncorrelated, and the scale does not shift with the data. Every
    cell is also printed as a number.
    """
    labels = list(corr.columns)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    fig.patch.set_facecolor(SURFACE)

    image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1.0, vmax=1.0)

    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.tick_params(colors=MUTED_INK, length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.iloc[i, j]
            # dark text on the pale middle of the scale, white on the ends
            text_color = "#FFFFFF" if abs(value) > 0.6 else INK
            ax.text(
                j, i, f"{value:.2f}", ha="center", va="center",
                color=text_color, fontsize=9.5,
            )

    bar = fig.colorbar(image, ax=ax, shrink=0.82)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED_INK, labelsize=9)
    bar.set_label("Correlation of daily returns", color=MUTED_INK, fontsize=9.5)

    ax.set_title("Return correlations", color=INK, fontsize=13, loc="left", pad=12)
    return _save(fig, Path(out_dir) / "correlation_heatmap.png")


# ---------------------------------------------------------------------------
# 5. Return distribution with VaR
# ---------------------------------------------------------------------------
def plot_return_distributions(
    returns: pd.DataFrame,
    confidence: float = config.VAR_CONFIDENCE,
    out_dir: Path = config.CHART_DIR,
) -> Path:
    """Histogram of daily returns per asset, with the VaR cutoff marked."""
    colors = color_map(returns.columns)
    names = list(returns.columns)
    rows = int(np.ceil(len(names) / 2))
    level = int(confidence * 100)

    fig, axes = plt.subplots(rows, 2, figsize=(10.0, 3.0 * rows))
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        series = returns[name].dropna()
        var = metrics.historical_var(series, confidence)
        cvar = metrics.historical_cvar(series, confidence)

        ax.hist(series, bins=60, color=colors[name], alpha=0.55, linewidth=0)
        ax.axvline(-var, color=INK, linewidth=1.4, linestyle="--")
        ax.axvline(-cvar, color=MUTED_INK, linewidth=1.2, linestyle=":")

        ax.annotate(
            f"VaR {level}%: {var * 100:.2f}%\nCVaR {level}%: {cvar * 100:.2f}%",
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            va="top",
            fontsize=9,
            color=INK,
        )

        _style_axes(ax, "Days")
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%")
        )
        ax.set_xlabel("Daily return", color=MUTED_INK, fontsize=9.5)
        ax.set_title(name, color=INK, fontsize=11, loc="left")

    for ax in axes[len(names):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Daily return distribution with {level}% VaR "
        "(dashed) and CVaR (dotted)",
        color=INK,
        fontsize=13,
        x=0.012,
        ha="left",
        y=0.995,
    )
    return _save(fig, Path(out_dir) / "return_distributions.png")


# ---------------------------------------------------------------------------
# All of them
# ---------------------------------------------------------------------------
def make_all_charts(
    cumulative: pd.DataFrame,
    rolling_vol: pd.DataFrame,
    drawdowns: pd.DataFrame,
    corr: pd.DataFrame,
    daily_returns: pd.DataFrame,
    out_dir: Path = config.CHART_DIR,
) -> dict[str, Path]:
    """Draw every chart and return a name -> path map for the report."""
    return {
        "cumulative_returns": plot_cumulative_returns(cumulative, out_dir),
        "rolling_volatility": plot_rolling_volatility(rolling_vol, out_dir=out_dir),
        "drawdowns": plot_drawdowns(drawdowns, out_dir),
        "correlation_heatmap": plot_correlation_heatmap(corr, out_dir),
        "return_distributions": plot_return_distributions(daily_returns, out_dir=out_dir),
    }
