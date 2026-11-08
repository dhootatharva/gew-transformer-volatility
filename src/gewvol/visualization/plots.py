"""Static figures written to results/figures/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
SEQUENTIAL = "Blues"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
})


def plot_predictions(dates: pd.DatetimeIndex, y: np.ndarray, preds: dict[str, np.ndarray],
                     path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(dates, y, color=INK, linewidth=2.2, label="Actual RV")
    for i, (name, p) in enumerate(preds.items()):
        ax.plot(dates, p, color=SERIES[i], linewidth=1.4, alpha=0.95, label=name)
    ax.set_title(title, loc="left")
    ax.set_ylabel("Annualised realized volatility")
    ax.legend(ncol=len(preds) + 1, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_metric_bars(df: pd.DataFrame, metrics: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.2))
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        vals = df[metric].sort_values()
        colors = [SERIES[0] if "GEW" in n else "#a7a6a0" for n in vals.index]
        bars = ax.barh(vals.index, vals.values, color=colors, height=0.6)
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in vals.values], padding=3, fontsize=8, color=INK_2)
        ax.set_xlim(0, vals.max() * 1.18)
        ax.set_title(f"{metric} on the test set (lower is better)", loc="left")
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_market_spillover(spill: np.ndarray, markets: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(spill, cmap=SEQUENTIAL, vmin=0, vmax=max(0.7, spill.max()))
    for i in range(len(markets)):
        for j in range(len(markets)):
            v = spill[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="white" if v > 0.5 else INK, fontsize=10)
    ax.set_xticks(range(len(markets)), markets)
    ax.set_yticks(range(len(markets)), markets)
    ax.set_xlabel("Attended market (key)")
    ax.set_ylabel("Querying market")
    ax.grid(False)
    ax.set_title(title, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_attention_heatmap(attn: np.ndarray, num_markets: int, path: Path, title: str) -> None:
    """Token-level attention collapsed over markets to a (T, T) time x time heatmap."""
    T = attn.shape[0] // num_markets
    per_time = attn.reshape(T, num_markets, T, num_markets).mean(axis=(1, 3))
    lags = np.arange(-T, 0)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(per_time, cmap=SEQUENTIAL, aspect="auto",
                   extent=(lags[0] - 0.5, lags[-1] + 0.5, lags[-1] + 0.5, lags[0] - 0.5))
    ax.set_xlabel("Attended day (relative to forecast origin)")
    ax.set_ylabel("Querying day")
    ax.grid(False)
    ax.set_title(title, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_lag_importance(lags: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, (name, imp) in enumerate(lags.items()):
        x = np.arange(-len(imp), 0)
        ax.plot(x, imp, color=SERIES[i], linewidth=2, label=name)
    ax.set_xlabel("Day relative to forecast origin")
    ax.set_ylabel("Mean attention received")
    ax.set_title("Attention by look-back day", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
