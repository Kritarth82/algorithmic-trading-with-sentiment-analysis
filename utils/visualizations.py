"""
Phase 6 — Evaluation & Visualizations
All plots: price history, sentiment timeline, training curves, predictions,
equity curve, regime timeline (new), expert utilization (new).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")           # headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from config import NUM_EXPERTS, RESULTS_DIR

logger = logging.getLogger(__name__)

_REGIME_COLORS = {0: "#4CAF50", 1: "#FFC107", 2: "#F44336"}
_REGIME_LABELS = {0: "Trending (low vol)", 1: "Normal", 2: "Crisis (high vol)"}
_EXPERT_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                   "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path, tight: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Existing plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_price_history(
    price_datasets: Dict[str, pd.DataFrame],
    save_path: Path = RESULTS_DIR / "price_history.png",
):
    n      = len(price_datasets)
    ncols  = min(4, n)
    nrows  = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                             squeeze=False)
    for ax, (ticker, df) in zip(axes.flat, price_datasets.items()):
        ax.plot(df.index, df["close"], linewidth=0.8)
        ax.set_title(ticker, fontsize=10)
        ax.set_xlabel(""); ax.set_ylabel("Close")
        ax.tick_params(axis="x", rotation=30, labelsize=7)

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Price History", fontsize=12, y=1.01)
    _save(fig, save_path)


def plot_sentiment_timeline(
    sentiment_data: Dict[str, pd.DataFrame],
    ticker: str,
    save_path: Path = RESULTS_DIR / "sentiment_timeline.png",
):
    df  = sentiment_data.get(ticker)
    if df is None or "finbert_compound" not in df.columns:
        logger.warning(f"No sentiment data for {ticker}")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(df.index, df["finbert_compound"], color="#1976D2", linewidth=0.7)
    axes[0].axhline(0, color="grey", linestyle="--", linewidth=0.5)
    axes[0].set_ylabel("Compound Score")
    axes[0].set_title(f"{ticker} — FinBERT Sentiment")

    if "news_volume" in df.columns:
        axes[1].bar(df.index, df["news_volume"], color="#FFA726", width=1.0, alpha=0.7)
        axes[1].set_ylabel("Daily Articles")

    if "close" in df.columns:
        axes[2].plot(df.index, df["close"], color="#43A047", linewidth=0.7)
        axes[2].set_ylabel("Close Price")

    _save(fig, save_path)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Path = RESULTS_DIR / "training_curves.png",
):
    keys = list(history.keys())
    ncols = 2
    nrows = (len(keys) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), squeeze=False)

    for ax, key in zip(axes.flat, keys):
        ax.plot(history[key], linewidth=0.9)
        ax.set_title(key); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")

    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)

    fig.suptitle("Training Curves", fontsize=12)
    _save(fig, save_path)


def plot_predictions_vs_actual(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    dates:    pd.DatetimeIndex,
    ticker:   str = "",
    save_path: Path = RESULTS_DIR / "predictions.png",
):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)

    axes[0].plot(dates, y_true, label="Actual",    linewidth=0.8, alpha=0.8)
    axes[0].plot(dates, y_pred, label="Predicted", linewidth=0.8, alpha=0.8)
    axes[0].axhline(0, color="grey", linestyle="--", linewidth=0.4)
    axes[0].set_ylabel("5-Day Forward Return")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"{ticker} — Predicted vs Actual Returns")

    axes[1].scatter(y_true, y_pred, s=3, alpha=0.4)
    lim = max(abs(y_true).max(), abs(y_pred).max()) * 1.1
    axes[1].set_xlim(-lim, lim); axes[1].set_ylim(-lim, lim)
    axes[1].axline((0, 0), slope=1, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Predicted")

    _save(fig, save_path)


def plot_equity_curve(
    portfolio_df: pd.DataFrame,
    per_asset:    Dict[str, pd.DataFrame],
    save_path:    Path = RESULTS_DIR / "equity_curve.png",
):
    fig, ax = plt.subplots(figsize=(14, 6))

    for ticker, df in per_asset.items():
        ax.plot(df.index, df["equity"], linewidth=0.6, alpha=0.5, label=ticker)

    if "portfolio_equity" in portfolio_df.columns:
        ax.plot(portfolio_df.index, portfolio_df["portfolio_equity"],
                linewidth=2.0, color="black", label="Portfolio")

    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Equity Curves")
    ax.legend(fontsize=7, ncol=4)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    _save(fig, save_path)


def plot_ablation(
    results: Dict[str, Dict[str, float]],
    metric:  str = "sharpe",
    save_path: Path = RESULTS_DIR / "ablation.png",
):
    models  = list(results.keys())
    values  = [results[m].get(metric, 0.0) for m in models]
    colors  = ["#1976D2" if v >= 0 else "#E53935" for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars    = ax.bar(models, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.6)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Ablation Study — {metric.upper()}")
    ax.set_xticklabels(models, rotation=20, ha="right")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# New: Regime Timeline
# ─────────────────────────────────────────────────────────────────────────────

def plot_regime_timeline(
    test_dates:      pd.DatetimeIndex,
    regime_preds:    np.ndarray,
    regime_actuals:  np.ndarray,
    equity_curve:    pd.Series,
    save_path:       Path = RESULTS_DIR / "regime_timeline.png",
):
    """
    Top panel:    Equity curve segments colored by predicted regime.
    Bottom panel: Predicted vs actual regime bar chart.
    """
    fig = plt.figure(figsize=(15, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.35)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    # ── Top: equity curve colored by regime ──────────────────────────────────
    n = min(len(test_dates), len(equity_curve), len(regime_preds))
    dates_n  = test_dates[:n]
    eq_n     = equity_curve.values[:n]
    reg_n    = regime_preds[:n]

    for i in range(n - 1):
        color = _REGIME_COLORS.get(int(reg_n[i]), "#888888")
        ax_top.plot(dates_n[i : i + 2], eq_n[i : i + 2],
                    color=color, linewidth=1.5)

    # Legend patches
    handles = [
        mpatches.Patch(color=_REGIME_COLORS[r], label=_REGIME_LABELS[r])
        for r in [0, 1, 2]
    ]
    ax_top.legend(handles=handles, fontsize=8, loc="upper left")
    ax_top.set_ylabel("Portfolio Value ($)")
    ax_top.set_title("Equity Curve — Colored by Predicted Regime")
    ax_top.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )

    # ── Bottom: predicted vs actual regime ───────────────────────────────────
    n2 = min(n, len(regime_actuals))
    x  = np.arange(n2)

    ax_bot.bar(x, [1] * n2,
               color=[_REGIME_COLORS.get(int(r), "#888") for r in regime_preds[:n2]],
               alpha=0.9, label="Predicted", width=1.0)
    ax_bot.bar(x, [-1] * n2,
               color=[_REGIME_COLORS.get(int(r), "#888") for r in regime_actuals[:n2]],
               alpha=0.9, label="Actual", width=1.0)
    ax_bot.axhline(0, color="black", linewidth=0.5)
    ax_bot.set_ylim(-1.5, 1.5)
    ax_bot.set_yticks([-1, 1]); ax_bot.set_yticklabels(["Actual", "Predicted"])
    ax_bot.set_xlabel("Trading Day (test set)")
    ax_bot.set_title("Regime: Predicted (top) vs Actual (bottom)")

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# New: Expert Utilization (MoE router weights)
# ─────────────────────────────────────────────────────────────────────────────

def plot_expert_utilization(
    router_weights: np.ndarray,           # (N, num_experts)
    test_dates:     pd.DatetimeIndex,
    save_path:      Path = RESULTS_DIR / "expert_utilization.png",
):
    """
    Stacked area chart showing each MoE expert's routing weight over time.
    Reveals which expert activates during different market periods.
    """
    n_experts = router_weights.shape[1]
    n         = min(len(test_dates), len(router_weights))
    dates     = test_dates[:n]
    weights   = router_weights[:n]           # (n, E)

    # Smooth with 5-day rolling average for legibility
    smooth = pd.DataFrame(weights).rolling(5, min_periods=1).mean().values

    fig, ax = plt.subplots(figsize=(15, 5))
    labels  = [f"Expert {i}" for i in range(n_experts)]
    colors  = _EXPERT_PALETTE[:n_experts]

    ax.stackplot(
        range(n),
        smooth.T,
        labels=labels,
        colors=colors,
        alpha=0.85,
    )

    # Tick dates sparsely
    step = max(1, n // 10)
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels(
        [str(d.date()) for d in dates[::step]],
        rotation=30, ha="right", fontsize=7,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Routing Weight")
    ax.set_xlabel("Trading Day (test set)")
    ax.set_title("MoE Expert Utilization Over Time")
    ax.legend(loc="upper right", fontsize=8, ncol=n_experts)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_correlation(
    df:        pd.DataFrame,
    feature_cols: List[str],
    save_path: Path = RESULTS_DIR / "feature_correlation.png",
):
    corr = df[feature_cols].dropna().corr()
    fig, ax = plt.subplots(figsize=(max(8, len(feature_cols) * 0.5),
                                    max(7, len(feature_cols) * 0.45)))
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, linewidths=0.4,
                annot=len(feature_cols) <= 20, fmt=".1f", annot_kws={"size": 6})
    ax.set_title("Feature Correlation Matrix")
    _save(fig, save_path)
