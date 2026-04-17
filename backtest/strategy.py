"""
Phase 5 — Economic Relevance Engine

This module does not claim alpha. It stress-tests whether model outputs remain
economically meaningful after realistic frictions (costs, lagged execution,
position caps, and regime scaling).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    BUY_THRESHOLD, SELL_THRESHOLD, TRANSACTION_COST,
    MAX_POSITION, INITIAL_CAPITAL, RISK_FREE_RATE,
    ROLLING_SHARPE_WINDOW, TICKERS, BT_EXECUTION_LAG_DAYS,
    FORECAST_HORIZON,
    RISK_VOL_TARGET_DAILY, RISK_VOL_LOOKBACK,
    RISK_MIN_POSITION_SCALE, RISK_MAX_POSITION_SCALE,
    RISK_MAX_DRAWDOWN_SOFT, RISK_DRAWDOWN_BRAKE,
    RISK_PORTFOLIO_WEIGHTING, RISK_PORTFOLIO_VOL_LOOKBACK,
    RISK_PORTFOLIO_TREND_LOOKBACK, RISK_PORTFOLIO_TREND_THRESHOLD,
    RISK_PORTFOLIO_MAX_WEIGHT,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Position sizing helpers
# ─────────────────────────────────────────────────────────────────────────────

_REGIME_SCALE: Dict[int, float] = {0: 1.0, 1: 0.8, 2: 0.5}


def regime_adjusted_position(
    base_signal: float,
    regime: int,
    regime_position_scale: Dict[int, float] = _REGIME_SCALE,
) -> float:
    """
    Scale position size by predicted regime.

    Regime convention:
      0 = low-risk/trending,
      1 = normal,
      2 = high-risk/crisis.
    """
    scale = float(regime_position_scale.get(int(regime), 1.0))
    return float(base_signal) * scale


def _signal_from_prediction(
    pred: float,
    buy_threshold: float = BUY_THRESHOLD,
    sell_threshold: float = SELL_THRESHOLD,
) -> float:
    """Map a predicted return to a raw position weight {-1, 0, 1}."""
    if pred > buy_threshold:
        return 1.0
    elif pred < sell_threshold:
        return -1.0
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-asset backtest
# ─────────────────────────────────────────────────────────────────────────────

def backtest_single(
    prices:   pd.Series,
    preds:    np.ndarray,
    regimes:  Optional[np.ndarray] = None,
    dates:    Optional[pd.Index] = None,
    buy_threshold: Optional[float] = None,
    sell_threshold: Optional[float] = None,
    hold_period: Optional[int] = None,
    transaction_cost: Optional[float] = None,
    max_position: Optional[float] = None,
    vol_target_daily: Optional[float] = None,
    vol_lookback: Optional[int] = None,
    min_position_scale: Optional[float] = None,
    max_position_scale: Optional[float] = None,
    max_drawdown_soft: Optional[float] = None,
    drawdown_brake: Optional[float] = None,
) -> pd.DataFrame:
    """
    Run a single-asset backtest.

    Args:
        prices:  Close prices aligned with `preds` (same length).
        preds:   Predicted 5-day forward returns, shape (N,).
        regimes: Predicted regime labels (0/1/2), shape (N,). None → regime 1.
        dates:   DatetimeIndex for the result.

    Returns:
        DataFrame with columns: signal, position, returns, strategy_returns,
                                 equity, transaction_costs, cum_cost.
    """
    N       = len(preds)
    if dates is None:
        dates = pd.RangeIndex(N)

    actual_returns = prices.pct_change().fillna(0).values[-N:]

    buy_thr = BUY_THRESHOLD if buy_threshold is None else float(buy_threshold)
    sell_thr = SELL_THRESHOLD if sell_threshold is None else float(sell_threshold)
    hold_days = max(int(FORECAST_HORIZON if hold_period is None else hold_period), 1)
    tx_cost = TRANSACTION_COST if transaction_cost is None else float(transaction_cost)
    pos_cap = MAX_POSITION if max_position is None else float(max_position)
    target_vol = RISK_VOL_TARGET_DAILY if vol_target_daily is None else float(vol_target_daily)
    vol_lb = max(2, int(RISK_VOL_LOOKBACK if vol_lookback is None else vol_lookback))
    min_scale = float(RISK_MIN_POSITION_SCALE if min_position_scale is None else min_position_scale)
    max_scale = float(RISK_MAX_POSITION_SCALE if max_position_scale is None else max_position_scale)
    dd_soft = float(RISK_MAX_DRAWDOWN_SOFT if max_drawdown_soft is None else max_drawdown_soft)
    dd_brake = float(RISK_DRAWDOWN_BRAKE if drawdown_brake is None else drawdown_brake)

    if regimes is None:
        # Build simple realized-volatility regimes from lagged returns.
        rv = pd.Series(actual_returns).rolling(20, min_periods=5).std().shift(1)
        q1, q2 = rv.quantile(0.33), rv.quantile(0.66)
        regimes = np.where(rv <= q1, 0, np.where(rv <= q2, 1, 2))
        regimes = np.nan_to_num(regimes, nan=1.0).astype(int)
    else:
        regimes = np.asarray(regimes)[:N].astype(int)

    signals = np.array([
        _signal_from_prediction(p, buy_threshold=buy_thr, sell_threshold=sell_thr)
        for p in preds
    ], dtype=float)

    # Hold each directional forecast for the full forecast horizon.
    # The position only changes when the hold expires or the prediction flips.
    raw_positions = np.zeros(N, dtype=float)
    current_direction = 0.0
    days_left = 0

    for i, signal in enumerate(signals):
        desired_direction = float(np.sign(signal))

        if current_direction == 0.0:
            if desired_direction != 0.0:
                current_direction = desired_direction
                days_left = hold_days
        else:
            if desired_direction != 0.0 and desired_direction != current_direction:
                current_direction = desired_direction
                days_left = hold_days
            elif days_left <= 0:
                if desired_direction != 0.0:
                    current_direction = desired_direction
                    days_left = hold_days
                else:
                    current_direction = 0.0
                    days_left = 0

        base_pos = current_direction * pos_cap
        raw_positions[i] = regime_adjusted_position(base_pos, int(regimes[i]))
        if current_direction != 0.0:
            days_left -= 1

    # Volatility-target scaling based on lagged realized volatility.
    realized_vol = (
        pd.Series(actual_returns)
        .rolling(vol_lb, min_periods=max(5, vol_lb // 2))
        .std()
        .shift(1)
        .to_numpy()
    )
    vol_scale = np.ones(N, dtype=float)
    valid_vol = np.isfinite(realized_vol)
    vol_scale[valid_vol] = target_vol / (np.abs(realized_vol[valid_vol]) + 1e-9)
    vol_scale = np.clip(vol_scale, min_scale, max_scale)

    # Dynamic drawdown brake applied during stress.
    positions = np.zeros(N, dtype=float)
    costs = np.zeros(N, dtype=float)
    strat_returns = np.zeros(N, dtype=float)
    equity = np.zeros(N, dtype=float)

    eq = float(INITIAL_CAPITAL)
    peak = float(INITIAL_CAPITAL)
    prev_pos = 0.0

    for i in range(N):
        pos_i = raw_positions[i] * vol_scale[i]
        dd_now = (eq / peak) - 1.0
        if dd_soft > 0 and dd_now <= -abs(dd_soft):
            pos_i *= max(0.0, min(1.0, dd_brake))

        change = abs(pos_i - prev_pos)
        cost_i = change * tx_cost
        ret_i = pos_i * actual_returns[i] - cost_i

        eq = max(1.0, eq * (1.0 + ret_i))
        peak = max(peak, eq)

        positions[i] = pos_i
        costs[i] = cost_i
        strat_returns[i] = ret_i
        equity[i] = eq
        prev_pos = pos_i

    result = pd.DataFrame({
        "prediction":        preds,
        "signal":            signals,
        "position":          positions,
        "actual_return":     actual_returns,
        "strategy_return":   strat_returns,
        "equity":            equity,
        "transaction_cost":  costs,
        "cum_cost":          np.cumsum(costs),
    }, index=dates)

    return result


def aggregate_portfolio_returns(
    all_returns: pd.DataFrame,
    strategy_params: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Aggregate per-asset strategy returns into a single portfolio return stream.

    Supports:
      - equal: equal-weight with cash on missing sessions (legacy behavior)
      - risk_trend: lagged inverse-volatility weights with momentum gate
    """
    if all_returns is None or all_returns.empty:
        return pd.Series(dtype=float)

    strategy_params = strategy_params or {}
    weighting_mode = str(
        strategy_params.get("portfolio_weighting", RISK_PORTFOLIO_WEIGHTING)
    ).lower()

    if weighting_mode != "risk_trend":
        return all_returns.fillna(0.0).mean(axis=1)

    vol_lb = max(
        4,
        int(strategy_params.get("portfolio_vol_lookback", RISK_PORTFOLIO_VOL_LOOKBACK)),
    )
    trend_lb = max(
        4,
        int(strategy_params.get("portfolio_trend_lookback", RISK_PORTFOLIO_TREND_LOOKBACK)),
    )
    trend_thr = float(
        strategy_params.get("portfolio_trend_threshold", RISK_PORTFOLIO_TREND_THRESHOLD)
    )
    max_w = float(
        strategy_params.get("portfolio_max_weight", RISK_PORTFOLIO_MAX_WEIGHT)
    )

    available = all_returns.notna()
    equal_w = (
        available.astype(float)
        .div(available.sum(axis=1).replace(0, np.nan), axis=0)
        .fillna(0.0)
    )

    rolling_vol = (
        all_returns
        .rolling(vol_lb, min_periods=max(4, vol_lb // 3))
        .std()
        .shift(1)
    )
    inv_vol = (1.0 / (rolling_vol.abs() + 1e-9)).where(available)

    trailing_mean = (
        all_returns
        .rolling(trend_lb, min_periods=max(4, trend_lb // 3))
        .mean()
        .shift(1)
    )
    raw_w = inv_vol.where(trailing_mean > trend_thr, 0.0)
    row_sum = raw_w.sum(axis=1)

    fallback_w = inv_vol.div(inv_vol.sum(axis=1).replace(0, np.nan), axis=0)
    weights = raw_w.div(row_sum.replace(0, np.nan), axis=0)

    zero_rows = row_sum <= 0
    if bool(np.any(zero_rows.to_numpy())):
        weights.loc[zero_rows] = fallback_w.loc[zero_rows]

    weights = weights.fillna(fallback_w).fillna(equal_w).fillna(0.0)

    if max_w < 0.999:
        weights = weights.clip(lower=0.0, upper=max_w)
        weights = (
            weights
            .div(weights.sum(axis=1).replace(0, np.nan), axis=0)
            .fillna(equal_w)
            .fillna(0.0)
        )

    portfolio_returns = (all_returns.fillna(0.0) * weights).sum(axis=1)
    return portfolio_returns


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio-level backtest
# ─────────────────────────────────────────────────────────────────────────────

def backtest_portfolio(
    price_datasets:   Dict[str, pd.DataFrame],
    predictions_dict: Dict[str, np.ndarray],
    prediction_dates_dict: Optional[Dict[str, pd.DatetimeIndex]] = None,
    regimes_dict:     Optional[Dict[str, np.ndarray]] = None,
    test_start:       Optional[str] = None,
    strategy_params: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run portfolio-level economic relevance check across selected tickers.

    Returns:
        portfolio_df: Daily portfolio return/equity series.
        per_asset:    Per-ticker strategy diagnostics.
    """
    from config import TEST_START
    test_start = test_start or TEST_START

    per_asset: Dict[str, pd.DataFrame] = {}
    strategy_params = strategy_params or {}

    for ticker, price_df in price_datasets.items():
        if ticker not in predictions_dict:
            continue

        test_df = price_df[price_df.index >= test_start].copy()
        preds   = predictions_dict[ticker]

        # If model output dates are available, align execution strictly to those
        # timestamps to avoid lookback-offset drift.
        if prediction_dates_dict and ticker in prediction_dates_dict:
            pred_dates = pd.DatetimeIndex(pd.to_datetime(prediction_dates_dict[ticker]))
            n0 = min(len(preds), len(pred_dates))
            preds = preds[:n0]
            pred_dates = pred_dates[:n0]

            valid_mask = pred_dates.isin(test_df.index)
            pred_dates = pred_dates[valid_mask]
            preds = preds[np.asarray(valid_mask)]
            test_df = test_df.loc[pred_dates]
            n = len(test_df)
        else:
            n = min(len(test_df), len(preds))
            test_df = test_df.iloc[:n]
            preds = preds[:n]

        regimes = regimes_dict[ticker][:n] if regimes_dict and ticker in regimes_dict else None

        result = backtest_single(
            prices=test_df["close"],
            preds=preds,
            regimes=regimes,
            dates=test_df.index[:n],
            buy_threshold=strategy_params.get("buy_threshold"),
            sell_threshold=strategy_params.get("sell_threshold"),
            hold_period=strategy_params.get("hold_period"),
            transaction_cost=strategy_params.get("transaction_cost"),
            max_position=strategy_params.get("max_position"),
            vol_target_daily=strategy_params.get("vol_target_daily"),
            vol_lookback=strategy_params.get("vol_lookback"),
            min_position_scale=strategy_params.get("min_position_scale"),
            max_position_scale=strategy_params.get("max_position_scale"),
            max_drawdown_soft=strategy_params.get("max_drawdown_soft"),
            drawdown_brake=strategy_params.get("drawdown_brake"),
        )
        per_asset[ticker] = result

    # Equal-weight portfolio
    if not per_asset:
        return pd.DataFrame(), per_asset

    all_returns = pd.DataFrame({
        t: df["strategy_return"] for t, df in per_asset.items()
    })

    portfolio_return = aggregate_portfolio_returns(
        all_returns,
        strategy_params=strategy_params,
    )
    portfolio_equity = INITIAL_CAPITAL * (1 + portfolio_return).cumprod()

    portfolio_df = pd.DataFrame({
        "portfolio_return": portfolio_return,
        "portfolio_equity": portfolio_equity,
    })

    return portfolio_df, per_asset


# ─────────────────────────────────────────────────────────────────────────────
# Performance metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    strategy_returns: pd.Series,
    label: str = "",
) -> Dict[str, float]:
    """
    Compute annualised Sharpe, Sortino, max drawdown, CAGR, hit rate.
    Assumes daily returns.
    """
    r  = strategy_returns.dropna()
    if len(r) < 2:
        return {}

    ann    = 252
    daily_rf = (1 + RISK_FREE_RATE) ** (1 / ann) - 1

    excess = r - daily_rf
    sharpe = excess.mean() / (excess.std() + 1e-9) * np.sqrt(ann)

    downside_std = r[r < daily_rf].std() + 1e-9
    sortino      = excess.mean() / downside_std * np.sqrt(ann)

    cum_returns  = (1 + r).cumprod()
    rolling_max  = cum_returns.cummax()
    drawdowns    = cum_returns / rolling_max - 1
    max_drawdown = float(drawdowns.min())

    total_return = float(cum_returns.iloc[-1] - 1)
    n_years      = len(r) / ann
    cagr         = float((1 + total_return) ** (1 / max(n_years, 1e-9)) - 1)

    hit_rate = float((r > 0).mean())

    # Rolling Sharpe
    rolling_sharpe = (
        r.rolling(ROLLING_SHARPE_WINDOW).mean() /
        (r.rolling(ROLLING_SHARPE_WINDOW).std() + 1e-9) * np.sqrt(ann)
    ).dropna()

    metrics = {
        "sharpe":            round(sharpe, 4),
        "sortino":           round(sortino, 4),
        "max_drawdown":      round(max_drawdown, 4),
        "total_return":      round(total_return, 4),
        "cagr":              round(cagr, 4),
        "hit_rate":          round(hit_rate, 4),
        "rolling_sharpe_mean": round(float(rolling_sharpe.mean()), 4),
    }

    if label:
        logger.info(
            f"[{label}] Sharpe={sharpe:.3f} Sortino={sortino:.3f} "
            f"MaxDD={max_drawdown:.3f} CAGR={cagr:.3f} HitRate={hit_rate:.3f}"
        )
    return metrics
