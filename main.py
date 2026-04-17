"""
main.py — News Impact Hypothesis Pipeline

This pipeline is framed as a forecasting study to test whether the available
news signal adds incremental predictive power to market data for 5-day returns.

Usage:
    python main.py --mode full        # run full hypothesis evaluation
    python main.py --mode data        # build price + aligned news proxies
    python main.py --mode nlp         # build sentiment proxy features
    python main.py --mode train       # train predictive models
    python main.py --mode backtest    # evaluate tradability under costs
    python main.py --mode eval        # export study diagnostics/plots
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from itertools import combinations
from pathlib import Path
import pandas as pd
import numpy as np
import torch

# ── Must import config first so directories are created ──────────────────────
import config as cfg

from data.price_data import (
    build_price_dataset, chronological_split, normalize_features,
)
from data.sentiment_data import build_sentiment_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(cfg.LOG_DIR / "pipeline.log"),
    ],
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def _seed_everything(seed: int = cfg.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Phase helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_data(args) -> dict:
    """Phase 1+2: Build market dataset and aligned news-proxy dataset."""
    logger.info("═" * 60)
    logger.info("PHASE 1 — Price Data Collection")
    logger.info("═" * 60)
    price_datasets = build_price_dataset(
        tickers=cfg.TICKERS, use_cache=not args.no_cache
    )
    logger.info(f"Downloaded {len(price_datasets)} tickers.")

    logger.info("═" * 60)
    logger.info("PHASE 2 — Sentiment Data (FinancialPhraseBank + Synthetic Fill)")
    logger.info("═" * 60)
    news_datasets = build_sentiment_dataset(
        price_datasets=price_datasets,
        from_date=cfg.TRAIN_START,
        to_date=cfg.TEST_END,
    )
    logger.info(f"Sentiment articles ready for {len(news_datasets)} tickers.")

    return {"price": price_datasets, "news": news_datasets}


def run_nlp(price_datasets: dict, news_datasets: dict) -> dict:
    """Phase 2b: Build sentiment proxy features for hypothesis testing."""
    from nlp.sentiment_analyzer import (
        FinBERTScorer, build_full_sentiment_features,
    )

    logger.info("═" * 60)
    logger.info("PHASE 2b — Sentiment Scoring")
    logger.info("═" * 60)

    scorer = FinBERTScorer() if cfg.USE_FINBERT else None
    merged = build_full_sentiment_features(news_datasets, price_datasets, scorer)
    logger.info(f"Sentiment features built for {len(merged)} tickers.")
    return {"merged": merged}


def run_train(merged_datasets: dict, args) -> dict:
    """Phase 3+4: Train predictive models used in the news-impact study."""
   
    from models.lstm_model import (
        LSTMAttnModel, ModelTrainer, BaselineModels,
        build_sequences, evaluate_predictions,
    )
    from fusion.fusion_model import (
        CrossAttentionFusion, FusionTrainer, build_fusion_sequences,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")

    def _dir_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) == 0:
            return 0.0
        return float(np.mean(np.sign(y_true) == np.sign(y_pred)))

    def _aligned_dir_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        n = min(len(y_true), len(y_pred))
        if n == 0:
            return 0.0
        return _dir_acc(np.asarray(y_true)[:n], np.asarray(y_pred)[:n])

    def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        n = min(len(y_true), len(y_pred))
        if n < 2:
            return 0.0
        yt = np.asarray(y_true, dtype=float)[:n]
        yp = np.asarray(y_pred, dtype=float)[:n]
        if np.std(yt) < 1e-9 or np.std(yp) < 1e-9:
            return 0.0
        corr = float(np.corrcoef(yt, yp)[0, 1])
        return corr if np.isfinite(corr) else 0.0

    corr_w = float(getattr(cfg, "MODEL_VAL_SELECTION_CORR_WEIGHT", 0.70))
    dir_w = float(getattr(cfg, "MODEL_VAL_SELECTION_DIR_WEIGHT", 0.30))
    norm_w = max(corr_w + dir_w, 1e-9)
    corr_w /= norm_w
    dir_w /= norm_w

    def _val_score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        da = _aligned_dir_acc(y_true, y_pred)
        corr = _safe_corr(y_true, y_pred)
        # Shift directional accuracy around random baseline (0.5)
        score = corr_w * corr + dir_w * (da - 0.5)
        return {
            "score": float(score),
            "dir_acc": float(da),
            "corr": float(corr),
        }

    def _simplex_units(n: int, total_units: int):
        if n == 1:
            yield [total_units]
            return
        for i in range(total_units + 1):
            for rest in _simplex_units(n - 1, total_units - i):
                yield [i] + rest

    def _optimize_ensemble(
        y_val: np.ndarray,
        val_pred_map: dict,
        test_pred_map: dict,
        step: float = 0.1,
    ):
        keys = list(val_pred_map.keys())
        if not keys:
            return None, np.array([]), np.array([]), {
                "weights": {}, "val_dir_acc": 0.0, "val_corr": 0.0
            }

        yv = np.asarray(y_val, dtype=float)
        val_min_len = min([len(yv)] + [len(np.asarray(val_pred_map[k])) for k in keys])
        test_min_len = min([len(np.asarray(test_pred_map[k])) for k in keys])
        if val_min_len == 0 or test_min_len == 0:
            return None, np.array([]), np.array([]), {
                "weights": {}, "val_dir_acc": 0.0, "val_corr": 0.0
            }

        yv = yv[:val_min_len]
        val_mat = np.column_stack([
            np.asarray(val_pred_map[k], dtype=float)[:val_min_len] for k in keys
        ])
        test_mat = np.column_stack([
            np.asarray(test_pred_map[k], dtype=float)[:test_min_len] for k in keys
        ])

        total_units = max(1, int(round(1.0 / step)))
        best_score = (-1e9, -1e9, -1e9)
        best_w = np.zeros(len(keys), dtype=float)
        best_w[0] = 1.0

        for units in _simplex_units(len(keys), total_units):
            w = np.asarray(units, dtype=float) / float(total_units)
            if np.count_nonzero(w) == 0:
                continue
            pred_val = val_mat @ w
            val_stats = _val_score(yv, pred_val)
            score = (val_stats["score"], val_stats["corr"], val_stats["dir_acc"])
            if score > best_score:
                best_score = score
                best_w = w

        pred_val = val_mat @ best_w
        pred_test = test_mat @ best_w
        weights = {
            k: float(w) for k, w in zip(keys, best_w) if w > 1e-9
        }
        info = {
            "weights": weights,
            "val_score": float(best_score[0]),
            "val_corr": float(best_score[1]),
            "val_dir_acc": float(best_score[2]),
        }
        return "ensemble", pred_test, pred_val, info

    results_all: dict = {}

    for ticker, merged_df in merged_datasets.items():
        logger.info(f"\n{'─'*40}\nTraining on {ticker}\n{'─'*40}")

        price_cols = [c for c in cfg.PRICE_FEATURES if c in merged_df.columns]
        sent_cols  = [c for c in cfg.SENTIMENT_FEATURES if c in merged_df.columns]

        df_tr, df_vl, df_te = chronological_split(merged_df)

        all_feature_cols = price_cols + sent_cols
        df_tr, df_vl, df_te, scaler = normalize_features(
            df_tr, df_vl, df_te, all_feature_cols
        )

        # ── Phase 3: LSTM-only ───────────────────────────────────────────────
        logger.info("  Building LSTM sequences ...")
        X_tr, y_tr = build_sequences(df_tr, price_cols, cfg.TARGET_COL)
        X_vl, y_vl = build_sequences(df_vl, price_cols, cfg.TARGET_COL)
        X_te, y_te = build_sequences(df_te, price_cols, cfg.TARGET_COL)
        val_dates = df_vl.index[cfg.LOOKBACK:]
        test_dates = df_te.index[cfg.LOOKBACK:]

        if len(X_tr) == 0:
            logger.warning(f"No training samples for {ticker} — skipping.")
            continue

        lstm_model = LSTMAttnModel(
            input_size=len(price_cols),
            hidden_size=cfg.HIDDEN_SIZE,
            num_layers=cfg.NUM_LAYERS,
            n_heads=cfg.ATTN_HEADS,
            dropout=cfg.DROPOUT,
        )
        save_path = cfg.MODEL_DIR / f"{ticker}_lstm.pt"
        trainer   = ModelTrainer(lstm_model, device=device)
        history   = trainer.fit(X_tr, y_tr, X_vl, y_vl,
                                epochs=cfg.EPOCHS, save_path=save_path)
        lstm_vl_preds, _ = trainer.predict(X_vl)
        lstm_preds, _    = trainer.predict(X_te)
        lstm_metrics    = evaluate_predictions(y_te, lstm_preds, f"{ticker}/LSTM")
        lstm_val_da     = _dir_acc(y_vl, lstm_vl_preds)
        logger.info(f"[{ticker}/LSTM] Val DirAcc={lstm_val_da:.4f}")

        # ── Baselines ────────────────────────────────────────────────────────
        baseline  = BaselineModels()
        baseline.fit(X_tr, y_tr)
        bl_vl_preds = baseline.predict(X_vl)
        bl_preds  = baseline.predict(X_te)
        baseline_val_da = {}
        for name, prd in bl_preds.items():
            evaluate_predictions(y_te, prd[:len(y_te)], f"{ticker}/{name.upper()}")
            da = _dir_acc(y_vl, bl_vl_preds[name][:len(y_vl)])
            baseline_val_da[name] = da
            logger.info(f"[{ticker}/{name.upper()}] Val DirAcc={da:.4f}")

        # ── Phase 4: CrossAttentionFusion ────────────────────────────────────
        logger.info("  Building Fusion sequences ...")
        Xp_tr, Xs_tr, yr_tr, yreg_tr = build_fusion_sequences(df_tr, price_cols, sent_cols)
        Xp_vl, Xs_vl, yr_vl, yreg_vl = build_fusion_sequences(df_vl, price_cols, sent_cols)
        Xp_te, Xs_te, yr_te, yreg_te = build_fusion_sequences(df_te, price_cols, sent_cols)

        if len(Xp_tr) == 0:
            logger.warning(f"No fusion training samples for {ticker} — skipping fusion.")
            continue

        fusion_model = CrossAttentionFusion(
            n_price=len(price_cols),
            n_sent=len(sent_cols),
            hidden=cfg.HIDDEN_SIZE,
            n_layers=cfg.NUM_LAYERS,
            n_heads=cfg.ATTN_HEADS,
            dropout=cfg.DROPOUT,
        )
        fusion_save = cfg.MODEL_DIR / f"{ticker}_fusion.pt"
        ftrainer    = FusionTrainer(fusion_model, device=device)
        ftrainer.fit(
            Xp_tr, Xs_tr, yr_tr, yreg_tr,
            Xp_vl, Xs_vl, yr_vl, yreg_vl,
            epochs=cfg.EPOCHS, save_path=fusion_save,
        )
        fusion_vl_preds, _, _ = ftrainer.predict(Xp_vl, Xs_vl)
        fusion_preds, attn_w, router_w = ftrainer.predict(Xp_te, Xs_te)
        fusion_metrics = evaluate_predictions(yr_te, fusion_preds, f"{ticker}/Fusion")
        fusion_val_da  = _dir_acc(yr_vl, fusion_vl_preds)
        logger.info(f"[{ticker}/Fusion] Val DirAcc={fusion_val_da:.4f}")

        # Choose prediction stream by validation directional accuracy.
        # This avoids forcing Fusion when a simpler model generalizes better.
        candidate_preds = {
            "lstm": lstm_preds,
            "fusion": fusion_preds,
            **{k: v[:len(yr_te)] for k, v in bl_preds.items()},
        }
        candidate_val_preds = {
            "lstm": lstm_vl_preds,
            "fusion": fusion_vl_preds,
            **{k: v for k, v in bl_vl_preds.items()},
        }
        val_target = yr_vl if len(yr_vl) > 0 else y_vl
        candidate_val_stats = {
            name: _val_score(val_target, pred)
            for name, pred in candidate_val_preds.items()
        }
        candidate_val_da = {k: float(v["dir_acc"]) for k, v in candidate_val_stats.items()}
        candidate_val_corr = {k: float(v["corr"]) for k, v in candidate_val_stats.items()}
        candidate_val_score = {k: float(v["score"]) for k, v in candidate_val_stats.items()}

        for name in candidate_val_stats:
            logger.info(
                f"[{ticker}/{name.upper()}] ValScore={candidate_val_score[name]:.4f} "
                f"ValCorr={candidate_val_corr[name]:.4f} ValDirAcc={candidate_val_da[name]:.4f}"
            )

        best_single = max(
            candidate_val_score.keys(),
            key=lambda k: float(candidate_val_score.get(k, -1e9)),
        )
        selected_model = best_single
        selected_preds = candidate_preds[best_single]
        selected_val_preds = candidate_val_preds[best_single]
        selected_val_da = float(candidate_val_da[best_single])
        selected_val_corr = float(candidate_val_corr[best_single])
        selected_val_score = float(candidate_val_score[best_single])

        ensemble_info = {
            "weights": {},
            "val_score": 0.0,
            "val_corr": 0.0,
            "val_dir_acc": 0.0,
        }
        if bool(getattr(cfg, "MODEL_USE_ENSEMBLE", False)):
            ensemble_model, ensemble_preds, ensemble_val_preds, ensemble_info = _optimize_ensemble(
                y_val=val_target,
                val_pred_map=candidate_val_preds,
                test_pred_map=candidate_preds,
                step=0.1,
            )
            ensemble_margin = float(getattr(cfg, "MODEL_ENSEMBLE_MARGIN", 0.01))
            if (
                ensemble_model is not None
                and len(ensemble_preds) > 0
                and ensemble_info["val_score"] >= candidate_val_score[best_single] + ensemble_margin
            ):
                selected_model = ensemble_model
                selected_preds = ensemble_preds
                selected_val_preds = ensemble_val_preds
                selected_val_da = float(ensemble_info["val_dir_acc"])
                selected_val_corr = float(ensemble_info["val_corr"])
                selected_val_score = float(ensemble_info["val_score"])
                logger.info(
                    f"[{ticker}] Ensemble selected with weights={ensemble_info['weights']} "
                    f"(Val Score={ensemble_info['val_score']:.4f}, "
                    f"best single score={candidate_val_score[best_single]:.4f})"
                )

        # Post-hoc linear calibration on validation predictions to reduce
        # near-zero output compression and improve thresholded trading utility.
        n_cal = min(len(selected_val_preds), len(val_target))
        if n_cal >= 25:
            x = np.asarray(selected_val_preds[:n_cal], dtype=float)
            y = np.asarray(val_target[:n_cal], dtype=float)
            if np.std(x) > 1e-9:
                slope = float(np.cov(x, y)[0, 1] / (np.var(x) + 1e-9))
                intercept = float(np.mean(y) - slope * np.mean(x))

                # Keep calibration stable and sign-preserving.
                if slope <= 0:
                    slope = 1.0
                    intercept = 0.0
                slope = float(np.clip(slope, 0.5, 3.0))
                intercept = float(np.clip(intercept, -0.01, 0.01))

                selected_preds = slope * np.asarray(selected_preds) + intercept
                selected_val_preds = slope * np.asarray(selected_val_preds) + intercept
                logger.info(
                    f"[{ticker}] Applied output calibration: slope={slope:.3f}, intercept={intercept:.5f}"
                )

        logger.info(
            f"[{ticker}] Selected model for backtest: {selected_model.upper()} "
            f"(ValScore={selected_val_score:.4f}, ValCorr={selected_val_corr:.4f}, "
            f"ValDirAcc={selected_val_da:.4f})"
        )

        ensemble_metrics = None
        if selected_model == "ensemble":
            eval_n = min(len(yr_te), len(selected_preds))
            ensemble_metrics = evaluate_predictions(
                yr_te[:eval_n], selected_preds[:eval_n], f"{ticker}/ENSEMBLE"
            )



        results_all[ticker] = {
            "lstm":     lstm_metrics,
            "fusion":   fusion_metrics,
            "ensemble": ensemble_metrics,
            "preds":    selected_preds,
            "val_preds": np.asarray(selected_val_preds),
            "y_val": np.asarray(val_target),
            "selected_model": selected_model,
            "val_dir_acc": candidate_val_da,
            "val_corr": candidate_val_corr,
            "val_score": candidate_val_score,
            "ensemble_info": ensemble_info,
            "attn_w":   attn_w,
            "router_w": router_w,
            "dates_val": val_dates.tolist(),
            "dates_te": test_dates.tolist(),
            "y_te":     yr_te,
        }

    return results_all


def run_backtest(price_datasets: dict, train_results: dict) -> dict:
    """Phase 5: Economic relevance check via constrained backtest."""
    from backtest.strategy import (
        aggregate_portfolio_returns,
        backtest_portfolio,
        backtest_single,
        compute_metrics,
    )

    logger.info("═" * 60)
    logger.info("PHASE 5 — Backtesting")
    logger.info("═" * 60)

    def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        n = min(len(y_true), len(y_pred))
        if n < 2:
            return 0.0
        yt = np.asarray(y_true, dtype=float)[:n]
        yp = np.asarray(y_pred, dtype=float)[:n]
        if np.std(yt) < 1e-9 or np.std(yp) < 1e-9:
            return 0.0
        corr = float(np.corrcoef(yt, yp)[0, 1])
        return corr if np.isfinite(corr) else 0.0


    predictions_dict = {}
    prediction_dates_dict = {}
    all_predictions = {}
    all_prediction_dates = {}
    ticker_quality = {}

    tuned_params = {
        "buy_threshold": float(cfg.BUY_THRESHOLD),
        "sell_threshold": float(cfg.SELL_THRESHOLD),
        "hold_period": int(max(cfg.FORECAST_HORIZON, 1)),
        "transaction_cost": float(cfg.TRANSACTION_COST),
        "max_position": float(cfg.MAX_POSITION),
        "vol_target_daily": float(cfg.RISK_VOL_TARGET_DAILY),
        "vol_lookback": int(cfg.RISK_VOL_LOOKBACK),
        "min_position_scale": float(cfg.RISK_MIN_POSITION_SCALE),
        "max_position_scale": float(cfg.RISK_MAX_POSITION_SCALE),
        "max_drawdown_soft": float(cfg.RISK_MAX_DRAWDOWN_SOFT),
        "drawdown_brake": float(cfg.RISK_DRAWDOWN_BRAKE),
        "portfolio_weighting": str(cfg.RISK_PORTFOLIO_WEIGHTING),
        "portfolio_vol_lookback": int(cfg.RISK_PORTFOLIO_VOL_LOOKBACK),
        "portfolio_trend_lookback": int(cfg.RISK_PORTFOLIO_TREND_LOOKBACK),
        "portfolio_trend_threshold": float(cfg.RISK_PORTFOLIO_TREND_THRESHOLD),
        "portfolio_max_weight": float(cfg.RISK_PORTFOLIO_MAX_WEIGHT),
    }

    def _validation_portfolio_returns(params: dict) -> pd.Series:
        per_asset_val = {}

        for ticker, res in train_results.items():
            if ticker not in price_datasets:
                continue

            val_preds = np.asarray(res.get("val_preds", []), dtype=float)
            val_dates = pd.DatetimeIndex(pd.to_datetime(res.get("dates_val", [])))

            if len(val_preds) == 0 or len(val_dates) == 0:
                continue

            n0 = min(len(val_preds), len(val_dates))
            val_preds = val_preds[:n0]
            val_dates = val_dates[:n0]

            price_df = price_datasets.get(ticker)
            if price_df is None or "close" not in price_df.columns:
                continue

            valid_mask = val_dates.isin(price_df.index)
            if int(np.sum(valid_mask)) < 2:
                continue

            aligned_dates = val_dates[valid_mask]
            aligned_preds = val_preds[np.asarray(valid_mask)]
            close_series = price_df.loc[aligned_dates, "close"]

            n = min(len(close_series), len(aligned_preds))
            if n < 2:
                continue

            bt_df = backtest_single(
                prices=close_series.iloc[:n],
                preds=aligned_preds[:n],
                dates=aligned_dates[:n],
                buy_threshold=params["buy_threshold"],
                sell_threshold=params["sell_threshold"],
                hold_period=params["hold_period"],
                transaction_cost=params["transaction_cost"],
                max_position=params["max_position"],
                vol_target_daily=params["vol_target_daily"],
                vol_lookback=params["vol_lookback"],
                min_position_scale=params["min_position_scale"],
                max_position_scale=params["max_position_scale"],
                max_drawdown_soft=params["max_drawdown_soft"],
                drawdown_brake=params["drawdown_brake"],
            )
            per_asset_val[ticker] = bt_df["strategy_return"]

        if not per_asset_val:
            return pd.Series(dtype=float)

        val_df = pd.DataFrame(per_asset_val)
        return aggregate_portfolio_returns(val_df, strategy_params=params)

    def _validation_asset_returns(ticker: str, params: dict) -> pd.Series:
        res = train_results.get(ticker, {})
        val_preds = np.asarray(res.get("val_preds", []), dtype=float)
        val_dates = pd.DatetimeIndex(pd.to_datetime(res.get("dates_val", [])))
        if len(val_preds) == 0 or len(val_dates) == 0:
            return pd.Series(dtype=float)

        price_df = price_datasets.get(ticker)
        if price_df is None or "close" not in price_df.columns:
            return pd.Series(dtype=float)

        n0 = min(len(val_preds), len(val_dates))
        val_preds = val_preds[:n0]
        val_dates = val_dates[:n0]
        valid_mask = val_dates.isin(price_df.index)
        if int(np.sum(valid_mask)) < 2:
            return pd.Series(dtype=float)

        aligned_dates = val_dates[valid_mask]
        aligned_preds = val_preds[np.asarray(valid_mask)]
        close_series = price_df.loc[aligned_dates, "close"]
        n = min(len(close_series), len(aligned_preds))
        if n < 2:
            return pd.Series(dtype=float)

        bt_df = backtest_single(
            prices=close_series.iloc[:n],
            preds=aligned_preds[:n],
            dates=aligned_dates[:n],
            buy_threshold=params["buy_threshold"],
            sell_threshold=params["sell_threshold"],
            hold_period=params["hold_period"],
            transaction_cost=params["transaction_cost"],
            max_position=params["max_position"],
            vol_target_daily=params["vol_target_daily"],
            vol_lookback=params["vol_lookback"],
            min_position_scale=params["min_position_scale"],
            max_position_scale=params["max_position_scale"],
            max_drawdown_soft=params["max_drawdown_soft"],
            drawdown_brake=params["drawdown_brake"],
        )
        return bt_df["strategy_return"]

    thr_grid = sorted(set([0.0, 0.0015, 0.0025, abs(float(cfg.BUY_THRESHOLD)), 0.0040, 0.0050]))
    hold_grid = sorted(set([
        max(1, int(cfg.FORECAST_HORIZON) - 2),
        int(cfg.FORECAST_HORIZON),
        int(cfg.FORECAST_HORIZON) + 2,
    ]))

    best_score = (-1e9, -1e9, -1e9)
    best_val_metrics = {}
    best_candidates = 0

    for thr in thr_grid:
        for hold_days in hold_grid:
            params = dict(tuned_params)
            params["buy_threshold"] = float(thr)
            params["sell_threshold"] = float(-thr)
            params["hold_period"] = int(hold_days)

            val_returns = _validation_portfolio_returns(params)
            if len(val_returns) < 30:
                continue

            m = compute_metrics(val_returns, label="")
            score = (
                float(m.get("sharpe", -1e9)),
                float(m.get("sortino", -1e9)),
                float(m.get("total_return", -1e9)),
            )
            best_candidates += 1

            if score > best_score:
                best_score = score
                tuned_params = params
                best_val_metrics = {
                    "sharpe": float(m.get("sharpe", 0.0)),
                    "sortino": float(m.get("sortino", 0.0)),
                    "max_drawdown": float(m.get("max_drawdown", 0.0)),
                    "total_return": float(m.get("total_return", 0.0)),
                    "cagr": float(m.get("cagr", 0.0)),
                    "hit_rate": float(m.get("hit_rate", 0.0)),
                    "rolling_sharpe_mean": float(m.get("rolling_sharpe_mean", 0.0)),
                }

    if best_candidates > 0:
        logger.info(
            "Tuned strategy params from validation grid search "
            f"(candidates={best_candidates}): buy={tuned_params['buy_threshold']:.4f}, "
            f"sell={tuned_params['sell_threshold']:.4f}, hold={tuned_params['hold_period']}"
        )
        logger.info(
            "Validation portfolio metrics with tuned params: "
            f"Sharpe={best_val_metrics.get('sharpe', 0.0):.3f}, "
            f"Sortino={best_val_metrics.get('sortino', 0.0):.3f}, "
            f"TotalRet={best_val_metrics.get('total_return', 0.0):.3f}"
        )
    else:
        logger.warning("Validation tuning skipped (insufficient validation backtest samples).")

    for ticker, res in train_results.items():
        raw_preds = np.asarray(res.get("preds", []), dtype=float)
        val_preds = np.asarray(res.get("val_preds", []), dtype=float)
        y_val = np.asarray(res.get("y_val", []), dtype=float)
        n_corr = min(len(y_val), len(val_preds))
        val_dir = float(np.mean(np.sign(y_val[:n_corr]) == np.sign(val_preds[:n_corr]))) if n_corr > 0 else 0.0
        val_corr = _safe_corr(y_val[:n_corr], val_preds[:n_corr]) if n_corr > 1 else 0.0

        all_predictions[ticker] = raw_preds
        all_prediction_dates[ticker] = pd.DatetimeIndex(pd.to_datetime(res.get("dates_te", [])))
        ticker_quality[ticker] = {
            "val_dir_acc": float(val_dir),
            "val_corr": float(val_corr),
            "score": float(0.8 * val_dir + 0.2 * max(0.0, val_corr)),
        }

        logger.info(
            f"[{ticker}] Using raw sign-based predictions (no threshold calibration), "
            f"selected_model={res.get('selected_model', 'unknown')}, "
            f"val_dir_acc={val_dir:.4f}, val_corr={val_corr:.4f}"
        )

    if not all_predictions:
        return {"portfolio_df": pd.DataFrame(), "per_asset": {}, "metrics": {}}

    selected_tickers = list(all_predictions.keys())
    if bool(getattr(cfg, "RISK_FILTER_ENABLE", False)):
        val_asset_returns = {}
        val_asset_metrics = {}

        for t in all_predictions.keys():
            sr = _validation_asset_returns(t, tuned_params)
            if len(sr) < 30:
                continue
            m = compute_metrics(sr, label="")
            if not m:
                continue
            val_asset_returns[t] = sr
            val_asset_metrics[t] = m

        if val_asset_returns:
            ranked = sorted(
                val_asset_returns.keys(),
                key=lambda t: (
                    float(val_asset_metrics[t].get("sharpe", -1e9)),
                    float(val_asset_metrics[t].get("sortino", -1e9)),
                    float(val_asset_metrics[t].get("total_return", -1e9)),
                ),
                reverse=True,
            )

            eligible = [
                t for t in ranked
                if ticker_quality.get(t, {}).get("val_dir_acc", 0.0) >= float(cfg.RISK_MIN_VAL_DIR_ACC)
                and ticker_quality.get(t, {}).get("val_corr", 0.0) >= float(cfg.RISK_MIN_VAL_CORR)
            ]
            if not eligible:
                eligible = ranked

            candidate_pool = eligible[: min(6, len(eligible))]
            max_k = max(1, min(int(getattr(cfg, "RISK_TOP_K", len(candidate_pool))), len(candidate_pool)))

            best_combo = None
            best_combo_score = (-1e9, -1e9, -1e9, -1e9)
            for k in range(1, max_k + 1):
                for combo in combinations(candidate_pool, k):
                    combo_df = pd.DataFrame({t: val_asset_returns[t] for t in combo})
                    combo_ret = aggregate_portfolio_returns(combo_df, strategy_params=tuned_params)
                    cm = compute_metrics(combo_ret, label="")
                    if not cm:
                        continue

                    combo_dir = float(np.mean([
                        ticker_quality.get(t, {}).get("val_dir_acc", 0.0) for t in combo
                    ]))
                    combo_corr = float(np.mean([
                        ticker_quality.get(t, {}).get("val_corr", 0.0) for t in combo
                    ]))
                    corr_score = float(np.clip((combo_corr + 0.05) / 0.15, 0.0, 1.0))
                    dir_score = float(np.clip((combo_dir - 0.48) / 0.10, 0.0, 1.0))
                    sharpe_score = float(np.clip((float(cm.get("sharpe", 0.0)) + 0.5) / 1.5, 0.0, 1.0))
                    data_score = 0.90
                    edge_proxy = 100.0 * (
                        0.35 * corr_score +
                        0.30 * dir_score +
                        0.25 * sharpe_score +
                        0.10 * data_score
                    )

                    score = (
                        float(edge_proxy),
                        float(cm.get("sharpe", -1e9)),
                        float(cm.get("sortino", -1e9)),
                        float(cm.get("total_return", -1e9)),
                    )
                    if score > best_combo_score:
                        best_combo_score = score
                        best_combo = list(combo)

            if best_combo:
                selected_tickers = best_combo

        logger.info(
            f"Risk filter active: selected tickers={selected_tickers} "
            f"from universe={list(all_predictions.keys())}"
        )

    predictions_dict = {t: all_predictions[t] for t in selected_tickers}
    prediction_dates_dict = {t: all_prediction_dates[t] for t in selected_tickers}

    portfolio_df, per_asset = backtest_portfolio(
        price_datasets,
        predictions_dict,
        prediction_dates_dict=prediction_dates_dict,
        strategy_params=tuned_params,
    )

    # Persist detailed backtest outputs for inspection.
    results_dir = cfg.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    if not portfolio_df.empty:
        portfolio_path = results_dir / "portfolio_backtest.csv"
        portfolio_df.to_csv(portfolio_path, index=True)
        logger.info(f"Saved → {portfolio_path}")

    for ticker, df in per_asset.items():
        if df is None or df.empty:
            continue
        out_path = results_dir / f"{ticker}_backtest.csv"
        df.to_csv(out_path, index=True)

        trade_turnover = int((df["position"].diff().abs().fillna(0) > 0).sum())
        active_days = int((df["position"].abs() > 0).sum())
        logger.info(
            f"[{ticker}] Backtest rows={len(df)} active_days={active_days} "
            f"position_changes={trade_turnover}"
        )
        logger.info(f"Saved → {out_path}")

    if not portfolio_df.empty:
        metrics = compute_metrics(portfolio_df["portfolio_return"], "Portfolio")
        logger.info(f"Portfolio metrics: {metrics}")
    else:
        metrics = {}

    return {
        "portfolio_df": portfolio_df,
        "per_asset": per_asset,
        "metrics": metrics,
        "strategy_params": tuned_params,
        "validation_metrics": best_val_metrics,
    }


def run_eval(
    price_datasets:   dict,
    merged_datasets:  dict,
    train_results:    dict,
    backtest_results: dict,
):
    """Phase 6: Export hypothesis-oriented diagnostics and plots."""
    import pandas as pd
    from utils.visualizations import (
        plot_price_history, plot_sentiment_timeline,
        plot_equity_curve, plot_predictions_vs_actual,
        plot_regime_timeline, plot_expert_utilization,
        plot_feature_correlation,
    )

    logger.info("═" * 60)
    logger.info("PHASE 6 — Evaluation & Visualisations")
    logger.info("═" * 60)

    def _news_impact_assessment(
        train_results: dict,
        backtest_results: dict,
        merged_datasets: dict,
    ) -> dict:
        """Assess whether news sentiment adds meaningful predictive power."""
        per_ticker = []
        corr_vals = []
        dir_vals = []
        mae_vals = []

        use_tradable_only = bool(getattr(cfg, "EVAL_USE_TRADABLE_ONLY", True))
        tradable_tickers = set(backtest_results.get("per_asset", {}).keys())

        for ticker, res in train_results.items():
            if use_tradable_only and tradable_tickers and ticker not in tradable_tickers:
                continue
            y_true = np.asarray(res.get("y_te", []), dtype=float)
            y_pred = np.asarray(res.get("preds", []), dtype=float)
            n = min(len(y_true), len(y_pred))
            if n < 2:
                continue

            yt = y_true[:n]
            yp = y_pred[:n]
            mae = float(np.mean(np.abs(yt - yp)))
            dir_acc = float(np.mean(np.sign(yt) == np.sign(yp)))

            if np.std(yt) < 1e-9 or np.std(yp) < 1e-9:
                corr = 0.0
            else:
                c = float(np.corrcoef(yt, yp)[0, 1])
                corr = c if np.isfinite(c) else 0.0

            corr_vals.append(corr)
            dir_vals.append(dir_acc)
            mae_vals.append(mae)
            per_ticker.append({
                "ticker": ticker,
                "n_test_samples": int(n),
                "corr": round(corr, 4),
                "dir_acc": round(dir_acc, 4),
                "mae": round(mae, 6),
                "selected_model": res.get("selected_model", "unknown"),
            })

        avg_corr = float(np.mean(corr_vals)) if corr_vals else 0.0
        avg_dir_acc = float(np.mean(dir_vals)) if dir_vals else 0.0
        avg_mae = float(np.mean(mae_vals)) if mae_vals else 0.0

        merged_rows = [len(df) for df in merged_datasets.values()] if merged_datasets else []
        avg_rows = float(np.mean(merged_rows)) if merged_rows else 0.0

        portfolio_df = backtest_results.get("portfolio_df", pd.DataFrame())
        if isinstance(portfolio_df, pd.DataFrame) and not portfolio_df.empty:
            pr = portfolio_df["portfolio_return"].dropna()
            if len(pr) > 1 and float(pr.std()) > 1e-9:
                sharpe_rf0 = float(pr.mean() / pr.std() * np.sqrt(252.0))
            else:
                sharpe_rf0 = 0.0
        else:
            sharpe_rf0 = 0.0

        # Predictive edge score (higher is better forecasting performance).
        corr_score = np.clip((avg_corr + 0.05) / 0.15, 0.0, 1.0)
        dir_score = np.clip((avg_dir_acc - 0.48) / 0.10, 0.0, 1.0)
        sharpe_score = np.clip((sharpe_rf0 + 0.5) / 1.5, 0.0, 1.0)
        data_score = np.clip(avg_rows / 1200.0, 0.0, 1.0)
        predictive_edge_score = float(
            100.0 * (0.35 * corr_score + 0.30 * dir_score + 0.25 * sharpe_score + 0.10 * data_score)
        )

        if predictive_edge_score >= 70:
            predictive_verdict = "strong-predictive-edge"
        elif predictive_edge_score >= 45:
            predictive_verdict = "moderate-edge"
        else:
            predictive_verdict = "weak-edge"

        # Hypothesis evidence score: higher means stronger support that
        # news sentiment has limited incremental impact in this setup.
        near_zero_corr = np.clip((0.08 - abs(avg_corr)) / 0.08, 0.0, 1.0)
        near_random_dir = np.clip((0.06 - abs(avg_dir_acc - 0.5)) / 0.06, 0.0, 1.0)
        non_positive_sharpe = np.clip((-sharpe_rf0 + 0.2) / 1.2, 0.0, 1.0)
        limited_impact_evidence_score = float(
            100.0 * (0.45 * near_zero_corr + 0.35 * near_random_dir + 0.20 * non_positive_sharpe)
        )

        if limited_impact_evidence_score >= 65:
            hypothesis_verdict = "supports_limited_news_impact"
        elif limited_impact_evidence_score >= 45:
            hypothesis_verdict = "partially_supports_limited_news_impact"
        else:
            hypothesis_verdict = "inconclusive"

        return {
            "objective": cfg.EXPERIMENT_OBJECTIVE,
            "hypothesis": cfg.EXPERIMENT_HYPOTHESIS,
            "predictive_edge_score": round(predictive_edge_score, 2),
            "predictive_verdict": predictive_verdict,
            "limited_impact_evidence_score": round(limited_impact_evidence_score, 2),
            "hypothesis_verdict": hypothesis_verdict,
            "summary": {
                "avg_test_corr": round(avg_corr, 4),
                "avg_test_dir_acc": round(avg_dir_acc, 4),
                "avg_test_mae": round(avg_mae, 6),
                "sharpe_rf0": round(sharpe_rf0, 4),
                "tickers_evaluated": int(len(per_ticker)),
                "avg_rows_per_ticker": round(avg_rows, 1),
                "use_finbert": bool(cfg.USE_FINBERT),
                "sentiment_source": "all-data.csv + lagged_ohlcv_proxy",
            },
            "per_ticker": per_ticker,
        }

    plot_price_history(price_datasets)

    for ticker in list(merged_datasets.keys())[:2]:
        plot_sentiment_timeline(merged_datasets, ticker,
                                cfg.RESULTS_DIR / f"{ticker}_sentiment.png")
        plot_feature_correlation(
            merged_datasets[ticker],
            cfg.PRICE_FEATURES[:10] + cfg.SENTIMENT_FEATURES[:6],
            cfg.RESULTS_DIR / f"{ticker}_correlation.png",
        )

    if backtest_results.get("per_asset"):
        plot_equity_curve(
            backtest_results["portfolio_df"],
            backtest_results["per_asset"],
        )

    for ticker, res in train_results.items():
        if "preds" not in res:
            continue

        dates_te = pd.to_datetime(res["dates_te"])
        plot_predictions_vs_actual(
            res["y_te"], res["preds"], dates_te, ticker,
            cfg.RESULTS_DIR / f"{ticker}_predictions.png",
        )

        if len(res.get("router_w", [])) > 0:
            plot_expert_utilization(
                res["router_w"], dates_te,
                cfg.RESULTS_DIR / f"{ticker}_expert_utilization.png",
            )

        if ticker in merged_datasets and "per_asset" in backtest_results:
            merged_df   = merged_datasets[ticker]
            test_merged = merged_df[merged_df.index >= cfg.TEST_START]
            n           = min(len(dates_te), len(test_merged))
            vol_test    = (
                test_merged["vol_20"].values[:n]
                if "vol_20" in test_merged else np.zeros(n)
            )
            from fusion.fusion_model import _regime_labels
            regime_actuals = _regime_labels(vol_test)
            regime_preds   = np.ones(n, dtype=int)

            equity_series = backtest_results["per_asset"].get(ticker, {})
            eq = (
                equity_series.get("equity", pd.Series(dtype=float))
                if hasattr(equity_series, "get")
                else pd.Series(dtype=float)
            )

            if len(eq) >= n:
                plot_regime_timeline(
                    dates_te[:n], regime_preds[:n], regime_actuals[:n],
                    eq.iloc[:n],
                    cfg.RESULTS_DIR / f"{ticker}_regime_timeline.png",
                )

    metrics_out = dict(backtest_results.get("metrics", {}))
    if backtest_results.get("strategy_params"):
        metrics_out["strategy_params"] = backtest_results["strategy_params"]
    if backtest_results.get("validation_metrics"):
        metrics_out["validation_metrics"] = backtest_results["validation_metrics"]

    assessment = _news_impact_assessment(
        train_results=train_results,
        backtest_results=backtest_results,
        merged_datasets=merged_datasets,
    )
    metrics_out["news_impact_report"] = assessment
    metrics_out["resource_report"] = assessment

    metrics_path = cfg.RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="News Impact Hypothesis Pipeline v3.0")
    parser.add_argument(
        "--mode",
        choices=["full", "data", "nlp", "train", "backtest", "eval"],
        default="full",
        help="Pipeline stage to run",
    )
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-download even if cache exists")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Override ticker list (e.g. --tickers AAPL MSFT)")
    return parser.parse_args()


def main():
    args = parse_args()
    _seed_everything()

    if args.tickers:
        cfg.TICKERS = args.tickers
        logger.info(f"Overriding tickers: {cfg.TICKERS}")

    state: dict = {}

    if args.mode in ("full", "data"):
        state.update(run_data(args))

    if args.mode in ("full", "nlp"):
        if "price" not in state or "news" not in state:
            logger.info("Loading cached data for NLP phase ...")
            state.update(run_data(args))
        nlp_out = run_nlp(state["price"], state["news"])
        state.update(nlp_out)

    if args.mode in ("full", "train"):
        if "merged" not in state:
            import pandas as pd
            merged = {}
            for ticker in cfg.TICKERS:
                p = cfg.DATA_DIR / f"{ticker}_sentiment_features.csv"
                if p.exists():
                    merged[ticker] = pd.read_csv(p, index_col="date", parse_dates=True)
                    logger.info(f"Loaded merged features for {ticker} from {p}")
            if not merged:
                logger.error("No merged datasets found. Run --mode nlp first.")
                sys.exit(1)
            state["merged"] = merged

        train_out = run_train(state["merged"], args)
        state["train_results"] = train_out

    if args.mode in ("full", "backtest"):
        if "price" not in state:
            state["price"] = build_price_dataset(use_cache=True)
        if "train_results" not in state:
            logger.error("No train results. Run --mode train first.")
            sys.exit(1)
        bt_out = run_backtest(state["price"], state["train_results"])
        state["backtest_results"] = bt_out

    if args.mode in ("full", "eval"):
        if "price" not in state:
            state["price"] = build_price_dataset(use_cache=True)
        if "merged" not in state or "train_results" not in state or \
                "backtest_results" not in state:
            logger.error("Eval requires price, merged, train_results, and backtest_results.")
            sys.exit(1)
        run_eval(state["price"], state["merged"],
                 state["train_results"], state["backtest_results"])

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()