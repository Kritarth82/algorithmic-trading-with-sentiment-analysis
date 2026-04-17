"""
Phase 2b — Sentiment Scoring (all-data.csv + OHLCV — NO API, NO FinBERT)

IMPORTANT: This module produces COMPUTED SENTIMENT PROXIES, not outputs
from a FinBERT transformer model.  The finbert_* feature names are kept
for downstream schema compatibility only.

Two scoring paths, blended together:

  PATH A — CSV-derived (primary, weight = 0.8)
    Aggregates the real per-sentence lexicon scores stored in aligned_df['scores']
    (pre-computed by PhraseBankLoader._lexicon_score).  Produces a compound score
    with natural variance from actual sentence text.

  PATH B — OHLCV-derived (anchor / fallback, weight = 0.2)
    OHLCVSentimentComputer derives compound from LAGGED (t-1) price signals:
      tanh(w_ret * norm_return + w_rsi * rsi_signal + w_bb * bb_pos + w_vol * vol_surge)
    Using lagged features avoids look-ahead bias: the sentiment estimate
    for day t is based on price action through day t-1.

  Blend rule (per trading day):
    IF day has real CSV scores  →  0.8 * csv_compound + 0.2 * ohlcv_compound
    IF day is null / fill-only  →  ohlcv_compound (100%)

All finbert_* features are derived from the final compound in a consistent way
so downstream LSTM/Fusion models see the same schema they always expected.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    FINBERT_MODEL, FINBERT_DEVICE, NLP_BATCH_SIZE,
    MAX_SEQ_LENGTH, MAX_TEXTS_PER_DAY, DATA_DIR,
    SENTIMENT_FEATURES, LABEL_SCORE_MAP, USE_FINBERT,
    OHLCV_SENT_W_RETURN, OHLCV_SENT_W_RSI,
    OHLCV_SENT_W_BB, OHLCV_SENT_W_VOL,
    SENT_CSV_BLEND_WEIGHT,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper — derive all finbert_* features from a single compound score
# ─────────────────────────────────────────────────────────────────────────────

def _features_from_compound(
    compound: float,
    n_texts: int,
    dispersion: float = 0.0,
) -> dict:
    """
    Convert a single compound score in (-1, 1) into the full set of
    finbert_* features expected by the downstream models.

    No model, no lookup — pure arithmetic.
    """
    c  = float(np.clip(compound, -1.0, 1.0))
    ac = abs(c)

    # Requested mapping from compound to finbert-style probabilities.
    pos = float(1.0 / (1.0 + math.exp(-c * 3.0)))
    neg = float(1.0 / (1.0 + math.exp(+c * 3.0)))
    neu = float(max(0.0, 1.0 - ac))

    confidence  = ac
    uncertainty = float(math.log(3.0) * (1.0 - ac))

    return {
        "finbert_positive":     pos,
        "finbert_negative":     neg,
        "finbert_neutral":      neu,
        "finbert_compound":     c,
        "finbert_confidence":   confidence,
        "finbert_uncertainty":  uncertainty,
        "news_volume":          n_texts,
        "sentiment_dispersion": dispersion,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PATH A — CSV-derived aggregator
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_csv_scores(scores: List[float], labels: List[str]) -> Optional[float]:
    """
    Aggregate per-sentence lexicon scores into a single compound value.

    Weights:
      - Sentences with more extreme scores (|score|) contribute more.
      - Falls back to simple mean if no weights available.

    Returns None if the list is empty.
    """
    if not scores:
        return None

    arr = np.array(scores, dtype=float)
    weights = np.abs(arr) + 0.05     # avoid zero weights on near-neutral sentences
    compound = float(np.average(arr, weights=weights))
    return float(np.clip(compound, -1.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# PATH B — OHLCV-derived sentiment computer
# ─────────────────────────────────────────────────────────────────────────────

class OHLCVSentimentComputer:
    """
    Derive a sentiment compound score from LAGGED (t-1) OHLCV price features.

    Using the previous day's price action avoids look-ahead bias: the
    sentiment proxy for day t only uses information available through t-1.

    Formula (applied inside tanh so result is in (-1, 1)):
        signal = w_ret * (lagged_return / vol_baseline)
               + w_rsi * rsi_signal       # (RSI-14 - 50) / 50
               + w_bb  * bb_position      # (close - bb_lower) / (bb_upper - bb_lower) * 2 - 1
               + w_vol * volume_surge     # log(volume / volume_ma20)

    Requires columns (subset of PRICE_FEATURES):
        returns, rsi_14, bb_upper, bb_lower, close, volume, vol_20
    Falls back gracefully if columns are absent.
    """

    def __init__(
        self,
        w_return: float = OHLCV_SENT_W_RETURN,
        w_rsi:    float = OHLCV_SENT_W_RSI,
        w_bb:     float = OHLCV_SENT_W_BB,
        w_vol:    float = OHLCV_SENT_W_VOL,
    ):
        self.w_return = w_return
        self.w_rsi    = w_rsi
        self.w_bb     = w_bb
        self.w_vol    = w_vol

    def compute(self, price_row: pd.Series) -> float:
        """
        Compute compound sentiment from one LAGGED day's OHLCV row.

        Parameters
        ----------
        price_row : pd.Series  — the PREVIOUS day's row from the price DataFrame
                                 (already computed technical indicators).

        Returns
        -------
        float in (-1, 1).
        """
        signal = 0.0

        # --- Return signal ---------------------------------------------------
        ret = self._get(price_row, ["returns", "log_returns"], 0.0)
        vol_baseline = max(self._get(price_row, ["vol_20"], 0.015), 1e-4)
        signal += self.w_return * (ret / vol_baseline)

        # --- RSI signal ------------------------------------------------------
        rsi = self._get(price_row, ["rsi_14"], 50.0)
        rsi_signal = (rsi - 50.0) / 50.0   # maps [0,100] to [-1, 1]
        signal += self.w_rsi * rsi_signal

        # --- Bollinger Band position -----------------------------------------
        bb_upper = self._get(price_row, ["bb_upper"], None)
        bb_lower = self._get(price_row, ["bb_lower"], None)
        close    = self._get(price_row, ["close"],    None)
        if bb_upper is not None and bb_lower is not None and close is not None:
            band_width = bb_upper - bb_lower
            if band_width > 1e-6:
                bb_pos = ((close - bb_lower) / band_width) * 2.0 - 1.0  # [-1, 1]
                signal += self.w_bb * float(np.clip(bb_pos, -1.0, 1.0))

        # --- Volume surge ----------------------------------------------------
        volume = self._get(price_row, ["volume"], None)
        vol_ma20 = self._get(price_row, ["volume_ma20", "vol_ma20"], None)
        if volume is not None and vol_ma20 is not None and volume > 0 and vol_ma20 > 0:
            vol_surge = math.log(max(volume / vol_ma20, 1e-6))
            signal += self.w_vol * vol_surge

        return float(np.tanh(signal))

    @staticmethod
    def _get(row: pd.Series, keys: List[str], default):
        """Try multiple column names; return first found value or default."""
        for k in keys:
            if k in row.index:
                v = row[k]
                if pd.notna(v):
                    return float(v)
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Legacy stubs — kept for import compatibility (no longer used in the pipeline)
# ─────────────────────────────────────────────────────────────────────────────

_LABEL_PROBS = {
    "positive": {"positive": 0.85, "negative": 0.08, "neutral": 0.07, "compound": +0.85},
    "negative": {"positive": 0.07, "negative": 0.85, "neutral": 0.08, "compound": -0.85},
    "neutral":  {"positive": 0.10, "negative": 0.10, "neutral": 0.80, "compound":  0.00},
}


def scores_from_labels(
    texts_with_labels: List[dict],
    label_map: dict = LABEL_SCORE_MAP,
) -> pd.DataFrame:
    """Legacy stub — no longer used; kept for backward compatibility."""
    rows = []
    for item in texts_with_labels:
        label = str(item.get("label", "neutral")).lower()
        probs = _LABEL_PROBS.get(label, _LABEL_PROBS["neutral"])
        rows.append({
            "positive": probs["positive"],
            "negative": probs["negative"],
            "neutral":  probs["neutral"],
        })
    return pd.DataFrame(rows, columns=["positive", "negative", "neutral"])


class FinBERTScorer:
    """
    Legacy stub — FinBERT is intentionally disabled.
    Raises RuntimeError if instantiated so callers notice quickly.
    """
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "FinBERT is disabled. Set USE_FINBERT=False (already done in config.py). "
            "The pipeline now uses all-data.csv + OHLCV sentiment only."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Topic relevance (self-contained)
# ─────────────────────────────────────────────────────────────────────────────

_TICKER_TO_COMPANY = {
    "AAPL":        "Apple",
    "TSLA":        "Tesla",
    "MSFT":        "Microsoft",
    "NVDA":        "NVIDIA",
    "AMZN":        "Amazon",
    "META":        "Meta",
    "GOOGL":       "Google",
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS":      "TCS",
}


def _topic_relevance(
    texts: List[str],
    ticker: str,
    labels: List[str] | None = None,
    sources: List[str] | None = None,
) -> float:
    company  = _TICKER_TO_COMPANY.get(ticker, ticker)
    keywords = {ticker.lower(), ticker.split(".")[0].lower(), company.lower()}
    labels  = labels or []
    sources = sources or []

    if not texts:
        return 0.0

    mention_ratio = sum(
        1 for t in texts if any(kw in t.lower() for kw in keywords)
    ) / len(texts)

    if sources:
        src_scores = [
            0.85 if s == "synthetic" else 0.55 if s == "synthetic_fill" else 0.7
            for s in sources
        ]
        source_quality = float(np.mean(src_scores))
    else:
        source_quality = 0.5

    label_quality = 1.0 if labels and len(labels) == len(texts) else 0.6
    score = 0.6 * mention_ratio + 0.3 * source_quality + 0.1 * label_quality
    return float(np.clip(score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Shared aggregation helper (legacy — kept for any external imports)
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_scores_df(scores: pd.DataFrame, n_texts: int) -> dict:
    """Legacy helper — aggregates a (N,3) scores DataFrame. Kept for compat."""
    pos = scores["positive"].values
    neg = scores["negative"].values
    neu = scores["neutral"].values
    compound_per   = pos - neg
    confidence_per = np.max(scores.values, axis=1)
    eps    = 1e-9
    entropy = -(
        pos * np.log(pos + eps) +
        neg * np.log(neg + eps) +
        neu * np.log(neu + eps)
    )
    return {
        "finbert_positive":     float(pos.mean()),
        "finbert_negative":     float(neg.mean()),
        "finbert_neutral":      float(neu.mean()),
        "finbert_compound":     float(compound_per.mean()),
        "finbert_confidence":   float(confidence_per.mean()),
        "finbert_uncertainty":  float(entropy.mean()),
        "news_volume":          n_texts,
        "sentiment_dispersion": float(compound_per.std()) if n_texts > 1 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# build_full_sentiment_features  (main entry point)
# ─────────────────────────────────────────────────────────────────────────────

def build_full_sentiment_features(
    news_aligned: Dict[str, pd.DataFrame],
    price_data:   Dict[str, pd.DataFrame],
    scorer=None,                                    # ignored (FinBERT disabled)
) -> Dict[str, pd.DataFrame]:
    """
    Build daily sentiment feature vectors for each ticker.

    Scoring pipeline per trading day
    ─────────────────────────────────
    1. PATH A — CSV scores:
       Pull pre-computed per-sentence lexicon scores from aligned_df['scores'].
       Aggregate via weighted mean → csv_compound.

    2. PATH B — OHLCV compound:
       Always computed (cheap arithmetic on price_df row).

    3. Blend:
       good_day  (has real CSV scores):  0.7 * csv_compound + 0.3 * ohlcv_compound
       null_day  (no/fill-only scores):  ohlcv_compound

    4. Derive finbert_* features from final compound (arithmetic, not model).
    5. Add temporal rolling features (ma3, ma7, momentum, volatility).
    6. Merge with price_df and save.
    """
    if scorer is not None:
        logger.warning(
            "scorer argument is ignored — FinBERT is disabled. "
            "Using all-data.csv + OHLCV sentiment instead."
        )

    ohlcv_computer  = OHLCVSentimentComputer()
    merged_datasets: Dict[str, pd.DataFrame] = {}

    for ticker, aligned_df in news_aligned.items():
        logger.info(f"Scoring sentiment for {ticker} ...")
        price_df = price_data[ticker]
        price_df_ext = price_df.copy()
        if "volume" in price_df_ext.columns:
            price_df_ext["volume_ma20"] = price_df_ext["volume"].rolling(20, min_periods=1).mean()

        rows: List[dict] = []

        for date, row in aligned_df.iterrows():
            texts   = row["texts"]   if isinstance(row.get("texts"),   list) else []
            labels  = row["labels"]  if isinstance(row.get("labels"),  list) else []
            sources = row["sources"] if isinstance(row.get("sources"), list) else []
            scores  = row["scores"]  if isinstance(row.get("scores"),  list) else []

            texts = [t for t in texts if isinstance(t, str) and t.strip()]

            # ── PATH B: OHLCV compound (always computed from LAGGED t-1 row) ─
            if date in price_df_ext.index:
                date_loc = price_df_ext.index.get_loc(date)
                if date_loc > 0:
                    prev_row = price_df_ext.iloc[date_loc - 1]
                    ohlcv_compound = ohlcv_computer.compute(prev_row)
                else:
                    ohlcv_compound = 0.0
            else:
                ohlcv_compound = 0.0

            # ── PATH A: CSV compound ──────────────────────────────────────
            real_scores = [
                float(s)
                for s in scores
                if isinstance(s, (int, float, np.number)) and pd.notna(s)
            ]
            is_fill_only = all(src == "synthetic_fill" for src in sources) if sources else False
            no_articles = len(texts) == 0
            csv_compound = _aggregate_csv_scores(real_scores, labels) if real_scores else None
            csv_invalid = (csv_compound is None) or (not np.isfinite(csv_compound))

            # ── Blend ─────────────────────────────────────────────────────
            if (not csv_invalid) and (not is_fill_only) and (not no_articles):
                w = SENT_CSV_BLEND_WEIGHT
                final_compound = w * csv_compound + (1.0 - w) * ohlcv_compound
            else:
                # Null or fill-only day: trust OHLCV fully
                final_compound = ohlcv_compound

            # ── Dispersion (variance across per-sentence scores) ──────────
            if len(real_scores) > 1:
                dispersion = float(np.std(real_scores))
            else:
                dispersion = 0.0

            # ── Build feature dict ────────────────────────────────────────
            daily = _features_from_compound(
                compound   = final_compound,
                n_texts    = len(texts),
                dispersion = dispersion,
            )

            text_relevance = _topic_relevance(
                texts=texts, ticker=ticker, labels=labels, sources=sources,
            )

            # Relevance blends topic signal with score quality
            conf = float(np.clip(daily.get("finbert_confidence", 0.0), 0.0, 1.0))
            unc  = float(np.clip(
                daily.get("finbert_uncertainty", math.log(3)) / math.log(3), 0.0, 1.0
            ))
            disp = float(np.clip(daily.get("sentiment_dispersion", 0.0), 0.0, 1.0))
            quality   = 0.6 * conf + 0.4 * (1.0 - unc)
            stability = 1.0 - 0.35 * disp
            daily["topic_relevance_score"] = float(
                np.clip(text_relevance * quality * stability, 0.0, 1.0)
            )
            daily["date"] = date
            rows.append(daily)

        sent_df = pd.DataFrame(rows).set_index("date")
        sent_df.index = pd.to_datetime(sent_df.index)
        sent_df       = sent_df.reindex(price_df.index)

        # ── Temporal rolling features ──────────────────────────────────────
        c = sent_df["finbert_compound"]
        sent_df["sentiment_ma3"]        = c.rolling(3,  min_periods=1).mean()
        sent_df["sentiment_ma7"]        = c.rolling(7,  min_periods=1).mean()
        sent_df["sentiment_momentum"]   = c - sent_df["sentiment_ma7"]
        sent_df["sentiment_volatility"] = c.rolling(5,  min_periods=1).std().fillna(0)

        sent_df.ffill(inplace=True)
        sent_df.fillna(0, inplace=True)

        # ── Merge with price data ──────────────────────────────────────────
        merged = price_df.copy()
        for col in SENTIMENT_FEATURES:
            merged[col] = sent_df[col] if col in sent_df.columns else 0.0

        out_path = DATA_DIR / f"{ticker}_sentiment_features.csv"
        merged.to_csv(out_path)
        logger.info(f"Saved merged features → {out_path}")

        merged_datasets[ticker] = merged

    return merged_datasets