"""
Phase 2 — News Proxy Dataset Construction (all-data.csv ONLY — no API, no FinBERT)

Changes from previous version:
  1. PhraseBankLoader  — now pre-computes a real per-sentence `score` using a
     financial lexicon so each sentence has a unique numeric value, not just a
     flat label bucket.  Variance within the positive/negative/neutral pools is
     what gives the LSTM meaningful signal.
  2. PerformanceBasedGenerator — passes the pre-computed sentence scores through
     in the returned article dicts (key: "score") so sentiment_analyzer.py can
     use them directly instead of a hardcoded lookup.
  3. No external API calls anywhere in this module.

Output schema:
    aligned DataFrame with columns 'texts', 'labels', 'sources', 'scores'
    indexed by trading date for downstream hypothesis evaluation.
"""

from __future__ import annotations

import logging
import math
import random
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    FINANCIAL_PHRASE_CSV,
    SYNTH_ARTICLES_PER_DAY,
    SYNTH_STRONG_RETURN_THR,
    SYNTH_MILD_RETURN_THR,
    LABEL_SCORE_MAP,
    CACHE_DIR,
    TRAIN_START,
    TEST_END,
    TICKERS,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


def _parse_cached_list(value: Any) -> list[Any]:
    """Parse cached list-like payloads from CSV safely."""
    if not isinstance(value, str):
        return []

    import ast

    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else []


def _list_len(value: Any) -> int:
    """Type-safe list length helper for cached payload checks."""
    return len(value) if isinstance(value, list) else 0

# Company name map (kept for topic_relevance in sentiment_analyzer)
TICKER_TO_COMPANY = {
    "AAPL":        "Apple",
    "TSLA":        "Tesla",
    "MSFT":        "Microsoft",
    "NVDA":        "NVIDIA",
    "AMZN":        "Amazon",
    "META":        "Meta",
    "GOOGL":       "Google",
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "TCS"
}


# ─────────────────────────────────────────────────────────────────────────────
# Financial lexicon — positive & negative word sets
# Used by PhraseBankLoader to compute a real per-sentence score with variance.
# ─────────────────────────────────────────────────────────────────────────────

_POS_WORDS = frozenset({
    "profit", "growth", "revenue", "increase", "gain", "rise", "rising",
    "record", "strong", "upgrade", "beat", "exceeded", "improved", "surged",
    "outperform", "recovery", "expansion", "robust", "accelerating", "positive",
    "higher", "boost", "breakthrough", "confident", "momentum", "soared",
    "record-high", "opportunity", "upside", "efficient", "margin", "demand",
    "return", "dividend", "surplus", "innovation", "agreement", "partnership",
    "raised", "hire", "hiring", "acquisition", "win", "winning", "benefited",
})

_NEG_WORDS = frozenset({
    "loss", "decline", "decrease", "fell", "falling", "drop", "dropped",
    "miss", "missed", "weak", "downgrade", "warning", "shortfall", "deficit",
    "layoff", "layoffs", "restructuring", "lawsuit", "recall", "fine", "penalty",
    "debt", "risk", "concern", "disappointing", "cut", "cuts", "reduced",
    "slowdown", "slower", "contraction", "bankruptcy", "default", "lower",
    "negative", "uncertainty", "pressure", "lost", "losses", "impairment",
    "write-off", "writedown", "liquidation", "sell-off", "downside", "missed",
})


def _lexicon_score(text: str, base_label: str) -> float:
    """
    Compute a numeric sentiment score for a sentence.

    Method:
      - Count positive & negative lexicon hits (normalised by text length)
      - The raw score = (pos_hits - neg_hits) / sqrt(n_words)
      - Result is clipped to [-1, 1]

    The raw lexicon score naturally carries the correct direction:
      - Text with more positive words → positive score
      - Text with more negative words → negative score
      - Balanced text → near-zero score

    Returns a float in [-1.0, +1.0].
    """
    # Backward-compatible fallback path used when TF-IDF weights are not provided.
    words = [w.rstrip(".,;:!?") for w in text.lower().split()]
    n = max(len(words), 1)

    pos_hits = sum(1 for w in words if w in _POS_WORDS)
    neg_hits = sum(1 for w in words if w in _NEG_WORDS)
    lexicon_score = (pos_hits - neg_hits) / math.sqrt(n)

    return float(np.clip(lexicon_score, -1.0, 1.0))


def _tokenize(text: str) -> List[str]:
    """Lightweight tokenizer for lexicon matching and TF-IDF stats."""
    return re.findall(r"[a-zA-Z][a-zA-Z\-]*", text.lower())


def _build_lexicon_idf(texts: pd.Series) -> Dict[str, float]:
    """
    Build IDF weights for financial lexicon terms over the phrasebank corpus.
    """
    lexicon_terms = _POS_WORDS | _NEG_WORDS
    n_docs = int(len(texts))
    doc_freq = {term: 0 for term in lexicon_terms}

    for text in texts.fillna(""):
        unique_terms = set(_tokenize(str(text)))
        for term in lexicon_terms:
            if term in unique_terms:
                doc_freq[term] += 1

    # Smoothed IDF: log((1 + N) / (1 + df)) + 1
    return {
        term: float(math.log((1.0 + n_docs) / (1.0 + df)) + 1.0)
        for term, df in doc_freq.items()
    }


def _tfidf_lexicon_score(text: str, base_label: str, idf_map: Dict[str, float]) -> float:
    """
    TF-IDF + financial lexicon weighted score for one sentence.

    Score rule:
        lexicon_score = (sum(tfidf_pos) - sum(tfidf_neg)) / sqrt(n_words)
        score = clip(lexicon_score, -1, 1)

    The raw lexicon score naturally carries the correct direction:
      - Text with more positive words → positive score
      - Text with more negative words → negative score
    """
    tokens = _tokenize(text)
    n_words = max(len(tokens), 1)
    if not tokens:
        return 0.0

    tf: Dict[str, int] = {}
    for tok in tokens:
        tf[tok] = tf.get(tok, 0) + 1

    pos_sum = 0.0
    neg_sum = 0.0
    for term, count in tf.items():
        if term in _POS_WORDS:
            pos_sum += count * idf_map.get(term, 1.0)
        elif term in _NEG_WORDS:
            neg_sum += count * idf_map.get(term, 1.0)

    lexicon_score = (pos_sum - neg_sum) / math.sqrt(n_words)
    return float(np.clip(lexicon_score, -1.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# PhraseBankLoader
# ─────────────────────────────────────────────────────────────────────────────

class PhraseBankLoader:
    """
    Loads all-data.csv (FinancialPhraseBank) and exposes per-label text pools.

    Additions vs previous version:
      - Each entry now stores a real `score` (float in [-1, 1]) computed via
        the financial lexicon above.  Sampling via `.sample_with_scores()` returns
        (text, score) pairs — giving the downstream scorer varied, meaningful
        compound values rather than identical fixed numbers per label bucket.

    CSV format (no header):
        column 0 → label  : "positive" | "negative" | "neutral"
        column 1 → text   : financial sentence
    Encoding: latin-1
    """

    def __init__(self, csv_path=FINANCIAL_PHRASE_CSV):
        self.csv_path = csv_path
        self.df: Optional[pd.DataFrame] = None
        # Each pool entry is a tuple (text, score)
        self.pos_pool: List[tuple] = []
        self.neg_pool: List[tuple] = []
        self.neu_pool: List[tuple] = []
        self._rng = random.Random(RANDOM_SEED)

    @staticmethod
    def _pool_for_label(label: str, pos_pool: List[tuple], neg_pool: List[tuple], neu_pool: List[tuple]) -> List[tuple]:
        return {
            "positive": pos_pool,
            "negative": neg_pool,
            "neutral": neu_pool,
        }.get(label, neu_pool)

    def load(self) -> "PhraseBankLoader":
        """Load CSV, compute per-sentence scores, partition into label pools."""
        logger.info(f"Loading FinancialPhraseBank from {self.csv_path}")
        self.df = pd.read_csv(
            self.csv_path,
            encoding="latin-1",
            header=None,
            names=["label", "text"],
        )
        self.df["label"] = self.df["label"].str.strip().str.lower()
        self.df["text"]  = self.df["text"].str.strip()
        self.df.dropna(subset=["text"], inplace=True)

        # Compute corpus-level IDF for financial lexicon terms then per-sentence scores.
        idf_map = _build_lexicon_idf(self.df["text"])

        # Compute real per-sentence score (TF-IDF + lexicon + label sign)
        self.df["score"] = self.df.apply(
            lambda row: _tfidf_lexicon_score(row["text"], row["label"], idf_map),
            axis=1,
        )

        self.pos_pool = list(
            zip(
                self.df[self.df["label"] == "positive"]["text"].tolist(),
                self.df[self.df["label"] == "positive"]["score"].tolist(),
            )
        )
        self.neg_pool = list(
            zip(
                self.df[self.df["label"] == "negative"]["text"].tolist(),
                self.df[self.df["label"] == "negative"]["score"].tolist(),
            )
        )
        self.neu_pool = list(
            zip(
                self.df[self.df["label"] == "neutral"]["text"].tolist(),
                self.df[self.df["label"] == "neutral"]["score"].tolist(),
            )
        )

        logger.info(
            f"PhraseBankLoader ready — "
            f"pos={len(self.pos_pool)}, neg={len(self.neg_pool)}, neu={len(self.neu_pool)}"
        )
        return self

    def sample(self, label: str, n: int) -> List[str]:
        """Sample n texts (with replacement) — text only (backward compat.)."""
        return [text for text, _ in self.sample_with_scores(label, n)]

    def sample_with_scores(self, label: str, n: int) -> List[tuple]:
        """Sample n (text, score) pairs (with replacement) from label pool."""
        pool = self._pool_for_label(label, self.pos_pool, self.neg_pool, self.neu_pool)

        if not pool:
            logger.warning(f"Empty pool for label '{label}', falling back to neutral.")
            pool = self.neu_pool or [("No relevant news available.", 0.0)]

        return self._rng.choices(pool, k=n)

    def sample_for_ticker(self, label: str, ticker: str, n: int) -> List[tuple]:
        """
        Sample n (text, score) pairs for a ticker.

        Preference order:
          1) Phrases of requested label that already mention ticker/company.
          2) Any phrase of requested label as fallback.
        """
        pool = self._pool_for_label(label, self.pos_pool, self.neg_pool, self.neu_pool)
        if not pool:
            pool = self.neu_pool or [("No relevant news available.", 0.0)]

        company = TICKER_TO_COMPANY.get(ticker, ticker)
        keys = {
            ticker.lower(),
            ticker.split(".")[0].lower(),
            company.lower(),
        }

        matched = []
        for text, score in pool:
            text_l = str(text).lower()
            if any(k and k in text_l for k in keys):
                matched.append((text, score))

        all_terms = set()
        for tk, comp in TICKER_TO_COMPANY.items():
            all_terms.add(tk.lower())
            all_terms.add(tk.split(".")[0].lower())
            all_terms.add(comp.lower())

        generic = []
        for text, score in pool:
            text_l = str(text).lower()
            if not any(term and term in text_l for term in all_terms):
                generic.append((text, score))

        if matched:
            return self._rng.choices(matched, k=n)

        if generic:
            return self._rng.choices(generic, k=n)

        return self._rng.choices(pool, k=n)

    def get_all(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("Call .load() before .get_all()")
        return self.df


# ─────────────────────────────────────────────────────────────────────────────
# PerformanceBasedGenerator
# ─────────────────────────────────────────────────────────────────────────────

class PerformanceBasedGenerator:
    """
    Given a day's OHLCV return, samples FinancialPhraseBank sentences whose
    sentiment tone matches the market performance.

    Each returned article dict now includes a 'score' key — the per-sentence
    lexicon score from PhraseBankLoader — so the downstream scorer can aggregate
    real numeric values (not flat label buckets).

    Tone selection:
        return >  STRONG_THR  → 3 positive
        return >  MILD_THR    → 2 positive + 1 neutral
        return < -STRONG_THR  → 3 negative
        return < -MILD_THR    → 2 negative + 1 neutral
        otherwise             → 3 neutral
    """

    def __init__(
        self,
        loader: PhraseBankLoader,
        strong_thr: float = SYNTH_STRONG_RETURN_THR,
        mild_thr:   float = SYNTH_MILD_RETURN_THR,
        n_per_day:  int   = SYNTH_ARTICLES_PER_DAY,
    ):
        self.loader     = loader
        self.strong_thr = strong_thr
        self.mild_thr   = mild_thr
        self.n_per_day  = n_per_day
        self._rng       = random.Random(RANDOM_SEED)

    def _tone_plan(self, daily_return: float) -> List[str]:
        """Return a list of labels (one per article) based on lagged daily_return.

        Uses probabilistic sampling rather than hard thresholds to produce
        more realistic (varied) sentiment distributions.
        """
        rng = self._rng

        # Base probabilities shift with return magnitude
        if daily_return > self.strong_thr:
            weights = {"positive": 0.75, "neutral": 0.15, "negative": 0.10}
        elif daily_return > self.mild_thr:
            weights = {"positive": 0.55, "neutral": 0.30, "negative": 0.15}
        elif daily_return < -self.strong_thr:
            weights = {"positive": 0.10, "neutral": 0.15, "negative": 0.75}
        elif daily_return < -self.mild_thr:
            weights = {"positive": 0.15, "neutral": 0.30, "negative": 0.55}
        else:
            weights = {"positive": 0.20, "neutral": 0.60, "negative": 0.20}

        labels_pool = list(weights.keys())
        probs = [weights[l] for l in labels_pool]
        return rng.choices(labels_pool, weights=probs, k=self.n_per_day)

    def generate(
        self,
        date: pd.Timestamp,
        ticker: str,
        prev_day_return: float,
    ) -> List[dict]:
        """
        Generate articles for one trading day based on the PREVIOUS day's return.

        Using lagged (t-1) returns avoids look-ahead bias: the sentiment
        assigned to a day reflects information that was already public
        (yesterday's close), not the day being predicted.

        Parameters
        ----------
        date             : trading date for which articles are generated
        ticker           : e.g. "AAPL"
        prev_day_return  : previous day's close-to-close return

        Returns
        -------
        List of dicts with keys: date, text, label, score, source, ticker
        """
        labels  = self._tone_plan(prev_day_return)
        company = TICKER_TO_COMPANY.get(ticker, ticker)
        articles = []
        for label in labels:
            pairs = self.loader.sample_for_ticker(label, ticker, 1)
            base_text, sent_score = pairs[0]

            base_l = str(base_text).lower()
            keys = {
                ticker.lower(),
                ticker.split(".")[0].lower(),
                company.lower(),
            }
            if any(k and k in base_l for k in keys):
                text = str(base_text)
            else:
                text = f"{company} ({ticker}): {base_text}"

            articles.append({
                "date":   date.strftime("%Y-%m-%d"),
                "text":   text,
                "label":  label,
                "score":  sent_score,      # real lexicon score
                "source": "synthetic",
                "ticker": ticker,
            })
        return articles


# ─────────────────────────────────────────────────────────────────────────────
# align_news_to_trading_days  (kept for compatibility with sentiment_analyzer)
# ─────────────────────────────────────────────────────────────────────────────

def align_news_to_trading_days(
    news_df: pd.DataFrame,
    trading_idx: pd.DatetimeIndex,
    fill_empty: bool = True,
) -> pd.DataFrame:
    """
    Group article payload by trading date.
    Returns DataFrame indexed by trading_idx with columns:
        - texts: List[str]
        - labels: List[str]
        - scores: List[float]   ← new column (real per-sentence scores)
        - sources: List[str]
    """
    news_df = news_df.copy()
    news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")
    news_df.dropna(subset=["date"], inplace=True)

    def _next_trading_day(d: pd.Timestamp):
        candidates = trading_idx[trading_idx >= d]
        return candidates[0] if len(candidates) > 0 else None

    news_df["trading_date"] = news_df["date"].apply(_next_trading_day)
    news_df.dropna(subset=["trading_date"], inplace=True)

    grouped_texts   = news_df.groupby("trading_date")["text"].apply(list)
    grouped_labels  = news_df.groupby("trading_date")["label"].apply(list)
    grouped_sources = news_df.groupby("trading_date")["source"].apply(list)

    # Aggregate real scores per day
    grouped_scores = (
        news_df.groupby("trading_date")["score"].apply(list)
        if "score" in news_df.columns
        else None
    )

    aligned = pd.DataFrame(index=trading_idx)
    aligned.index.name = "date"
    aligned["texts"]   = grouped_texts.reindex(trading_idx)
    aligned["labels"]  = grouped_labels.reindex(trading_idx)
    aligned["sources"] = grouped_sources.reindex(trading_idx)
    if grouped_scores is not None:
        aligned["scores"] = grouped_scores.reindex(trading_idx)
    else:
        aligned["scores"] = np.nan

    if fill_empty:
        for col in ["texts", "labels", "sources"]:
            aligned[col] = aligned[col].apply(
                lambda x: x if isinstance(x, list) else []
            )
        aligned["scores"] = aligned["scores"].apply(
            lambda x: x if isinstance(x, list) else []
        )

    return aligned


# ─────────────────────────────────────────────────────────────────────────────
# build_sentiment_dataset — main entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_sentiment_dataset(
    price_datasets: Dict[str, pd.DataFrame],
    from_date: str = TRAIN_START,
    to_date:   str = TEST_END,
) -> Dict[str, pd.DataFrame]:
    """
    For every ticker, produce a daily-aligned DataFrame with columns
    'texts', 'labels', 'scores', 'sources'.

    Strategy:
      - For trading days that exist in price_datasets[ticker]:
          use PerformanceBasedGenerator (tone driven by daily OHLCV return,
          scores are real lexicon values from PhraseBankLoader)
      - For business days with no price data:
          fill with neutral sentences + their lexicon scores

    Returns
    -------
    dict[ticker → aligned DataFrame]  (extended schema — includes 'scores')
    """
    loader    = PhraseBankLoader().load()
    generator = PerformanceBasedGenerator(loader)

    aligned_datasets: Dict[str, pd.DataFrame] = {}

    # Use runtime universe from available price datasets so CLI --tickers override works.
    tickers = list(price_datasets.keys()) if price_datasets else list(TICKERS)

    for ticker in tickers:
        price_df   = price_datasets.get(ticker)
        cache_path = CACHE_DIR / f"{ticker}_news_aligned.csv"

        # ── Try loading from cache ────────────────────────────────────────
        if cache_path.exists():
            logger.info(f"[{ticker}] Loading aligned news from cache: {cache_path}")
            cached = pd.read_csv(cache_path, index_col="date", parse_dates=True)

            for col in ["texts", "labels", "sources", "scores"]:
                if col in cached.columns:
                    cached[col] = cached[col].apply(_parse_cached_list)
                else:
                    cached[col] = [[] for _ in range(len(cached))]

            has_metadata = (
                cached["labels"].map(_list_len).sum() > 0
                and cached["sources"].map(_list_len).sum() > 0
            )
            has_scores = (
                cached["scores"]
                .map(
                    lambda x: isinstance(x, list)
                    and len(x) > 0
                    and any(isinstance(s, (int, float)) for s in x)
                )
                .sum()
                > 0
            )

            if has_metadata and has_scores:
                aligned_datasets[ticker] = cached
                continue

            logger.info(
                f"[{ticker}] Cache missing scores or metadata; rebuilding sentiment cache."
            )

        # ── Build from scratch ────────────────────────────────────────────
        logger.info(f"[{ticker}] Generating sentiment articles ...")
        biz_days = pd.date_range(from_date, to_date, freq="B")

        # Pre-compute lagged (t-1) daily returns for tone selection.
        # This avoids look-ahead bias: tone is based on *yesterday's* return,
        # not the day being predicted.
        if price_df is not None and "close" in price_df.columns:
            daily_returns = price_df["close"].pct_change().fillna(0.0)
        else:
            daily_returns = pd.Series(0.0, index=biz_days)

        all_articles: List[dict] = []

        for date in biz_days:
            if price_df is not None and date in price_df.index:
                # Get PREVIOUS trading day's return (lagged t-1)
                date_loc = price_df.index.get_loc(date)
                if not isinstance(date_loc, (int, np.integer)):
                    prev_return = 0.0
                elif date_loc > 0:
                    prev_return = float(daily_returns.iloc[date_loc - 1])
                else:
                    prev_return = 0.0

                articles = generator.generate(date, ticker, prev_return)
            else:
                # No price data → neutral synthetic fill (with real lexicon scores)
                company = TICKER_TO_COMPANY.get(ticker, ticker)
                pairs   = loader.sample_for_ticker("neutral", ticker, SYNTH_ARTICLES_PER_DAY)
                articles = []
                for t, s in pairs:
                    t_str = str(t)
                    t_l = t_str.lower()
                    keys = {
                        ticker.lower(),
                        ticker.split(".")[0].lower(),
                        company.lower(),
                    }
                    if any(k and k in t_l for k in keys):
                        text = t_str
                    else:
                        text = f"{company} ({ticker}): {t_str}"

                    articles.append(
                        {
                            "date":   date.strftime("%Y-%m-%d"),
                            "text":   text,
                            "label":  "neutral",
                            "score":  s,
                            "source": "synthetic_fill",
                            "ticker": ticker,
                        }
                    )
            all_articles.extend(articles)

        raw_df      = pd.DataFrame(all_articles)
        trading_idx: pd.DatetimeIndex
        if price_df is not None:
            trading_idx = pd.DatetimeIndex(price_df.index)
        else:
            trading_idx = pd.DatetimeIndex(biz_days)
        aligned     = align_news_to_trading_days(raw_df, trading_idx)

        # Cache (store lists as string repr)
        aligned.to_csv(cache_path)
        logger.info(f"[{ticker}] Cached aligned news → {cache_path} ({len(aligned)} rows)")

        aligned_datasets[ticker] = aligned

    return aligned_datasets