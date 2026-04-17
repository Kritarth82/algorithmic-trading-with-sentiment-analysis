import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
CACHE_DIR   = DATA_DIR / "cache"
MODEL_DIR   = BASE_DIR / "models" / "saved"
LOG_DIR     = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

for _d in [DATA_DIR, CACHE_DIR, MODEL_DIR, LOG_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Universe ──────────────────────────────────────────────────
TICKERS = [
    "MSFT",
  "TCS.NS",
]
#  "AAPL",
#     "TSLA",
#     "MSFT",
#     "NVDA",
#     "AMZN",
#     "META",
#   "TCS.NS",
#     "GOOGL",
# "RELIANCE.NS",
# ── Date range ────────────────────────────────────────────────
TRAIN_START = "2020-01-01"
TRAIN_END   = "2022-12-31"
VAL_START   = "2023-01-01"
VAL_END     = "2023-06-30"
TEST_START  = "2023-07-01"
TEST_END    = "2024-06-30"

# ── Features ──────────────────────────────────────────────────
PRICE_FEATURES = [
    "open", "high", "low", "close", "volume",
    "sma_20", "sma_50", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width",
    "atr_14", "obv", "stoch_k", "stoch_d",
    "returns", "log_returns", "vol_20",
]
SENTIMENT_FEATURES = [
    "finbert_positive", "finbert_negative", "finbert_neutral",
    "finbert_compound", "finbert_confidence", "finbert_uncertainty",
    "news_volume", "sentiment_ma3", "sentiment_ma7",
    "sentiment_momentum", "sentiment_volatility",
    "sentiment_dispersion", "topic_relevance_score",
]
TARGET_COL       = "future_return_5d"
FORECAST_HORIZON = 5

# ── Price data ────────────────────────────────────────────────
PRICE_CACHE_DIR = CACHE_DIR
PRICE_SOURCE    = "yfinance"

# ── Sentiment source — FinancialPhraseBank CSV ─────────────────
# Place all-data.csv in the same directory as main.py
FINANCIAL_PHRASE_CSV = BASE_DIR / "all-data.csv"

# Synthetic gap-fill settings
SYNTH_ARTICLES_PER_DAY  = 3      # articles generated per trading day
SYNTH_STRONG_RETURN_THR = 0.015  # |return| > this → strong pos/neg tone
SYNTH_MILD_RETURN_THR   = 0.005  # |return| > this → mild pos/neg tone

# Label → numeric score (used in fast path without FinBERT)
LABEL_SCORE_MAP = {
    "positive": +1.0,
    "negative": -1.0,
    "neutral":   0.0,
}

# Set False to skip FinBERT and use fast label→score mapping (no GPU needed)
# Forced False — we use all-data.csv + OHLCV-computed sentiment only (no API, no GPU)
USE_FINBERT = False

# ── NLP ───────────────────────────────────────────────────────
FINBERT_MODEL     = "ProsusAI/finbert"
FINBERT_DEVICE    = "cuda"
NLP_BATCH_SIZE    = 32
MAX_SEQ_LENGTH    = 512
MAX_TEXTS_PER_DAY = 10
SENTIMENT_AGG     = "score_weighted_mean"

# ── Sequence model ────────────────────────────────────────────
LOOKBACK            = 20
BATCH_SIZE          = 64
EPOCHS              = 100
LR                  = 3e-4
DROPOUT             = 0.3
HIDDEN_SIZE         = 128
NUM_LAYERS          = 2
ATTN_HEADS          = 8
EARLY_STOP_PATIENCE = 15
GRAD_CLIP           = 1.0
WEIGHT_DECAY        = 1e-4
LR_SCHEDULER        = "ReduceLROnPlateau"
LOSS_FUNCTION       = "HuberLoss"

# Correlation-aware training objectives
MODEL_CORR_LOSS_WEIGHT      = 0.55
MODEL_DIR_LOSS_WEIGHT       = 0.20
MODEL_DIR_LOGIT_TEMP        = 0.01
MODEL_MINI_BATCH_CORR_CLAMP = 0.99

# Validation model-selection objective
MODEL_VAL_SELECTION_CORR_WEIGHT = 0.90
MODEL_VAL_SELECTION_DIR_WEIGHT  = 0.10
MODEL_USE_ENSEMBLE              = False
MODEL_ENSEMBLE_MARGIN           = 0.01

# Evaluation scope
EVAL_USE_TRADABLE_ONLY          = False

# ── TCN ───────────────────────────────────────────────────────
TCN_CHANNELS      = [32, 64]
TCN_KERNEL_SIZE   = 3
TCN_DILATION_BASE = 2

# ── MoE ───────────────────────────────────────────────────────
NUM_EXPERTS        = 4
TOP_K_EXPERTS      = 2
EXPERT_HIDDEN_MULT = 2

# ── Regime ────────────────────────────────────────────────────
NUM_REGIMES        = 3
REGIME_THRESHOLDS  = [0.01, 0.02]
REGIME_LOSS_WEIGHT = 0.2

# ── Trading ───────────────────────────────────────────────────
BUY_THRESHOLD         = 0.003
SELL_THRESHOLD        = -0.003
TRANSACTION_COST      = 0.0005
MAX_POSITION          = 1.0
INITIAL_CAPITAL       = 100_000
RISK_FREE_RATE        = 0.00
ROLLING_SHARPE_WINDOW = 30
BT_EXECUTION_LAG_DAYS = 0
BT_SIGNAL_SMOOTH_SPAN = 3

# ── Risk-Adjusted Portfolio Controls ──────────────────────────
# Train on all tickers, but allow trading to focus on higher-quality
# validation performers to improve risk-adjusted returns.
RISK_FILTER_ENABLE       = False
RISK_TOP_K               = 2
RISK_MIN_VAL_DIR_ACC     = 0.50
RISK_MIN_VAL_CORR        = 0.00

# Position-level risk control: volatility targeting + drawdown brake.
RISK_VOL_TARGET_DAILY    = 0.012
RISK_VOL_LOOKBACK        = 20
RISK_MIN_POSITION_SCALE  = 0.25
RISK_MAX_POSITION_SCALE  = 1.00
RISK_MAX_DRAWDOWN_SOFT   = 0.10
RISK_DRAWDOWN_BRAKE      = 0.35

# Portfolio-level risk overlay: use lagged inverse-volatility weighting
# gated by trailing strategy momentum to suppress weak sleeves.
RISK_PORTFOLIO_WEIGHTING       = "risk_trend"
RISK_PORTFOLIO_VOL_LOOKBACK    = 40
RISK_PORTFOLIO_TREND_LOOKBACK  = 60
RISK_PORTFOLIO_TREND_THRESHOLD = -0.0010
RISK_PORTFOLIO_MAX_WEIGHT      = 0.80

# ── Reproducibility ───────────────────────────────────────────
RANDOM_SEED = 42

# ── Sentiment proxy weights (hypothesis study inputs) ─────────
# NOTE: These features are COMPUTED SENTIMENT PROXIES derived from
# price/volume data and lexicon-scored PhraseBank sentences.
# They are NOT outputs of a FinBERT transformer model — the finbert_*
# naming is kept only for downstream schema compatibility.
#
# The OHLCV sentiment computer uses LAGGED (t-1) price signals to
# derive a momentum-based market mood indicator. This avoids look-ahead
# bias: yesterday's price action influences today's sentiment estimate.
#
# Weights are applied inside a tanh() so the result stays in (-1, 1).
OHLCV_SENT_W_RETURN = 4.0   # lagged return signal (dominant)
OHLCV_SENT_W_RSI    = 1.5   # RSI-14 normalised: (RSI-50)/50
OHLCV_SENT_W_BB     = 1.0   # Bollinger Band position: (close-lower)/(upper-lower) * 2 - 1
OHLCV_SENT_W_VOL    = 0.5   # log volume surge vs 20-day avg (directional)

# Blend ratio: CSV-derived proxy vs lagged OHLCV anchor on informative days.
# 0.8 means the study primarily tests text-derived signal.
SENT_CSV_BLEND_WEIGHT = 0.8

# Demo leakage mode is removed from the pipeline.
# These are kept as dead config for backward compat only.
DEMO_LEAKAGE_MODE = False
DEMO_LEAKAGE_NOISE_STD = 0.0

# Study framing metadata used in evaluation reports
EXPERIMENT_OBJECTIVE = "evaluate_news_impact_on_returns"
EXPERIMENT_HYPOTHESIS = "news_signal_has_limited_incremental_predictive_power"