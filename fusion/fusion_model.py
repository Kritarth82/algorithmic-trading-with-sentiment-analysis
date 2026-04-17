"""
Phase 4 — Upgraded Fusion Model
Variants: EarlyFusion, LateFusion, CrossAttentionFusion (upgraded with TAG + RegimeHead).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import (
    HIDDEN_SIZE, NUM_LAYERS, ATTN_HEADS, DROPOUT,
    NUM_EXPERTS, TOP_K_EXPERTS, EXPERT_HIDDEN_MULT,
    NUM_REGIMES, REGIME_THRESHOLDS, REGIME_LOSS_WEIGHT,
    MODEL_CORR_LOSS_WEIGHT, MODEL_DIR_LOSS_WEIGHT,
    MODEL_DIR_LOGIT_TEMP, MODEL_MINI_BATCH_CORR_CLAMP,
    LOOKBACK, BATCH_SIZE, EPOCHS, LR,
    EARLY_STOP_PATIENCE, GRAD_CLIP, WEIGHT_DECAY,
    MODEL_DIR, RANDOM_SEED,
    PRICE_FEATURES, SENTIMENT_FEATURES, TARGET_COL,
)
from models.lstm_model import (
    TCNBlock, MultiHeadSelfAttentionRoPE, MixtureOfExpertsFF,
    TCN_CHANNELS, TCN_KERNEL_SIZE, TCN_DILATION_BASE,
    evaluate_predictions,
)

logger = logging.getLogger(__name__)
torch.manual_seed(RANDOM_SEED)


def _corrcoef_torch(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable Pearson correlation on a mini-batch."""
    yp = y_pred - torch.mean(y_pred)
    yt = y_true - torch.mean(y_true)
    cov = torch.mean(yp * yt)
    yp_std = torch.sqrt(torch.mean(yp * yp) + eps)
    yt_std = torch.sqrt(torch.mean(yt * yt) + eps)
    corr = cov / (yp_std * yt_std + eps)
    return torch.clamp(corr, -MODEL_MINI_BATCH_CORR_CLAMP, MODEL_MINI_BATCH_CORR_CLAMP)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _regime_labels(vol_series: np.ndarray) -> np.ndarray:
    """Map vol_20 values to regime integer labels 0/1/2."""
    labels = np.ones(len(vol_series), dtype=np.int64)
    labels[vol_series < REGIME_THRESHOLDS[0]]  = 0
    labels[vol_series > REGIME_THRESHOLDS[1]]  = 2
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FusionDataset(TensorDataset):
    def __init__(
        self,
        X_price:  torch.Tensor,
        X_sent:   torch.Tensor,
        y_return: torch.Tensor,
        y_regime: torch.Tensor,
    ):
        super().__init__(X_price, X_sent, y_return, y_regime)


def build_fusion_sequences(
    merged_df,
    price_cols:  List[str] = PRICE_FEATURES,
    sent_cols:   List[str] = SENTIMENT_FEATURES,
    target_col:  str       = TARGET_COL,
    lookback:    int       = LOOKBACK,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences for the fusion model.
    Returns (X_price, X_sent, y_return, y_regime).
    """
    available_price = [c for c in price_cols if c in merged_df.columns]
    available_sent  = [c for c in sent_cols  if c in merged_df.columns]

    price_vals  = merged_df[available_price].fillna(0).values.astype(np.float32)
    sent_vals   = merged_df[available_sent].fillna(0).values.astype(np.float32)
    target_vals = merged_df[target_col].fillna(0).values.astype(np.float32)

    vol_vals = merged_df["vol_20"].fillna(0).values if "vol_20" in merged_df.columns \
               else np.zeros(len(merged_df))

    X_p, X_s, y_ret, y_reg = [], [], [], []
    for i in range(lookback, len(merged_df)):
        if not np.isnan(target_vals[i]):
            X_p.append(price_vals[i - lookback : i])
            X_s.append(sent_vals[i - lookback : i])
            y_ret.append(target_vals[i])
            y_reg.append(_regime_labels(vol_vals[i : i + 1])[0])

    return (
        np.array(X_p,   dtype=np.float32),
        np.array(X_s,   dtype=np.float32),
        np.array(y_ret, dtype=np.float32),
        np.array(y_reg, dtype=np.int64),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Enhancement 1 — Temporal Attention Gate
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAttentionGate(nn.Module):
    """
    Learns per-timestep, per-feature gate weights for sentiment input.
    Suppresses noisy days (high uncertainty, low volume) before cross-attention.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_net = nn.Linear(d_model, d_model)

    def forward(self, x_sent: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_net(x_sent))
        return x_sent * gate


# ─────────────────────────────────────────────────────────────────────────────
# Enhancement 2 — Regime Head
# ─────────────────────────────────────────────────────────────────────────────

class RegimeHead(nn.Module):
    """
    Auxiliary classification head predicting market regime.
    Acts as a regularizer for the shared representation.
    """

    def __init__(self, hidden_size: int = HIDDEN_SIZE,
                 num_regimes: int = NUM_REGIMES, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_regimes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # raw logits (B, num_regimes)


# ─────────────────────────────────────────────────────────────────────────────
# Fusion Variant 1 — Early Fusion (concat then LSTM)
# ─────────────────────────────────────────────────────────────────────────────

class EarlyFusionModel(nn.Module):
    """Concatenate price + sentiment features then run through a single LSTM."""

    def __init__(
        self,
        n_price:     int,
        n_sent:      int,
        hidden:      int   = HIDDEN_SIZE,
        n_layers:    int   = NUM_LAYERS,
        dropout:     float = DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(n_price + n_sent, hidden, n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.regime_head = RegimeHead(hidden)
        self.norm        = nn.LayerNorm(hidden)

    def forward(
        self, x_price: torch.Tensor, x_sent: torch.Tensor
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        x        = torch.cat([x_price, x_sent], dim=-1)
        out, _   = self.lstm(x)
        pooled   = self.norm(out[:, -1, :])
        pred     = self.head(pooled).squeeze(-1)
        regime   = self.regime_head(pooled)
        return pred, None, regime


# ─────────────────────────────────────────────────────────────────────────────
# Fusion Variant 2 — Late Fusion (separate encoders, combine at prediction)
# ─────────────────────────────────────────────────────────────────────────────

class LateFusionModel(nn.Module):
    """Separate LSTM for each modality; combine pooled representations."""

    def __init__(
        self,
        n_price:  int,
        n_sent:   int,
        hidden:   int   = HIDDEN_SIZE,
        n_layers: int   = NUM_LAYERS,
        dropout:  float = DROPOUT,
    ):
        super().__init__()
        self.price_lstm = nn.LSTM(n_price, hidden, n_layers, batch_first=True,
                                  dropout=dropout if n_layers > 1 else 0.0)
        self.sent_lstm  = nn.LSTM(n_sent,  hidden, n_layers, batch_first=True,
                                  dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.regime_head = RegimeHead(hidden * 2)
        self.norm_p      = nn.LayerNorm(hidden)
        self.norm_s      = nn.LayerNorm(hidden)

    def forward(
        self, x_price: torch.Tensor, x_sent: torch.Tensor
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        p_out, _ = self.price_lstm(x_price)
        s_out, _ = self.sent_lstm(x_sent)
        p_pool   = self.norm_p(p_out[:, -1, :])
        s_pool   = self.norm_s(s_out[:, -1, :])
        combined = torch.cat([p_pool, s_pool], dim=-1)
        pred     = self.head(combined).squeeze(-1)
        regime   = self.regime_head(combined)
        return pred, None, regime


# ─────────────────────────────────────────────────────────────────────────────
# Fusion Variant 3 — Cross-Attention Fusion (upgraded, best performer)
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Price LSTM → cross-attention with gated sentiment → RoPE self-attention
    → MoE feed-forward → prediction head + regime head.
    """

    def __init__(
        self,
        n_price:  int,
        n_sent:   int,
        hidden:   int   = HIDDEN_SIZE,
        n_layers: int   = NUM_LAYERS,
        n_heads:  int   = ATTN_HEADS,
        dropout:  float = DROPOUT,
    ):
        super().__init__()
        # Price encoder
        self.tcn_price    = TCNBlock(n_price, TCN_CHANNELS, TCN_KERNEL_SIZE,
                                     TCN_DILATION_BASE, dropout)
        tcn_out_size      = self.tcn_price.out_size
        self.price_lstm   = nn.LSTM(tcn_out_size, hidden, n_layers,
                                    batch_first=True,
                                    dropout=dropout if n_layers > 1 else 0.0)

        # Temporal Attention Gate + sentiment projection
        self.tag       = TemporalAttentionGate(n_sent)
        self.sent_proj = nn.Linear(n_sent, hidden)

        # Cross-attention (price queries sentiment)
        self.cross_attn = nn.MultiheadAttention(hidden, n_heads,
                                                dropout=dropout, batch_first=True)
        self.ca_norm    = nn.LayerNorm(hidden)
        self.ca_drop    = nn.Dropout(dropout)

        # Self-attention with RoPE
        self.self_attn = MultiHeadSelfAttentionRoPE(hidden, n_heads, dropout)

        # MoE feed-forward
        self.moe_ff = MixtureOfExpertsFF(hidden, NUM_EXPERTS, TOP_K_EXPERTS,
                                         EXPERT_HIDDEN_MULT, dropout)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # Regime head
        self.regime_head = RegimeHead(hidden * 2)

    def forward(
        self,
        x_price: torch.Tensor,   # (B, T, n_price)
        x_sent:  torch.Tensor,   # (B, T, n_sent)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Gate + project sentiment
        gate_sent  = self.tag(x_sent)                    # (B, T, n_sent)
        sent_enc   = self.sent_proj(gate_sent)            # (B, T, hidden)

        # 2. Encode price
        tcn_out    = self.tcn_price(x_price)             # (B, T, tcn_ch)
        price_enc, _ = self.price_lstm(tcn_out)          # (B, T, hidden)

        # 3. Cross-attention: price queries sentiment
        price_ca, cross_attn_w = self.cross_attn(price_enc, sent_enc, sent_enc)
        price_ca   = self.ca_norm(price_enc + self.ca_drop(price_ca))

        # 4. Self-attention with RoPE
        price_sa   = self.self_attn(price_ca)            # (B, T, hidden)

        # 5. MoE feed-forward
        price_ff, router_w = self.moe_ff(price_sa)      # (B, T, hidden), (B, T, E)

        # 6. Global pooling: mean + last
        pooled = torch.cat([price_ff.mean(1), price_ff[:, -1, :]], dim=-1)  # (B, 2H)

        # 7. Heads
        pred         = self.head(pooled).squeeze(-1)
        regime_logits = self.regime_head(pooled)

        return pred, cross_attn_w, regime_logits, router_w[:, -1, :]


# ─────────────────────────────────────────────────────────────────────────────
# FusionTrainer
# ─────────────────────────────────────────────────────────────────────────────

class FusionTrainer:
    def __init__(
        self,
        model:        CrossAttentionFusion,
        device:       Optional[torch.device] = None,
        lr:           float = LR,
        weight_decay: float = WEIGHT_DECAY,
        grad_clip:    float = GRAD_CLIP,
        patience:     int   = EARLY_STOP_PATIENCE,
    ):
        self.model  = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                          weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=5)
        self.criterion = nn.HuberLoss(delta=0.01)
        self.corr_weight = float(MODEL_CORR_LOSS_WEIGHT)
        self.dir_weight = float(MODEL_DIR_LOSS_WEIGHT)
        self.dir_temp = float(max(MODEL_DIR_LOGIT_TEMP, 1e-4))
        self.grad_clip = grad_clip
        self.patience  = patience

    def _compute_loss(
        self,
        pred:          torch.Tensor,
        regime_logits: torch.Tensor,
        y_return:      torch.Tensor,
        y_regime:      torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return_loss = self.criterion(pred, y_return)
        corr = _corrcoef_torch(pred, y_return)
        corr_loss = 1.0 - corr
        dir_targets = (y_return > 0).float()
        dir_logits = pred / self.dir_temp
        dir_loss = F.binary_cross_entropy_with_logits(dir_logits, dir_targets)
        regime_loss = F.cross_entropy(regime_logits, y_regime)
        total = (
            return_loss
            + self.corr_weight * corr_loss
            + self.dir_weight * dir_loss
            + REGIME_LOSS_WEIGHT * regime_loss
        )
        return total, return_loss, regime_loss

    def _make_loader(
        self,
        X_p: np.ndarray, X_s: np.ndarray,
        y_r: np.ndarray, y_reg: np.ndarray,
        shuffle: bool,
    ) -> DataLoader:
        ds = FusionDataset(
            torch.from_numpy(X_p),
            torch.from_numpy(X_s),
            torch.from_numpy(y_r),
            torch.from_numpy(y_reg),
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        total = ret_sum = reg_sum = 0.0
        count = 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for X_p, X_s, y_r, y_reg in loader:
                X_p, X_s = X_p.to(self.device), X_s.to(self.device)
                y_r       = y_r.to(self.device)
                y_reg     = y_reg.to(self.device)

                pred, _, regime_logits, _ = self.model(X_p, X_s)
                loss, ret_l, reg_l    = self._compute_loss(pred, regime_logits, y_r, y_reg)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                n      = len(y_r)
                total  += loss.item()  * n
                ret_sum += ret_l.item() * n
                reg_sum += reg_l.item() * n
                count  += n

        n = max(count, 1)
        return {"total": total / n, "return": ret_sum / n, "regime": reg_sum / n}

    def fit(
        self,
        X_price_tr: np.ndarray, X_sent_tr: np.ndarray,
        y_ret_tr:   np.ndarray, y_reg_tr: np.ndarray,
        X_price_vl: np.ndarray, X_sent_vl: np.ndarray,
        y_ret_vl:   np.ndarray, y_reg_vl: np.ndarray,
        epochs:     int  = EPOCHS,
        save_path:  Optional[Path] = None,
    ) -> Dict[str, List[float]]:
        tr_loader = self._make_loader(X_price_tr, X_sent_tr, y_ret_tr, y_reg_tr, True)
        vl_loader = self._make_loader(X_price_vl, X_sent_vl, y_ret_vl, y_reg_vl, False)

        best_val, no_improve = float("inf"), 0
        history: Dict[str, List[float]] = {
            "train_total": [], "val_total": [],
            "train_return": [], "val_return": [],
            "train_regime": [], "val_regime": [],
        }

        for epoch in range(1, epochs + 1):
            tr = self._run_epoch(tr_loader, True)
            vl = self._run_epoch(vl_loader, False)
            self.scheduler.step(vl["total"])

            for k in ["total", "return", "regime"]:
                history[f"train_{k}"].append(tr[k])
                history[f"val_{k}"].append(vl[k])

            if vl["total"] < best_val:
                best_val   = vl["total"]
                no_improve = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch:03d} | "
                    f"tr_total={tr['total']:.5f} vl_total={vl['total']:.5f} | "
                    f"tr_ret={tr['return']:.5f} tr_reg={tr['regime']:.5f}"
                )

        if save_path and save_path.exists():
            self.model.load_state_dict(torch.load(save_path, map_location=self.device))

        return history

    @torch.no_grad()
    def predict(
        self,
        X_price: np.ndarray,
        X_sent:  np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (pred_returns, cross_attn_weights, router_weights)."""
        self.model.eval()
        ds     = TensorDataset(torch.from_numpy(X_price), torch.from_numpy(X_sent))
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

        preds, attn_ws, router_ws = [], [], []
        for X_p, X_s in loader:
            pred, attn_w, regime_logits, router_w = self.model(
                X_p.to(self.device), X_s.to(self.device)
            )
            preds.append(pred.cpu().numpy())
            if attn_w is not None:
                attn_ws.append(attn_w.cpu().numpy())
            if router_w is not None:
                router_ws.append(router_w.cpu().numpy())

        preds_np = np.concatenate(preds)
        attn_np  = np.concatenate(attn_ws) if attn_ws else np.array([])
        router_np = np.concatenate(router_ws) if router_ws else np.array([])
        return preds_np, attn_np, router_np
