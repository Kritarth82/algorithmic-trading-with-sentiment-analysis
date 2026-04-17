"""
Phase 3 — Upgraded LSTM + Attention Model
Enhancements: TCN preprocessing, RoPE self-attention, Mixture-of-Experts FF.
"""

from __future__ import annotations

import logging
import math
import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import xgboost as xgb
except Exception:
    xgb = None

from config import (
    HIDDEN_SIZE, NUM_LAYERS, ATTN_HEADS, DROPOUT,
    TCN_CHANNELS, TCN_KERNEL_SIZE, TCN_DILATION_BASE,
    NUM_EXPERTS, TOP_K_EXPERTS, EXPERT_HIDDEN_MULT,
    LOOKBACK, BATCH_SIZE, EPOCHS, LR,
    EARLY_STOP_PATIENCE, GRAD_CLIP, WEIGHT_DECAY,
    MODEL_CORR_LOSS_WEIGHT, MODEL_DIR_LOSS_WEIGHT,
    MODEL_DIR_LOGIT_TEMP, MODEL_MINI_BATCH_CORR_CLAMP,
    MODEL_DIR, RANDOM_SEED,
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
# Enhancement 1 — TCN Preprocessing Block
# ─────────────────────────────────────────────────────────────────────────────

class _CausalConv1dBlock(nn.Module):
    """Single dilated causal Conv1d layer with GELU, dropout, residual."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int,
                 dilation: int, dropout: float):
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.conv    = nn.Conv1d(in_ch, out_ch, kernel,
                                 padding=self.padding, dilation=dilation)
        self.act     = nn.GELU()
        self.drop    = nn.Dropout(dropout)
        self.res     = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm    = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.conv(x)
        if self.padding:
            out = out[:, :, :-self.padding]   # remove future leakage
        out = self.act(out)
        out = self.drop(out)
        out = out + self.res(x)
        # LayerNorm over channel dim: transpose → norm → transpose
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return out


class TCNBlock(nn.Module):
    """
    Stack of dilated causal conv layers.
    Input/Output: (B, T, input_size) → (B, T, TCN_CHANNELS[-1])
    """

    def __init__(
        self,
        input_size: int,
        channels:   List[int] = TCN_CHANNELS,
        kernel:     int        = TCN_KERNEL_SIZE,
        dilation_base: int     = TCN_DILATION_BASE,
        dropout:    float      = DROPOUT,
    ):
        super().__init__()
        layers = []
        in_ch  = input_size
        for i, out_ch in enumerate(channels):
            dil = dilation_base ** i
            layers.append(_CausalConv1dBlock(in_ch, out_ch, kernel, dil, dropout))
            in_ch = out_ch
        self.net      = nn.Sequential(*layers)
        self.out_size = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = x.transpose(1, 2)          # → (B, C, T)
        x = self.net(x)
        x = x.transpose(1, 2)          # → (B, T, C)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Enhancement 2 — Rotary Position Embeddings (RoPE)
# ─────────────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device):
        t     = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)   # (T, dim/2)
        emb   = torch.cat([freqs, freqs], dim=-1)            # (T, dim)
        return emb.cos(), emb.sin()                          # (T, dim), (T, dim)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    @staticmethod
    def apply_rotary(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, heads, T, head_dim)
        cos = cos[:x.shape[2]].unsqueeze(0).unsqueeze(0)  # (1, 1, T, dim)
        sin = sin[:x.shape[2]].unsqueeze(0).unsqueeze(0)
        return x * cos + RotaryEmbedding.rotate_half(x) * sin


class MultiHeadSelfAttentionRoPE(nn.Module):
    """Multi-head self-attention with Rotary Position Embeddings on Q and K."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = DROPOUT):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5

        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)                                     # (B, T, H, hd)
        q = q.transpose(1, 2); k = k.transpose(1, 2)                # (B, H, T, hd)
        v = v.transpose(1, 2)

        cos, sin = self.rope(T, x.device)
        q = RotaryEmbedding.apply_rotary(q, cos, sin)
        k = RotaryEmbedding.apply_rotary(k, cos, sin)

        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        attn = self.drop(attn)
        out  = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out  = self.proj(out)
        return self.norm(x + self.drop(out))                         # residual + norm


# ─────────────────────────────────────────────────────────────────────────────
# Enhancement 3 — Mixture of Experts Feed-Forward
# ─────────────────────────────────────────────────────────────────────────────

class MixtureOfExpertsFF(nn.Module):
    """
    Gated Top-K MoE feed-forward layer.
    Returns (output, router_weights) where router_weights are logged for interpretability.
    """

    def __init__(
        self,
        d_model:     int,
        num_experts: int   = NUM_EXPERTS,
        top_k:       int   = TOP_K_EXPERTS,
        hidden_mult: int   = EXPERT_HIDDEN_MULT,
        dropout:     float = DROPOUT,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        hidden           = d_model * hidden_mult

        self.router  = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, d_model),
            )
            for _ in range(num_experts)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, d_model)
        B, T, D = x.shape

        logits  = self.router(x)                    # (B, T, num_experts)
        weights = torch.softmax(logits, dim=-1)

        # Top-k gating
        top_vals, top_idx = torch.topk(weights, self.top_k, dim=-1)
        mask    = torch.zeros_like(weights).scatter_(-1, top_idx, 1.0)
        weights = weights * mask
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-9)   # renormalize

        # Weighted sum of expert outputs
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            w_i = weights[..., i].unsqueeze(-1)     # (B, T, 1)
            out = out + w_i * expert(x)

        out = self.norm(x + self.drop(out))
        return out, weights                          # (B, T, D), (B, T, num_experts)


# ─────────────────────────────────────────────────────────────────────────────
# Upgraded TemporalBlock
# ─────────────────────────────────────────────────────────────────────────────

class TemporalBlock(nn.Module):
    """
    TCNBlock → LSTM → RoPE Self-Attention → MoE-FF → global pool.
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int   = HIDDEN_SIZE,
        num_layers:  int   = NUM_LAYERS,
        n_heads:     int   = ATTN_HEADS,
        dropout:     float = DROPOUT,
    ):
        super().__init__()
        self.tcn  = TCNBlock(input_size, TCN_CHANNELS, TCN_KERNEL_SIZE,
                             TCN_DILATION_BASE, dropout)
        tcn_out   = self.tcn.out_size

        self.lstm = nn.LSTM(tcn_out, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attn = MultiHeadSelfAttentionRoPE(hidden_size, n_heads, dropout)
        self.moe  = MixtureOfExpertsFF(hidden_size, NUM_EXPERTS, TOP_K_EXPERTS,
                                       EXPERT_HIDDEN_MULT, dropout)
        self.out_size = hidden_size * 2   # mean + last

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, input_size)
        tcn_out       = self.tcn(x)          # (B, T, tcn_ch)
        lstm_out, _   = self.lstm(tcn_out)   # (B, T, hidden)
        attended      = self.attn(lstm_out)  # (B, T, hidden)
        ff_out, rw    = self.moe(attended)   # (B, T, hidden), (B, T, E)

        pooled = torch.cat([ff_out.mean(1), ff_out[:, -1, :]], dim=-1)  # (B, 2*H)
        return pooled, rw[:, -1, :]           # router weights of last timestep


# ─────────────────────────────────────────────────────────────────────────────
# Full LSTM-Attention Model
# ─────────────────────────────────────────────────────────────────────────────

class LSTMAttnModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, n_heads: int = ATTN_HEADS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.temporal = TemporalBlock(input_size, hidden_size, num_layers,
                                      n_heads, dropout)
        self.head = nn.Sequential(
            nn.Linear(self.temporal.out_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled, rw = self.temporal(x)
        return self.head(pooled).squeeze(-1), rw


# ─────────────────────────────────────────────────────────────────────────────
# Sequence builder
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col:   str,
    lookback:     int = LOOKBACK,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) sliding-window arrays from a DataFrame."""
    import pandas as pd
    X_list, y_list = [], []
    vals = df[feature_cols].fillna(0).values
    tgt  = df[target_col].fillna(0).values
    for i in range(lookback, len(df)):
        if not np.isnan(tgt[i]):
            X_list.append(vals[i - lookback : i])
            y_list.append(tgt[i])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class ModelTrainer:
    def __init__(
        self,
        model:      LSTMAttnModel,
        device:     Optional[torch.device] = None,
        lr:         float = LR,
        weight_decay: float = WEIGHT_DECAY,
        grad_clip:  float = GRAD_CLIP,
        patience:   int   = EARLY_STOP_PATIENCE,
    ):
        self.model  = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer  = torch.optim.Adam(model.parameters(), lr=lr,
                                           weight_decay=weight_decay)
        self.scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=5)
        self.criterion  = nn.HuberLoss(delta=0.01)
        self.corr_weight = float(MODEL_CORR_LOSS_WEIGHT)
        self.dir_weight = float(MODEL_DIR_LOSS_WEIGHT)
        self.dir_temp = float(max(MODEL_DIR_LOGIT_TEMP, 1e-4))
        self.grad_clip  = grad_clip
        self.patience   = patience

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        total, count = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                pred, _ = self.model(X)

                reg_loss = self.criterion(pred, y)
                corr = _corrcoef_torch(pred, y)
                corr_loss = 1.0 - corr
                dir_targets = (y > 0).float()
                dir_logits = pred / self.dir_temp
                dir_loss = F.binary_cross_entropy_with_logits(dir_logits, dir_targets)

                loss = reg_loss + self.corr_weight * corr_loss + self.dir_weight * dir_loss
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                total += loss.item() * len(y)
                count += len(y)
        return total / max(count, 1)

    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
        epochs:  int   = EPOCHS,
        save_path: Optional[Path] = None,
    ) -> Dict[str, List[float]]:
        def _loader(X, y, shuffle):
            ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
            return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

        tr_loader = _loader(X_train, y_train, True)
        vl_loader = _loader(X_val,   y_val,   False)

        best_val, no_improve = float("inf"), 0
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            tr_loss = self._run_epoch(tr_loader, train=True)
            vl_loss = self._run_epoch(vl_loader, train=False)
            self.scheduler.step(vl_loss)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(vl_loss)

            if vl_loss < best_val:
                best_val   = vl_loss
                no_improve = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    logger.info(f"Epoch {epoch:03d} | val_loss={vl_loss:.6f} ✓ saved")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:03d} | train={tr_loss:.6f} val={vl_loss:.6f}")

        if save_path and save_path.exists():
            self.model.load_state_dict(torch.load(save_path, map_location=self.device))

        return history

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        ds     = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        preds, rws = [], []
        for (xb,) in loader:
            p, rw = self.model(xb.to(self.device))
            preds.append(p.cpu().numpy())
            rws.append(rw.cpu().numpy())
        return np.concatenate(preds), np.concatenate(rws)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline models
# ─────────────────────────────────────────────────────────────────────────────

class BaselineModels:
    """Ridge regression and XGBoost baselines for ablation study."""

    def __init__(self):
        self.ridge  = Ridge(alpha=1.0)
        # XGBoost can segfault on macOS in some binary builds; prefer stable fallback.
        use_xgb = (
            xgb is not None
            and platform.system() != "Darwin"
            and sys.version_info < (3, 14)
        )
        if use_xgb:
            self.tree_model = xgb.XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED,
                verbosity=0,
            )
            self.tree_name = "xgb"
        else:
            self.tree_model = HistGradientBoostingRegressor(
                max_depth=4,
                learning_rate=0.05,
                random_state=RANDOM_SEED,
            )
            self.tree_name = "hgb"

    def fit(self, X: np.ndarray, y: np.ndarray):
        X2d = X.reshape(len(X), -1)
        self.ridge.fit(X2d, y)
        self.tree_model.fit(X2d, y)

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        X2d = X.reshape(len(X), -1)
        return {
            "ridge": self.ridge.predict(X2d),
            self.tree_name: self.tree_model.predict(X2d),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label:  str = "",
) -> Dict[str, float]:
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    # Directional accuracy
    da   = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    metrics = {"mse": mse, "rmse": rmse, "mae": mae, "corr": corr, "dir_acc": da}
    if label:
        logger.info(f"[{label}] RMSE={rmse:.6f} MAE={mae:.6f} Corr={corr:.4f} DirAcc={da:.4f}")
    return metrics
