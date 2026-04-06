"""
Spectral regression models (CNN/MLP/Transformer/Fourier-CNN), wrapped in Lightning modules.
"""
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR

from config import Config


def build_warmup_cosine_scheduler(optimizer, cfg):
    """Create epoch-wise warmup + cosine annealing scheduler."""
    total_epochs = max(int(cfg.max_epochs), 1)
    warmup_epochs = max(int(getattr(cfg, "warmup_epochs", 0)), 0)
    min_lr = float(getattr(cfg, "cosine_min_lr", 0.0))
    base_lr = float(cfg.lr)
    if min_lr < 0.0:
        raise ValueError("cosine_min_lr must be >= 0.")
    if min_lr > base_lr:
        raise ValueError("cosine_min_lr must be <= lr.")

    # Keep cosine phase valid even when warmup >= total epochs.
    cosine_epochs = max(total_epochs - warmup_epochs, 1)

    def lr_lambda(epoch: int):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)

        cosine_epoch = max(epoch - warmup_epochs, 0)
        progress = min(float(cosine_epoch) / float(cosine_epochs), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = min_lr + (base_lr - min_lr) * cosine
        return lr / base_lr

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------------------------------------------------------
# 1D-CNN backbone
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv1d → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_ch, out_ch, kernel_size=7, pool_size=4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
        )

    def forward(self, x):
        return self.block(x)


class FourierSpectralConv1d(nn.Module):
    """
    FNO-style spectral convolution for 1D signals.

    Learns channel mixing on truncated low-frequency Fourier modes.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_modes: int,
        softshrink_lambda: float = 0.0,
    ):
        super().__init__()
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1.")
        if softshrink_lambda < 0.0:
            raise ValueError("softshrink_lambda must be >= 0.")

        self.out_ch = int(out_ch)
        self.n_modes = int(n_modes)
        self.softshrink_lambda = float(softshrink_lambda)

        scale = 1.0 / max(in_ch * out_ch, 1)
        self.weight = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, self.n_modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        x_ft = torch.fft.rfft(x, dim=-1, norm="ortho")  # (B, C, F)
        n_freq = x_ft.size(-1)
        n_modes = min(self.n_modes, n_freq)

        out_ft = torch.zeros(
            x.size(0),
            self.out_ch,
            n_freq,
            dtype=x_ft.dtype,
            device=x.device,
        )
        if n_modes > 0:
            out_ft[:, :, :n_modes] = torch.einsum(
                "bcm,com->bom",
                x_ft[:, :, :n_modes],
                self.weight[:, :, :n_modes],
            )

        if self.softshrink_lambda > 0.0:
            out_ft = torch.complex(
                F.softshrink(out_ft.real, lambd=self.softshrink_lambda),
                F.softshrink(out_ft.imag, lambd=self.softshrink_lambda),
            )

        return torch.fft.irfft(out_ft, n=x.size(-1), dim=-1, norm="ortho")


class FourierConvBlock1d(nn.Module):
    """
    Hybrid local+global block inspired by FFC:
    - local path: standard Conv1d
    - global path: Fourier spectral convolution
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        pool_size: int,
        n_modes: int,
        dropout: float = 0.0,
        softshrink_lambda: float = 0.0,
    ):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )
        self.global_in = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.global_fourier = FourierSpectralConv1d(
            out_ch,
            out_ch,
            n_modes=n_modes,
            softshrink_lambda=softshrink_lambda,
        )
        self.global_norm = nn.BatchNorm1d(out_ch)

        self.fuse = nn.Sequential(
            nn.Conv1d(out_ch * 2, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )
        self.residual_proj = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_local = self.local(x)
        h_global = self.global_in(x)
        h_global = self.global_fourier(h_global)
        h_global = self.global_norm(h_global)

        h = self.fuse(torch.cat([h_local, h_global], dim=1))
        h = self.dropout(h)
        h = h + self.residual_proj(x)
        return self.pool(h)


class SpectralFourierCNN(nn.Module):
    """Hybrid Fourier-CNN for spectral regression."""

    def __init__(self, cfg: Config, input_length: int):
        super().__init__()
        channels = [1] + list(cfg.cnn_channels)
        kernel_size = int(cfg.kernel_size)
        pool_size = int(cfg.pool_size)
        softshrink_lambda = float(getattr(cfg, "fourier_softshrink_lambda", 0.0))
        dropout = float(getattr(cfg, "dropout", 0.0))

        layers = []
        L = input_length
        for i in range(len(channels) - 1):
            n_modes = self._resolve_n_modes(cfg, seq_len=max(L, 1))
            layers.append(
                FourierConvBlock1d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size,
                    pool_size=pool_size,
                    n_modes=n_modes,
                    dropout=dropout,
                    softshrink_lambda=softshrink_lambda,
                )
            )
            L = L // pool_size

        final_len = max(L, 1)
        head_pool_out_len = int(getattr(cfg, "fourier_head_pool_out_len", 8))
        if head_pool_out_len < 1:
            raise ValueError("fourier_head_pool_out_len must be >= 1.")
        head_pool_out_len = min(head_pool_out_len, final_len)

        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(head_pool_out_len),
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
            nn.Linear(channels[-1] * head_pool_out_len, cfg.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fc_hidden, cfg.n_targets),
        )

    @staticmethod
    def _resolve_n_modes(cfg: Config, seq_len: int) -> int:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1.")
        n_freq = (seq_len // 2) + 1

        explicit_modes = getattr(cfg, "fourier_n_modes", None)
        min_modes = int(getattr(cfg, "fourier_min_modes", 8))
        ratio = float(getattr(cfg, "fourier_modes_ratio", 0.25))
        if min_modes < 1:
            raise ValueError("fourier_min_modes must be >= 1.")
        if ratio <= 0.0:
            raise ValueError("fourier_modes_ratio must be > 0.")

        if explicit_modes is not None:
            n_modes = int(explicit_modes)
        else:
            n_modes = max(min_modes, int(round(n_freq * ratio)))
        return max(1, min(n_modes, n_freq))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.head(h)


class SpectralCNN(nn.Module):
    """1D-CNN backbone + regression head."""

    def __init__(self, cfg: Config, input_length: int):
        super().__init__()

        # Build convolutional blocks
        channels = [1] + list(cfg.cnn_channels)  # e.g. [1, 32, 64, 128]
        layers = []
        L = input_length
        for i in range(len(channels) - 1):
            layers.append(
                ConvBlock(channels[i], channels[i + 1], cfg.kernel_size, cfg.pool_size)
            )
            L = L // cfg.pool_size  # track spatial dimension after pooling

        self.encoder = nn.Sequential(*layers)

        # Flatten dimension after encoder
        flat_dim = channels[-1] * max(L, 1)

        # Regression head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
            nn.Linear(flat_dim, cfg.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fc_hidden, cfg.n_targets),
        )

    def forward(self, x):
        """x: (B, 1, L) → (B, n_targets)"""
        h = self.encoder(x)
        return self.head(h)


class SpectralMaskedModelingCNN(nn.Module):
    """CNN masked-spectrum model with SpectralCNN-compatible encoder."""

    def __init__(self, cfg: Config, input_length: int):
        super().__init__()
        channels = [1] + list(cfg.cnn_channels)
        layers = []
        L = input_length
        for i in range(len(channels) - 1):
            layers.append(
                ConvBlock(channels[i], channels[i + 1], cfg.kernel_size, cfg.pool_size)
            )
            L = L // cfg.pool_size
        self.encoder = nn.Sequential(*layers)

        flat_dim = channels[-1] * max(L, 1)
        hidden_dim = int(getattr(cfg, "fc_hidden", 256))
        dropout = float(getattr(cfg, "dropout", 0.0))
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_length),
        )

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)


class SpectralDualBranchCNN(nn.Module):
    """Dual-branch CNN: raw/derivative halves use different kernels."""

    def __init__(self, cfg: Config, input_length: int):
        super().__init__()
        if input_length % 2 != 0:
            raise ValueError(
                "cnn_dual requires even input_length (raw + derivative concatenation)."
            )

        half_len = input_length // 2
        channels = [1] + list(cfg.cnn_channels)

        def build_encoder(kernel_size: int):
            layers = []
            L = half_len
            for i in range(len(channels) - 1):
                layers.append(
                    ConvBlock(channels[i], channels[i + 1], kernel_size, cfg.pool_size)
                )
                L = L // cfg.pool_size
            return nn.Sequential(*layers), max(L, 1)

        self.encoder_raw, len_raw = build_encoder(int(cfg.dual_kernel_size_raw))
        self.encoder_deriv, len_deriv = build_encoder(int(cfg.dual_kernel_size_derivative))
        flat_dim = channels[-1] * (len_raw + len_deriv)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
            nn.Linear(flat_dim, cfg.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fc_hidden, cfg.n_targets),
        )

    def forward(self, x):
        """x: (B, 1, 2L) where first L=raw, second L=derivative."""
        total_len = x.size(-1)
        half_len = total_len // 2
        x_raw = x[:, :, :half_len]
        x_der = x[:, :, half_len:]
        h_raw = self.encoder_raw(x_raw)
        h_der = self.encoder_deriv(x_der)
        h = torch.cat([h_raw, h_der], dim=2)
        return self.head(h)


class SpectralMLP(nn.Module):
    """Simple MLP baseline for spectral regression."""

    def __init__(self, cfg: Config, input_length: int):
        super().__init__()
        hidden_dims = list(cfg.mlp_hidden_dims)
        dims = [input_length] + hidden_dims + [cfg.n_targets]
        layers = [nn.Flatten()]
        for i in range(len(dims) - 2):
            layers.extend(
                [
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(cfg.dropout),
                ]
            )
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: (B, 1, L) or (B, L) → (B, n_targets)"""
        return self.net(x)


class SpectralTransformer(nn.Module):
    """
    ProTformer-style spectral regressor.

    Interface is kept compatible with the previous implementation:
      - __init__(cfg, input_length, feature_wavenumbers=None)
      - forward(x) where x is (B, 1, L) or (B, L)
    """

    def __init__(
        self,
        cfg,
        input_length: int,
        feature_wavenumbers: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # ---- ProTformer-like defaults ----
        patch_size = int(getattr(cfg, "transformer_patch_size", 25))
        d_model = int(getattr(cfg, "transformer_d_model", 50))      # paper: 50-d patch embedding
        nhead = int(getattr(cfg, "transformer_nhead", 5))           # 50 divisible by 5
        num_layers = int(getattr(cfg, "transformer_num_layers", 3)) # paper: 3 transformer blocks
        ffn_dim = int(getattr(cfg, "transformer_ffn_dim", 128))
        dropout = float(getattr(cfg, "transformer_dropout", 0.1))

        # Regression head dimensions close to the paper
        head_hidden1 = int(getattr(cfg, "transformer_head_hidden", 128))
        head_hidden2 = int(getattr(cfg, "transformer_head_hidden2", 32))

        if patch_size < 1:
            raise ValueError("transformer_patch_size must be >= 1.")
        if d_model < 2:
            raise ValueError("transformer_d_model must be >= 2.")
        if d_model % nhead != 0:
            raise ValueError(
                "transformer_d_model must be divisible by transformer_nhead."
            )
        if input_length < 1:
            raise ValueError("input_length must be >= 1.")

        self.input_length = int(input_length)
        self.patch_size = patch_size
        self.d_model = d_model

        # Number of patches after right padding
        self.num_patches = math.ceil(self.input_length / self.patch_size)
        self.padded_length = self.num_patches * self.patch_size
        self.pad_length = self.padded_length - self.input_length

        # Keep compatibility with previous constructor signature.
        # If actual feature wavenumbers are given, use patch-center wavenumbers for sinusoidal PE.
        if feature_wavenumbers is None:
            wn = torch.arange(self.input_length, dtype=torch.float32)
        else:
            wn = torch.as_tensor(feature_wavenumbers, dtype=torch.float32).flatten()
            if int(wn.numel()) != int(self.input_length):
                raise ValueError(
                    "feature_wavenumbers length must match input_length. "
                    f"got {int(wn.numel())} vs {int(input_length)}"
                )

        self.register_buffer("feature_wavenumbers", wn, persistent=True)

        patch_centers = self._build_patch_centers(wn, self.patch_size, self.num_patches)
        pos_enc = self._build_wavenumber_sincos(patch_centers, d_model)
        self.register_buffer("positional_encoding", pos_enc, persistent=True)

        # Learnable CLS token and its positional embedding.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_positional = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_positional, mean=0.0, std=0.02)

        # ---- Patch embedding: (patch_size) -> d_model ----
        self.patch_embed = nn.Linear(self.patch_size, d_model)

        self.embed_norm = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # ---- Transformer blocks ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ---- Regression head: CLS token -> 128 -> 32 -> n_targets ----
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden1, head_hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden2, cfg.n_targets),
        )

    @staticmethod
    def _build_patch_centers(
        wavenumbers: torch.Tensor,
        patch_size: int,
        num_patches: int,
    ) -> torch.Tensor:
        """
        Build one representative wavenumber per patch using the mean of each patch.
        For the last incomplete patch, repeat the last wavenumber to match padded length.
        """
        if wavenumbers.numel() == 0:
            raise ValueError("wavenumbers must be non-empty.")

        padded_len = num_patches * patch_size
        if padded_len > wavenumbers.numel():
            pad_len = padded_len - wavenumbers.numel()
            pad_val = wavenumbers[-1].expand(pad_len)
            wavenumbers = torch.cat([wavenumbers, pad_val], dim=0)

        patch_wn = wavenumbers.view(num_patches, patch_size)
        return patch_wn.mean(dim=1)

    @staticmethod
    def _build_wavenumber_sincos(
        wavenumbers: torch.Tensor,
        d_model: int,
    ) -> torch.Tensor:
        """
        Sinusoidal positional encoding built from absolute wavenumbers.
        """
        wn = wavenumbers.float()
        wn = (wn - wn.mean()) / (wn.std(unbiased=False) + 1e-8)

        half = d_model // 2
        freq = torch.exp(
            torch.arange(half, dtype=wn.dtype, device=wn.device)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = wn.unsqueeze(1) * freq.unsqueeze(0)
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

        if pe.size(1) < d_model:
            pe = torch.cat([pe, torch.sin(wn).unsqueeze(1)], dim=1)

        return pe[:, :d_model]  # (num_patches, d_model)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L) -> (B, num_patches, patch_size)
        Right-pad with zeros if L is not divisible by patch_size.
        """
        if x.size(-1) != self.input_length:
            raise ValueError(
                f"Expected input length {self.input_length}, got {x.size(-1)}."
            )

        if self.pad_length > 0:
            x = torch.nn.functional.pad(x, (0, self.pad_length), mode="constant", value=0.0)

        return x.view(x.size(0), self.num_patches, self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, L) or (B, L)
        return: (B, n_targets)
        """
        if x.dim() == 3:
            if x.size(1) != 1:
                raise ValueError(
                    f"Expected x shape (B, 1, L) when x.dim()==3, got {tuple(x.shape)}"
                )
            x = x.squeeze(1)
        elif x.dim() != 2:
            raise ValueError(
                f"Expected x to have shape (B, L) or (B, 1, L), got {tuple(x.shape)}"
            )

        # 1) Patchify spectral sequence
        x = self._patchify(x)  # (B, num_patches, patch_size)

        # 2) Linear patch embedding
        token = self.patch_embed(x)  # (B, num_patches, d_model)

        # 3) Prepend CLS token
        cls_token = self.cls_token.expand(token.size(0), -1, -1)  # (B, 1, d_model)
        token = torch.cat([cls_token, token], dim=1)  # (B, 1 + num_patches, d_model)

        # 4) Add positional encoding (learnable for CLS, sinusoidal for patch tokens)
        pos = torch.cat(
            [self.cls_positional, self.positional_encoding.unsqueeze(0)],
            dim=1,
        )  # (1, 1 + num_patches, d_model)
        token = token + pos
        token = self.embed_norm(token)
        token = self.embed_dropout(token)

        # 5) Transformer blocks
        h = self.encoder(token)  # (B, 1 + num_patches, d_model)

        # 6) Use CLS output for regression
        cls_out = h[:, 0, :]  # (B, d_model)

        # 7) Regression head
        return self.head(cls_out)


class SpectralReconstructionMLP(nn.Module):
    """MLP autoencoder for spectrum reconstruction."""

    def __init__(self, cfg: Config, input_length: int):
        super().__init__()
        hidden_dims = list(cfg.recon_hidden_dims)
        latent_dim = int(cfg.recon_latent_dim)

        # Encoder: input -> ... -> latent
        enc_dims = [input_length] + hidden_dims + [latent_dim]
        enc_layers = [nn.Flatten()]
        for i in range(len(enc_dims) - 2):
            enc_layers.extend(
                [
                    nn.Linear(enc_dims[i], enc_dims[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(cfg.dropout),
                ]
            )
        enc_layers.append(nn.Linear(enc_dims[-2], enc_dims[-1]))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: latent -> ... -> input
        dec_dims = [latent_dim] + hidden_dims[::-1] + [input_length]
        dec_layers = []
        for i in range(len(dec_dims) - 2):
            dec_layers.extend(
                [
                    nn.Linear(dec_dims[i], dec_dims[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(cfg.dropout),
                ]
            )
        dec_layers.append(nn.Linear(dec_dims[-2], dec_dims[-1]))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        """x: (B, 1, L) or (B, L) -> reconstructed spectrum (B, L)."""
        z = self.encoder(x)
        return self.decoder(z)


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class _BaseSpectralLightningModule(pl.LightningModule):
    """Common optimizer/scheduler and LR logging for spectral tasks."""

    cfg: Config

    def on_train_epoch_start(self):
        optimizer = self.optimizers()
        if optimizer is None:
            return
        if isinstance(optimizer, (list, tuple)):
            optimizer = optimizer[0]
        lr = optimizer.param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = build_warmup_cosine_scheduler(optimizer, self.cfg)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class SpectralRegressionModule(_BaseSpectralLightningModule):
    """Wraps selectable backbone (CNN/MLP/Transformer) with train/val logic."""

    def __init__(
        self,
        cfg: Config,
        input_length: int,
        feature_wavenumbers: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        model_type = str(cfg.model_type).lower()
        self.model_type = model_type
        self.group_lasso_lambda = float(getattr(cfg, "mlp_group_lasso_lambda", 0.0))
        if self.group_lasso_lambda < 0.0:
            raise ValueError("mlp_group_lasso_lambda must be >= 0.")
        if model_type == "cnn":
            self.model = SpectralCNN(cfg, input_length)
        elif model_type == "cnn_fourier":
            self.model = SpectralFourierCNN(cfg, input_length)
        elif model_type == "cnn_dual":
            if not cfg.use_derivative_features:
                raise ValueError(
                    "cnn_dual requires use_derivative_features=True in config."
                )
            if cfg.use_second_derivative_features:
                raise ValueError(
                    "cnn_dual supports raw+first-derivative only. "
                    "Set use_second_derivative_features=False."
                )
            self.model = SpectralDualBranchCNN(cfg, input_length)
        elif model_type == "mlp":
            self.model = SpectralMLP(cfg, input_length)
        elif model_type == "transformer":
            self.model = SpectralTransformer(
                cfg,
                input_length,
                feature_wavenumbers=feature_wavenumbers,
            )
        else:
            raise ValueError(
                "Unsupported model_type="
                f"'{cfg.model_type}'. Use 'cnn', 'cnn_fourier', 'cnn_dual', 'mlp', or 'transformer'."
            )
        self.loss_fn = nn.MSELoss()
        self._val_preds = []
        self._val_targets = []

    def _first_mlp_linear_weight(self):
        """Return first Linear weight in MLP backbone, or None."""
        if self.model_type != "mlp":
            return None
        for layer in self.model.net:
            if isinstance(layer, nn.Linear):
                return layer.weight
        return None

    def _group_lasso_penalty(self, ref_tensor: torch.Tensor) -> torch.Tensor:
        """Group-Lasso over input-feature groups of first MLP layer."""
        if self.group_lasso_lambda <= 0.0:
            return ref_tensor.new_zeros(())
        w = self._first_mlp_linear_weight()
        if w is None:
            return ref_tensor.new_zeros(())
        return torch.norm(w, p=2, dim=0).sum()

    # ----- forward -----
    def forward(self, x):
        return self.model(x)

    # ----- training -----
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        mse_loss = self.loss_fn(pred, y)
        group_lasso = self._group_lasso_penalty(mse_loss)
        loss = mse_loss + self.group_lasso_lambda * group_lasso
        mae = torch.mean(torch.abs(pred - y))
        rmse = torch.sqrt(mse_loss)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_mse", mse_loss, prog_bar=False, on_step=True, on_epoch=True)
        if self.group_lasso_lambda > 0.0 and self.model_type == "mlp":
            self.log(
                "train_group_lasso",
                group_lasso,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
            )
        self.log("train_mae", mae, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train_rmse", rmse, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    # ----- validation -----
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        mae = torch.mean(torch.abs(pred - y))
        rmse = torch.sqrt(loss)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mae", mae, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_rmse", rmse, prog_bar=False, on_step=False, on_epoch=True)
        self._val_preds.append(pred.detach().cpu())
        self._val_targets.append(y.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return

        preds = torch.cat(self._val_preds, dim=0)
        targets = torch.cat(self._val_targets, dim=0)
        self._val_preds.clear()
        self._val_targets.clear()

        ss_res = torch.sum((targets - preds) ** 2, dim=0)
        target_mean = torch.mean(targets, dim=0, keepdim=True)
        ss_tot = torch.sum((targets - target_mean) ** 2, dim=0)
        r2_per_target = 1.0 - ss_res / (ss_tot + 1e-8)
        rmse_per_target = torch.sqrt(torch.mean((targets - preds) ** 2, dim=0))

        metrics = {
            "val_r2_mean": torch.mean(r2_per_target),
            "val_rmse_mean": torch.mean(rmse_per_target),
        }
        for i in range(self.cfg.n_targets):
            metrics[f"val_r2_target_{i+1}"] = r2_per_target[i]
            metrics[f"val_rmse_target_{i+1}"] = rmse_per_target[i]

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)


class SpectralReconstructionModule(_BaseSpectralLightningModule):
    """MLP autoencoder LightningModule for spectral reconstruction."""

    def __init__(self, cfg: Config, input_length: int):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = SpectralReconstructionMLP(cfg, input_length)
        self.loss_fn = nn.MSELoss()
        self._val_preds = []
        self._val_targets = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, target = batch
        pred = self(x)
        loss = self.loss_fn(pred, target)
        mae = torch.mean(torch.abs(pred - target))
        rmse = torch.sqrt(loss)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_mae", mae, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train_rmse", rmse, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        pred = self(x)
        loss = self.loss_fn(pred, target)
        mae = torch.mean(torch.abs(pred - target))
        rmse = torch.sqrt(loss)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mae", mae, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_rmse", rmse, prog_bar=False, on_step=False, on_epoch=True)
        self._val_preds.append(pred.detach().cpu())
        self._val_targets.append(target.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return

        preds = torch.cat(self._val_preds, dim=0)
        targets = torch.cat(self._val_targets, dim=0)
        self._val_preds.clear()
        self._val_targets.clear()

        eps = 1e-8
        target_mean = torch.mean(targets, dim=0, keepdim=True)
        ss_res = torch.sum((targets - preds) ** 2, dim=0)
        ss_tot = torch.sum((targets - target_mean) ** 2, dim=0)
        r2_per_feature = 1.0 - ss_res / (ss_tot + eps)
        rmse_per_feature = torch.sqrt(torch.mean((targets - preds) ** 2, dim=0))

        self.log("val_r2_mean", torch.mean(r2_per_feature), on_step=False, on_epoch=True)
        self.log(
            "val_rmse_mean",
            torch.mean(rmse_per_feature),
            on_step=False,
            on_epoch=True,
        )


class SpectralMaskedModelingModule(_BaseSpectralLightningModule):
    """LightningModule for masked spectra modeling with CNN encoder."""

    def __init__(self, cfg: Config, input_length: int):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = SpectralMaskedModelingCNN(cfg, input_length)
        self.mask_ratio = float(getattr(cfg, "mask_ratio", 0.3))
        self.mask_value = float(getattr(cfg, "mask_value", 0.0))
        if not 0.0 <= self.mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be in [0, 1].")
        self._val_preds = []
        self._val_targets = []
        self._val_masks = []

    def forward(self, x):
        return self.model(x)

    def apply_mask(self, x):
        if self.mask_ratio <= 0.0:
            mask = torch.zeros_like(x, dtype=torch.bool)
        elif self.mask_ratio >= 1.0:
            mask = torch.ones_like(x, dtype=torch.bool)
        else:
            mask = torch.rand_like(x) < self.mask_ratio
        x_masked = x.masked_fill(mask, self.mask_value)
        return x_masked, mask

    @staticmethod
    def _masked_mse_loss(pred, target, mask):
        mask_2d = mask.squeeze(1).float()
        mse = (pred - target) ** 2
        denom = torch.sum(mask_2d).clamp_min(1.0)
        return torch.sum(mse * mask_2d) / denom

    def training_step(self, batch, batch_idx):
        x, target = batch
        x_masked, mask = self.apply_mask(x)
        pred = self(x_masked)
        loss = self._masked_mse_loss(pred, target, mask)
        full_mse = torch.mean((pred - target) ** 2)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "train_full_rmse",
            torch.sqrt(full_mse),
            prog_bar=False,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        x_masked, mask = self.apply_mask(x)
        pred = self(x_masked)
        loss = self._masked_mse_loss(pred, target, mask)
        full_mse = torch.mean((pred - target) ** 2)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "val_full_rmse",
            torch.sqrt(full_mse),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self._val_preds.append(pred.detach().cpu())
        self._val_targets.append(target.detach().cpu())
        self._val_masks.append(mask.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return

        preds = torch.cat(self._val_preds, dim=0)
        targets = torch.cat(self._val_targets, dim=0)
        masks = torch.cat(self._val_masks, dim=0).squeeze(1).bool()
        self._val_preds.clear()
        self._val_targets.clear()
        self._val_masks.clear()

        if torch.any(masks):
            pred_masked = preds[masks]
            target_masked = targets[masks]
            masked_rmse = torch.sqrt(torch.mean((pred_masked - target_masked) ** 2))
            ss_res = torch.sum((target_masked - pred_masked) ** 2)
            target_mean = torch.mean(target_masked)
            ss_tot = torch.sum((target_masked - target_mean) ** 2)
            masked_r2 = 1.0 - ss_res / (ss_tot + 1e-8)
        else:
            masked_rmse = torch.tensor(0.0)
            masked_r2 = torch.tensor(0.0)

        self.log("val_masked_rmse", masked_rmse, on_step=False, on_epoch=True)
        self.log("val_masked_r2", masked_r2, on_step=False, on_epoch=True)
