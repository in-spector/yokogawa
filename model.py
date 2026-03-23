"""
Spectral regression models (CNN/MLP), wrapped in a PyTorch Lightning module.
"""
import math
import torch
import torch.nn as nn
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
    """Wraps selectable backbone (CNN/MLP) with training / validation logic."""

    def __init__(self, cfg: Config, input_length: int):
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
        else:
            raise ValueError(
                "Unsupported model_type="
                f"'{cfg.model_type}'. Use 'cnn', 'cnn_dual', or 'mlp'."
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
