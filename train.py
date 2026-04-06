"""
Train spectral regression models using K-fold cross-validation.

Usage:
    python train.py
"""
import os
import json
import pickle
import math
from copy import deepcopy
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from dataclasses import asdict
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cross_decomposition import PLSRegression

from checkpoint_utils import average_lightning_checkpoints
from config import Config
from cv_utils import (
    evaluate_ensemble_on_dataset,
    save_all_folds_scatter_plot,
    save_fold_scatter_plot,
)
from dataset import (
    load_raw_data,
    preprocess,
    SpectrumDataModule,
    apply_wavenumber_range,
    apply_savgol_to_base_spectra,
    append_derivative_features,
)
from model import SpectralRegressionModule
from path_utils import normalize_data_paths
from train_io import prepare_run_dir, write_latest_run


def save_wavenumber_selection_plot(
    X_raw: np.ndarray,
    wavenum: np.ndarray,
    selected_mask: np.ndarray,
    out_path: str,
    n_samples: int = 5,
):
    """Visualize selected/non-selected spectral regions for a few samples."""
    if X_raw.shape[0] == 0:
        return

    n_show = min(int(n_samples), X_raw.shape[0])
    sample_indices = np.linspace(0, X_raw.shape[0] - 1, n_show, dtype=int)
    fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.8 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for ax, idx in zip(axes, sample_indices):
        y = X_raw[idx]
        y_sel = np.where(selected_mask, y, np.nan)
        y_non = np.where(~selected_mask, y, np.nan)
        ax.plot(wavenum, y_non, color="#E67E22", lw=1.0, alpha=0.9, label="Not selected")
        ax.plot(wavenum, y_sel, color="#1F77B4", lw=1.2, alpha=0.95, label="Selected")
        ax.set_ylabel(f"Sample {idx}")
        ax.grid(alpha=0.2)

    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Wavenumber")
    fig.suptitle("Wavenumber selection (selected vs not selected)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def augment_with_additive_noise(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
):
    """Create augmented samples by adding Gaussian noise to X."""
    if (not cfg.use_additive_noise_aug) or cfg.additive_noise_copies <= 0:
        return X, y

    copies = int(cfg.additive_noise_copies)
    X_list = [X]
    y_list = [y]
    for _ in range(copies):
        noise = rng.normal(0.0, float(cfg.additive_noise_std), size=X.shape).astype(
            np.float32
        )
        X_list.append((X + noise).astype(np.float32))
        y_list.append(y)
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def oversample_by_target_bins(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
):
    """Oversample sparse target regions using quantile-bin bootstrap."""
    if not bool(getattr(cfg, "use_target_bin_oversampling", False)):
        return X, y, {"enabled": False, "applied": False, "reason": "disabled"}
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if X.shape[0] < 2:
        return X, y, {"enabled": True, "applied": False, "reason": "too_few_samples"}

    target_idx = int(getattr(cfg, "oversampling_target_index", 0))
    if not (0 <= target_idx < y.shape[1]):
        raise ValueError(
            "oversampling_target_index is out of range. "
            f"got {target_idx}, but y has {y.shape[1]} target column(s)."
        )

    n_bins_req = int(getattr(cfg, "oversampling_n_bins", 10))
    if n_bins_req < 2:
        raise ValueError("oversampling_n_bins must be >= 2.")

    target_ratio = float(getattr(cfg, "oversampling_target_ratio", 1.0))
    if not (0.0 < target_ratio <= 1.0):
        raise ValueError("oversampling_target_ratio must be in (0, 1].")

    max_multiplier = float(getattr(cfg, "oversampling_max_multiplier", 2.0))
    if max_multiplier < 1.0:
        raise ValueError("oversampling_max_multiplier must be >= 1.0.")
    noise_std = float(getattr(cfg, "oversampling_feature_noise_std", 0.0))
    if noise_std < 0.0:
        raise ValueError("oversampling_feature_noise_std must be >= 0.")

    y_ref = y[:, target_idx].astype(np.float64)
    edges = np.quantile(y_ref, np.linspace(0.0, 1.0, n_bins_req + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return X, y, {"enabled": True, "applied": False, "reason": "constant_target"}

    # Bin IDs in [0, n_bins_eff - 1].
    bin_ids = np.digitize(y_ref, edges[1:-1], right=False)
    n_bins_eff = edges.size - 1
    counts = np.bincount(bin_ids, minlength=n_bins_eff)
    nonzero_counts = counts[counts > 0]
    if nonzero_counts.size < 2:
        return X, y, {"enabled": True, "applied": False, "reason": "single_nonempty_bin"}

    target_bin_size = int(math.ceil(float(np.max(nonzero_counts)) * target_ratio))
    if target_bin_size <= 0:
        return X, y, {"enabled": True, "applied": False, "reason": "zero_target_bin_size"}

    max_extra = int(math.floor(X.shape[0] * (max_multiplier - 1.0)))
    if max_extra <= 0:
        return X, y, {"enabled": True, "applied": False, "reason": "max_multiplier_limit"}

    extra_chunks = []
    for b in range(n_bins_eff):
        count_b = int(counts[b])
        if count_b == 0 or count_b >= target_bin_size:
            continue
        need = target_bin_size - count_b
        idx_b = np.flatnonzero(bin_ids == b)
        extra_chunks.append(rng.choice(idx_b, size=need, replace=True))

    if not extra_chunks:
        return X, y, {
            "enabled": True,
            "applied": False,
            "reason": "already_balanced",
            "before_n": int(X.shape[0]),
            "after_n": int(X.shape[0]),
        }

    extra_idx = np.concatenate(extra_chunks, axis=0)
    if extra_idx.size > max_extra:
        extra_idx = rng.choice(extra_idx, size=max_extra, replace=False)

    X_extra = X[extra_idx].astype(np.float32, copy=True)
    if noise_std > 0.0:
        # Smoothed bootstrap: add small feature-wise Gaussian noise to replicas.
        feat_std = np.std(X, axis=0, dtype=np.float64).astype(np.float32)
        X_extra += rng.normal(0.0, noise_std, size=X_extra.shape).astype(np.float32) * feat_std

    X_out = np.concatenate([X, X_extra], axis=0).astype(np.float32)
    y_out = np.concatenate([y, y[extra_idx]], axis=0).astype(np.float32)
    return X_out, y_out, {
        "enabled": True,
        "applied": True,
        "target_index": target_idx,
        "n_bins_eff": int(n_bins_eff),
        "min_bin_before": int(np.min(nonzero_counts)),
        "max_bin_before": int(np.max(nonzero_counts)),
        "target_bin_size": int(target_bin_size),
        "noise_std": float(noise_std),
        "before_n": int(X.shape[0]),
        "after_n": int(X_out.shape[0]),
    }


def save_additive_noise_augmentation_plot(
    X: np.ndarray,
    wavenum: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
    out_path: str,
    n_samples: int = 5,
):
    """Save before/after examples of additive-noise augmentation."""
    if (not cfg.use_additive_noise_aug) or X.shape[0] == 0:
        return

    n_show = min(int(n_samples), X.shape[0])
    sample_indices = np.linspace(0, X.shape[0] - 1, n_show, dtype=int)
    X_base = X[sample_indices]
    noise = rng.normal(
        0.0, float(cfg.additive_noise_std), size=X_base.shape
    ).astype(np.float32)
    X_aug = X_base + noise

    fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.8 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for ax, idx, x0, x1 in zip(axes, sample_indices, X_base, X_aug):
        ax.plot(wavenum, x0, color="#1F77B4", lw=1.2, alpha=0.95, label="Original")
        ax.plot(wavenum, x1, color="#D62728", lw=1.0, alpha=0.85, label="Augmented")
        ax.set_ylabel(f"Sample {idx}")
        ax.grid(alpha=0.2)

    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Wavenumber")
    fig.suptitle("Additive-noise augmentation examples")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_spectrum_samples_plot(
    X: np.ndarray,
    wavenum: np.ndarray,
    title: str,
    line_label: str,
    color: str,
    out_path: str,
    n_samples: int = 5,
):
    """Save representative spectra in separate plots."""
    if X.shape[0] == 0:
        return

    n_show = min(int(n_samples), X.shape[0])
    sample_indices = np.linspace(0, X.shape[0] - 1, n_show, dtype=int)
    fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.8 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for ax, idx in zip(axes, sample_indices):
        ax.plot(wavenum, X[idx], color=color, lw=1.2, alpha=0.95, label=line_label)
        ax.set_ylabel(f"Sample {idx}")
        ax.grid(alpha=0.2)

    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Wavenumber")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_target_distribution_before_after_plot(
    y_before: np.ndarray,
    y_after: np.ndarray,
    target_name: str,
    out_path: str,
    n_bins: int = 40,
):
    """Save target distribution histograms before/after oversampling."""
    if y_before.size == 0 or y_after.size == 0:
        return

    y_before = np.asarray(y_before, dtype=np.float64).ravel()
    y_after = np.asarray(y_after, dtype=np.float64).ravel()
    x_min = float(min(np.min(y_before), np.min(y_after)))
    x_max = float(max(np.max(y_before), np.max(y_after)))
    if x_max <= x_min:
        x_max = x_min + 1e-6

    bins = np.linspace(x_min, x_max, int(max(5, n_bins)) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # Use density=True to compare shape fairly when sample counts differ.
    axes[0].hist(y_before, bins=bins, color="#1F77B4", alpha=0.85, density=True)
    axes[0].set_title(f"Before oversampling (n={y_before.size})")
    axes[0].set_ylabel("Density")
    axes[0].grid(alpha=0.2)

    axes[1].hist(y_after, bins=bins, color="#D62728", alpha=0.85, density=True)
    axes[1].set_title(f"After oversampling (n={y_after.size})")
    axes[1].set_ylabel("Density")
    axes[1].set_xlabel(target_name)
    axes[1].grid(alpha=0.2)

    fig.suptitle(f"Target distribution before/after oversampling: {target_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_rmsecv_from_fold_results(fold_results, n_targets: int):
    """Compute exact RMSECV from out-of-fold predictions."""
    if not fold_results:
        raise ValueError("fold_results is empty; cannot compute RMSECV.")

    y_true_all = np.concatenate([r["y_true"] for r in fold_results], axis=0)
    y_pred_all = np.concatenate([r["y_pred"] for r in fold_results], axis=0)
    if y_true_all.shape != y_pred_all.shape:
        raise ValueError(
            "Shape mismatch while computing RMSECV: "
            f"{y_true_all.shape} vs {y_pred_all.shape}"
        )
    if y_true_all.ndim == 1:
        y_true_all = y_true_all[:, None]
        y_pred_all = y_pred_all[:, None]
    if y_true_all.shape[1] != int(n_targets):
        raise ValueError(
            "Target dimension mismatch while computing RMSECV: "
            f"got {y_true_all.shape[1]} vs expected {int(n_targets)}"
        )

    sq_err = (y_true_all.astype(np.float64) - y_pred_all.astype(np.float64)) ** 2
    rmsecv = np.sqrt(np.mean(sq_err, axis=0))
    return rmsecv.tolist(), int(y_true_all.shape[0])


def initialize_model_from_checkpoint(model: SpectralRegressionModule, ckpt_path: str):
    """Initialize model with compatible weights from an existing checkpoint."""
    if not ckpt_path:
        return 0
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"init_checkpoint_path not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Invalid checkpoint format: {ckpt_path}")

    current_state = model.state_dict()
    compatible = {}
    for k, v in state_dict.items():
        if k in current_state and torch.is_tensor(v) and current_state[k].shape == v.shape:
            compatible[k] = v

    current_state.update(compatible)
    model.load_state_dict(current_state, strict=True)
    return len(compatible)


def initialize_mlp_first_layer_by_input_std(
    model: SpectralRegressionModule,
    X_train: np.ndarray,
    eps: float = 1e-12,
):
    """Scale first MLP layer weights by per-feature std from training data."""
    if model.model_type != "mlp":
        return None

    first_linear = None
    for layer in model.model.net:
        if isinstance(layer, torch.nn.Linear):
            first_linear = layer
            break
    if first_linear is None:
        return None

    std_vec = np.std(X_train, axis=0, dtype=np.float64).astype(np.float32)
    std_t = torch.from_numpy(std_vec).to(
        device=first_linear.weight.device,
        dtype=first_linear.weight.dtype,
    )

    mean_std = float(std_t.mean().item())
    if mean_std > eps:
        scale = std_t / (mean_std + eps)
    else:
        scale = torch.ones_like(std_t)

    with torch.no_grad():
        first_linear.weight.mul_(scale.unsqueeze(0))

    return {
        "mean_std": mean_std,
        "min_std": float(std_t.min().item()),
        "max_std": float(std_t.max().item()),
    }


def train_one_fold_nn(
    fold: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
    run_dir: str,
    feature_wavenumbers: np.ndarray = None,
):
    """Train a single fold and return the best checkpoint path + val metrics."""

    input_length = X_train.shape[1]
    dm = SpectrumDataModule(X_train, y_train, X_val, y_val, cfg.batch_size)
    model = SpectralRegressionModule(
        cfg, input_length, feature_wavenumbers=feature_wavenumbers
    )
    init_ckpt = getattr(cfg, "init_checkpoint_path", None)
    use_std_init = bool(getattr(cfg, "mlp_init_first_layer_by_input_std", False))
    if use_std_init and model.model_type == "mlp":
        if init_ckpt:
            print(
                "  Std-based MLP first-layer init skipped "
                "(init_checkpoint_path is set)."
            )
        else:
            stats = initialize_mlp_first_layer_by_input_std(model, X_train)
            if stats is not None:
                print(
                    "  Std-based MLP first-layer init applied "
                    f"(std min/mean/max={stats['min_std']:.4g}/"
                    f"{stats['mean_std']:.4g}/{stats['max_std']:.4g})"
                )
    if init_ckpt:
        loaded = initialize_model_from_checkpoint(model, init_ckpt)
        print(
            f"  Warm start from checkpoint: {init_ckpt} "
            f"(loaded {loaded}/{len(model.state_dict())} tensors)"
        )

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    tb_save_dir = os.path.join(run_dir, cfg.tensorboard_dir)
    logger = TensorBoardLogger(
        save_dir=tb_save_dir,
        name=cfg.experiment_name,
        version=f"fold_{fold + 1}",
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=cfg.early_stop_patience, mode="min"),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"fold{fold}_best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=dm)

    # Load best checkpoint and compute validation metrics
    best_path = callbacks[1].best_model_path
    best_model = SpectralRegressionModule.load_from_checkpoint(
        best_path,
        cfg=cfg,
        input_length=input_length,
        feature_wavenumbers=feature_wavenumbers,
        weights_only=False,
    )
    best_model.eval()
    with torch.no_grad():
        X_val_t = torch.from_numpy(X_val).unsqueeze(1).to(best_model.device)
        preds = best_model(X_val_t).cpu().numpy()

    r2_per_target = [r2_score(y_val[:, i], preds[:, i]) for i in range(cfg.n_targets)]
    rmse_per_target = [
        np.sqrt(mean_squared_error(y_val[:, i], preds[:, i]))
        for i in range(cfg.n_targets)
    ]

    return best_path, r2_per_target, rmse_per_target, logger.log_dir, y_val, preds


def train_one_fold_pls(
    fold: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
    run_dir: str,
):
    """Train one PLS fold and return model path + validation metrics."""
    max_allowed = min(X_train.shape[1], max(X_train.shape[0] - 1, 1))
    n_components = min(int(cfg.pls_n_components), max_allowed)
    if n_components < 1:
        raise ValueError(
            "Invalid PLS components. Please set pls_n_components >= 1 "
            "and ensure training fold has at least 2 samples."
        )

    pls = PLSRegression(
        n_components=n_components,
        scale=bool(cfg.pls_scale),
        max_iter=int(cfg.pls_max_iter),
        tol=float(cfg.pls_tol),
        copy=True,
    )
    pls.fit(X_train, y_train)
    preds = pls.predict(X_val)
    if preds.ndim == 1:
        preds = preds[:, None]

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    best_path = os.path.join(ckpt_dir, f"fold{fold}_pls.pkl")
    with open(best_path, "wb") as f:
        pickle.dump(pls, f)

    r2_per_target = [r2_score(y_val[:, i], preds[:, i]) for i in range(cfg.n_targets)]
    rmse_per_target = [
        np.sqrt(mean_squared_error(y_val[:, i], preds[:, i]))
        for i in range(cfg.n_targets)
    ]
    return best_path, r2_per_target, rmse_per_target, None, y_val, preds


def _load_processed_data_for_path(cfg: Config, data_path: str):
    """Run current data pipeline for one path and return processed arrays."""
    cfg_local = deepcopy(cfg)
    cfg_local.data_path = data_path

    X_raw_all, y, X_eval_raw_all, wavenum_all = load_raw_data(cfg_local)
    X_raw, X_eval_raw, wavenum, selected_mask = apply_wavenumber_range(
        X_raw_all, X_eval_raw_all, wavenum_all, cfg_local
    )
    X_raw, X_eval_raw, wavenum, selected_mask = _apply_std_top_r_wavenumber_selection(
        X_raw,
        X_eval_raw,
        wavenum,
        selected_mask,
        cfg_local,
    )
    # Apply SG to the original spectra first, then build optional extra features.
    X_sg, X_eval_sg = apply_savgol_to_base_spectra(X_raw, X_eval_raw, cfg_local)
    X_feat, X_eval_feat = append_derivative_features(X_sg, X_eval_sg, cfg_local)
    X, X_eval, _ = preprocess(X_feat, X_eval_feat, cfg_local, apply_savgol=False)
    return X_raw_all, X_raw, y, X_eval, X, wavenum_all, selected_mask, wavenum


def _apply_std_top_r_wavenumber_selection(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    wavenum: np.ndarray,
    selected_mask: np.ndarray,
    cfg: Config,
):
    """Optionally keep top-r% wavelengths by training-set std."""
    r = getattr(cfg, "std_top_r_percent", None)
    if r is None:
        return X_train, X_eval, wavenum, selected_mask

    r = float(r)
    if not (0.0 < r <= 100.0):
        raise ValueError("std_top_r_percent must be in (0, 100].")

    n_features = int(X_train.shape[1])
    if n_features < 1:
        raise ValueError("No features available for std-based selection.")

    n_keep = max(1, int(math.ceil(n_features * (r / 100.0))))
    if n_keep >= n_features:
        keep_local = np.ones(n_features, dtype=bool)
    else:
        std_vec = np.std(X_train, axis=0, dtype=np.float64)
        order = np.argsort(-std_vec, kind="mergesort")
        keep_idx = order[:n_keep]
        keep_local = np.zeros(n_features, dtype=bool)
        keep_local[keep_idx] = True

    selected_idx = np.flatnonzero(selected_mask)
    if selected_idx.size != n_features:
        raise ValueError(
            "Internal mismatch in wavenumber masks during std-based selection."
        )
    selected_mask_out = np.zeros_like(selected_mask, dtype=bool)
    selected_mask_out[selected_idx[keep_local]] = True

    return (
        X_train[:, keep_local],
        X_eval[:, keep_local],
        wavenum[keep_local],
        selected_mask_out,
    )


def _build_feature_wavenumbers(cfg: Config, wavenum: np.ndarray) -> np.ndarray:
    """Expand selected wavenumbers to match feature construction order."""
    parts = [wavenum]
    if cfg.use_derivative_features:
        parts.append(wavenum)
    if cfg.use_second_derivative_features:
        parts.append(wavenum)
    return np.concatenate(parts, axis=0).astype(np.float32)


def main():
    cfg = Config()
    model_type = str(cfg.model_type).lower()
    if model_type not in {"cnn", "cnn_fourier", "cnn_dual", "mlp", "transformer", "pls"}:
        raise ValueError(
            f"Unsupported model_type='{cfg.model_type}'. "
            "Use 'cnn', 'cnn_fourier', 'cnn_dual', 'mlp', 'transformer', or 'pls'."
        )
    if model_type == "pls" and getattr(cfg, "init_checkpoint_path", None):
        print("Warning: init_checkpoint_path is ignored when model_type='pls'.")

    std_top_r_percent = getattr(cfg, "std_top_r_percent", None)
    if std_top_r_percent is not None:
        std_top_r_percent = float(std_top_r_percent)
        if not (0.0 < std_top_r_percent <= 100.0):
            raise ValueError("std_top_r_percent must be in (0, 100].")
        if model_type in {"cnn", "cnn_fourier", "cnn_dual", "transformer"}:
            print(
                "Warning: std_top_r_percent is mainly intended for model_type='mlp'. "
                "For CNN/Transformer, selecting only high-std wavelengths may break "
                "local receptive-field/patch assumptions."
            )

    pl.seed_everything(cfg.seed, workers=True)
    # Make PyTorch behavior deterministic as much as possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = prepare_run_dir(cfg.output_dir, run_id, cfg.tensorboard_dir)
    write_latest_run(cfg.output_dir, "latest_run.txt", run_dir)

    # Load and preprocess data (supports one or many paths)
    data_paths = normalize_data_paths(cfg.data_path)
    X_raw_all_list, X_raw_list, y_list, X_eval_list, X_list = [], [], [], [], []
    wavenum_all_ref = None
    selected_mask_ref = None
    wavenum_ref = None

    for i, path in enumerate(data_paths):
        (
            X_raw_all_i,
            X_raw_i,
            y_i,
            X_eval_i,
            X_i,
            wavenum_all_i,
            selected_mask_i,
            wavenum_i,
        ) = _load_processed_data_for_path(cfg, path)

        if i == 0:
            wavenum_all_ref = wavenum_all_i
            selected_mask_ref = selected_mask_i
            wavenum_ref = wavenum_i
        else:
            if X_i.shape[1] != X_list[0].shape[1]:
                raise ValueError(
                    f"Feature dimension mismatch across data_path. "
                    f"'{path}' gives {X_i.shape[1]} but first path gives {X_list[0].shape[1]}."
                )
            if not np.array_equal(wavenum_all_i, wavenum_all_ref):
                raise ValueError(
                    f"Wavenumber axis mismatch across data_path. "
                    f"All paths must have the same spectral columns to concatenate."
                )
            if not np.array_equal(selected_mask_i, selected_mask_ref):
                raise ValueError(
                    f"Selected wavenumber mask mismatch across data_path. "
                    f"Check wavenumber_min/max and source data columns."
                )
            if not np.array_equal(wavenum_i, wavenum_ref):
                raise ValueError(
                    f"Selected wavenumber values mismatch across data_path."
                )

        X_raw_all_list.append(X_raw_all_i)
        X_raw_list.append(X_raw_i)
        y_list.append(y_i)
        X_eval_list.append(X_eval_i)
        X_list.append(X_i)

    X_raw_all = np.concatenate(X_raw_all_list, axis=0)
    X_raw = np.concatenate(X_raw_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    X_eval = np.concatenate(X_eval_list, axis=0)
    X = np.concatenate(X_list, axis=0)
    wavenum_all = wavenum_all_ref
    selected_mask = selected_mask_ref
    wavenum = wavenum_ref
    input_length = X.shape[1]
    feature_wavenumbers = _build_feature_wavenumbers(cfg, wavenum)
    if feature_wavenumbers.shape[0] != input_length:
        raise ValueError(
            "Feature wavenumber length mismatch. "
            f"got {feature_wavenumbers.shape[0]} vs input_length={input_length}"
        )

    model_txt = os.path.join(run_dir, "artifacts", "model_structure.txt")
    with open(model_txt, "w") as f:
        if model_type in {"cnn", "cnn_fourier", "cnn_dual", "mlp", "transformer"}:
            model_for_export = SpectralRegressionModule(
                cfg,
                input_length,
                feature_wavenumbers=feature_wavenumbers,
            )
            f.write(str(model_for_export) + "\n")
        else:
            max_allowed_global = min(X.shape[1], max(X.shape[0] - 1, 1))
            effective_global = min(int(cfg.pls_n_components), max_allowed_global)
            f.write("PLSRegression model\n")
            f.write(f"requested_n_components={cfg.pls_n_components}\n")
            f.write(f"effective_n_components_upper_bound={effective_global}\n")
            f.write(f"scale={cfg.pls_scale}\n")
            f.write(f"max_iter={cfg.pls_max_iter}\n")
            f.write(f"tol={cfg.pls_tol}\n")

    hparams = asdict(cfg)
    hparams.update(
        {
            "run_id": run_id,
            "run_dir": run_dir,
            "resolved_data_paths": data_paths,
            "input_length": int(input_length),
            "n_train_samples": int(X.shape[0]),
            "n_eval_samples": int(X_eval.shape[0]),
        }
    )
    hparams_json = os.path.join(run_dir, "artifacts", "hparams.json")
    with open(hparams_json, "w") as f:
        json.dump(hparams, f, indent=2, ensure_ascii=False)

    print(f"Training samples : {X.shape[0]}")
    print(f"Spectral features: {X.shape[1]} (selected from {X_raw_all.shape[1]})")
    if cfg.sg_window and int(cfg.sg_window) > 1:
        print(
            "Savitzky-Golay: "
            f"deriv={int(getattr(cfg, 'sg_deriv', 0))}, "
            f"window={int(cfg.sg_window)}, "
            f"polyorder={int(cfg.sg_polyorder)}"
        )
    else:
        print("Savitzky-Golay: disabled")
    if cfg.std_top_r_percent is not None:
        print(
            "Std-based wavelength selection: "
            f"top {float(cfg.std_top_r_percent):g}% by train-set std"
        )
        print(f"Selected base wavelengths: {wavenum.shape[0]}")
    if cfg.use_derivative_features or cfg.use_second_derivative_features:
        parts = ["original"]
        if cfg.use_derivative_features:
            parts.append("first derivative")
        if cfg.use_second_derivative_features:
            parts.append("second derivative")
        print(f"Feature construction: {' + '.join(parts)}")
    if cfg.use_target_bin_oversampling:
        print(
            "Target-bin oversampling: "
            f"target_index={int(cfg.oversampling_target_index)}, "
            f"bins={int(cfg.oversampling_n_bins)}, "
            f"target_ratio={float(cfg.oversampling_target_ratio):g}, "
            f"max_multiplier={float(cfg.oversampling_max_multiplier):g}, "
            f"feature_noise_std={float(cfg.oversampling_feature_noise_std):g}"
        )
    else:
        print("Target-bin oversampling: disabled")
    if getattr(cfg, "eval_sheet", None):
        print(f"Evaluation samples: {X_eval.shape[0]}")
    else:
        print("Evaluation samples: 0 (eval_sheet is None)")
    print(f"Model type: {model_type}")
    print(f"Targets: {cfg.target_cols}")
    if len(data_paths) > 1:
        print(f"Data paths: {len(data_paths)} paths merged")
        for p in data_paths:
            print(f"  - {p}")
    else:
        print(f"Data path: {data_paths[0]}")
    if cfg.wavenumber_min is None and cfg.wavenumber_max is None:
        print("Wavenumber range: full range")
    else:
        print(
            "Wavenumber range: "
            f"[{float(np.min(wavenum)):.2f}, {float(np.max(wavenum)):.2f}]"
        )
    print(f"Run directory: {run_dir}")
    print()

    # K-fold cross-validation
    kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    all_r2, all_rmse = [], []
    best_paths = []
    tb_log_dirs = []
    fold_plot_results = []
    plots_dir = os.path.join(run_dir, "artifacts", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    selection_plot_path = os.path.join(plots_dir, "wavenumber_selection_samples.pdf")
    save_wavenumber_selection_plot(
        X_raw_all,
        wavenum_all,
        selected_mask,
        selection_plot_path,
        n_samples=cfg.range_plot_n_samples,
    )
    noise_aug_plot_path = os.path.join(
        plots_dir, "additive_noise_augmentation_samples.pdf"
    )
    save_additive_noise_augmentation_plot(
        X_raw,
        wavenum,
        cfg,
        np.random.default_rng(cfg.seed),
        noise_aug_plot_path,
        n_samples=cfg.range_plot_n_samples,
    )
    savgol_before_plot_path = os.path.join(plots_dir, "savgol_before_samples.pdf")
    savgol_after_plot_path = os.path.join(plots_dir, "savgol_after_samples.pdf")
    sg_enabled = bool(cfg.sg_window and int(cfg.sg_window) > 1)
    if sg_enabled:
        X_sg_plot, _ = apply_savgol_to_base_spectra(
            X_raw,
            np.empty((0, X_raw.shape[1]), dtype=np.float32),
            cfg,
        )
        save_spectrum_samples_plot(
            X_raw,
            wavenum,
            "Before Savitzky-Golay preprocessing",
            "Before SG",
            "#1F77B4",
            savgol_before_plot_path,
            n_samples=cfg.range_plot_n_samples,
        )
        save_spectrum_samples_plot(
            X_sg_plot,
            wavenum,
            "After Savitzky-Golay preprocessing "
            f"(deriv={int(getattr(cfg, 'sg_deriv', 0))}, "
            f"window={int(cfg.sg_window)}, polyorder={int(cfg.sg_polyorder)})",
            "After SG",
            "#D62728",
            savgol_after_plot_path,
            n_samples=cfg.range_plot_n_samples,
        )

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        # Re-seed per fold for stable initialization/shuffling across runs.
        pl.seed_everything(cfg.seed + fold, workers=True)
        print(f"\n{'='*60}")
        print(f"  Fold {fold + 1} / {cfg.n_splits}")
        print(f"{'='*60}")

        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        fold_rng = np.random.default_rng(cfg.seed + fold)
        X_tr_bal, y_tr_bal, os_info = oversample_by_target_bins(
            X_tr, y_tr, cfg, fold_rng
        )
        X_tr_fit, y_tr_fit = augment_with_additive_noise(
            X_tr_bal, y_tr_bal, cfg, fold_rng
        )
        if cfg.use_target_bin_oversampling:
            target_idx_plot = int(getattr(cfg, "oversampling_target_index", 0))
            if 0 <= target_idx_plot < len(cfg.target_cols):
                target_name_plot = cfg.target_cols[target_idx_plot]
            else:
                target_name_plot = f"target[{target_idx_plot}]"
            dist_plot_path = os.path.join(
                plots_dir, f"fold_{fold + 1}_target_oversampling_dist.pdf"
            )
            save_target_distribution_before_after_plot(
                y_before=y_tr[:, target_idx_plot],
                y_after=y_tr_bal[:, target_idx_plot],
                target_name=target_name_plot,
                out_path=dist_plot_path,
            )
        if os_info.get("applied", False):
            target_idx_print = int(os_info["target_index"])
            if 0 <= target_idx_print < len(cfg.target_cols):
                target_name = cfg.target_cols[target_idx_print]
            else:
                target_name = f"target[{target_idx_print}]"
            print(
                "  Target-bin oversampling: "
                f"{os_info['before_n']} -> {os_info['after_n']} samples "
                f"(target={target_name}, "
                f"bins={os_info['n_bins_eff']}, "
                f"bin min/max before={os_info['min_bin_before']}/"
                f"{os_info['max_bin_before']}, "
                f"target_bin_size={os_info['target_bin_size']}, "
                f"feature_noise_std={os_info['noise_std']:g})"
            )
        elif cfg.use_target_bin_oversampling:
            print(
                "  Target-bin oversampling skipped "
                f"({os_info.get('reason', 'no_action')})."
            )

        if model_type == "pls":
            path, r2, rmse, tb_log_dir, y_true, y_pred = train_one_fold_pls(
                fold, X_tr_fit, y_tr_fit, X_va, y_va, cfg, run_dir
            )
            max_allowed = min(X_tr_fit.shape[1], max(X_tr_fit.shape[0] - 1, 1))
            effective_components = min(int(cfg.pls_n_components), max_allowed)
            print(
                "  PLS components: "
                f"requested={cfg.pls_n_components}, effective={effective_components}"
            )
        else:
            path, r2, rmse, tb_log_dir, y_true, y_pred = train_one_fold_nn(
                fold,
                X_tr_fit,
                y_tr_fit,
                X_va,
                y_va,
                cfg,
                run_dir,
                feature_wavenumbers=feature_wavenumbers,
            )
        if cfg.use_additive_noise_aug and cfg.additive_noise_copies > 0:
            print(
                "  Additive-noise augmentation: "
                f"{X_tr_bal.shape[0]} -> {X_tr_fit.shape[0]} samples "
                f"(std={cfg.additive_noise_std}, copies={cfg.additive_noise_copies})"
            )
        best_paths.append(path)
        if tb_log_dir is not None:
            tb_log_dirs.append(tb_log_dir)
        all_r2.append(r2)
        all_rmse.append(rmse)
        fold_plot_results.append({"fold": fold, "y_true": y_true, "y_pred": y_pred})

        fold_plot_path = os.path.join(plots_dir, f"fold_{fold + 1}_pred_vs_true.pdf")
        save_fold_scatter_plot(y_true, y_pred, cfg.target_cols, fold, fold_plot_path)

        for i, name in enumerate(cfg.target_cols):
            print(f"  {name}:  R²={r2[i]:.4f}  RMSE={rmse[i]:.4f}")

    # Summary
    rmsecv_vals, n_oof = compute_rmsecv_from_fold_results(
        fold_plot_results, cfg.n_targets
    )

    summary_lines = [
        f"{'='*60}",
        "  Cross-validation summary (mean ± std + RMSECV)",
        f"{'='*60}",
        f"  OOF samples used for RMSECV: {n_oof}",
    ]
    for i, name in enumerate(cfg.target_cols):
        r2_vals = [r[i] for r in all_r2]
        rmse_vals = [r[i] for r in all_rmse]
        summary_lines.append(
            f"  {name}:  R²={np.mean(r2_vals):.4f}±{np.std(r2_vals):.4f}  "
            f"RMSE={np.mean(rmse_vals):.4f}±{np.std(rmse_vals):.4f}  "
            f"RMSECV={rmsecv_vals[i]:.4f}"
        )
    print()
    for line in summary_lines:
        print(line)

    # Evaluate ensemble of fold checkpoints on full training dataset
    full_train_r2, full_train_rmse = evaluate_ensemble_on_dataset(
        best_paths=best_paths,
        X_data=X,
        y_data=y,
        cfg=cfg,
        model_type=model_type,
        input_length=input_length,
        feature_wavenumbers=feature_wavenumbers,
    )
    full_train_lines = [
        f"{'='*60}",
        "  Ensemble evaluation on full training data",
        f"{'='*60}",
    ]
    for i, name in enumerate(cfg.target_cols):
        full_train_lines.append(
            f"  {name}:  R²={full_train_r2[i]:.4f}  RMSE={full_train_rmse[i]:.4f}"
        )
    print()
    for line in full_train_lines:
        print(line)

    summary_file = os.path.join(run_dir, "artifacts", "cv_summary.txt")
    with open(summary_file, "w") as f:
        f.write("\n".join(summary_lines + [""] + full_train_lines) + "\n")

    all_folds_plot_path = os.path.join(plots_dir, "all_folds_pred_vs_true.pdf")
    save_all_folds_scatter_plot(fold_plot_results, cfg.target_cols, all_folds_plot_path)

    # Save the list of best checkpoint paths for ensemble prediction
    paths_file = os.path.join(run_dir, "best_checkpoints.txt")
    with open(paths_file, "w") as f:
        for p in best_paths:
            f.write(p + "\n")

    avg_ckpt_path = None
    if model_type in {"cnn", "cnn_fourier", "cnn_dual", "mlp"}:
        avg_ckpt_path = os.path.join(run_dir, "checkpoints", "folds_avg.ckpt")
        average_lightning_checkpoints(best_paths, avg_ckpt_path)

    print(f"\nCheckpoint paths saved to {paths_file}")
    if avg_ckpt_path:
        print(f"Averaged checkpoint saved to {avg_ckpt_path}")
    print(f"Model structure saved to {model_txt}")
    print(f"Hyperparameters saved to {hparams_json}")
    print(f"Cross-validation summary saved to {summary_file}")
    print(f"Fold scatter plots saved to {plots_dir}")
    print(f"Combined fold scatter plot saved to {all_folds_plot_path}")
    print(f"Wavenumber selection plot saved to {selection_plot_path}")
    if cfg.use_target_bin_oversampling:
        print(
            "Oversampling distribution plots saved to "
            f"{plots_dir}/fold_*_target_oversampling_dist.pdf"
        )
    if sg_enabled:
        print(f"Savitzky-Golay before plot saved to {savgol_before_plot_path}")
        print(f"Savitzky-Golay after plot saved to {savgol_after_plot_path}")
    if cfg.use_additive_noise_aug:
        print(f"Additive-noise augmentation plot saved to {noise_aug_plot_path}")
    if tb_log_dirs:
        print("TensorBoard log dirs:")
        for d in tb_log_dirs:
            print(f"  - {d}")
    else:
        print("TensorBoard log dirs: (not used for PLS)")


if __name__ == "__main__":
    main()
