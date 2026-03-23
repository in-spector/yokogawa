"""
Train spectral regression models using K-fold cross-validation.

Usage:
    python train.py
"""
import os
import json
import pickle
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
from dataset import (
    load_raw_data,
    preprocess,
    SpectrumDataModule,
    apply_wavenumber_range,
    append_derivative_features,
)
from model import SpectralRegressionModule
from path_utils import normalize_data_paths
from train_io import prepare_run_dir, write_latest_run


def save_fold_scatter_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols,
    fold: int,
    out_path: str,
):
    """Save one pred-vs-true scatter image for a single fold."""
    n_targets = len(target_cols)
    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4))
    if n_targets == 1:
        axes = [axes]

    for i, name in enumerate(target_cols):
        ax = axes[i]
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        min_v = min(float(np.min(true_i)), float(np.min(pred_i)))
        max_v = max(float(np.max(true_i)), float(np.max(pred_i)))
        ax.scatter(true_i, pred_i, s=20, alpha=0.8)
        ax.plot([min_v, max_v], [min_v, max_v], "r--", lw=1)
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.set_title(name)

    fig.suptitle(f"Fold {fold + 1}: Pred vs True")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_all_folds_scatter_plot(fold_results, target_cols, out_path: str):
    """Save one pred-vs-true scatter image that overlays all folds."""
    n_targets = len(target_cols)
    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4))
    if n_targets == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(target_cols):
        ax = axes[i]
        all_true = np.concatenate([r["y_true"][:, i] for r in fold_results], axis=0)
        all_pred = np.concatenate([r["y_pred"][:, i] for r in fold_results], axis=0)
        min_v = min(float(np.min(all_true)), float(np.min(all_pred)))
        max_v = max(float(np.max(all_true)), float(np.max(all_pred)))

        for fold_idx, fold_result in enumerate(fold_results):
            ax.scatter(
                fold_result["y_true"][:, i],
                fold_result["y_pred"][:, i],
                s=18,
                alpha=0.6,
                color=cmap(fold_idx % 10),
                label=f"Fold {fold_result['fold'] + 1}",
            )
        ax.plot([min_v, max_v], [min_v, max_v], "k--", lw=1)
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.set_title(name)

    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("All Folds: Pred vs True")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate_ensemble_on_dataset(
    best_paths,
    X_data: np.ndarray,
    y_data: np.ndarray,
    cfg: Config,
    model_type: str,
    input_length: int,
):
    """Evaluate fold-ensemble (prediction average) on a given dataset."""
    preds_list = []

    if model_type == "pls":
        for path in best_paths:
            with open(path, "rb") as f:
                pls = pickle.load(f)
            pred = pls.predict(X_data)
            if pred.ndim == 1:
                pred = pred[:, None]
            preds_list.append(pred)
    else:
        X_t = torch.from_numpy(X_data).unsqueeze(1)
        for path in best_paths:
            model = SpectralRegressionModule.load_from_checkpoint(
                path, cfg=cfg, input_length=input_length, weights_only=False
            )
            model.eval()
            with torch.no_grad():
                pred = model(X_t.to(model.device)).cpu().numpy()
            preds_list.append(pred)

    ensemble_pred = np.mean(preds_list, axis=0)
    r2_per_target = [
        r2_score(y_data[:, i], ensemble_pred[:, i]) for i in range(cfg.n_targets)
    ]
    rmse_per_target = [
        np.sqrt(mean_squared_error(y_data[:, i], ensemble_pred[:, i]))
        for i in range(cfg.n_targets)
    ]
    return r2_per_target, rmse_per_target


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


def train_one_fold_nn(
    fold: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
    run_dir: str,
):
    """Train a single fold and return the best checkpoint path + val metrics."""

    input_length = X_train.shape[1]
    dm = SpectrumDataModule(X_train, y_train, X_val, y_val, cfg.batch_size)
    model = SpectralRegressionModule(cfg, input_length)
    init_ckpt = getattr(cfg, "init_checkpoint_path", None)
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
        best_path, cfg=cfg, input_length=input_length, weights_only=False
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
    X_feat, X_eval_feat = append_derivative_features(X_raw, X_eval_raw, cfg_local)
    X, X_eval, _ = preprocess(X_feat, X_eval_feat, cfg_local)
    return X_raw_all, X_raw, y, X_eval, X, wavenum_all, selected_mask, wavenum


def main():
    cfg = Config()
    model_type = str(cfg.model_type).lower()
    if model_type not in {"cnn", "cnn_dual", "mlp", "pls"}:
        raise ValueError(
            f"Unsupported model_type='{cfg.model_type}'. "
            "Use 'cnn', 'cnn_dual', 'mlp', or 'pls'."
        )
    if model_type == "pls" and getattr(cfg, "init_checkpoint_path", None):
        print("Warning: init_checkpoint_path is ignored when model_type='pls'.")

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

    model_txt = os.path.join(run_dir, "artifacts", "model_structure.txt")
    with open(model_txt, "w") as f:
        if model_type in {"cnn", "cnn_dual", "mlp"}:
            model_for_export = SpectralRegressionModule(cfg, input_length)
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
    if cfg.use_derivative_features or cfg.use_second_derivative_features:
        parts = ["original"]
        if cfg.use_derivative_features:
            parts.append("first derivative")
        if cfg.use_second_derivative_features:
            parts.append("second derivative")
        print(f"Feature construction: {' + '.join(parts)}")
    print(f"Evaluation samples: {X_eval.shape[0]}")
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

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        # Re-seed per fold for stable initialization/shuffling across runs.
        pl.seed_everything(cfg.seed + fold, workers=True)
        print(f"\n{'='*60}")
        print(f"  Fold {fold + 1} / {cfg.n_splits}")
        print(f"{'='*60}")

        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        X_tr_fit, y_tr_fit = augment_with_additive_noise(
            X_tr, y_tr, cfg, np.random.default_rng(cfg.seed + fold)
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
                fold, X_tr_fit, y_tr_fit, X_va, y_va, cfg, run_dir
            )
        if cfg.use_additive_noise_aug and cfg.additive_noise_copies > 0:
            print(
                "  Additive-noise augmentation: "
                f"{X_tr.shape[0]} -> {X_tr_fit.shape[0]} samples "
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
    summary_lines = [
        f"{'='*60}",
        "  Cross-validation summary (mean ± std)",
        f"{'='*60}",
    ]
    for i, name in enumerate(cfg.target_cols):
        r2_vals = [r[i] for r in all_r2]
        rmse_vals = [r[i] for r in all_rmse]
        summary_lines.append(
            f"  {name}:  R²={np.mean(r2_vals):.4f}±{np.std(r2_vals):.4f}  "
            f"RMSE={np.mean(rmse_vals):.4f}±{np.std(rmse_vals):.4f}"
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
    if model_type in {"cnn", "cnn_dual", "mlp"}:
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
