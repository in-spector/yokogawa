"""
Train CNN-based Masked Spectra Modeling (MSM) using K-fold CV.

Usage:
    python train_masked_spectra_modeling.py
"""
import json
import os
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from config_masked_spectra_modeling import MaskedSpectraModelingConfig
from checkpoint_utils import average_lightning_checkpoints
from dataset import SpectrumDataModule, load_reconstruction_raw_data, preprocess
from model import SpectralMaskedModelingModule
from train_io import prepare_run_dir, write_latest_run


def save_fold_waveform_plot(
    x_true: np.ndarray,
    x_masked: np.ndarray,
    x_pred: np.ndarray,
    wavenum: np.ndarray,
    fold: int,
    out_path: str,
    n_examples: int = 4,
):
    """Save masked/reconstructed/true waveform overlays for one fold."""
    if wavenum is None:
        raise ValueError("wavenum must be provided for waveform plotting.")
    x_axis = np.asarray(wavenum, dtype=np.float32).reshape(-1)
    if x_axis.shape[0] != x_true.shape[1]:
        raise ValueError(
            "wavenum length does not match spectral feature length: "
            f"{x_axis.shape[0]} vs {x_true.shape[1]}"
        )

    n_show = min(n_examples, x_true.shape[0])
    fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.8 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i in range(n_show):
        axes[i].plot(x_axis, x_true[i], label="True", lw=1.2, color="#1F77B4")
        axes[i].plot(
            x_axis,
            x_masked[i],
            label="Masked input",
            lw=1.0,
            alpha=0.85,
            color="#7F7F7F",
        )
        axes[i].plot(
            x_axis,
            x_pred[i],
            label="Reconstructed",
            lw=1.0,
            alpha=0.85,
            color="#D62728",
        )
        axes[i].set_ylabel(f"Sample {i}")
        axes[i].grid(alpha=0.2)
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Wavenumber")

    fig.suptitle(f"Fold {fold + 1}: Masked spectra modeling")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def train_one_fold(
    fold: int,
    X_train: np.ndarray,
    X_val: np.ndarray,
    cfg: MaskedSpectraModelingConfig,
    run_dir: str,
):
    """Train one MSM fold and return best checkpoint + metrics."""
    input_length = X_train.shape[1]

    dm = SpectrumDataModule(
        X_train=X_train,
        y_train=X_train,
        X_val=X_val,
        y_val=X_val,
        batch_size=cfg.batch_size,
    )
    model = SpectralMaskedModelingModule(cfg, input_length)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    tb_save_dir = os.path.join(run_dir, cfg.tensorboard_dir)
    logger = TensorBoardLogger(
        save_dir=tb_save_dir,
        name=f"{cfg.experiment_name}_msm",
        version=f"fold_{fold + 1}",
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=cfg.early_stop_patience, mode="min"),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"msm_fold{fold}_best",
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

    best_path = callbacks[1].best_model_path
    best_model = SpectralMaskedModelingModule.load_from_checkpoint(
        best_path, cfg=cfg, input_length=input_length, weights_only=False
    )
    best_model.eval()

    with torch.no_grad():
        X_val_t = torch.from_numpy(X_val).unsqueeze(1).to(best_model.device)
        X_masked_t, mask_t = best_model.apply_mask(X_val_t)
        preds = best_model(X_masked_t).cpu().numpy()
        x_masked = X_masked_t.squeeze(1).cpu().numpy()
        mask_np = mask_t.squeeze(1).cpu().numpy().astype(bool)

    flat_true_full = X_val.reshape(-1)
    flat_pred_full = preds.reshape(-1)
    full_rmse = np.sqrt(mean_squared_error(flat_true_full, flat_pred_full))

    if np.any(mask_np):
        masked_true = X_val[mask_np]
        masked_pred = preds[mask_np]
        masked_r2 = r2_score(masked_true, masked_pred)
        masked_rmse = np.sqrt(mean_squared_error(masked_true, masked_pred))
    else:
        masked_r2 = np.nan
        masked_rmse = np.nan

    return (
        best_path,
        masked_r2,
        masked_rmse,
        full_rmse,
        logger.log_dir,
        X_val,
        x_masked,
        preds,
    )


def main():
    cfg = MaskedSpectraModelingConfig()
    pl.seed_everything(cfg.seed, workers=True)
    run_id = datetime.now().strftime("msm_%Y%m%d_%H%M%S")
    run_dir = prepare_run_dir(cfg.output_dir, run_id, cfg.tensorboard_dir)
    write_latest_run(cfg.output_dir, "latest_run_msm.txt", run_dir)

    X_raw, X_eval_raw, wavenum = load_reconstruction_raw_data(cfg)
    if X_eval_raw.shape[0] > 0:
        X, X_eval, _ = preprocess(X_raw, X_eval_raw, cfg)
    else:
        X, _, _ = preprocess(X_raw, X_raw[:1], cfg)
        X_eval = np.empty((0, X.shape[1]), dtype=np.float32)
    input_length = X.shape[1]

    model_for_export = SpectralMaskedModelingModule(cfg, input_length)
    model_txt = os.path.join(run_dir, "artifacts", "model_structure.txt")
    with open(model_txt, "w") as f:
        f.write(str(model_for_export) + "\n")

    hparams = asdict(cfg)
    hparams.update(
        {
            "task": "masked_spectra_modeling",
            "run_id": run_id,
            "run_dir": run_dir,
            "input_length": int(input_length),
            "n_train_samples": int(X.shape[0]),
            "n_eval_samples": int(X_eval.shape[0]),
        }
    )
    hparams_json = os.path.join(run_dir, "artifacts", "hparams.json")
    with open(hparams_json, "w") as f:
        json.dump(hparams, f, indent=2, ensure_ascii=False)

    print(f"Training samples : {X.shape[0]}")
    print(f"Wavenumber count : {len(wavenum)}")
    print(f"Spectral features: {X.shape[1]}")
    print(f"Evaluation samples: {X_eval.shape[0]}")
    print(f"Mask ratio: {cfg.mask_ratio}")
    print(f"Run directory: {run_dir}")
    print()

    kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    all_masked_r2, all_masked_rmse, all_full_rmse = [], [], []
    best_paths = []
    tb_log_dirs = []

    plots_dir = os.path.join(run_dir, "artifacts", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n{'='*60}")
        print(f"  MSM Fold {fold + 1} / {cfg.n_splits}")
        print(f"{'='*60}")

        X_tr, X_va = X[train_idx], X[val_idx]
        path, masked_r2, masked_rmse, full_rmse, tb_log_dir, x_true, x_masked, x_pred = (
            train_one_fold(fold, X_tr, X_va, cfg, run_dir)
        )

        best_paths.append(path)
        tb_log_dirs.append(tb_log_dir)
        all_masked_r2.append(masked_r2)
        all_masked_rmse.append(masked_rmse)
        all_full_rmse.append(full_rmse)

        fold_plot_path = os.path.join(plots_dir, f"fold_{fold + 1}_msm_waveform.pdf")
        save_fold_waveform_plot(
            x_true,
            x_masked,
            x_pred,
            wavenum,
            fold,
            fold_plot_path,
        )

        print(
            f"  Fold {fold + 1}: masked R²={masked_r2:.4f}  "
            f"masked RMSE={masked_rmse:.4f}  full RMSE={full_rmse:.4f}"
        )

    summary_lines = [
        f"{'='*60}",
        "  MSM CV summary (mean ± std)",
        f"{'='*60}",
        (
            "  masked R²="
            f"{np.nanmean(all_masked_r2):.4f}±{np.nanstd(all_masked_r2):.4f}"
        ),
        (
            "  masked RMSE="
            f"{np.nanmean(all_masked_rmse):.4f}±{np.nanstd(all_masked_rmse):.4f}"
        ),
        f"  full RMSE={np.mean(all_full_rmse):.4f}±{np.std(all_full_rmse):.4f}",
    ]
    print()
    for line in summary_lines:
        print(line)

    summary_file = os.path.join(run_dir, "artifacts", "cv_summary.txt")
    with open(summary_file, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    paths_file = os.path.join(run_dir, "best_checkpoints_msm.txt")
    with open(paths_file, "w") as f:
        for p in best_paths:
            f.write(p + "\n")

    averaged_ckpt_path = os.path.join(run_dir, "checkpoints", "msm_fold_average.ckpt")
    average_lightning_checkpoints(best_paths, averaged_ckpt_path)

    print(f"\nCheckpoint paths saved to {paths_file}")
    print(f"Averaged checkpoint saved to {averaged_ckpt_path}")
    print(f"Model structure saved to {model_txt}")
    print(f"Hyperparameters saved to {hparams_json}")
    print(f"Cross-validation summary saved to {summary_file}")
    print(f"Waveform plots saved to {plots_dir}")
    print("TensorBoard log dirs:")
    for d in tb_log_dirs:
        print(f"  - {d}")


if __name__ == "__main__":
    main()
