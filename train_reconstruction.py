"""
Train MLP autoencoder models for spectral reconstruction using K-fold CV.

Usage:
    python train_reconstruction.py
"""
import os
import json
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import asdict
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from config_reconstruction import ReconstructionConfig
from dataset import load_reconstruction_raw_data, preprocess, SpectrumDataModule
from model import SpectralReconstructionModule
from train_io import prepare_run_dir, write_latest_run


def save_fold_waveform_plot(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    wavenum: np.ndarray,
    fold: int,
    out_path: str,
    n_examples: int = 4,
):
    """Save reconstructed-vs-true waveform overlays for one fold."""
    n_show = min(n_examples, x_true.shape[0])
    fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.8 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    x_axis = np.arange(x_true.shape[1]) if wavenum is None else wavenum
    for i in range(n_show):
        axes[i].plot(x_axis, x_true[i], label="True", lw=1.2)
        axes[i].plot(x_axis, x_pred[i], label="Reconstructed", lw=1.0, alpha=0.85)
        axes[i].set_ylabel(f"Sample {i}")
        axes[i].grid(alpha=0.2)
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Wavenumber" if wavenum is not None else "Index")

    fig.suptitle(f"Fold {fold + 1}: Waveform reconstruction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def train_one_fold(
    fold: int,
    X_train: np.ndarray,
    X_val: np.ndarray,
    cfg: ReconstructionConfig,
    run_dir: str,
):
    """Train one reconstruction fold and return best checkpoint + metrics."""
    input_length = X_train.shape[1]

    dm = SpectrumDataModule(
        X_train=X_train,
        y_train=X_train,
        X_val=X_val,
        y_val=X_val,
        batch_size=cfg.batch_size,
    )
    model = SpectralReconstructionModule(cfg, input_length)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    tb_save_dir = os.path.join(run_dir, cfg.tensorboard_dir)
    logger = TensorBoardLogger(
        save_dir=tb_save_dir,
        name=f"{cfg.experiment_name}_reconstruction",
        version=f"fold_{fold + 1}",
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=cfg.early_stop_patience, mode="min"),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"recon_fold{fold}_best",
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
    best_model = SpectralReconstructionModule.load_from_checkpoint(
        best_path, cfg=cfg, input_length=input_length, weights_only=False
    )
    best_model.eval()

    with torch.no_grad():
        X_val_t = torch.from_numpy(X_val).unsqueeze(1).to(best_model.device)
        preds = best_model(X_val_t).cpu().numpy()

    flat_true = X_val.reshape(-1)
    flat_pred = preds.reshape(-1)
    r2 = r2_score(flat_true, flat_pred)
    rmse = np.sqrt(mean_squared_error(flat_true, flat_pred))

    return best_path, r2, rmse, logger.log_dir, X_val, preds


def main():
    cfg = ReconstructionConfig()
    pl.seed_everything(cfg.seed, workers=True)
    run_id = datetime.now().strftime("reconstruction_%Y%m%d_%H%M%S")
    run_dir = prepare_run_dir(cfg.output_dir, run_id, cfg.tensorboard_dir)
    write_latest_run(cfg.output_dir, "latest_run_reconstruction.txt", run_dir)

    X_raw, X_eval_raw, wavenum = load_reconstruction_raw_data(cfg)
    if X_eval_raw.shape[0] > 0:
        X, X_eval, _ = preprocess(X_raw, X_eval_raw, cfg)
    else:
        # Fit scaler on train set only when eval sheet is not provided.
        X, _, _ = preprocess(X_raw, X_raw[:1], cfg)
        X_eval = np.empty((0, X.shape[1]), dtype=np.float32)
    input_length = X.shape[1]

    model_for_export = SpectralReconstructionModule(cfg, input_length)
    model_txt = os.path.join(run_dir, "artifacts", "model_structure.txt")
    with open(model_txt, "w") as f:
        f.write(str(model_for_export) + "\n")

    hparams = asdict(cfg)
    hparams.update(
        {
            "task": "reconstruction",
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
    print(f"Spectral features: {X.shape[1]}")
    print(f"Evaluation samples: {X_eval.shape[0]}")
    print(f"Run directory: {run_dir}")
    print()

    kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    all_r2, all_rmse = [], []
    best_paths = []
    tb_log_dirs = []

    plots_dir = os.path.join(run_dir, "artifacts", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n{'='*60}")
        print(f"  Reconstruction Fold {fold + 1} / {cfg.n_splits}")
        print(f"{'='*60}")

        X_tr, X_va = X[train_idx], X[val_idx]
        path, r2, rmse, tb_log_dir, x_true, x_pred = train_one_fold(
            fold, X_tr, X_va, cfg, run_dir
        )

        best_paths.append(path)
        tb_log_dirs.append(tb_log_dir)
        all_r2.append(r2)
        all_rmse.append(rmse)

        fold_plot_path = os.path.join(plots_dir, f"fold_{fold + 1}_waveform_recon.pdf")
        save_fold_waveform_plot(x_true, x_pred, wavenum, fold, fold_plot_path)

        print(f"  Fold {fold + 1}: R²={r2:.4f}  RMSE={rmse:.4f}")

    summary_lines = [
        f"{'='*60}",
        "  Reconstruction CV summary (mean ± std)",
        f"{'='*60}",
        f"  R²={np.mean(all_r2):.4f}±{np.std(all_r2):.4f}",
        f"  RMSE={np.mean(all_rmse):.4f}±{np.std(all_rmse):.4f}",
    ]
    print()
    for line in summary_lines:
        print(line)

    summary_file = os.path.join(run_dir, "artifacts", "cv_summary.txt")
    with open(summary_file, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    paths_file = os.path.join(run_dir, "best_checkpoints_reconstruction.txt")
    with open(paths_file, "w") as f:
        for p in best_paths:
            f.write(p + "\n")

    print(f"\nCheckpoint paths saved to {paths_file}")
    print(f"Model structure saved to {model_txt}")
    print(f"Hyperparameters saved to {hparams_json}")
    print(f"Cross-validation summary saved to {summary_file}")
    print(f"Waveform plots saved to {plots_dir}")
    print("TensorBoard log dirs:")
    for d in tb_log_dirs:
        print(f"  - {d}")


if __name__ == "__main__":
    main()
