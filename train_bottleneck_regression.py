"""
Train target regression using bottleneck features from a pretrained encoder.
"""
import json
import os
import pickle
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from config_bottleneck import BottleneckConfig
from checkpoint_utils import average_lightning_checkpoints
from dataset import SpectrumDataModule, preprocess, resolve_excel_files
from model import (
    SpectralMaskedModelingModule,
    SpectralReconstructionModule,
    SpectralRegressionModule,
)
from train_io import prepare_run_dir


def resolve_ckpt_from_latest(
    output_dir: str,
    latest_run_filename: str,
    checkpoints_filename: str,
    missing_hint: str,
) -> str:
    latest_run_file = os.path.join(output_dir, latest_run_filename)
    if not os.path.isfile(latest_run_file):
        raise FileNotFoundError(missing_hint)
    with open(latest_run_file) as f:
        run_dir = f.readline().strip()
    if not run_dir or not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Invalid run dir in {latest_run_file}: {run_dir}")

    paths_file = os.path.join(run_dir, checkpoints_filename)
    if not os.path.isfile(paths_file):
        raise FileNotFoundError(f"Checkpoint list not found: {paths_file}")
    with open(paths_file) as f:
        paths = [line.strip() for line in f if line.strip()]
    if not paths:
        raise FileNotFoundError(f"No checkpoints listed in: {paths_file}")
    return paths[0]


def resolve_encoder_ckpt(cfg: BottleneckConfig) -> str:
    task = str(getattr(cfg, "encoder_pretrain_task", "reconstruction")).lower()

    if task == "msm":
        if cfg.msm_ckpt_path:
            if not os.path.isfile(cfg.msm_ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {cfg.msm_ckpt_path}")
            return cfg.msm_ckpt_path
        return resolve_ckpt_from_latest(
            output_dir=cfg.msm_output_dir,
            latest_run_filename="latest_run_msm.txt",
            checkpoints_filename="best_checkpoints_msm.txt",
            missing_hint=(
                "latest_run_msm.txt not found. Set msm_ckpt_path in "
                "config_bottleneck.py or run train_masked_spectra_modeling.py."
            ),
        )

    if task == "reconstruction":
        if cfg.recon_ckpt_path:
            if not os.path.isfile(cfg.recon_ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {cfg.recon_ckpt_path}")
            return cfg.recon_ckpt_path
        return resolve_ckpt_from_latest(
            output_dir=cfg.reconstruction_output_dir,
            latest_run_filename="latest_run_reconstruction.txt",
            checkpoints_filename="best_checkpoints_reconstruction.txt",
            missing_hint=(
                "latest_run_reconstruction.txt not found. Set recon_ckpt_path in "
                "config_bottleneck.py or run train_reconstruction.py."
            ),
        )

    raise ValueError(
        "Unsupported encoder_pretrain_task="
        f"'{cfg.encoder_pretrain_task}'. Use 'msm' or 'reconstruction'."
    )


def load_supervised_data(cfg: BottleneckConfig):
    """Load spectra + targets from train_sheet (targets can be in target_sheet)."""
    file_paths = resolve_excel_files(cfg.data_path)
    if not file_paths:
        raise FileNotFoundError(f"No xlsx files found: {cfg.data_path}")

    spectra_dfs = []
    target_dfs = []
    common_spec_cols = None
    for file_path in file_paths:
        try:
            spec_df = pd.read_excel(file_path, sheet_name=cfg.train_sheet)
            target_df = (
                pd.read_excel(file_path, sheet_name=cfg.target_sheet)
                if cfg.target_sheet
                else spec_df
            )
        except Exception as exc:
            print(f"[WARN] Skipped file '{file_path}': {exc}")
            continue

        spec_cols = [c for c in spec_df.columns if isinstance(c, (int, float))]
        if not spec_cols:
            print(
                f"[WARN] Skipped file '{file_path}': no spectral columns in "
                f"'{cfg.train_sheet}'"
            )
            continue
        if any(col not in target_df.columns for col in cfg.target_cols):
            print(
                f"[WARN] Skipped file '{file_path}': target columns not found in "
                f"'{cfg.target_sheet or cfg.train_sheet}'"
            )
            continue
        if len(spec_df) != len(target_df):
            print(
                f"[WARN] Skipped file '{file_path}': row count mismatch between "
                f"'{cfg.train_sheet}' and '{cfg.target_sheet or cfg.train_sheet}'"
            )
            continue

        if common_spec_cols is None:
            common_spec_cols = set(spec_cols)
        else:
            common_spec_cols &= set(spec_cols)

        spectra_dfs.append(spec_df)
        target_dfs.append(target_df)

    if not spectra_dfs:
        raise RuntimeError("No valid files found for supervised bottleneck training.")
    if not common_spec_cols:
        raise RuntimeError("No common spectral columns across valid files.")

    spec_cols = sorted(common_spec_cols, reverse=True)
    X = np.concatenate(
        [df[spec_cols].values.astype(np.float32) for df in spectra_dfs], axis=0
    )
    y = np.concatenate(
        [df[cfg.target_cols].values.astype(np.float32) for df in target_dfs], axis=0
    )
    return X, y


def encode_with_pretrained_encoder(
    X: np.ndarray,
    cfg: BottleneckConfig,
    ckpt_path: str,
) -> np.ndarray:
    task = str(getattr(cfg, "encoder_pretrain_task", "reconstruction")).lower()
    if task == "msm":
        model = SpectralMaskedModelingModule.load_from_checkpoint(
            ckpt_path,
            cfg=cfg,
            input_length=X.shape[1],
            weights_only=False,
        )
    elif task == "reconstruction":
        model = SpectralReconstructionModule.load_from_checkpoint(
            ckpt_path,
            cfg=cfg,
            input_length=X.shape[1],
            weights_only=False,
        )
    else:
        raise ValueError(
            "Unsupported encoder_pretrain_task="
            f"'{cfg.encoder_pretrain_task}'. Use 'msm' or 'reconstruction'."
        )
    model.eval()
    model.cpu()

    feats = []
    bs = int(cfg.encoder_batch_size)
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            xb = torch.from_numpy(X[i:i + bs]).unsqueeze(1)
            zb = model.model.encoder(xb)
            if zb.ndim > 2:
                zb = torch.flatten(zb, start_dim=1)
            feats.append(zb.cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def train_one_fold_nn(
    fold: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: BottleneckConfig,
    run_dir: str,
):
    input_length = X_train.shape[1]
    dm = SpectrumDataModule(X_train, y_train, X_val, y_val, cfg.batch_size)
    model = SpectralRegressionModule(cfg, input_length)

    logger = TensorBoardLogger(
        save_dir=os.path.join(run_dir, cfg.tensorboard_dir),
        name=cfg.experiment_name,
        version=f"fold_{fold + 1}",
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=cfg.early_stop_patience, mode="min"),
        ModelCheckpoint(
            dirpath=os.path.join(run_dir, "checkpoints"),
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

    best_path = callbacks[1].best_model_path
    best_model = SpectralRegressionModule.load_from_checkpoint(
        best_path, cfg=cfg, input_length=input_length, weights_only=False
    )
    best_model.eval()
    with torch.no_grad():
        pred = best_model(torch.from_numpy(X_val).unsqueeze(1).to(best_model.device))
        pred = pred.cpu().numpy()

    r2 = [r2_score(y_val[:, i], pred[:, i]) for i in range(cfg.n_targets)]
    rmse = [
        np.sqrt(mean_squared_error(y_val[:, i], pred[:, i])) for i in range(cfg.n_targets)
    ]
    return best_path, r2, rmse, logger.log_dir


def train_one_fold_pls(
    fold: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: BottleneckConfig,
    run_dir: str,
):
    max_allowed = min(X_train.shape[1], max(X_train.shape[0] - 1, 1))
    n_components = min(int(cfg.pls_n_components), max_allowed)
    if n_components < 1:
        raise ValueError("Invalid PLS components for bottleneck training.")

    model = PLSRegression(
        n_components=n_components,
        scale=bool(cfg.pls_scale),
        max_iter=int(cfg.pls_max_iter),
        tol=float(cfg.pls_tol),
        copy=True,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    if pred.ndim == 1:
        pred = pred[:, None]

    best_path = os.path.join(run_dir, "checkpoints", f"fold{fold}_pls.pkl")
    with open(best_path, "wb") as f:
        pickle.dump(model, f)

    r2 = [r2_score(y_val[:, i], pred[:, i]) for i in range(cfg.n_targets)]
    rmse = [
        np.sqrt(mean_squared_error(y_val[:, i], pred[:, i])) for i in range(cfg.n_targets)
    ]
    return best_path, r2, rmse, None


def main():
    cfg = BottleneckConfig()
    model_type = str(cfg.model_type).lower()
    if model_type not in {"pls", "cnn", "mlp"}:
        raise ValueError(
            f"Unsupported model_type='{cfg.model_type}'. Use 'pls', 'cnn', or 'mlp'."
        )

    pl.seed_everything(cfg.seed, workers=True)
    run_id = datetime.now().strftime("bottleneck_%Y%m%d_%H%M%S")
    run_dir = prepare_run_dir(cfg.output_dir, run_id, cfg.tensorboard_dir)

    ckpt_path = resolve_encoder_ckpt(cfg)
    X_raw, y = load_supervised_data(cfg)
    X_enc_in, _, _ = preprocess(X_raw, X_raw[:1], cfg)
    X = encode_with_pretrained_encoder(X_enc_in, cfg, ckpt_path)

    model_txt = os.path.join(run_dir, "artifacts", "model_structure.txt")
    with open(model_txt, "w") as f:
        if model_type in {"cnn", "mlp"}:
            f.write(str(SpectralRegressionModule(cfg, X.shape[1])) + "\n")
        else:
            f.write("PLSRegression on bottleneck features\n")
            f.write(f"requested_n_components={cfg.pls_n_components}\n")

    hparams = asdict(cfg)
    hparams.update(
        {
            "task": "bottleneck_regression",
            "run_id": run_id,
            "run_dir": run_dir,
            "encoder_checkpoint": ckpt_path,
            "encoder_pretrain_task": cfg.encoder_pretrain_task,
            "input_length": int(X.shape[1]),
            "n_train_samples": int(X.shape[0]),
        }
    )
    with open(os.path.join(run_dir, "artifacts", "hparams.json"), "w") as f:
        json.dump(hparams, f, indent=2, ensure_ascii=False)

    print(f"Encoder pretraining task: {cfg.encoder_pretrain_task}")
    print(f"Encoder checkpoint: {ckpt_path}")
    print(f"Bottleneck features shape: {X.shape}")
    print(f"Targets: {cfg.target_cols}")
    print(f"Model type: {model_type}")
    print(f"Run directory: {run_dir}")

    kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    all_r2, all_rmse, best_paths, tb_log_dirs = [], [], [], []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        if model_type == "pls":
            path, r2, rmse, tb_log_dir = train_one_fold_pls(
                fold, X_tr, y_tr, X_va, y_va, cfg, run_dir
            )
        else:
            path, r2, rmse, tb_log_dir = train_one_fold_nn(
                fold, X_tr, y_tr, X_va, y_va, cfg, run_dir
            )

        best_paths.append(path)
        if tb_log_dir is not None:
            tb_log_dirs.append(tb_log_dir)
        all_r2.append(r2)
        all_rmse.append(rmse)

        print(f"\nFold {fold + 1}/{cfg.n_splits}")
        for i, name in enumerate(cfg.target_cols):
            print(f"  {name}: R²={r2[i]:.4f}  RMSE={rmse[i]:.4f}")

    summary_lines = ["Cross-validation summary (mean ± std)"]
    for i, name in enumerate(cfg.target_cols):
        r2_vals = [r[i] for r in all_r2]
        rmse_vals = [r[i] for r in all_rmse]
        summary_lines.append(
            f"{name}: R²={np.mean(r2_vals):.4f}±{np.std(r2_vals):.4f}  "
            f"RMSE={np.mean(rmse_vals):.4f}±{np.std(rmse_vals):.4f}"
        )
    print()
    for line in summary_lines:
        print(line)

    with open(os.path.join(run_dir, "best_checkpoints_bottleneck.txt"), "w") as f:
        for p in best_paths:
            f.write(p + "\n")
    with open(os.path.join(run_dir, "artifacts", "cv_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    np.savez_compressed(
        os.path.join(run_dir, "artifacts", "bottleneck_train.npz"),
        X_bottleneck=X,
        y=y,
        target_cols=np.array(cfg.target_cols, dtype=object),
    )

    if model_type != "pls":
        averaged_ckpt_path = os.path.join(
            run_dir, "checkpoints", "bottleneck_fold_average.ckpt"
        )
        average_lightning_checkpoints(best_paths, averaged_ckpt_path)
        print(f"Averaged checkpoint: {averaged_ckpt_path}")
    else:
        print("Averaged checkpoint: skipped for PLS model_type.")

    if tb_log_dirs:
        print("TensorBoard log dirs:")
        for d in tb_log_dirs:
            print(f"  - {d}")


if __name__ == "__main__":
    main()
