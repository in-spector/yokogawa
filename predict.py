"""
Generate predictions for the evaluation set using an ensemble of K-fold models.

Usage:
    python predict.py
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch

from config import Config
from dataset import (
    load_raw_data,
    preprocess,
    apply_wavenumber_range,
    append_derivative_features,
    load_eval_sample_ids,
)
from model import SpectralRegressionModule


def resolve_run_dir(cfg: Config) -> str:
    """Return run directory to use for checkpoints/predictions."""
    latest_run_file = os.path.join(cfg.output_dir, "latest_run.txt")
    if os.path.exists(latest_run_file):
        with open(latest_run_file) as f:
            run_dir = f.readline().strip()
        if run_dir and os.path.isdir(run_dir):
            return run_dir
    return cfg.output_dir


def main():
    cfg = Config()
    model_type = str(cfg.model_type).lower()
    if model_type not in {"cnn", "cnn_dual", "mlp", "pls"}:
        raise ValueError(
            f"Unsupported model_type='{cfg.model_type}'. "
            "Use 'cnn', 'cnn_dual', 'mlp', or 'pls'."
        )
    os.makedirs(cfg.output_dir, exist_ok=True)
    run_dir = resolve_run_dir(cfg)

    # Load and preprocess
    X_raw_all, _, X_eval_raw_all, wavenum_all = load_raw_data(cfg)
    X_raw, X_eval_raw, _, _ = apply_wavenumber_range(
        X_raw_all, X_eval_raw_all, wavenum_all, cfg
    )
    X_feat, X_eval_feat = append_derivative_features(X_raw, X_eval_raw, cfg)
    X_train, X_eval, _ = preprocess(X_feat, X_eval_feat, cfg)
    input_length = X_train.shape[1]

    # Load checkpoint paths
    paths_file = os.path.join(run_dir, "best_checkpoints.txt")
    with open(paths_file) as f:
        ckpt_paths = [line.strip() for line in f if line.strip()]

    print(f"Ensembling {len(ckpt_paths)} models...")

    preds_list = []

    if model_type == "pls":
        for path in ckpt_paths:
            with open(path, "rb") as f:
                pls = pickle.load(f)
            pred = pls.predict(X_eval)
            if pred.ndim == 1:
                pred = pred[:, None]
            preds_list.append(pred)
    else:
        # Ensemble prediction (average) for neural models
        X_eval_t = torch.from_numpy(X_eval).unsqueeze(1)  # (N, 1, L)
        for path in ckpt_paths:
            model = SpectralRegressionModule.load_from_checkpoint(
                path, cfg=cfg, input_length=input_length, weights_only=False
            )
            model.eval()
            with torch.no_grad():
                pred = model(X_eval_t.to(model.device)).cpu().numpy()
            preds_list.append(pred)

    # Mean of all fold predictions
    ensemble_pred = np.mean(preds_list, axis=0)

    # Read evaluation sample IDs (supports merged directory input)
    eval_ids = load_eval_sample_ids(cfg)
    if len(eval_ids) != ensemble_pred.shape[0]:
        raise RuntimeError(
            "Mismatch between number of evaluation IDs and predictions: "
            f"{len(eval_ids)} vs {ensemble_pred.shape[0]}"
        )

    # Build output DataFrame
    result = pd.DataFrame()
    result["SampleID"] = eval_ids
    for i, name in enumerate(cfg.target_cols):
        result[name] = ensemble_pred[:, i]

    out_path = os.path.join(run_dir, cfg.predictions_file)
    result.to_excel(out_path, index=False)
    print(f"\nUsing run directory: {run_dir}")
    print(f"Predictions saved to {out_path}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
