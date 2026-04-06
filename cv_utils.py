"""
Cross-validation utilities for plotting and ensemble evaluation.
"""
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score

from model import SpectralRegressionModule


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
    cfg,
    model_type: str,
    input_length: int,
    feature_wavenumbers: np.ndarray = None,
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
                path,
                cfg=cfg,
                input_length=input_length,
                feature_wavenumbers=feature_wavenumbers,
                weights_only=False,
            )
            model.eval()
            with torch.no_grad():
                pred = model(X_t.to(model.device)).cpu().numpy()
            preds_list.append(pred)

    ensemble_pred = np.mean(preds_list, axis=0)
    n_targets = y_data.shape[1]
    r2_per_target = [r2_score(y_data[:, i], ensemble_pred[:, i]) for i in range(n_targets)]
    rmse_per_target = [
        np.sqrt(mean_squared_error(y_data[:, i], ensemble_pred[:, i]))
        for i in range(n_targets)
    ]
    return r2_per_target, rmse_per_target
