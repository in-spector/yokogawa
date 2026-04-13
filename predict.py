"""
Generate predictions for the evaluation set using an ensemble of K-fold models.

Usage:
    python predict.py
    python predict.py --run_dir /home/member/cao/yokogawa/outputs/DS1/20260404_150257
"""
import argparse
import glob
import json
import os
import math
import pickle
import re
from datetime import datetime
from dataclasses import fields
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from dataset import (
    load_raw_data,
    preprocess,
    apply_wavenumber_range,
    apply_savgol_to_base_spectra,
    append_derivative_features,
    load_eval_sample_ids,
    resolve_excel_files,
)
from model import SpectralRegressionModule


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Training run directory to load checkpoints/hparams from.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Override data_path used to load train/eval sheets for prediction.",
    )
    parser.add_argument(
        "--train_sheet",
        type=str,
        default=None,
        help="Override train sheet name used for preprocessing fit.",
    )
    parser.add_argument(
        "--eval_sheet",
        type=str,
        default=None,
        help="Override evaluation sheet name used for prediction targets.",
    )
    parser.add_argument(
        "--save_pred_to_xlsx",
        action="store_true",
        help=(
            "Save a copy of the source xlsx file(s) with prediction columns "
            "added to the evaluation sheet."
        ),
    )
    parser.add_argument(
        "--pred_xlsx_suffix",
        type=str,
        default="_with_predictions",
        help="Suffix added to copied xlsx filenames when --save_pred_to_xlsx is used.",
    )
    return parser.parse_args()


def resolve_run_dir(base_cfg: Config, run_dir_arg: str | None) -> str:
    """Resolve run directory from arg/config/latest pointer."""
    if run_dir_arg:
        run_dir = os.path.abspath(run_dir_arg)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
        return run_dir

    config_run_dir = getattr(base_cfg, "predict_run_dir", None)
    if config_run_dir:
        run_dir = os.path.abspath(config_run_dir)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"predict_run_dir does not exist: {run_dir}")
        return run_dir

    latest_run_file = os.path.join(base_cfg.output_dir, "latest_run.txt")
    if os.path.exists(latest_run_file):
        with open(latest_run_file) as f:
            run_dir = f.readline().strip()
        if run_dir and os.path.isdir(run_dir):
            return os.path.abspath(run_dir)
    return os.path.abspath(base_cfg.output_dir)


def load_cfg_from_run_hparams(run_dir: str):
    """Load config values from run artifacts/hparams.json."""
    hparams_path = os.path.join(run_dir, "artifacts", "hparams.json")
    if not os.path.isfile(hparams_path):
        raise FileNotFoundError(f"hparams.json not found: {hparams_path}")

    with open(hparams_path) as f:
        hparams = json.load(f)

    cfg = Config()
    valid_keys = {f.name for f in fields(Config)}
    loaded_keys = []
    for key, value in hparams.items():
        if key in valid_keys:
            setattr(cfg, key, value)
            loaded_keys.append(key)
    return cfg, hparams_path, loaded_keys


def _fold_sort_key(path: str):
    """Sort checkpoints by fold index, then filename."""
    name = os.path.basename(path)
    match = re.search(r"fold(\d+)_", name)
    fold = int(match.group(1)) if match else 10**9
    return (fold, name)


def discover_fold_checkpoint_paths(run_dir: str, model_type: str):
    """Auto-discover fold checkpoints under run_dir/checkpoints."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    if model_type == "pls":
        pattern = os.path.join(ckpt_dir, "fold*_pls.pkl")
    else:
        pattern = os.path.join(ckpt_dir, "fold*_best*.ckpt")

    ckpt_paths = sorted(glob.glob(pattern), key=_fold_sort_key)
    if not ckpt_paths:
        raise FileNotFoundError(
            f"No fold checkpoints found in {ckpt_dir} (pattern={os.path.basename(pattern)})."
        )
    return ckpt_paths


def _build_feature_wavenumbers(cfg: Config, wavenum: np.ndarray) -> np.ndarray:
    """Expand selected wavenumbers to match feature construction order."""
    parts = [wavenum]
    if cfg.use_derivative_features:
        parts.append(wavenum)
    if cfg.use_second_derivative_features:
        parts.append(wavenum)
    return np.concatenate(parts, axis=0).astype(np.float32)


def prepare_prediction_artifact_dirs(run_dir: str):
    """Create a timestamped prediction artifact directory under the run."""
    predict_root = os.path.join(run_dir, "artifacts", "predict")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(predict_root, timestamp)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    latest_predict_file = os.path.join(predict_root, "latest_predict.txt")
    with open(latest_predict_file, "w") as f:
        f.write(out_dir + "\n")

    return out_dir, plots_dir, latest_predict_file


def save_prediction_distribution_plot(
    result_df: pd.DataFrame,
    target_cols,
    out_path: str,
):
    """Save per-target prediction distributions as one PDF figure."""
    n_targets = len(target_cols)
    if n_targets < 1:
        return

    n_cols = 1 if n_targets == 1 else 2
    n_rows = int(math.ceil(n_targets / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.8 * n_rows))
    axes = np.array(axes, dtype=object).reshape(-1)

    for i, name in enumerate(target_cols):
        ax = axes[i]
        values = pd.to_numeric(result_df[name], errors="coerce").dropna().to_numpy()
        if values.size == 0:
            ax.text(0.5, 0.5, "No valid predictions", ha="center", va="center")
            ax.set_title(str(name))
            ax.set_xlabel("Predicted value")
            ax.set_ylabel("Count")
            ax.grid(alpha=0.2)
            continue

        bins = min(40, max(10, int(np.sqrt(values.size))))
        ax.hist(
            values,
            bins=bins,
            color="#1F77B4",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )
        mean_v = float(np.mean(values))
        median_v = float(np.median(values))
        ax.axvline(mean_v, color="#D62728", lw=1.6, label=f"mean={mean_v:.4g}")
        ax.axvline(
            median_v,
            color="#2CA02C",
            lw=1.4,
            ls="--",
            label=f"median={median_v:.4g}",
        )
        ax.set_title(str(name))
        ax.set_xlabel("Predicted value")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=8)

    for j in range(n_targets, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Prediction distributions on evaluation dataset")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_prediction_readme(
    out_dir: str,
    predictions_path: str,
    plot_path: str,
):
    """Write a short artifact index for easier navigation."""
    readme_path = os.path.join(out_dir, "README.txt")
    lines = [
        "Prediction artifacts",
        "",
        f"- predictions : {os.path.basename(predictions_path)}",
        f"- plots       : {os.path.relpath(plot_path, out_dir)}",
        "",
        "Structure:",
        f"{os.path.basename(out_dir)}/",
        f"  {os.path.basename(predictions_path)}",
        "  plots/",
        f"    {os.path.basename(plot_path)}",
    ]
    with open(readme_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return readme_path


def _get_common_train_spec_cols(cfg: Config):
    """Collect train-sheet spectral columns shared across all valid source files."""
    file_paths = resolve_excel_files(cfg.data_path)
    train_spec_cols_list = []

    for file_path in file_paths:
        try:
            train_df = pd.read_excel(file_path, sheet_name=cfg.train_sheet)
        except Exception as exc:
            print(f"[WARN] Train sheet load failed for '{file_path}': {exc}")
            continue

        spec_cols = [c for c in train_df.columns if isinstance(c, (int, float))]
        if spec_cols:
            train_spec_cols_list.append(spec_cols)

    if not train_spec_cols_list:
        raise RuntimeError("No valid train-sheet spectral columns found for XLSX export.")

    common_cols = set(train_spec_cols_list[0])
    for spec_cols in train_spec_cols_list[1:]:
        common_cols &= set(spec_cols)
    if not common_cols:
        raise RuntimeError("No common spectral columns found for XLSX export.")

    return sorted(common_cols, reverse=True)


def save_predictions_to_source_xlsx(
    cfg: Config,
    pred_mean: np.ndarray,
    suffix: str,
):
    """Save copied xlsx files with prediction columns added to eval_sheet."""
    eval_sheet = getattr(cfg, "eval_sheet", None)
    if not eval_sheet:
        raise ValueError("--save_pred_to_xlsx requires eval_sheet to be set.")

    file_paths = resolve_excel_files(cfg.data_path)
    common_spec_cols = _get_common_train_spec_cols(cfg)
    pred_offset = 0
    saved_paths = []

    for file_path in file_paths:
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        eval_df = all_sheets.get(eval_sheet)
        if eval_df is None:
            print(f"[WARN] Eval sheet '{eval_sheet}' not found in '{file_path}'. Skipped.")
            continue

        missing_cols = [c for c in common_spec_cols if c not in eval_df.columns]
        if missing_cols:
            print(
                f"[WARN] Eval sheet in '{file_path}' is missing required spectral columns. Skipped."
            )
            continue

        valid_mask = ~eval_df.loc[:, common_spec_cols].apply(
            pd.to_numeric, errors="coerce"
        ).isna().any(axis=1)
        valid_index = eval_df.index[valid_mask]
        n_valid = len(valid_index)
        next_offset = pred_offset + n_valid
        if next_offset > pred_mean.shape[0]:
            raise RuntimeError(
                f"Prediction row count is insufficient for '{file_path}': "
                f"need {next_offset}, got {pred_mean.shape[0]}."
            )

        updated_eval_df = eval_df.copy()
        for i, name in enumerate(cfg.target_cols):
            updated_eval_df[name] = np.nan
            updated_eval_df.loc[valid_index, name] = pred_mean[pred_offset:next_offset, i]
        all_sheets[eval_sheet] = updated_eval_df

        root, ext = os.path.splitext(str(file_path))
        out_path = f"{root}{suffix}{ext}"
        with pd.ExcelWriter(out_path) as writer:
            for sheet_name, sheet_df in all_sheets.items():
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

        saved_paths.append(out_path)
        pred_offset = next_offset

    if pred_offset != pred_mean.shape[0]:
        raise RuntimeError(
            "Not all predictions were written back to source xlsx files: "
            f"used {pred_offset}, total {pred_mean.shape[0]}."
        )

    return saved_paths


def main():
    args = parse_args()
    base_cfg = Config()
    run_dir = resolve_run_dir(base_cfg, args.run_dir)
    cfg, hparams_path, loaded_hparams_keys = load_cfg_from_run_hparams(run_dir)

    # Optional command-line overrides for evaluation data source.
    if args.data_path is not None:
        cfg.data_path = args.data_path
    if args.train_sheet is not None:
        cfg.train_sheet = args.train_sheet
    if args.eval_sheet is not None:
        cfg.eval_sheet = args.eval_sheet

    model_type = str(cfg.model_type).lower()
    if model_type not in {"cnn", "cnn_fourier", "cnn_dual", "mlp", "transformer", "pls"}:
        raise ValueError(
            f"Unsupported model_type='{cfg.model_type}'. "
            "Use 'cnn', 'cnn_fourier', 'cnn_dual', 'mlp', 'transformer', or 'pls'."
        )
    os.makedirs(run_dir, exist_ok=True)

    # Load and preprocess
    X_raw_all, _, X_eval_raw_all, wavenum_all = load_raw_data(cfg)
    X_raw, X_eval_raw, _, wavenum = apply_wavenumber_range(
        X_raw_all, X_eval_raw_all, wavenum_all, cfg
    )
    # Keep train/predict pipelines consistent: SG on base spectra first.
    X_sg, X_eval_sg = apply_savgol_to_base_spectra(X_raw, X_eval_raw, cfg)
    X_feat, X_eval_feat = append_derivative_features(X_sg, X_eval_sg, cfg)
    X_train, X_eval, _ = preprocess(X_feat, X_eval_feat, cfg, apply_savgol=False)
    input_length = X_train.shape[1]
    feature_wavenumbers = _build_feature_wavenumbers(cfg, wavenum)

    # Auto-discover fold checkpoints under run_dir/checkpoints
    ckpt_paths = discover_fold_checkpoint_paths(run_dir, model_type)

    print(f"Ensembling {len(ckpt_paths)} models...")
    print(f"Run directory: {run_dir}")
    print(f"Loaded hparams: {hparams_path} ({len(loaded_hparams_keys)} keys)")
    if any(v is not None for v in [args.data_path, args.train_sheet, args.eval_sheet]):
        print("CLI overrides:")
        print(f"  data_path  : {cfg.data_path}")
        print(f"  train_sheet: {cfg.train_sheet}")
        print(f"  eval_sheet : {cfg.eval_sheet}")
    print("Checkpoint paths:")
    for p in ckpt_paths:
        print(f"  - {p}")
    if int(getattr(cfg, "n_splits", 0)) > 0 and len(ckpt_paths) != int(cfg.n_splits):
        print(
            f"[WARN] Number of checkpoints ({len(ckpt_paths)}) != n_splits ({cfg.n_splits})."
        )

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
                path,
                cfg=cfg,
                input_length=input_length,
                feature_wavenumbers=feature_wavenumbers,
                weights_only=False,
            )
            model.eval()
            with torch.no_grad():
                pred = model(X_eval_t.to(model.device)).cpu().numpy()
            preds_list.append(pred)

    # Aggregate fold predictions per sample/target.
    # pred_stack shape: (n_folds, n_samples, n_targets)
    pred_stack = np.stack(preds_list, axis=0)
    pred_mean = np.mean(pred_stack, axis=0)
    pred_std = np.std(pred_stack, axis=0)
    pred_min = np.min(pred_stack, axis=0)
    pred_max = np.max(pred_stack, axis=0)

    # Read evaluation sample IDs (supports merged directory input)
    eval_ids = load_eval_sample_ids(cfg)
    if len(eval_ids) != pred_mean.shape[0]:
        raise RuntimeError(
            "Mismatch between number of evaluation IDs and predictions: "
            f"{len(eval_ids)} vs {pred_mean.shape[0]}"
        )

    # Build output DataFrame
    result = pd.DataFrame()
    result["SampleID"] = eval_ids
    for i, name in enumerate(cfg.target_cols):
        # Keep legacy mean column name for backward compatibility.
        result[name] = pred_mean[:, i]
        result[f"{name}_std"] = pred_std[:, i]
        result[f"{name}_min"] = pred_min[:, i]
        result[f"{name}_max"] = pred_max[:, i]

    predict_out_dir, predict_plots_dir, latest_predict_file = prepare_prediction_artifact_dirs(
        run_dir
    )
    out_path = os.path.join(predict_out_dir, cfg.predictions_file)
    legacy_out_path = os.path.join(run_dir, cfg.predictions_file)
    result.to_excel(out_path, index=False)
    # Backward-compatible location used by existing downstream tooling.
    result.to_excel(legacy_out_path, index=False)

    dist_plot_path = os.path.join(predict_plots_dir, "prediction_distributions.pdf")
    save_prediction_distribution_plot(result, cfg.target_cols, dist_plot_path)
    readme_path = write_prediction_readme(predict_out_dir, out_path, dist_plot_path)

    print(f"\nUsing run directory: {run_dir}")
    print(f"Prediction artifact directory: {predict_out_dir}")
    print(f"Predictions saved to {out_path}")
    print(f"Distribution plot saved to {dist_plot_path}")
    print(f"Artifact index saved to {readme_path}")
    print(f"Latest prediction pointer saved to {latest_predict_file}")
    print(f"Backward-compatible copy saved to {legacy_out_path}")
    if args.save_pred_to_xlsx:
        saved_xlsx_paths = save_predictions_to_source_xlsx(
            cfg,
            pred_mean=pred_mean,
            suffix=args.pred_xlsx_suffix,
        )
        print("Prediction-added xlsx files:")
        for path in saved_xlsx_paths:
            print(f"  - {path}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
