"""
Data loading, preprocessing, and PyTorch Lightning DataModule.
"""
import sys
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
import pytorch_lightning as pl

from config import Config


# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------

def load_raw_data(cfg: Config):
    """Load training and evaluation sheets from the Excel file.

    Returns
    -------
    X_train : np.ndarray, shape (N_train, n_features)
    y_train : np.ndarray, shape (N_train, n_targets)
    X_eval  : np.ndarray, shape (N_eval, n_features)
    wavenum : np.ndarray, wavenumber axis (for reference)
    """
    file_paths = resolve_excel_files(cfg.data_path)
    if not file_paths:
        raise FileNotFoundError(f"No xlsx files found: {cfg.data_path}")

    train_dfs = []
    eval_dfs = []
    common_cols_all = None
    for file_path in file_paths:
        try:
            train_df = pd.read_excel(file_path, sheet_name=cfg.train_sheet)
            eval_df = pd.read_excel(file_path, sheet_name=cfg.eval_sheet)
        except Exception as exc:
            print(
                f"[WARN] Skipped file '{file_path}': {exc}",
                file=sys.stderr,
            )
            continue

        # Identify spectral columns (numeric column names = wavenumbers)
        spec_cols = [c for c in train_df.columns if isinstance(c, (int, float))]
        # Use only the wavenumber columns present in BOTH sheets
        common_cols = [c for c in spec_cols if c in eval_df.columns]
        if not common_cols:
            print(
                f"[WARN] Skipped file '{file_path}': no common spectral columns "
                f"between '{cfg.train_sheet}' and '{cfg.eval_sheet}'",
                file=sys.stderr,
            )
            continue

        if common_cols_all is None:
            common_cols_all = set(common_cols)
        else:
            common_cols_all &= set(common_cols)

        train_dfs.append(train_df)
        eval_dfs.append(eval_df)

    if not train_dfs or not eval_dfs:
        raise RuntimeError(
            "No valid Excel files found. "
            f"Check sheet names: train='{cfg.train_sheet}', eval='{cfg.eval_sheet}'."
        )
    if not common_cols_all:
        raise RuntimeError("No common spectral columns across all valid files.")

    common_cols_sorted = sorted(common_cols_all, reverse=True)  # high wavenumber first
    wavenum = np.array(common_cols_sorted)

    try:
        X_train = np.concatenate(
            [df[common_cols_sorted].values.astype(np.float32) for df in train_dfs],
            axis=0,
        )
        y_train = np.concatenate(
            [df[cfg.target_cols].values.astype(np.float32) for df in train_dfs],
            axis=0,
        )
    except KeyError as exc:
        raise KeyError(
            f"Target column(s) missing in train sheet. target_cols={cfg.target_cols}"
        ) from exc

    X_eval = np.concatenate(
        [df[common_cols_sorted].values.astype(np.float32) for df in eval_dfs],
        axis=0,
    )

    return X_train, y_train, X_eval, wavenum


def resolve_excel_files(data_path: Union[str, Sequence[str]]):
    """Resolve one file or recursively list all xlsx files under a directory."""
    if isinstance(data_path, (list, tuple)):
        files = []
        for path in data_path:
            files.extend(resolve_excel_files(str(path)))
        unique = sorted({p.resolve() for p in files})
        return unique

    path = Path(data_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            p for p in path.rglob("*.xlsx")
            if p.is_file() and not p.name.startswith("~$")
        )
    raise FileNotFoundError(f"Path not found: {data_path}")


def load_eval_sample_ids(cfg: Config) -> np.ndarray:
    """Load evaluation sample IDs in the same file order as `load_raw_data`."""
    file_paths = resolve_excel_files(cfg.data_path)
    ids = []
    for file_path in file_paths:
        try:
            eval_df = pd.read_excel(file_path, sheet_name=cfg.eval_sheet)
        except Exception as exc:
            print(
                f"[WARN] Skipped file '{file_path}' while loading eval IDs: {exc}",
                file=sys.stderr,
            )
            continue
        id_col = eval_df.columns[0]  # first column is sample ID
        file_ids = eval_df[id_col].astype(str)
        if len(file_paths) == 1:
            ids.extend(file_ids.tolist())
        else:
            # keep source trace when multiple files are merged
            ids.extend([f"{file_path.stem}:{v}" for v in file_ids.tolist()])
    return np.array(ids)


def load_reconstruction_raw_data(cfg: Config):
    """Load spectral-only train/eval arrays for reconstruction tasks.

    Unlike `load_raw_data`, this function does not require target columns.
    """
    file_paths = resolve_excel_files(cfg.data_path)
    if not file_paths:
        raise FileNotFoundError(f"No xlsx files found: {cfg.data_path}")

    train_dfs = []
    eval_dfs = []
    eval_sheet = getattr(cfg, "eval_sheet", None)
    use_eval_sheet = bool(eval_sheet)
    common_cols_all = None
    for file_path in file_paths:
        try:
            train_df = pd.read_excel(file_path, sheet_name=cfg.train_sheet)
        except Exception as exc:
            print(
                f"[WARN] Skipped file '{file_path}': {exc}",
                file=sys.stderr,
            )
            continue
        eval_df = None
        if use_eval_sheet:
            try:
                eval_df = pd.read_excel(file_path, sheet_name=eval_sheet)
            except Exception as exc:
                print(
                    f"[WARN] Skipped file '{file_path}': {exc}",
                    file=sys.stderr,
                )
                continue

        spec_cols = [c for c in train_df.columns if isinstance(c, (int, float))]
        if use_eval_sheet and eval_df is not None:
            common_cols = [c for c in spec_cols if c in eval_df.columns]
        else:
            common_cols = spec_cols
        if not common_cols:
            if use_eval_sheet:
                print(
                    f"[WARN] Skipped file '{file_path}': no common spectral columns "
                    f"between '{cfg.train_sheet}' and '{eval_sheet}'",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[WARN] Skipped file '{file_path}': no spectral columns in "
                    f"'{cfg.train_sheet}'",
                    file=sys.stderr,
                )
            continue

        if common_cols_all is None:
            common_cols_all = set(common_cols)
        else:
            common_cols_all &= set(common_cols)

        train_dfs.append(train_df)
        if eval_df is not None:
            eval_dfs.append(eval_df)

    if not train_dfs:
        if use_eval_sheet:
            raise RuntimeError(
                "No valid Excel files found. "
                f"Check sheet names: train='{cfg.train_sheet}', eval='{eval_sheet}'."
            )
        raise RuntimeError(
            "No valid Excel files found. "
            f"Check train sheet name: train='{cfg.train_sheet}'."
        )
    if not common_cols_all:
        raise RuntimeError("No common spectral columns across all valid files.")

    common_cols_sorted = sorted(common_cols_all, reverse=True)
    wavenum = np.array(common_cols_sorted)

    X_train = np.concatenate(
        [df[common_cols_sorted].values.astype(np.float32) for df in train_dfs],
        axis=0,
    )
    if use_eval_sheet and eval_dfs:
        X_eval = np.concatenate(
            [df[common_cols_sorted].values.astype(np.float32) for df in eval_dfs],
            axis=0,
        )
    else:
        X_eval = np.empty((0, len(common_cols_sorted)), dtype=np.float32)
    return X_train, X_eval, wavenum


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(X_train, X_eval, cfg: Config):
    """Apply Savitzky-Golay smoothing + standardization.

    Returns scaled arrays and the fitted StandardScaler (for inverse transform).
    """
    # Savitzky-Golay smoothing along the wavelength axis
    if cfg.sg_window and cfg.sg_window > 1:
        X_train = savgol_filter(X_train, cfg.sg_window, cfg.sg_polyorder, axis=1)
        X_eval = savgol_filter(X_eval, cfg.sg_window, cfg.sg_polyorder, axis=1)

    # Standardize each wavelength channel
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train).astype(np.float32)
    X_eval = scaler_X.transform(X_eval).astype(np.float32)

    return X_train, X_eval, scaler_X


def apply_wavenumber_range(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    wavenum: np.ndarray,
    cfg: Config,
):
    """Filter spectral channels by configured wavenumber range.

    Returns
    -------
    X_train_sel : np.ndarray
    X_eval_sel  : np.ndarray
    wavenum_sel : np.ndarray
    selected_mask : np.ndarray[bool], shape (n_original_features,)
        Boolean mask on the original wavenumber axis.
    """
    wn_min = cfg.wavenumber_min
    wn_max = cfg.wavenumber_max

    if wn_min is None and wn_max is None:
        selected_mask = np.ones_like(wavenum, dtype=bool)
        return X_train, X_eval, wavenum, selected_mask

    if wn_min is None:
        lower, upper = float(np.min(wavenum)), float(wn_max)
    elif wn_max is None:
        lower, upper = float(wn_min), float(np.max(wavenum))
    else:
        lower, upper = sorted([float(wn_min), float(wn_max)])

    selected_mask = (wavenum >= lower) & (wavenum <= upper)
    if not np.any(selected_mask):
        raise ValueError(
            "No spectral channels were selected by wavenumber range: "
            f"[{lower}, {upper}]"
        )

    return (
        X_train[:, selected_mask],
        X_eval[:, selected_mask],
        wavenum[selected_mask],
        selected_mask,
    )


def append_derivative_features(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    cfg: Config,
):
    """Optionally append first/second-derivative features."""
    if (not cfg.use_derivative_features) and (not cfg.use_second_derivative_features):
        return X_train, X_eval

    feats_train = [X_train]
    feats_eval = [X_eval]

    if cfg.use_derivative_features or cfg.use_second_derivative_features:
        dX_train = np.gradient(X_train, axis=1).astype(np.float32)
        dX_eval = np.gradient(X_eval, axis=1).astype(np.float32)
        if cfg.use_derivative_features:
            feats_train.append(dX_train)
            feats_eval.append(dX_eval)

    if cfg.use_second_derivative_features:
        ddX_train = np.gradient(dX_train, axis=1).astype(np.float32)
        ddX_eval = np.gradient(dX_eval, axis=1).astype(np.float32)
        feats_train.append(ddX_train)
        feats_eval.append(ddX_eval)

    X_train_out = np.concatenate(feats_train, axis=1).astype(np.float32)
    X_eval_out = np.concatenate(feats_eval, axis=1).astype(np.float32)
    return X_train_out, X_eval_out


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SpectrumDataset(Dataset):
    """Simple dataset wrapping spectral features and (optional) targets."""

    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        # Add channel dimension: (N, 1, L) for Conv1d
        self.X = torch.from_numpy(X).unsqueeze(1)
        self.y = torch.from_numpy(y) if y is not None else None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ---------------------------------------------------------------------------
# Lightning DataModule
# ---------------------------------------------------------------------------

class SpectrumDataModule(pl.LightningDataModule):
    """DataModule that accepts pre-split arrays (for K-fold compatibility)."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32,
    ):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size

    def train_dataloader(self):
        ds = SpectrumDataset(self.X_train, self.y_train)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        ds = SpectrumDataset(self.X_val, self.y_val)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
