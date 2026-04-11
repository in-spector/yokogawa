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

def _coerce_numeric_with_valid_mask(df: pd.DataFrame, cols):
    """Convert selected columns to numeric and return validity mask per row."""
    numeric_df = df.loc[:, cols].apply(pd.to_numeric, errors="coerce")
    valid_mask = ~numeric_df.isna().any(axis=1)
    return numeric_df, valid_mask


def _warn_dropped_rows(
    file_path: Path,
    sheet_name: str,
    dropped: int,
    total: int,
    context: str,
):
    """Emit a compact warning when rows are skipped due to parse failures."""
    if dropped <= 0:
        return
    print(
        f"[WARN] Dropped {dropped}/{total} row(s) in '{file_path}' "
        f"(sheet='{sheet_name}', {context}) due to non-numeric values.",
        file=sys.stderr,
    )


def load_raw_data(cfg: Config):
    """Load training and evaluation sheets from the Excel file.

    Returns
    -------
    X_train : np.ndarray, shape (N_train, n_features)
    y_train : np.ndarray, shape (N_train, n_targets)
    X_eval  : np.ndarray, shape (N_eval, n_features)
        If cfg.eval_sheet is None/empty, returns shape (0, n_features).
    wavenum : np.ndarray, wavenumber axis (for reference)
    """
    file_paths = resolve_excel_files(cfg.data_path)
    if not file_paths:
        raise FileNotFoundError(f"No xlsx files found: {cfg.data_path}")

    eval_sheet = getattr(cfg, "eval_sheet", None)
    use_eval_sheet = bool(eval_sheet)

    train_records = []
    eval_dfs = []
    for file_path in file_paths:
        train_df = None
        eval_df = None

        try:
            train_df = pd.read_excel(file_path, sheet_name=cfg.train_sheet)
        except Exception as exc:
            print(
                f"[WARN] Train sheet load failed for '{file_path}': {exc}",
                file=sys.stderr,
            )
        if use_eval_sheet:
            try:
                eval_df = pd.read_excel(file_path, sheet_name=eval_sheet)
            except Exception as exc:
                print(
                    f"[WARN] Eval sheet load failed for '{file_path}': {exc}",
                    file=sys.stderr,
                )
        if train_df is not None:
            spec_cols = [c for c in train_df.columns if isinstance(c, (int, float))]
            if not spec_cols:
                print(
                    f"[WARN] Skipped train data in '{file_path}': no spectral columns "
                    f"in '{cfg.train_sheet}'",
                    file=sys.stderr,
                )
            else:
                train_records.append((file_path, train_df, spec_cols))
        if eval_df is not None:
            eval_dfs.append((file_path, eval_df))

    if not train_records:
        raise RuntimeError(
            "No valid Excel files found. "
            + (
                f"Check sheet names: train='{cfg.train_sheet}', eval='{eval_sheet}'."
                if use_eval_sheet
                else f"Check train sheet name: train='{cfg.train_sheet}'."
            )
        )

    common_cols_all = set(train_records[0][2])
    for _, _, spec_cols in train_records[1:]:
        common_cols_all &= set(spec_cols)
    if not common_cols_all:
        raise RuntimeError("No common spectral columns across all train-sheet files.")

    common_cols_sorted = sorted(common_cols_all, reverse=True)  # high wavenumber first
    wavenum = np.array(common_cols_sorted)
    train_dfs = [(file_path, train_df) for file_path, train_df, _ in train_records]

    X_train_parts = []
    y_train_parts = []
    for file_path, df in train_dfs:
        try:
            spec_num, spec_valid = _coerce_numeric_with_valid_mask(df, common_cols_sorted)
            y_num, y_valid = _coerce_numeric_with_valid_mask(df, cfg.target_cols)
        except KeyError as exc:
            raise KeyError(
                f"Target column(s) missing in train sheet. target_cols={cfg.target_cols}"
            ) from exc

        valid = spec_valid & y_valid
        _warn_dropped_rows(
            file_path,
            cfg.train_sheet,
            dropped=int((~valid).sum()),
            total=int(df.shape[0]),
            context="train spectral/target parsing",
        )
        if bool(valid.any()):
            X_train_parts.append(spec_num.loc[valid].to_numpy(dtype=np.float32))
            y_train_parts.append(y_num.loc[valid].to_numpy(dtype=np.float32))

    if not X_train_parts:
        raise RuntimeError(
            "No valid numeric training rows remain after parsing spectral/target values."
        )

    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)

    if use_eval_sheet and eval_dfs:
        X_eval_parts = []
        for file_path, df in eval_dfs:
            missing_cols = [c for c in common_cols_sorted if c not in df.columns]
            if missing_cols:
                print(
                    f"[WARN] Skipped eval data in '{file_path}': missing required "
                    "spectral columns from train-sheet preprocessing.",
                    file=sys.stderr,
                )
                continue
            spec_num, valid = _coerce_numeric_with_valid_mask(df, common_cols_sorted)
            _warn_dropped_rows(
                file_path,
                eval_sheet,
                dropped=int((~valid).sum()),
                total=int(df.shape[0]),
                context="eval spectral parsing",
            )
            if bool(valid.any()):
                X_eval_parts.append(spec_num.loc[valid].to_numpy(dtype=np.float32))
        if X_eval_parts:
            X_eval = np.concatenate(X_eval_parts, axis=0)
        else:
            X_eval = np.empty((0, len(common_cols_sorted)), dtype=np.float32)
    else:
        X_eval = np.empty((0, len(common_cols_sorted)), dtype=np.float32)

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
    eval_sheet = getattr(cfg, "eval_sheet", None)
    if not eval_sheet:
        return np.array([], dtype=str)

    file_paths = resolve_excel_files(cfg.data_path)
    train_spec_cols_list = []
    eval_dfs = []
    for file_path in file_paths:
        train_df = None
        eval_df = None
        try:
            train_df = pd.read_excel(file_path, sheet_name=cfg.train_sheet)
        except Exception as exc:
            print(
                f"[WARN] Train sheet load failed for '{file_path}' while loading eval IDs: {exc}",
                file=sys.stderr,
            )
        try:
            eval_df = pd.read_excel(file_path, sheet_name=eval_sheet)
        except Exception as exc:
            print(
                f"[WARN] Eval sheet load failed for '{file_path}' while loading eval IDs: {exc}",
                file=sys.stderr,
            )

        if train_df is not None:
            spec_cols = [c for c in train_df.columns if isinstance(c, (int, float))]
            if spec_cols:
                train_spec_cols_list.append(spec_cols)
        if eval_df is not None:
            eval_dfs.append((file_path, eval_df))

    if not eval_dfs or not train_spec_cols_list:
        return np.array([], dtype=str)

    common_cols_all = set(train_spec_cols_list[0])
    for spec_cols in train_spec_cols_list[1:]:
        common_cols_all &= set(spec_cols)
    if not common_cols_all:
        return np.array([], dtype=str)

    common_cols_sorted = sorted(common_cols_all, reverse=True)
    ids = []
    for file_path, eval_df in eval_dfs:
        missing_cols = [c for c in common_cols_sorted if c not in eval_df.columns]
        if missing_cols:
            print(
                f"[WARN] Skipped eval IDs in '{file_path}': missing required spectral "
                "columns from train-sheet preprocessing.",
                file=sys.stderr,
            )
            continue
        _, valid = _coerce_numeric_with_valid_mask(eval_df, common_cols_sorted)
        _warn_dropped_rows(
            file_path,
            eval_sheet,
            dropped=int((~valid).sum()),
            total=int(eval_df.shape[0]),
            context="eval spectral parsing for IDs",
        )
        id_col = eval_df.columns[0]  # first column is sample ID
        file_ids = eval_df.loc[valid, id_col].astype(str)
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

        train_dfs.append((file_path, train_df))
        if eval_df is not None:
            eval_dfs.append((file_path, eval_df))

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

    X_train_parts = []
    for file_path, df in train_dfs:
        spec_num, valid = _coerce_numeric_with_valid_mask(df, common_cols_sorted)
        _warn_dropped_rows(
            file_path,
            cfg.train_sheet,
            dropped=int((~valid).sum()),
            total=int(df.shape[0]),
            context="reconstruction train spectral parsing",
        )
        if bool(valid.any()):
            X_train_parts.append(spec_num.loc[valid].to_numpy(dtype=np.float32))
    if not X_train_parts:
        raise RuntimeError("No valid numeric train rows remain for reconstruction.")
    X_train = np.concatenate(X_train_parts, axis=0)

    if use_eval_sheet and eval_dfs:
        X_eval_parts = []
        for file_path, df in eval_dfs:
            spec_num, valid = _coerce_numeric_with_valid_mask(df, common_cols_sorted)
            _warn_dropped_rows(
                file_path,
                eval_sheet,
                dropped=int((~valid).sum()),
                total=int(df.shape[0]),
                context="reconstruction eval spectral parsing",
            )
            if bool(valid.any()):
                X_eval_parts.append(spec_num.loc[valid].to_numpy(dtype=np.float32))
        if X_eval_parts:
            X_eval = np.concatenate(X_eval_parts, axis=0)
        else:
            X_eval = np.empty((0, len(common_cols_sorted)), dtype=np.float32)
    else:
        X_eval = np.empty((0, len(common_cols_sorted)), dtype=np.float32)
    return X_train, X_eval, wavenum


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def apply_savgol_to_base_spectra(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    cfg: Config,
):
    """Apply optional Savitzky-Golay transform on base spectra only."""
    sg_window = getattr(cfg, "sg_window", None)
    if sg_window is None:
        return X_train, X_eval

    sg_window = int(sg_window)
    if sg_window <= 1:
        return X_train, X_eval

    sg_polyorder = int(getattr(cfg, "sg_polyorder", 2))
    sg_deriv = int(getattr(cfg, "sg_deriv", 0))

    if sg_window % 2 == 0:
        raise ValueError("sg_window must be an odd integer greater than 1.")
    if sg_polyorder < 0:
        raise ValueError("sg_polyorder must be >= 0.")
    if sg_polyorder >= sg_window:
        raise ValueError("sg_polyorder must be smaller than sg_window.")
    if sg_deriv not in {0, 1, 2}:
        raise ValueError("sg_deriv must be one of {0, 1, 2}.")
    if sg_deriv > sg_polyorder:
        raise ValueError("sg_deriv must be <= sg_polyorder.")
    # mode='interp' requires window_length <= size of x along the target axis.
    if sg_window > X_train.shape[1]:
        raise ValueError(
            "sg_window must be <= number of spectral channels. "
            f"got sg_window={sg_window}, channels={X_train.shape[1]}"
        )

    # SG is applied along wavelength axis per spectrum row.
    X_train_out = savgol_filter(
        X_train,
        window_length=sg_window,
        polyorder=sg_polyorder,
        deriv=sg_deriv,
        axis=1,
        mode="interp",
    ).astype(np.float32)

    # SciPy's internal least-squares path can fail on empty sample batches.
    # Keep empty eval arrays untouched and consistent in shape/dtype.
    if X_eval.shape[0] == 0:
        X_eval_out = np.empty((0, X_train.shape[1]), dtype=np.float32)
        return X_train_out, X_eval_out

    if sg_window > X_eval.shape[1]:
        raise ValueError(
            "sg_window must be <= number of eval spectral channels. "
            f"got sg_window={sg_window}, channels={X_eval.shape[1]}"
        )

    X_eval_out = savgol_filter(
        X_eval,
        window_length=sg_window,
        polyorder=sg_polyorder,
        deriv=sg_deriv,
        axis=1,
        mode="interp",
    ).astype(np.float32)
    return X_train_out, X_eval_out


def preprocess(
    X_train,
    X_eval,
    cfg: Config,
    apply_savgol: bool = True,
):
    """Apply feature standardization with optional SG transform.

    Returns scaled arrays and the fitted StandardScaler (for inverse transform).
    """
    if apply_savgol:
        X_train, X_eval = apply_savgol_to_base_spectra(X_train, X_eval, cfg)

    # Standardize each feature channel
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train).astype(np.float32)
    if X_eval.shape[0] == 0:
        # Keep empty eval array when eval sheet is disabled (e.g., eval_sheet=None).
        X_eval = np.empty((0, X_train.shape[1]), dtype=np.float32)
    else:
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
        # Derivative features are built from the already-prepared base spectra.
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
