#!/usr/bin/env python3
"""
Plot mean and standard deviation spectra from multiple Excel files in a directory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "指定ディレクトリ内の複数xlsxを読み込み、各ファイルの"
            "スペクトル平均値と標準偏差をファイルごとに保存します。"
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="xlsxファイルを含む入力ディレクトリ",
    )
    parser.add_argument(
        "--sheet-name",
        type=str,
        default="学習用",
        help="読み込むシート名（既定: 学習用）",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.xlsx",
        help="対象ファイルパターン（再帰探索、既定: *.xlsx）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="出力先ディレクトリ（未指定時: <input-dir>）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="出力画像のDPI（既定: 180）",
    )
    return parser.parse_args()


def get_spectral_columns(df: pd.DataFrame) -> List[Tuple[str, float]]:
    spectral_cols: List[Tuple[str, float]] = []
    for col in df.columns:
        if isinstance(col, (int, float, np.integer, np.floating)):
            spectral_cols.append((col, float(col)))
            continue
        try:
            wn = float(col)
        except (TypeError, ValueError):
            continue
        spectral_cols.append((col, wn))
    spectral_cols.sort(key=lambda x: x[1], reverse=True)
    return spectral_cols


def compute_mean_std_for_file(
    xlsx_path: Path,
    sheet_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    spectral_cols = get_spectral_columns(df)
    if not spectral_cols:
        raise ValueError(
            f"No spectral columns found in '{xlsx_path.name}' (sheet='{sheet_name}')."
        )

    col_names = [name for name, _ in spectral_cols]
    wavenumbers = np.array([wn for _, wn in spectral_cols], dtype=np.float64)

    values = df[col_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if values.size == 0:
        raise ValueError(
            f"No spectral values found in '{xlsx_path.name}' (sheet='{sheet_name}')."
        )

    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0, ddof=0)

    return wavenumbers, mean, std


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(
        p for p in input_dir.rglob(args.pattern)
        if p.is_file() and not p.name.startswith("~$")
    )
    if not files:
        raise FileNotFoundError(
            f"No files matched pattern '{args.pattern}' in: {input_dir}"
        )

    output_dir = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    skipped_messages: List[str] = []
    for path in files:
        try:
            wavenumbers, mean, std = compute_mean_std_for_file(path, args.sheet_name)
        except Exception as exc:
            skipped_messages.append(f"{path}: {exc}")
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(wavenumbers, mean, linewidth=1.5, label=path.name)
        ax.fill_between(
            wavenumbers,
            mean - std,
            mean + std,
            alpha=0.18,
        )
        ax.set_title(f"Mean ± Std Spectrum: {path.stem}")
        ax.set_xlabel("Wavenumber")
        ax.set_ylabel("Intensity")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=9)
        ax.invert_xaxis()
        fig.tight_layout()

        rel = path.relative_to(input_dir).with_suffix("")
        out_path = output_dir / rel.parent / f"{rel.name}_mean_std.pdf"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        saved_paths.append(out_path)

    for msg in skipped_messages:
        print(f"[WARN] Skipped file: {msg}", file=sys.stderr)

    if not saved_paths:
        raise RuntimeError(
            "No valid files were processed. "
            "Check sheet name and spectral column format."
        )

    print(f"Found files: {len(files)}")
    print(f"Processed files: {len(saved_paths)}")
    print(f"Skipped files: {len(skipped_messages)}")
    for p in saved_paths:
        print(f"Saved figure: {p}")


if __name__ == "__main__":
    main()
