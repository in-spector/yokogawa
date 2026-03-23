"""
Path helpers shared by configs and training scripts.
"""
from pathlib import Path
from typing import List, Sequence, Union


GenericPathInput = Union[str, Sequence[str]]


def infer_dataset_name(data_path: GenericPathInput) -> str:
    """Infer dataset name from a file/dir path or a list of paths."""
    if isinstance(data_path, (list, tuple)):
        paths = [str(p).strip() for p in data_path if str(p).strip()]
        if not paths:
            return "default_ds"
        if len(paths) == 1:
            return infer_dataset_name(paths[0])
        return "multi_input"

    path = Path(str(data_path)).expanduser()
    if path.suffix.lower() == ".xlsx":
        parent_name = path.parent.name
        if parent_name and parent_name.lower() not in {"data", "dataset", "datasets"}:
            return parent_name
        return path.stem
    return path.name or "default_ds"


def append_dataset_dir(base_dir: str, data_path: GenericPathInput) -> str:
    """Append dataset name to base_dir when the leaf name differs."""
    ds_name = infer_dataset_name(data_path)
    out_path = Path(base_dir)
    if out_path.name != ds_name:
        out_path = out_path / ds_name
    return str(out_path)


def normalize_data_paths(data_path: GenericPathInput) -> List[str]:
    """Normalize data path input to a non-empty list of strings."""
    if isinstance(data_path, (list, tuple)):
        paths = [str(p).strip() for p in data_path if str(p).strip()]
    elif isinstance(data_path, str):
        if "," in data_path:
            parts = [p.strip() for p in data_path.split(",") if p.strip()]
            paths = parts if len(parts) > 1 else [data_path.strip()]
        else:
            paths = [data_path.strip()]
    else:
        raise TypeError("data_path must be str or list/tuple of str.")

    if not paths:
        raise ValueError("data_path is empty.")
    return paths
