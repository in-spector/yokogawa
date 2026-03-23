"""
Utilities for checkpoint post-processing.
"""
import os
from typing import List

import torch


def average_lightning_checkpoints(checkpoint_paths: List[str], output_path: str) -> str:
    """Average state_dict weights from multiple Lightning checkpoints."""
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths must not be empty.")

    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in checkpoint_paths
    ]
    state_dicts = [ckpt["state_dict"] for ckpt in checkpoints]
    ref_keys = list(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        if list(sd.keys()) != ref_keys:
            raise ValueError("Checkpoint state_dict keys do not match across folds.")

    n_ckpts = float(len(state_dicts))
    averaged_state_dict = {}
    for key in ref_keys:
        value0 = state_dicts[0][key]
        if torch.is_tensor(value0) and torch.is_floating_point(value0):
            acc = value0.detach().clone().float()
            for sd in state_dicts[1:]:
                acc += sd[key].detach().float()
            averaged_state_dict[key] = (acc / n_ckpts).to(dtype=value0.dtype)
        elif torch.is_tensor(value0):
            averaged_state_dict[key] = value0.detach().clone()
        else:
            averaged_state_dict[key] = value0

    merged = checkpoints[0]
    merged["state_dict"] = averaged_state_dict
    merged["epoch"] = -1
    merged["global_step"] = 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(merged, output_path)
    return output_path
