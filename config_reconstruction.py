"""
Reconstruction-only configuration.
Only parameters used by `train_reconstruction.py` and reconstruction model/data flow.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from path_utils import append_dataset_dir


@dataclass
class ReconstructionConfig:
    # ---- data ----
    data_path: str = "/home/member/yokogawa_data/DS6/DS6_G4.xlsx"
    train_sheet: str = "学習用"
    eval_sheet: Optional[str] = None

    # ---- preprocessing ----
    sg_window: Optional[int] = None
    sg_polyorder: int = 2

    # ---- reconstruction model ----
    recon_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    recon_latent_dim: int = 64
    dropout: float = 0.1

    # ---- training ----
    max_epochs: int = 1000
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    cosine_min_lr: float = 1e-6
    early_stop_patience: int = 500
    n_splits: int = 5

    # ---- output ----
    output_dir: str = "outputs"
    tensorboard_dir: str = "tensorboard"
    experiment_name: str = "spectral_regression"

    # ---- reproducibility ----
    seed: int = 42

    def __post_init__(self):
        self.output_dir = append_dataset_dir(self.output_dir, self.data_path)
