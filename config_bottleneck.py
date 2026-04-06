"""
Config for bottleneck-feature regression training.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from path_utils import append_dataset_dir


@dataclass
class BottleneckConfig:
    # ---- supervised data ----
    data_path: str = "/home/member/yokogawa_data/DS6/GX_2009.xlsx"
    train_sheet: str = "学習用"
    # If labels are in another sheet (e.g., "value"), set this.
    # If None, labels are read from train_sheet.
    target_sheet: Optional[str] = None
    target_cols: List[str] = field(default_factory=lambda: ["G1", "G2", "G3"])
    n_targets: int = 3

    # ---- encoder pretraining source ----
    # "msm" (masked spectra modeling) or "reconstruction"
    encoder_pretrain_task: str = "msm"
    # Optional explicit MSM checkpoint path. If None, resolve from latest MSM run.
    msm_ckpt_path: Optional[str] = "/home/member/cao/yokogawa/outputs/DS6/msm_20260322_235030/checkpoints/msm_fold_average.ckpt"
    # Used only when msm_ckpt_path is None.
    msm_output_dir: str = "outputs"
    # Optional explicit reconstruction checkpoint path.
    recon_ckpt_path: Optional[str] = None
    # Used only when recon_ckpt_path is None.
    reconstruction_output_dir: str = "outputs"
    # Must match encoder training setup used for the checkpoint.
    sg_window: Optional[int] = None
    sg_polyorder: int = 2
    recon_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    recon_latent_dim: int = 64
    dropout: float = 0.0
    encoder_batch_size: int = 256

    # ---- regression model ----
    model_type: str = "mlp"  # "pls", "cnn", or "mlp"
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_size: int = 5
    pool_size: int = 4
    fc_hidden: int = 256
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64, 32])

    # PLS
    pls_n_components: int = 8
    pls_scale: bool = False
    pls_max_iter: int = 500
    pls_tol: float = 1e-6

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
    experiment_name: str = "bottleneck_regression"

    # ---- reproducibility ----
    seed: int = 42

    def __post_init__(self):
        self.output_dir = append_dataset_dir(self.output_dir, self.data_path)
        self.msm_output_dir = append_dataset_dir(self.msm_output_dir, self.data_path)
        self.reconstruction_output_dir = append_dataset_dir(
            self.reconstruction_output_dir,
            self.data_path,
        )
