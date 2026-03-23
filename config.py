"""
Hyperparameters and paths for spectral regression.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union

from path_utils import append_dataset_dir


@dataclass
class Config:
    # ---- data ----
    data_path: Union[str, List[str]] = field(
        default_factory=lambda: [
            "/home/member/yokogawa_data/DS3"
        ]
    )
    train_sheet: str = "学習用"
    eval_sheet: str = "評価用"
    target_cols: List[str] = field(default_factory=lambda: ["H2O"])
    n_targets: int = 1

    # ---- preprocessing ----
    # Savitzky-Golay smoothing applied before training (reduces noise)
    sg_window: int = None
    sg_polyorder: int = 2
    # PCA dimensionality reduction (None = no PCA, keep raw spectrum)
    pca_components: int = None
    # Wavenumber range filter for regression task
    # If both are None, all wavenumbers are used.
    wavenumber_min: Optional[float] = None
    wavenumber_max: Optional[float] = None
    range_plot_n_samples: int = 5
    # Append first-derivative spectra to input features
    use_derivative_features: bool = True
    # Append second-derivative spectra to input features
    use_second_derivative_features: bool = False
    # Additive-noise data augmentation (regression training only)
    use_additive_noise_aug: bool = False
    additive_noise_std: float = 0.05
    additive_noise_copies: int = 1

    # ---- model (1D-CNN) ----
    model_type: str = "cnn"     # "cnn", "cnn_dual", "mlp", or "pls"

    # CNN parameters
    # Number of filters per conv block
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_size: int = 5
    dual_kernel_size_raw: int = 7
    dual_kernel_size_derivative: int = 5
    pool_size: int = 4
    fc_hidden: int = 256
    dropout: float = 0.5

    # MLP parameters
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    # Group Lasso coefficient for first MLP layer (0 disables sparsity regularization)
    mlp_group_lasso_lambda: float = 0
    # PLS parameters
    pls_n_components: int = 8
    # Since X is already standardized in preprocess(), keep internal PLS scaling off.
    pls_scale: bool = False
    pls_max_iter: int = 500
    pls_tol: float = 1e-6
    # Reconstruction MLP autoencoder parameters
    recon_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    recon_latent_dim: int = 64

    # ---- training ----
    init_checkpoint_path: Optional[str] = None  # Optional warm-start checkpoint
    max_epochs: int = 1000
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    cosine_min_lr: float = 1e-6
    early_stop_patience: int = 500
    n_splits: int = 5           # K-fold cross-validation

    # ---- output ----
    output_dir: str = "outputs"
    tensorboard_dir: str = "tensorboard"
    experiment_name: str = "spectral_regression"
    predictions_file: str = "predictions.xlsx"

    # ---- reproducibility ----
    seed: int = 42

    def __post_init__(self):
        self.output_dir = append_dataset_dir(self.output_dir, self.data_path)
