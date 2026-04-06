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
            "/home/member/yokogawa_data/DS6/GX_2009.xlsx"
        ]
    )
    train_sheet: str = "学習用"
    # Optional evaluation sheet. If None, training runs without evaluation-sheet loading.
    eval_sheet: Optional[str] = None
    target_cols: List[str] = field(default_factory=lambda: ['G1', 'G2', 'G3'])
    n_targets: int = 3

    # ---- preprocessing ----
    # Savitzky-Golay transform applied to base spectra before feature expansion.
    # Set sg_window to an odd integer (e.g., 23/69). None disables SG.
    sg_window: int | None = 67
    sg_polyorder: int = 2
    # Derivative order for SG: 0=smoothing, 1=first derivative, 2=second derivative
    sg_deriv: int = 1
    # PCA dimensionality reduction (None = no PCA, keep raw spectrum)
    pca_components: int = None
    # Wavenumber range filter for regression task
    # If both are None, all wavenumbers are used.
    wavenumber_min: Optional[float] = None
    wavenumber_max: Optional[float] = None
    # Optional feature selection by per-wavelength std.
    # Keep top r% wavelengths (after wavenumber range filtering). None disables.
    std_top_r_percent: Optional[float] = None
    range_plot_n_samples: int = 5
    # Append first-derivative spectra to input features
    use_derivative_features: bool = False
    # Append second-derivative spectra to input features
    use_second_derivative_features: bool = False
    # Additive-noise data augmentation (regression training only)
    use_additive_noise_aug: bool = False
    additive_noise_std: float = 0.005
    additive_noise_copies: int = 1
    # Optional target-distribution oversampling for regression.
    # Training targets are quantile-binned; sparse bins are bootstrapped.
    use_target_bin_oversampling: bool = False
    oversampling_target_index: int = 0
    oversampling_n_bins: int = 10
    oversampling_target_ratio: float = 0.8
    oversampling_max_multiplier: float = 5.0
    oversampling_feature_noise_std: float = 0.01

    # ---- model (1D-CNN) ----
    model_type: str = "cnn"     # "cnn", "cnn_fourier", "cnn_dual", "mlp", "transformer", or "pls"

    # CNN parameters
    # Number of filters per conv block
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_size: int = 5
    dual_kernel_size_raw: int = 7
    dual_kernel_size_derivative: int = 5
    pool_size: int = 4
    fc_hidden: int = 256
    dropout: float = 0.1

    # Fourier-CNN parameters (model_type="cnn_fourier")
    # If set, use fixed number of low-frequency modes in each block.
    fourier_n_modes: Optional[int] = None
    # Used when fourier_n_modes is None.
    fourier_modes_ratio: float = 0.25
    fourier_min_modes: int = 8
    # AFNO-style coefficient shrinkage in frequency domain (0 disables).
    fourier_softshrink_lambda: float = 0.0
    # Output length of AdaptiveAvgPool1d in cnn_fourier head.
    fourier_head_pool_out_len: int = 8

    # MLP parameters
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    # Group Lasso coefficient for first MLP layer (0 disables sparsity regularization)
    mlp_group_lasso_lambda: float = 0
    # If True, scale first-layer MLP weights by per-feature std from fold training data.
    mlp_init_first_layer_by_input_std: bool = False

    # Transformer parameters
    transformer_patch_size: int = 25
    transformer_d_model: int = 250
    transformer_nhead: int = 5
    transformer_num_layers: int = 3
    transformer_ffn_dim: int = 128
    transformer_dropout: float = 0.0

    transformer_head_hidden: int = 128
    transformer_head_hidden2: int = 32

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
    max_epochs: int = 500
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    cosine_min_lr: float = 1e-6
    early_stop_patience: int = 100
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
