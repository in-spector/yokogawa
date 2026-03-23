# Spectral Regression: CNN / MLP for Component Concentration Prediction

Predicts 3 component concentrations (**G1**, **G2**, **G3**) from infrared spectral data (~3713 wavenumber features) using a selectable backbone (**1D-CNN** or **MLP**) with K-fold cross-validation ensemble.

## Project Structure

```
spectral_regression/
├── run.sh              # One-command pipeline (install → train → predict)
├── requirements.txt    # Python dependencies
├── config.py           # All hyperparameters in one place
├── dataset.py          # Data loading, preprocessing, DataModule
├── model.py            # 1D-CNN architecture + Lightning module
├── train.py            # K-fold cross-validation training (regression)
├── train_reconstruction.py  # K-fold training for waveform reconstruction
├── predict.py          # Ensemble inference on evaluation set
├── data/
│   └── GX_2009.xlsx    # ← Place the data file here
└── outputs/            # (created at runtime)
    ├── latest_run.txt  # Path to the most recent run directory
    └── 20260319_103000/  # One directory per training run (timestamp)
        ├── artifacts/
        │   ├── model_structure.txt
        │   └── hparams.json
        ├── checkpoints/      # Best model per fold
        ├── tensorboard/      # TensorBoard logs
        ├── best_checkpoints.txt
        └── predictions.xlsx
```

## Quick Start

```bash
# 1. Place the data file
mkdir -p data
cp /path/to/GX_2009.xlsx data/

# 2. Run the full pipeline
bash run.sh
```

Or step by step:

```bash
pip install -r requirements.txt
python train.py       # Train with 5-fold CV
python predict.py     # Generate evaluation predictions
python train_reconstruction.py  # Train MLP autoencoder for waveform reconstruction
```

TensorBoard visualization:

```bash
tensorboard --logdir outputs/tensorboard
```

Open the displayed URL in your browser, then select each fold run
(`fold_1` ... `fold_5`) to compare metrics.

Each `python train.py` run creates a new timestamp-based directory under
`outputs/`, and all artifacts are saved there.
`python predict.py` automatically uses the run path written in
`outputs/latest_run.txt`.

For waveform reconstruction, `python train_reconstruction.py` writes to
`outputs/reconstruction_YYYYMMDD_HHMMSS/` and updates
`outputs/latest_run_reconstruction.txt`.

## Model Architecture

Two model options are available via `config.py`:

- `model_type="cnn"`: 1D-CNN for local spectral pattern extraction
- `model_type="mlp"`: simple MLP baseline on flattened spectra
- `model_type="pls"`: Partial Least Squares regression (`sklearn` implementation)

**CNN (`model_type="cnn"`)**:

```
Input spectrum (1 × 3713)
  → Conv1d(1→32, k=7) → BN → ReLU → MaxPool(4)
  → Conv1d(32→64, k=7) → BN → ReLU → MaxPool(4)
  → Conv1d(64→128, k=7) → BN → ReLU → MaxPool(4)
  → Flatten → Dropout(0.3) → FC(128) → ReLU → Dropout(0.3) → FC(3)
```

**Why this works well here:**
- Conv1d captures local correlations between neighbouring wavenumbers
- Aggressive pooling (4×) reduces 3713 → ~58 dimensions in 3 layers
- BatchNorm + Dropout prevent overfitting on 147 samples
- Savitzky-Golay pre-smoothing reduces spectral noise
- K-fold ensemble improves robustness

## Configuration

All hyperparameters are in `config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | cnn | Backbone selection (`cnn`, `mlp`, or `pls`) |
| `cnn_channels` | [32, 64, 128] | Filters per conv block |
| `mlp_hidden_dims` | [256, 128] | Hidden layer sizes for MLP |
| `pls_n_components` | 8 | Number of latent components for PLS |
| `pls_scale` | False | Internal scaling in `PLSRegression` |
| `pls_max_iter` | 500 | Max iterations in PLS power method |
| `pls_tol` | 1e-6 | Convergence tolerance in PLS |
| `wavenumber_min` | None | Lower bound of used wavenumber range |
| `wavenumber_max` | None | Upper bound of used wavenumber range |
| `range_plot_n_samples` | 5 | Number of samples in selection visualization |
| `kernel_size` | 7 | Conv1d kernel size |
| `dropout` | 0.3 | Dropout rate |
| `max_epochs` | 300 | Max training epochs |
| `n_splits` | 5 | K-fold splits |
| `early_stop_patience` | 50 | Early stopping patience |
| `tensorboard_dir` | tensorboard | TensorBoard log root under `output_dir` |
| `experiment_name` | spectral_regression | TensorBoard run group name |

## Reference Performance

The provided benchmark (Sheet2 in the Excel file):

| Target | R² | RMSEC (%) |
|--------|-----|-----------|
| G1 | 0.974 | 0.12 |
| G2 | 0.976 | 0.03 |
| G3 | 0.977 | 0.07 |
