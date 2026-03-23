# Yokogawa Spectral Learning Toolkit

赤外スペクトルを対象に、以下の4系統の学習ワークフローを扱うコードベースです。

- 回帰学習 (`train.py`) : スペクトル -> 目的変数（例: `H2O`, `G1/G2/G3`）
- 再構成事前学習 (`train_reconstruction.py`) : スペクトル自己再構成
- マスク再構成事前学習 (`train_masked_spectra_modeling.py`) : Masked Spectra Modeling (MSM)
- ボトルネック回帰 (`train_bottleneck_regression.py`) : 事前学習エンコーダ特徴 -> 目的変数

モデルは `CNN / Dual-branch CNN / MLP / PLS` を用途に応じて選択できます。

## 1. 主なファイル

- `config.py` : 回帰学習と推論の設定
- `config_reconstruction.py` : 再構成学習の設定
- `config_masked_spectra_modeling.py` : MSM学習の設定
- `config_bottleneck.py` : ボトルネック回帰の設定
- `dataset.py` : Excel読み込み・前処理・DataModule
- `model.py` : 各モデルと LightningModule
- `train.py` : K-fold 回帰学習
- `predict.py` : 回帰モデルの推論
- `train_reconstruction.py` : K-fold 再構成学習
- `train_masked_spectra_modeling.py` : K-fold MSM学習
- `train_bottleneck_regression.py` : K-fold ボトルネック回帰
- `checkpoint_utils.py` : fold checkpoint平均化
- `path_utils.py` : data_path / output_dir 関連ユーティリティ
- `train_io.py` : 学習runディレクトリ作成ユーティリティ

## 2. セットアップ

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## 3. データ配置の考え方

各 `config_*.py` の `data_path` で、以下のどちらも指定可能です。

- 単一Excelファイル (`.../xxx.xlsx`)
- Excel群を含むディレクトリ (`.../DS3` など)

`output_dir` は内部で `outputs/<dataset_name>/...` 形式に補正されます。

## 4. 学習・推論フロー

### A. 回帰学習 + 推論（標準フロー）

1. `config.py` を調整
2. 学習

```bash
python3 train.py
```

3. 推論

```bash
python3 predict.py
```

回帰学習では `latest_run.txt` が更新され、`predict.py` はそのrunを優先利用します。

### B. 再構成事前学習

1. `config_reconstruction.py` を調整
2. 実行

```bash
python3 train_reconstruction.py
```

`latest_run_reconstruction.txt` が更新されます。

### C. MSM事前学習

1. `config_masked_spectra_modeling.py` を調整
2. 実行

```bash
python3 train_masked_spectra_modeling.py
```

`latest_run_msm.txt` が更新されます。

### D. ボトルネック回帰

1. 先に B または C を実行してエンコーダ checkpoint を用意
2. `config_bottleneck.py` を調整
   - `encoder_pretrain_task = "reconstruction"` または `"msm"`
   - 明示checkpointを使うなら `recon_ckpt_path` / `msm_ckpt_path` を指定
3. 実行

```bash
python3 train_bottleneck_regression.py
```

## 5. 出力物

### 回帰 (`train.py`)

- `best_checkpoints.txt`
- `checkpoints/folds_avg.ckpt`（NNモデル時）
- `artifacts/hparams.json`
- `artifacts/model_structure.txt`
- `artifacts/cv_summary.txt`
- `artifacts/plots/*.pdf`
- `predictions.xlsx`（`predict.py`実行時）

### 再構成 (`train_reconstruction.py`)

- `best_checkpoints_reconstruction.txt`
- `artifacts/hparams.json`
- `artifacts/model_structure.txt`
- `artifacts/cv_summary.txt`
- `artifacts/plots/fold_*_waveform_recon.pdf`

### MSM (`train_masked_spectra_modeling.py`)

- `best_checkpoints_msm.txt`
- `checkpoints/msm_fold_average.ckpt`
- `artifacts/hparams.json`
- `artifacts/model_structure.txt`
- `artifacts/cv_summary.txt`
- `artifacts/plots/fold_*_msm_waveform.pdf`

### ボトルネック回帰 (`train_bottleneck_regression.py`)

- `best_checkpoints_bottleneck.txt`
- `checkpoints/bottleneck_fold_average.ckpt`（NNモデル時）
- `artifacts/hparams.json`
- `artifacts/model_structure.txt`
- `artifacts/cv_summary.txt`
- `artifacts/bottleneck_train.npz`

## 6. 設定ファイルの使い分け

- `config.py`
  - `model_type`: `cnn`, `cnn_dual`, `mlp`, `pls`
  - 波数範囲選択・微分特徴・ノイズ拡張・checkpoint warm start など
- `config_reconstruction.py`
  - 再構成MLPの hidden/latent と学習条件
- `config_masked_spectra_modeling.py`
  - MSMの `mask_ratio`, `mask_value`, CNN構成
- `config_bottleneck.py`
  - 事前学習タスク選択、encoder checkpoint解決、回帰ヘッド設定

## 7. 補助スクリプト

- `plot_spectra_mean_std.py` : スペクトル平均・分散の可視化
- `filter_xlsx_by_threshold.py` : 閾値によるExcelフィルタ処理

## 8. 注意事項

- `run.sh` は現行リポジトリには存在しません。上記の個別コマンドで実行してください。
- `model_type="pls"` では TensorBoard/NN checkpoint 平均化は一部スキップされます（仕様）。
- ボトルネック回帰は事前学習チェックポイントの前提が一致している必要があります（前処理設定・入力長など）。
