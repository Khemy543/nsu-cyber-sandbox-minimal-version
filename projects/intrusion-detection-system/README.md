# Intrusion Detection System (X-AI-IDS Integration)

This sandbox service operationalizes the `X-AI-IDS-Architecture` workflow as a containerized batch pipeline.

## What This Service Does

- Runs an NSL-KDD 5-class IDS pipeline with class order:
  - `0: Normal`
  - `1: DoS`
  - `2: Probe`
  - `3: R2L`
  - `4: U2R`
- Supports:
  - preprocess (raw NSL-KDD -> train/test CSV)
  - train
  - evaluate
  - explain
  - full (train + evaluate + explain)
- Produces:
  - model artifacts (`joblib`, scaler, feature list, manifest)
  - evaluation reports/plots (confusion matrices, per-class metrics, ROC)
  - explainability outputs (SHAP global importance + beeswarm plot, LIME local explanations)

## Runtime Paths in Container

- Input data mount: `/app/data`
- Raw NSL-KDD defaults:
  - `/app/data/intrusion-detection-system/raw/KDDTrain+.txt`
  - `/app/data/intrusion-detection-system/raw/KDDTest+.txt`
- Expected default split files:
  - `/app/data/intrusion-detection-system/nsl-kdd/train.csv`
  - `/app/data/intrusion-detection-system/nsl-kdd/test.csv`
- Results output:
  - `/app/results/intrusion-detection-system`

## MinIO Dataset Loading

If local files are missing, the service can pull from MinIO (`datasets` bucket by default).

Primary object keys:

- `intrusion-detection-system/nsl-kdd/train.csv`
- `intrusion-detection-system/nsl-kdd/test.csv`

Raw-object keys for preprocess mode:

- `intrusion-detection-system/nsl-kdd/raw/KDDTrain+.txt`
- `intrusion-detection-system/nsl-kdd/raw/KDDTest+.txt`

Fallback keys are configurable.

## Default Compose Behavior

`docker/docker-compose.yml` runs:

```bash
python /app/src/main.py
```

with defaults:

- `mode=full`
- `dataset=nsl-kdd`
- `download_from_minio=true`

## Common Commands

Inside the `intrusion-detection-system` container:

```bash
python /app/src/main.py --mode preprocess
python /app/src/main.py --mode full
python /app/src/main.py --mode train --use-smote true
python /app/src/main.py --mode evaluate --download-from-minio false
python /app/src/main.py --mode explain --download-from-minio false
```

From repo root, you can also use Make targets:

```bash
make preprocess
make train
make evaluate
make explain
make ids-pipeline
```

Example with overrides:

```bash
make preprocess PREPROCESS_SELECTION_METHOD=kbest PREPROCESS_TOP_K=20
make train USE_SMOTE=true
```

Recommended end-to-end sequence when starting from raw NSL-KDD files:

```bash
python /app/src/main.py --mode preprocess --download-from-minio true
python /app/src/main.py --mode train --download-from-minio false
python /app/src/main.py --mode evaluate --download-from-minio false
python /app/src/main.py --mode explain --download-from-minio false
```

## Key Configuration Flags

- `--mode`: `preprocess|full|train|evaluate|explain`
- `--raw-train-path`, `--raw-test-path`
- `--train-path`, `--test-path`
- `--download-from-minio true|false`
- `--minio-bucket`, `--minio-train-key`, `--minio-test-key`
- `--minio-raw-train-key`, `--minio-raw-test-key`
- `--preprocess-selection-method none|kbest|info_gain`
- `--preprocess-top-k <n>`
- `--preprocess-target-column label|binary_attack`
- `--use-class-weights true|false`
- `--use-smote true|false`
- `--explain-samples <n>`
- `--skip-explain true|false`
- `make` variable overrides:
  - `DOWNLOAD_FROM_MINIO`
  - `PREPROCESS_SELECTION_METHOD`
  - `PREPROCESS_TOP_K`
  - `PREPROCESS_TARGET_COLUMN`
  - `USE_SMOTE`

## Artifacts

Saved under `/app/results/intrusion-detection-system/artifacts`:

- `ids_xgboost.joblib`
- `ids_scaler.joblib`
- `feature_columns.json`
- `model_manifest.json`
- `preprocess_scaler.joblib`
- `preprocess_selected_features.json`
- `preprocess_selected_features_kbest.json`
- `preprocess_top_info_gain_features.json`
- `class_names.npy`

## Notes

- `preprocess` mode mirrors your Colab preprocessing flow:
  - duplicate removal
  - 5-class label mapping
  - LabelEncoder + OneHotEncoder on `protocol_type`, `service`, `flag`
  - train/test column alignment
  - StandardScaler on numeric features
  - optional feature selection (`kbest` or `info_gain`)
- Label mapping for string attack names follows the 5-class NSL-KDD grouping used in your original notebooks.
- `evaluate` and `explain` modes require existing artifacts.
