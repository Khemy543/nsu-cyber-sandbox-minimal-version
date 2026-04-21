from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_RESULTS_DIR = Path("/app/results/intrusion-detection-system")
DEFAULT_ARTIFACTS_DIR = DEFAULT_RESULTS_DIR / "artifacts"
DEFAULT_DATA_DIR = Path("/app/data/intrusion-detection-system/nsl-kdd")
DEFAULT_RAW_DATA_DIR = Path("/app/data/intrusion-detection-system/raw")


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class PipelineConfig:
    mode: str
    dataset: str
    raw_train_path: Path
    raw_test_path: Path
    train_path: Path
    test_path: Path
    results_dir: Path
    artifacts_dir: Path
    model_path: Path
    scaler_path: Path
    features_path: Path
    manifest_path: Path
    download_from_minio: bool
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str
    minio_train_key: str
    minio_test_key: str
    minio_train_key_fallbacks: list[str]
    minio_test_key_fallbacks: list[str]
    minio_raw_train_key: str
    minio_raw_test_key: str
    minio_raw_train_key_fallbacks: list[str]
    minio_raw_test_key_fallbacks: list[str]
    random_seed: int
    preprocess_selection_method: str
    preprocess_top_k: int
    preprocess_target_column: str
    preprocess_save_class_names: bool
    preprocess_save_scaler: bool
    use_class_weights: bool
    use_smote: bool
    smote_k_neighbors: int
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    explain_samples: int
    explain_background_size: int
    enable_llm_explanations: bool
    llm_model: str
    llm_base_url: str
    llm_api_key: str
    llm_temperature: float
    llm_max_tokens: int
    llm_top_k_features: int
    skip_explain: bool
    save_predictions: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Intrusion Detection System pipeline for sandbox execution."
    )
    parser.add_argument(
        "--mode",
        choices=["preprocess", "full", "train", "evaluate", "explain"],
        default=os.getenv("IDS_MODE", "full"),
        help="Pipeline run mode.",
    )
    parser.add_argument(
        "--dataset",
        choices=["nsl-kdd"],
        default=os.getenv("IDS_DATASET", "nsl-kdd"),
        help="Dataset profile to run.",
    )
    parser.add_argument(
        "--raw-train-path",
        type=Path,
        default=Path(
            os.getenv("IDS_RAW_TRAIN_PATH", str(DEFAULT_RAW_DATA_DIR / "KDDTrain+.txt"))
        ),
        help="Raw NSL-KDD train file path (KDDTrain+.txt style).",
    )
    parser.add_argument(
        "--raw-test-path",
        type=Path,
        default=Path(
            os.getenv("IDS_RAW_TEST_PATH", str(DEFAULT_RAW_DATA_DIR / "KDDTest+.txt"))
        ),
        help="Raw NSL-KDD test file path (KDDTest+.txt style).",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path(os.getenv("IDS_TRAIN_PATH", str(DEFAULT_DATA_DIR / "train.csv"))),
        help="Preprocessed train dataset path.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path(os.getenv("IDS_TEST_PATH", str(DEFAULT_DATA_DIR / "test.csv"))),
        help="Preprocessed test dataset path.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(os.getenv("IDS_RESULTS_DIR", str(DEFAULT_RESULTS_DIR))),
        help="Directory for metrics, plots, and explainability outputs.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(os.getenv("IDS_ARTIFACTS_DIR", str(DEFAULT_ARTIFACTS_DIR))),
        help="Directory for saved model artifacts.",
    )
    parser.add_argument(
        "--download-from-minio",
        type=parse_bool,
        default=parse_bool(os.getenv("IDS_DOWNLOAD_FROM_MINIO", "true")),
        help="Download train/test data from MinIO if local files are missing.",
    )
    parser.add_argument(
        "--minio-endpoint",
        default=os.getenv("IDS_MINIO_ENDPOINT", "http://storage:9000"),
        help="MinIO endpoint URL.",
    )
    parser.add_argument(
        "--minio-access-key",
        default=os.getenv("IDS_MINIO_ACCESS_KEY", "admin"),
        help="MinIO access key.",
    )
    parser.add_argument(
        "--minio-secret-key",
        default=os.getenv("IDS_MINIO_SECRET_KEY", "password123"),
        help="MinIO secret key.",
    )
    parser.add_argument(
        "--minio-bucket",
        default=os.getenv("IDS_MINIO_BUCKET", "datasets"),
        help="MinIO bucket containing IDS datasets.",
    )
    parser.add_argument(
        "--minio-train-key",
        default=os.getenv("IDS_MINIO_TRAIN_KEY", "intrusion-detection-system/nsl-kdd/train.csv"),
        help="Primary MinIO object key for train split.",
    )
    parser.add_argument(
        "--minio-test-key",
        default=os.getenv("IDS_MINIO_TEST_KEY", "intrusion-detection-system/nsl-kdd/test.csv"),
        help="Primary MinIO object key for test split.",
    )
    parser.add_argument(
        "--minio-train-key-fallbacks",
        type=parse_csv_list,
        default=parse_csv_list(
            os.getenv(
                "IDS_MINIO_TRAIN_KEY_FALLBACKS",
                "nsl-kdd/train.csv,NSL-KDD/train.csv,train.csv",
            )
        ),
        help="Comma-separated fallback object keys for train data.",
    )
    parser.add_argument(
        "--minio-test-key-fallbacks",
        type=parse_csv_list,
        default=parse_csv_list(
            os.getenv(
                "IDS_MINIO_TEST_KEY_FALLBACKS",
                "nsl-kdd/test.csv,NSL-KDD/test.csv,test.csv",
            )
        ),
        help="Comma-separated fallback object keys for test data.",
    )
    parser.add_argument(
        "--minio-raw-train-key",
        default=os.getenv(
            "IDS_MINIO_RAW_TRAIN_KEY",
            "intrusion-detection-system/nsl-kdd/raw/KDDTrain+.txt",
        ),
        help="Primary MinIO object key for raw train data.",
    )
    parser.add_argument(
        "--minio-raw-test-key",
        default=os.getenv(
            "IDS_MINIO_RAW_TEST_KEY",
            "intrusion-detection-system/nsl-kdd/raw/KDDTest+.txt",
        ),
        help="Primary MinIO object key for raw test data.",
    )
    parser.add_argument(
        "--minio-raw-train-key-fallbacks",
        type=parse_csv_list,
        default=parse_csv_list(
            os.getenv(
                "IDS_MINIO_RAW_TRAIN_KEY_FALLBACKS",
                "nsl-kdd/raw/KDDTrain+.txt,NSL-KDD/KDDTrain+.txt,KDDTrain+.txt",
            )
        ),
        help="Comma-separated fallback object keys for raw train data.",
    )
    parser.add_argument(
        "--minio-raw-test-key-fallbacks",
        type=parse_csv_list,
        default=parse_csv_list(
            os.getenv(
                "IDS_MINIO_RAW_TEST_KEY_FALLBACKS",
                "nsl-kdd/raw/KDDTest+.txt,NSL-KDD/KDDTest+.txt,KDDTest+.txt",
            )
        ),
        help="Comma-separated fallback object keys for raw test data.",
    )
    parser.add_argument("--random-seed", type=int, default=int(os.getenv("IDS_RANDOM_SEED", "42")))
    parser.add_argument(
        "--preprocess-selection-method",
        choices=["none", "kbest", "info_gain"],
        default=os.getenv("IDS_PREPROCESS_SELECTION_METHOD", "info_gain"),
        help="Feature-selection method used by preprocess mode.",
    )
    parser.add_argument(
        "--preprocess-top-k",
        type=int,
        default=int(os.getenv("IDS_PREPROCESS_TOP_K", "15")),
        help="Top-K feature count for k-best/information-gain selection.",
    )
    parser.add_argument(
        "--preprocess-target-column",
        choices=["label", "binary_attack"],
        default=os.getenv("IDS_PREPROCESS_TARGET_COLUMN", "binary_attack"),
        help="Target column name to write into preprocessed CSV outputs.",
    )
    parser.add_argument(
        "--preprocess-save-class-names",
        type=parse_bool,
        default=parse_bool(os.getenv("IDS_PREPROCESS_SAVE_CLASS_NAMES", "true")),
        help="Save class_names.npy artifact during preprocess mode.",
    )
    parser.add_argument(
        "--preprocess-save-scaler",
        type=parse_bool,
        default=parse_bool(os.getenv("IDS_PREPROCESS_SAVE_SCALER", "true")),
        help="Persist fitted preprocessing scaler artifact during preprocess mode.",
    )
    parser.add_argument(
        "--use-class-weights",
        type=parse_bool,
        default=parse_bool(os.getenv("IDS_USE_CLASS_WEIGHTS", "true")),
        help="Use balanced class weights during training.",
    )
    parser.add_argument(
        "--use-smote",
        type=parse_bool,
        default=parse_bool(os.getenv("IDS_USE_SMOTE", "false")),
        help="Apply SMOTE to the training split only.",
    )
    parser.add_argument(
        "--smote-k-neighbors",
        type=int,
        default=int(os.getenv("IDS_SMOTE_K_NEIGHBORS", "3")),
        help="k_neighbors parameter for SMOTE.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=int(os.getenv("IDS_N_ESTIMATORS", "300")),
        help="XGBoost number of estimators.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=int(os.getenv("IDS_MAX_DEPTH", "8")),
        help="XGBoost max depth.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(os.getenv("IDS_LEARNING_RATE", "0.08")),
        help="XGBoost learning rate.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=float(os.getenv("IDS_SUBSAMPLE", "0.9")),
        help="XGBoost subsample ratio.",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=float(os.getenv("IDS_COLSAMPLE_BYTREE", "0.8")),
        help="XGBoost colsample_bytree ratio.",
    )
    parser.add_argument(
        "--explain-samples",
        type=int,
        default=int(os.getenv("IDS_EXPLAIN_SAMPLES", "5")),
        help="Number of test rows to explain with LIME.",
    )
    parser.add_argument(
        "--explain-background-size",
        type=int,
        default=int(os.getenv("IDS_EXPLAIN_BACKGROUND_SIZE", "500")),
        help="Background sample size for explainability computations.",
    )
    parser.add_argument(
        "--enable-llm-explanations",
        type=parse_bool,
        default=parse_bool(os.getenv("IDS_ENABLE_LLM_EXPLANATIONS", "false")),
        help="Enable LLM-generated natural language explanations from SHAP outputs.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("IDS_LLM_MODEL", "gpt-4o-mini"),
        help="Model name for OpenAI-compatible chat completions endpoint.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=os.getenv("IDS_LLM_BASE_URL", "https://api.openai.com/v1"),
        help="Base URL for OpenAI-compatible chat API.",
    )
    parser.add_argument(
        "--llm-api-key",
        default=os.getenv("IDS_LLM_API_KEY", ""),
        help="API key for the configured OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=float(os.getenv("IDS_LLM_TEMPERATURE", "0.4")),
        help="Sampling temperature for LLM explanations.",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=int(os.getenv("IDS_LLM_MAX_TOKENS", "400")),
        help="Max tokens for each LLM explanation response.",
    )
    parser.add_argument(
        "--llm-top-k-features",
        type=int,
        default=int(os.getenv("IDS_LLM_TOP_K_FEATURES", "5")),
        help="Top-K SHAP features to include in the LLM prompt per sample.",
    )
    parser.add_argument(
        "--skip-explain",
        type=parse_bool,
        default=parse_bool(os.getenv("IDS_SKIP_EXPLAIN", "false")),
        help="Skip explainability stage even in full mode.",
    )
    parser.add_argument(
        "--save-predictions",
        type=parse_bool,
        default=parse_bool(os.getenv("IDS_SAVE_PREDICTIONS", "true")),
        help="Save per-row predictions CSV.",
    )
    return parser


def parse_config() -> PipelineConfig:
    args = build_parser().parse_args()
    model_path = args.artifacts_dir / "ids_xgboost.joblib"
    scaler_path = args.artifacts_dir / "ids_scaler.joblib"
    features_path = args.artifacts_dir / "feature_columns.json"
    manifest_path = args.artifacts_dir / "model_manifest.json"
    return PipelineConfig(
        mode=args.mode,
        dataset=args.dataset,
        raw_train_path=args.raw_train_path,
        raw_test_path=args.raw_test_path,
        train_path=args.train_path,
        test_path=args.test_path,
        results_dir=args.results_dir,
        artifacts_dir=args.artifacts_dir,
        model_path=model_path,
        scaler_path=scaler_path,
        features_path=features_path,
        manifest_path=manifest_path,
        download_from_minio=args.download_from_minio,
        minio_endpoint=args.minio_endpoint,
        minio_access_key=args.minio_access_key,
        minio_secret_key=args.minio_secret_key,
        minio_bucket=args.minio_bucket,
        minio_train_key=args.minio_train_key,
        minio_test_key=args.minio_test_key,
        minio_train_key_fallbacks=args.minio_train_key_fallbacks,
        minio_test_key_fallbacks=args.minio_test_key_fallbacks,
        minio_raw_train_key=args.minio_raw_train_key,
        minio_raw_test_key=args.minio_raw_test_key,
        minio_raw_train_key_fallbacks=args.minio_raw_train_key_fallbacks,
        minio_raw_test_key_fallbacks=args.minio_raw_test_key_fallbacks,
        random_seed=args.random_seed,
        preprocess_selection_method=args.preprocess_selection_method,
        preprocess_top_k=args.preprocess_top_k,
        preprocess_target_column=args.preprocess_target_column,
        preprocess_save_class_names=args.preprocess_save_class_names,
        preprocess_save_scaler=args.preprocess_save_scaler,
        use_class_weights=args.use_class_weights,
        use_smote=args.use_smote,
        smote_k_neighbors=args.smote_k_neighbors,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        explain_samples=args.explain_samples,
        explain_background_size=args.explain_background_size,
        enable_llm_explanations=args.enable_llm_explanations,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        llm_top_k_features=args.llm_top_k_features,
        skip_explain=args.skip_explain,
        save_predictions=args.save_predictions,
    )
