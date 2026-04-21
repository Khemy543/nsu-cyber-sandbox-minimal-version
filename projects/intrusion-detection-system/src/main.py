from __future__ import annotations

from pathlib import Path
import json

from artifacts import load_artifacts, save_artifacts
from config import parse_config
from data_loader import ensure_dataset_file
from evaluate import evaluate_ids_model, save_evaluation_outputs
from explain import run_explainability
from preprocess import (
    CLASS_NAMES,
    load_nsl_kdd_dataframe,
    load_raw_nsl_kdd_dataframe,
    prepare_nsl_kdd_dataset,
    preprocess_raw_nsl_kdd,
    save_preprocessed_outputs,
)
from train import train_xgboost_model


def _ensure_paths(config) -> None:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)


def _resolve_train_test_files(config) -> tuple[Path, Path]:
    train_path = ensure_dataset_file(
        file_path=config.train_path,
        download_from_minio=config.download_from_minio,
        minio_endpoint=config.minio_endpoint,
        minio_access_key=config.minio_access_key,
        minio_secret_key=config.minio_secret_key,
        minio_bucket=config.minio_bucket,
        minio_key=config.minio_train_key,
        fallback_keys=config.minio_train_key_fallbacks,
    )
    test_path = ensure_dataset_file(
        file_path=config.test_path,
        download_from_minio=config.download_from_minio,
        minio_endpoint=config.minio_endpoint,
        minio_access_key=config.minio_access_key,
        minio_secret_key=config.minio_secret_key,
        minio_bucket=config.minio_bucket,
        minio_key=config.minio_test_key,
        fallback_keys=config.minio_test_key_fallbacks,
    )
    return train_path, test_path


def _resolve_raw_train_test_files(config) -> tuple[Path, Path]:
    raw_train_path = ensure_dataset_file(
        file_path=config.raw_train_path,
        download_from_minio=config.download_from_minio,
        minio_endpoint=config.minio_endpoint,
        minio_access_key=config.minio_access_key,
        minio_secret_key=config.minio_secret_key,
        minio_bucket=config.minio_bucket,
        minio_key=config.minio_raw_train_key,
        fallback_keys=config.minio_raw_train_key_fallbacks,
    )
    raw_test_path = ensure_dataset_file(
        file_path=config.raw_test_path,
        download_from_minio=config.download_from_minio,
        minio_endpoint=config.minio_endpoint,
        minio_access_key=config.minio_access_key,
        minio_secret_key=config.minio_secret_key,
        minio_bucket=config.minio_bucket,
        minio_key=config.minio_raw_test_key,
        fallback_keys=config.minio_raw_test_key_fallbacks,
    )
    return raw_train_path, raw_test_path


def _load_frames(train_path: Path, test_path: Path):
    train_df = load_nsl_kdd_dataframe(train_path)
    test_df = load_nsl_kdd_dataframe(test_path)
    return train_df, test_df


def _run_preprocess_mode(config) -> dict:
    raw_train_path, raw_test_path = _resolve_raw_train_test_files(config)
    raw_train_df = load_raw_nsl_kdd_dataframe(raw_train_path)
    raw_test_df = load_raw_nsl_kdd_dataframe(raw_test_path)

    preprocess_result = preprocess_raw_nsl_kdd(
        raw_train_df=raw_train_df,
        raw_test_df=raw_test_df,
        selection_method=config.preprocess_selection_method,
        top_k=config.preprocess_top_k,
        target_column_name=config.preprocess_target_column,
        random_seed=config.random_seed,
    )
    preprocess_summary = save_preprocessed_outputs(
        result=preprocess_result,
        train_output_path=config.train_path,
        test_output_path=config.test_path,
        artifacts_dir=config.artifacts_dir,
        results_dir=config.results_dir,
        save_scaler=config.preprocess_save_scaler,
        save_class_names=config.preprocess_save_class_names,
    )
    return {
        "mode": config.mode,
        "dataset": config.dataset,
        "raw_train_path": str(raw_train_path),
        "raw_test_path": str(raw_test_path),
        "train_path": str(config.train_path),
        "test_path": str(config.test_path),
        "preprocess": preprocess_summary,
    }


def _train_and_maybe_explain(config) -> dict:
    train_path, test_path = _resolve_train_test_files(config)
    train_df, test_df = _load_frames(train_path, test_path)

    dataset = prepare_nsl_kdd_dataset(train_df, test_df, fit_scaler=True)
    model, training_metadata = train_xgboost_model(dataset, config)

    save_artifacts(
        model=model,
        scaler=dataset.scaler,
        feature_columns=dataset.feature_columns,
        class_names=CLASS_NAMES,
        model_path=config.model_path,
        scaler_path=config.scaler_path,
        features_path=config.features_path,
        manifest_path=config.manifest_path,
        metadata=training_metadata,
    )

    eval_result = evaluate_ids_model(
        model=model,
        x_test=dataset.X_test,
        y_test=dataset.y_test,
        class_names=CLASS_NAMES,
    )
    metrics_summary = save_evaluation_outputs(
        result=eval_result,
        y_test=dataset.y_test,
        class_names=CLASS_NAMES,
        output_dir=config.results_dir,
        save_predictions=config.save_predictions,
    )

    explain_report = {"status": "skipped"}
    if config.mode == "full" and not config.skip_explain:
        explain_report = run_explainability(
            model=model,
            x_train=dataset.X_train,
            x_test=dataset.X_test,
            feature_names=dataset.feature_columns,
            class_names=CLASS_NAMES,
            output_dir=config.results_dir / "explainability",
            explain_samples=config.explain_samples,
            explain_background_size=config.explain_background_size,
            random_seed=config.random_seed,
        )

    summary = {
        "mode": config.mode,
        "training": training_metadata,
        "evaluation": metrics_summary,
        "explainability": explain_report,
        "artifacts_dir": str(config.artifacts_dir),
        "results_dir": str(config.results_dir),
    }
    return summary


def _evaluate_or_explain_with_existing_artifacts(config) -> dict:
    train_path, test_path = _resolve_train_test_files(config)
    train_df, test_df = _load_frames(train_path, test_path)
    loaded = load_artifacts(
        model_path=config.model_path,
        scaler_path=config.scaler_path,
        features_path=config.features_path,
        manifest_path=config.manifest_path,
    )

    dataset = prepare_nsl_kdd_dataset(
        train_df=train_df,
        test_df=test_df,
        scaler=loaded.scaler,
        feature_columns=loaded.feature_columns,
        fit_scaler=False,
    )

    summary = {
        "mode": config.mode,
        "artifacts_dir": str(config.artifacts_dir),
        "results_dir": str(config.results_dir),
        "manifest": loaded.manifest,
    }

    if config.mode == "evaluate":
        eval_result = evaluate_ids_model(
            model=loaded.model,
            x_test=dataset.X_test,
            y_test=dataset.y_test,
            class_names=CLASS_NAMES,
        )
        metrics_summary = save_evaluation_outputs(
            result=eval_result,
            y_test=dataset.y_test,
            class_names=CLASS_NAMES,
            output_dir=config.results_dir,
            save_predictions=config.save_predictions,
        )
        summary["evaluation"] = metrics_summary
        return summary

    explain_report = run_explainability(
        model=loaded.model,
        x_train=dataset.X_train,
        x_test=dataset.X_test,
        feature_names=dataset.feature_columns,
        class_names=CLASS_NAMES,
        output_dir=config.results_dir / "explainability",
        explain_samples=config.explain_samples,
        explain_background_size=config.explain_background_size,
        random_seed=config.random_seed,
    )
    summary["explainability"] = explain_report
    return summary


def main() -> None:
    config = parse_config()
    _ensure_paths(config)

    print(f"Starting IDS pipeline in mode={config.mode}", flush=True)
    if config.mode == "preprocess":
        summary = _run_preprocess_mode(config)
    elif config.mode in {"full", "train"}:
        summary = _train_and_maybe_explain(config)
    else:
        summary = _evaluate_or_explain_with_existing_artifacts(config)

    summary_path = config.results_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Run summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
