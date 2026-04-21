from __future__ import annotations

from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _mean_abs_shap(shap_values: object) -> np.ndarray:
    if isinstance(shap_values, list):
        stacked = np.stack([np.abs(values) for values in shap_values], axis=0)
        return np.mean(stacked, axis=(0, 1))
    values = np.asarray(shap_values)
    if values.ndim == 3:
        return np.mean(np.abs(values), axis=(0, 2))
    if values.ndim == 2:
        return np.mean(np.abs(values), axis=0)
    raise ValueError(f"Unsupported SHAP array shape: {values.shape}")


def _select_beeswarm_values(
    shap_values: object,
    x_samples: np.ndarray,
) -> tuple[np.ndarray, int | None]:
    if isinstance(shap_values, list):
        if not shap_values:
            raise ValueError("Received empty SHAP value list.")
        class_scores = np.array([np.mean(np.abs(values)) for values in shap_values])
        class_index = int(np.argmax(class_scores))
        return np.asarray(shap_values[class_index]), class_index

    values = np.asarray(shap_values)
    if values.ndim == 2:
        return values, None

    if values.ndim != 3:
        raise ValueError(f"Unsupported SHAP shape for beeswarm plot: {values.shape}")

    n_samples, n_features = x_samples.shape

    if values.shape[0] == n_samples and values.shape[1] == n_features:
        class_scores = np.mean(np.abs(values), axis=(0, 1))
        class_index = int(np.argmax(class_scores))
        return values[:, :, class_index], class_index

    if values.shape[0] == n_samples and values.shape[2] == n_features:
        class_scores = np.mean(np.abs(values), axis=(0, 2))
        class_index = int(np.argmax(class_scores))
        return values[:, class_index, :], class_index

    if values.shape[1] == n_samples and values.shape[2] == n_features:
        class_scores = np.mean(np.abs(values), axis=(1, 2))
        class_index = int(np.argmax(class_scores))
        return values[class_index, :, :], class_index

    raise ValueError(
        "Unsupported SHAP shape for beeswarm plot "
        f"(expected to align with samples={n_samples}, features={n_features}): {values.shape}"
    )


def _run_shap_global(
    model: object,
    x_background: np.ndarray,
    x_samples: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
) -> dict:
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_samples)
    importance = _mean_abs_shap(shap_values)

    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df.to_csv(output_dir / "shap_global_importance.csv", index=False)

    top_df = df.head(20).iloc[::-1]
    plt.figure(figsize=(8, 7))
    plt.barh(top_df["feature"], top_df["mean_abs_shap"], color="#2E8B57")
    plt.title("Top SHAP Global Feature Importances")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_global_importance.png", dpi=220, bbox_inches="tight")
    plt.close()

    beeswarm_path = output_dir / "shap_beeswarm.png"
    beeswarm = {"status": "ok", "path": str(beeswarm_path)}
    try:
        beeswarm_values, class_index = _select_beeswarm_values(
            shap_values=shap_values,
            x_samples=x_samples,
        )
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            beeswarm_values,
            x_samples,
            feature_names=feature_names,
            max_display=min(20, len(feature_names)),
            plot_type="dot",
            show=False,
        )
        plt.title("SHAP Beeswarm Plot")
        plt.tight_layout()
        plt.savefig(beeswarm_path, dpi=220, bbox_inches="tight")
        plt.close()
        if class_index is not None:
            beeswarm["class_index"] = class_index
    except Exception as exc:  # noqa: BLE001
        plt.close("all")
        beeswarm = {"status": "error", "message": str(exc)}

    return {
        "status": "ok",
        "background_rows": int(len(x_background)),
        "explained_rows": int(len(x_samples)),
        "top_feature": str(df.iloc[0]["feature"]) if len(df) else "",
        "beeswarm": beeswarm,
    }


def _run_lime_local(
    model: object,
    x_train: np.ndarray,
    x_test: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    output_dir: Path,
    sample_indices: np.ndarray,
    random_seed: int,
) -> dict:
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=x_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=random_seed,
    )

    rows: list[dict] = []
    for order, row_idx in enumerate(sample_indices, start=1):
        exp = explainer.explain_instance(
            x_test[row_idx],
            model.predict_proba,
            num_features=min(15, len(feature_names)),
            top_labels=1,
        )
        output_html = output_dir / f"lime_explanation_{order}.html"
        exp.save_to_file(str(output_html))
        top_label = exp.top_labels[0] if exp.top_labels else None
        for feature, weight in exp.as_list(label=top_label):
            rows.append(
                {
                    "sample_order": order,
                    "row_index": int(row_idx),
                    "predicted_class": int(top_label) if top_label is not None else None,
                    "feature_rule": feature,
                    "weight": float(weight),
                }
            )

    pd.DataFrame(rows).to_csv(output_dir / "lime_local_explanations.csv", index=False)
    return {"status": "ok", "explained_rows": int(len(sample_indices))}


def run_explainability(
    model: object,
    x_train: np.ndarray,
    x_test: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    output_dir: Path,
    explain_samples: int,
    explain_background_size: int,
    random_seed: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_seed)

    bg_size = min(explain_background_size, len(x_train))
    sample_size = min(explain_samples, len(x_test))
    background_indices = rng.choice(len(x_train), size=bg_size, replace=False)
    sample_indices = rng.choice(len(x_test), size=sample_size, replace=False)
    x_background = x_train[background_indices]
    x_samples = x_test[sample_indices]

    report = {
        "shap": {"status": "skipped"},
        "lime": {"status": "skipped"},
        "sample_size": int(sample_size),
        "background_size": int(bg_size),
    }

    try:
        report["shap"] = _run_shap_global(
            model=model,
            x_background=x_background,
            x_samples=x_samples,
            feature_names=feature_names,
            output_dir=output_dir,
        )
    except Exception as exc:  # noqa: BLE001
        report["shap"] = {"status": "error", "message": str(exc)}

    try:
        report["lime"] = _run_lime_local(
            model=model,
            x_train=x_train,
            x_test=x_test,
            feature_names=feature_names,
            class_names=class_names,
            output_dir=output_dir,
            sample_indices=sample_indices,
            random_seed=random_seed,
        )
    except Exception as exc:  # noqa: BLE001
        report["lime"] = {"status": "error", "message": str(exc)}

    (output_dir / "explainability_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report
