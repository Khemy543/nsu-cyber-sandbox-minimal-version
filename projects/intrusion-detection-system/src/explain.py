from __future__ import annotations

from pathlib import Path
import json
import os
from itertools import permutations

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


def _infer_dataset_name(feature_names: list[str]) -> str:
    joined = " ".join(feature_names).lower()
    if "destination port" in joined or "flow" in joined or "bwd" in joined:
        return "CICIDS-2017"
    return "NSL-KDD"


def _to_shap_arr(
    shap_values: object,
    n_samples: int,
    n_features: int,
) -> np.ndarray:
    if isinstance(shap_values, list):
        if not shap_values:
            raise ValueError("Received empty SHAP value list.")
        arr = np.stack([np.asarray(values) for values in shap_values], axis=0)
        if arr.ndim != 3:
            raise ValueError(f"Unsupported SHAP list shape: {arr.shape}")
        return arr

    values = np.asarray(shap_values)
    if values.ndim == 2:
        if values.shape == (n_features, n_samples):
            values = values.T
        if values.shape != (n_samples, n_features):
            raise ValueError(
                "Unsupported SHAP 2D shape "
                f"(expected ({n_samples}, {n_features}), got {values.shape})"
            )
        return values[None, :, :]

    if values.ndim != 3:
        raise ValueError(f"Unsupported SHAP array shape for LLM explanations: {values.shape}")

    for class_axis, sample_axis, feature_axis in permutations(range(3), 3):
        if values.shape[sample_axis] == n_samples and values.shape[feature_axis] == n_features:
            return np.transpose(values, (class_axis, sample_axis, feature_axis))

    raise ValueError(
        "Could not map SHAP array axes to (classes, samples, features). "
        f"shape={values.shape}, samples={n_samples}, features={n_features}"
    )


def _extract_base_value(explainer: object, class_index: int) -> float:
    expected = getattr(explainer, "expected_value", 0.0)
    values = np.asarray(expected)
    if values.ndim == 0:
        return float(values)
    flat = values.reshape(-1)
    if len(flat) == 0:
        return 0.0
    if 0 <= class_index < len(flat):
        return float(flat[class_index])
    return float(flat[0])


def _build_llm_prompt(
    model_name: str,
    dataset_name: str,
    predicted_label: str,
    predicted_prob: float,
    base_value: float,
    top_rows: pd.DataFrame,
) -> str:
    top_feature_lines = "\n".join(
        f"- {row.feature}: value={row.value}, shap={row.shap:+.4f}"
        for _, row in top_rows.iterrows()
    )
    top_k = len(top_rows)
    return (
        "You are an explainable-AI cybersecurity analyst. "
        "Explain this single prediction from an IDS.\n\n"
        f"Model: {model_name}\n"
        f"Dataset: {dataset_name}\n"
        f"Predicted Class: {predicted_label} (p = {predicted_prob:.3f})\n"
        f"Base Value (f(x)_base): {base_value:.4f}\n\n"
        f"Top {top_k} SHAP contributors for this instance "
        "(positive = pushes toward the predicted class):\n"
        f"{top_feature_lines}\n\n"
        "Explain briefly and clearly:\n"
        f'1) Why the model predicted "{predicted_label}" for THIS flow.\n'
        "2) Which features were decisive and how their values influenced the outcome.\n"
        "3) What a human analyst should check next (1 to 3 concrete steps).\n\n"
        "Avoid math-heavy language; use practical cybersecurity terms."
    )


def _run_llm_explanations(
    model: object,
    x_samples: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    sample_indices: np.ndarray,
    output_dir: Path,
    enable_llm_explanations: bool,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_top_k_features: int,
) -> dict:
    if not enable_llm_explanations:
        return {"status": "skipped", "message": "LLM explanations are disabled."}

    if llm_max_tokens <= 0:
        return {"status": "error", "message": "llm_max_tokens must be > 0."}

    if llm_top_k_features <= 0:
        return {"status": "error", "message": "llm_top_k_features must be > 0."}

    try:
        import shap
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": f"Failed to import shap: {exc}"}

    try:
        from openai import OpenAI
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": f"Failed to import openai package: {exc}"}

    if "api.openai.com" in llm_base_url and not llm_api_key:
        return {
            "status": "error",
            "message": (
                "IDS_LLM_API_KEY is required when using OpenAI hosted endpoint. "
                "Set IDS_LLM_API_KEY or use a different IDS_LLM_BASE_URL."
            ),
        }

    client_kwargs: dict[str, object] = {}
    if llm_api_key:
        client_kwargs["api_key"] = llm_api_key
    if llm_base_url:
        client_kwargs["base_url"] = llm_base_url
    client = OpenAI(**client_kwargs)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_samples)
    shap_arr = _to_shap_arr(
        shap_values=shap_values,
        n_samples=len(x_samples),
        n_features=len(feature_names),
    )

    prob_matrix = np.asarray(model.predict_proba(x_samples))
    if prob_matrix.ndim == 1:
        prob_matrix = np.column_stack([1.0 - prob_matrix, prob_matrix])

    dataset_name = _infer_dataset_name(feature_names)
    rows: list[dict] = []
    failures: list[dict] = []

    for local_idx in range(len(x_samples)):
        row_index = int(sample_indices[local_idx])
        probs = np.asarray(prob_matrix[local_idx]).reshape(-1)
        predicted_class_idx = int(np.argmax(probs))
        predicted_prob = float(probs[predicted_class_idx])
        label = (
            class_names[predicted_class_idx]
            if 0 <= predicted_class_idx < len(class_names)
            else str(predicted_class_idx)
        )

        shap_class_idx = predicted_class_idx
        if shap_class_idx >= shap_arr.shape[0]:
            shap_class_idx = 0

        shap_vec = np.asarray(shap_arr[shap_class_idx, local_idx, :]).reshape(-1)
        if len(shap_vec) != len(feature_names):
            failures.append(
                {
                    "sample_order": int(local_idx + 1),
                    "row_index": row_index,
                    "message": (
                        f"SHAP vector length mismatch: {len(shap_vec)} vs "
                        f"{len(feature_names)} features."
                    ),
                }
            )
            continue

        df = pd.DataFrame(
            {
                "feature": feature_names,
                "value": x_samples[local_idx],
                "shap": shap_vec,
            }
        )
        df = df.sort_values("shap", key=lambda series: series.abs(), ascending=False)
        top_rows = df.head(llm_top_k_features).reset_index(drop=True)
        base_value = _extract_base_value(explainer=explainer, class_index=shap_class_idx)
        prompt = _build_llm_prompt(
            model_name=type(model).__name__,
            dataset_name=dataset_name,
            predicted_label=label,
            predicted_prob=predicted_prob,
            base_value=base_value,
            top_rows=top_rows,
        )

        top_features_str = " | ".join(
            f"{item.feature}={item.value} (shap={item.shap:+.4f})"
            for _, item in top_rows.iterrows()
        )

        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful AI assistant specialized in "
                            "explainable ML for cybersecurity. Be precise and concise."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=llm_max_tokens,
                temperature=llm_temperature,
            )
            explanation = (
                response.choices[0].message.content.strip()
                if response.choices and response.choices[0].message
                else ""
            )
            rows.append(
                {
                    "sample_order": int(local_idx + 1),
                    "row_index": row_index,
                    "predicted_class_index": int(predicted_class_idx),
                    "predicted_class_label": label,
                    "predicted_probability": predicted_prob,
                    "top_features": top_features_str,
                    "explanation": explanation,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "sample_order": int(local_idx + 1),
                    "row_index": row_index,
                    "message": str(exc),
                }
            )

    output_csv = output_dir / "llm_shap_explanations.csv"
    output_json = output_dir / "llm_shap_explanations.json"
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    output_json.write_text(
        json.dumps(
            {
                "model": llm_model,
                "endpoint": llm_base_url,
                "generated_count": len(rows),
                "failed_count": len(failures),
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    status = "ok"
    if failures and rows:
        status = "partial"
    if failures and not rows:
        status = "error"

    return {
        "status": status,
        "generated_count": int(len(rows)),
        "failed_count": int(len(failures)),
        "output_csv": str(output_csv),
        "output_json": str(output_json),
        "model": llm_model,
        "endpoint": llm_base_url,
    }


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
    enable_llm_explanations: bool,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_top_k_features: int,
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
        "llm": {"status": "skipped"},
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

    try:
        report["llm"] = _run_llm_explanations(
            model=model,
            x_samples=x_samples,
            feature_names=feature_names,
            class_names=class_names,
            sample_indices=sample_indices,
            output_dir=output_dir,
            enable_llm_explanations=enable_llm_explanations,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            llm_top_k_features=llm_top_k_features,
        )
    except Exception as exc:  # noqa: BLE001
        report["llm"] = {"status": "error", "message": str(exc)}

    (output_dir / "explainability_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report
