from __future__ import annotations

from dataclasses import dataclass
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
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


@dataclass
class EvaluationResult:
    accuracy: float
    roc_auc_macro: float
    roc_auc_micro: float
    y_pred: np.ndarray
    y_prob: np.ndarray
    confusion: np.ndarray
    confusion_normalized: np.ndarray
    classification_report: dict
    per_class_metrics: pd.DataFrame


def evaluate_ids_model(
    model: object,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> EvaluationResult:
    y_prob = model.predict_proba(x_test)
    y_pred = np.argmax(y_prob, axis=1)

    accuracy = float(accuracy_score(y_test, y_pred))
    confusion = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
    confusion_norm = confusion_matrix(
        y_test, y_pred, labels=range(len(class_names)), normalize="true"
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=range(len(class_names)),
        zero_division=0,
    )
    per_class = pd.DataFrame(
        {
            "class_id": list(range(len(class_names))),
            "class_name": class_names,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
        }
    )

    y_bin = label_binarize(y_test, classes=list(range(len(class_names))))
    try:
        roc_auc_macro = float(
            roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
        )
        roc_auc_micro = float(
            roc_auc_score(y_bin, y_prob, average="micro", multi_class="ovr")
        )
    except ValueError:
        roc_auc_macro = float("nan")
        roc_auc_micro = float("nan")

    return EvaluationResult(
        accuracy=accuracy,
        roc_auc_macro=roc_auc_macro,
        roc_auc_micro=roc_auc_micro,
        y_pred=y_pred,
        y_prob=y_prob,
        confusion=confusion,
        confusion_normalized=confusion_norm,
        classification_report=report,
        per_class_metrics=per_class,
    )


def _plot_confusion(cm: np.ndarray, class_names: list[str], output_file: Path, title: str) -> None:
    plt.figure(figsize=(8.5, 6.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if np.issubdtype(cm.dtype, np.floating) else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close()


def _plot_roc_curves(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    output_file: Path,
) -> None:
    y_bin = label_binarize(y_test, classes=list(range(len(class_names))))
    plt.figure(figsize=(9.5, 6.5))
    for class_index, class_name in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, class_index], y_prob[:, class_index])
            auc_value = roc_auc_score(y_bin[:, class_index], y_prob[:, class_index])
            plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC={auc_value:.3f})")
        except ValueError:
            continue

    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.title("One-vs-Rest ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close()


def save_evaluation_outputs(
    result: EvaluationResult,
    y_test: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    save_predictions: bool = True,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_summary = {
        "accuracy": result.accuracy,
        "roc_auc_macro_ovr": result.roc_auc_macro,
        "roc_auc_micro_ovr": result.roc_auc_micro,
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(metrics_summary, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(result.confusion, index=class_names, columns=class_names).to_csv(
        output_dir / "confusion_matrix.csv", index=True
    )
    pd.DataFrame(
        result.confusion_normalized, index=class_names, columns=class_names
    ).to_csv(output_dir / "confusion_matrix_normalized.csv", index=True)
    pd.DataFrame(result.classification_report).transpose().to_csv(
        output_dir / "classification_report.csv", index=True
    )
    result.per_class_metrics.to_csv(output_dir / "per_class_metrics.csv", index=False)

    if save_predictions:
        predictions = pd.DataFrame(
            {
                "y_true": y_test,
                "y_pred": result.y_pred,
            }
        )
        for idx, class_name in enumerate(class_names):
            predictions[f"prob_{class_name.lower()}"] = result.y_prob[:, idx]
        predictions.to_csv(output_dir / "predictions.csv", index=False)

    _plot_confusion(
        result.confusion,
        class_names,
        output_dir / "confusion_matrix.png",
        "Confusion Matrix (Raw Counts)",
    )
    _plot_confusion(
        result.confusion_normalized,
        class_names,
        output_dir / "confusion_matrix_normalized.png",
        "Confusion Matrix (Row-Normalized)",
    )
    _plot_roc_curves(y_test, result.y_prob, class_names, output_dir / "roc_curves_ovr.png")
    return metrics_summary
