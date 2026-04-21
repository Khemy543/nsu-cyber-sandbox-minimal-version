from __future__ import annotations

from collections import Counter

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from config import PipelineConfig
from preprocess import PreparedDataset


def _compute_sample_weights(labels: np.ndarray) -> tuple[np.ndarray, dict[int, float]]:
    classes = np.unique(labels)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    weight_map = {int(class_id): float(weight) for class_id, weight in zip(classes, class_weights)}
    sample_weights = np.array([weight_map[int(label)] for label in labels], dtype=np.float32)
    return sample_weights, weight_map


def train_xgboost_model(
    dataset: PreparedDataset,
    config: PipelineConfig,
) -> tuple[XGBClassifier, dict]:
    x_train = dataset.X_train
    y_train = dataset.y_train
    original_distribution = {int(k): int(v) for k, v in Counter(y_train).items()}

    smote_applied = False
    if config.use_smote:
        class_counts = Counter(y_train)
        min_count = min(class_counts.values())
        if min_count >= 2:
            k_neighbors = min(config.smote_k_neighbors, min_count - 1)
            smote = SMOTE(random_state=config.random_seed, k_neighbors=k_neighbors)
            x_train, y_train = smote.fit_resample(x_train, y_train)
            smote_applied = True
        else:
            print(
                "SMOTE skipped because one class has fewer than 2 samples.",
                flush=True,
            )

    sample_weights = None
    class_weight_map = {}
    if config.use_class_weights:
        sample_weights, class_weight_map = _compute_sample_weights(y_train)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        eval_metric="mlogloss",
        random_state=config.random_seed,
        n_jobs=1,
        tree_method="hist",
    )

    fit_kwargs: dict = {}
    if sample_weights is not None:
        fit_kwargs["sample_weight"] = sample_weights
    model.fit(x_train, y_train, **fit_kwargs)

    metadata = {
        "training_rows": int(len(y_train)),
        "feature_count": int(x_train.shape[1]),
        "original_class_distribution": original_distribution,
        "resampled_class_distribution": {int(k): int(v) for k, v in Counter(y_train).items()},
        "use_smote": config.use_smote,
        "smote_applied": smote_applied,
        "use_class_weights": config.use_class_weights,
        "class_weights": class_weight_map,
        "xgboost_params": {
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "learning_rate": config.learning_rate,
            "subsample": config.subsample,
            "colsample_bytree": config.colsample_bytree,
            "random_seed": config.random_seed,
        },
    }
    return model, metadata
