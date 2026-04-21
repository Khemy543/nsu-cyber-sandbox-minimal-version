from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

import joblib
from sklearn.preprocessing import StandardScaler


@dataclass
class LoadedArtifacts:
    model: object
    scaler: StandardScaler
    feature_columns: list[str]
    manifest: dict


def save_artifacts(
    model: object,
    scaler: StandardScaler,
    feature_columns: list[str],
    class_names: list[str],
    model_path: Path,
    scaler_path: Path,
    features_path: Path,
    manifest_path: Path,
    metadata: dict,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    features_path.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "class_names": class_names,
        "feature_count": len(feature_columns),
        "paths": {
            "model": str(model_path),
            "scaler": str(scaler_path),
            "features": str(features_path),
        },
        "metadata": metadata,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_artifacts(
    model_path: Path,
    scaler_path: Path,
    features_path: Path,
    manifest_path: Path,
) -> LoadedArtifacts:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler artifact: {scaler_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing feature list artifact: {features_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = json.loads(features_path.read_text(encoding="utf-8"))
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return LoadedArtifacts(
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        manifest=manifest,
    )
