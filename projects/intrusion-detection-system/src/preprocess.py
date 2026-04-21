from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
CLASS_IDS = [0, 1, 2, 3, 4]
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

RAW_NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "attack",
    "level",
]

NSL_KDD_COLUMNS_43 = RAW_NSL_KDD_COLUMNS[:-2] + ["label", "difficulty"]
NSL_KDD_COLUMNS_42 = NSL_KDD_COLUMNS_43[:-1]

ATTACK_MAPPING = {
    "normal": 0,
    "neptune": 1,
    "back": 1,
    "land": 1,
    "pod": 1,
    "smurf": 1,
    "teardrop": 1,
    "mailbomb": 1,
    "apache2": 1,
    "processtable": 1,
    "udpstorm": 1,
    "worm": 1,
    "ipsweep": 2,
    "nmap": 2,
    "portsweep": 2,
    "satan": 2,
    "mscan": 2,
    "saint": 2,
    "ftp_write": 3,
    "guess_passwd": 3,
    "imap": 3,
    "multihop": 3,
    "phf": 3,
    "spy": 3,
    "warezclient": 3,
    "warezmaster": 3,
    "sendmail": 3,
    "named": 3,
    "snmpgetattack": 3,
    "snmpguess": 3,
    "xlock": 3,
    "xsnoop": 3,
    "httptunnel": 3,
    "buffer_overflow": 4,
    "loadmodule": 4,
    "perl": 4,
    "rootkit": 4,
    "ps": 4,
    "sqlattack": 4,
    "xterm": 4,
}


@dataclass
class PreparedDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_columns: list[str]
    scaler: StandardScaler
    train_rows: int
    test_rows: int


@dataclass
class ColabStylePreprocessResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    selected_features: list[str]
    selected_features_kbest: list[str]
    top_info_gain_features: list[str]
    scaler: StandardScaler
    summary: dict


def load_nsl_kdd_dataframe(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    lowered = {str(col).strip().lower(): col for col in df.columns}
    label_aliases = {"label", "binary_attack", "attack", "target", "class"}
    if any(alias in lowered for alias in label_aliases):
        return df

    df_headerless = pd.read_csv(file_path, header=None)
    if df_headerless.shape[1] == len(NSL_KDD_COLUMNS_43):
        df_headerless.columns = NSL_KDD_COLUMNS_43
        return df_headerless
    if df_headerless.shape[1] == len(NSL_KDD_COLUMNS_42):
        df_headerless.columns = NSL_KDD_COLUMNS_42
        return df_headerless

    raise ValueError(
        f"Unsupported NSL-KDD file format for {file_path}. "
        "Expected preprocessed CSV with target column or raw 42/43 NSL-KDD columns."
    )


def load_raw_nsl_kdd_dataframe(file_path: Path) -> pd.DataFrame:
    df_headerless = pd.read_csv(file_path, header=None)
    if df_headerless.shape[1] == len(RAW_NSL_KDD_COLUMNS):
        df_headerless.columns = RAW_NSL_KDD_COLUMNS
        return df_headerless

    df = pd.read_csv(file_path)
    lowered = {str(col).strip().lower(): col for col in df.columns}
    rename_map: dict[str, str] = {}
    if "label" in lowered:
        rename_map[lowered["label"]] = "attack"
    if "difficulty" in lowered:
        rename_map[lowered["difficulty"]] = "level"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "attack" not in df.columns:
        raise ValueError(
            f"Raw dataset file {file_path} must contain attack labels "
            "or be headerless NSL-KDD (43 columns)."
        )
    if "level" not in df.columns:
        df["level"] = 0
    return df


def _map_labels(raw_labels: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(raw_labels):
        values = raw_labels.astype(int).to_numpy()
        if set(np.unique(values)).issubset(set(CLASS_IDS)):
            return values

    normalized = raw_labels.astype(str).str.strip().str.lower()
    mapped = normalized.map(ATTACK_MAPPING)
    if mapped.isna().any():
        unknown = sorted(set(normalized[mapped.isna()].tolist()))
        preview = unknown[:10]
        raise ValueError(f"Unknown attack labels encountered: {preview}")
    return mapped.astype(int).to_numpy()


def _normalize_label_column(df: pd.DataFrame) -> pd.DataFrame:
    label_aliases = [
        "label",
        "binary_attack",
        "Label",
        "class",
        "Class",
        "target",
        "Target",
        "attack",
    ]
    for alias in label_aliases:
        if alias in df.columns and alias != "label":
            df = df.rename(columns={alias: "label"})
            break
    if "label" not in df.columns:
        raise ValueError("Missing label column after normalization.")
    return df


def _build_feature_tables(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = _normalize_label_column(train_df.copy())
    test_df = _normalize_label_column(test_df.copy())
    for frame in (train_df, test_df):
        for extra_col in ("difficulty", "level"):
            if extra_col in frame.columns:
                frame.drop(columns=[extra_col], inplace=True)

    x_train = train_df.drop(columns=["label"])
    x_test = test_df.drop(columns=["label"])

    categorical_cols = [
        col
        for col in x_train.columns
        if str(x_train[col].dtype) in {"object", "category"}
    ]

    combined = pd.concat([x_train, x_test], axis=0, keys=["train", "test"])
    if categorical_cols:
        combined = pd.get_dummies(combined, columns=categorical_cols, drop_first=False)

    x_train = combined.xs("train").copy()
    x_test = combined.xs("test").copy()
    x_train = x_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x_test = x_test.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if feature_columns is not None:
        x_train = x_train.reindex(columns=feature_columns, fill_value=0.0)
        x_test = x_test.reindex(columns=feature_columns, fill_value=0.0)

    return x_train, x_test


def prepare_nsl_kdd_dataset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    feature_columns: list[str] | None = None,
    fit_scaler: bool = True,
) -> PreparedDataset:
    train_df = _normalize_label_column(train_df.copy())
    test_df = _normalize_label_column(test_df.copy())
    y_train = _map_labels(train_df["label"])
    y_test = _map_labels(test_df["label"])

    x_train_df, x_test_df = _build_feature_tables(
        train_df=train_df,
        test_df=test_df,
        feature_columns=feature_columns,
    )

    feature_names = list(x_train_df.columns)
    if scaler is None:
        scaler = StandardScaler()
        fit_scaler = True

    if fit_scaler:
        x_train = scaler.fit_transform(x_train_df)
    else:
        x_train = scaler.transform(x_train_df)
    x_test = scaler.transform(x_test_df)

    return PreparedDataset(
        X_train=np.asarray(x_train, dtype=np.float32),
        y_train=np.asarray(y_train, dtype=np.int64),
        X_test=np.asarray(x_test, dtype=np.float32),
        y_test=np.asarray(y_test, dtype=np.int64),
        feature_columns=feature_names,
        scaler=scaler,
        train_rows=len(train_df),
        test_rows=len(test_df),
    )


def _make_attack_labels_numeric(df: pd.DataFrame) -> pd.DataFrame:
    if "attack" not in df.columns:
        raise ValueError("Raw dataframe must include `attack` column.")
    mapped = df["attack"].astype(str).str.strip().str.lower().map(ATTACK_MAPPING)
    if mapped.isna().any():
        unknown = sorted(set(df.loc[mapped.isna(), "attack"].astype(str).tolist()))
        raise ValueError(f"Unknown attack labels in raw dataset: {unknown[:10]}")
    output = df.copy()
    output["label"] = mapped.astype(int)
    output.drop(columns=["attack"], inplace=True)
    return output


def preprocess_raw_nsl_kdd(
    raw_train_df: pd.DataFrame,
    raw_test_df: pd.DataFrame,
    selection_method: str,
    top_k: int,
    target_column_name: str,
    random_seed: int,
) -> ColabStylePreprocessResult:
    train_df = raw_train_df.copy()
    test_df = raw_test_df.copy()

    train_rows_before = len(train_df)
    test_rows_before = len(test_df)
    train_df.drop_duplicates(inplace=True)
    test_df.drop_duplicates(inplace=True)
    train_rows_after = len(train_df)
    test_rows_after = len(test_df)

    train_missing_counts = train_df.isnull().sum().to_dict()
    test_missing_counts = test_df.isnull().sum().to_dict()

    train_df = _make_attack_labels_numeric(train_df)
    test_df = _make_attack_labels_numeric(test_df)

    for col in CATEGORICAL_COLUMNS:
        if col not in train_df.columns or col not in test_df.columns:
            raise ValueError(f"Missing categorical column `{col}` required by preprocessing.")

    # 1) LabelEncoder per categorical feature fitted on train+test combined.
    le_map = {
        col: LabelEncoder().fit(pd.concat([train_df[col], test_df[col]], axis=0).astype(str))
        for col in CATEGORICAL_COLUMNS
    }
    train_int = train_df[CATEGORICAL_COLUMNS].apply(
        lambda series: le_map[series.name].transform(series.astype(str))
    )
    test_int = test_df[CATEGORICAL_COLUMNS].apply(
        lambda series: le_map[series.name].transform(series.astype(str))
    )

    # 2) OneHotEncoder with handle_unknown=ignore.
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    train_ohe = pd.DataFrame(
        ohe.fit_transform(train_int),
        columns=ohe.get_feature_names_out(CATEGORICAL_COLUMNS),
        index=train_df.index,
    )
    test_ohe = pd.DataFrame(
        ohe.transform(test_int),
        columns=ohe.get_feature_names_out(CATEGORICAL_COLUMNS),
        index=test_df.index,
    )

    # 3) Drop original categoricals and concat one-hot columns.
    train_df = pd.concat([train_df.drop(columns=CATEGORICAL_COLUMNS), train_ohe], axis=1)
    test_df = pd.concat([test_df.drop(columns=CATEGORICAL_COLUMNS), test_ohe], axis=1)

    # 4) Align train/test columns, then reorder test to train.
    missing_in_test = set(train_df.columns) - set(test_df.columns)
    for col in missing_in_test:
        test_df[col] = 0
    missing_in_train = set(test_df.columns) - set(train_df.columns)
    for col in missing_in_train:
        train_df[col] = 0
    test_df = test_df[train_df.columns]

    scaler = StandardScaler()
    numeric_cols = train_df.select_dtypes(include="number").columns.difference(["label"])
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

    x_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    x_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    effective_k = max(1, min(top_k, x_train.shape[1]))
    kbest_selector = SelectKBest(score_func=mutual_info_classif, k=effective_k)
    kbest_selector.fit(x_train, y_train)
    selected_features_kbest = x_train.columns[kbest_selector.get_support()].tolist()

    mi_scores = mutual_info_classif(x_train, y_train, random_state=random_seed)
    mi_series = pd.Series(mi_scores, index=x_train.columns)
    top_info_gain_features = mi_series.sort_values(ascending=False).head(effective_k).index.tolist()

    if selection_method == "kbest":
        selected_features = selected_features_kbest
    elif selection_method == "info_gain":
        selected_features = top_info_gain_features
    else:
        selected_features = list(x_train.columns)

    x_train_selected = x_train[selected_features]
    x_test_selected = x_test[selected_features]

    train_out = x_train_selected.copy()
    train_out[target_column_name] = y_train.values
    test_out = x_test_selected.copy()
    test_out[target_column_name] = y_test.values

    summary = {
        "raw_rows": {
            "train_before_dedup": int(train_rows_before),
            "test_before_dedup": int(test_rows_before),
            "train_after_dedup": int(train_rows_after),
            "test_after_dedup": int(test_rows_after),
        },
        "missing_values": {
            "train_total_missing_cells": int(sum(train_missing_counts.values())),
            "test_total_missing_cells": int(sum(test_missing_counts.values())),
        },
        "selection_method": selection_method,
        "top_k": int(effective_k),
        "selected_feature_count": int(len(selected_features)),
        "selected_features": selected_features,
        "selected_features_kbest": selected_features_kbest,
        "top_info_gain_features": top_info_gain_features,
        "target_column": target_column_name,
        "class_names": CLASS_NAMES,
        "class_distribution_train": {
            str(k): int(v) for k, v in y_train.value_counts().sort_index().to_dict().items()
        },
        "class_distribution_test": {
            str(k): int(v) for k, v in y_test.value_counts().sort_index().to_dict().items()
        },
    }

    return ColabStylePreprocessResult(
        train_df=train_out,
        test_df=test_out,
        selected_features=selected_features,
        selected_features_kbest=selected_features_kbest,
        top_info_gain_features=top_info_gain_features,
        scaler=scaler,
        summary=summary,
    )


def save_preprocessed_outputs(
    result: ColabStylePreprocessResult,
    train_output_path: Path,
    test_output_path: Path,
    artifacts_dir: Path,
    results_dir: Path,
    save_scaler: bool,
    save_class_names: bool,
) -> dict:
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    test_output_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    result.train_df.to_csv(train_output_path, index=False)
    result.test_df.to_csv(test_output_path, index=False)

    if save_scaler:
        joblib.dump(result.scaler, artifacts_dir / "preprocess_scaler.joblib")
    if save_class_names:
        np.save(artifacts_dir / "class_names.npy", np.array(CLASS_NAMES))

    (artifacts_dir / "preprocess_selected_features.json").write_text(
        json.dumps(result.selected_features, indent=2),
        encoding="utf-8",
    )
    (artifacts_dir / "preprocess_selected_features_kbest.json").write_text(
        json.dumps(result.selected_features_kbest, indent=2),
        encoding="utf-8",
    )
    (artifacts_dir / "preprocess_top_info_gain_features.json").write_text(
        json.dumps(result.top_info_gain_features, indent=2),
        encoding="utf-8",
    )

    output_summary = {
        **result.summary,
        "output_paths": {
            "train_csv": str(train_output_path),
            "test_csv": str(test_output_path),
            "artifacts_dir": str(artifacts_dir),
            "results_dir": str(results_dir),
        },
    }
    (results_dir / "preprocess_summary.json").write_text(
        json.dumps(output_summary, indent=2),
        encoding="utf-8",
    )
    return output_summary
