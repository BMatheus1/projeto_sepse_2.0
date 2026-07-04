from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .config import (
    DEFAULT_THRESHOLD,
    HYPERPARAMETER_SPACE,
    ID_COLUMNS,
    LOGS_DIR,
    ORIGINAL_MODEL_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
)


MODEL_KEYS = [
    "modelo",
    "model",
    "clf",
    "classifier",
    "pipeline",
    "best_model",
    "melhor_modelo",
    "modelo_final",
]


def setup_logging(name: str, log_path: Path) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def load_artifact(path: Path = ORIGINAL_MODEL_PATH) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Artefato nao encontrado: {path}")
    return joblib.load(path)


def extract_model(artifact: Any) -> Any:
    if hasattr(artifact, "predict_proba") or hasattr(artifact, "predict"):
        return artifact
    if isinstance(artifact, dict):
        for key in MODEL_KEYS:
            candidate = artifact.get(key)
            if hasattr(candidate, "predict_proba") or hasattr(candidate, "predict"):
                return candidate
    raise ValueError("Nao foi possivel extrair um modelo preditivo do artefato.")


def extract_threshold(artifact: Any) -> float:
    if isinstance(artifact, dict):
        for key in ["threshold_validacao_modelo", "threshold_base"]:
            value = artifact.get(key)
            if value is not None:
                return float(value)
    return DEFAULT_THRESHOLD


def extract_feature_names(artifact: Any, model: Any | None = None) -> List[str]:
    if isinstance(artifact, dict):
        features = artifact.get("features")
        if isinstance(features, list) and features:
            return [str(col) for col in features]

    if model is not None and hasattr(model, "get_booster"):
        booster = model.get_booster()
        if getattr(booster, "feature_names", None):
            return [str(col) for col in booster.feature_names]

    if model is not None and hasattr(model, "feature_names_in_"):
        return [str(col) for col in model.feature_names_in_]

    raise ValueError("Nao foi possivel identificar as features do modelo.")


def extract_medians(artifact: Any) -> Dict[str, float]:
    if isinstance(artifact, dict):
        medians = artifact.get("medianas_treino")
        if isinstance(medians, dict):
            return {str(k): float(v) for k, v in medians.items()}
    return {}


def load_split(path: Path, sample_size: Optional[int] = None, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de dados nao encontrado: {path}")
    df = pd.read_parquet(path)
    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)
    return df


def prepare_features_target(
    df: pd.DataFrame,
    features: Iterable[str],
    medians: Optional[Dict[str, float]] = None,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"Coluna alvo ausente: {target_column}")

    medians = medians or {}
    feature_list = [str(col) for col in features if str(col) not in ID_COLUMNS and str(col) != target_column]
    x = df.copy()
    for col in feature_list:
        if col not in x.columns:
            x[col] = medians.get(col, 0.0)

    x = x.reindex(columns=feature_list)
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
        if col in medians:
            x[col] = x[col].fillna(medians[col])

    y = pd.to_numeric(df[target_column], errors="coerce").fillna(0).astype(int)
    return x, y


def get_original_resources() -> Tuple[Any, Any, List[str], Dict[str, float], float]:
    artifact = load_artifact()
    model = extract_model(artifact)
    features = extract_feature_names(artifact, model)
    medians = extract_medians(artifact)
    threshold = extract_threshold(artifact)
    return artifact, model, features, medians, threshold


def positive_class_weight(y: pd.Series) -> float:
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0:
        return 1.0
    return max(1.0, negatives / positives)


def normalize_hyperparameters(params: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in params.items():
        if key not in HYPERPARAMETER_SPACE:
            normalized[key] = value
            continue
        low, high, kind = HYPERPARAMETER_SPACE[key]
        if kind == "int":
            normalized[key] = int(np.clip(round(float(value)), low, high))
        else:
            normalized[key] = float(np.clip(float(value), low, high))
    return normalized


def build_xgb_classifier(params: Dict[str, Any], scale_pos_weight: float = 1.0) -> XGBClassifier:
    safe_params = normalize_hyperparameters(params)
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        **safe_params,
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")
