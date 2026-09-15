from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd

from .config import PHASE2_OPTIMIZED_MODEL_PATH, ROOT


def _collect_clinical_values(patient: Dict[str, Any]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for section in [patient.get("sinais_vitais", {}), patient.get("exames", {})]:
        for key, value in section.items():
            try:
                values[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    return values


def _fallback_score(values: Dict[str, float]) -> Tuple[float, List[str]]:
    factors: List[str] = []
    if values.get("MAP", 999) < 65:
        factors.append("pressão arterial média baixa")
    if values.get("Lactate", 0) >= 2:
        factors.append("lactato elevado")
    if values.get("Resp", 0) >= 22:
        factors.append("frequência respiratória aumentada")
    temp = values.get("Temp")
    if temp is not None and (temp >= 38 or temp < 36):
        factors.append("temperatura alterada")
    if values.get("WBC", 0) >= 12000 or ("WBC" in values and values.get("WBC", 99999) < 4000):
        factors.append("leucócitos alterados")
    if values.get("HR", 0) >= 100:
        factors.append("frequência cardíaca elevada")
    return round(len(factors) / 6, 4), factors


def _risk_level(score: float) -> str:
    if score >= 0.60:
        return "alto"
    if score >= 0.33:
        return "moderado"
    return "baixo"


def _extract_artifact_parts(artifact: Any) -> Tuple[Any, List[str], Dict[str, float], float]:
    model = artifact
    features: List[str] = []
    medians: Dict[str, float] = {}
    threshold = 0.15
    if isinstance(artifact, dict):
        for key in ["modelo", "model", "clf", "classifier", "pipeline", "best_model", "melhor_modelo", "modelo_final"]:
            if key in artifact:
                model = artifact[key]
                break
        features = [str(item) for item in artifact.get("features", [])]
        medians = {str(k): float(v) for k, v in artifact.get("medianas_treino", {}).items()} if isinstance(artifact.get("medianas_treino"), dict) else {}
        for key in ["best_threshold", "threshold_validacao_modelo", "threshold_base"]:
            if artifact.get(key) is not None:
                threshold = float(artifact[key])
                break
    if not features and hasattr(model, "feature_names_in_"):
        features = [str(col) for col in model.feature_names_in_]
    if not features:
        features_path = ROOT / "data" / "processed" / "features_modelo_sem_tempo_admin.csv"
        if features_path.exists():
            df = pd.read_csv(features_path)
            features = df.iloc[:, 0].dropna().astype(str).tolist()
    return model, features, medians, threshold


def estimate_sepsis_risk(patient: Dict[str, Any], model_path: Path = PHASE2_OPTIMIZED_MODEL_PATH) -> Dict[str, Any]:
    values = _collect_clinical_values(patient)
    fallback_score, fallback_factors = _fallback_score(values)

    if model_path.exists():
        try:
            artifact = joblib.load(model_path)
            model, features, medians, threshold = _extract_artifact_parts(artifact)
            if hasattr(model, "predict_proba") and features:
                row = {feature: values.get(feature, medians.get(feature, 0.0)) for feature in features}
                df = pd.DataFrame([row]).reindex(columns=features).fillna(0)
                probability = float(model.predict_proba(df)[0, 1])
                return {
                    "risk_score": round(probability, 4),
                    "risk_level": _risk_level(max(probability, threshold if probability >= threshold else probability)),
                    "used_model": "fase2_optimized_model",
                    "factors": fallback_factors,
                }
        except Exception:
            pass

    return {
        "risk_score": fallback_score,
        "risk_level": _risk_level(fallback_score),
        "used_model": "clinical_rule_fallback",
        "factors": fallback_factors,
    }

