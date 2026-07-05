from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .config import MODELS_DIR, ORIGINAL_MODEL_PATH, REPORTS_DIR, ensure_project_dirs
from .llm_explainer import generate_explanation
from .project_io import extract_feature_names, extract_medians, extract_model, extract_threshold, load_artifact, write_json


CLINICAL_OVERRIDES = {
    "MAP": 58.0,
    "Lactate": 3.1,
    "Resp": 28.0,
    "Temp": 38.4,
    "WBC": 16.0,
    "HR": 112.0,
}


def load_preferred_artifact() -> Tuple[Any, Path, str]:
    optimized_path = MODELS_DIR / "optimized_model.pkl"
    if optimized_path.exists():
        return load_artifact(optimized_path), optimized_path, "optimized"
    return load_artifact(ORIGINAL_MODEL_PATH), ORIGINAL_MODEL_PATH, "baseline"


def build_example_patient(features: List[str], medians: Dict[str, float], overrides: Dict[str, Any]) -> Dict[str, float]:
    patient = {feature: float(medians.get(feature, 0.0)) for feature in features}
    for key, value in overrides.items():
        if key in patient:
            patient[key] = float(value)
    return patient


def select_clinical_variables(patient: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["MAP", "Lactate", "Resp", "Temp", "WBC", "HR"]
    return {key: patient[key] for key in keys if key in patient}


def select_influencing_factors(clinical_variables: Dict[str, Any]) -> List[str]:
    factors: List[str] = []
    map_value = clinical_variables.get("MAP")
    lactate = clinical_variables.get("Lactate")
    resp = clinical_variables.get("Resp")
    temp = clinical_variables.get("Temp")
    wbc = clinical_variables.get("WBC")
    hr = clinical_variables.get("HR")

    if map_value is not None and float(map_value) < 65:
        factors.append("pressao arterial media baixa")
    if lactate is not None and float(lactate) >= 2:
        factors.append("lactato elevado")
    if resp is not None and float(resp) >= 22:
        factors.append("frequencia respiratoria aumentada")
    if temp is not None and (float(temp) >= 38 or float(temp) < 36):
        factors.append("temperatura alterada")
    if wbc is not None and (float(wbc) >= 12 or float(wbc) < 4):
        factors.append("leucocitos alterados")
    if hr is not None and float(hr) >= 100:
        factors.append("frequencia cardiaca elevada")

    return factors or ["nenhum fator clinico simples destacado pelas regras locais"]


def predict_and_explain(overrides: Dict[str, Any] | None = None, use_llm: bool | None = None) -> Dict[str, Any]:
    ensure_project_dirs()
    artifact, artifact_path, artifact_source = load_preferred_artifact()
    model = extract_model(artifact)
    features = extract_feature_names(artifact, model)
    medians = extract_medians(artifact)
    threshold = extract_threshold(artifact)
    patient = build_example_patient(features, medians, overrides or CLINICAL_OVERRIDES)

    input_frame = pd.DataFrame([patient]).reindex(columns=features)
    probability = float(model.predict_proba(input_frame)[0, 1])
    predicted_class = int(probability >= threshold)
    predicted_label = "sepse" if predicted_class == 1 else "sem sepse"
    clinical_variables = select_clinical_variables(patient)
    influencing_factors = select_influencing_factors(clinical_variables)
    explanation = generate_explanation(
        probability=probability,
        predicted_class=predicted_class,
        clinical_variables=clinical_variables,
        influencing_factors=influencing_factors,
        use_llm=bool(os.getenv("OPENAI_API_KEY")) if use_llm is None else use_llm,
    )

    result = {
        "artifact_source": artifact_source,
        "artifact_path": str(artifact_path),
        "probability": probability,
        "threshold": threshold,
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "clinical_variables": clinical_variables,
        "influencing_factors": influencing_factors,
        "explanation_mode": explanation["mode"],
        "explanation": explanation["explanation"],
    }
    write_json(REPORTS_DIR / "predict_and_explain_example.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa uma predicao exemplo e gera explicacao.")
    for key in CLINICAL_OVERRIDES:
        parser.add_argument(f"--{key}", type=float, default=None)
    parser.add_argument("--mock", action="store_true", help="Forca explicacao local sem LLM.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        key: value
        for key in CLINICAL_OVERRIDES
        if (value := getattr(args, key)) is not None
    }
    predict_and_explain(overrides=overrides or None, use_llm=False if args.mock else None)


if __name__ == "__main__":
    main()
