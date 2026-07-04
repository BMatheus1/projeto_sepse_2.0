from __future__ import annotations

import argparse
import ast
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from .config import LOGS_DIR, MODELS_DIR, REPORTS_DIR, TEST_PATH, TRAIN_PATH, VALIDATION_PATH, ensure_project_dirs
from .metrics import evaluate_probabilities, save_confusion_matrix_png
from .project_io import (
    build_xgb_classifier,
    get_original_resources,
    load_split,
    positive_class_weight,
    prepare_features_target,
    setup_logging,
    write_json,
)


def load_best_hyperparameters(summary_path: Path = REPORTS_DIR / "ga_experiments_summary.csv") -> Dict[str, Any]:
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Resumo dos experimentos nao encontrado: {summary_path}. "
            "Execute primeiro: python -m src.tc_fase2.run_ga_experiments --quick"
        )
    df = pd.read_csv(summary_path)
    if df.empty:
        raise ValueError("Resumo dos experimentos esta vazio.")
    best_row = df.sort_values("best_fitness", ascending=False).iloc[0]
    params = best_row["best_hyperparameters"]
    if isinstance(params, str):
        return ast.literal_eval(params)
    return dict(params)


def train_optimized_model(quick: bool = False, train_sample: int | None = None, test_sample: int | None = None) -> dict:
    ensure_project_dirs()
    logger = setup_logging("optimized_training", LOGS_DIR / "training.log")
    logger.info("Inicio do treino otimizado | quick=%s", quick)
    start = time.perf_counter()

    _, _, features, medians, threshold = get_original_resources()
    best_params = load_best_hyperparameters()
    train_sample = train_sample if train_sample is not None else (5000 if quick else None)
    test_sample = test_sample if test_sample is not None else (3000 if quick else None)

    train_df = load_split(TRAIN_PATH, sample_size=train_sample)
    val_df = load_split(VALIDATION_PATH, sample_size=train_sample) if quick else load_split(VALIDATION_PATH)
    full_train = pd.concat([train_df, val_df], ignore_index=True)
    test_df = load_split(TEST_PATH, sample_size=test_sample)

    x_train, y_train = prepare_features_target(full_train, features, medians)
    x_test, y_test = prepare_features_target(test_df, features, medians)

    model = build_xgb_classifier(best_params, scale_pos_weight=positive_class_weight(y_train))
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = evaluate_probabilities(y_test, probabilities, threshold)
    metrics["execution_time_seconds"] = round(time.perf_counter() - start, 4)
    metrics["hyperparameters"] = best_params
    metrics["quick"] = quick

    artifact = {
        "modelo": model,
        "features": features,
        "medianas_treino": medians,
        "threshold_validacao_modelo": threshold,
        "best_hyperparameters": best_params,
        "origem": "Tech Challenge Fase 2 - Algoritmo Genetico",
    }
    model_path = MODELS_DIR / "optimized_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    write_json(REPORTS_DIR / "optimized_metrics.json", metrics)
    save_confusion_matrix_png(
        metrics["confusion_matrix"],
        REPORTS_DIR / "optimized_confusion_matrix.png",
        "Matriz de confusao - modelo otimizado",
    )
    logger.info("Fim do treino otimizado | recall=%.5f | f1=%.5f", metrics["recall"], metrics["f1_score"])
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o modelo final otimizado pelo GA.")
    parser.add_argument("--quick", action="store_true", help="Usa amostras menores para validacao rapida.")
    parser.add_argument("--train-sample", type=int, default=None)
    parser.add_argument("--test-sample", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_optimized_model(quick=args.quick, train_sample=args.train_sample, test_sample=args.test_sample)


if __name__ == "__main__":
    main()
