from __future__ import annotations

import argparse
import time

from .config import LOGS_DIR, REPORTS_DIR, TEST_PATH, VALIDATION_PATH, ensure_project_dirs
from .metrics import evaluate_probabilities, save_confusion_matrix_png
from .project_io import get_original_resources, load_split, prepare_features_target, setup_logging, write_json


def run_baseline(split: str = "test", sample_size: int | None = None) -> dict:
    ensure_project_dirs()
    logger = setup_logging("baseline", LOGS_DIR / "training.log")
    logger.info("Inicio do baseline | split=%s | sample_size=%s", split, sample_size)
    start = time.perf_counter()

    artifact, model, features, medians, threshold = get_original_resources()
    data_path = VALIDATION_PATH if split == "val" else TEST_PATH
    df = load_split(data_path, sample_size=sample_size)
    x, y = prepare_features_target(df, features, medians)
    probabilities = model.predict_proba(x)[:, 1]
    metrics = evaluate_probabilities(y, probabilities, threshold)
    metrics["execution_time_seconds"] = round(time.perf_counter() - start, 4)
    metrics["split"] = split
    metrics["sample_size"] = sample_size
    metrics["hyperparameters"] = model.get_params() if hasattr(model, "get_params") else {}
    metrics["model_name"] = artifact.get("nome_modelo") if isinstance(artifact, dict) else type(model).__name__

    write_json(REPORTS_DIR / "baseline_metrics.json", metrics)
    save_confusion_matrix_png(
        metrics["confusion_matrix"],
        REPORTS_DIR / "baseline_confusion_matrix.png",
        "Matriz de confusao - baseline",
    )
    logger.info("Fim do baseline | recall=%.5f | f1=%.5f", metrics["recall"], metrics["f1_score"])
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia o modelo baseline original.")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--sample", type=int, default=None, help="Amostra opcional para execucao rapida.")
    parser.add_argument("--quick", action="store_true", help="Usa uma amostra pequena do split escolhido.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_size = 5000 if args.quick and args.sample is None else args.sample
    run_baseline(split=args.split, sample_size=sample_size)


if __name__ == "__main__":
    main()
