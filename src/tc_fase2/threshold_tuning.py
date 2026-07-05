from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .config import REPORTS_DIR, ensure_project_dirs
from .metrics import calculate_metrics
from .project_io import write_json


DEFAULT_THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def threshold_fitness(metrics: Dict[str, Any]) -> float:
    positives = int(metrics.get("true_positives", 0)) + int(metrics.get("false_negatives", 0))
    fn_penalty = int(metrics.get("false_negatives", 0)) / max(1, positives)
    return float(
        metrics["recall"] * 0.50
        + metrics["f1_score"] * 0.35
        + metrics["precision"] * 0.10
        - fn_penalty * 0.05
    )


def evaluate_thresholds(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    y_values = list(y_true)
    prob_values = list(probabilities)
    rows = []

    for threshold in thresholds or DEFAULT_THRESHOLDS:
        predictions = [int(prob >= threshold) for prob in prob_values]
        metrics = calculate_metrics(y_values, predictions)
        metrics["threshold"] = float(threshold)
        metrics["fitness"] = threshold_fitness(metrics)
        rows.append(metrics)

    return pd.DataFrame(rows).sort_values(
        ["fitness", "recall", "f1_score", "precision"],
        ascending=[False, False, False, False],
    )


def choose_best_threshold(results: pd.DataFrame) -> Dict[str, Any]:
    if results.empty:
        raise ValueError("Nenhum threshold foi avaliado.")
    return results.iloc[0].to_dict()


def tune_threshold(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    thresholds: Optional[List[float]] = None,
    reports_dir: Path = REPORTS_DIR,
    save_outputs: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    results = evaluate_thresholds(y_true, probabilities, thresholds)
    best = choose_best_threshold(results)

    if save_outputs:
        ensure_project_dirs()
        reports_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(reports_dir / "threshold_tuning.csv", index=False)
        write_json(
            reports_dir / "best_threshold.json",
            {
                "best_threshold": float(best["threshold"]),
                "threshold_strategy": "validation_fitness",
                "threshold_source": "validation_set",
                "fitness_formula": "recall * 0.50 + f1_score * 0.35 + precision * 0.10 - fn_penalty * 0.05",
                "metrics": best,
            },
        )

    return results, best
