from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def calculate_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, Any]:
    y_true_array = np.asarray(list(y_true)).astype(int)
    y_pred_array = np.asarray(list(y_pred)).astype(int)
    matrix = confusion_matrix(y_true_array, y_pred_array, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "recall": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "precision": float(precision_score(y_true_array, y_pred_array, zero_division=0)),
        "f1_score": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def predictions_from_proba(proba: Iterable[float], threshold: float) -> np.ndarray:
    return (np.asarray(list(proba), dtype=float) >= float(threshold)).astype(int)


def fitness_score(metrics: Dict[str, Any]) -> float:
    positives = int(metrics.get("true_positives", 0)) + int(metrics.get("false_negatives", 0))
    fn_penalty = int(metrics.get("false_negatives", 0)) / max(1, positives)
    return float(metrics["recall"] * 0.55 + metrics["f1_score"] * 0.35 - fn_penalty * 0.10)


def evaluate_probabilities(y_true: Iterable[int], probabilities: Iterable[float], threshold: float) -> Dict[str, Any]:
    y_pred = predictions_from_proba(probabilities, threshold)
    metrics = calculate_metrics(y_true, y_pred)
    metrics["threshold"] = float(threshold)
    metrics["fitness"] = fitness_score(metrics)
    return metrics


def save_confusion_matrix_png(matrix: Iterable[Iterable[int]], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_array = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix_array, cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Sem sepse", "Sepse"],
        yticklabels=["Sem sepse", "Sepse"],
        ylabel="Classe real",
        xlabel="Classe predita",
        title=title,
    )
    for i in range(matrix_array.shape[0]):
        for j in range(matrix_array.shape[1]):
            ax.text(j, i, int(matrix_array[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
