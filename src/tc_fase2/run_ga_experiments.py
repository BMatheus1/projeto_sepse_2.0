from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List

import pandas as pd

from .config import LOGS_DIR, REPORTS_DIR, TRAIN_PATH, VALIDATION_PATH, ensure_project_dirs
from .genetic_algorithm import GeneticHyperparameterOptimizer
from .project_io import (
    get_original_resources,
    load_split,
    prepare_features_target,
    setup_logging,
    write_json,
)


DEFAULT_EXPERIMENTS = [
    {"experiment": 1, "population_size": 10, "generations": 5, "mutation_rate": 0.10},
    {"experiment": 2, "population_size": 20, "generations": 8, "mutation_rate": 0.20},
    {"experiment": 3, "population_size": 30, "generations": 10, "mutation_rate": 0.30},
]


QUICK_EXPERIMENTS = [
    {"experiment": 1, "population_size": 4, "generations": 2, "mutation_rate": 0.10},
    {"experiment": 2, "population_size": 5, "generations": 2, "mutation_rate": 0.20},
    {"experiment": 3, "population_size": 6, "generations": 2, "mutation_rate": 0.30},
]


def run_experiments(quick: bool = False, train_sample: int | None = None, val_sample: int | None = None) -> List[Dict[str, Any]]:
    ensure_project_dirs()
    logger = setup_logging("ga_experiments", LOGS_DIR / "ga_experiments.log")
    experiments = QUICK_EXPERIMENTS if quick else DEFAULT_EXPERIMENTS
    train_sample = train_sample if train_sample is not None else (2500 if quick else None)
    val_sample = val_sample if val_sample is not None else (1200 if quick else None)

    logger.info("Inicio dos experimentos GA | quick=%s", quick)
    _, _, features, medians, threshold = get_original_resources()

    train_df = load_split(TRAIN_PATH, sample_size=train_sample)
    val_df = load_split(VALIDATION_PATH, sample_size=val_sample)
    x_train, y_train = prepare_features_target(train_df, features, medians)
    x_val, y_val = prepare_features_target(val_df, features, medians)

    summary_rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []

    for config in experiments:
        logger.info("Inicio do experimento %s | config=%s", config["experiment"], config)
        start = time.perf_counter()
        optimizer = GeneticHyperparameterOptimizer(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            threshold=threshold,
            population_size=config["population_size"],
            generations=config["generations"],
            mutation_rate=config["mutation_rate"],
            logger=logger,
        )
        result = optimizer.run()
        elapsed = round(time.perf_counter() - start, 4)
        metrics = result["best_metrics"]
        payload = {
            **config,
            "quick": quick,
            "execution_time_seconds": elapsed,
            "best_fitness": result["best_fitness"],
            "best_hyperparameters": result["best_hyperparameters"],
            "accuracy": metrics["accuracy"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "f1_score": metrics["f1_score"],
            "false_negatives": metrics["false_negatives"],
            "false_positives": metrics["false_positives"],
            "threshold": metrics["threshold"],
            "history": result["history"],
        }
        write_json(REPORTS_DIR / f"ga_experiment_{config['experiment']}.json", payload)
        summary_rows.append({k: v for k, v in payload.items() if k not in {"history"}})
        for row in result["history"]:
            history_rows.append({"experiment": config["experiment"], **row})
        logger.info("Fim do experimento %s | best_fitness=%.5f", config["experiment"], result["best_fitness"])

    pd.DataFrame(summary_rows).to_csv(REPORTS_DIR / "ga_experiments_summary.csv", index=False)
    pd.DataFrame(history_rows).to_csv(REPORTS_DIR / "ga_fitness_history.csv", index=False)
    logger.info("Fim dos experimentos GA")
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa 3 experimentos de algoritmo genetico.")
    parser.add_argument("--quick", action="store_true", help="Usa populacoes e amostras pequenas.")
    parser.add_argument("--train-sample", type=int, default=None)
    parser.add_argument("--val-sample", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiments(quick=args.quick, train_sample=args.train_sample, val_sample=args.val_sample)


if __name__ == "__main__":
    main()
