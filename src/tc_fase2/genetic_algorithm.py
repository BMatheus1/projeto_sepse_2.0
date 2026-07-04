from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import HYPERPARAMETER_SPACE, RANDOM_STATE
from .metrics import evaluate_probabilities
from .project_io import build_xgb_classifier, normalize_hyperparameters, positive_class_weight


@dataclass
class Individual:
    genes: Dict[str, Any]
    fitness: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class GeneticHyperparameterOptimizer:
    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        threshold: float,
        population_size: int = 10,
        generations: int = 5,
        mutation_rate: float = 0.10,
        elitism: int = 1,
        random_state: int = RANDOM_STATE,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.threshold = threshold
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.mutation_rate = float(mutation_rate)
        self.elitism = max(1, int(elitism))
        self.random = random.Random(random_state)
        self.logger = logger or logging.getLogger(__name__)
        self.scale_pos_weight = positive_class_weight(y_train)

    def random_gene_value(self, name: str) -> Any:
        low, high, kind = HYPERPARAMETER_SPACE[name]
        if kind == "int":
            return self.random.randint(int(low), int(high))
        return round(self.random.uniform(float(low), float(high)), 5)

    def create_individual(self) -> Individual:
        return Individual({name: self.random_gene_value(name) for name in HYPERPARAMETER_SPACE})

    def generate_initial_population(self) -> List[Individual]:
        return [self.create_individual() for _ in range(self.population_size)]

    def is_valid_genes(self, genes: Dict[str, Any]) -> bool:
        try:
            normalized = normalize_hyperparameters(genes)
        except Exception:
            return False
        return set(HYPERPARAMETER_SPACE).issubset(normalized)

    def evaluate_individual(self, individual: Individual) -> Individual:
        start = time.perf_counter()
        model = build_xgb_classifier(individual.genes, scale_pos_weight=self.scale_pos_weight)
        model.fit(self.x_train, self.y_train)
        probabilities = model.predict_proba(self.x_val)[:, 1]
        metrics = evaluate_probabilities(self.y_val, probabilities, self.threshold)
        metrics["execution_time_seconds"] = round(time.perf_counter() - start, 4)
        individual.metrics = metrics
        individual.fitness = float(metrics["fitness"])
        return individual

    def tournament_selection(self, population: List[Individual], tournament_size: int = 3) -> Individual:
        sample_size = min(tournament_size, len(population))
        candidates = self.random.sample(population, sample_size)
        return max(candidates, key=lambda item: item.fitness if item.fitness is not None else float("-inf"))

    def crossover(self, parent_a: Individual | Dict[str, Any], parent_b: Individual | Dict[str, Any]) -> Individual:
        genes_a = parent_a.genes if isinstance(parent_a, Individual) else parent_a
        genes_b = parent_b.genes if isinstance(parent_b, Individual) else parent_b
        child_genes = {}
        for name in HYPERPARAMETER_SPACE:
            child_genes[name] = genes_a[name] if self.random.random() < 0.5 else genes_b[name]
        return Individual(normalize_hyperparameters(child_genes))

    def mutate(self, individual: Individual | Dict[str, Any]) -> Individual:
        genes = dict(individual.genes if isinstance(individual, Individual) else individual)
        for name in HYPERPARAMETER_SPACE:
            if self.random.random() < self.mutation_rate:
                genes[name] = self.random_gene_value(name)
        return Individual(normalize_hyperparameters(genes))

    def run(self) -> Dict[str, Any]:
        population = self.generate_initial_population()
        history: List[Dict[str, Any]] = []
        best: Optional[Individual] = None

        for generation in range(1, self.generations + 1):
            self.logger.info("Inicio da geracao %s", generation)
            evaluated = [self.evaluate_individual(individual) for individual in population]
            evaluated.sort(key=lambda item: item.fitness if item.fitness is not None else float("-inf"), reverse=True)
            generation_best = evaluated[0]
            if best is None or (generation_best.fitness or 0.0) > (best.fitness or float("-inf")):
                best = generation_best

            history.append(
                {
                    "generation": generation,
                    "best_fitness": generation_best.fitness,
                    "best_genes": generation_best.genes,
                    "accuracy": generation_best.metrics.get("accuracy"),
                    "recall": generation_best.metrics.get("recall"),
                    "precision": generation_best.metrics.get("precision"),
                    "f1_score": generation_best.metrics.get("f1_score"),
                    "false_negatives": generation_best.metrics.get("false_negatives"),
                    "false_positives": generation_best.metrics.get("false_positives"),
                }
            )
            self.logger.info(
                "Geracao %s | melhor fitness %.5f | recall %.5f | f1 %.5f | FN %s",
                generation,
                generation_best.fitness,
                generation_best.metrics.get("recall"),
                generation_best.metrics.get("f1_score"),
                generation_best.metrics.get("false_negatives"),
            )

            next_population = [Individual(dict(item.genes), item.fitness, dict(item.metrics)) for item in evaluated[: self.elitism]]
            while len(next_population) < self.population_size:
                parent_a = self.tournament_selection(evaluated)
                parent_b = self.tournament_selection(evaluated)
                child = self.crossover(parent_a, parent_b)
                child = self.mutate(child)
                next_population.append(child)
            population = next_population

        if best is None:
            raise RuntimeError("Algoritmo genetico finalizou sem avaliar individuos.")

        return {
            "best_fitness": best.fitness,
            "best_hyperparameters": best.genes,
            "best_metrics": best.metrics,
            "history": history,
        }
