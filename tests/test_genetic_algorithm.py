import pandas as pd

from src.tc_fase2.config import HYPERPARAMETER_SPACE
from src.tc_fase2.genetic_algorithm import GeneticHyperparameterOptimizer, Individual
from src.tc_fase2.metrics import fitness_score


def make_optimizer():
    x = pd.DataFrame({"a": [0.0, 1.0, 0.5, 1.5], "b": [1.0, 0.0, 1.5, 0.5]})
    y = pd.Series([0, 1, 0, 1])
    return GeneticHyperparameterOptimizer(
        x_train=x,
        y_train=y,
        x_val=x,
        y_val=y,
        threshold=0.5,
        population_size=4,
        generations=1,
        mutation_rate=1.0,
    )


def assert_valid_genes(genes):
    for name, value in genes.items():
        low, high, kind = HYPERPARAMETER_SPACE[name]
        assert low <= value <= high
        if kind == "int":
            assert isinstance(value, int)
        else:
            assert isinstance(value, float)


def test_initial_population_generates_valid_individuals():
    optimizer = make_optimizer()
    population = optimizer.generate_initial_population()
    assert len(population) == 4
    for individual in population:
        assert set(individual.genes) == set(HYPERPARAMETER_SPACE)
        assert_valid_genes(individual.genes)


def test_mutation_keeps_hyperparameters_inside_limits():
    optimizer = make_optimizer()
    individual = optimizer.create_individual()
    mutated = optimizer.mutate(individual)
    assert_valid_genes(mutated.genes)


def test_crossover_returns_valid_child():
    optimizer = make_optimizer()
    parent_a = optimizer.create_individual()
    parent_b = optimizer.create_individual()
    child = optimizer.crossover(parent_a, parent_b)
    assert isinstance(child, Individual)
    assert_valid_genes(child.genes)


def test_fitness_score_returns_number():
    metrics = {
        "recall": 0.8,
        "f1_score": 0.6,
        "true_positives": 8,
        "false_negatives": 2,
    }
    value = fitness_score(metrics)
    assert isinstance(value, float)
