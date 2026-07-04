from src.tc_fase2.metrics import calculate_metrics, evaluate_probabilities


def test_basic_metrics_calculation():
    metrics = calculate_metrics([0, 0, 1, 1], [0, 1, 0, 1])
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 1
    assert metrics["true_negatives"] == 1
    assert metrics["recall"] == 0.5


def test_evaluate_probabilities_adds_threshold_and_fitness():
    metrics = evaluate_probabilities([0, 1], [0.1, 0.9], threshold=0.5)
    assert metrics["threshold"] == 0.5
    assert isinstance(metrics["fitness"], float)
