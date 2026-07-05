from src.tc_fase2.threshold_tuning import choose_best_threshold, tune_threshold


def test_threshold_tuning_returns_tested_thresholds(tmp_path):
    results, best = tune_threshold(
        y_true=[0, 0, 1, 1],
        probabilities=[0.05, 0.25, 0.70, 0.95],
        thresholds=[0.10, 0.50],
        reports_dir=tmp_path,
    )
    assert list(results["threshold"].sort_values()) == [0.10, 0.50]
    assert "fitness" in results.columns
    assert (tmp_path / "threshold_tuning.csv").exists()
    assert (tmp_path / "best_threshold.json").exists()
    assert best["threshold"] in [0.10, 0.50]


def test_choose_best_threshold_validates_non_empty():
    results, _ = tune_threshold(
        y_true=[0, 1],
        probabilities=[0.1, 0.9],
        thresholds=[0.5],
        save_outputs=False,
    )
    best = choose_best_threshold(results)
    assert best["threshold"] == 0.5
