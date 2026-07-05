import pytest

from src.tc_fase2.train_optimized_model import QUICK_RESULTS_ERROR, load_best_hyperparameters


def test_train_optimized_blocks_quick_results_without_flag(tmp_path):
    summary = tmp_path / "ga_experiments_summary.csv"
    summary.write_text(
        "experiment,quick,best_fitness,best_hyperparameters\n"
        "1,True,0.9,\"{'max_depth': 3, 'learning_rate': 0.1}\"\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="quick=True"):
        load_best_hyperparameters(summary, allow_quick_results=False)


def test_train_optimized_allows_quick_results_with_flag(tmp_path):
    summary = tmp_path / "ga_experiments_summary.csv"
    summary.write_text(
        "experiment,quick,best_fitness,best_hyperparameters\n"
        "1,True,0.9,\"{'max_depth': 3, 'learning_rate': 0.1}\"\n",
        encoding="utf-8",
    )
    params = load_best_hyperparameters(summary, allow_quick_results=True)
    assert params["max_depth"] == 3
    assert "sem --quick" in QUICK_RESULTS_ERROR
