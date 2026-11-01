from pathlib import Path

from data_generation.generate_transitions import generate
from ml.transition.train import rule_predict, train


def test_transition_pipeline_and_rule_baseline(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    frame = generate(data_dir, seed=9)
    report = train(
        data_dir / "synthetic_intent_transitions.csv",
        tmp_path / "models",
        tmp_path / "reports",
        seed=9,
    )
    assert report["split"]["overlapping_groups"] == 0
    assert set(report["models"]) == {"rule_based", "logistic_regression", "random_forest"}
    assert len(rule_predict(frame)) == 1000
    assert (tmp_path / "models" / "logistic_regression.joblib").exists()
    assert (tmp_path / "models" / "random_forest.joblib").exists()
