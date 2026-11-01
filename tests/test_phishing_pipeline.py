from pathlib import Path

from data_generation.generate_phishing import generate
from ml.phishing.train import train


def test_phishing_pipeline_saves_models_and_group_split(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    report_dir = tmp_path / "reports"
    model_dir = tmp_path / "models"
    generate(data_dir, seed=5)
    report = train(data_dir / "synthetic_phishing_emails.csv", model_dir, report_dir, seed=5)
    assert report["split"]["overlapping_groups"] == 0
    assert set(report["models"]) == {"logistic_regression", "random_forest"}
    for model_name in report["models"]:
        assert 0 <= report["models"][model_name]["accuracy"] <= 1
        assert (model_dir / f"{model_name}.joblib").exists()
    assert (report_dir / "phishing_metrics.json").exists()
