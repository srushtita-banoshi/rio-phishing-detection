from pathlib import Path

from data_generation.generate_intents import generate
from ml.intent.train import extract_requested_action, train


def test_intent_pipeline_baseline_and_action_extraction(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    generate(data_dir, seed=7)
    report = train(
        data_dir / "synthetic_intent_emails.csv",
        tmp_path / "models",
        tmp_path / "reports",
        seed=7,
        skip_semantic=True,
    )
    assert report["split"]["overlapping_groups"] == 0
    assert "tfidf_logistic_regression" in report["models"]
    assert 0 <= report["models"]["tfidf_logistic_regression"]["mean_confidence"] <= 1
    assert extract_requested_action("Please review the project brief.") == "review the project brief"
