from pathlib import Path

from data_generation.generate_transitions import generate
from ml.fallback.train import _context, train


def test_fallback_conditions_use_same_current_email(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    frame = generate(data_dir, seed=11)
    current_from_full = _context(frame, "full_history").map(lambda text: text.partition("[CURRENT]")[2].strip())
    assert (current_from_full == _context(frame, "no_previous_messages")).all()
    report = train(
        data_dir / "synthetic_intent_transitions.csv",
        tmp_path / "models",
        tmp_path / "reports",
        seed=11,
    )
    assert report["experiment"]["same_current_email"] is True
    assert report["experiment"]["missing_history_feature_used"] is False
    assert set(report["models"]) == {"full_history", "partial_history", "no_previous_messages"}
