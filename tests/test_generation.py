from pathlib import Path

import pandas as pd

from data_generation.generate_intents import generate as generate_intents
from data_generation.generate_phishing import generate as generate_phishing
from data_generation.generate_transitions import generate as generate_transitions


def test_generators_are_reproducible(tmp_path: Path) -> None:
    first = generate_phishing(tmp_path / "first", seed=17)
    second = generate_phishing(tmp_path / "second", seed=17)
    pd.testing.assert_frame_equal(first, second)


def test_dataset_a_balance_and_quality(tmp_path: Path) -> None:
    frame = generate_phishing(tmp_path, seed=3)
    assert len(frame) == 2000
    assert frame["label_name"].value_counts().to_dict() == {"legitimate": 1000, "phishing": 1000}
    assert frame["synthetic"].all()
    assert frame["template_family"].nunique() >= 10
    assert frame.duplicated().sum() == 0


def test_dataset_b_intents(tmp_path: Path) -> None:
    frame = generate_intents(tmp_path, seed=3)
    assert len(frame) == 1000
    assert frame["intent_label"].nunique() == 10
    assert frame["intent_label"].value_counts().min() >= 90
    assert frame.isna().sum().sum() == 0
    assert frame["email_text"].duplicated().sum() == 0


def test_dataset_c_counts_and_linked_threads(tmp_path: Path) -> None:
    frame = generate_transitions(tmp_path, seed=3)
    assert len(frame) == 1000
    assert frame["transition_label"].value_counts().to_dict() == {
        "normal continuation": 350,
        "legitimate intent change": 250,
        "suspicious intent shift": 400,
    }
    assert frame["transition_subtype"].value_counts().to_dict()["clearly suspicious transition"] == 150
    assert frame["thread_id"].is_unique
    assert frame["previous_email_id"].is_unique
    assert frame["current_email_id"].is_unique
    assert frame["previous_email"].duplicated().sum() == 0
    assert frame["current_email"].duplicated().sum() == 0
    normal = frame[frame["transition_label"] == "normal continuation"]
    assert (normal["previous_intent"] == normal["current_intent"]).all()
