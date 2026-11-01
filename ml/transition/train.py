"""Train and evaluate synthetic intent-transition detectors."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import Bunch

SEED = 20261031
SENSITIVE_INTENTS = {"credential_submission", "financial_transfer", "account_verification", "payment_confirmation", "access_permission"}
URGENT_PATTERN = re.compile(r"\b(?:urgent|urgently|immediately|asap|limited)\b", re.IGNORECASE)


def _cosine_similarity(previous: list[str], current: list[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    combined = vectorizer.fit_transform(previous + current)
    previous_vectors = combined[: len(previous)]
    current_vectors = combined[len(previous) :]
    numerator = previous_vectors.multiply(current_vectors).sum(axis=1).A1
    denominator = np.sqrt(previous_vectors.multiply(previous_vectors).sum(axis=1).A1) * np.sqrt(
        current_vectors.multiply(current_vectors).sum(axis=1).A1
    )
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)


def rule_predict(frame: pd.DataFrame) -> np.ndarray:
    """Conservative baseline: flag sensitive, urgent intent shifts only."""
    changed = frame["previous_intent"] != frame["current_intent"]
    sensitive = frame["current_intent"].isin(SENSITIVE_INTENTS)
    urgent = frame["current_email"].map(lambda text: bool(URGENT_PATTERN.search(text)))
    return np.where(~changed, "normal continuation", np.where(sensitive & urgent, "suspicious intent shift", "legitimate intent change"))


def _features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    similarity = _cosine_similarity(frame["previous_email"].tolist(), frame["current_email"].tolist())
    previous_action = frame["previous_requested_action"].str.lower()
    current_action = frame["current_requested_action"].str.lower()
    return np.column_stack(
        [
            (frame["previous_intent"] == frame["current_intent"]).astype(int),
            (previous_action == current_action).astype(int),
            similarity,
            frame["current_intent"].isin(SENSITIVE_INTENTS).astype(int),
            frame["current_email"].map(lambda text: int(bool(URGENT_PATTERN.search(text)))),
        ]
    ), ["intent_same", "action_same", "text_similarity", "current_intent_sensitive", "current_email_urgent"]


def _evaluate(predictions: np.ndarray, labels: pd.Series) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "by_transition_label": {
            label: {
                "count": int((labels == label).sum()),
                "predicted_correctly": int(((labels == label) & (predictions == label)).sum()),
            }
            for label in sorted(labels.unique())
        },
    }


def train(data_path: Path, model_dir: Path, report_dir: Path, seed: int = SEED) -> dict[str, object]:
    frame = pd.read_csv(data_path)
    required = {
        "previous_email", "current_email", "previous_intent", "current_intent",
        "previous_requested_action", "current_requested_action", "transition_label",
        "scenario_id", "synthetic",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if not frame["synthetic"].astype(bool).all():
        raise ValueError("This pipeline expects the synthetic Dataset C marker.")

    labels = frame["transition_label"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_indices, test_indices = next(splitter.split(frame, labels, groups=frame["scenario_id"]))
    train_frame = frame.iloc[train_indices]
    test_frame = frame.iloc[test_indices]
    train_features, feature_names = _features(train_frame)
    test_features, _ = _features(test_frame)

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_features)
    scaled_test = scaler.transform(test_features)
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1, class_weight="balanced"),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"rule_based": _evaluate(rule_predict(test_frame), labels.iloc[test_indices])}
    for name, model in models.items():
        model.fit(scaled_train, labels.iloc[train_indices])
        joblib.dump({"model": model, "scaler": scaler, "feature_names": feature_names}, model_dir / f"{name}.joblib")
        results[name] = _evaluate(model.predict(scaled_test), labels.iloc[test_indices])

    report = {
        "dataset": "synthetic_intent_transitions.csv",
        "synthetic_only_warning": "These metrics measure performance on synthetic data and do not demonstrate real-world intent-shift detection.",
        "seed": seed,
        "features": feature_names,
        "semantic_similarity": "TF-IDF cosine similarity between previous and current email text",
        "split": {
            "method": "GroupShuffleSplit by scenario_id",
            "train_rows": len(train_indices),
            "test_rows": len(test_indices),
            "overlapping_groups": int(len(set(train_frame["scenario_id"]) & set(test_frame["scenario_id"]))),
        },
        "models": results,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "transition_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/synthetic/synthetic_intent_transitions.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/transition"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train(args.data, args.model_dir, args.report_dir, args.seed)
