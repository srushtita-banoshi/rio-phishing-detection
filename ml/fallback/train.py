"""Evaluate intent-transition detection with progressively missing context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

SEED = 20261031


def _context(frame: pd.DataFrame, condition: str) -> pd.Series:
    current = frame["current_email"].fillna("").astype(str)
    previous = frame["previous_email"].fillna("").astype(str)
    if condition == "full_history":
        return previous + " [CURRENT] " + current
    if condition == "partial_history":
        return previous.str[-80:] + " [CURRENT] " + current
    if condition == "no_previous_messages":
        return current
    raise ValueError(f"Unknown context condition: {condition}")


def _evaluate(model, features, labels: pd.Series) -> dict[str, object]:
    predictions = model.predict(features)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "by_transition_label": {
            label: {
                "count": int((labels == label).sum()),
                "correct": int(((labels == label) & (predictions == label)).sum()),
            }
            for label in sorted(labels.unique())
        },
    }


def train(data_path: Path, model_dir: Path, report_dir: Path, seed: int = SEED) -> dict[str, object]:
    frame = pd.read_csv(data_path)
    required = {"previous_email", "current_email", "transition_label", "scenario_id", "synthetic"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if not frame["synthetic"].astype(bool).all():
        raise ValueError("This pipeline expects the synthetic Dataset C marker.")

    labels = frame["transition_label"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_indices, test_indices = next(splitter.split(frame, labels, groups=frame["scenario_id"]))
    train_labels = labels.iloc[train_indices]
    test_labels = labels.iloc[test_indices]
    model_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}

    for condition in ("full_history", "partial_history", "no_previous_messages"):
        texts = _context(frame, condition)
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        train_features = vectorizer.fit_transform(texts.iloc[train_indices])
        test_features = vectorizer.transform(texts.iloc[test_indices])
        model = LogisticRegression(max_iter=1000, random_state=seed)
        model.fit(train_features, train_labels)
        joblib.dump({"model": model, "vectorizer": vectorizer, "condition": condition}, model_dir / f"{condition}.joblib")
        results[condition] = _evaluate(model, test_features, test_labels)

    report = {
        "dataset": "synthetic_intent_transitions.csv",
        "synthetic_only_warning": "These metrics measure synthetic-data fallback behavior and do not demonstrate real-world performance.",
        "seed": seed,
        "experiment": {
            "same_current_email": True,
            "conditions": ["full_history", "partial_history", "no_previous_messages"],
            "missing_history_feature_used": False,
            "split": "GroupShuffleSplit by scenario_id",
            "train_rows": len(train_indices),
            "test_rows": len(test_indices),
            "overlapping_groups": int(len(set(frame.iloc[train_indices]["scenario_id"]) & set(frame.iloc[test_indices]["scenario_id"]))),
        },
        "models": results,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "fallback_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/synthetic/synthetic_intent_transitions.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/fallback"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train(args.data, args.model_dir, args.report_dir, args.seed)
