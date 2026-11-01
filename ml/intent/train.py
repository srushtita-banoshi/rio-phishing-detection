"""Train synthetic email intent classifiers and extract requested actions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

SEED = 20261031
ACTION_PATTERN = re.compile(r"\b(?:review|check|schedule|share|submit|transfer|verify|confirm|download|grant|approve|read)\b[^.!?]*", re.IGNORECASE)


def extract_requested_action(text: str) -> str:
    """Extract the first imperative/request phrase as a lightweight baseline."""
    match = ACTION_PATTERN.search(text)
    return match.group(0).strip() if match else ""


def _evaluate(model: Any, features, labels: pd.Series) -> dict[str, float]:
    predictions = model.predict(features)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def _confidence(model: Any, features) -> np.ndarray:
    return np.max(model.predict_proba(features), axis=1)


def _semantic_features(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model_name)
    return np.asarray(encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True))


def train(
    data_path: Path,
    model_dir: Path,
    report_dir: Path,
    seed: int = SEED,
    semantic_model: str = "all-MiniLM-L6-v2",
    skip_semantic: bool = False,
) -> dict[str, object]:
    frame = pd.read_csv(data_path)
    required = {"email_text", "intent_label", "requested_action", "scenario_id", "synthetic"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if not frame["synthetic"].astype(bool).all():
        raise ValueError("This pipeline expects the synthetic Dataset B marker.")

    labels = frame["intent_label"].astype(str)
    texts = frame["email_text"].fillna("").astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_indices, test_indices = next(splitter.split(frame, labels, groups=frame["scenario_id"]))
    train_labels = labels.iloc[train_indices]
    test_labels = labels.iloc[test_indices]

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    train_features = vectorizer.fit_transform(texts.iloc[train_indices])
    test_features = vectorizer.transform(texts.iloc[test_indices])
    baseline = LogisticRegression(max_iter=1000, random_state=seed)
    baseline.fit(train_features, train_labels)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": baseline, "vectorizer": vectorizer}, model_dir / "tfidf_logistic_regression.joblib")
    results: dict[str, object] = {
        "tfidf_logistic_regression": {
            **_evaluate(baseline, test_features, test_labels),
            "mean_confidence": float(_confidence(baseline, test_features).mean()),
        }
    }

    semantic_status = "skipped"
    if not skip_semantic:
        try:
            semantic_train = _semantic_features(texts.iloc[train_indices].tolist(), semantic_model)
            semantic_test = _semantic_features(texts.iloc[test_indices].tolist(), semantic_model)
            semantic_classifier = LogisticRegression(max_iter=1000, random_state=seed)
            semantic_classifier.fit(semantic_train, train_labels)
            joblib.dump(
                {"model": semantic_classifier, "encoder_name": semantic_model},
                model_dir / "sentence_transformer_logistic_regression.joblib",
            )
            results["sentence_transformer_logistic_regression"] = {
                **_evaluate(semantic_classifier, semantic_test, test_labels),
                "mean_confidence": float(_confidence(semantic_classifier, semantic_test).mean()),
            }
            semantic_status = "completed"
        except ImportError:
            semantic_status = "unavailable: install sentence-transformers to run this comparison"

    report = {
        "dataset": "synthetic_intent_emails.csv",
        "synthetic_only_warning": "These metrics measure performance on synthetic data and do not demonstrate real-world intent understanding.",
        "seed": seed,
        "split": {
            "method": "GroupShuffleSplit by scenario_id",
            "train_rows": len(train_indices),
            "test_rows": len(test_indices),
            "overlapping_groups": int(len(set(frame.iloc[train_indices]["scenario_id"]) & set(frame.iloc[test_indices]["scenario_id"]))),
        },
        "requested_action_extraction": {
            "method": "lightweight regular-expression request phrase extraction",
            "test_nonempty_rate": float((texts.iloc[test_indices].map(extract_requested_action) != "").mean()),
        },
        "semantic_model": {"name": semantic_model, "status": semantic_status},
        "models": results,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "intent_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/synthetic/synthetic_intent_emails.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/intent"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--semantic-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--skip-semantic", action="store_true")
    args = parser.parse_args()
    train(args.data, args.model_dir, args.report_dir, args.seed, args.semantic_model, args.skip_semantic)
