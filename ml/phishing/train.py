"""Train and evaluate the initial synthetic phishing classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 20261031
TEXT_COLUMNS = ["subject", "body"]
NUMERIC_COLUMNS = ["url_count", "domain_uses_https", "domain_has_subdomain", "has_attachment", "urgency_indicator_count"]
CATEGORICAL_COLUMNS = ["authentication_result", "attachment_type"]


def _metrics(model, features, labels) -> dict[str, object]:
    predictions = model.predict(features)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "confusion_matrix": matrix.tolist(),
        "support": int(len(labels)),
    }


def train(data_path: Path, model_dir: Path, report_dir: Path, seed: int = SEED) -> dict[str, object]:
    frame = pd.read_csv(data_path)
    required = set(TEXT_COLUMNS + NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + ["label", "scenario_id"])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if not frame["synthetic"].astype(bool).all():
        raise ValueError("This pipeline expects the synthetic Dataset A marker.")

    text = frame[TEXT_COLUMNS].fillna("").agg(" ".join, axis=1)
    labels = frame["label"].astype(int)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_indices, test_indices = next(splitter.split(frame, labels, groups=frame["scenario_id"]))

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    train_text = vectorizer.fit_transform(text.iloc[train_indices])
    test_text = vectorizer.transform(text.iloc[test_indices])

    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(with_mean=False), NUMERIC_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )
    train_meta = preprocessor.fit_transform(frame.iloc[train_indices])
    test_meta = preprocessor.transform(frame.iloc[test_indices])
    train_features = hstack([train_text, train_meta]).tocsr()
    test_features = hstack([test_text, test_meta]).tocsr()

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1, class_weight="balanced"),
    }
    results: dict[str, object] = {}
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        model.fit(train_features, labels.iloc[train_indices])
        joblib.dump(
            {"model": model, "vectorizer": vectorizer, "preprocessor": preprocessor, "text_columns": TEXT_COLUMNS},
            model_dir / f"{name}.joblib",
        )
        results[name] = _metrics(model, test_features, labels.iloc[test_indices])

    report = {
        "dataset": "synthetic_phishing_emails.csv",
        "synthetic_only_warning": "These metrics measure performance on synthetic data and do not demonstrate real-world performance.",
        "seed": seed,
        "split": {
            "method": "GroupShuffleSplit by scenario_id",
            "train_rows": len(train_indices),
            "test_rows": len(test_indices),
            "train_groups": int(frame.iloc[train_indices]["scenario_id"].nunique()),
            "test_groups": int(frame.iloc[test_indices]["scenario_id"].nunique()),
            "overlapping_groups": int(len(set(frame.iloc[train_indices]["scenario_id"]) & set(frame.iloc[test_indices]["scenario_id"]))),
        },
        "models": results,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phishing_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/synthetic/synthetic_phishing_emails.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/phishing"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train(args.data, args.model_dir, args.report_dir, args.seed)
