"""Unified real-time inference and analysis engine for RIO."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack

from ml.explain import (
    explain_intent_prediction,
    explain_phishing_prediction,
    explain_transition_shift,
)
from ml.intent.train import extract_requested_action
from ml.phishing.extractor import extract_email_features, to_model_metadata_dataframe
from ml.transition.train import _cosine_similarity, _features, rule_predict

ROOT = Path(__file__).resolve().parent.parent


def _required(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required model artifact was not found: {path}")
    return path


class RIOEngine:
    """Production-ready real-time analyzer orchestrating all RIO models."""

    def __init__(self, models_dir: Path | str | None = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else (ROOT / "models")
        self._artifacts: dict[str, Any] | None = None

    @property
    def artifacts(self) -> dict[str, Any]:
        if self._artifacts is None:
            self.load_models()
        return self._artifacts  # type: ignore

    def load_models(self) -> None:
        """Load and cache all trained model pipelines."""
        m_dir = self.models_dir
        self._artifacts = {
            "phishing": joblib.load(_required(m_dir / "phishing/logistic_regression.joblib")),
            "phishing_rf": (
                joblib.load(m_dir / "phishing/random_forest.joblib")
                if (m_dir / "phishing/random_forest.joblib").exists()
                else None
            ),
            "intent": joblib.load(_required(m_dir / "intent/tfidf_logistic_regression.joblib")),
            "transition": joblib.load(_required(m_dir / "transition/logistic_regression.joblib")),
            "transition_rf": (
                joblib.load(m_dir / "transition/random_forest.joblib")
                if (m_dir / "transition/random_forest.joblib").exists()
                else None
            ),
            "fallback": {
                condition: joblib.load(_required(m_dir / f"fallback/{condition}.joblib"))
                for condition in ("full_history", "partial_history", "no_previous_messages")
            },
        }

    def analyze_raw_email(
        self,
        subject: str,
        body: str,
        sender: str = "",
        headers: dict[str, str] | None = None,
        attachment_filename: str | None = None,
    ) -> dict[str, Any]:
        """Perform real-time multi-feature phishing detection, intent extraction, and explainability."""
        start_time = time.perf_counter()

        # 1. Automatic multi-feature extraction
        features = extract_email_features(
            subject=subject,
            body=body,
            sender=sender,
            headers=headers,
            attachment_filename=attachment_filename,
        )
        metadata_df = to_model_metadata_dataframe(features)

        # 2. Phishing classification
        pkg = self.artifacts["phishing"]
        text_features = pkg["vectorizer"].transform([features["combined_text"]])
        meta_features = pkg["preprocessor"].transform(metadata_df)
        combined_features = hstack([text_features, meta_features]).tocsr()

        phishing_pred = int(pkg["model"].predict(combined_features)[0])
        phishing_probs = pkg["model"].predict_proba(combined_features)[0]
        phishing_confidence = float(np.max(phishing_probs))
        phishing_label = "phishing" if phishing_pred else "legitimate"

        # 3. Intent extraction
        intent_pkg = self.artifacts["intent"]
        intent_feats = intent_pkg["vectorizer"].transform([features["combined_text"]])
        intent_probs = intent_pkg["model"].predict_proba(intent_feats)[0]
        intent_idx = int(np.argmax(intent_probs))
        predicted_intent = str(intent_pkg["model"].classes_[intent_idx])
        intent_confidence = float(intent_probs[intent_idx])
        action_phrase = extract_requested_action(features["combined_text"])

        # 4. Explainable predictions
        phishing_explanation = explain_phishing_prediction(
            package=pkg,
            subject=subject,
            body=body,
            metadata_df=metadata_df,
            prediction_label=phishing_label,
            confidence=phishing_confidence,
        )

        intent_explanation = explain_intent_prediction(
            package=intent_pkg,
            text=features["combined_text"],
            predicted_intent=predicted_intent,
            confidence=intent_confidence,
            action_phrase=action_phrase,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "execution_time_ms": round(elapsed_ms, 2),
            "phishing": {
                "prediction": phishing_label,
                "confidence": phishing_confidence,
                "probabilities": {
                    "legitimate": float(phishing_probs[0]),
                    "phishing": float(phishing_probs[1]),
                },
                "explanation": phishing_explanation,
            },
            "intent": {
                "intent": predicted_intent,
                "confidence": intent_confidence,
                "requested_action": action_phrase,
                "explanation": intent_explanation,
            },
            "extracted_features": features,
        }

    def analyze_transition(
        self,
        previous_email: str,
        current_email: str,
        previous_intent: str | None = None,
        current_intent: str | None = None,
    ) -> dict[str, Any]:
        """Analyze intent transition between consecutive email turns."""
        start_time = time.perf_counter()

        intent_pkg = self.artifacts["intent"]
        if previous_intent is None:
            p_feat = intent_pkg["vectorizer"].transform([previous_email])
            previous_intent = str(intent_pkg["model"].classes_[int(np.argmax(intent_pkg["model"].predict_proba(p_feat)[0]))])
        if current_intent is None:
            c_feat = intent_pkg["vectorizer"].transform([current_email])
            current_intent = str(intent_pkg["model"].classes_[int(np.argmax(intent_pkg["model"].predict_proba(c_feat)[0]))])

        p_action = extract_requested_action(previous_email)
        c_action = extract_requested_action(current_email)

        record = pd.DataFrame([{
            "previous_email": previous_email,
            "current_email": current_email,
            "previous_intent": previous_intent,
            "current_intent": current_intent,
            "previous_requested_action": p_action,
            "current_requested_action": c_action,
        }])

        trans_pkg = self.artifacts["transition"]
        trans_feats, _ = _features(record)
        scaled = trans_pkg["scaler"].transform(trans_feats)
        probs = trans_pkg["model"].predict_proba(scaled)[0]
        pred_idx = int(np.argmax(probs))
        label = str(trans_pkg["model"].classes_[pred_idx])
        confidence = float(probs[pred_idx])
        rule = str(rule_predict(record)[0])

        sim = float(_cosine_similarity([previous_email], [current_email])[0])
        urgency = bool(trans_feats[0][4])

        explanation = explain_transition_shift(
            previous_intent=previous_intent,
            current_intent=current_intent,
            previous_action=p_action,
            current_action=c_action,
            transition_label=label,
            confidence=confidence,
            rule_baseline=rule,
            urgency_detected=urgency,
            similarity_score=sim,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "execution_time_ms": round(elapsed_ms, 2),
            "transition_label": label,
            "confidence": confidence,
            "rule_consensus": rule,
            "similarity_score": round(sim, 4),
            "previous_intent": previous_intent,
            "current_intent": current_intent,
            "previous_action": p_action,
            "current_action": c_action,
            "explanation": explanation,
        }
