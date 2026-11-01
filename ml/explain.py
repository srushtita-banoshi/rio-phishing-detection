"""Explainability module for RIO phishing, intent, and intent-shift models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack


def explain_phishing_prediction(
    package: dict[str, Any],
    subject: str,
    body: str,
    metadata_df: pd.DataFrame,
    prediction_label: str,
    confidence: float,
    top_n: int = 5,
) -> dict[str, Any]:
    """Generate transparent feature attribution explanations for phishing predictions."""
    model = package["model"]
    vectorizer = package["vectorizer"]
    preprocessor = package["preprocessor"]

    text = f"{subject} {body}"
    text_features = vectorizer.transform([text])
    meta_features = preprocessor.transform(metadata_df)
    features = hstack([text_features, meta_features]).tocsr()

    vocab = vectorizer.get_feature_names_out()
    meta_feature_names = preprocessor.get_feature_names_out()
    all_feature_names = np.concatenate([vocab, meta_feature_names])

    contributing_factors = []
    text_evidence = []

    # If linear model (Logistic Regression), inspect weights
    if hasattr(model, "coef_"):
        weights = model.coef_[0]
        # Calculate contribution for each active feature: x_i * w_i
        feature_indices = features.indices
        feature_values = features.data
        contributions = []
        for idx, val in zip(feature_indices, feature_values):
            contribution = weights[idx] * val
            contributions.append((all_feature_names[idx], contribution, val))

        # Sort by contribution: positive towards phishing (class 1), negative towards legitimate (class 0)
        sorted_contributions = sorted(contributions, key=lambda x: x[1], reverse=(prediction_label == "phishing"))

        for name, contrib, val in sorted_contributions[:top_n]:
            clean_name = name.replace("numeric__", "").replace("categorical__", "")
            direction = "increases phishing risk" if contrib > 0 else "indicates legitimacy"
            contributing_factors.append({
                "feature": clean_name,
                "impact": direction,
                "score": round(float(contrib), 4),
            })
            if name in vocab and contrib > 0:
                text_evidence.append(name)

    # Metadata rule-based risk indicators
    risk_flags = []
    row = metadata_df.iloc[0]
    if row.get("url_count", 0) > 0 and not row.get("domain_uses_https", 1):
        risk_flags.append("Insecure URLs detected (HTTP instead of HTTPS).")
    if row.get("domain_has_subdomain", 0):
        risk_flags.append("Sender or URL domain contains multiple subdomains.")
    if row.get("urgency_indicator_count", 0) > 0:
        risk_flags.append(f"High urgency language detected ({row.get('urgency_indicator_count')} urgency indicators).")
    if row.get("authentication_result") in ("none", "softfail"):
        risk_flags.append(f"Email authentication verification flagged: {row.get('authentication_result')}.")
    if row.get("has_attachment", 0) and row.get("attachment_type") in ("exe", "zip"):
        risk_flags.append(f"Potentially hazardous attachment type: {row.get('attachment_type')}.")

    # Generate narrative summary
    if prediction_label == "phishing":
        if risk_flags:
            summary = "Flagged as PHISHING primarily due to: " + " ".join(risk_flags)
        else:
            summary = "Flagged as PHISHING due to linguistic indicators matching phishing patterns in the subject and body."
    else:
        summary = "Classified as LEGITIMATE. Email exhibits standard organizational characteristics without anomalous urgency or credential prompts."

    return {
        "prediction": prediction_label,
        "confidence": confidence,
        "summary": summary,
        "risk_flags": risk_flags,
        "top_contributing_features": contributing_factors,
        "suspicious_terms_found": list(set(text_evidence)),
    }


def explain_intent_prediction(
    package: dict[str, Any],
    text: str,
    predicted_intent: str,
    confidence: float,
    action_phrase: str,
    top_n: int = 4,
) -> dict[str, Any]:
    """Explain why an email was classified with a specific intent."""
    model = package["model"]
    vectorizer = package["vectorizer"]

    features = vectorizer.transform([text])
    vocab = vectorizer.get_feature_names_out()

    active_tokens = []
    if hasattr(model, "classes_") and hasattr(model, "coef_"):
        class_idx = list(model.classes_).index(predicted_intent)
        weights = model.coef_[class_idx]
        for idx in features.indices:
            token = vocab[idx]
            weight = weights[idx]
            if weight > 0:
                active_tokens.append((token, float(weight)))

        active_tokens = sorted(active_tokens, key=lambda x: x[1], reverse=True)[:top_n]

    token_names = [t[0] for t in active_tokens]
    rationale = f"Classified as '{predicted_intent}'"
    if token_names:
        rationale += f" driven by keyword cues: {', '.join(repr(t) for t in token_names)}."
    if action_phrase:
        rationale += f" Extracted requested action: '{action_phrase}'."

    return {
        "intent": predicted_intent,
        "confidence": confidence,
        "action_phrase": action_phrase or "None detected",
        "rationale": rationale,
        "key_intent_cues": token_names,
    }


def explain_transition_shift(
    previous_intent: str,
    current_intent: str,
    previous_action: str,
    current_action: str,
    transition_label: str,
    confidence: float,
    rule_baseline: str,
    urgency_detected: bool,
    similarity_score: float,
) -> dict[str, Any]:
    """Provide structured explanation of why a conversational transition is flagged or cleared."""
    sensitive_intents = {
        "credential_submission",
        "financial_transfer",
        "account_verification",
        "payment_confirmation",
        "access_permission",
    }
    is_sensitive = current_intent in sensitive_intents
    intent_changed = previous_intent != current_intent

    reasons = []
    if intent_changed:
        reasons.append(f"Intent changed from '{previous_intent}' to '{current_intent}'.")
    else:
        reasons.append(f"Intent remained consistent ('{current_intent}').")

    if is_sensitive:
        reasons.append(f"Current message involves sensitive action category ('{current_intent}').")
    if urgency_detected:
        reasons.append("Current message introduces artificial urgency or immediate deadline pressure.")
    if similarity_score < 0.15 and intent_changed:
        reasons.append(f"Low semantic continuity with previous conversation (text similarity: {similarity_score:.1%}).")

    if transition_label == "suspicious intent shift":
        assessment = (
            "SUSPICIOUS INTENT SHIFT: The conversation pivots toward a sensitive action with heightened urgency "
            "or without prior context established in earlier thread turns."
        )
    elif transition_label == "legitimate intent change":
        assessment = (
            "LEGITIMATE INTENT CHANGE: The conversation naturally transitions to a new business topic "
            "without abnormal pressure, credential requests, or unauthorized financial demands."
        )
    else:
        assessment = "NORMAL CONTINUATION: The conversation maintains continuity on the same topic and purpose."

    return {
        "transition_label": transition_label,
        "confidence": confidence,
        "rule_consensus": rule_baseline,
        "assessment": assessment,
        "details": reasons,
        "metrics": {
            "intent_changed": intent_changed,
            "is_sensitive_intent": is_sensitive,
            "has_urgency": urgency_detected,
            "thread_text_similarity": round(similarity_score, 4),
        },
    }
