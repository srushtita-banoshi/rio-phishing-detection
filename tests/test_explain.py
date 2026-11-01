from pathlib import Path

import joblib
import pandas as pd

from ml.explain import (
    explain_intent_prediction,
    explain_phishing_prediction,
    explain_transition_shift,
)


def test_explain_phishing_prediction() -> None:
    root = Path(__file__).resolve().parent.parent
    package = joblib.load(root / "models/phishing/logistic_regression.joblib")

    metadata = pd.DataFrame([{
        "url_count": 2,
        "domain_uses_https": 0,
        "domain_has_subdomain": 1,
        "has_attachment": 0,
        "urgency_indicator_count": 2,
        "authentication_result": "none",
        "attachment_type": "none",
    }])

    exp = explain_phishing_prediction(
        package=package,
        subject="Urgent account review",
        body="Sign in immediately to verify your account.",
        metadata_df=metadata,
        prediction_label="phishing",
        confidence=0.98,
    )

    assert exp["prediction"] == "phishing"
    assert len(exp["risk_flags"]) >= 1
    assert len(exp["top_contributing_features"]) >= 1
    assert "PHISHING" in exp["summary"]


def test_explain_intent_prediction() -> None:
    root = Path(__file__).resolve().parent.parent
    package = joblib.load(root / "models/intent/tfidf_logistic_regression.joblib")

    exp = explain_intent_prediction(
        package=package,
        text="Could we schedule a planning meeting next week?",
        predicted_intent="meeting_scheduling",
        confidence=0.95,
        action_phrase="schedule a planning meeting",
    )

    assert exp["intent"] == "meeting_scheduling"
    assert exp["action_phrase"] == "schedule a planning meeting"
    assert "meeting_scheduling" in exp["rationale"]


def test_explain_transition_shift() -> None:
    exp = explain_transition_shift(
        previous_intent="meeting_scheduling",
        current_intent="credential_submission",
        previous_action="schedule meeting",
        current_action="submit credentials",
        transition_label="suspicious intent shift",
        confidence=0.92,
        rule_baseline="suspicious intent shift",
        urgency_detected=True,
        similarity_score=0.08,
    )

    assert exp["transition_label"] == "suspicious intent shift"
    assert exp["metrics"]["intent_changed"] is True
    assert exp["metrics"]["is_sensitive_intent"] is True
    assert len(exp["details"]) >= 2
