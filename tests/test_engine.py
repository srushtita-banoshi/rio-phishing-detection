from ml.engine import RIOEngine


def test_rio_engine_analyze_raw_email() -> None:
    engine = RIOEngine()
    result = engine.analyze_raw_email(
        subject="Urgent: Account Verification Required",
        body="Please click https://secure-verification.portal.com/login and verify your credentials immediately.",
        sender="Security Desk <security@alert.portal.com>",
    )

    assert "execution_time_ms" in result
    assert result["execution_time_ms"] > 0
    assert result["phishing"]["prediction"] in ("phishing", "legitimate")
    assert 0 <= result["phishing"]["confidence"] <= 1.0
    assert result["intent"]["intent"] is not None
    assert "explanation" in result["phishing"]
    assert "top_contributing_features" in result["phishing"]["explanation"]


def test_rio_engine_analyze_transition() -> None:
    engine = RIOEngine()
    res = engine.analyze_transition(
        previous_email="Can we schedule a planning meeting next week?",
        current_email="Also, urgently submit your employee credentials before the review can continue.",
    )

    assert "execution_time_ms" in res
    assert res["transition_label"] in ("normal continuation", "legitimate intent change", "suspicious intent shift")
    assert res["explanation"]["assessment"] != ""
