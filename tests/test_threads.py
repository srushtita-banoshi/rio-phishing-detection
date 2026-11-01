from pathlib import Path

import joblib
import pandas as pd

from ml.threads import (
    ConversationMessage,
    ConversationThread,
    analyze_thread_trajectory,
    clean_body_text,
    evaluate_missing_context_on_thread,
    load_conversation_threads,
)


def test_clean_body_text() -> None:
    raw = "Here is the plan =20 for tomorrow.\n -----Original Message-----\n From: boss@corp.com\n Sent: yesterday\n Old notes"
    cleaned = clean_body_text(raw)
    assert "=20" not in cleaned
    assert "Original Message" not in cleaned
    assert "Here is the plan for tomorrow" in cleaned


def test_load_conversation_threads(tmp_path: Path) -> None:
    sample_csv = tmp_path / "email_thread_details.csv"
    data = [
        {"thread_id": 101, "subject": "Review Request", "timestamp": "2026-01-01 10:00:00", "from": "alice@corp.com", "to": "bob@corp.com", "body": "Please review the attached notes."},
        {"thread_id": 101, "subject": "Re: Review Request", "timestamp": "2026-01-01 11:00:00", "from": "bob@corp.com", "to": "alice@corp.com", "body": "Looks good to me."},
        {"thread_id": 102, "subject": "One-off", "timestamp": "2026-01-01 12:00:00", "from": "charlie@corp.com", "to": "david@corp.com", "body": "Single message."},
    ]
    pd.DataFrame(data).to_csv(sample_csv, index=False)

    threads = load_conversation_threads(sample_csv, min_turns=2)
    assert len(threads) == 1
    assert "101" in threads or 101 in threads
    t = threads.get("101") or threads.get(101)
    assert t.turn_count == 2
    assert t.messages[0].sender == "alice@corp.com"
    assert t.messages[1].sender == "bob@corp.com"


def test_thread_trajectory_and_fallback(tmp_path: Path) -> None:
    # Load actual project models
    root = Path(__file__).resolve().parent.parent
    artifacts = {
        "intent": joblib.load(root / "models/intent/tfidf_logistic_regression.joblib"),
        "transition": joblib.load(root / "models/transition/logistic_regression.joblib"),
        "fallback": {
            condition: joblib.load(root / f"models/fallback/{condition}.joblib")
            for condition in ("full_history", "partial_history", "no_previous_messages")
        },
    }

    thread = ConversationThread(thread_id="test-1", subject="Project Update")
    thread.messages.append(
        ConversationMessage(
            turn_index=0,
            thread_id="test-1",
            timestamp="2026-01-01 09:00:00",
            sender="pm@company.com",
            recipient="dev@company.com",
            subject="Project Update",
            body="Could we schedule a planning meeting next week?",
            cleaned_body="Could we schedule a planning meeting next week?",
        )
    )
    thread.messages.append(
        ConversationMessage(
            turn_index=1,
            thread_id="test-1",
            timestamp="2026-01-01 10:00:00",
            sender="pm@company.com",
            recipient="dev@company.com",
            subject="Re: Project Update",
            body="Also, urgently submit your employee credentials before the review can continue.",
            cleaned_body="Also, urgently submit your employee credentials before the review can continue.",
        )
    )

    trajectory = analyze_thread_trajectory(thread, artifacts)
    assert len(trajectory) == 2
    assert trajectory[0]["intent"] is not None
    assert trajectory[1]["transition"] is not None
    assert trajectory[1]["transition"]["label"] in ("normal continuation", "legitimate intent change", "suspicious intent shift")

    fallback_eval = evaluate_missing_context_on_thread(thread, artifacts)
    assert "full_history" in fallback_eval["evaluations"]
    assert "no_previous_messages" in fallback_eval["evaluations"]
