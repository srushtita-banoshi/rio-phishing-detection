"""Multi-turn conversation thread processing and trajectory analysis for RIO."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.explain import explain_transition_shift
from ml.intent.train import extract_requested_action
from ml.transition.train import _cosine_similarity, _features, rule_predict


@dataclass
class ConversationMessage:
    turn_index: int
    thread_id: int | str
    timestamp: str
    sender: str
    recipient: str
    subject: str
    body: str
    cleaned_body: str


@dataclass
class ConversationThread:
    thread_id: int | str
    subject: str
    messages: list[ConversationMessage] = field(default_factory=list)

    @property
    def turn_count(self) -> int:
        return len(self.messages)


def clean_body_text(raw_text: str) -> str:
    """Clean quoted-printable artifacts, email headers, and reply blocks from raw email body."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return ""

    text = raw_text

    # Decode quoted-printable artifacts common in enterprise email logs like Enron
    text = text.replace("=20", " ").replace("=09", " ").replace("=\n", "")
    text = re.sub(r"=([0-9A-Fa-f]{2})", " ", text)

    # Remove forwarded / reply quotation headers
    text = re.sub(
        r"-+\s*(?:Original Message|Forwarded by|Forwarded Message)\s*-+[\s\S]*?(?:Subject:|Date:|To:|From:)[^\n]*\n",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove standard quoting prefix '>'
    text = re.sub(r"^\s*>.*$", "", text, flags=re.MULTILINE)

    # Collapse excess whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_conversation_threads(
    csv_path: Path | str,
    max_threads: int = 50,
    min_turns: int = 2,
    max_rows_to_scan: int = 15000,
) -> dict[int | str, ConversationThread]:
    """
    Safely load conversation threads from email_thread_details.csv.
    Groups messages by thread_id and orders each thread chronologically by timestamp.
    Does NOT invent or assign unverified labels.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Thread dataset not found at {path}")

    # Read with chunking/limit for responsive loading
    chunks = pd.read_csv(
        path,
        nrows=max_rows_to_scan,
        usecols=["thread_id", "subject", "timestamp", "from", "to", "body"],
        dtype={"thread_id": str, "subject": str, "timestamp": str, "from": str, "to": str, "body": str},
    )

    chunks["timestamp"] = pd.to_datetime(chunks["timestamp"], errors="coerce")
    chunks = chunks.sort_values(by=["thread_id", "timestamp"]).reset_index(drop=True)

    threads: dict[int | str, ConversationThread] = {}

    for thread_id, group in chunks.groupby("thread_id", sort=False):
        if len(group) < min_turns:
            continue

        first_subject = str(group["subject"].dropna().iloc[0]) if not group["subject"].dropna().empty else "No Subject"
        conv_thread = ConversationThread(thread_id=thread_id, subject=first_subject)

        for turn_idx, (_, row) in enumerate(group.iterrows()):
            raw_body = str(row["body"]) if pd.notna(row["body"]) else ""
            cleaned = clean_body_text(raw_body)
            # If cleaning left it empty, preserve snippet of raw body
            if not cleaned and raw_body:
                cleaned = raw_body[:300].strip()

            msg = ConversationMessage(
                turn_index=turn_idx,
                thread_id=thread_id,
                timestamp=str(row["timestamp"]) if pd.notna(row["timestamp"]) else "",
                sender=str(row["from"]) if pd.notna(row["from"]) else "Unknown",
                recipient=str(row["to"]) if pd.notna(row["to"]) else "Unknown",
                subject=str(row["subject"]) if pd.notna(row["subject"]) else "",
                body=raw_body,
                cleaned_body=cleaned,
            )
            conv_thread.messages.append(msg)

        threads[thread_id] = conv_thread
        if len(threads) >= max_threads:
            break

    return threads


def analyze_thread_trajectory(
    thread: ConversationThread,
    artifacts: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Analyze intent progression and detect intent-shifts across all turns of a conversation thread.
    Uses trained models to predict intents and transitions dynamically.
    """
    intent_package = artifacts["intent"]
    transition_package = artifacts["transition"]

    trajectory: list[dict[str, Any]] = []

    for i, msg in enumerate(thread.messages):
        turn_text = msg.cleaned_body or msg.subject or "No content"
        # Predict intent for this message
        features = intent_package["vectorizer"].transform([turn_text])
        probabilities = intent_package["model"].predict_proba(features)[0]
        intent_idx = int(np.argmax(probabilities))
        intent = str(intent_package["model"].classes_[intent_idx])
        confidence = float(probabilities[intent_idx])
        action = extract_requested_action(turn_text)

        turn_result: dict[str, Any] = {
            "turn_index": i,
            "sender": msg.sender,
            "timestamp": msg.timestamp,
            "snippet": turn_text[:140] + ("..." if len(turn_text) > 140 else ""),
            "intent": intent,
            "intent_confidence": confidence,
            "requested_action": action,
            "transition": None,
        }

        # If turn > 0, compute transition relative to previous turn
        if i > 0:
            prev_turn = trajectory[i - 1]
            prev_text = thread.messages[i - 1].cleaned_body or thread.messages[i - 1].subject or ""
            curr_text = turn_text

            record = pd.DataFrame([{
                "previous_email": prev_text,
                "current_email": curr_text,
                "previous_intent": prev_turn["intent"],
                "current_intent": intent,
                "previous_requested_action": prev_turn["requested_action"],
                "current_requested_action": action,
            }])

            trans_feat, _ = _features(record)
            scaled = transition_package["scaler"].transform(trans_feat)
            trans_probs = transition_package["model"].predict_proba(scaled)[0]
            trans_idx = int(np.argmax(trans_probs))
            transition_label = str(transition_package["model"].classes_[trans_idx])
            transition_conf = float(trans_probs[trans_idx])
            rule_label = str(rule_predict(record)[0])

            sim = float(_cosine_similarity([prev_text], [curr_text])[0])
            urgency_flag = bool(re.search(r"\b(?:urgent|urgently|immediately|asap|limited)\b", curr_text, re.IGNORECASE))

            explanation = explain_transition_shift(
                previous_intent=prev_turn["intent"],
                current_intent=intent,
                previous_action=prev_turn["requested_action"],
                current_action=action,
                transition_label=transition_label,
                confidence=transition_conf,
                rule_baseline=rule_label,
                urgency_detected=urgency_flag,
                similarity_score=sim,
            )

            turn_result["transition"] = {
                "label": transition_label,
                "confidence": transition_conf,
                "rule_baseline": rule_label,
                "similarity_score": sim,
                "urgency_flag": urgency_flag,
                "explanation": explanation,
            }

        trajectory.append(turn_result)

    return trajectory


def evaluate_missing_context_on_thread(
    thread: ConversationThread,
    artifacts: dict[str, Any],
    target_turn_idx: int = -1,
) -> dict[str, Any]:
    """
    Evaluate how the missing-context fallback model behaves on a real thread turn
    under Full History, Partial History, and No History.
    """
    if len(thread.messages) < 2:
        raise ValueError("Thread must contain at least 2 turns for missing context evaluation.")

    idx = target_turn_idx if target_turn_idx >= 0 else (len(thread.messages) - 1)
    current_msg = thread.messages[idx]
    current_text = current_msg.cleaned_body or current_msg.subject or ""

    prior_texts = [
        m.cleaned_body or m.subject or ""
        for m in thread.messages[:idx]
    ]
    full_prior = " \n ".join(prior_texts)
    immediate_prior = prior_texts[-1] if prior_texts else ""

    fallback_models = artifacts["fallback"]
    results = {}

    for condition in ("full_history", "partial_history", "no_previous_messages"):
        model_pkg = fallback_models[condition]
        if condition == "full_history":
            text = f"{full_prior} [CURRENT] {current_text}"
        elif condition == "partial_history":
            text = f"{immediate_prior[-80:]} [CURRENT] {current_text}"
        else:
            text = current_text

        feats = model_pkg["vectorizer"].transform([text])
        probs = model_pkg["model"].predict_proba(feats)[0]
        max_idx = int(np.argmax(probs))
        results[condition] = {
            "prediction": str(model_pkg["model"].classes_[max_idx]),
            "confidence": float(probs[max_idx]),
        }

    return {
        "thread_id": thread.thread_id,
        "target_turn_index": idx,
        "current_sender": current_msg.sender,
        "current_snippet": current_text[:120],
        "evaluations": results,
    }
