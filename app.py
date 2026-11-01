"""RIO Streamlit demonstration UI for real-time phishing detection and conversation analysis."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack

from ml.engine import RIOEngine
from ml.explain import (
    explain_intent_prediction,
    explain_phishing_prediction,
    explain_transition_shift,
)
from ml.intent.train import extract_requested_action
from ml.phishing.extractor import extract_email_features, to_model_metadata_dataframe
from ml.threads import (
    analyze_thread_trajectory,
    evaluate_missing_context_on_thread,
    load_conversation_threads,
)
from ml.transition.train import _cosine_similarity, _features, rule_predict

ROOT = Path(__file__).resolve().parent


def _required(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required model artifact was not found: {path}")
    return path


@st.cache_resource
def load_artifacts() -> dict[str, Any]:
    return {
        "phishing": joblib.load(_required(ROOT / "models/phishing/logistic_regression.joblib")),
        "intent": joblib.load(_required(ROOT / "models/intent/tfidf_logistic_regression.joblib")),
        "transition": joblib.load(_required(ROOT / "models/transition/logistic_regression.joblib")),
        "fallback": {
            condition: joblib.load(_required(ROOT / f"models/fallback/{condition}.joblib"))
            for condition in ("full_history", "partial_history", "no_previous_messages")
        },
    }


@st.cache_resource
def get_rio_engine() -> RIOEngine:
    return RIOEngine(models_dir=ROOT / "models")


@st.cache_data(show_spinner=False)
def get_cached_threads(csv_path: str, max_threads: int = 40):
    try:
        return load_conversation_threads(csv_path, max_threads=max_threads, min_turns=2)
    except Exception as err:
        return {}


def predict_phishing(
    artifacts: dict[str, Any],
    subject: str,
    body: str,
    sender_domain: str,
    url_count: int,
    uses_https: bool,
    has_subdomain: bool,
    authentication_result: str,
    has_attachment: bool,
    attachment_type: str,
    urgency_count: int,
) -> tuple[str, float]:
    package = artifacts["phishing"]
    metadata = pd.DataFrame([{
        "url_count": url_count,
        "domain_uses_https": int(uses_https),
        "domain_has_subdomain": int(has_subdomain),
        "has_attachment": int(has_attachment),
        "urgency_indicator_count": urgency_count,
        "authentication_result": authentication_result,
        "attachment_type": attachment_type if has_attachment else "none",
    }])
    text_features = package["vectorizer"].transform([f"{subject} {body}"])
    meta_features = package["preprocessor"].transform(metadata)
    features = hstack([text_features, meta_features]).tocsr()
    model = package["model"]
    prediction = int(model.predict(features)[0])
    confidence = float(np.max(model.predict_proba(features)[0]))
    return ("phishing" if prediction else "legitimate", confidence)


def predict_intent(artifacts: dict[str, Any], text: str) -> tuple[str, float, str]:
    package = artifacts["intent"]
    features = package["vectorizer"].transform([text])
    model = package["model"]
    probabilities = model.predict_proba(features)[0]
    index = int(np.argmax(probabilities))
    return str(model.classes_[index]), float(probabilities[index]), extract_requested_action(text)


def transition_record(previous: str, current: str, previous_intent: str, current_intent: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "previous_email": previous,
        "current_email": current,
        "previous_intent": previous_intent,
        "current_intent": current_intent,
        "previous_requested_action": extract_requested_action(previous),
        "current_requested_action": extract_requested_action(current),
    }])


def predict_transition(artifacts: dict[str, Any], record: pd.DataFrame) -> tuple[str, float, str]:
    features, _ = _features(record)
    package = artifacts["transition"]
    scaled = package["scaler"].transform(features)
    model = package["model"]
    probabilities = model.predict_proba(scaled)[0]
    index = int(np.argmax(probabilities))
    rule = str(rule_predict(record)[0])
    return str(model.classes_[index]), float(probabilities[index]), rule


def predict_fallback(
    artifacts: dict[str, Any],
    previous: str,
    current: str,
    condition: str,
) -> tuple[str, float]:
    package = artifacts["fallback"][condition]
    if condition == "full_history":
        text = f"{previous} [CURRENT] {current}"
    elif condition == "partial_history":
        text = f"{previous[-80:]} [CURRENT] {current}"
    else:
        text = current
    features = package["vectorizer"].transform([text])
    model = package["model"]
    probabilities = model.predict_proba(features)[0]
    index = int(np.argmax(probabilities))
    return str(model.classes_[index]), float(probabilities[index])


def render_explainability_box(explanation: dict[str, Any]) -> None:
    """Render structured XAI visual cards in Streamlit."""
    with st.expander("🔍 Explainable AI (XAI) Prediction Breakdown", expanded=True):
        st.markdown(f"**Diagnosis:** {explanation.get('summary', '')}")

        risk_flags = explanation.get("risk_flags", [])
        if risk_flags:
            st.markdown("##### ⚠️ Risk Indicator Flags:")
            for flag in risk_flags:
                st.markdown(f"- 🔴 {flag}")

        features = explanation.get("top_contributing_features", [])
        if features:
            st.markdown("##### 📊 Top Feature Attribution Factors:")
            f_df = pd.DataFrame(features)
            st.dataframe(f_df, hide_index=True, use_container_width=True)

        terms = explanation.get("suspicious_terms_found", [])
        if terms:
            st.markdown(f"**High-Risk Terms Flagged:** `{', '.join(terms)}`")


def main() -> None:
    st.set_page_config(page_title="RIO Phishing & Intent Analysis", page_icon="🛡️", layout="wide")
    st.title("🛡️ RIO — AI-Based Real-Time Phishing & Intent-Shift Detection")
    st.caption("Multi-feature email classification, intent extraction, conversational intent-shift detection, and explainable AI.")

    try:
        artifacts = load_artifacts()
        engine = get_rio_engine()
    except FileNotFoundError as error:
        st.error(str(error))
        st.info("Run the training commands from README.md before starting the UI.")
        st.stop()

    tab_phishing, tab_intent, tab_transition, tab_fallback, tab_threads = st.tabs(
        [
            "Phishing Detection",
            "Intent Extraction",
            "Intent Shift",
            "Missing Context",
            "Conversation Threads",
        ]
    )

    # ---------------- TAB 1: PHISHING DETECTION ----------------
    with tab_phishing:
        st.subheader("Multi-Feature Phishing Email Detection")
        mode = st.radio(
            "Input Mode:",
            ["Automatic Feature Extraction (Raw Email)", "Manual Parameter Customization"],
            horizontal=True,
        )

        if mode == "Automatic Feature Extraction (Raw Email)":
            st.info("Paste any email text. RIO automatically parses URLs, domains, urgency, authentication, and attachments in real time.")
            col_s, col_from = st.columns([2, 1])
            with col_s:
                raw_subject = st.text_input("Email Subject", "Urgent: Review required for your account immediately")
            with col_from:
                raw_sender = st.text_input("Sender Header", "Security Alerts <support@account-verification.example.com>")

            raw_body = st.text_area(
                "Email Body",
                "Dear customer,\n\nPlease sign in to http://account-verification.example.com/portal/login to confirm your recent account activity.\nFailure to respond within 24 hours will result in account suspension.\n\nThank you,\nAccount Security Team",
                height=150,
            )
            raw_attachment = st.text_input("Attachment Filename (optional)", "", placeholder="e.g. invoice.pdf")

            if st.button("🚀 Analyze Email in Real-Time", type="primary"):
                result = engine.analyze_raw_email(
                    subject=raw_subject,
                    body=raw_body,
                    sender=raw_sender,
                    attachment_filename=raw_attachment or None,
                )

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    is_phish = result["phishing"]["prediction"] == "phishing"
                    badge_color = "red" if is_phish else "green"
                    st.metric(
                        "Prediction",
                        result["phishing"]["prediction"].upper(),
                        f"{result['phishing']['confidence']:.1%} confidence",
                    )
                with c2:
                    st.metric("Detected Intent", result["intent"]["intent"])
                with c3:
                    st.metric("Inference Latency", f"{result['execution_time_ms']} ms")
                with c4:
                    st.metric("Extracted URLs", result["extracted_features"]["url_count"])

                # Extracted feature overview pill
                ext = result["extracted_features"]
                with st.expander("Extracted Multi-Modal Features", expanded=False):
                    st.json({
                        "sender_domain": ext["sender_domain"],
                        "url_count": ext["url_count"],
                        "urls_found": ext["urls"],
                        "domain_uses_https": bool(ext["domain_uses_https"]),
                        "domain_has_subdomain": bool(ext["domain_has_subdomain"]),
                        "urgency_indicator_count": ext["urgency_indicator_count"],
                        "has_attachment": bool(ext["has_attachment"]),
                        "attachment_type": ext["attachment_type"],
                    })

                # Explainability Card
                render_explainability_box(result["phishing"]["explanation"])

        else:
            # Manual parameter mode (preserves original tab behaviour)
            st.subheader("Manual Feature Parameters")
            subject = st.text_input("Subject", "Review required for your account", key="m_sub")
            body = st.text_area("Email body", "Please sign in to confirm the recent account activity.", height=120, key="m_body")
            sender_domain = st.text_input("Sender domain", "example.com", key="m_dom")
            col1, col2, col3 = st.columns(3)
            with col1:
                url_count = st.number_input("URL count", min_value=0, max_value=20, value=1, key="m_url")
                uses_https = st.checkbox("URLs use HTTPS", value=True, key="m_https")
            with col2:
                has_subdomain = st.checkbox("Domain has subdomain", value=False, key="m_subdom")
                has_attachment = st.checkbox("Has attachment", value=False, key="m_att")
            with col3:
                authentication = st.selectbox("Authentication result", ["pass", "neutral", "softfail", "none"], key="m_auth")
                attachment_type = st.selectbox("Attachment type", ["none", "pdf", "docx", "xlsx"], key="m_att_type")
            urgency = st.number_input("Urgency indicator count", min_value=0, max_value=10, value=0, key="m_urg")

            if st.button("Detect Phishing (Manual Parameters)", type="primary"):
                label, confidence = predict_phishing(
                    artifacts, subject, body, sender_domain, int(url_count), uses_https,
                    has_subdomain, authentication, has_attachment, attachment_type, int(urgency),
                )
                st.metric("Prediction", label.upper(), f"{confidence:.1%} confidence")

                # Generate explanation for manual parameters
                metadata_df = pd.DataFrame([{
                    "url_count": int(url_count),
                    "domain_uses_https": int(uses_https),
                    "domain_has_subdomain": int(has_subdomain),
                    "has_attachment": int(has_attachment),
                    "urgency_indicator_count": int(urgency),
                    "authentication_result": authentication,
                    "attachment_type": attachment_type if has_attachment else "none",
                }])
                exp = explain_phishing_prediction(
                    artifacts["phishing"], subject, body, metadata_df, label, confidence,
                )
                render_explainability_box(exp)

    # ---------------- TAB 2: INTENT EXTRACTION ----------------
    with tab_intent:
        st.subheader("Extract Email Intent and Action")
        text = st.text_area(
            "Email text",
            "Please review the project brief and leave comments by Friday.",
            key="intent_text",
            height=130,
        )
        if st.button("Extract Intent", type="primary"):
            t_start = time.perf_counter()
            label, confidence, action = predict_intent(artifacts, text)
            t_ms = (time.perf_counter() - t_start) * 1000.0

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Intent Class", label, f"{confidence:.1%} confidence")
            with c2:
                st.metric("Requested Action", action or "None detected")
            with c3:
                st.metric("Inference Time", f"{t_ms:.2f} ms")

            # Intent explanation
            exp = explain_intent_prediction(
                artifacts["intent"], text, label, confidence, action,
            )
            with st.expander("🔍 Intent Classification Explanation", expanded=True):
                st.write(f"**Rationale:** {exp['rationale']}")
                if exp["key_intent_cues"]:
                    st.write(f"**Key intent cue tokens:** `{', '.join(exp['key_intent_cues'])}`")

    # ---------------- TAB 3: INTENT SHIFT ----------------
    with tab_transition:
        st.subheader("Detect Suspicious Intent-Shift Across Conversational Turns")
        previous = st.text_area("Previous Email (Turn t-1)", "Please review the project brief.", height=100)
        current = st.text_area("Current Email (Turn t)", "Also, urgently submit your employee credentials before the review can continue.", height=100)

        if st.button("Detect Intent Shift", type="primary"):
            res = engine.analyze_transition(previous_email=previous, current_email=current)

            c1, c2, c3 = st.columns(3)
            with c1:
                badge = res["transition_label"].upper()
                st.metric("Transition Status", badge, f"{res['confidence']:.1%} confidence")
            with c2:
                st.metric("Previous → Current Intent", f"{res['previous_intent']} → {res['current_intent']}")
            with c3:
                st.metric("Text Continuity Similarity", f"{res['similarity_score']:.1%}")

            with st.expander("🔍 Intent-Shift Explanation & Security Assessment", expanded=True):
                st.markdown(f"**Security Assessment:** {res['explanation']['assessment']}")
                st.markdown("##### Assessment Factors:")
                for item in res["explanation"]["details"]:
                    st.markdown(f"- {item}")
                st.caption(f"Rule baseline consensus: **{res['rule_consensus']}** | Execution time: {res['execution_time_ms']} ms")

    # ---------------- TAB 4: MISSING CONTEXT ----------------
    with tab_fallback:
        st.subheader("Test Detection Robustness with Incomplete / Missing Conversation History")
        previous_fallback = st.text_area("Available Previous History", "Please review the project brief.", key="fb_prev", height=90)
        current_fallback = st.text_area("Current Email Message", "Also, urgently submit your employee credentials before the review can continue.", key="fb_curr", height=90)

        if st.button("Compare Context Conditions", type="primary"):
            rows = []
            for condition in ("full_history", "partial_history", "no_previous_messages"):
                label, confidence = predict_fallback(artifacts, previous_fallback, current_fallback, condition)
                rows.append({
                    "Context Condition": condition.replace("_", " ").title(),
                    "Predicted Transition": label,
                    "Confidence": f"{confidence:.1%}",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("Note: History availability itself is not supplied as a feature to prevent artificial bias.")

    # ---------------- TAB 5: CONVERSATION THREADS DATASET EXPLORER ----------------
    with tab_threads:
        st.subheader("Real-World Conversation Thread Explorer")
        st.markdown(
            "Explore authentic multi-turn email conversation threads (from `email_thread_details.csv`). "
            "*Note: This dataset represents real enterprise conversations without pre-assigned ground truth labels. "
            "RIO reconstructs threads chronologically and dynamically evaluates intent trajectory and transition shifts.*"
        )

        csv_path = ROOT / "data/raw/email_thread_details.csv"
        if not csv_path.exists():
            # Check Downloads directory fallback
            dl_path = Path("C:/Users/LENOVO/Downloads/email_thread_details.csv")
            if dl_path.exists():
                csv_path = dl_path

        if not csv_path.exists():
            st.warning(f"Thread dataset not found at `{csv_path}`. Place `email_thread_details.csv` in `data/raw/` to enable this explorer.")
        else:
            with st.spinner("Loading conversation threads from dataset..."):
                threads = get_cached_threads(str(csv_path), max_threads=50)

            if not threads:
                st.error("No multi-turn threads could be loaded from the dataset.")
            else:
                thread_ids = list(threads.keys())
                col_sel, col_info = st.columns([1, 2])
                with col_sel:
                    selected_id = st.selectbox("Select Conversation Thread:", thread_ids, format_func=lambda tid: f"Thread #{tid} ({threads[tid].turn_count} turns)")
                
                selected_thread = threads[selected_id]
                with col_info:
                    st.write(f"**Subject:** {selected_thread.subject}")
                    st.write(f"**Total Turns:** {selected_thread.turn_count}")

                st.markdown("#### Chronological Conversation Turns & Dynamic Intent Trajectory")
                trajectory = analyze_thread_trajectory(selected_thread, artifacts)

                for turn in trajectory:
                    turn_idx = turn["turn_index"]
                    msg = selected_thread.messages[turn_idx]
                    trans = turn.get("transition")

                    with st.container():
                        st.markdown(f"---")
                        c_header, c_meta = st.columns([3, 1])
                        with c_header:
                            st.markdown(f"**Turn {turn_idx + 1}** — From: `{msg.sender}` | Date: `{msg.timestamp}`")
                        with c_meta:
                            st.markdown(f"**Intent:** `{turn['intent']}` ({turn['intent_confidence']:.1%})")

                        st.info(turn["snippet"])

                        if trans:
                            t_label = trans["label"]
                            color = "orange" if t_label == "suspicious intent shift" else "green"
                            st.markdown(f"↳ **Transition from previous turn:** :{color}[**{t_label.upper()}**] (Confidence: {trans['confidence']:.1%}, Continuity similarity: {trans['similarity_score']:.1%})")
                            with st.expander(f"View Turn {turn_idx + 1} Transition Assessment", expanded=False):
                                st.write(trans["explanation"]["assessment"])
                                for d in trans["explanation"]["details"]:
                                    st.write(f"- {d}")

                st.markdown("#### Stress-Test Missing History on this Thread")
                if st.button("Evaluate Target Turn Under Missing Context", key="btn_thread_fb"):
                    fb_eval = evaluate_missing_context_on_thread(selected_thread, artifacts)
                    st.write(f"Evaluating latest turn by `{fb_eval['current_sender']}`:")
                    fb_rows = [
                        {
                            "Condition": cond.replace("_", " ").title(),
                            "Predicted State": fb_eval["evaluations"][cond]["prediction"],
                            "Confidence": f"{fb_eval['evaluations'][cond]['confidence']:.1%}",
                        }
                        for cond in ("full_history", "partial_history", "no_previous_messages")
                    ]
                    st.dataframe(pd.DataFrame(fb_rows), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
