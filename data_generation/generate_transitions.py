"""Generate Dataset C: linked synthetic email intent transitions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import DEFAULT_SEED, choose, make_rng, write_metadata

INTENT_TEXT = {
    "document_review": ("Please review the project brief.", "review the project brief"),
    "meeting_scheduling": ("Could we schedule a planning meeting?", "schedule a planning meeting"),
    "information_sharing": ("Here is the latest project status.", "share the project status"),
    "credential_submission": ("Please submit your credentials through the portal.", "submit credentials"),
    "financial_transfer": ("Please transfer the approved funds.", "transfer approved funds"),
    "account_verification": ("Please verify the account details.", "verify account details"),
    "file_download": ("Please download the project file.", "download the project file"),
    "payment_confirmation": ("Please confirm the invoice payment.", "confirm the payment"),
    "access_permission": ("Please grant access to the research folder.", "grant folder access"),
}
INTENTS = list(INTENT_TEXT)


def _pair(
    rng,
    index: int,
    label: str,
    previous_intent: str,
    current_intent: str,
    subtype: str,
) -> dict[str, object]:
    previous_text, previous_action = INTENT_TEXT[previous_intent]
    current_text, current_action = INTENT_TEXT[current_intent]
    reference = f"Related work item RIO-{index:04d}."
    previous_text = f"{previous_text} {reference}"
    current_text = f"{current_text} {reference}"
    evidence = {
        "normal continuation": "Same thread topic and requested action remain aligned.",
        "legitimate intent change": "The topic changes with an explicit business context and no access or payment pressure.",
        "suspicious intent shift": "The request changes toward sensitive action without a clear thread rationale.",
    }[label]
    if label == "suspicious intent shift":
        current_text = f"Also, urgently {current_action} before the review can continue. {reference}"
    return {
        "thread_id": f"thread-{index:04d}",
        "template_family": f"transition-{subtype.replace(' ', '_')}",
        "scenario_id": f"{subtype.replace(' ', '_')}-{index // 20:03d}",
        "previous_email_id": f"syn-c-prev-{index:04d}",
        "current_email_id": f"syn-c-current-{index:04d}",
        "previous_email": previous_text,
        "current_email": current_text,
        "previous_intent": previous_intent,
        "current_intent": current_intent,
        "previous_requested_action": previous_action,
        "current_requested_action": current_action,
        "transition_label": label,
        "transition_subtype": subtype,
        "supporting_evidence": evidence,
        "synthetic": True,
    }


def generate(output_dir: Path, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    rng = make_rng(seed)
    specs = [
        ("normal continuation", "normal continuation", 350),
        ("legitimate intent change", "legitimate intent change", 250),
        ("suspicious intent shift", "suspicious intent shift", 250),
        ("suspicious intent shift", "clearly suspicious transition", 150),
    ]
    rows: list[dict[str, object]] = []
    for label, subtype, count in specs:
        for offset in range(count):
            previous = choose(rng, INTENTS)
            current = previous if label == "normal continuation" else choose(rng, [intent for intent in INTENTS if intent != previous])
            rows.append(_pair(rng, len(rows), label, previous, current, subtype))
    frame = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "synthetic_intent_transitions.csv"
    frame.to_csv(output, index=False)
    write_metadata(output_dir / "synthetic_intent_transitions.metadata.json", dataset_name="C_intent_transitions", row_count=len(frame), seed=seed, output_file=output.name, columns=list(frame.columns))
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)
