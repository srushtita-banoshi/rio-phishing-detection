"""Generate Dataset B: synthetic email intent extraction examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import DEFAULT_SEED, choose, make_rng, write_metadata

INTENTS = {
    "document_review": ("review the project brief", "the project brief", ["Could you review the project brief?", "Please check the updated brief and leave comments."]),
    "meeting_scheduling": ("schedule a planning meeting", "the planning session", ["Can we schedule a planning meeting next week?", "Please find a time for the planning session."]),
    "information_sharing": ("share the status update", "the delivery team", ["I am sharing the latest status update.", "Here is the information from the delivery team."]),
    "credential_submission": ("submit your employee credentials", "the employee portal", ["Please submit your employee credentials through the portal.", "The employee portal needs your credentials for the request."]),
    "financial_transfer": ("transfer the approved funds", "the project account", ["Please transfer the approved funds to the project account.", "The finance team requested a transfer to the project account."]),
    "account_verification": ("verify the account details", "your customer account", ["Could you verify the account details?", "Please confirm the details for your customer account."]),
    "file_download": ("download the project file", "the shared workspace", ["Please download the project file from the workspace.", "The shared workspace contains a file for you to download."]),
    "payment_confirmation": ("confirm the payment", "the invoice record", ["Please confirm that the payment was received.", "Could you update the invoice record with payment confirmation?"]),
    "access_permission": ("grant access to the folder", "the research folder", ["Please grant access to the research folder.", "Could you approve my access to the folder?"]),
    "other": ("review the announcement", "the team", ["Please read the announcement when you have time.", "I wanted to share a general note with the team."]),
}


def generate(output_dir: Path, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    rng = make_rng(seed)
    rows: list[dict[str, object]] = []
    for index in range(1000):
        intent = list(INTENTS)[index % len(INTENTS)]
        action, target, variants = INTENTS[intent]
        text = choose(rng, variants)
        if rng.random() < 0.45:
            text = f"Hello team, {text}"
        text = f"{text} Reference RIO-{index:04d} is associated with this request."
        rows.append({
            "email_id": f"syn-b-{index:04d}",
            "template_family": f"intent-{intent}",
            "scenario_id": f"{intent}-{index // len(INTENTS):03d}",
            "email_text": text,
            "intent_label": intent,
            "requested_action": action,
            "action_target": target,
            "synthetic": True,
        })
    frame = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "synthetic_intent_emails.csv"
    frame.to_csv(output, index=False)
    write_metadata(output_dir / "synthetic_intent_emails.metadata.json", dataset_name="B_intent_extraction", row_count=len(frame), seed=seed, output_file=output.name, columns=list(frame.columns))
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)
