"""Generate Dataset A: balanced synthetic phishing/legitimate emails."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import DEFAULT_SEED, choose, json_list, make_rng, write_metadata

LEGITIMATE_SCENARIOS = [
    ("project_update", "The design review is ready for your comments.", "Please review the attached plan and share feedback by Thursday."),
    ("operations_notice", "Planned maintenance notice", "The service window is scheduled for Saturday from 22:00 to 23:00 UTC."),
    ("team_coordination", "Agenda for tomorrow's team meeting", "Here is the agenda and the notes from our last discussion."),
    ("customer_followup", "Following up on our conversation", "Thank you for the discussion. We will send the revised estimate next week."),
    ("newsletter", "Monthly community update", "This month's update includes events, learning resources, and office news."),
]
PHISHING_SCENARIOS = [
    ("account_task", "Review required for your account", "Please sign in to confirm the recent account activity before access is limited."),
    ("invoice", "Updated invoice available", "Please review the attached invoice and use the supplier portal if payment details need attention."),
    ("shared_document", "A document was shared with you", "The document is ready for review. Open the workspace link to see the latest version."),
    ("delivery", "Delivery status update", "Your delivery could not be completed. Confirm the delivery details to arrange another attempt."),
    ("payroll", "Payroll information update", "Please check the payroll notice and confirm your details through the employee service page."),
]
DOMAINS = ["northstar.example", "lakeside.example", "brightpath.example", "meridian.example", "oakridge.example"]
SUSPICIOUS_DOMAINS = ["northstar-support.example", "secure-lakeside.example", "brightpath-alerts.example", "meridian-check.example"]
VARIATIONS = [
    "I have included the relevant context below.",
    "The team can answer questions during normal business hours.",
    "This message is part of the current workstream.",
    "Please use the usual company process if you need assistance.",
]


def _row(rng: np.random.Generator, label: int, index: int) -> dict[str, object]:
    scenarios = PHISHING_SCENARIOS if label else LEGITIMATE_SCENARIOS
    family, subject, body = choose(rng, scenarios)
    sender_domain = choose(rng, SUSPICIOUS_DOMAINS if label else DOMAINS)
    sender_prefix = choose(rng, ["team", "support", "operations", "noreply", "contact", "alex"])
    url_count = int(rng.choice([0, 0, 1, 1, 2, 3]))
    urls: list[str] = []
    for url_index in range(url_count):
        domain = sender_domain if rng.random() < 0.65 else choose(rng, DOMAINS + SUSPICIOUS_DOMAINS)
        path = choose(rng, ["portal/review", "documents/agenda", "account/status", "help/article", "files/latest"])
        urls.append(f"https://{domain}/{path}/{index}-{url_index}")
    attachment = bool(rng.random() < (0.28 if label else 0.22))
    attachment_type = choose(rng, ["pdf", "docx", "xlsx", "none"]) if attachment else "none"
    urgency = int(rng.choice([0, 0, 1, 1, 2]))
    auth_status = choose(rng, ["pass", "pass", "neutral", "softfail", "none"])
    if rng.random() < 0.35:
        body = f"{body} {choose(rng, VARIATIONS)}"
    if urgency and rng.random() < 0.5:
        body = f"{body} Please respond within {choose(rng, ['one business day', '48 hours', 'this week'])}."
    return {
        "email_id": f"syn-a-{index:04d}",
        "template_family": family,
        "scenario_id": f"{family}-{index % 10}",
        "subject": subject,
        "body": body,
        "sender_email": f"{sender_prefix}@{sender_domain}",
        "sender_domain": sender_domain,
        "urls": json_list(urls),
        "url_count": url_count,
        "domain_uses_https": int(all(url.startswith("https://") for url in urls)) if urls else 1,
        "domain_has_subdomain": int(any(domain.count(".") > 1 for domain in [sender_domain])),
        "authentication_result": auth_status,
        "has_attachment": int(attachment),
        "attachment_type": attachment_type,
        "urgency_indicator_count": urgency,
        "label": label,
        "label_name": "phishing" if label else "legitimate",
        "synthetic": True,
    }


def generate(output_dir: Path, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    rng = make_rng(seed)
    rows = [_row(rng, label, index) for label in (0, 1) for index in range(label * 1000, (label + 1) * 1000)]
    frame = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "synthetic_phishing_emails.csv"
    frame.to_csv(output, index=False)
    write_metadata(output_dir / "synthetic_phishing_emails.metadata.json", dataset_name="A_phishing_emails", row_count=len(frame), seed=seed, output_file=output.name, columns=list(frame.columns))
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)
