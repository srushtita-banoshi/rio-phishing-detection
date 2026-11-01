"""Feature extraction utilities for raw email text and headers."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import pandas as pd

URL_PATTERN = re.compile(
    r"(?:https?://|www\.)[^\s<>'\"`]+",
    re.IGNORECASE,
)

EMAIL_ADDRESS_PATTERN = re.compile(
    r"[a-zA-Z0-9_.+-]+@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)",
)

URGENCY_PATTERN = re.compile(
    r"\b(urgent|urgently|immediate|immediately|asap|action required|verify immediately|"
    r"account suspended|limited time|unauthorized|critical|expire|within \d+ (?:hours?|days?)|"
    r"failure to respond|suspend your access)\b",
    re.IGNORECASE,
)

COMMON_ATTACHMENT_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".xlsx": "xlsx",
    ".xls": "xlsx",
    ".exe": "exe",
    ".zip": "zip",
}


def extract_sender_domain(sender_str: str) -> str:
    """Extract domain name from sender string, e.g., 'Alice <alice@example.com>' -> 'example.com'."""
    if not sender_str:
        return "unknown"
    match = EMAIL_ADDRESS_PATTERN.search(sender_str)
    if match:
        return match.group(1).lower()
    # If just a domain or plain string
    if "@" in sender_str:
        return sender_str.split("@")[-1].strip().lower()
    return sender_str.strip().lower()


def extract_urls(text: str) -> list[str]:
    """Find all URLs in subject and body text."""
    if not text:
        return []
    return [match.group(0).rstrip(".,;:)") for match in URL_PATTERN.finditer(text)]


def check_domain_characteristics(urls: list[str], sender_domain: str) -> tuple[int, int]:
    """Return (uses_https, has_subdomain)."""
    if not urls:
        # Default behavior when no URLs: HTTPS=1 (no unencrypted link risk), subdomain=0
        return 1, 0

    all_https = True
    has_subdomain = False

    for raw_url in urls:
        url = raw_url if raw_url.startswith(("http://", "https://")) else f"http://{raw_url}"
        parsed = urlparse(url)
        if parsed.scheme.lower() != "https":
            all_https = False
        hostname = parsed.hostname or ""
        parts = hostname.split(".")
        if len(parts) > 2:
            has_subdomain = True

    # Also check sender domain for subdomain
    sender_parts = sender_domain.split(".")
    if len(sender_parts) > 2:
        has_subdomain = True

    return int(all_https), int(has_subdomain)


def extract_urgency_indicators(text: str) -> int:
    """Count urgency indicators in text."""
    if not text:
        return 0
    return len(URGENCY_PATTERN.findall(text))


def detect_attachment_info(text: str, attachment_filename: str | None = None) -> tuple[int, str]:
    """Detect if an attachment exists and its type."""
    if attachment_filename:
        ext = "." + attachment_filename.split(".")[-1].lower() if "." in attachment_filename else ""
        att_type = COMMON_ATTACHMENT_EXTENSIONS.get(ext, "none")
        return 1, att_type

    # Check for text hints like "attached invoice.pdf"
    for ext, att_type in COMMON_ATTACHMENT_EXTENSIONS.items():
        if re.search(rf"\w+{re.escape(ext)}\b", text, re.IGNORECASE):
            return 1, att_type

    return 0, "none"


def detect_authentication_result(headers: dict[str, str] | None = None) -> str:
    """Parse authentication headers (SPF, DKIM, DMARC) if available."""
    if not headers:
        return "none"

    auth_str = (
        headers.get("Authentication-Results", "")
        + " "
        + headers.get("Received-SPF", "")
    ).lower()

    if "pass" in auth_str:
        return "pass"
    if "softfail" in auth_str:
        return "softfail"
    if "fail" in auth_str:
        return "none"
    if "neutral" in auth_str:
        return "neutral"
    return "none"


def extract_email_features(
    subject: str,
    body: str,
    sender: str = "",
    headers: dict[str, str] | None = None,
    attachment_filename: str | None = None,
) -> dict[str, Any]:
    """Extract full feature set required by RIO models from raw email content."""
    combined_text = f"{subject} {body}"
    sender_domain = extract_sender_domain(sender)
    urls = extract_urls(combined_text)
    url_count = len(urls)
    uses_https, has_subdomain = check_domain_characteristics(urls, sender_domain)
    urgency_count = extract_urgency_indicators(combined_text)
    has_attachment, attachment_type = detect_attachment_info(combined_text, attachment_filename)
    auth_result = detect_authentication_result(headers)

    return {
        "subject": subject,
        "body": body,
        "combined_text": combined_text.strip(),
        "sender_domain": sender_domain,
        "urls": urls,
        "url_count": url_count,
        "domain_uses_https": uses_https,
        "domain_has_subdomain": has_subdomain,
        "urgency_indicator_count": urgency_count,
        "has_attachment": has_attachment,
        "attachment_type": attachment_type,
        "authentication_result": auth_result,
    }


def to_model_metadata_dataframe(extracted: dict[str, Any]) -> pd.DataFrame:
    """Convert extracted features dictionary into the exact DataFrame required by the preprocessor."""
    return pd.DataFrame([
        {
            "url_count": extracted["url_count"],
            "domain_uses_https": extracted["domain_uses_https"],
            "domain_has_subdomain": extracted["domain_has_subdomain"],
            "has_attachment": extracted["has_attachment"],
            "urgency_indicator_count": extracted["urgency_indicator_count"],
            "authentication_result": extracted["authentication_result"],
            "attachment_type": extracted["attachment_type"],
        }
    ])
