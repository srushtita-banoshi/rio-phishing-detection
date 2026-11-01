from ml.phishing.extractor import (
    extract_email_features,
    extract_sender_domain,
    extract_urgency_indicators,
    extract_urls,
    to_model_metadata_dataframe,
)


def test_extract_sender_domain() -> None:
    assert extract_sender_domain("Support <support@paypal-security.com>") == "paypal-security.com"
    assert extract_sender_domain("john@company.org") == "company.org"
    assert extract_sender_domain("") == "unknown"


def test_extract_urls() -> None:
    text = "Visit https://secure-login.bank.com/auth and http://insecure.site.org now."
    urls = extract_urls(text)
    assert len(urls) == 2
    assert "https://secure-login.bank.com/auth" in urls
    assert "http://insecure.site.org" in urls


def test_extract_urgency_indicators() -> None:
    text = "This is urgent! Immediate action required within 24 hours."
    assert extract_urgency_indicators(text) >= 2


def test_to_model_metadata_dataframe() -> None:
    extracted = extract_email_features(
        subject="Urgent Security Alert",
        body="Please check attached invoice.pdf and click https://portal.example.com immediately.",
        sender="Security Team <security@corp.example.com>",
    )
    df = to_model_metadata_dataframe(extracted)
    assert len(df) == 1
    assert df["url_count"].iloc[0] == 1
    assert df["urgency_indicator_count"].iloc[0] >= 1
    assert df["has_attachment"].iloc[0] == 1
    assert df["attachment_type"].iloc[0] == "pdf"
