"""Redaction: scrub secrets and PII from text before logging or export."""

from __future__ import annotations

from aire.safety.patterns import (
    CREDIT_CARD_RE,
    EMAIL_RE,
    PHONE_RE,
    SECRET_PATTERNS,
    SSN_RE,
)


def redact_secrets(text: str, *, replacement: str = "[REDACTED]") -> str:
    """Replace detected credentials with a placeholder."""
    result = text
    for pattern in SECRET_PATTERNS.values():
        result = pattern.sub(replacement, result)
    return result


def redact_pii(text: str, *, replacement: str = "[PII]") -> str:
    """Replace detected PII with a placeholder."""
    result = text
    for pattern in (EMAIL_RE, PHONE_RE, SSN_RE, CREDIT_CARD_RE):
        result = pattern.sub(replacement, result)
    return result


def redact(text: str, *, secrets: bool = True, pii: bool = False) -> str:
    """Combined redaction helper."""
    if secrets:
        text = redact_secrets(text)
    if pii:
        text = redact_pii(text)
    return text
