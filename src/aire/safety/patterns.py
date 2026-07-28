"""Detection patterns: PII, prompt injection, and secrets.

These are conservative heuristics — defense in depth, not a guarantee. They
power guardrails, dataset quality reports and redaction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "bearer_token": re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{20,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "generic_secret": re.compile(
        r"(?i)(?:api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?[A-Za-z0-9._~+/=-]{16,}"
    ),
}

INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)ignore (?:all |any |the )?(?:previous|prior|above) (?:instructions|prompts)"),
    re.compile(r"(?i)disregard (?:your|all|the) (?:instructions|rules|guidelines)"),
    re.compile(r"(?i)you are now (?:a|an) (?:new |evil |unfiltered )"),
    re.compile(
        r"(?i)\b(?:system prompt|developer mode|jailbreak)\b.{0,40}(?:override|ignore|bypass)"
    ),
    re.compile(r"(?i)reveal (?:your|the) (?:system |initial )?(?:prompt|instructions)"),
    re.compile(r"(?i)forget (?:everything|all) (?:you (?:know|were told)|above)"),
]


@dataclass(frozen=True)
class Match:
    kind: str
    span: tuple[int, int]
    text: str


def detect_pii(text: str) -> list[Match]:
    """Find suspected PII occurrences."""
    matches: list[Match] = []
    for kind, pattern in {
        "email": EMAIL_RE,
        "phone": PHONE_RE,
        "ssn": SSN_RE,
        "credit_card": CREDIT_CARD_RE,
    }.items():
        for m in pattern.finditer(text):
            matches.append(Match(kind=kind, span=(m.start(), m.end()), text=m.group()))
    return matches


def detect_injection(text: str) -> list[Match]:
    """Find prompt-injection attempts by heuristic phrase matching."""
    matches: list[Match] = []
    for pattern in INJECTION_PATTERNS:
        for m in pattern.finditer(text):
            matches.append(
                Match(kind="prompt_injection", span=(m.start(), m.end()), text=m.group())
            )
    return matches


def detect_secrets(text: str) -> list[Match]:
    """Find likely leaked credentials."""
    matches: list[Match] = []
    for kind, pattern in SECRET_PATTERNS.items():
        for m in pattern.finditer(text):
            matches.append(Match(kind=kind, span=(m.start(), m.end()), text=m.group()))
    return matches
