"""Credential detection/redaction and constant-time token comparison."""
from __future__ import annotations

import hmac
import re
from typing import Iterable

# Well-known credential shapes. Conservative on purpose: a false positive
# only asks the user to rephrase; a false negative stores a secret.
_PATTERNS = [
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(
        r"(?i)\b(?:password|passwd|secret|api[_-]?key|token)\b\s*[:=]\s*\S{8,}"
    ),
]
REDACTED = "[REDACTED]"


def contains_credential(text: str, known_secrets: Iterable[str] = ()) -> bool:
    if any(s and s in text for s in known_secrets):
        return True
    return any(p.search(text) for p in _PATTERNS)


def redact(text: str, known_secrets: Iterable[str] = ()) -> str:
    """Replace exact known secrets and credential-shaped strings."""
    for secret in known_secrets:
        if secret and len(secret) >= 8:
            text = text.replace(secret, REDACTED)
    for pattern in _PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def tokens_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
