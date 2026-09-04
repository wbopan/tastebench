"""Credential redaction applied before any hash, token count, request, or artifact."""

from __future__ import annotations

import re

SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)


def redact_sensitive_text(text: str) -> str:
    """Redact common credential shapes before text enters prompts or artifacts."""
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED_SECRET]", text)
    return text
