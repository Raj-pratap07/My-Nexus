"""
Quote-fidelity helpers shared by the offline graders.

Pure functions, no I/O and no app imports, so any judge module can use them
without import cycles. Moved verbatim from ``github_prompt_eval`` (which
re-exports every name, so existing imports keep working).
"""
from __future__ import annotations

import re
from typing import Sequence

QUOTE_RE = re.compile(r'"([^"\n]+)"|“([^”\n]+)”')
MIN_QUOTE_CHARS = 12
MIN_SEGMENT_CHARS = 8


def _normalize(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("**", "").replace("`", "")
    return " ".join(text.split())


def extract_quotes(answer: str) -> list[str]:
    """Quoted spans long enough to be meaningful source quotations."""
    quotes = []
    for m in QUOTE_RE.finditer(answer):
        q = _normalize(m.group(1) or m.group(2)).strip().rstrip(".,;:").strip()
        if len(q) >= MIN_QUOTE_CHARS:
            quotes.append(q)
    return quotes


def unmatched_quotes(quotes: Sequence[str], sources: Sequence[str]) -> list[str]:
    """Quotes (or "..."-separated segments) not found verbatim in any source."""
    corpus = _normalize("\n".join(sources))
    bad = []
    for q in quotes:
        segments = [s.strip(" .,;:") for s in re.split(r"\.\.\.|…", q)]
        if not all(s in corpus for s in segments if len(s) >= MIN_SEGMENT_CHARS):
            bad.append(q)
    return bad
