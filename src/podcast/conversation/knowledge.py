"""Knowledge rendering and speech sanitization utilities.

This module strictly separates internal evidence metadata from the natural-language
content that agents and TTS systems are allowed to see and speak.
"""

from __future__ import annotations

import re
from typing import Any

_FORBIDDEN_PATTERNS = (
    r"\bsource\s*[:\-]?\s*\{?",
    r"\bchunk\s*\d+",
    r"\bpage\s*\d+",
    r"\bclaim[_ -]?id\b",
    r"\bdocument[_ -]?id\b",
    r"\bretrieval[_ -]?score\b",
    r"\bvalue\s*[:\-]",
    r"\bmetadata\b",
)


def _collect_claim_texts(value: Any, *, collected: list[str] | None = None) -> list[str]:
    """Recursively collect natural-language claim text while excluding metadata keys."""
    collected = collected if collected is not None else []
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned and not cleaned.lower().startswith("source") and not cleaned.lower().startswith("chunk"):
            collected.append(cleaned)
        return collected

    if isinstance(value, list):
        for item in value:
            _collect_claim_texts(item, collected=collected)
        return collected

    if isinstance(value, dict):
        if isinstance(value.get("value"), str):
            cleaned = value["value"].strip()
            if cleaned:
                collected.append(cleaned)
            return collected

        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in {"source", "chunk", "page", "metadata", "source_metadata", "source_id", "document_id", "claim_id", "retrieval_score", "url", "title", "summary"}:
                continue
            _collect_claim_texts(item, collected=collected)
        return collected

    return collected


def render_evidence_for_agent(evidence: list[dict[str, Any]] | tuple[dict[str, Any], ...] | dict[str, Any] | None) -> str:
    """Return only claim text that is safe to expose to conversational agents."""
    if isinstance(evidence, dict):
        evidence = [evidence]
    elif evidence is None:
        evidence = []

    facts: list[str] = []
    for entry in evidence or []:
        if isinstance(entry, dict):
            for key in ("claim", "text", "value", "summary", "fact"):
                if key in entry and isinstance(entry[key], str):
                    facts.append(entry[key].strip())
            if not any(key in entry for key in ("claim", "text", "value", "summary", "fact")):
                _collect_claim_texts(entry, collected=facts)
        else:
            _collect_claim_texts(entry, collected=facts)

    unique_claims: list[str] = []
    seen: set[str] = set()
    for claim in facts:
        normalized = claim.strip()
        if normalized and normalized.lower() not in seen and "source" not in normalized.lower() and "chunk" not in normalized.lower() and "page" not in normalized.lower():
            seen.add(normalized.lower())
            unique_claims.append(normalized)

    if not unique_claims:
        return "No additional evidence is available."

    return "\n".join(f"- {claim}" for claim in unique_claims)


def scrub_speech_text(text: str) -> str:
    """Remove internal metadata artifacts from speech text instead of rejecting it.

    Used on the TTS path where we prefer degraded-but-clean speech over a crash.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned

    # Dict/JSON-ish fragments that leaked from structured material, e.g.
    # "{'value': 'Some claim', 'source': {'chunk': 1}}" -> "Some claim".
    cleaned = re.sub(
        r"\{['\"]value['\"]\s*:\s*['\"](?P<claim>[^'\"]*)['\"][^}]*\}(\s*\})*",
        lambda match: match.group("claim"),
        cleaned,
    )
    for pattern in _FORBIDDEN_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[{}\[\]]", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def sanitize_speech_text(text: str) -> str:
    """Reject speech that reveals internal metadata or schema names."""
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned

    lowered = cleaned.lower()
    for pattern in _FORBIDDEN_PATTERNS:
        if re.search(pattern, lowered):
            raise ValueError("Speech contains prohibited internal metadata patterns.")

    return cleaned
