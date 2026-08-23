"""Dialogue generation for the podcast script."""

from __future__ import annotations

import json
import re
from typing import Any


class PodcastDialogue:
    """Generate a structured dialogue JSON object from a plan using the local LLM."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def _extract_json(self, raw_response: str) -> dict[str, Any]:
        """Extract the first valid JSON object from raw chat output containing reasoning blocks."""
        text = raw_response.strip()
        if not text:
            raise ValueError("Dialogue model returned empty output.")

        cleaned = re.sub(r"(?is)<think>.*?</think>", " ", text)
        cleaned = re.sub(r"(?is)```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"(?is)\s*```\s*$", "", cleaned)

        candidates: list[str] = []
        for start in range(len(cleaned)):
            if cleaned[start] != "{":
                continue
            depth = 0
            for end in range(start, len(cleaned)):
                if cleaned[end] == "{":
                    depth += 1
                elif cleaned[end] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[start : end + 1]
                        candidates.append(candidate)
                        break

        if not candidates:
            raise ValueError(f"Could not parse dialogue JSON: {text[:200]}")

        valid_payloads: list[dict[str, Any]] = []
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                valid_payloads.append(payload)

        if not valid_payloads:
            raise ValueError(f"Malformed dialogue JSON: {candidates[0][:300]}")

        preferred = max(valid_payloads, key=lambda item: ("dialogue" in item, len(str(item))))
        if "dialogue" not in preferred or not isinstance(preferred["dialogue"], list):
            raise ValueError("Model response did not include a dialogue list.")
        return preferred

    def build(self, plan: Any) -> dict[str, Any]:
        """Ask the LLM to produce a realistic two-speaker podcast conversation."""
        if self.llm is None or not getattr(self.llm, "available", False):
            raise RuntimeError("Qwen model unavailable.")

        title = "Untitled Podcast"
        material: dict[str, Any] = {}
        if isinstance(plan, dict):
            title = str(plan.get("title", title))
            material = plan.get("material", {})

        prompt = f"""
You are producing a podcast script from source material.

Title: {title}

Material:
{json.dumps(material, ensure_ascii=False, indent=2)}

Create a realistic two-person conversation with HOST and EXPERT.

Rules:
- Every factual claim must be supported by the supplied material.
- Do not invent data.
- Keep it natural and engaging.
- Use a conversation structure with a hook, explanation, key facts, examples, and a final takeaway.
- Output valid JSON in this exact shape:

{{
  "title": "...",
  "speakers": ["HOST", "EXPERT"],
  "dialogue": [
    {{"speaker": "HOST", "text": "..."}},
    {{"speaker": "EXPERT", "text": "..."}}
  ]
}}
"""

        if hasattr(self.llm, "generate_json"):
            script = self.llm.generate_json(
                "You are a careful podcast script writer.",
                prompt,
                max_tokens=3000,
                retries=2,
            )
        else:
            response = self.llm.generate(
                "You are a careful podcast script writer.",
                prompt,
                max_tokens=3000,
            )
            script = self._extract_json(response)

        if "dialogue" not in script or not isinstance(script["dialogue"], list):
            raise ValueError("Model response did not include a dialogue list.")

        return script
