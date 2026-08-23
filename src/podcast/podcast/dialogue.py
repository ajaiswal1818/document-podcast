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
        """Ask the LLM to produce a realistic two-speaker podcast conversation that follows the story blueprint."""
        if self.llm is None or not getattr(self.llm, "available", False):
            raise RuntimeError("Qwen model unavailable.")

        title = "Untitled Podcast"
        material: dict[str, Any] = {}
        story_blueprint: dict[str, Any] = {}
        if isinstance(plan, dict):
            title = str(plan.get("title", title))
            material = plan.get("material", {})
            story_blueprint = plan.get("story", {}) or {}
            audience = plan.get("audience", {}) or {}
            teaching = plan.get("teaching", []) or []
        else:
            audience = {}
            teaching = []

        prompt = f"""
You are producing a podcast story for a healthcare / biotech marketing field audience.

Title: {title}

Audience: a non-biologist field agent who needs to understand the science quickly and confidently.
Goal: tell the most compelling story that can honestly be drawn from this material, not a generic summary.

Story Blueprint:
{json.dumps(story_blueprint, ensure_ascii=False, indent=2)}

Audience Model:
{json.dumps(audience, ensure_ascii=False, indent=2)}

Teaching Priorities:
{json.dumps(teaching, ensure_ascii=False, indent=2)}

Material:
{json.dumps(material, ensure_ascii=False, indent=2)}

Create a realistic two-person conversation with HOST and EXPERT.

Requirements:
- Every factual claim must be supported by the supplied material.
- The script must follow the story blueprint, especially the central question, tension, turning point, and takeaway.
- The HOST should ask the kinds of questions a real listener would ask before the science makes sense.
- The EXPERT should explain complex ideas simply and clearly, without talking like a textbook.
- Translate technical biology and medical terms into plain English for a lay audience.
- Keep it conversational and human, with a natural rhythm: HOST question, brief expert answer, then a follow-up question.
- Use a structure that matches the blueprint instead of generic sections.
- If the source is complex or technical, explain it in simple analogies without losing meaning.
- Do not invent data, product claims, or unsupported conclusions.
- Make the numbers meaningful by explaining why they matter, not just stating them.
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
