"""Podcast planning logic."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

SYSTEM_PROMPT = """
You are a professional podcast research producer.

You are given part of a source document.
Extract only information that is supported by the source.

Identify:
1. Main ideas
2. Important facts
3. Important numbers
4. Arguments
5. Examples
6. Interesting or surprising points
7. Conclusions

Do not invent information.

Return valid JSON.
"""


def _extract_json_object(raw_response: str) -> dict:
    """Extract the first JSON object from an LLM response."""
    text = raw_response.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not extract JSON from response: {raw_response[:200]}")
    return json.loads(text[start : end + 1])


@dataclass
class PodcastPlan:
    """Simple podcast plan container."""

    title: str
    segments: list[str] = field(default_factory=list)


class PodcastPlanner:
    """Creates a podcast plan from source text."""

    def __init__(self, title: str = "Untitled Podcast") -> None:
        self.title = title

    def plan(self, source_text: str) -> PodcastPlan:
        """Split source text into a simple segment plan."""
        chunks = [part.strip() for part in source_text.split("\n\n") if part.strip()]
        return PodcastPlan(title=self.title, segments=chunks[:5])


def analyse_chunk(llm, chunk: str) -> dict:
    """Analyse a source chunk and return a JSON object with structured evidence."""
    prompt = f"""
Analyse this document section.

SOURCE:

{chunk}

Return JSON in this format:

{{
  "main_ideas": [],
  "facts": [],
  "numbers": [],
  "arguments": [],
  "examples": [],
  "interesting_points": [],
  "conclusions": []
}}
"""

    response = llm.generate(
        SYSTEM_PROMPT,
        prompt,
        max_tokens=2000,
    )
    return _extract_json_object(response)


def build_episode_plan(analyses: list[dict], title: str = "Local Podcast Episode") -> dict:
    """Merge document analyses into a single episode plan JSON with source attribution."""
    material_keys = [
        "main_ideas",
        "facts",
        "numbers",
        "arguments",
        "examples",
        "interesting_points",
        "conclusions",
    ]
    merged: dict[str, list[dict]] = {key: [] for key in material_keys}

    for chunk_index, analysis in enumerate(analyses, start=1):
        for key in material_keys:
            items = analysis.get(key, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    payload = dict(item)
                    payload.setdefault("source", {"chunk": chunk_index})
                    merged[key].append(payload)
                else:
                    merged[key].append({"value": str(item), "source": {"chunk": chunk_index}})

    return {
        "title": title,
        "segments": [
            {"title": "Hook", "focus": "Why this matters"},
            {"title": "Core ideas", "focus": "The main arguments and facts"},
            {"title": "Examples and evidence", "focus": "Concrete examples and numbers"},
            {"title": "Takeaway", "focus": "What the audience should remember"},
        ],
        "material": merged,
    }
