"""Story blueprint generation for documentary-style podcast scripting."""

from __future__ import annotations

import json
from typing import Any


def build_story_blueprint(llm: Any, analyses: list[dict], title: str = "Local Podcast Episode") -> dict[str, Any]:
    """Create a story-first blueprint that explains the central question, audience needs, and teaching points."""
    if llm is None or not getattr(llm, "available", False):
        raise RuntimeError("Qwen model unavailable.")

    aggregated = {
        "main_ideas": [],
        "facts": [],
        "numbers": [],
        "examples": [],
        "interesting_points": [],
        "conclusions": [],
    }
    for analysis in analyses:
        for key, values in aggregated.items():
            items = analysis.get(key, [])
            if isinstance(items, list):
                aggregated[key].extend(str(item) for item in items)

    prompt = f"""
You are helping turn source material into a memorable learning story for a non-biologist sales or field audience.

Title: {title}

Material:
{json.dumps(aggregated, ensure_ascii=False, indent=2)}

Your job is to discover the most compelling story that can honestly be told from this material.

Return valid JSON with this exact structure:

{{
  "story": {{
    "central_question": "...",
    "hook": "...",
    "problem": "...",
    "tension": "...",
    "turning_point": "...",
    "resolution": "...",
    "big_takeaway": "..."
  }},
  "audience": {{
    "what_they_already_know": ["..."],
    "what_will_confuse_them": ["..."],
    "what_they_need_to_remember": ["..."],
    "why_they_should_care": "..."
  }},
  "teaching": [
    {{
      "concept": "...",
      "explanation": "...",
      "analogy": "...",
      "source": {{"chunk": 1}}
    }}
  ]
}}

Rules:
- Do not invent facts.
- Write for someone with no biology background.
- Make the story feel like a real narrative, not a report summary.
- Identify the key concepts they need to understand before the story makes sense.
- The teaching items should help a sales or field person understand the science simply.
"""
    response = llm.generate(
        "You are a story architect for scientific content.",
        prompt,
        max_tokens=3000,
    )
    blueprint = json.loads(response)
    if not isinstance(blueprint, dict):
        raise ValueError("Story blueprint output was not valid JSON.")
    return blueprint
