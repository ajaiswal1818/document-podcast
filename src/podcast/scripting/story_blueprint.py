"""Story blueprint generation for documentary-style podcast scripting."""

from __future__ import annotations

import json
from typing import Any


def build_story_blueprint(
    llm: Any, analyses: list[dict], title: str = "Local Podcast Episode", *, technical: bool = False
) -> dict[str, Any]:
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

    audience = "a clinical or medical audience" if technical else "a non-biologist sales or field audience"
    terminology = (
        "Preserve medical terminology from the source. Do not add lay explanations or analogies."
        if technical
        else "Explain scientific ideas simply for someone with no biology background."
    )
    prompt = f"""
You are helping turn source material into a memorable learning story for {audience}.

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
- {terminology}
- Make the story feel like a real narrative, not a report summary.
- Identify the key concepts they need to understand before the story makes sense.
- The teaching items should preserve the source's clinical framing.
"""
    system = "You are a story architect for scientific content."
    if hasattr(llm, "generate_json"):
        blueprint = llm.generate_json(system, prompt, max_tokens=10000, retries=2)
    else:
        response = llm.generate(system, prompt, max_tokens=10000)
        start = response.find("{")
        end = response.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"Story blueprint output contained no JSON object: {response[:200]}")
        blueprint = json.loads(response[start : end + 1])
    if not isinstance(blueprint, dict):
        raise ValueError("Story blueprint output was not valid JSON.")
    return blueprint
