"""Podcast planning logic."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

from .story import build_story_blueprint

SYSTEM_PROMPT = """
You are a medical-storytelling analyst for a marketing and field-sales audience.

Your job is to turn a scientific document into a clear, credible story for a non-biologist field agent.

You are given a section of a source document. Extract only information that is supported by the source.

Identify and prioritize:
1. Main ideas in plain English
2. Important facts and discoveries
3. Key numbers and statistics
4. Why this matters to practitioners or customers
5. Examples or case signals
6. Surprising or important takeaways
7. Conclusions and practical implications
8. Technical terms that need plain-language translation

Rules:
- Translate scientific terms into clear, non-technical language whenever possible.
- Keep the meaning faithful to the source; do not oversimplify away the core claim.
- For every technical concept, prefer a simple explanation that a field professional can understand quickly.
- Do not invent information or stretch interpretations beyond the source.
- Use wording that feels relevant to a healthcare / biotech / marketing audience.

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
Analyse this document section for a non-expert marketing field audience.

SOURCE:

{chunk}

We need a story that helps a field agent explain the science in plain English.

Please identify:
- the core scientific insight
- the business or practical significance
- technical terms that should be translated to simple language
- evidence that is strong and source-backed
- anything surprising or worth highlighting

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

Important:
- Prefer plain-language explanations in the values.
- Keep the claims grounded in the source text.
- If a term is technical, explain it in a way a non-biologist can understand.
"""

    response = llm.generate(
        SYSTEM_PROMPT,
        prompt,
        max_tokens=10000,
    )
    return _extract_json_object(response)


def translate_technical_terms(llm, material: dict) -> dict:
    """Translate complex scientific claims into plain-English language for non-experts."""
    if llm is None or not getattr(llm, "available", False):
        raise RuntimeError("Qwen model unavailable.")

    prompt = f"""
You are a medical translator for a marketing field audience.

If I am a sales person with no biology background, what do I need to understand before this story makes sense?

Material:
{json.dumps(material, ensure_ascii=False, indent=2)}

Return JSON with:
- "plain_english_summary": one short paragraph in simple language
- "facts": a list of plain-English fact statements
- "main_ideas": a list of simple explanations of the main ideas

Rules:
- Keep it accurate to the source.
- Translate biology / medical terms into simple language.
- Write for a non-scientist field professional.
- Do not invent claims.
- Focus on what someone needs to understand before the story becomes clear.
"""
    response = llm.generate(
        "You translate dense science into plain English for field teams.",
        prompt,
        max_tokens=10000,
    )
    translated = _extract_json_object(response)
    if not isinstance(translated, dict):
        raise ValueError("Medical translation output was not valid JSON.")

    return translated


def build_episode_plan(analyses: list[dict], title: str = "Local Podcast Episode", llm=None) -> dict:
    """Merge document analyses into a single episode plan JSON with source attribution and story-first structure."""
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

    blueprint: dict = {}
    if llm is not None and getattr(llm, "available", False):
        try:
            blueprint = build_story_blueprint(llm, analyses, title=title)
        except Exception as exc:
            print(f"Warning: story blueprint generation failed, continuing without one: {exc}", file=sys.stderr)
            blueprint = {}

    return {
        "title": title,
        "segments": [
            {"title": "Hook", "focus": "The compelling reason this matters"},
            {"title": "Problem and tension", "focus": "What is confusing or risky"},
            {"title": "Turning point", "focus": "The evidence that changes the story"},
            {"title": "Resolution and takeaway", "focus": "What the audience should understand and remember"},
        ],
        "story": blueprint.get("story", {}),
        "audience": blueprint.get("audience", {}),
        "teaching": blueprint.get("teaching", []),
        "material": merged,
    }
