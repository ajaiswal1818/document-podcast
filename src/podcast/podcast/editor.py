"""Story-quality evaluation for generated podcast scripts."""

from __future__ import annotations

import json
import re
from typing import Any

_LEAK_PATTERNS = (
    r"\{['\"]?value['\"]?\s*:",
    r"['\"]?source['\"]?\s*:\s*\{",
    r"\bchunk\s*\d+",
    r"\bclaim[_ -]?id\b",
    r"\bretrieval[_ -]?score\b",
)


class StoryEditor:
    """A lightweight editor pass that checks if a draft script is story-led and audience-aware."""

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    def review(self, script: dict[str, Any], blueprint: dict[str, Any] | None = None) -> dict[str, Any]:
        """Evaluate whether the script follows a real story arc and explains the issue for a lay audience."""
        blueprint = blueprint or {}
        story = blueprint.get("story", {}) if isinstance(blueprint, dict) else {}
        audience = blueprint.get("audience", {}) if isinstance(blueprint, dict) else {}

        dialogue = script.get("dialogue", []) if isinstance(script, dict) else []
        host_questions = 0
        for turn in dialogue:
            if isinstance(turn, dict) and turn.get("speaker") == "HOST":
                text = str(turn.get("text", "")).lower()
                if "?" in text:
                    host_questions += 1

        final_expert_text = ""
        for turn in reversed(dialogue):
            if isinstance(turn, dict) and turn.get("speaker") == "EXPERT":
                final_expert_text = str(turn.get("text", "")).lower()
                break

        has_story_arc = bool(
            story.get("central_question")
            and (story.get("hook") or story.get("tension") or story.get("turning_point"))
            and (story.get("resolution") or story.get("big_takeaway") or story.get("turning_point"))
        )
        explicit_takeaway_markers = [
            "why this matters",
            "the takeaway",
            "bottom line",
            "this means",
            "that means",
            "important because",
            "matters because",
            "why it matters",
            "the key point",
            "so the real story is",
        ]
        takeaway_is_clear = bool(
            story.get("big_takeaway")
            or story.get("resolution")
            or any(marker in final_expert_text for marker in explicit_takeaway_markers)
        )

        turn_texts = [
            str(turn.get("text", "")).strip()
            for turn in dialogue
            if isinstance(turn, dict) and str(turn.get("text", "")).strip()
        ]
        unique_texts = {text.lower() for text in turn_texts}
        dialogue_is_diverse = (
            len(unique_texts) / len(turn_texts) >= 0.6 if len(turn_texts) > 3 else bool(turn_texts)
        )
        no_metadata_leakage = not any(
            re.search(pattern, text, flags=re.IGNORECASE) for text in turn_texts for pattern in _LEAK_PATTERNS
        )

        checks = {
            "has_two_humans": bool(script.get("speakers")),
            "host_asks_questions": host_questions >= 1,
            "story_is_present": bool(story.get("central_question") or story.get("hook") or story.get("tension")),
            "story_has_arc": has_story_arc,
            "technical_terms_are_explained": bool(blueprint.get("teaching") or blueprint.get("audience")),
            "takeaway_is_clear": takeaway_is_clear,
            "facts_are_present": bool(dialogue),
            "dialogue_is_diverse": dialogue_is_diverse,
            "no_metadata_leakage": no_metadata_leakage,
        }

        failed_checks = [name for name, passed in checks.items() if not passed]
        approved = not failed_checks
        return {
            "approved": approved,
            "checks": checks,
            "failed_checks": failed_checks,
            "notes": [
                "The script should sound like two humans talking.",
                "The host should ask clarifying questions a listener would ask.",
                "The story needs a central question, tension, and a clear payoff.",
                "The listener should understand why the finding matters by the end.",
            ],
        }

    def build_regeneration_prompt(self, script: dict[str, Any], blueprint: dict[str, Any] | None, verdict: dict[str, Any]) -> str:
        """Build a short list of repair instructions for a regenerate pass."""
        blueprint = blueprint or {}
        story = blueprint.get("story", {}) if isinstance(blueprint, dict) else {}
        failed = verdict.get("failed_checks", [])
        questions = []

        if "host_asks_questions" in failed:
            questions.append("Add at least one HOST question that a listener would genuinely ask before understanding the science.")
        if "story_is_present" in failed or "story_has_arc" in failed:
            questions.append("Make the story explicit: define the central question, the tension, and the turning point; the script should have a clear arc rather than a list of facts.")
        if "technical_terms_are_explained" in failed:
            questions.append("Explain the technical concepts in plain English before using them in a way a non-biologist can follow.")
        if "takeaway_is_clear" in failed:
            questions.append("End with a clear takeaway that tells the audience why this matters and what they should remember.")
        if "facts_are_present" in failed:
            questions.append("Ensure the dialogue includes concrete source-backed facts and not just generic discussion.")
        if "dialogue_is_diverse" in failed:
            questions.append("Every turn must say something new; do not repeat the same sentence or restate the same point across turns.")
        if "no_metadata_leakage" in failed:
            questions.append("Remove all internal bookkeeping from the spoken text: no source references, chunk numbers, ids, scores, or dict/JSON fragments.")

        if not questions:
            return "Keep the current story and tighten the wording, but do not change the core structure."

        return (
            "Regenerate this podcast script to fix the following issues:\n- "
            + "\n- ".join(questions)
            + f"\n\nStory goal: {story.get('central_question', 'Explain why this matters')}."
        )
