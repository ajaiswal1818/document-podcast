"""Story-quality evaluation for generated podcast scripts."""

from __future__ import annotations

import json
from typing import Any


class StoryEditor:
    """A lightweight editor pass that checks if a draft script is story-led and audience-aware."""

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    def review(self, script: dict[str, Any], blueprint: dict[str, Any] | None = None) -> dict[str, Any]:
        """Evaluate whether the script follows a real story arc and explains the issue for a lay audience."""
        blueprint = blueprint or {}
        story = blueprint.get("story", {}) if isinstance(blueprint, dict) else {}

        dialogue = script.get("dialogue", []) if isinstance(script, dict) else []
        host_questions = 0
        for turn in dialogue:
            if isinstance(turn, dict) and turn.get("speaker") == "HOST":
                text = str(turn.get("text", "")).lower()
                if "?" in text:
                    host_questions += 1

        checks = {
            "has_two_humans": bool(script.get("speakers")),
            "host_asks_questions": host_questions >= 1,
            "story_is_present": bool(story.get("central_question") or story.get("hook") or story.get("tension")),
            "technical_terms_are_explained": bool(blueprint.get("teaching") or blueprint.get("audience")),
            "facts_are_present": bool(dialogue),
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
                "The content should explain the story, not just list fact sections.",
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
        if "story_is_present" in failed:
            questions.append("Make the story explicit: define the central question, the tension, and the turning point.")
        if "technical_terms_are_explained" in failed:
            questions.append("Explain the technical concepts in plain English before using them in a way a non-biologist can follow.")
        if "facts_are_present" in failed:
            questions.append("Ensure the dialogue includes concrete source-backed facts and not just generic discussion.")

        if not questions:
            return "Keep the current story and tighten the wording, but do not change the core structure."

        return (
            "Regenerate this podcast script to fix the following issues:\n- "
            + "\n- ".join(questions)
            + f"\n\nStory goal: {story.get('central_question', 'Explain why this matters')}."
        )
