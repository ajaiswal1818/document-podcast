"""Dialogue generation for the podcast script."""

from __future__ import annotations


class PodcastDialogue:
    """Placeholder dialogue generator."""

    def build(self, plan: list[str]) -> list[str]:
        """Convert a plan into a simple dialogue script."""
        return [f"Narrator: Introduce the topic: {segment[:80]}" for segment in plan]
