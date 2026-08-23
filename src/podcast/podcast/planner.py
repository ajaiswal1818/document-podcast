"""Podcast planning logic."""

from __future__ import annotations

from dataclasses import dataclass, field


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
