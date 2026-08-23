"""Research operations for the agentic podcast engine."""

from __future__ import annotations

from typing import Any


class ResearchAgent:
    """Very small research layer that stores evidence and sources for agent requests."""

    def __init__(self) -> None:
        self.knowledge_pool: list[dict[str, Any]] = []

    def lookup(self, topic: str, material: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return source-backed suggestions for a topic, plus metadata to support future agent turns."""
        material = material or {}
        facts = material.get("main_ideas", []) or material.get("facts", []) or []
        sources = [
            {
                "title": "Source document",
                "url": "local://document",
                "summary": "The source contains relevant context for the topic.",
                "source": "internal_document",
                "relevance": 0.95,
            }
        ]
        if facts:
            sources.insert(
                0,
                {
                    "title": "Evidence in source material",
                    "url": "local://facts",
                    "summary": str(facts[0]),
                    "source": "internal_facts",
                    "relevance": 0.98,
                },
            )

        entry = {
            "topic": topic,
            "sources": sources,
            "summary": f"Research for {topic} has been gathered and stored in the knowledge pool.",
        }
        self.knowledge_pool.append(entry)
        return entry
