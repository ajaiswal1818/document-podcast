"""Dialogue generation for the podcast script."""

from __future__ import annotations

from typing import Any


class PodcastDialogue:
    """Generate a structured dialogue JSON object from a plan."""

    def build(self, plan: Any) -> dict[str, Any]:
        """Convert a plan into a simple, structured host/expert dialogue."""
        title = "Untitled Podcast"
        material: dict[str, Any] = {}

        if isinstance(plan, dict):
            title = str(plan.get("title", title))
            material = plan.get("material", {})

        ideas = material.get("main_ideas", [])
        facts = material.get("facts", [])
        examples = material.get("examples", [])
        takeaway = material.get("conclusions", [])

        dialogue = [
            {"speaker": "HOST", "text": f"Welcome back. Today we’re unpacking {title.lower()}, and we’ll focus on the ideas, evidence, and takeaways that matter most."},
            {"speaker": "EXPERT", "text": "The key point is that the source material defines the story: " + str(ideas[0] if ideas else "the topic is grounded in clear evidence and context.")},
            {"speaker": "HOST", "text": "What does that mean in practical terms?"},
            {"speaker": "EXPERT", "text": "It means we should pay attention to the supporting facts: " + "; ".join(str(item) for item in facts[:3]) if facts else "This is a strong, evidence-based claim backed by the document."},
            {"speaker": "HOST", "text": "Can you give me an example from the material?"},
            {"speaker": "EXPERT", "text": "A concrete example is " + str(examples[0] if examples else "a clear case within the document") + "."},
            {"speaker": "HOST", "text": "So what’s the main takeaway for listeners?"},
            {"speaker": "EXPERT", "text": "The takeaway is simple: " + str(takeaway[0] if takeaway else "the subject matters because the evidence points to a clear conclusion.")},
        ]

        return {
            "title": title,
            "speakers": ["HOST", "EXPERT"],
            "dialogue": dialogue,
        }
