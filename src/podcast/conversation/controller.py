"""Stateful conversation manager for turn-based podcast generation."""

from __future__ import annotations

import json
from typing import Any

from podcast.agents.conversational_agent import ConversationalAgent
from podcast.research.researcher import ResearchAgent


class ConversationState:
    """Tracks the current conversational history and relevant knowledge."""

    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.turns: list[dict[str, str]] = []
        self.facts: list[str] = []
        self.open_questions: list[str] = []
        self.summary = "The conversation is beginning."

    def add_turn(self, speaker: str, text: str) -> None:
        if not text:
            return
        self.turns.append({"speaker": speaker, "text": text})
        if len(self.turns) > 32:
            self.turns = self.turns[-32:]
        self.summary = self._build_summary()

    def _build_summary(self) -> str:
        if not self.turns:
            return "The conversation is beginning."
        recent = " | ".join(f"{turn['speaker']}: {turn['text']}" for turn in self.turns[-4:])
        return f"Current thread: {self.topic}. Recent turns: {recent}"

    def render_for_agent(self, speaker: str) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "summary": self.summary,
            "recent_turns": self.turns[-6:],
            "speaker": speaker,
        }


class ConversationController:
    """Coordinates a small two-agent discussion into a podcast script."""

    def __init__(self, llm: Any, *, max_turns: int = 24, agent_names: tuple[str, str] = ("A", "B")) -> None:
        self.llm = llm
        self.max_turns = max_turns
        self.agent_names = agent_names
        self.agents = [
            ConversationalAgent(agent_names[0], "curious and probing", llm),
            ConversationalAgent(agent_names[1], "calm and explanatory", llm),
        ]
        self.research_agent = ResearchAgent()

    def run(self, material: dict[str, Any] | None = None) -> dict[str, Any]:
        title = "Demo"
        source_material = material or {}
        if isinstance(source_material, dict):
            title = str(source_material.get("title", title))
            base_material = source_material.get("material", source_material)
        else:
            base_material = {}

        state = ConversationState(title)
        current_index = 0

        for _ in range(self.max_turns):
            agent = self.agents[current_index]
            response = agent.respond(state, base_material)
            metadata = response.get("metadata", {})
            if metadata.get("needs_research"):
                topic = metadata.get("topic") or title
                research = self.research_agent.lookup(str(topic), base_material)
                state.summary = f"Research retrieved for {topic}: {research['summary']}"
            speech = response.get("text") or response.get("speech") or ""
            state.add_turn(agent.name, speech)
            current_index = 1 - current_index

        script = {
            "title": title,
            "speakers": list(self.agent_names),
            "dialogue": [{"speaker": turn["speaker"], "text": turn["text"]} for turn in state.turns],
        }
        return script
