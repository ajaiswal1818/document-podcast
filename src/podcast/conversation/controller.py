"""Stateful conversation manager for turn-based podcast generation."""

from __future__ import annotations

from typing import Any

from podcast.agents.conversational_agent import ConversationalAgent
from podcast.conversation.knowledge import render_evidence_for_agent
from podcast.research.researcher import ResearchAgent


class ConversationState:
    """Tracks the conversational thread and the knowledge it has generated."""

    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.turns: list[dict[str, str]] = []
        self.facts: list[str] = []
        self.open_questions: list[str] = []

    @property
    def summary(self) -> str:
        if not self.turns:
            return "The conversation is beginning."
        return f"Discussion of {self.topic}, {len(self.turns)} turns so far."

    def add_turn(self, speaker: str, text: str) -> None:
        if not text:
            return
        self.turns.append({"speaker": speaker, "text": text})

    def add_facts(self, facts: list[str]) -> None:
        for fact in facts:
            cleaned = str(fact).strip()
            if cleaned and cleaned not in self.facts:
                self.facts.append(cleaned)

    def render_for_agent(self, speaker: str) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "summary": self.summary,
            "recent_turns": self.turns[-8:],
            "speaker": speaker,
            "safe_fact_context": render_evidence_for_agent(
                [{"claim": fact} for fact in self.facts]
            ),
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
        try:
            research = self.research_agent.lookup(title, base_material)
            state.add_facts(
                [str(finding.get("claim", "")) for finding in research.get("findings", []) if isinstance(finding, dict)]
            )
        except Exception:
            pass

        current_index = 0
        for _ in range(self.max_turns):
            agent = self.agents[current_index]
            response = agent.respond(state, source_material if isinstance(source_material, dict) else base_material)
            speech = response.get("text") or response.get("speech") or ""
            state.add_turn(agent.name, speech)
            current_index = 1 - current_index

        return {
            "title": title,
            "speakers": list(self.agent_names),
            "dialogue": [{"speaker": turn["speaker"], "text": turn["text"]} for turn in state.turns],
        }
