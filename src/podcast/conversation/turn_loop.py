"""Stateful conversation manager for turn-based podcast generation."""

from __future__ import annotations

from typing import Any

from podcast.conversation.agent import ConversationalAgent
from podcast.conversation.evidence import render_evidence_for_agent
from podcast.research.researcher import ResearchAgent


class ConversationState:
    """Tracks the conversational thread and the knowledge it has generated."""

    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.turns: list[dict[str, str]] = []
        self.facts: list[str] = []
        self.open_questions: list[str] = []
        self.current_focus = ""

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
            "current_focus": self.current_focus,
            "safe_fact_context": render_evidence_for_agent(
                [{"claim": fact} for fact in self.facts]
            ),
        }


class ConversationController:
    """Coordinates a small two-agent discussion into a podcast script."""

    def __init__(
        self,
        llm: Any,
        *,
        max_turns: int = 24,
        agent_names: tuple[str, str] = ("A", "B"),
        target_words: int | None = None,
    ) -> None:
        self.llm = llm
        self.max_turns = max_turns
        self.agent_names = agent_names
        self.target_words = target_words
        self.agents = [
            ConversationalAgent(
                agent_names[0],
                "curious and probing; asks short pointed questions and reacts briefly to keep the conversation moving",
                llm,
            ),
            ConversationalAgent(
                agent_names[1],
                "calm and explanatory; gives generous, example-rich explanations with analogies a non-expert can follow",
                llm,
            ),
        ]
        self.research_agent = ResearchAgent()

    @staticmethod
    def _build_focus_plan(material: dict[str, Any]) -> list[str]:
        """Build an ordered list of conversation segment foci from the story arc and teaching points."""
        foci: list[str] = []
        story = material.get("story", {}) if isinstance(material.get("story"), dict) else {}
        for key in ("hook", "problem", "tension", "turning_point", "resolution", "big_takeaway"):
            value = story.get(key)
            if isinstance(value, str) and value.strip():
                foci.append(f"{key.replace('_', ' ').title()}: {value.strip()}")

        teaching = material.get("teaching", [])
        if isinstance(teaching, list):
            for item in teaching:
                if isinstance(item, dict) and str(item.get("concept", "")).strip():
                    concept = str(item["concept"]).strip()
                    explanation = str(item.get("explanation", "")).strip()
                    analogy = str(item.get("analogy", "")).strip()
                    focus = f"Teach the concept '{concept}'"
                    if explanation:
                        focus += f": {explanation}"
                    if analogy:
                        focus += f" (a useful analogy: {analogy})"
                    foci.append(focus)

        if not foci:
            foci = ["Explore the main findings, what they mean in plain language, and why they matter."]
        return foci

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

        foci = self._build_focus_plan(source_material if isinstance(source_material, dict) else {})
        target_words = self.target_words
        turn_cap = self.max_turns if not target_words else max(self.max_turns, target_words // 8)

        total_words = 0
        current_index = 0
        for turn_count in range(turn_cap):
            if target_words:
                if total_words >= target_words:
                    break
                progress = total_words / target_words
                if progress >= 0.92:
                    state.current_focus = (
                        "Wrap up the episode: land the big takeaway, make clear why it matters, "
                        "and close the conversation naturally."
                    )
                else:
                    state.current_focus = foci[min(len(foci) - 1, int(progress * len(foci)))]
            else:
                state.current_focus = foci[min(len(foci) - 1, (turn_count * len(foci)) // max(1, self.max_turns))]

            agent = self.agents[current_index]
            response = agent.respond(state, source_material if isinstance(source_material, dict) else base_material)
            speech = response.get("text") or response.get("speech") or ""
            state.add_turn(agent.name, speech)
            total_words += len(speech.split())
            current_index = 1 - current_index

        return {
            "title": title,
            "speakers": list(self.agent_names),
            "dialogue": [{"speaker": turn["speaker"], "text": turn["text"]} for turn in state.turns],
        }
