"""Two-agent conversational turn generation for a podcast simulation."""

from __future__ import annotations

import json
import re
from typing import Any

from podcast.conversation.evidence import render_evidence_for_agent, scrub_speech_text


class ConversationalAgent:
    """A conversational participant that responds from prior turns and source material."""

    def __init__(self, name: str, persona: str, llm: Any, *, voice: str | None = None) -> None:
        self.name = name
        self.persona = persona
        self.llm = llm
        self.voice = voice or "af_heart"

    def _extract_text(self, raw_response: str) -> str:
        """Pull the actual conversational text out of plain output or JSON-wrapped output."""
        text = (raw_response or "").strip()
        if not text:
            return ""

        cleaned = re.sub(r"(?is)<think>.*?</think>", " ", text)
        # Truncated generations leave an unclosed <think>; the rest is reasoning, not speech.
        cleaned = re.sub(r"(?is)<think>.*$", " ", cleaned)
        cleaned = re.sub(r"(?is)```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"(?is)\s*```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            payload = json.loads(cleaned)
            if isinstance(payload, dict):
                for key in ("text", "speech"):
                    if isinstance(payload.get(key), str):
                        return payload[key].strip()
                if isinstance(payload.get("dialogue"), list):
                    for turn in payload["dialogue"]:
                        if isinstance(turn, dict) and isinstance(turn.get("text"), str):
                            return turn["text"].strip()
            elif isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        return item["text"].strip()
        except json.JSONDecodeError:
            pass

        if cleaned.startswith("{"):
            start = cleaned.find("\"")
            if start != -1:
                end = cleaned.rfind("}")
                if end > start:
                    candidate = cleaned[start : end + 1]
                    try:
                        payload = json.loads(candidate)
                        if isinstance(payload, dict):
                            for key in ("text", "speech"):
                                if isinstance(payload.get(key), str):
                                    return payload[key].strip()
                    except json.JSONDecodeError:
                        pass

        return cleaned

    def _strip_speaker_prefix(self, text: str) -> str:
        """Remove a leading speaker label the model may add despite instructions."""
        cleaned = text.strip()
        cleaned = re.sub(rf"^(?:{re.escape(self.name)}|HOST|EXPERT)\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
        if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) > 2:
            cleaned = cleaned[1:-1].strip()
        return cleaned

    def _idea_texts(self, material: dict[str, Any] | None) -> list[str]:
        """Collect plain-language idea strings from structured material items."""
        source = material or {}
        if isinstance(source, dict):
            source = source.get("material", source)
        if not isinstance(source, dict):
            return []

        ideas: list[str] = []
        for key in ("translated_main_ideas", "main_ideas", "translated_facts", "facts", "conclusions"):
            items = source.get(key) or []
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    value = item.get("value")
                    if isinstance(value, str) and value.strip():
                        ideas.append(value.strip())
                elif isinstance(item, str) and item.strip():
                    ideas.append(item.strip())
        return ideas

    def _fallback_text(self, state: Any, material: dict[str, Any] | None = None) -> str:
        """Create a context-specific fallback when the model output is unusable."""
        topic = getattr(state, "topic", "the topic")
        turns = getattr(state, "turns", []) or []
        ideas = self._idea_texts(material)
        idea = ideas[len(turns) % len(ideas)] if ideas else f"the central finding behind {topic}"

        if not turns:
            return f"Let's start with {topic}: what is actually going on here, and why does it matter?"

        templates = [
            f"Hold on, let's slow down on one point: {idea} What should a listener make of that?",
            f"The piece I keep coming back to is this: {idea}",
            f"Here's what stands out to me: {idea} That's the part worth sitting with.",
        ]
        return templates[len(turns) % len(templates)]

    def respond(self, state: Any, material: dict[str, Any] | None = None) -> dict[str, Any]:
        """Generate the next spoken turn as plain text from conversation state and source material."""
        if self.llm is None:
            raise RuntimeError("No LLM available for conversational turn generation.")

        context = state.render_for_agent(self.name)
        evidence = render_evidence_for_agent(material if isinstance(material, dict) else None)
        language = str((material or {}).get("language", "en")).lower() if isinstance(material, dict) else "en"
        language_name = {"en": "English", "nl": "Dutch"}.get(language, language)
        recent_turns = context.get("recent_turns", [])
        transcript = "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in recent_turns)
        if not transcript:
            transcript = "(The conversation has not started yet. You speak first.)"

        system_prompt = (
            f"You are {self.name}, one of two people in a natural spoken conversation about a technical topic. "
            f"Your persona: {self.persona}. "
            "React to what the other person just said before deciding what to say. "
            "Do NOT always explain. Sometimes react briefly, ask a short question, or push on a surprising point. "
            "You are not a lecturer or a scripted interviewer: sound like a thoughtful human who is listening in real time. "
            "Make each turn do one clear conversational job—react, clarify, challenge, connect, or move the story forward. "
            "Avoid a rigid question-and-answer pattern, generic transitions, and recap language. "
            "Vary your length naturally: a brief reaction, a short question, a normal reply, or occasionally a deeper explanation. "
            "Ground every factual claim in the evidence provided; never invent data. "
            "Never mention internal bookkeeping such as sources, chunks, pages, ids, scores, or metadata. "
            "Do not repeat points that have already been made; move the conversation forward. "
            f"Speak only in {language_name}. "
            f"Reply with ONLY the words {self.name} speaks next - no JSON, no quotes, no speaker label, no stage directions."
        )
        focus = str(context.get("current_focus") or "").strip()
        focus_line = f"Current segment focus (steer the conversation here):\n{focus}\n\n" if focus else ""
        user_prompt = (
            f"Topic: {context['topic']}\n\n"
            f"Evidence you can draw on:\n{evidence}\n\n"
            f"Extra facts gathered during the conversation:\n{context.get('safe_fact_context', 'None yet.')}\n\n"
            f"{focus_line}"
            f"Conversation so far:\n{transcript}\n\n"
            f"What does {self.name} say next? Reply with the spoken words only."
        )

        raw = self.llm.generate(system_prompt, user_prompt, max_tokens=1600)
        text = self._strip_speaker_prefix(self._extract_text(raw))
        text = scrub_speech_text(text)

        if not text:
            text = self._fallback_text(state, material)

        # Repeating a substantive turn is degeneration, but short back-channels
        # ("Right.", "Exactly.") repeat naturally and are allowed through.
        if len(text.split()) > 5:
            recent_texts = {str(turn.get("text", "")).strip().lower() for turn in recent_turns}
            if text.strip().lower() in recent_texts:
                text = self._fallback_text(state, material)

        return {"speaker": self.name, "text": text, "metadata": {"persona": self.persona}}
