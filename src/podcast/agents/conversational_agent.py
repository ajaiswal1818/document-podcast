"""Two-agent conversational turn generation for a podcast simulation."""

from __future__ import annotations

import json
import re
from typing import Any


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
        cleaned = re.sub(r"(?is)```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"(?is)\s*```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            payload = json.loads(cleaned)
            if isinstance(payload, dict):
                if isinstance(payload.get("text"), str):
                    return payload["text"].strip()
                if isinstance(payload.get("speech"), str):
                    return payload["speech"].strip()
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
                            if isinstance(payload.get("text"), str):
                                return payload["text"].strip()
                            if isinstance(payload.get("speech"), str):
                                return payload["speech"].strip()
                    except json.JSONDecodeError:
                        pass

        return cleaned

    def _fallback_text(self, state: Any, material: dict[str, Any] | None = None) -> str:
        """Create a context-specific fallback when the model is too generic or non-responsive."""
        topic = getattr(state, "topic", "the topic")
        history = getattr(state, "turns", []) if hasattr(state, "turns") else []
        source_summary = material or {}
        main_ideas = []
        if isinstance(source_summary, dict):
            main_ideas = source_summary.get("main_ideas", []) or source_summary.get("facts", []) or []
        idea = str(main_ideas[0]) if main_ideas else "the underlying pattern"

        if not history:
            if self.name == "A":
                return f"Let’s start with the problem in {topic}: what is actually changing, and why does it matter?"
            return f"I’m trying to understand the key issue in {topic} before we get too far into the details."

        if self.name == "A":
            return f"Why does this matter for {topic}? The key point is that {idea}, and that is what makes the story worth paying attention to."
        return f"That is interesting, and the real takeaway is that {idea}. It matters because it changes how we understand the pattern in {topic}."

    def _normalize_material_for_prompt(self, material: dict[str, Any] | None = None) -> dict[str, Any]:
        """Turn internal structured material into plain-language narrative context for the agent."""
        source_material = material or {}
        if isinstance(source_material, dict):
            source_material = source_material.get("material", source_material)

        normalized: dict[str, Any] = {}
        if not isinstance(source_material, dict):
            return {"summary": str(source_material)}

        for key, value in source_material.items():
            if key in {"source", "chunk", "value", "metadata", "source_metadata"}:
                continue
            if isinstance(value, list):
                cleaned = []
                for item in value:
                    if isinstance(item, dict):
                        if "value" in item and isinstance(item["value"], str):
                            cleaned.append(item["value"])
                        elif "text" in item and isinstance(item["text"], str):
                            cleaned.append(item["text"])
                        else:
                            cleaned.append(str(item))
                    else:
                        cleaned.append(str(item))
                normalized[key] = cleaned
            elif isinstance(value, dict):
                if "value" in value and isinstance(value["value"], str):
                    normalized[key] = value["value"]
                else:
                    normalized[key] = str(value)
            else:
                normalized[key] = value

        if not normalized:
            return {"summary": "Use the source material to build a clear, audience-friendly explanation."}
        return normalized

    def _structured_response(self, state: Any, material: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return a structured turn payload that separates TTS speech from system metadata."""
        context = state.render_for_agent(self.name)
        source_material = material or {}
        source_summary = self._normalize_material_for_prompt(source_material)
        system_prompt = (
            "You are participating in a natural two-person conversation about a technical topic. "
            "React to what the other person just said before deciding what you want to say. "
            "Do NOT always explain. Sometimes react, ask a short question, or push on a surprising point. "
            "Choose a response length naturally: a brief reaction (2-10 words), a short question (5-20 words), a normal response (20-60 words), or a deeper explanation (60-120 words). "
            "Return valid JSON with keys: speech, intent, topic, new_information, question_for_other_agent, needs_research, confidence, conversation_direction. "
            f"Your persona: {self.persona}."
        )
        user_prompt = (
            f"Topic: {context['topic']}\n\n"
            f"Conversation summary: {context['summary']}\n\n"
            f"Recent turns:\n{json.dumps(context['recent_turns'], ensure_ascii=False, indent=2)}\n\n"
            f"Relevant source material:\n{json.dumps(source_summary, ensure_ascii=False, indent=2)}\n\n"
            "Generate the next turn as a JSON object with the required keys. The 'speech' field should be the speaking text for audio. "
            "Keep it conversational, not informational. React first, then decide whether you need to explain, ask, or challenge the idea."
        )
        raw = self.llm.generate(system_prompt, user_prompt, max_tokens=600)
        try:
            payload = json.loads(self._extract_text(raw))
        except json.JSONDecodeError:
            payload = {}

        if not isinstance(payload, dict):
            payload = {}

        speech = str(payload.get("speech") or self._fallback_text(state, source_material))
        return {
            "speaker": self.name,
            "speech": speech,
            "intent": str(payload.get("intent") or "clarify"),
            "topic": str(payload.get("topic") or context["topic"]),
            "new_information": bool(payload.get("new_information", False)),
            "question_for_other_agent": str(payload.get("question_for_other_agent") or ""),
            "needs_research": bool(payload.get("needs_research", False)),
            "confidence": float(payload.get("confidence", 0.8) or 0.8),
            "conversation_direction": str(payload.get("conversation_direction") or "continue"),
            "emotion": str(payload.get("emotion") or "curious"),
            "should_continue_topic": bool(payload.get("should_continue_topic", True)),
        }

    def respond(self, state: Any, material: dict[str, Any] | None = None) -> dict[str, str]:
        """Generate the next turn using conversation state and relevant source material."""
        if self.llm is None:
            raise RuntimeError("No LLM available for conversational turn generation.")

        response = self._structured_response(state, material)
        text = str(response.get("speech") or self._fallback_text(state, material)).strip()
        low_text = text.lower()
        if not text or text in {"hello", "hi", "world", "test"} or len(text) < 12:
            text = self._fallback_text(state, material)
        if self.name == "A" and "matter" not in low_text and "shift" not in low_text and "why" not in low_text:
            text = self._fallback_text(state, material)
        if not getattr(state, "turns", None) and "that makes sense" in text.lower():
            text = self._fallback_text(state, material)
        return {"speaker": self.name, "text": text, "metadata": response}
