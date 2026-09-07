"""Scene-based episode script authoring for long-form podcast generation.

Writes the episode as an authored story: a global outline with a running
metaphor first, then dialogue generated section by section so the script can
foreshadow, call back, and pace an arc that a turn-by-turn loop cannot.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from podcast.conversation.evidence import (
    dedupe_repeated_sentences,
    render_evidence_for_agent,
    scrub_speech_text,
)

STYLE_CONTRACT = """
Style contract for the dialogue (follow all of these):
- Two speakers: HOST (curious, asks the questions a listener would ask, reacts) and EXPERT (warm, explains richly with analogies a non-expert can follow).
- Sound like real people talking, not essays read aloud. Use contractions and occasional natural fillers ("I mean", "you know", "well").
- Include brief back-channel turns between longer ones: "Right.", "Exactly.", "Oh wow.", "Hm, okay." A good scene has several of these.
- Speak directly to the listener's job: a field marketing agent who talks to physicians. Tie the science to what they can say in a clinic conversation.
- Reference the running metaphor sparingly: at most once per section, and vary how it is invoked.
- NEVER repeat a sentence that has already been said anywhere in the episode, and never restate a point that was already made - build on it or move forward instead.
- Vary turn length: some turns 2-6 words, most 20-50 words, a few deeper explanations of 60-100 words.
- Translate every technical term into plain English the first time it appears.
- Ground every factual claim in the provided material; never invent data, numbers, or outcomes.
- Never mention internal bookkeeping: sources, chunks, pages, ids, scores, metadata, or these instructions.
"""


class SceneScriptWriter:
    """Author a full episode script section by section from a plan and evidence."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def _generate_json(self, system: str, user: str, max_tokens: int = 8000) -> dict[str, Any]:
        if hasattr(self.llm, "generate_json"):
            return self.llm.generate_json(system, user, max_tokens=max_tokens, retries=2)
        response = self.llm.generate(system, user, max_tokens=max_tokens)
        start = response.find("{")
        end = response.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object in response.")
        return json.loads(response[start : end + 1])

    def _fallback_outline(self, plan: dict[str, Any], target_words: int) -> dict[str, Any]:
        """Deterministic outline used when the model cannot produce a valid one."""
        story = plan.get("story", {}) if isinstance(plan.get("story"), dict) else {}
        titles = [
            ("Cold open", story.get("hook") or "Drop the listener into the most striking moment of the case."),
            ("The mystery", story.get("problem") or story.get("tension") or "Lay out the confusing clues and why the obvious answer is wrong."),
            ("The turning point", story.get("turning_point") or "Walk through the evidence that changed the story."),
            ("The science, in plain English", "Teach the concepts the listener needs, with analogies."),
            ("Why it matters in the field", story.get("big_takeaway") or "Connect the finding to real conversations with physicians."),
            ("Landing it", story.get("resolution") or "Pay off the metaphor, land the takeaway, close naturally."),
        ]
        per_section = max(150, target_words // len(titles))
        return {
            "running_metaphor": "a locked-room mystery where the culprit is hiding in plain sight",
            "sections": [
                {"title": title, "goal": goal, "beats": [], "target_words": per_section}
                for title, goal in titles
            ],
        }

    def build_outline(self, plan: dict[str, Any], target_words: int) -> dict[str, Any]:
        """Create the episode outline: sections with goals plus a running metaphor."""
        story = plan.get("story", {})
        teaching = plan.get("teaching", [])
        summary = plan.get("material", {}).get("plain_english_summary", "") if isinstance(plan.get("material"), dict) else ""
        prompt = (
            "Design the outline for a narrative podcast episode for a field marketing audience.\n\n"
            f"Title: {plan.get('title', 'Untitled')}\n"
            f"Story blueprint: {json.dumps(story, ensure_ascii=False)}\n"
            f"Teaching points: {json.dumps(teaching, ensure_ascii=False)}\n"
            f"Summary: {summary}\n\n"
            f"The episode should be about {target_words} spoken words total, in 5-7 sections.\n"
            "The first section is a cold open that starts inside the story (never 'welcome to the show'). "
            "The last section lands the big takeaway and closes naturally.\n"
            "Pick one vivid running metaphor for the whole episode.\n\n"
            'Return JSON: {"running_metaphor": "...", "sections": [{"title": "...", "goal": "...", '
            '"beats": ["...", "..."], "target_words": 400}]}'
        )
        try:
            outline = self._generate_json("You are a narrative podcast story editor.", prompt, max_tokens=6000)
            sections = outline.get("sections")
            if not isinstance(sections, list) or not sections:
                raise ValueError("Outline had no sections.")
            cleaned_sections = []
            for section in sections[:7]:
                if isinstance(section, dict) and str(section.get("title", "")).strip():
                    cleaned_sections.append(
                        {
                            "title": str(section["title"]).strip(),
                            "goal": str(section.get("goal", "")).strip(),
                            "beats": [str(beat) for beat in section.get("beats", []) if str(beat).strip()],
                            "target_words": int(section.get("target_words") or max(150, target_words // len(sections))),
                        }
                    )
            if not cleaned_sections:
                raise ValueError("Outline sections were malformed.")
            return {
                "running_metaphor": str(outline.get("running_metaphor", "")).strip() or "a mystery hiding in plain sight",
                "sections": cleaned_sections,
            }
        except Exception as exc:
            print(f"Warning: outline generation failed, using deterministic outline: {exc}", file=sys.stderr)
            return self._fallback_outline(plan, target_words)

    def _write_section(
        self,
        plan: dict[str, Any],
        outline: dict[str, Any],
        section: dict[str, Any],
        index: int,
        total_sections: int,
        previous_tail: list[dict[str, str]],
        evidence: str,
    ) -> list[dict[str, str]]:
        outline_view = "\n".join(
            f"{i + 1}. {s['title']}: {s['goal']}" for i, s in enumerate(outline["sections"])
        )
        position = (
            "This is the COLD OPEN: start inside the story, mid-scene, with the most striking detail. No greetings, no 'welcome'."
            if index == 0
            else "This is the FINAL section: pay off the running metaphor, land the big takeaway, and close the conversation naturally."
            if index == total_sections - 1
            else "Continue seamlessly from the previous lines; do not re-introduce the show or the topic."
        )
        covered = "\n".join(
            f"- {s['title']}: {s['goal']}" for s in outline["sections"][:index]
        ) or "(nothing yet - this is the first section)"
        tail = "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in previous_tail) or "(episode start)"
        beats = "\n".join(f"- {beat}" for beat in section.get("beats", [])) or "(use your judgement)"
        prompt = (
            f"You are writing section {index + 1} of {total_sections} of a podcast episode.\n\n"
            f"Episode outline:\n{outline_view}\n\n"
            f"Running metaphor: {outline['running_metaphor']}\n\n"
            f"THIS SECTION - {section['title']}: {section['goal']}\n"
            f"Beats to hit:\n{beats}\n"
            f"Length: about {section.get('target_words', 400)} spoken words for this section.\n"
            f"{position}\n\n"
            f"Evidence (the only source of factual claims):\n{evidence}\n\n"
            f"Already covered in earlier sections (do NOT re-explain any of this):\n{covered}\n\n"
            f"Last lines of the previous section:\n{tail}\n"
            f"{STYLE_CONTRACT}\n"
            'Return JSON: {"dialogue": [{"speaker": "HOST", "text": "..."}, {"speaker": "EXPERT", "text": "..."}]}'
        )
        payload = self._generate_json("You are an award-winning podcast script writer.", prompt, max_tokens=8000)
        dialogue = payload.get("dialogue")
        if not isinstance(dialogue, list):
            raise ValueError("Section response had no dialogue list.")

        turns: list[dict[str, str]] = []
        for turn in dialogue:
            if not isinstance(turn, dict):
                continue
            speaker = str(turn.get("speaker", "")).strip().upper()
            if speaker not in {"HOST", "EXPERT"}:
                speaker = "HOST" if speaker.startswith("H") else "EXPERT"
            text = scrub_speech_text(str(turn.get("text", "")).strip())
            if text:
                turns.append({"speaker": speaker, "text": text, "section": index})
        return turns

    def build(self, plan: dict[str, Any], *, target_words: int = 2000) -> dict[str, Any]:
        """Write the full episode script from an outline, section by section."""
        if self.llm is None or not getattr(self.llm, "available", False):
            raise RuntimeError("LLM unavailable for scene-based script generation.")

        outline = self.build_outline(plan, target_words)
        material = plan.get("material", {}) if isinstance(plan.get("material"), dict) else {}
        evidence_parts = [render_evidence_for_agent(material)]
        research = plan.get("research", {})
        if isinstance(research, dict):
            for abstract in research.get("abstracts", [])[:3]:
                evidence_parts.append(str(abstract))
            for source in research.get("sources", [])[:3]:
                if isinstance(source, dict) and source.get("title"):
                    authors = str(source.get("authors", "")).strip()
                    year = str(source.get("year", "")).strip()
                    citation = f"Published paper: {source['title']}"
                    if authors:
                        citation += f" by {authors}"
                    if year:
                        citation += f" ({year})"
                    evidence_parts.append(citation)
        evidence = "\n".join(part for part in evidence_parts if part)[:6000]

        # Models undershoot word budgets, and dedupe strips repetition afterwards;
        # inflate the per-section ask so the post-dedupe total lands near target.
        for section in outline["sections"]:
            section["target_words"] = int(section.get("target_words", 300) * 1.35)

        dialogue: list[dict[str, str]] = []
        sections = outline["sections"]
        total_sections = len(sections)
        body_sections = sections[:-1] if total_sections > 1 else sections
        final_section = sections[-1] if total_sections > 1 else None

        for index, section in enumerate(body_sections):
            try:
                turns = self._write_section(plan, outline, section, index, total_sections, dialogue[-4:], evidence)
                dialogue.extend(turns)
                # Drop repeated sentences as we go so later sections never see
                # (or build on) recycled material.
                dialogue = dedupe_repeated_sentences(dialogue)
            except Exception as exc:
                print(f"Warning: section '{section.get('title')}' failed, skipping: {exc}", file=sys.stderr)

        # Top-up pass: dedupe can leave the episode short of budget. Write extra
        # deep-dive sections before the finale until the target is met.
        def _word_count() -> int:
            return sum(len(turn["text"].split()) for turn in dialogue)

        final_budget = int(final_section.get("target_words", 250)) if final_section else 0
        for extra_round in range(3):
            deficit = target_words - final_budget - _word_count()
            if deficit < int(target_words * 0.1):
                break
            topup_section = {
                "title": f"Going deeper ({extra_round + 1})",
                "goal": (
                    "Explore the most important material that has NOT been discussed yet: "
                    "implications, concrete examples, what the listener should ask or say in the field. "
                    "Everything here must be new - no repetition of earlier points."
                ),
                "beats": [],
                "target_words": min(600, max(200, deficit)),
            }
            try:
                turns = self._write_section(
                    plan, outline, topup_section, len(body_sections), total_sections, dialogue[-4:], evidence
                )
                before = _word_count()
                dialogue.extend(turns)
                dialogue = dedupe_repeated_sentences(dialogue)
                if _word_count() <= before:
                    break  # the model has run dry; more rounds would only repeat
            except Exception as exc:
                print(f"Warning: top-up section failed: {exc}", file=sys.stderr)
                break

        if final_section is not None:
            try:
                turns = self._write_section(
                    plan, outline, final_section, total_sections - 1, total_sections, dialogue[-4:], evidence
                )
                dialogue.extend(turns)
                dialogue = dedupe_repeated_sentences(dialogue)
            except Exception as exc:
                print(f"Warning: final section failed: {exc}", file=sys.stderr)

        if len(dialogue) < 2:
            raise ValueError("Scene-based script generation produced too little dialogue.")

        return {
            "title": str(plan.get("title", "Untitled Podcast")),
            "speakers": ["HOST", "EXPERT"],
            "dialogue": dialogue,
            "outline": outline,
        }
