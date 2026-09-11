"""Command-line entry points for the podcast project."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import soundfile as sf

import time
from typing import Any

from podcast.conversation.turn_loop import ConversationController
from podcast.conversation.evidence import dedupe_repeated_sentences, scrub_speech_text
from podcast.document.parser import DocumentParser, chunk_text
from podcast.llm.openai import DEFAULT_MODEL as DEFAULT_OPENAI_MODEL
from podcast.llm.openai import OpenAIChatGPT
from podcast.llm.openai import has_openai_api_key
from podcast.llm.qwen import Qwen
from podcast.scripting.script_builder import PodcastDialogue
from podcast.scripting.script_editor import StoryEditor
from podcast.scripting.episode_planner import analyse_chunk, build_episode_plan, translate_technical_terms
from podcast.scripting.scene_writer import SceneScriptWriter
from podcast.scripting.story_blueprint import build_story_blueprint
from podcast.research.researcher import ResearchAgent
from podcast.tts import CartesiaTTS, VibeVoiceTTS, get_tts_backend
from podcast.tts.assembly import assemble_audio_files
from podcast.tts.cartesia import LANGUAGE_NAMES as CARTESIA_LANGUAGE_NAMES
from podcast.tts.cartesia import get_voice_pair
from podcast.tts.cartesia import has_cartesia_api_key
from podcast.tts.kokoro import KokoroTTS


def _make_tts_backend(name: str, *, language: str = "en") -> Any:
    """Create a TTS backend while preserving compatibility with existing test monkeypatches."""
    backend_name = (name or "cartesia").lower()
    if backend_name == "cartesia":
        ctor = globals().get("CartesiaTTS") or get_tts_backend("cartesia").__class__
        return ctor(language=language)
    if backend_name == "kokoro":
        ctor = globals().get("KokoroTTS") or get_tts_backend("kokoro").__class__
        return ctor()
    if backend_name == "vibevoice":
        ctor = globals().get("VibeVoiceTTS") or get_tts_backend("vibevoice").__class__
        return ctor()
    if backend_name == "dia":
        return get_tts_backend("dia")
    raise ValueError(f"Unsupported TTS backend: {name}")


def _resolve_llm_backend(requested: str) -> str:
    """Prefer OpenAI when configured, otherwise retain an offline-capable CLI."""
    if requested in {"auto", "openai"} and has_openai_api_key():
        return "openai"
    return "qwen"


def _resolve_tts_backend(requested: str) -> str:
    """Prefer Cartesia when configured, otherwise retain an offline-capable CLI."""
    if requested not in {"auto", "cartesia"}:
        return requested
    if requested in {"auto", "cartesia"} and has_cartesia_api_key():
        return "cartesia"
    return "kokoro"


def _record_step(metrics: list[dict[str, Any]], name: str, *, start_time: float | None = None, tokens: int | None = None, chars: int | None = None) -> float:
    """Record timing, token count, and throughput information for a pipeline step."""
    wall_time = time.perf_counter() - start_time if start_time is not None else 0.0
    entry: dict[str, Any] = {
        "name": name,
        "time_seconds": round(wall_time, 4),
    }
    if tokens is not None and wall_time > 0:
        entry["tokens_per_second"] = round(tokens / wall_time, 2)
    if chars is not None and wall_time > 0:
        entry["chars_per_second"] = round(chars / wall_time, 2)
    metrics.append(entry)
    return wall_time


MAX_EPISODE_MINUTES = 15.0
# Conservative budget that leaves room for natural Cartesia pauses while keeping
# a 15-minute requested episode below the hard audio-duration ceiling in practice.
SPOKEN_WORDS_PER_MINUTE = 140
NATURAL_SPEECH_WPM = 175

_ACKNOWLEDGEMENT_OPENERS = {
    "exactly", "right", "yeah", "yes", "oh", "wow", "totally", "absolutely",
    "hm", "hmm", "okay", "ok", "true", "sure", "wait", "really", "no",
}


def _plan_turn_gaps(turns: list[dict[str, Any]]) -> list[float]:
    """Humanize inter-turn timing: instant latch-ons for reactions, beats before deep dives.

    Returns one pause duration (seconds) per boundary between consecutive turns.
    """
    gaps: list[float] = []
    for previous, current in zip(turns, turns[1:]):
        words = str(current.get("text", "")).split()
        opener = words[0].strip(".,!?…'\"").lower() if words else ""
        prev_section = previous.get("section")
        if prev_section is not None and current.get("section") != prev_section:
            gaps.append(random.uniform(0.7, 1.1))
        elif len(words) <= 5 or opener in _ACKNOWLEDGEMENT_OPENERS:
            gaps.append(random.uniform(0.0, 0.15))
        elif len(words) > 60:
            gaps.append(random.uniform(0.5, 0.9))
        else:
            gaps.append(random.uniform(0.2, 0.55))
    return gaps


def _word_count(turns: list[dict[str, Any]]) -> int:
    return sum(len(str(turn.get("text", "")).split()) for turn in turns)


def _trim_text_to_words(text: str, limit: int) -> str:
    """Keep complete sentences where possible when enforcing an episode ceiling."""
    words = text.split()
    if len(words) <= limit:
        return text
    kept: list[str] = []
    used = 0
    for sentence in re.split(r"(?<=[.!?…])\s+", text):
        sentence_words = sentence.split()
        if used + len(sentence_words) > limit:
            break
        kept.append(sentence)
        used += len(sentence_words)
    return " ".join(kept).strip() or " ".join(words[:limit]).rstrip(".,;:") + "."


def _cap_dialogue_words(dialogue: list[dict[str, Any]], max_words: int | None) -> list[dict[str, Any]]:
    """Keep the story ending while enforcing the maximum configured duration."""
    if not max_words or _word_count(dialogue) <= max_words:
        return dialogue

    closing = dialogue[-4:]
    closing_words = _word_count(closing)
    closing_budget = min(closing_words, max_words, max(120, max_words // 8))
    if closing_words > closing_budget:
        closing = []
        remaining = closing_budget
        for turn in reversed(dialogue[-4:]):
            text = _trim_text_to_words(str(turn.get("text", "")), remaining)
            if text:
                closing.insert(0, {**turn, "text": text})
                remaining -= len(text.split())
            if remaining <= 0:
                break

    retained: list[dict[str, Any]] = []
    remaining = max_words - _word_count(closing)
    for turn in dialogue[:-4]:
        text = _trim_text_to_words(str(turn.get("text", "")), remaining)
        if text:
            retained.append({**turn, "text": text})
            remaining -= len(text.split())
        if remaining <= 0:
            break
    return retained + closing


def run_pipeline(
    input_path: str,
    output_dir: str | Path = "data/output",
    *,
    llm: Any | None = None,
    max_chunks: int = 5,
    # Keep programmatic runs and tests local unless a backend is explicitly chosen.
    tts_backend: str = "kokoro",
    language: str = "en",
    target_minutes: float = 15.0,
    research: bool = True,
) -> dict:
    """Run the local document-to-podcast pipeline on a single source file."""
    source = Path(input_path)
    target_dir = Path(output_dir) / source.stem
    target_dir.mkdir(parents=True, exist_ok=True)
    benchmark_steps: list[dict[str, Any]] = []
    benchmark_started = time.perf_counter()

    parser = DocumentParser(source)
    parse_started = time.perf_counter()
    extracted_text = parser.parse()
    _record_step(benchmark_steps, "parse_document", start_time=parse_started, chars=len(extracted_text))
    chunks = chunk_text(extracted_text)
    _record_step(benchmark_steps, "chunk_document", start_time=time.perf_counter(), chars=sum(len(chunk) for chunk in chunks))

    (target_dir / "extracted.txt").write_text(extracted_text, encoding="utf-8")
    (target_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    llm = llm or Qwen()
    if not llm.available:
        raise RuntimeError("Selected LLM is unavailable.")

    research_report: dict[str, Any] = {}
    if research:
        research_started = time.perf_counter()
        try:
            research_report = ResearchAgent(llm).research_document(extracted_text)
        except Exception as exc:
            print(f"Warning: research retrieval failed, continuing with local material: {exc}", file=sys.stderr)
        _record_step(benchmark_steps, "research_document", start_time=research_started)
        if research_report:
            (target_dir / "research.json").write_text(
                json.dumps(research_report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            retrieved = [source.get("title", "?") for source in research_report.get("sources", []) if source.get("has_full_text")]
            if retrieved:
                print(f"Research: retrieved full text for {len(retrieved)} paper(s).", file=sys.stderr)

    research_chunks: list[str] = []
    for full_text in research_report.get("full_texts", []):
        research_chunks.extend(chunk_text(str(full_text.get("text", ""))))

    analyses: list[dict] = []
    for chunk in (chunks[:max_chunks] + research_chunks[:max_chunks]):
        chunk_start = time.perf_counter()
        analysis = analyse_chunk(llm, chunk)
        tokens = max(1, len(str(analysis).split()))
        _record_step(benchmark_steps, "analyse_chunk", start_time=chunk_start, tokens=tokens, chars=len(chunk))
        analyses.append(analysis)

    plan = build_episode_plan(analyses, title=source.stem.replace("_", " ").title(), llm=llm)
    story_blueprint = plan.get("story") or build_story_blueprint(llm, analyses, title=source.stem.replace("_", " ").title())
    translated_material = translate_technical_terms(llm, plan.get("material", {}))
    plan["material"] = {
        **plan.get("material", {}),
        "plain_english_summary": translated_material.get("plain_english_summary", ""),
        "translated_facts": translated_material.get("facts", []),
        "translated_main_ideas": translated_material.get("main_ideas", []),
    }
    plan["story"] = story_blueprint.get("story", plan.get("story", {}))
    plan["audience"] = story_blueprint.get("audience", plan.get("audience", {}))
    plan["teaching"] = story_blueprint.get("teaching", plan.get("teaching", []))
    plan["language"] = language
    if research_report:
        plan["research"] = {
            "query": research_report.get("query", ""),
            "sources": [
                {key: source.get(key, "") for key in ("title", "authors", "journal", "year", "doi")}
                for source in research_report.get("sources", [])
            ],
            "abstracts": research_report.get("abstracts", []),
        }
    (target_dir / "document_analysis.json").write_text(json.dumps(analyses, ensure_ascii=False, indent=2), encoding="utf-8")
    (target_dir / "episode_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (target_dir / "story_blueprint.json").write_text(json.dumps(story_blueprint, ensure_ascii=False, indent=2), encoding="utf-8")
    (target_dir / "plain_english_translation.json").write_text(json.dumps(translated_material, ensure_ascii=False, indent=2), encoding="utf-8")

    dialogue_builder = PodcastDialogue(llm)
    editor = StoryEditor(llm)
    dialogue_started = time.perf_counter()

    controller_material = {
        "title": plan.get("title", source.stem.replace("_", " ").title()),
        "material": plan.get("material", {}),
        "story": story_blueprint.get("story", {}),
        "audience": story_blueprint.get("audience", {}),
        "teaching": story_blueprint.get("teaching", []),
    }
    effective_minutes = min(target_minutes, MAX_EPISODE_MINUTES) if target_minutes else None
    if target_minutes and effective_minutes != target_minutes:
        print(f"Script: limiting requested duration to {MAX_EPISODE_MINUTES:g} minutes.", file=sys.stderr)
    target_words = int(effective_minutes * SPOKEN_WORDS_PER_MINUTE) if effective_minutes else None

    script: dict[str, Any] | None = None
    try:
        script = SceneScriptWriter(llm).build(plan, target_words=target_words or 2000)
    except Exception as exc:
        print(f"Warning: scene-based script generation failed, falling back to turn loop: {exc}", file=sys.stderr)

    if not script or len(script.get("dialogue", [])) < 2:
        controller = ConversationController(llm, max_turns=24, agent_names=("HOST", "EXPERT"))
        controller.target_words = target_words
        script = controller.run(controller_material)

    if not script.get("dialogue") or len(script.get("dialogue", [])) < 2:
        script = dialogue_builder.build(plan)

    script["dialogue"] = _cap_dialogue_words(dedupe_repeated_sentences(script.get("dialogue", [])), target_words)
    verdict = editor.review(script, story_blueprint, target_words=target_words)
    critical_checks = {"dialogue_is_diverse", "no_metadata_leakage"}
    for _attempt in range(2):
        if verdict["approved"]:
            break
        regeneration_prompt = editor.build_regeneration_prompt(script, story_blueprint, verdict)
        candidate = dialogue_builder.build(plan, regeneration_prompt=regeneration_prompt)
        candidate["dialogue"] = _cap_dialogue_words(dedupe_repeated_sentences(candidate.get("dialogue", [])), target_words)
        candidate_verdict = editor.review(candidate, story_blueprint, target_words=target_words)
        script_is_degenerate = critical_checks & set(verdict["failed_checks"])
        candidate_is_degenerate = critical_checks & set(candidate_verdict["failed_checks"])
        if (script_is_degenerate and not candidate_is_degenerate) or len(
            candidate_verdict["failed_checks"]
        ) < len(verdict["failed_checks"]):
            script, verdict = candidate, candidate_verdict
    script["dialogue"] = _cap_dialogue_words(dedupe_repeated_sentences(script.get("dialogue", [])), target_words)
    (target_dir / "editor_verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")

    script_chars = len(json.dumps(script, ensure_ascii=False))
    _record_step(benchmark_steps, "generate_dialogue", start_time=dialogue_started, tokens=max(1, script_chars // 4), chars=script_chars)
    (target_dir / "podcast_script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")

    # Script work is done; free the LLM's memory before loading TTS models.
    if hasattr(llm, "release"):
        llm.release()

    backend_name = (tts_backend or "kokoro").lower()
    tts = _make_tts_backend(backend_name, language=language)
    voice_map = (
        {speaker: profile["id"] for speaker, profile in get_voice_pair(language).items()}
        if backend_name == "cartesia"
        else {"HOST": "af_heart", "EXPERT": "am_adam"}
    )
    tts_manifest: dict[str, Any] = {
        "backend": backend_name,
        "segments": 0,
    }
    if backend_name == "cartesia":
        tts_manifest.update(
            {
                "provider": "Cartesia",
                "model": getattr(tts, "model_id", "sonic-3.6"),
                "language": language,
                "voice_roles": get_voice_pair(language),
                "contexts": {"enabled": True, "scope": "one context per spoken turn"},
            }
        )
        print(
            f"TTS: using Cartesia Sonic in {CARTESIA_LANGUAGE_NAMES[language]} with "
            f"{get_voice_pair(language)['HOST']['name']} (HOST) and "
            f"{get_voice_pair(language)['EXPERT']['name']} (EXPERT); WebSocket contexts enabled.",
            file=sys.stderr,
        )
    else:
        print(f"TTS: using local backend={backend_name}.", file=sys.stderr)
    audio_files: list[str] = []
    dialogue_turns = script.get("dialogue", [])
    if hasattr(tts, "synthesize_dialogue"):
        tts_start = time.perf_counter()
        audio_files = list(tts.synthesize_dialogue(dialogue_turns, str(target_dir)))
        _record_step(
            benchmark_steps,
            "tts_dialogue",
            start_time=tts_start,
            chars=sum(len(str(turn.get("text", ""))) for turn in dialogue_turns),
        )
        spoken_turns: list[dict[str, Any]] = []
    else:
        spoken_turns = []
        for turn_index, turn in enumerate(dialogue_turns, start=1):
            speaker = str(turn.get("speaker", "HOST")).strip() or "HOST"
            text = scrub_speech_text(str(turn.get("text", "")).strip())
            if not text:
                continue
            segment_path = target_dir / f"{speaker.lower()}_{turn_index}.wav"
            tts_start = time.perf_counter()
            tts.synthesize(text, str(segment_path), voice=voice_map.get(speaker, "am_adam"))
            _record_step(benchmark_steps, f"tts_{speaker.lower()}_{turn_index}", start_time=tts_start, chars=len(text))
            audio_files.append(str(segment_path))
            spoken_turns.append({**turn, "text": text})

    if audio_files:
        final_audio_path = target_dir / "podcast.wav"
        assemble_started = time.perf_counter()
        if spoken_turns:
            assemble_audio_files(audio_files, str(final_audio_path), gaps=_plan_turn_gaps(spoken_turns))
        else:
            # Dia chunks already contain multi-turn prosody; join them tightly.
            # Dia also over-paces its delivery, so calibrate a pitch-preserving
            # slow-down from the measured words-per-minute of the raw chunks.
            words = sum(len(scrub_speech_text(str(turn.get("text", "")).strip()).split()) for turn in dialogue_turns)
            seconds = sum(sf.info(path).duration for path in audio_files)
            stretch_rate = None
            if words and seconds:
                actual_wpm = words / (seconds / 60)
                stretch_rate = max(0.75, min(1.0, NATURAL_SPEECH_WPM / actual_wpm))
                if stretch_rate >= 0.97:
                    stretch_rate = None
            assemble_audio_files(audio_files, str(final_audio_path), gap_seconds=0.15, stretch_rate=stretch_rate)
        _record_step(benchmark_steps, "assemble_audio", start_time=assemble_started, chars=sum(len(str(path)) for path in audio_files))
        (target_dir / "audio_manifest.json").write_text(
            json.dumps({"audio_files": audio_files, "output": str(final_audio_path)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    tts_manifest["segments"] = len(audio_files)
    (target_dir / "tts_manifest.json").write_text(
        json.dumps(tts_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_time = time.perf_counter() - benchmark_started
    benchmark_payload = {
        "title": source.stem.replace("_", " ").title(),
        "total_time_seconds": round(total_time, 4),
        "steps": benchmark_steps,
    }
    (target_dir / "benchmark.json").write_text(json.dumps(benchmark_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"title": script.get("title", source.stem), "total_time_seconds": round(total_time, 4), "steps": benchmark_steps}, ensure_ascii=False))

    return script


def main() -> None:
    """CLI entry point for local podcast generation."""
    parser = argparse.ArgumentParser(description="Generate a local podcast package from a document.")
    parser.add_argument("input", help="Path to the input PDF or text document")
    parser.add_argument("--output-dir", default="data/output", help="Directory for generated output artifacts")
    parser.add_argument(
        "--language",
        choices=["en", "da"],
        default="en",
        help="Spoken podcast language. Cartesia uses only a voice pair configured for this language.",
    )
    parser.add_argument(
        "--llm",
        choices=["auto", "openai", "qwen"],
        default="auto",
        help="LLM provider. Auto uses OpenAI when OPENAI_API_KEY is set, otherwise local Qwen.",
    )
    parser.add_argument(
        "--openai-model",
        default=DEFAULT_OPENAI_MODEL,
        help=f"OpenAI model for --llm openai (default: {DEFAULT_OPENAI_MODEL}).",
    )
    parser.add_argument(
        "--model-version",
        choices=["Qwen3-4B-4bit", "Qwen3-4B-6bit", "Qwen3-4B-8bit", "Qwen3-8B-4bit"],
        default="Qwen3-4B-4bit",
        help="Name of the local Qwen3 model variant to use.",
    )
    parser.add_argument("--max-chunks", type=int, default=5, help="Maximum number of chunks to process in v0.1.")
    parser.add_argument(
        "--tts",
        choices=["auto", "cartesia", "kokoro", "vibevoice", "dia"],
        default="auto",
        help="TTS backend. Auto uses Cartesia when CARTESIA_API_KEY is set, otherwise local Kokoro.",
    )
    parser.add_argument("--skip-research", action="store_true", help="Skip scholarly research retrieval.")
    parser.add_argument(
        "--target-minutes",
        type=float,
        default=15.0,
        help="Target episode duration in minutes; drives how long the conversation runs. Use 0 for the legacy fixed 24-turn behavior.",
    )
    args = parser.parse_args()

    try:
        llm_backend = _resolve_llm_backend(args.llm)
        tts_backend = _resolve_tts_backend(args.tts)
        llm = OpenAIChatGPT(model=args.openai_model) if llm_backend == "openai" else Qwen(model_version=args.model_version)
        if not llm.available:
            raise RuntimeError(f"Selected LLM is unavailable: {getattr(llm, 'load_error', None)}")
        if llm_backend == "openai":
            print(f"LLM: using OpenAI model={args.openai_model}.", file=sys.stderr)
        else:
            print(f"LLM: using local Qwen model={args.model_version}.", file=sys.stderr)
        if args.llm in {"auto", "openai"} and llm_backend == "qwen":
            print("LLM: OpenAI key not found; fell back to local Qwen.", file=sys.stderr)
        if args.tts in {"auto", "cartesia"} and tts_backend == "kokoro":
            print("TTS: Cartesia key not found; fell back to local Kokoro.", file=sys.stderr)

        result = run_pipeline(
            args.input,
            args.output_dir,
            llm=llm,
            max_chunks=args.max_chunks,
            tts_backend=tts_backend,
            language=args.language,
            target_minutes=args.target_minutes,
            research=not args.skip_research,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(json.dumps({"title": result["title"], "dialogue_turns": len(result["dialogue"])}, indent=2))


if __name__ == "__main__":
    main()
