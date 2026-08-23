"""Command-line entry points for the podcast project."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import time
from typing import Any

from podcast.document.parser import DocumentParser, chunk_text
from podcast.llm.qwen import Qwen
from podcast.podcast.dialogue import PodcastDialogue
from podcast.podcast.planner import analyse_chunk, build_episode_plan
from podcast.tts.kokoro import KokoroTTS, assemble_audio_files


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


def run_pipeline(
    input_path: str,
    output_dir: str | Path = "data/output",
    *,
    llm: Qwen | None = None,
    max_chunks: int = 5,
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
        raise RuntimeError("Qwen model unavailable.")

    analyses: list[dict] = []
    for chunk in chunks[: max_chunks]:
        chunk_start = time.perf_counter()
        analysis = analyse_chunk(llm, chunk)
        tokens = max(1, len(str(analysis).split()))
        _record_step(benchmark_steps, "analyse_chunk", start_time=chunk_start, tokens=tokens, chars=len(chunk))
        analyses.append(analysis)

    plan = build_episode_plan(analyses, title=source.stem.replace("_", " ").title())
    (target_dir / "document_analysis.json").write_text(json.dumps(analyses, ensure_ascii=False, indent=2), encoding="utf-8")
    (target_dir / "episode_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    dialogue_started = time.perf_counter()
    script = PodcastDialogue(llm).build(plan)
    script_chars = len(json.dumps(script, ensure_ascii=False))
    _record_step(benchmark_steps, "generate_dialogue", start_time=dialogue_started, tokens=max(1, script_chars // 4), chars=script_chars)
    (target_dir / "podcast_script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")

    tts = KokoroTTS()
    audio_files: list[str] = []
    for turn_index, turn in enumerate(script.get("dialogue", []), start=1):
        speaker = str(turn.get("speaker", "HOST")).strip() or "HOST"
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        segment_path = target_dir / f"{speaker.lower()}_{turn_index}.wav"
        tts_start = time.perf_counter()
        if speaker != "HOST":
            tts.voice = "am_adam"
        tts.synthesize(text, str(segment_path), voice=tts.voice)
        _record_step(benchmark_steps, f"tts_{speaker.lower()}_{turn_index}", start_time=tts_start, chars=len(text))
        audio_files.append(str(segment_path))

    if audio_files:
        final_audio_path = target_dir / "podcast.wav"
        assemble_started = time.perf_counter()
        assemble_audio_files(audio_files, str(final_audio_path))
        _record_step(benchmark_steps, "assemble_audio", start_time=assemble_started, chars=sum(len(str(path)) for path in audio_files))
        (target_dir / "audio_manifest.json").write_text(
            json.dumps({"audio_files": audio_files, "output": str(final_audio_path)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
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
        "--model-version",
        choices=["Qwen3-4B-4bit", "Qwen3-4B-6bit", "Qwen3-4B-8bit", "Qwen3-8B-4bit"],
        default="Qwen3-4B-4bit",
        help="Name of the local Qwen3 model variant to use.",
    )
    parser.add_argument("--max-chunks", type=int, default=5, help="Maximum number of chunks to process in v0.1.")
    args = parser.parse_args()

    try:
        llm = Qwen(model_version=args.model_version)
        if not llm.available:
            raise RuntimeError("Qwen model unavailable.")

        result = run_pipeline(args.input, args.output_dir, llm=llm, max_chunks=args.max_chunks)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(json.dumps({"title": result["title"], "dialogue_turns": len(result["dialogue"])}, indent=2))


if __name__ == "__main__":
    main()
