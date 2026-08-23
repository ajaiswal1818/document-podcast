"""Command-line entry points for the podcast project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from podcast.document.parser import DocumentParser, chunk_text
from podcast.llm.qwen import Qwen
from podcast.podcast.dialogue import PodcastDialogue
from podcast.podcast.planner import analyse_chunk, build_episode_plan


def run_pipeline(input_path: str, output_dir: str | Path = "data/output") -> dict:
    """Run the local document-to-podcast pipeline on a single source file."""
    source = Path(input_path)
    target_dir = Path(output_dir) / source.stem
    target_dir.mkdir(parents=True, exist_ok=True)

    parser = DocumentParser(source)
    extracted_text = parser.parse()
    chunks = chunk_text(extracted_text)

    (target_dir / "extracted.txt").write_text(extracted_text, encoding="utf-8")
    (target_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    llm = Qwen()
    analyses: list[dict] = []
    if llm.available:
        for chunk in chunks[:5]:
            analyses.append(analyse_chunk(llm, chunk))
    else:
        analyses = [{
            "main_ideas": ["The document was processed locally but the model is not currently available."],
            "facts": [],
            "numbers": [],
            "arguments": [],
            "examples": [],
            "interesting_points": [],
            "conclusions": ["The pipeline is ready for local model execution."],
        }]

    plan = build_episode_plan(analyses, title=source.stem.replace("_", " ").title())
    (target_dir / "document_analysis.json").write_text(json.dumps(analyses, ensure_ascii=False, indent=2), encoding="utf-8")
    (target_dir / "episode_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    script = PodcastDialogue().build(plan)
    (target_dir / "podcast_script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")

    return script


def main() -> None:
    """CLI entry point for local podcast generation."""
    parser = argparse.ArgumentParser(description="Generate a local podcast package from a document.")
    parser.add_argument("input", help="Path to the input PDF or text document")
    parser.add_argument("--output-dir", default="data/output", help="Directory for generated output artifacts")
    args = parser.parse_args()

    result = run_pipeline(args.input, args.output_dir)
    print(json.dumps({"title": result["title"], "dialogue_turns": len(result["dialogue"])}, indent=2))


if __name__ == "__main__":
    main()
