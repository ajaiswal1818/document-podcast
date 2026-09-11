# document-podcast

A local-first podcast generation pipeline for Apple Silicon. This v0.1 targets a Mac mini M4 with 16 GB RAM and uses MLX-backed local inference for the document-to-podcast flow.

## Architecture

- PDF extraction with PyMuPDF
- document chunking for context-limited local LLMs
- OpenAI ChatGPT via the Responses API as the CLI reasoning model; Qwen remains available for local/offline runs
- structured episode planning and dialogue generation
- Cartesia Sonic TTS for CLI runs with exactly two consistent voices: Skylar (HOST) and Daniel (EXPERT)

## Project layout

- `models/` for local model artifacts
- `data/input/` for source PDFs or text files
- `data/output/` for generated artifacts
- `src/podcast/` for the package code:
  - `cli.py` — entry point and pipeline orchestration
  - `document/parser.py` — PDF/text extraction and chunking
  - `research/researcher.py` — scholarly retrieval (Europe PMC / Crossref, open-access full text)
  - `scripting/` — script generation: `episode_planner.py`, `story_blueprint.py`, `scene_writer.py` (primary), `script_builder.py` (single-shot fallback), `script_editor.py` (quality gate)
  - `conversation/` — turn-loop fallback: `turn_loop.py`, `agent.py`, `evidence.py` (metadata/speech hygiene)
  - `llm/openai.py` — OpenAI Responses API wrapper (CLI default)
  - `llm/qwen.py` — MLX Qwen wrapper for local/offline runs
  - `tts/` — speech: `cartesia.py` (CLI default, two fixed voices), `kokoro.py` (local default for programmatic use and tests), `dia.py` (whole-conversation prosody, optional), `assembly.py` (pauses, loudness, final join)

## Quick start

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
python -m podcast.cli ./data/input/document.pdf
```

## LLM configuration

The CLI auto-selects OpenAI and Cartesia when their keys are in the ignored `.env`; with no `.env` or missing keys, it uses local Qwen and Kokoro instead. To run:

```bash
uv run document-podcast ./data/input/document.pdf
```

Use `--llm qwen --tts kokoro` to force local providers. Programmatic `run_pipeline()` also remains local by default for tests.

## Podcast language

Use `--language da` for a Danish podcast. Cartesia synthesizes Danish with its fixed Katie (HOST) and Jameson (EXPERT) voice pair; English remains the default.
