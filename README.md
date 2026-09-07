# document-podcast

A local-first podcast generation pipeline for Apple Silicon. This v0.1 targets a Mac mini M4 with 16 GB RAM and uses MLX-backed local inference for the document-to-podcast flow.

## Architecture

- PDF extraction with PyMuPDF
- document chunking for context-limited local LLMs
- Qwen3 4B 4-bit via MLX as the default local reasoning model
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
  - `llm/qwen.py` — MLX Qwen wrapper
  - `tts/` — speech: `cartesia.py` (CLI default, two fixed voices), `kokoro.py` (local default for programmatic use and tests), `dia.py` (whole-conversation prosody, optional), `assembly.py` (pauses, loudness, final join)

## Quick start

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
python -m podcast.cli ./data/input/document.pdf
```

## Local model target

The default Qwen model is:

```python
mlx-community/Qwen3-4B-4bit
```

This is a comfortable first target for a 16 GB M4 Mac and remains a strong v0.1 baseline before benchmarking 6-bit and 8-bit variants.
