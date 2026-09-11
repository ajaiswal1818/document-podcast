"""A localhost-only web UI and API for the document podcast pipeline."""

from __future__ import annotations

import argparse
import json
import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from podcast.cli import _resolve_llm_backend, _resolve_tts_backend, run_pipeline
from podcast.config import AppConfig, DEFAULT_CONFIG
from podcast.llm.openai import DEFAULT_MODEL as DEFAULT_OPENAI_MODEL
from podcast.llm.openai import OpenAIChatGPT
from podcast.llm.qwen import Qwen


ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
STATIC_DIR = Path(__file__).with_name("static")


class GenerationRequest(BaseModel):
    """Options exposed to the browser for one local pipeline run."""

    input_file: str = Field(min_length=1, max_length=255)
    language: Literal["en", "da"] = "en"
    llm: Literal["auto", "openai", "qwen"] = "auto"
    tts: Literal["auto", "cartesia", "kokoro", "vibevoice", "dia"] = "auto"
    target_minutes: float = Field(default=15.0, gt=0, le=15.0)
    research: bool = True
    episode_format: Literal["plain", "technical"] = "plain"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_filename(filename: str) -> str:
    candidate = Path(filename or "document").name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip(".-")
    if not safe:
        raise HTTPException(status_code=400, detail="Please provide a valid file name.")
    if Path(safe).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
    return safe


def _within(directory: Path, candidate: Path) -> Path:
    """Resolve a user-provided child path without allowing traversal."""
    root = directory.resolve()
    resolved = candidate.resolve()
    if resolved.parent != root:
        raise HTTPException(status_code=404, detail="Item not found.")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _episode_payload(output_dir: Path, episode_dir: Path) -> dict[str, Any]:
    audio_path = episode_dir / "podcast.wav"
    script = _read_json(episode_dir / "podcast_script.json")
    plan = _read_json(episode_dir / "episode_plan.json")
    manifest = _read_json(episode_dir / "tts_manifest.json")
    translation = _read_json(episode_dir / "plain_english_translation.json")
    try:
        duration = round(sf.info(audio_path).duration, 2)
    except (RuntimeError, OSError):
        duration = None
    title = script.get("title") or plan.get("title") or episode_dir.name.replace("_", " ").title()
    return {
        "id": episode_dir.name,
        "title": str(title),
        "language": manifest.get("language") or plan.get("language") or "en",
        "tts_backend": manifest.get("backend"),
        "episode_format": manifest.get("format") or plan.get("format") or "plain",
        "duration_seconds": duration,
        "created_at": datetime.fromtimestamp(audio_path.stat().st_mtime, UTC).isoformat(),
        "audio_url": f"/api/episodes/{episode_dir.name}/audio",
        "script_url": f"/api/episodes/{episode_dir.name}/script",
        "summary": str(translation.get("plain_english_summary") or ""),
    }


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Build the self-hosted app. It intentionally has no network-facing defaults."""
    app_config = config or DEFAULT_CONFIG
    app_config.ensure_directories()
    app = FastAPI(title="Novo Radio", version="0.1.0")
    jobs: dict[str, dict[str, Any]] = {}
    pipeline_lock = threading.Lock()

    def find_episode(episode_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", episode_id):
            raise HTTPException(status_code=404, detail="Episode not found.")
        episode_dir = _within(app_config.output_dir, app_config.output_dir / episode_id)
        if not (episode_dir.is_dir() and (episode_dir / "podcast.wav").is_file()):
            raise HTTPException(status_code=404, detail="Episode not found.")
        return episode_dir

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/episodes")
    def list_episodes() -> list[dict[str, Any]]:
        episodes = [
            _episode_payload(app_config.output_dir, path)
            for path in app_config.output_dir.iterdir()
            if path.is_dir() and (path / "podcast.wav").is_file()
        ]
        return sorted(episodes, key=lambda item: item["created_at"], reverse=True)

    @app.get("/api/episodes/{episode_id}")
    def get_episode(episode_id: str) -> dict[str, Any]:
        episode_dir = find_episode(episode_id)
        payload = _episode_payload(app_config.output_dir, episode_dir)
        payload["script"] = _read_json(episode_dir / "podcast_script.json")
        payload["plan"] = _read_json(episode_dir / "episode_plan.json")
        return payload

    @app.get("/api/episodes/{episode_id}/audio")
    def get_audio(episode_id: str) -> FileResponse:
        episode_dir = find_episode(episode_id)
        return FileResponse(episode_dir / "podcast.wav", media_type="audio/wav", filename=f"{episode_id}.wav")

    @app.get("/api/episodes/{episode_id}/script")
    def get_script(episode_id: str) -> dict[str, Any]:
        episode_dir = find_episode(episode_id)
        script = _read_json(episode_dir / "podcast_script.json")
        if not script:
            raise HTTPException(status_code=404, detail="Script not found.")
        return script

    @app.post("/api/uploads", status_code=status.HTTP_201_CREATED)
    async def upload_source(file: UploadFile = File(...)) -> dict[str, str]:
        filename = _safe_filename(file.filename or "document")
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Files must be 50 MB or smaller.")
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        destination = app_config.input_dir / filename
        if destination.exists():
            destination = app_config.input_dir / f"{destination.stem}-{uuid.uuid4().hex[:8]}{destination.suffix}"
        destination.write_bytes(content)
        return {"filename": destination.name}

    def run_job(job_id: str, request: GenerationRequest, source: Path) -> None:
        jobs[job_id].update({"status": "queued", "updated_at": _now()})
        try:
            # MLX and local TTS models contend for the same hardware, so serialize jobs.
            with pipeline_lock:
                jobs[job_id].update({"status": "running", "updated_at": _now()})
                llm_backend = _resolve_llm_backend(request.llm)
                tts_backend = _resolve_tts_backend(request.tts)
                llm = (
                    OpenAIChatGPT(model=DEFAULT_OPENAI_MODEL)
                    if llm_backend == "openai"
                    else Qwen()
                )
                if not llm.available:
                    raise RuntimeError(f"Selected LLM is unavailable: {getattr(llm, 'load_error', None)}")
                result = run_pipeline(
                    str(source),
                    app_config.output_dir,
                    llm=llm,
                    tts_backend=tts_backend,
                    language=request.language,
                    target_minutes=request.target_minutes,
                    research=request.research,
                    episode_format=request.episode_format,
                )
                episode_id = source.stem
                jobs[job_id].update({
                    "status": "succeeded",
                    "episode_id": episode_id,
                    "title": result.get("title", episode_id),
                    "updated_at": _now(),
                })
        except Exception as exc:  # keep error visible to the local UI, not in a hung job
            jobs[job_id].update({"status": "failed", "error": str(exc), "updated_at": _now()})

    @app.post("/api/generate", status_code=status.HTTP_202_ACCEPTED)
    def generate_episode(request: GenerationRequest) -> dict[str, Any]:
        filename = _safe_filename(request.input_file)
        source = _within(app_config.input_dir, app_config.input_dir / filename)
        if not source.is_file():
            raise HTTPException(status_code=404, detail="Uploaded source file not found.")
        job_id = uuid.uuid4().hex
        jobs[job_id] = {"id": job_id, "status": "queued", "created_at": _now(), "updated_at": _now()}
        threading.Thread(target=run_job, args=(job_id, request, source), daemon=True).start()
        return jobs[job_id]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Generation job not found.")
        return job

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="web")
    return app


app = create_app()


def main() -> None:
    """Run the browser UI on loopback only."""
    parser = argparse.ArgumentParser(description="Run the local Novo Radio web app.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (defaults to localhost).")
    parser.add_argument("--port", default=8000, type=int, help="Port to listen on.")
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("podcast.web.app:app", host=args.host, port=args.port)
