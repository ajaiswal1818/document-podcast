from pathlib import Path

import soundfile as sf
from fastapi.testclient import TestClient

from podcast.config import AppConfig
from podcast.web.app import create_app


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        input_dir=tmp_path / "data" / "input",
        output_dir=tmp_path / "data" / "output",
        models_dir=tmp_path / "models",
    )


def test_default_config_uses_repository_data_directory() -> None:
    from podcast.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG.project_root.name == "document-podcast"
    assert DEFAULT_CONFIG.output_dir == DEFAULT_CONFIG.project_root / "data" / "output"


def test_web_lists_and_streams_generated_episode(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.ensure_directories()
    episode = config.output_dir / "demo_episode"
    episode.mkdir()
    sf.write(episode / "podcast.wav", [0.0, 0.1, 0.0], 8000)
    (episode / "podcast_script.json").write_text('{"title": "Demo episode"}', encoding="utf-8")
    (episode / "tts_manifest.json").write_text('{"backend": "kokoro", "language": "en"}', encoding="utf-8")

    client = TestClient(create_app(config))

    response = client.get("/api/episodes")
    assert response.status_code == 200
    assert response.json()[0]["title"] == "Demo episode"
    assert response.json()[0]["audio_url"] == "/api/episodes/demo_episode/audio"
    audio = client.get("/api/episodes/demo_episode/audio")
    assert audio.status_code == 200
    assert audio.headers["content-type"].startswith("audio/wav")


def test_web_rejects_paths_and_unsupported_uploads(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.ensure_directories()
    client = TestClient(create_app(config))

    assert client.get("/api/episodes/..%2Fsecret").status_code == 404
    response = client.post("/api/uploads", files={"file": ("unsafe.exe", b"not a document")})
    assert response.status_code == 400
