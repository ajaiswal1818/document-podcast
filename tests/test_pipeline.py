import json
from pathlib import Path

import fitz
import soundfile as sf

from podcast.cli import run_pipeline
from podcast.document.parser import DocumentParser, chunk_text, extract_pdf
from podcast.podcast.dialogue import PodcastDialogue
from podcast.podcast.planner import build_episode_plan
from podcast.tts.kokoro import assemble_audio_files


class FakeLLM:
    available = True

    def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
        return '{"title": "Demo", "speakers": ["HOST", "EXPERT"], "dialogue": [{"speaker": "HOST", "text": "Hello"}, {"speaker": "EXPERT", "text": "World"}]}'


class FakeInvalidLLM:
    available = True

    def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
        return '{"title": "Broken", "speakers": ["HOST", "EXPERT"], "dialogue": [{"speaker": "HOST", "text": "This string is invalid because it contains a trailing comma", }, {"speaker": "EXPERT", "text": "World"}]}'


class FakeThinkLLM:
    available = True

    def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
        return '''
<think>
I am reasoning internally about the source material.
</think>

```json
{"title": "Demo", "speakers": ["HOST", "EXPERT"], "dialogue": [{"speaker": "HOST", "text": "Hello"}, {"speaker": "EXPERT", "text": "World"}]}
```

{"title": "Demo", "speakers": ["HOST", "EXPERT"], "dialogue": [{"speaker": "HOST", "text": "Hello"}, {"speaker": "EXPERT", "text": "World"}]}
'''


def test_document_parser_reads_text_file(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello from the document parser", encoding="utf-8")

    parser = DocumentParser(file_path)

    assert parser.parse() == "hello from the document parser"


def test_chunk_text_splits_long_content() -> None:
    text = "\n\n".join(["A" * 2000, "B" * 2000, "C" * 2000])

    chunks = chunk_text(text, max_chars=2500)

    assert len(chunks) >= 2
    assert all(len(chunk) <= 3000 for chunk in chunks)


def test_extract_pdf_reads_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Hello local podcast!")
    document.save(pdf_path)
    document.close()

    extracted = extract_pdf(str(pdf_path))

    assert "Hello local podcast!" in extracted


def test_dialogue_uses_llm_json() -> None:
    script = PodcastDialogue(FakeLLM()).build({"title": "Demo", "material": {"main_ideas": ["AI"]}})

    assert script["title"] == "Demo"
    assert script["speakers"] == ["HOST", "EXPERT"]
    assert len(script["dialogue"]) == 2


def test_dialogue_rejects_malformed_json() -> None:
    try:
        PodcastDialogue(FakeInvalidLLM()).build({"title": "Broken", "material": {"main_ideas": ["AI"]}})
        raise AssertionError("Expected ValueError for malformed dialogue JSON")
    except ValueError as exc:
        assert "dialogue list" in str(exc) or "Malformed dialogue JSON" in str(exc)


def test_dialogue_extracts_valid_json_from_raw_think_output() -> None:
    script = PodcastDialogue(FakeThinkLLM()).build({"title": "Demo", "material": {"main_ideas": ["AI"]}})

    assert script["title"] == "Demo"
    assert script["speakers"] == ["HOST", "EXPERT"]
    assert len(script["dialogue"]) == 2


def test_episode_plan_tracks_source_chunks() -> None:
    plan = build_episode_plan([
        {"facts": ["A fact"], "numbers": ["42"], "main_ideas": ["Idea 1"]},
    ], title="Demo")

    assert plan["material"]["facts"][0]["source"]["chunk"] == 1
    assert plan["material"]["numbers"][0]["source"]["chunk"] == 1


def test_assemble_audio_files_writes_podcast_wav(tmp_path: Path) -> None:
    audio_path = tmp_path / "seg1.wav"
    sf.write(audio_path, [0.0, 0.1, 0.2], 8000)
    out_path = tmp_path / "podcast.wav"

    assembled = assemble_audio_files([str(audio_path)], str(out_path))

    assert assembled == str(out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_run_pipeline_generates_audio_file(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("This is a short document with a clear point.", encoding="utf-8")

    class FakeTTS:
        def __init__(self):
            self.calls = []
            self.voice = "af_heart"

        def synthesize(self, text: str, output_path: str, voice: str | None = None) -> str:
            self.calls.append((text, output_path, voice))
            sf.write(output_path, [0.0, 0.1, 0.2], 8000)
            return output_path

    fake_tts = FakeTTS()
    monkeypatch.setattr("podcast.cli.KokoroTTS", lambda: fake_tts)
    monkeypatch.setattr("podcast.cli.assemble_audio_files", lambda files, out: str(Path(out).write_bytes(b"x") or Path(out)))

    result = run_pipeline(str(source), output_dir=tmp_path / "output", llm=FakeLLM(), max_chunks=1)

    assert result["title"] == "Demo"
    assert fake_tts.calls
    assert (tmp_path / "output" / "notes" / "podcast.wav").exists()


def test_run_pipeline_records_benchmark_metrics(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("This is a short document with a clear point.", encoding="utf-8")

    class FakeTTS:
        def __init__(self):
            self.voice = "af_heart"

        def synthesize(self, text: str, output_path: str, voice: str | None = None) -> str:
            sf.write(output_path, [0.0, 0.1, 0.2], 8000)
            return output_path

    monkeypatch.setattr("podcast.cli.KokoroTTS", lambda: FakeTTS())
    monkeypatch.setattr("podcast.cli.assemble_audio_files", lambda files, out: str(Path(out).write_bytes(b"x") or Path(out)))

    run_pipeline(str(source), output_dir=tmp_path / "output", llm=FakeLLM(), max_chunks=1)

    benchmark_path = tmp_path / "output" / "notes" / "benchmark.json"
    assert benchmark_path.exists()

    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    assert "total_time_seconds" in payload
    assert "steps" in payload
    assert payload["steps"]
    assert payload["total_time_seconds"] >= 0.0
