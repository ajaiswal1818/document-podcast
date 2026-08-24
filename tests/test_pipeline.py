import json
from pathlib import Path

import fitz
import soundfile as sf

from podcast.cli import run_pipeline
from podcast.document.parser import DocumentParser, chunk_text, extract_pdf
from podcast.podcast.dialogue import PodcastDialogue
from podcast.podcast.editor import StoryEditor
from podcast.podcast.planner import build_episode_plan, build_story_blueprint, translate_technical_terms
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


def test_translate_technical_terms_to_plain_english() -> None:
    class FakeTranslatorLLM:
        available = True

        def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
            return '{"plain_english_summary": "This means the immune system is reacting more strongly than before.", "facts": ["The immune response is stronger than before."], "main_ideas": ["The body is reacting more strongly."]}'

    result = translate_technical_terms(FakeTranslatorLLM(), {
        "facts": ["The cytokine signaling pathway increased in the immune response."],
        "main_ideas": ["Immune response increased."],
    })

    assert "plain_english_summary" in result
    assert "immune" in result["plain_english_summary"].lower()
    assert result["facts"][0]


def test_story_blueprint_has_audience_and_teaching_layers() -> None:
    class FakeStoryLLM:
        available = True

        def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
            return '{"story": {"central_question": "Why does response improve?", "hook": "A small change changes outcomes.", "problem": "People do not understand why it matters.", "tension": "The science looks complex.", "turning_point": "The trial changed the response curve.", "resolution": "The mechanism is clearer.", "big_takeaway": "The result matters for adoption."}, "audience": {"what_they_already_know": ["a treatment exists"], "what_will_confuse_them": ["what the number means"], "what_they_need_to_remember": ["response improved"], "why_they_should_care": "It changes how quickly people can act."}, "teaching": [{"concept": "response rate", "explanation": "How many patients improved.", "analogy": "It is like conversion rate.", "source": {"chunk": 1}}]}'

    blueprint = build_story_blueprint(FakeStoryLLM(), [{"facts": ["A trial showed a response increase."], "main_ideas": ["The response improved."]}], title="Demo")

    assert "story" in blueprint
    assert blueprint["story"]["central_question"]
    assert blueprint["audience"]["what_will_confuse_them"]
    assert blueprint["teaching"][0]["analogy"]


def test_story_editor_flags_unstructured_podcast() -> None:
    editor = StoryEditor(None)
    script = {"title": "Demo", "dialogue": [{"speaker": "HOST", "text": "Today we talk about X."}, {"speaker": "EXPERT", "text": "The study showed a result."}]}
    verdict = editor.review(script, {"story": {"central_question": "Why does it matter?"}})

    assert "approved" in verdict
    assert isinstance(verdict["approved"], bool)
    assert verdict["failed_checks"]

    regen_prompt = editor.build_regeneration_prompt(script, {"story": {"central_question": "Why does it matter?"}}, verdict)
    assert "question" in regen_prompt.lower()


def test_story_editor_requires_story_arc_and_takeaway() -> None:
    editor = StoryEditor(None)
    script = {
        "title": "Demo",
        "dialogue": [
            {"speaker": "HOST", "text": "Why does this matter?"},
            {"speaker": "EXPERT", "text": "It is a biological response."},
        ],
    }
    verdict = editor.review(script, {
        "story": {"central_question": "Why does it matter?"},
        "audience": {"why_they_should_care": "It changes decision-making."},
    })

    assert "story_has_arc" in verdict["checks"]
    assert "story_has_arc" in verdict["failed_checks"]
    assert "takeaway_is_clear" in verdict["failed_checks"]


def test_conversation_controller_uses_stateful_turns() -> None:
    from podcast.agents.conversational_agent import ConversationalAgent
    from podcast.conversation.controller import ConversationController, ConversationState

    state = ConversationState("demo topic")
    state.add_turn("A", "Why does this matter?")
    state.add_turn("B", "Because the pattern shows a real shift.")

    agent = ConversationalAgent("A", "curious", FakeLLM())
    response = agent.respond(state, {"main_ideas": ["The shift matters"]})

    assert response["speaker"] == "A"
    assert response["text"]
    assert "metadata" in response

    controller = ConversationController(FakeLLM(), max_turns=2)
    script = controller.run({"title": "Demo", "material": {"main_ideas": ["The shift matters"]}})

    assert script["speakers"] == ["A", "B"]
    assert len(script["dialogue"]) >= 2


def test_run_pipeline_uses_stateful_controller_for_dialogue(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("This is a short document with a clear point.", encoding="utf-8")

    calls: list[str] = []

    class FakeController:
        def __init__(self, llm, *, max_turns=6, agent_names=("A", "B")) -> None:
            calls.append("init")

        def run(self, material=None):
            calls.append("run")
            return {
                "title": "Demo",
                "speakers": ["A", "B"],
                "dialogue": [
                    {"speaker": "A", "text": "Why does this matter?"},
                    {"speaker": "B", "text": "Because the pattern changes the decision."},
                    {"speaker": "A", "text": "So the real takeaway is the shift in meaning?"},
                    {"speaker": "B", "text": "Exactly, and that is why it matters."},
                ],
            }

    class FakeTTS:
        def __init__(self):
            self.voice = "af_heart"

        def synthesize(self, text: str, output_path: str, voice: str | None = None) -> str:
            sf.write(output_path, [0.0, 0.1, 0.2], 8000)
            return output_path

    class FailingSceneWriter:
        def __init__(self, llm) -> None:
            pass

        def build(self, plan, *, target_words=2000):
            raise ValueError("scene generation unavailable")

    monkeypatch.setattr("podcast.cli.SceneScriptWriter", FailingSceneWriter)
    monkeypatch.setattr("podcast.cli.ConversationController", FakeController)
    monkeypatch.setattr("podcast.cli.KokoroTTS", lambda: FakeTTS())
    monkeypatch.setattr("podcast.cli.assemble_audio_files", lambda files, out: str(Path(out).write_bytes(b"x") or Path(out)))

    result = run_pipeline(str(source), output_dir=tmp_path / "output", llm=FakeLLM(), max_chunks=1, research=False)

    assert "run" in calls
    assert len(result["dialogue"]) >= 4


def test_conversational_agent_prompts_for_reaction_before_explaining() -> None:
    from podcast.agents.conversational_agent import ConversationalAgent
    from podcast.conversation.controller import ConversationState

    class CaptureLLM:
        available = True
        captured = {}

        def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
            self.captured["system"] = system
            self.captured["user"] = user
            return '{"speech": "Wait, really?", "intent": "react", "topic": "demo topic", "new_information": false, "question_for_other_agent": "", "needs_research": false, "confidence": 0.9, "conversation_direction": "react", "emotion": "surprised", "should_continue_topic": true}'

    llm = CaptureLLM()
    agent = ConversationalAgent("A", "curious", llm)
    state = ConversationState("demo topic")
    state.add_turn("B", "The pattern changes the decision.")

    agent.respond(state, {"main_ideas": ["The pattern matters"]})

    assert "React to what the other person just said" in llm.captured["system"]
    assert "Do NOT always explain" in llm.captured["system"]


def test_conversation_state_allows_longer_podcast_blocks() -> None:
    from podcast.conversation.controller import ConversationState

    state = ConversationState("demo topic")
    for i in range(16):
        state.add_turn("A" if i % 2 == 0 else "B", f"Turn {i + 1} keeps the conversation moving.")

    assert len(state.turns) == 16


def test_conversational_agent_hides_internal_metadata_from_prompt() -> None:
    from podcast.agents.conversational_agent import ConversationalAgent
    from podcast.conversation.controller import ConversationState

    class CaptureLLM:
        available = True
        captured = {}

        def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
            self.captured["system"] = system
            self.captured["user"] = user
            return '{"speech": "Wait, really?", "intent": "react", "topic": "demo topic", "new_information": false, "question_for_other_agent": "", "needs_research": false, "confidence": 0.9, "conversation_direction": "react", "emotion": "surprised", "should_continue_topic": true}'

    llm = CaptureLLM()
    agent = ConversationalAgent("A", "curious", llm)
    state = ConversationState("demo topic")
    state.add_turn("B", "The pattern changes the decision.")

    agent.respond(state, {
        "main_ideas": [{"value": "The pattern matters", "source": {"chunk": 1}}],
        "facts": [{"value": "The trial showed a clear shift.", "source": {"chunk": 2}}],
    })

    prompt_text = llm.captured["user"].lower()
    assert "\"value\"" not in prompt_text
    assert "\"chunk\"" not in prompt_text
    assert "\"source\"" not in prompt_text
    assert "evidence you can draw on" in prompt_text
    assert "the pattern matters" in prompt_text
    assert "clear shift" in prompt_text


def test_research_agent_fetches_real_web_evidence(monkeypatch) -> None:
    from podcast.research.researcher import ResearchAgent

    class FakeResponse:
        def __init__(self, payload):
            self.payload = json.dumps(payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self.payload

    def fake_urlopen(request, timeout=10):
        assert "diagnostic+mismatch" in request.full_url
        return FakeResponse({
            "resultList": {
                "result": [
                    {
                        "title": "Diagnostic mismatch in clinical practice",
                        "abstractText": "The mismatch pattern is often caused by a delayed clue.",
                        "doi": "10.1000/mismatch",
                        "isOpenAccess": "N",
                        "pubYear": "2024",
                        "authorString": "Doe J",
                    }
                ]
            }
        })

    monkeypatch.setattr("podcast.research.researcher.urlopen", fake_urlopen)

    researcher = ResearchAgent()
    result = researcher.lookup("diagnostic mismatch", {"main_ideas": ["The pattern matters"]})

    assert result["topic"] == "diagnostic mismatch"
    assert result["findings"]
    assert result["findings"][0]["url"] == "https://doi.org/10.1000/mismatch"
    assert result["findings"][0]["source_title"] == "Diagnostic mismatch in clinical practice"
    assert "delayed clue" in result["findings"][0]["claim"]


def test_scene_writer_builds_multi_section_script() -> None:
    from podcast.podcast.scenes import SceneScriptWriter

    class FakeSceneLLM:
        available = True

        def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
            if "outline" in user.lower() and "running_metaphor" in user:
                return (
                    '{"running_metaphor": "a locked room mystery", "sections": ['
                    '{"title": "Cold open", "goal": "start in the scene", "beats": ["the belly"], "target_words": 100},'
                    '{"title": "Landing it", "goal": "close", "beats": [], "target_words": 100}]}'
                )
            return (
                '{"dialogue": [{"speaker": "HOST", "text": "Her belly was massive, and yet she felt no pain at all."},'
                '{"speaker": "EXPERT", "text": "Right. And that contradiction is exactly where this mystery starts."}]}'
            )

    writer = SceneScriptWriter(FakeSceneLLM())
    script = writer.build({"title": "Demo", "story": {}, "teaching": [], "material": {"main_ideas": ["The case"]}}, target_words=200)

    assert script["speakers"] == ["HOST", "EXPERT"]
    assert len(script["dialogue"]) == 4
    assert script["outline"]["running_metaphor"] == "a locked room mystery"
    assert all(turn["speaker"] in {"HOST", "EXPERT"} for turn in script["dialogue"])


def test_research_agent_ranks_results_by_title_overlap() -> None:
    from podcast.research.researcher import ResearchAgent

    agent = ResearchAgent()
    query = "Chronic ascites systemic lupus erythematosus case report"
    good = agent._title_overlap(query, "Chronic ascites as the initial presentation of systemic lupus erythematosus")
    bad = agent._title_overlap(query, "Proceedings of the 32nd European Paediatric Rheumatology Congress")

    assert good > 0.5
    assert bad < 0.3


def test_research_agent_stores_sources_for_agent_requests() -> None:
    from podcast.research.researcher import ResearchAgent

    researcher = ResearchAgent()
    result = researcher.lookup("diagnostic mismatch", {"main_ideas": ["The pattern matters"]})

    assert result["topic"] == "diagnostic mismatch"
    assert result["sources"]
    assert result["sources"][0]["source"]


def test_render_evidence_for_agent_omits_metadata_fields() -> None:
    from podcast.conversation.knowledge import render_evidence_for_agent

    evidence = [
        {"claim": "The patient had recurrent ascites.", "source": {"chunk": 4, "page": 12}, "id": "C17"},
        {"value": "Initial treatment did not resolve the issue.", "source": {"chunk": 7}, "id": "C18"},
    ]

    rendered = render_evidence_for_agent(evidence)
    assert "recurrent ascites" in rendered.lower()
    assert "initial treatment" in rendered.lower()
    assert "chunk" not in rendered.lower()
    assert "source" not in rendered.lower()
    assert "page" not in rendered.lower()


def test_speech_sanitizer_rejects_internal_metadata_patterns() -> None:
    from podcast.conversation.knowledge import sanitize_speech_text

    bad_text = "According to source chunk 4, the patient had recurrent ascites."
    ok_text = "The patient had recurrent ascites despite the initial treatment."

    assert sanitize_speech_text(ok_text) == ok_text
    try:
        sanitize_speech_text(bad_text)
        raise AssertionError("Expected metadata-laced speech to be rejected")
    except ValueError:
        pass


def test_run_pipeline_regenerates_when_story_editor_rejects(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("This is a short document with a clear point.", encoding="utf-8")

    class FakeRegeneratingLLM(FakeLLM):
        def __init__(self):
            self.calls = 0

        def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
            self.calls += 1
            if self.calls == 1:
                return '{"title": "Demo", "speakers": ["HOST", "EXPERT"], "dialogue": [{"speaker": "HOST", "text": "Today we talk about X."}, {"speaker": "EXPERT", "text": "The study showed a result."}]}'
            return '{"title": "Demo", "speakers": ["HOST", "EXPERT"], "dialogue": [{"speaker": "HOST", "text": "What problem are we trying to solve here?"}, {"speaker": "EXPERT", "text": "The treatment improved response by 37%, which means more patients improved than under standard care."}]}'

    class FakeTTS:
        def __init__(self):
            self.voice = "af_heart"

        def synthesize(self, text: str, output_path: str, voice: str | None = None) -> str:
            sf.write(output_path, [0.0, 0.1, 0.2], 8000)
            return output_path

    llm = FakeRegeneratingLLM()
    monkeypatch.setattr("podcast.cli.KokoroTTS", lambda: FakeTTS())
    monkeypatch.setattr("podcast.cli.assemble_audio_files", lambda files, out: str(Path(out).write_bytes(b"x") or Path(out)))

    run_pipeline(str(source), output_dir=tmp_path / "output", llm=llm, max_chunks=1, research=False)

    assert llm.calls >= 2


def test_tts_backend_factory_uses_selected_backend() -> None:
    from podcast.tts import get_tts_backend

    assert get_tts_backend("kokoro").__class__.__name__ == "KokoroTTS"
    assert get_tts_backend("vibevoice").__class__.__name__ == "VibeVoiceTTS"


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

    result = run_pipeline(str(source), output_dir=tmp_path / "output", llm=FakeLLM(), max_chunks=1, research=False)

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

    run_pipeline(str(source), output_dir=tmp_path / "output", llm=FakeLLM(), max_chunks=1, research=False)

    benchmark_path = tmp_path / "output" / "notes" / "benchmark.json"
    assert benchmark_path.exists()

    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    assert "total_time_seconds" in payload
    assert "steps" in payload
    assert payload["steps"]
    assert payload["total_time_seconds"] >= 0.0
