const state = { episodes: [], selected: -1, jobId: null };
const DEMO_MODE = true;
const $ = (id) => document.getElementById(id);
const audio = $("audio");
const playerPanel = $("playerPanel");

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return "0:00";
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}
function setStatus(message, kind = "") {
  const el = $("jobStatus"); el.textContent = message; el.className = `status ${kind}`;
}
function renderEpisodes() {
  const term = $("search").value.trim().toLowerCase();
  const list = $("episodeList"); list.innerHTML = "";
  const visible = state.episodes.filter((ep) => ep.title.toLowerCase().includes(term));
  if (!visible.length) { list.innerHTML = '<p class="empty">No podcasts match that search.</p>'; return; }
  const template = $("episodeTemplate");
  visible.forEach((episode) => {
    const node = template.content.firstElementChild.cloneNode(true);
    node.querySelector(".episode-title").textContent = episode.title;
    node.querySelector(".episode-meta").textContent = `${episode.language.toUpperCase()} · ${formatTime(episode.duration_seconds)}`;
    node.classList.toggle("active", state.selected >= 0 && state.episodes[state.selected].id === episode.id);
    node.addEventListener("click", () => selectEpisode(state.episodes.findIndex((item) => item.id === episode.id)));
    list.append(node);
  });
}
async function loadEpisodes(preserve = true) {
  const response = await fetch("/api/episodes");
  if (!response.ok) throw new Error("Could not load local episodes.");
  const selectedId = preserve && state.selected >= 0 ? state.episodes[state.selected].id : null;
  state.episodes = await response.json();
  state.selected = state.episodes.findIndex((item) => item.id === selectedId);
  renderEpisodes();
  if (state.selected === -1 && state.episodes.length) selectEpisode(0);
}
function selectEpisode(index) {
  if (index < 0 || index >= state.episodes.length) return;
  state.selected = index; const episode = state.episodes[index];
  audio.src = episode.audio_url; audio.load();
  $("nowTitle").textContent = episode.title; $("nowLanguage").textContent = episode.language;
  $("summary").textContent = episode.summary || "Audio and script are available for this episode.";
  $("progress").value = 0; $("progress").disabled = false; $("elapsed").textContent = "0:00"; $("duration").textContent = formatTime(episode.duration_seconds);
  renderEpisodes();
}
function moveEpisode(offset) { if (state.selected >= 0) selectEpisode((state.selected + offset + state.episodes.length) % state.episodes.length); }
$("playerToggle").addEventListener("click", () => {
  const collapsed = !playerPanel.classList.contains("collapsed");
  playerPanel.classList.toggle("collapsed", collapsed);
  $("summary").hidden = collapsed;
  $("playerToggle").textContent = collapsed ? "⌄" : "⌃";
  $("playerToggle").setAttribute("aria-expanded", String(!collapsed));
  $("playerToggle").setAttribute("aria-label", collapsed ? "Expand description" : "Collapse description");
});
$("play").addEventListener("click", async () => { if (!audio.src) return; if (audio.paused) await audio.play(); else audio.pause(); });
audio.addEventListener("play", () => { $("play").textContent = "Ⅱ"; $("play").setAttribute("aria-label", "Pause"); });
audio.addEventListener("pause", () => { $("play").textContent = "▶"; $("play").setAttribute("aria-label", "Play"); });
audio.addEventListener("timeupdate", () => { const percent = audio.duration ? (audio.currentTime / audio.duration) * 100 : 0; $("progress").value = percent; $("elapsed").textContent = formatTime(audio.currentTime); });
audio.addEventListener("loadedmetadata", () => { $("duration").textContent = formatTime(audio.duration); });
$("progress").addEventListener("input", (event) => { if (audio.duration) audio.currentTime = audio.duration * event.target.value / 100; });
$("back15").addEventListener("click", () => { audio.currentTime = Math.max(0, audio.currentTime - 15); });
$("forward15").addEventListener("click", () => { audio.currentTime = Math.min(audio.duration || Infinity, audio.currentTime + 15); });
$("previousEpisode").addEventListener("click", () => moveEpisode(-1)); $("nextEpisode").addEventListener("click", () => moveEpisode(1));
$("search").addEventListener("input", renderEpisodes); $("refresh").addEventListener("click", () => loadEpisodes().catch((err) => setStatus(err.message, "failed")));
$("sourceFile").addEventListener("change", (event) => { $("fileLabel").textContent = event.target.files[0]?.name || "Choose a PDF or text file"; });
$("episodeFormat").addEventListener("change", (event) => { const technical = event.target.value === "technical"; $("minutes").value = technical ? "3" : "15"; $("minutes").disabled = technical; });
async function pollJob() {
  if (!state.jobId) return;
  const response = await fetch(`/api/jobs/${state.jobId}`); const job = await response.json();
  if (job.status === "queued" || job.status === "running") { setStatus(job.status === "queued" ? "Queued" : "Making episode…", job.status); setTimeout(pollJob, 1500); return; }
  $("generate").disabled = false; state.jobId = null;
  if (job.status === "succeeded") { setStatus("Episode ready"); await loadEpisodes(false); const index = state.episodes.findIndex((ep) => ep.id === job.episode_id); if (index >= 0) selectEpisode(index); }
  else setStatus(job.error || "Generation failed", "failed");
}
$("generateForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (DEMO_MODE) { setStatus("Demo mode — generation is disabled"); return; }
  const file = $("sourceFile").files[0]; if (!file) return;
  try {
    $("generate").disabled = true; setStatus("Uploading…", "queued");
    const form = new FormData(); form.append("file", file); const upload = await fetch("/api/uploads", { method: "POST", body: form });
    if (!upload.ok) throw new Error((await upload.json()).detail || "Upload failed."); const { filename } = await upload.json();
    const response = await fetch("/api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ input_file: filename, language: $("language").value, llm: $("llm").value, tts: $("tts").value, target_minutes: Number($("minutes").value), research: $("research").checked, episode_format: $("episodeFormat").value }) });
    if (!response.ok) throw new Error((await response.json()).detail || "Could not start generation."); state.jobId = (await response.json()).id; pollJob();
  } catch (err) { $("generate").disabled = false; setStatus(err.message, "failed"); }
});
loadEpisodes().catch((err) => { $("episodeList").innerHTML = `<p class="empty">${err.message}</p>`; });
