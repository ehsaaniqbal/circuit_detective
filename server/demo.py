"""Judge-facing demo routes for the Circuit Detective Space."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

try:
    from ..models import CircuitDetectiveAction, CircuitDetectiveObservation
    from .backend import Head, PlantedLiteCausalChainBackend
    from .circuit_detective_environment import (
        CircuitDetectiveEnvironment,
        PLANTED_LITE_CAUSAL_DELTA_THRESHOLD,
    )
except ImportError:
    from models import CircuitDetectiveAction, CircuitDetectiveObservation
    from server.backend import Head, PlantedLiteCausalChainBackend
    from server.circuit_detective_environment import (
        CircuitDetectiveEnvironment,
        PLANTED_LITE_CAUSAL_DELTA_THRESHOLD,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO_SEED = 19
PHASE2_ARTIFACT_DIR = ROOT / "artifacts" / "planted_lite_naive_max_sft1536_grpo300_ctx1024"
PHASE1_ARTIFACT_DIR = ROOT / "artifacts" / "phase1_sft64_grpo200_a10g_large"


@dataclass
class DemoSession:
    env: CircuitDetectiveEnvironment
    transcript: list[dict[str, object]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


_SESSIONS: dict[str, DemoSession] = {}


def register_demo_routes(app: FastAPI) -> None:
    """Attach the demo UI and JSON helpers without disturbing OpenEnv routes."""

    @app.get("/", response_class=HTMLResponse)
    def demo_index() -> str:
        return DEMO_HTML

    @app.post("/demo/reset")
    def reset_demo(seed: int = DEFAULT_DEMO_SEED) -> dict[str, object]:
        return create_session(seed=seed)

    @app.post("/demo/step")
    def step_demo(payload: dict[str, object]) -> dict[str, object]:
        session_id = str(payload.get("session_id", ""))
        tool_name = str(payload.get("tool_name", ""))
        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            raise HTTPException(status_code=400, detail="arguments must be an object")
        return step_session(session_id=session_id, tool_name=tool_name, arguments=arguments)

    @app.get("/demo/baseline")
    def baseline_demo(seed: int = DEFAULT_DEMO_SEED) -> dict[str, object]:
        return run_policy_trace(policy="baseline", seed=seed)

    @app.get("/demo/protocol")
    def protocol_demo(seed: int = DEFAULT_DEMO_SEED) -> dict[str, object]:
        return run_policy_trace(policy="protocol", seed=seed)

    @app.get("/demo/results")
    def demo_results() -> dict[str, object]:
        return load_results_snapshot()


def make_planted_lite_env(seed: int = DEFAULT_DEMO_SEED) -> CircuitDetectiveEnvironment:
    return CircuitDetectiveEnvironment(
        backend=PlantedLiteCausalChainBackend(seed=seed),
        require_ablation=True,
        causal_delta_threshold=PLANTED_LITE_CAUSAL_DELTA_THRESHOLD,
    )


def create_session(*, seed: int = DEFAULT_DEMO_SEED) -> dict[str, object]:
    cleanup_sessions()
    env = make_planted_lite_env(seed=seed)
    observation = env.reset()
    session_id = str(uuid4())
    session = DemoSession(env=env)
    session.transcript.append(observation_event("reset", {}, observation))
    _SESSIONS[session_id] = session
    return demo_payload(session_id=session_id, session=session)


def step_session(
    *,
    session_id: str,
    tool_name: str,
    arguments: dict[str, object],
) -> dict[str, object]:
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown demo session. Reset the case.")
    if session.transcript and bool(session.transcript[-1].get("done")):
        raise HTTPException(status_code=400, detail="Episode is done. Reset the case.")
    observation = session.env.step(
        CircuitDetectiveAction(tool_name=tool_name, arguments=arguments)
    )
    session.transcript.append(observation_event(tool_name, arguments, observation))
    return demo_payload(session_id=session_id, session=session)


def run_policy_trace(*, policy: str, seed: int = DEFAULT_DEMO_SEED) -> dict[str, object]:
    payload = create_session(seed=seed)
    session_id = str(payload["session_id"])
    payload = step_session(
        session_id=session_id,
        tool_name="inspect_induction_scores",
        arguments={"top_k": 2},
    )
    candidates = [
        str(item["head_id"])
        for item in payload["transcript"][-1]["result"].get("scores", [])
        if isinstance(item, dict)
    ]
    if not candidates:
        return payload

    if policy == "baseline":
        return step_session(
            session_id=session_id,
            tool_name="submit_circuit",
            arguments={"heads": [candidates[0]]},
        )

    for head_id in candidates:
        head = Head.parse(head_id)
        payload = step_session(
            session_id=session_id,
            tool_name="ablate_head",
            arguments={"layer": head.layer, "head": head.head},
        )
    best_head = str(payload["best_ablated_head"] or candidates[-1])
    return step_session(
        session_id=session_id,
        tool_name="submit_circuit",
        arguments={"heads": [best_head]},
    )


def observation_event(
    tool_name: str,
    arguments: dict[str, object],
    observation: CircuitDetectiveObservation,
) -> dict[str, object]:
    return {
        "tool_name": tool_name,
        "arguments": arguments,
        "summary": observation.summary,
        "result": observation.result,
        "reward": observation.reward,
        "done": observation.done,
        "step_count": observation.step_count,
        "remaining_budget": observation.remaining_budget,
    }


def demo_payload(*, session_id: str, session: DemoSession) -> dict[str, object]:
    transcript = session.transcript
    latest = transcript[-1] if transcript else {}
    done = bool(latest.get("done"))
    submitted_heads = []
    score: dict[str, object] = {}
    phase2: dict[str, object] = {}
    if done:
        result = latest.get("result", {})
        if isinstance(result, dict):
            submitted_heads = list(result.get("submitted_heads", []))
            raw_score = result.get("score", {})
            raw_phase2 = result.get("phase2", {})
            score = raw_score if isinstance(raw_score, dict) else {}
            phase2 = raw_phase2 if isinstance(raw_phase2, dict) else {}

    return {
        "session_id": session_id,
        "scenario": "planted_lite_causal_chain",
        "headline": "The top-ranked head is a decoy. Ablation reveals causality.",
        "transcript": transcript,
        "candidates": candidate_rows(session),
        "best_ablated_head": best_ablated_head(transcript),
        "next_required_tool": latest_next_tool(transcript),
        "done": done,
        "submitted_heads": submitted_heads,
        "ground_truth_heads": session.env.ground_truth_heads() if done else [],
        "score": score,
        "phase2": phase2,
        "rubric": rubric(done=done, score=score, phase2=phase2, transcript=transcript),
    }


def candidate_rows(session: DemoSession) -> list[dict[str, object]]:
    heads = session.env.candidate_heads()
    rows = [{"head_id": head_id, "score": None, "behavior_delta": None} for head_id in heads]
    by_head = {str(row["head_id"]): row for row in rows}
    for event in session.transcript:
        result = event.get("result", {})
        if not isinstance(result, dict):
            continue
        for score in result.get("scores", []):
            if isinstance(score, dict):
                head_id = str(score.get("head_id", ""))
                if head_id in by_head:
                    by_head[head_id]["score"] = score.get("score")
        head_id = str(result.get("head_id", ""))
        if head_id in by_head:
            by_head[head_id]["behavior_delta"] = result.get("behavior_delta")
    return rows


def best_ablated_head(transcript: list[dict[str, object]]) -> str | None:
    best_head = None
    best_delta = float("-inf")
    for event in transcript:
        result = event.get("result", {})
        if not isinstance(result, dict):
            continue
        head_id = result.get("head_id")
        delta = result.get("behavior_delta")
        if head_id is not None and isinstance(delta, int | float) and float(delta) > best_delta:
            best_head = str(head_id)
            best_delta = float(delta)
    return best_head


def latest_next_tool(transcript: list[dict[str, object]]) -> str | None:
    for event in reversed(transcript):
        result = event.get("result", {})
        if isinstance(result, dict) and result.get("next_required_tool"):
            return str(result["next_required_tool"])
    return None


def rubric(
    *,
    done: bool,
    score: dict[str, object],
    phase2: dict[str, object],
    transcript: list[dict[str, object]],
) -> dict[str, object]:
    tools = [str(event["tool_name"]) for event in transcript]
    return {
        "inspected_scores": "inspect_induction_scores" in tools,
        "ran_ablation": "ablate_head" in tools,
        "submitted_circuit": "submit_circuit" in tools,
        "submitted_after_ablation": bool(phase2.get("ablate_submitted")),
        "causal_success": bool(phase2.get("causal_success")),
        "final_f1": score.get("f1", 0.0) if done else 0.0,
        "final_reward": transcript[-1].get("reward", 0.0) if transcript else 0.0,
    }


def load_results_snapshot() -> dict[str, object]:
    phase1 = load_metrics(PHASE1_ARTIFACT_DIR / "phase1_eval_metrics.json")
    phase2 = load_metrics(PHASE2_ARTIFACT_DIR / "phase1_eval_metrics.json")
    return {
        "phase1": summarize_metrics(phase1, "eval_after") if phase1 else None,
        "phase2": summarize_metrics(phase2, "eval_after") if phase2 else None,
        "phase2_status": "complete" if phase2 else "awaiting_final_run_artifact",
        "artifact_paths": {
            "phase1": str(PHASE1_ARTIFACT_DIR.relative_to(ROOT)),
            "phase2": str(PHASE2_ARTIFACT_DIR.relative_to(ROOT)),
        },
    }


def load_metrics(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_metrics(payload: dict[str, object], prefix: str) -> dict[str, object]:
    section = payload.get("after") or payload.get("eval_after") or payload
    if not isinstance(section, dict):
        return {}

    def get(name: str) -> object:
        return section.get(f"{prefix}_{name}", section.get(name))

    return {
        "success_rate": get("success_rate"),
        "submit_rate": get("submit_rate"),
        "causal_success_rate": get("causal_success_rate"),
        "terminal_ready_no_submit_rate": get("terminal_ready_no_submit_rate"),
        "mean_reward": get("mean_reward"),
        "rollouts": get("rollouts"),
    }


def cleanup_sessions(max_age_s: float = 3600.0) -> None:
    now = time.time()
    expired = [
        session_id
        for session_id, session in _SESSIONS.items()
        if now - session.created_at > max_age_s
    ]
    for session_id in expired:
        _SESSIONS.pop(session_id, None)


DEMO_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Circuit Detective Demo</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,650;9..144,800&family=IBM+Plex+Mono:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,650&display=swap');

    :root {
      --ink: #17201b;
      --muted: #637066;
      --paper: #f6f0de;
      --paper-strong: #fffaf0;
      --line: #d7c9a7;
      --accent: #b33f1f;
      --accent-2: #0f6b68;
      --good: #1f7a45;
      --bad: #ad3227;
      --shadow: 0 18px 45px rgba(45, 37, 16, 0.16);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Source Serif 4", Georgia, serif;
      background:
        radial-gradient(circle at 12% 12%, rgba(179, 63, 31, 0.16), transparent 28rem),
        radial-gradient(circle at 88% 8%, rgba(15, 107, 104, 0.16), transparent 30rem),
        linear-gradient(135deg, #efe0bd 0%, #f7f0df 52%, #e8d7ad 100%);
      min-height: 100vh;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(23, 32, 27, 0.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(23, 32, 27, 0.035) 1px, transparent 1px);
      background-size: 42px 42px;
      mask-image: linear-gradient(to bottom, black, transparent 85%);
    }

    header, main { width: min(1180px, calc(100vw - 28px)); margin: 0 auto; }
    header { padding: 34px 0 18px; }
    .eyebrow {
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
      letter-spacing: .14em;
      text-transform: uppercase;
      color: var(--accent-2);
      font-weight: 600;
    }
    h1 {
      font-family: "Fraunces", Georgia, serif;
      font-size: clamp(42px, 6vw, 86px);
      line-height: .91;
      letter-spacing: -0.055em;
      margin: 10px 0 14px;
      max-width: 920px;
    }
    .lede {
      max-width: 830px;
      font-size: 20px;
      line-height: 1.45;
      color: #334038;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1.35fr .9fr;
      gap: 16px;
      align-items: start;
      padding: 18px 0 36px;
    }
    .panel {
      background: rgba(255, 250, 240, 0.87);
      border: 1px solid rgba(113, 86, 34, 0.22);
      border-radius: 24px;
      box-shadow: var(--shadow);
      overflow: hidden;
      backdrop-filter: blur(12px);
    }
    .panel h2 {
      margin: 0;
      font-family: "Fraunces", Georgia, serif;
      font-size: 24px;
      letter-spacing: -0.03em;
    }
    .panel-head {
      padding: 18px 20px 12px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255, 252, 243, .96), rgba(246, 240, 222, .66));
    }
    .panel-body { padding: 18px 20px 20px; }
    .controls { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 0; }
    button {
      appearance: none;
      border: 1px solid #9f8551;
      background: #fff8e8;
      color: var(--ink);
      border-radius: 999px;
      padding: 9px 13px;
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: transform .12s ease, background .12s ease, border-color .12s ease;
    }
    button:hover { transform: translateY(-1px); background: #fff1cf; border-color: var(--accent); }
    button.primary { background: var(--accent); color: white; border-color: var(--accent); }
    button.secondary { background: var(--accent-2); color: white; border-color: var(--accent-2); }
    button:disabled { opacity: .45; cursor: not-allowed; transform: none; }

    table { width: 100%; border-collapse: collapse; font-family: "IBM Plex Mono", monospace; font-size: 13px; }
    th { text-align: left; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }
    td, th { padding: 10px 6px; border-bottom: 1px solid rgba(215, 201, 167, .85); }
    .chip {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 8px;
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
      font-weight: 600;
      background: #efe1bf;
      color: #3d321d;
    }
    .chip.good { background: #d8ebd2; color: var(--good); }
    .chip.bad { background: #f1d5ce; color: var(--bad); }
    .transcript { display: grid; gap: 10px; }
    .event {
      border: 1px solid rgba(107, 84, 41, .24);
      background: rgba(255, 253, 247, .78);
      border-radius: 16px;
      padding: 12px;
      animation: rise .28s ease both;
    }
    .event-title {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
      font-weight: 600;
      color: var(--accent-2);
      margin-bottom: 6px;
    }
    .event p { margin: 0 0 8px; line-height: 1.35; }
    pre {
      margin: 0;
      max-height: 170px;
      overflow: auto;
      background: #18231d;
      color: #f2ecd9;
      border-radius: 12px;
      padding: 10px;
      font-family: "IBM Plex Mono", monospace;
      font-size: 11px;
      line-height: 1.38;
      white-space: pre-wrap;
    }
    .rubric { display: grid; gap: 10px; }
    .rubric-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid rgba(215, 201, 167, .8);
      padding-bottom: 8px;
      font-size: 15px;
    }
    .metric {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: baseline;
      font-family: "IBM Plex Mono", monospace;
      padding: 8px 0;
      border-bottom: 1px solid rgba(215, 201, 167, .8);
    }
    .metric strong { font-size: 18px; }
    .note {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.4;
      margin-top: 12px;
    }
    .split { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .banner {
      margin: 18px 0 0;
      border-left: 5px solid var(--accent);
      padding: 12px 14px;
      background: rgba(255, 250, 240, .7);
      font-size: 16px;
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
      .split { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="eyebrow">OpenEnv Hackathon · Circuit Detective</div>
    <h1>Correlation points at a decoy. Intervention finds the circuit.</h1>
    <p class="lede">
      This demo is a controlled planted-lite mechanistic-interpretability task.
      The agent sees two plausible attention heads. The top score is intentionally misleading;
      only ablation identifies the causal head.
    </p>
    <div class="banner">
      Honest scope: this page demonstrates the environment and causal protocol. Final Phase 2 training metrics appear here once the HF Jobs artifact is uploaded.
    </div>
  </header>

  <main class="grid">
    <section class="panel">
      <div class="panel-head">
        <h2>Case File</h2>
        <div class="controls">
          <button class="primary" onclick="resetCase()">Reset Case</button>
          <button onclick="inspectScores()">Inspect</button>
          <button class="secondary" onclick="runProtocol()">Protocol Replay</button>
          <button onclick="runBaseline()">Baseline Replay</button>
        </div>
      </div>
      <div class="panel-body">
        <p class="note" id="next-hint">Reset the case to begin.</p>
        <table>
          <thead><tr><th>Head</th><th>Inspect score</th><th>Ablation delta</th></tr></thead>
          <tbody id="candidates"></tbody>
        </table>
        <div class="controls" id="candidate-actions"></div>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head"><h2>Lab Notebook</h2></div>
      <div class="panel-body">
        <div class="transcript" id="transcript"></div>
      </div>
    </section>

    <aside class="panel">
      <div class="panel-head"><h2>Verdict</h2></div>
      <div class="panel-body">
        <div class="rubric" id="rubric"></div>
        <p class="note" id="ground-truth"></p>
      </div>
      <div class="panel-head"><h2>Results Snapshot</h2></div>
      <div class="panel-body" id="results"></div>
    </aside>
  </main>

  <script>
    let state = null;

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options
      });
      if (!response.ok) throw new Error(await response.text());
      return await response.json();
    }

    function fmt(value) {
      if (value === null || value === undefined) return "—";
      if (typeof value === "number") return value.toFixed(value > 1 ? 2 : 3);
      return String(value);
    }

    function parseHead(headId) {
      const [layer, head] = headId.slice(1).split("H").map(Number);
      return { layer, head };
    }

    async function resetCase() {
      state = await api("/demo/reset", { method: "POST" });
      render();
    }

    async function inspectScores() {
      if (!state) return resetCase();
      state = await api("/demo/step", {
        method: "POST",
        body: JSON.stringify({
          session_id: state.session_id,
          tool_name: "inspect_induction_scores",
          arguments: { top_k: 2 }
        })
      });
      render();
    }

    async function ablate(headId) {
      const args = parseHead(headId);
      state = await api("/demo/step", {
        method: "POST",
        body: JSON.stringify({
          session_id: state.session_id,
          tool_name: "ablate_head",
          arguments: args
        })
      });
      render();
    }

    async function submit(headId) {
      state = await api("/demo/step", {
        method: "POST",
        body: JSON.stringify({
          session_id: state.session_id,
          tool_name: "submit_circuit",
          arguments: { heads: [headId] }
        })
      });
      render();
    }

    async function runBaseline() {
      state = await api("/demo/baseline");
      render();
    }

    async function runProtocol() {
      state = await api("/demo/protocol");
      render();
    }

    async function loadResults() {
      const results = await api("/demo/results");
      renderResults(results);
    }

    function render() {
      if (!state) return;
      document.getElementById("next-hint").innerHTML =
        state.done
          ? `<span class="chip ${state.rubric.causal_success ? "good" : "bad"}">${state.rubric.causal_success ? "Causal success" : "Failed causal protocol"}</span>`
          : `Next expected tool: <span class="chip">${state.next_required_tool || "choose from evidence"}</span>`;

      const candidateBody = document.getElementById("candidates");
      candidateBody.innerHTML = state.candidates.map(row => `
        <tr>
          <td><strong>${row.head_id}</strong></td>
          <td>${fmt(row.score)}</td>
          <td>${fmt(row.behavior_delta)}</td>
        </tr>
      `).join("");

      const actions = document.getElementById("candidate-actions");
      const top = state.candidates.find(row => row.score !== null);
      const best = state.best_ablated_head;
      actions.innerHTML = state.done ? "" : [
        ...state.candidates.map(row => `<button onclick="ablate('${row.head_id}')">Ablate ${row.head_id}</button>`),
        top ? `<button onclick="submit('${top.head_id}')">Submit top-ranked</button>` : "",
        best ? `<button class="secondary" onclick="submit('${best}')">Submit max-delta ${best}</button>` : ""
      ].join("");

      const transcript = document.getElementById("transcript");
      transcript.innerHTML = state.transcript.map((event, index) => `
        <div class="event">
          <div class="event-title">
            <span>${index}. ${event.tool_name}</span>
            <span>reward ${fmt(event.reward)}</span>
          </div>
          <p>${event.summary}</p>
          <pre>${JSON.stringify(event.result, null, 2)}</pre>
        </div>
      `).join("");

      const rubric = state.rubric;
      document.getElementById("rubric").innerHTML = Object.entries({
        "Inspected scores": rubric.inspected_scores,
        "Ran ablation": rubric.ran_ablation,
        "Submitted circuit": rubric.submitted_circuit,
        "Submitted ablated head": rubric.submitted_after_ablation,
        "Causal success": rubric.causal_success,
        "Final F1": fmt(rubric.final_f1),
        "Final reward": fmt(rubric.final_reward)
      }).map(([key, value]) => `
        <div class="rubric-row"><span>${key}</span><strong>${typeof value === "boolean" ? (value ? "yes" : "no") : value}</strong></div>
      `).join("");

      document.getElementById("ground-truth").textContent =
        state.done ? `Ground truth revealed after submit: ${state.ground_truth_heads.join(", ")}` : "";
    }

    function renderResults(results) {
      const phase2 = results.phase2;
      const phase1 = results.phase1;
      document.getElementById("results").innerHTML = `
        <div class="metric"><span>Phase 1 success</span><strong>${phase1 ? fmt(phase1.success_rate * 100) + "%" : "—"}</strong></div>
        <div class="metric"><span>Phase 2 status</span><strong>${results.phase2_status.replaceAll("_", " ")}</strong></div>
        <div class="metric"><span>Phase 2 causal success</span><strong>${phase2 && phase2.causal_success_rate !== null ? fmt(phase2.causal_success_rate * 100) + "%" : "pending"}</strong></div>
        <div class="metric"><span>Terminal no-submit</span><strong>${phase2 && phase2.terminal_ready_no_submit_rate !== null ? fmt(phase2.terminal_ready_no_submit_rate * 100) + "%" : "pending"}</strong></div>
        <p class="note">Artifacts are read from the Space repo when available; no metric is fabricated.</p>
      `;
    }

    resetCase();
    loadResults();
  </script>
</body>
</html>"""
