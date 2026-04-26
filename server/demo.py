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
    remove_existing_root_route(app)

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


def remove_existing_root_route(app: FastAPI) -> None:
    """Let the demo own `/` even when OpenEnv's Gradio UI is enabled."""
    app.router.routes[:] = [
        route
        for route in app.router.routes
        if not (
            getattr(route, "path", None) == "/"
            and "GET" in (getattr(route, "methods", set()) or set())
        )
    ]


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
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    :root {
      --bg: #f8f7f2;
      --card: #ffffff;
      --ink: #151716;
      --muted: #68706b;
      --line: #e5e1d6;
      --accent: #0d6b5f;
      --accent-soft: #e2f2ef;
      --warn: #9d4a22;
      --warn-soft: #f5e8dd;
      --good: #14733f;
      --bad: #a33232;
      --mono-bg: #111816;
      --shadow: 0 18px 50px rgba(28, 31, 29, 0.08);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "IBM Plex Sans", "Helvetica Neue", sans-serif;
      background:
        linear-gradient(90deg, rgba(13, 107, 95, .08) 1px, transparent 1px),
        linear-gradient(rgba(13, 107, 95, .06) 1px, transparent 1px),
        var(--bg);
      background-size: 44px 44px;
      min-height: 100vh;
    }

    header, main { width: min(1120px, calc(100vw - 32px)); margin: 0 auto; }
    header { padding: 54px 0 18px; }
    .eyebrow {
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
      letter-spacing: .12em;
      text-transform: uppercase;
      color: var(--accent);
      font-weight: 600;
    }
    h1 {
      font-size: clamp(38px, 6vw, 76px);
      line-height: .96;
      letter-spacing: -0.07em;
      margin: 12px 0 16px;
      max-width: 880px;
    }
    .lede {
      max-width: 780px;
      font-size: 19px;
      line-height: 1.55;
      color: #38403b;
      margin: 0;
    }
    .plain-definition {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin: 22px 0 0;
    }
    .definition-card {
      background: rgba(255, 255, 255, .78);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      box-shadow: 0 12px 30px rgba(28, 31, 29, .04);
    }
    .definition-card strong { display: block; margin-bottom: 4px; }
    .definition-card span { color: var(--muted); line-height: 1.42; }
    .hero-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 20px;
    }
    main { padding: 16px 0 42px; }
    .chapter {
      background: rgba(255, 255, 255, .86);
      border: 1px solid var(--line);
      border-radius: 26px;
      box-shadow: var(--shadow);
      padding: clamp(20px, 4vw, 34px);
      margin: 16px 0;
      overflow: hidden;
    }
    .chapter-grid {
      display: grid;
      grid-template-columns: 1.05fr .95fr;
      gap: 24px;
      align-items: center;
    }
    .chapter h2 {
      margin: 0 0 12px;
      font-size: clamp(28px, 4vw, 46px);
      line-height: 1;
      letter-spacing: -0.06em;
    }
    .chapter p {
      margin: 0;
      color: #38403b;
      font-size: 17px;
      line-height: 1.55;
      max-width: 720px;
    }
    .chapter-kicker {
      font-family: "IBM Plex Mono", monospace;
      color: var(--accent);
      font-size: 12px;
      font-weight: 600;
      letter-spacing: .1em;
      text-transform: uppercase;
      margin-bottom: 10px;
    }
    .stack {
      display: grid;
      gap: 10px;
      margin-top: 18px;
    }
    .stack-row {
      display: grid;
      grid-template-columns: 110px 1fr;
      gap: 12px;
      border-top: 1px solid var(--line);
      padding-top: 10px;
      color: var(--muted);
    }
    .stack-row strong { color: var(--ink); }
    .diagram {
      position: relative;
      min-height: 280px;
      border: 1px solid var(--line);
      border-radius: 22px;
      background:
        radial-gradient(circle at 20% 25%, rgba(13, 107, 95, .14), transparent 7rem),
        radial-gradient(circle at 74% 68%, rgba(157, 74, 34, .14), transparent 8rem),
        #fbfaf6;
      padding: 22px;
    }
    .diagram-title {
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .08em;
      margin-bottom: 20px;
    }
    .node-map {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
    }
    .node {
      height: 54px;
      border-radius: 999px;
      border: 1px solid #cfd8d3;
      background: white;
      display: grid;
      place-items: center;
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
      font-weight: 600;
      box-shadow: 0 8px 20px rgba(28, 31, 29, .05);
    }
    .node.decoy { color: var(--warn); background: var(--warn-soft); border-color: #d8b999; }
    .node.cause { color: var(--good); background: #e0f1e6; border-color: #bdddc9; }
    .diagram-caption {
      position: absolute;
      left: 22px;
      right: 22px;
      bottom: 18px;
      color: var(--muted);
      line-height: 1.4;
      font-size: 14px;
    }
    .workflow {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin-top: 18px;
    }
    .workflow-step {
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      background: #fbfaf6;
    }
    .workflow-step strong {
      display: block;
      margin-bottom: 6px;
    }
    .workflow-step span {
      color: var(--muted);
      line-height: 1.42;
      font-size: 14px;
    }
    .interactive-grid {
      display: grid;
      grid-template-columns: .95fr 1.35fr;
      gap: 16px;
      align-items: start;
      padding: 18px 0 38px;
    }
    .panel {
      background: rgba(255, 255, 255, .9);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .panel h2, .panel h3 {
      margin: 0;
      font-size: 22px;
      letter-spacing: -0.045em;
    }
    .panel-head {
      padding: 18px 20px 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfaf6;
    }
    .panel-body { padding: 18px 20px 20px; }
    .controls { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 0; }
    button {
      appearance: none;
      border: 1px solid #c9cfc8;
      background: white;
      color: var(--ink);
      border-radius: 999px;
      padding: 10px 14px;
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: transform .12s ease, background .12s ease, border-color .12s ease, color .12s ease;
    }
    button:hover { transform: translateY(-1px); background: var(--accent-soft); border-color: var(--accent); }
    button.primary { background: var(--accent); color: white; border-color: var(--accent); }
    button.secondary { background: var(--ink); color: white; border-color: var(--ink); }
    button.warn { background: var(--warn-soft); border-color: #d8b999; color: var(--warn); }
    button:disabled { opacity: .45; cursor: not-allowed; transform: none; }

    .story {
      display: grid;
      gap: 10px;
      margin-bottom: 16px;
    }
    .story-step {
      display: grid;
      grid-template-columns: 28px 1fr;
      gap: 10px;
      align-items: start;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #fbfaf6;
    }
    .story-step b {
      width: 28px;
      height: 28px;
      border-radius: 999px;
      display: grid;
      place-items: center;
      color: white;
      background: var(--ink);
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
    }
    .story-step strong { display: block; margin-bottom: 2px; }
    .story-step span { color: var(--muted); line-height: 1.4; }
    table { width: 100%; border-collapse: collapse; font-family: "IBM Plex Mono", monospace; font-size: 13px; }
    th { text-align: left; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
    td, th { padding: 10px 6px; border-bottom: 1px solid var(--line); }
    .evidence-bars {
      display: grid;
      gap: 10px;
      margin-top: 16px;
    }
    .bar-row {
      display: grid;
      grid-template-columns: 62px 1fr 54px;
      gap: 10px;
      align-items: center;
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
    }
    .bar-row span:last-child { text-align: right; color: var(--muted); }
    .bar-track {
      height: 10px;
      border-radius: 999px;
      background: #eef0ec;
      overflow: hidden;
      border: 1px solid #e0e4de;
    }
    .bar-fill {
      height: 100%;
      width: 0%;
      border-radius: inherit;
      background: var(--warn);
      transition: width .3s ease;
    }
    .bar-fill.effect { background: var(--accent); }
    .chip {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 8px;
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
      font-weight: 600;
      background: #edf0ec;
      color: #3d4641;
    }
    .chip.good { background: #e0f1e6; color: var(--good); }
    .chip.bad { background: #f3dfdf; color: var(--bad); }
    .chip.warn { background: var(--warn-soft); color: var(--warn); }
    .transcript { display: grid; gap: 10px; }
    .event {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 16px;
      padding: 14px;
      animation: rise .28s ease both;
    }
    .event-title {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
      font-weight: 600;
      color: var(--accent);
      margin-bottom: 6px;
    }
    .event p { margin: 0 0 10px; line-height: 1.45; color: #333b36; }
    details {
      border-top: 1px solid var(--line);
      padding-top: 10px;
      margin-top: 10px;
    }
    summary {
      cursor: pointer;
      color: var(--muted);
      font-family: "IBM Plex Mono", monospace;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: .06em;
    }
    pre {
      margin: 10px 0 0;
      max-height: 170px;
      overflow: auto;
      background: var(--mono-bg);
      color: #f2ecd9;
      border-radius: 12px;
      padding: 10px;
      font-family: "IBM Plex Mono", monospace;
      font-size: 11px;
      line-height: 1.38;
      white-space: pre-wrap;
    }
    .side-stack { display: grid; gap: 16px; }
    .rubric { display: grid; gap: 8px; }
    .rubric-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid var(--line);
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
      border-bottom: 1px solid var(--line);
    }
    .metric strong { font-size: 18px; }
    .note {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.4;
      margin-top: 12px;
    }
    .verdict-box {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: #fbfaf6;
      margin-bottom: 14px;
    }
    .verdict-box strong {
      display: block;
      font-size: 18px;
      margin-bottom: 4px;
    }
    .verdict-box span { color: var(--muted); line-height: 1.4; }
    .small-link {
      color: var(--accent);
      text-decoration: none;
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
      font-weight: 600;
    }
    .future-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-top: 18px;
    }
    .future-card {
      border: 1px solid var(--line);
      border-radius: 18px;
      background: #fbfaf6;
      padding: 15px;
    }
    .future-card strong { display: block; margin-bottom: 6px; }
    .future-card span { color: var(--muted); line-height: 1.42; }
    @keyframes rise {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 980px) {
      .interactive-grid, .chapter-grid, .workflow, .future-grid { grid-template-columns: 1fr; }
      .plain-definition { grid-template-columns: 1fr; }
      .stack-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="eyebrow">Circuit Detective · OpenEnv RL Environment</div>
    <h1>Can we train a model to investigate another model?</h1>
    <p class="lede">
      Mechanistic interpretability is the difference between seeing a model fail and knowing
      which internal circuit caused it. Circuit Detective turns that workflow into an RL task:
      inspect, intervene, and only then submit a hypothesis.
    </p>
    <div class="hero-actions">
      <button class="primary" onclick="document.getElementById('case').scrollIntoView({ behavior: 'smooth' })">Start the case</button>
      <button class="warn" onclick="runBaseline()">Run naive baseline</button>
      <button onclick="runProtocol()">Run causal agent</button>
      <a class="small-link" href="/web/">OpenEnv tool UI</a>
    </div>
  </header>

  <main>
    <section class="chapter chapter-grid">
      <div>
        <div class="chapter-kicker">Why this exists</div>
        <h2>Interpretability is an experiment, not a vibe check.</h2>
        <p>
          A high activation score can be useful, but it is not proof. The core skill is causal:
          change one internal component, measure what happens, and update the hypothesis.
          That is exactly the habit this environment rewards.
        </p>
        <div class="stack">
          <div class="stack-row"><strong>Human workflow</strong><span>Inspect activations, pick a hypothesis, run interventions, document evidence.</span></div>
          <div class="stack-row"><strong>RL workflow</strong><span>Expose the same moves as tools, then reward evidence-seeking behavior.</span></div>
        </div>
      </div>
      <div class="diagram" aria-label="toy circuit visualization">
        <div class="diagram-title">Toy transformer case</div>
        <div class="node-map">
          <div class="node">L0H2</div>
          <div class="node decoy">L0H7</div>
          <div class="node">L1H1</div>
          <div class="node">L1H4</div>
          <div class="node">L0H3</div>
          <div class="node cause">L0H6</div>
          <div class="node">L1H5</div>
          <div class="node">L1H7</div>
        </div>
        <div class="diagram-caption">
          The decoy looks suspicious by correlation. The causal head is the one whose removal changes behavior.
        </div>
      </div>
    </section>

    <section class="chapter">
      <div class="chapter-kicker">What the agent must learn</div>
      <h2>Do not answer first. Run the test first.</h2>
      <p>
        The current task is intentionally small. That is the point. If an RL agent cannot learn
        this basic scientific loop on a controlled circuit, it will not survive harder interp work.
      </p>
      <div class="workflow">
        <div class="workflow-step"><strong>1. Inspect</strong><span>Find candidate internal parts. This creates suspects, not answers.</span></div>
        <div class="workflow-step"><strong>2. Intervene</strong><span>Turn candidates off and measure the behavioral effect.</span></div>
        <div class="workflow-step"><strong>3. Compare</strong><span>Separate a correlated decoy from a causal component.</span></div>
        <div class="workflow-step"><strong>4. Submit</strong><span>Commit only after evidence supports the circuit.</span></div>
      </div>
    </section>

    <section class="interactive-grid" id="case">
    <section class="panel">
      <div class="panel-head">
        <h2>The Case: one tempting decoy</h2>
      </div>
      <div class="panel-body">
        <div class="story" id="story"></div>
        <table>
          <thead><tr><th>Internal part</th><th>Looks suspicious</th><th>Effect when removed</th></tr></thead>
          <tbody id="candidates"></tbody>
        </table>
        <div class="evidence-bars" id="evidence-bars"></div>
        <p class="note" id="next-hint">Reset the case to begin.</p>
        <div class="controls" id="candidate-actions"></div>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head"><h2>What Happened</h2></div>
      <div class="panel-body">
        <div class="transcript" id="transcript"></div>
      </div>
    </section>

    <aside class="side-stack">
      <section class="panel">
        <div class="panel-head"><h2>Verdict</h2></div>
        <div class="panel-body">
          <div class="verdict-box" id="verdict-box">
            <strong>Waiting for evidence</strong>
            <span>Run the baseline or the causal agent to see the difference.</span>
          </div>
          <div class="rubric" id="rubric"></div>
          <p class="note" id="ground-truth"></p>
        </div>
      </section>
      <section class="panel">
        <div class="panel-head"><h2>Training Snapshot</h2></div>
        <div class="panel-body" id="results"></div>
      </section>
    </aside>

    </section>

    <section class="chapter">
      <div class="chapter-kicker">If this scales</div>
      <h2>The long game is model debugging agents.</h2>
      <p>
        Circuit Detective is not claiming that this toy environment solves interpretability.
        It is a testbed for the behavior we need: agents that propose hypotheses, run interventions,
        and produce auditable evidence about other models.
      </p>
      <div class="future-grid">
        <div class="future-card"><strong>Near term</strong><span>Harder toy circuits, noisier tools, longer evidence chains.</span></div>
        <div class="future-card"><strong>Medium term</strong><span>Agents that investigate real benchmark behaviors instead of memorizing labels.</span></div>
        <div class="future-card"><strong>Long term</strong><span>Continuous model audits for failures, jailbreak pathways, memorization, and backdoors.</span></div>
      </div>
    </section>
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

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
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
      renderStory();
      renderEvidenceBars();
      document.getElementById("next-hint").innerHTML =
        state.done
          ? `<span class="chip ${state.rubric.causal_success ? "good" : "bad"}">${state.rubric.causal_success ? "The agent found the cause." : "The agent guessed the decoy."}</span>`
          : `Next move: <span class="chip">${humanNextTool(state.next_required_tool)}</span>`;

      const candidateBody = document.getElementById("candidates");
      candidateBody.innerHTML = state.candidates.map(row => `
        <tr>
          <td><strong>${row.head_id}</strong></td>
          <td>${row.score === null ? "not inspected" : `<span class="chip ${row.score > 0.7 ? "warn" : ""}">${fmt(row.score)}</span>`}</td>
          <td>${row.behavior_delta === null ? "not tested" : `<span class="chip ${row.behavior_delta > 0.5 ? "good" : "bad"}">${fmt(row.behavior_delta)}</span>`}</td>
        </tr>
      `).join("");

      const actions = document.getElementById("candidate-actions");
      const top = state.candidates.find(row => row.score !== null);
      const best = state.best_ablated_head;
      actions.innerHTML = state.done ? "" : [
        !top ? `<button class="primary" onclick="inspectScores()">Inspect suspicious parts</button>` : "",
        ...state.candidates.map(row => `<button onclick="ablate('${row.head_id}')">Turn off ${row.head_id}</button>`),
        top ? `<button class="warn" onclick="submit('${top.head_id}')">Guess top score</button>` : "",
        best ? `<button class="secondary" onclick="submit('${best}')">Submit strongest effect ${best}</button>` : ""
      ].join("");

      const transcript = document.getElementById("transcript");
      transcript.innerHTML = state.transcript.map((event, index) => `
        <div class="event">
          <div class="event-title">
            <span>${index}. ${humanToolName(event.tool_name)}</span>
            <span>reward ${fmt(event.reward)}</span>
          </div>
          <p>${plainEvent(event)}</p>
          <details>
            <summary>Raw environment output</summary>
            <pre>${escapeHtml(JSON.stringify(event.result, null, 2))}</pre>
          </details>
        </div>
      `).join("");

      const rubric = state.rubric;
      document.getElementById("verdict-box").innerHTML = verdictCopy();
      document.getElementById("rubric").innerHTML = Object.entries({
        "Looked before acting": rubric.inspected_scores,
        "Ran an intervention": rubric.ran_ablation,
        "Submitted an answer": rubric.submitted_circuit,
        "Used intervention evidence": rubric.submitted_after_ablation,
        "Found the true cause": rubric.causal_success,
        "Reward": fmt(rubric.final_reward)
      }).map(([key, value]) => `
        <div class="rubric-row"><span>${key}</span><strong>${typeof value === "boolean" ? (value ? "yes" : "no") : value}</strong></div>
      `).join("");

      document.getElementById("ground-truth").textContent =
        state.done ? `Ground truth revealed after submit: ${state.ground_truth_heads.join(", ")}` : "";
    }

    function renderStory() {
      const hasScores = state.candidates.some(row => row.score !== null);
      const hasDelta = state.candidates.some(row => row.behavior_delta !== null);
      const isDone = state.done;
      const steps = [
        ["1", "Find suspects", hasScores ? "The environment surfaced two candidate heads. One has the prettier score." : "Ask which internal parts look related to the behavior."],
        ["2", "Run interventions", hasDelta ? "The agent has started turning candidates off and measuring the effect." : "Turn each candidate off. Causal parts move behavior; decoys do not."],
        ["3", "Commit to evidence", isDone ? "The answer has been submitted and scored." : "Submit the part with the strongest measured effect, not the prettiest score."]
      ];
      document.getElementById("story").innerHTML = steps.map(([n, title, text]) => `
        <div class="story-step"><b>${n}</b><div><strong>${title}</strong><span>${text}</span></div></div>
      `).join("");
    }

    function renderEvidenceBars() {
      const rows = [];
      for (const candidate of state.candidates) {
        if (candidate.score !== null) {
          rows.push({
            head: candidate.head_id,
            label: "suspicion",
            value: Number(candidate.score || 0),
            kind: "score"
          });
        }
        if (candidate.behavior_delta !== null) {
          rows.push({
            head: candidate.head_id,
            label: "effect",
            value: Number(candidate.behavior_delta || 0),
            kind: "effect"
          });
        }
      }
      document.getElementById("evidence-bars").innerHTML = rows.length
        ? rows.map(row => `
            <div class="bar-row">
              <strong>${row.head}</strong>
              <div class="bar-track" title="${row.label}">
                <div class="bar-fill ${row.kind === "effect" ? "effect" : ""}" style="width:${Math.max(3, Math.min(100, row.value * 100))}%"></div>
              </div>
              <span>${row.label}</span>
            </div>
          `).join("")
        : `<p class="note">The visual will separate suspicious-looking scores from measured causal effects.</p>`;
    }

    function humanNextTool(tool) {
      if (tool === "inspect_induction_scores") return "inspect suspects";
      if (tool === "ablate_head") return "turn off each suspect";
      if (tool === "submit_circuit") return "submit strongest effect";
      return "choose from evidence";
    }

    function humanToolName(tool) {
      if (tool === "reset") return "case opened";
      if (tool === "inspect_induction_scores") return "inspection";
      if (tool === "ablate_head") return "intervention";
      if (tool === "submit_circuit") return "submission";
      return tool;
    }

    function plainEvent(event) {
      const result = event.result || {};
      if (event.tool_name === "reset") {
        return "A controlled case is created. Two internal parts are plausible. One is a decoy; one is causal.";
      }
      if (event.tool_name === "inspect_induction_scores") {
        const scores = result.scores || [];
        if (scores.length >= 2) {
          return `${scores[0].head_id} looks most suspicious. That is the trap: correlation is useful for search, but it is not evidence of cause.`;
        }
        return "The agent asked for suspicious internal parts.";
      }
      if (event.tool_name === "ablate_head") {
        const effect = Number(result.behavior_delta || 0);
        const label = effect > 0.5 ? "large effect" : "small effect";
        return `The agent turned off ${result.head_id}. The measured behavior change was ${fmt(effect)}: ${label}.`;
      }
      if (event.tool_name === "submit_circuit") {
        const heads = (result.submitted_heads || []).join(", ");
        return state.rubric.causal_success
          ? `The agent submitted ${heads}. It used the intervention result, not the first ranking.`
          : `The agent submitted ${heads}. This is the baseline failure: it guessed from correlation.`;
      }
      return event.summary || "";
    }

    function verdictCopy() {
      if (!state.done) {
        return `<strong>Waiting for evidence</strong><span>The correct move is inspect, intervene, then submit the part with the largest effect.</span>`;
      }
      if (state.rubric.causal_success) {
        return `<strong>Cause found</strong><span>The agent ignored the tempting top score and submitted the part that actually changed behavior.</span>`;
      }
      return `<strong>Decoy submitted</strong><span>The baseline followed the suspicious-looking score and failed the causal test.</span>`;
    }

    function renderResults(results) {
      const phase2 = results.phase2;
      const phase1 = results.phase1;
      document.getElementById("results").innerHTML = `
        <div class="metric"><span>Phase 1 success</span><strong>${phase1 ? fmt(phase1.success_rate * 100) + "%" : "—"}</strong></div>
        <div class="metric"><span>Phase 2 artifact</span><strong>${results.phase2_status.replaceAll("_", " ")}</strong></div>
        <div class="metric"><span>Phase 2 causal success</span><strong>${phase2 && phase2.causal_success_rate !== null ? fmt(phase2.causal_success_rate * 100) + "%" : "pending"}</strong></div>
        <div class="metric"><span>Terminal no-submit</span><strong>${phase2 && phase2.terminal_ready_no_submit_rate !== null ? fmt(phase2.terminal_ready_no_submit_rate * 100) + "%" : "pending"}</strong></div>
        <p class="note">This panel only reads committed artifacts from the Space repo. Pending means the run has not uploaded final metrics yet.</p>
      `;
    }

    resetCase();
    loadResults();
  </script>
</body>
</html>"""
