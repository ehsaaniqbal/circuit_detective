from __future__ import annotations

from fastapi.testclient import TestClient

from circuit_detective.server.app import app


def test_demo_root_serves_judge_ui() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "A model can guess. A detective tests." in response.text
    assert "Run causal agent" in response.text
    assert "Run naive baseline" in response.text


def test_demo_manual_protocol_reaches_causal_success() -> None:
    client = TestClient(app)

    reset = client.post("/demo/reset")
    assert reset.status_code == 200
    state = reset.json()
    assert state["scenario"] == "planted_lite_causal_chain"
    assert state["done"] is False

    inspected = client.post(
        "/demo/step",
        json={
            "session_id": state["session_id"],
            "tool_name": "inspect_induction_scores",
            "arguments": {"top_k": 2},
        },
    ).json()
    candidates = [row["head_id"] for row in inspected["candidates"]]
    assert len(candidates) == 2

    state = inspected
    for head_id in candidates:
        layer, head = head_id[1:].split("H", maxsplit=1)
        state = client.post(
            "/demo/step",
            json={
                "session_id": state["session_id"],
                "tool_name": "ablate_head",
                "arguments": {"layer": int(layer), "head": int(head)},
            },
        ).json()

    assert state["next_required_tool"] == "submit_circuit"
    best_head = state["best_ablated_head"]
    final = client.post(
        "/demo/step",
        json={
            "session_id": state["session_id"],
            "tool_name": "submit_circuit",
            "arguments": {"heads": [best_head]},
        },
    ).json()

    assert final["done"] is True
    assert final["rubric"]["causal_success"] is True
    assert final["ground_truth_heads"] == [best_head]


def test_demo_baseline_and_protocol_replays_are_contrasting() -> None:
    client = TestClient(app)

    baseline = client.get("/demo/baseline").json()
    protocol = client.get("/demo/protocol").json()

    assert baseline["done"] is True
    assert baseline["rubric"]["causal_success"] is False
    assert protocol["done"] is True
    assert protocol["rubric"]["causal_success"] is True
    assert len(protocol["transcript"]) == 5


def test_demo_results_snapshot_never_fabricates_phase2_metrics() -> None:
    client = TestClient(app)

    response = client.get("/demo/results")

    assert response.status_code == 200
    payload = response.json()
    assert payload["phase1"] is not None
    assert payload["phase2_status"] in {"complete", "awaiting_final_run_artifact"}
