from __future__ import annotations

import json
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "phase1_sft.py"
SPEC = importlib.util.spec_from_file_location("phase1_sft", SCRIPT_PATH)
assert SPEC is not None
phase1_sft = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(phase1_sft)


def test_tool_call_renders_trl_xml_format() -> None:
    rendered = phase1_sft.tool_call("submit_circuit", {"heads": ["L1H6"]})

    assert "<tool_call>" in rendered
    assert "<function=submit_circuit>" in rendered
    assert "<parameter=heads>" in rendered
    assert '["L1H6"]' in rendered


def test_synthetic_inspect_response_exposes_target_head_first() -> None:
    payload = json.loads(phase1_sft.synthetic_inspect_response("L1H6"))

    assert payload["result"]["scores"][0]["head_id"] == "L1H6"
    assert payload["done"] is False


def test_planted_synthetic_inspect_uses_decoy_first_and_target_later() -> None:
    target, decoys = phase1_sft.planted_heads_for_record(0)

    rank_two_payload = json.loads(
        phase1_sft.synthetic_planted_inspect_response(
            target_head=target,
            decoy_heads=decoys,
            target_position=1,
        )
    )
    rank_three_payload = json.loads(
        phase1_sft.synthetic_planted_inspect_response(
            target_head=target,
            decoy_heads=decoys,
            target_position=2,
        )
    )

    assert rank_two_payload["scenario_id"] == "planted_circuit_arena"
    assert rank_two_payload["result"]["scores"][0]["head_id"] == decoys[0]
    assert rank_two_payload["result"]["scores"][0]["head_id"] != target
    assert rank_two_payload["result"]["scores"][1]["head_id"] == target
    assert rank_three_payload["result"]["scores"][2]["head_id"] == target


def test_planted_synthetic_ablation_marks_only_target_as_causal() -> None:
    target, decoys = phase1_sft.planted_heads_for_record(0)

    decoy_payload = json.loads(
        phase1_sft.synthetic_planted_ablation_response(
            head_id=decoys[0],
            is_target=False,
        )
    )
    target_payload = json.loads(
        phase1_sft.synthetic_planted_ablation_response(
            head_id=target,
            is_target=True,
        )
    )

    assert decoy_payload["result"]["causal_verified"] is False
    assert target_payload["result"]["causal_verified"] is True
    assert target_payload["result"]["behavior_delta"] > decoy_payload["result"]["behavior_delta"]


def test_ioi_synthetic_trace_exposes_multi_head_target() -> None:
    inspect_payload = json.loads(phase1_sft.synthetic_ioi_inspect_response())
    ablation_payload = json.loads(phase1_sft.synthetic_ioi_ablation_response("L9H9"))

    ranked = [item["head_id"] for item in inspect_payload["result"]["scores"]]

    assert phase1_sft.ioi_target_heads() == ["L9H9", "L9H6", "L10H0"]
    assert set(phase1_sft.ioi_target_heads()).issubset(ranked)
    assert inspect_payload["scenario_id"] == "ioi_gpt2_small_name_mover"
    assert ablation_payload["result"]["causal_verified"] is True
