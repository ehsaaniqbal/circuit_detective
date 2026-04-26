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

    payload = json.loads(
        phase1_sft.synthetic_planted_inspect_response(
            target_head=target,
            decoy_heads=decoys,
        )
    )

    assert payload["scenario_id"] == "planted_circuit_arena"
    assert payload["result"]["scores"][0]["head_id"] == decoys[0]
    assert payload["result"]["scores"][0]["head_id"] != target
    assert payload["result"]["scores"][2]["head_id"] == target


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
