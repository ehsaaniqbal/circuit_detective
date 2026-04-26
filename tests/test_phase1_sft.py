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


class DummyTokenizer:
    def apply_chat_template(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        return "\n".join(message["content"] for message in messages)

    def __call__(self, text, add_special_tokens=False):  # type: ignore[no-untyped-def]
        return {"input_ids": text.split()}


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


def test_planted_lite_synthetic_trace_exposes_two_candidate_causal_chain() -> None:
    target, decoy = phase1_sft.planted_lite_heads_for_record(0)
    inspect_payload = json.loads(
        phase1_sft.synthetic_planted_lite_inspect_response(
            target_head=target,
            decoy_head=decoy,
        )
    )
    decoy_payload = json.loads(
        phase1_sft.synthetic_planted_lite_ablation_response(
            head_id=decoy,
            target_head=target,
            decoy_head=decoy,
            ablated_heads_so_far=[],
        )
    )
    target_payload = json.loads(
        phase1_sft.synthetic_planted_lite_ablation_response(
            head_id=target,
            target_head=target,
            decoy_head=decoy,
            ablated_heads_so_far=[decoy],
        )
    )

    assert inspect_payload["scenario_id"] == "planted_lite_causal_chain"
    assert inspect_payload["result"]["scores"][0]["head_id"] == decoy
    assert inspect_payload["result"]["scores"][1]["head_id"] == target
    assert inspect_payload["result"]["next_required_tool"] == "ablate_head"
    assert decoy_payload["result"]["best_ablated_head_so_far"] == decoy
    assert decoy_payload["result"]["must_submit"] is None
    assert decoy_payload["result"]["next_required_tool"] == "ablate_head"
    assert decoy_payload["result"]["terminal_action_required"] is False
    assert target_payload["result"]["best_ablated_head_so_far"] == target
    assert target_payload["result"]["must_submit"] == target
    assert target_payload["result"]["next_required_tool"] == "submit_circuit"
    assert target_payload["result"]["next_required_arguments"] == {"heads": [target]}
    assert target_payload["result"]["terminal_action_required"] is True


def test_planted_lite_sft_records_include_full_causal_chain() -> None:
    records = phase1_sft.build_sft_records(
        tokenizer=DummyTokenizer(),
        examples_per_prompt=1,
        target_head="L1H6",
        scenario="planted_lite",
    )

    assert records
    full_trace = records[0]["text"]
    assert len(records) == 72
    assert "inspect_induction_scores" in full_trace
    assert full_trace.count("<function=ablate_head>") == 2
    assert "best_ablated_head_so_far" in full_trace
    assert "next_required_tool" in full_trace
    assert "must_submit" in full_trace
    assert "<function=submit_circuit>" in full_trace
    assert any("One candidate remains; ablate it." in record["text"] for record in records)
    assert any("must_submit is now set." in record["text"] for record in records)
    assert any("Finish the episode now with submit_circuit." in record["text"] for record in records)
    assert any("Do not call another ablation" in record["text"] for record in records)
    assert any("Recovery case" in record["text"] for record in records)
    assert any("Recovery terminal case" in record["text"] for record in records)
    assert any("terminal_action_required is true" in record["text"] for record in records)
    assert any("Use next_required_arguments exactly" in record["text"] for record in records)


def test_planted_lite_sft_preflight_rejects_truncated_submit() -> None:
    records = [{"text": "one two <function=submit_circuit> three"}]

    try:
        phase1_sft.validate_sft_records_fit(
            records=records,
            tokenizer=DummyTokenizer(),
            max_seq_length=2,
            scenario="planted_lite",
        )
    except ValueError as exc:
        assert "truncated_submit" in str(exc)
    else:
        raise AssertionError("Expected planted-lite SFT preflight to reject truncated submit.")


def test_ioi_synthetic_trace_exposes_multi_head_target() -> None:
    inspect_payload = json.loads(phase1_sft.synthetic_ioi_inspect_response())
    ablation_payload = json.loads(phase1_sft.synthetic_ioi_ablation_response("L9H9"))

    ranked = [item["head_id"] for item in inspect_payload["result"]["scores"]]

    assert phase1_sft.ioi_target_heads() == ["L9H9", "L9H6", "L10H0"]
    assert set(phase1_sft.ioi_target_heads()).issubset(ranked)
    assert inspect_payload["scenario_id"] == "ioi_gpt2_small_name_mover"
    assert ablation_payload["result"]["causal_verified"] is True


def test_real_ioi_synthetic_trace_matches_observed_backend_ranking() -> None:
    inspect_payload = json.loads(
        phase1_sft.synthetic_ioi_inspect_response(scenario_id="ioi_gpt2_small_real")
    )
    ablation_payload = json.loads(
        phase1_sft.synthetic_ioi_ablation_response("L9H9", scenario_id="ioi_gpt2_small_real")
    )

    ranked = [item["head_id"] for item in inspect_payload["result"]["scores"]]

    assert ranked[0] == "L8H10"
    assert set(phase1_sft.ioi_target_heads()).issubset(ranked)
    assert phase1_sft.real_ioi_expert_ablation_heads()[0] == "L8H10"
    assert inspect_payload["scenario_id"] == "ioi_gpt2_small_real"
    assert ablation_payload["result"]["causal_delta_threshold"] == 0.01
    assert ablation_payload["result"]["causal_verified"] is True
