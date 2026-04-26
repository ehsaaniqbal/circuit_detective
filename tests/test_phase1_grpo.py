from __future__ import annotations

import json

from circuit_detective.phase1_grpo import CircuitDetectiveToolEnv
from circuit_detective.phase1_grpo import CurriculumCircuitToolEnv
from circuit_detective.phase1_grpo import IOICircuitToolEnv
from circuit_detective.phase1_grpo import Phase2CircuitDetectiveToolEnv
from circuit_detective.phase1_grpo import PlantedCircuitToolEnv
from circuit_detective.phase1_grpo import PlantedLiteCircuitToolEnv
from circuit_detective.phase1_grpo import RealIOICircuitToolEnv
from circuit_detective.phase1_grpo import consume_reward_trace, reset_reward_trace, reward_func
from circuit_detective.server.backend import (
    FakeInductionBackend,
    Head,
    PlantedLiteCausalChainBackend,
    RandomizedPlantedCircuitBackend,
)


def make_env() -> CircuitDetectiveToolEnv:
    return CircuitDetectiveToolEnv(backend_factory=FakeInductionBackend)


def make_phase2_env() -> Phase2CircuitDetectiveToolEnv:
    return Phase2CircuitDetectiveToolEnv(backend_factory=FakeInductionBackend)


def make_planted_env(seed: int = 17) -> PlantedCircuitToolEnv:
    return PlantedCircuitToolEnv(
        backend_factory=lambda: RandomizedPlantedCircuitBackend(seed=seed)
    )


def make_planted_lite_env(seed: int = 17) -> PlantedLiteCircuitToolEnv:
    return PlantedLiteCircuitToolEnv(
        backend_factory=lambda: PlantedLiteCausalChainBackend(seed=seed)
    )


def make_ioi_env() -> IOICircuitToolEnv:
    return IOICircuitToolEnv()


def test_wrapper_reset_returns_json_observation() -> None:
    env = make_env()

    payload = json.loads(env.reset() or "{}")

    assert payload["scenario_id"] == "l1_induction_attn_only_2l"
    assert payload["done"] is False


def test_wrapper_public_methods_are_only_trl_surface() -> None:
    env = make_env()

    public_methods = {
        name
        for name in dir(env)
        if not name.startswith("_") and callable(getattr(env, name))
    }

    assert public_methods == {
        "ablate_head",
        "inspect_induction_scores",
        "list_tools",
        "reset",
        "run_probe",
        "submit_circuit",
    }


def test_wrapper_shapes_useful_intermediate_rewards() -> None:
    env = make_env()
    env.reset()

    env.list_tools()
    env.inspect_induction_scores(top_k=3)

    assert env.reward > 0.0
    assert env.done is False


def test_wrapper_penalizes_repeated_inspection() -> None:
    env = make_env()
    env.reset()

    env.inspect_induction_scores(top_k=3)
    before = env.reward
    env.inspect_induction_scores(top_k=3)

    assert env.reward < before


def test_wrapper_keeps_terminal_reward() -> None:
    env = make_env()
    env.reset()

    env.submit_circuit(["L1H3"])

    assert env.done is True
    assert env.reward > 0.8


def test_reward_func_records_rollout_diagnostics() -> None:
    reset_reward_trace()
    env = make_env()
    env.reset()

    env.inspect_induction_scores(top_k=3)
    rewards = reward_func([env])
    records = consume_reward_trace()

    assert rewards[0] == 0.0
    assert records[0]["used_inspect"] is True
    assert records[0]["submitted"] is False


def test_reward_func_penalizes_probe_only_without_submission() -> None:
    reset_reward_trace()
    env = make_env()
    env.reset()

    env.run_probe()
    rewards = reward_func([env])

    assert rewards[0] < -0.3


def test_reward_func_mildly_penalizes_wrong_submission() -> None:
    reset_reward_trace()
    env = make_env()
    env.reset()

    env.submit_circuit(["L0H0"])
    rewards = reward_func([env])

    assert -0.1 < rewards[0] < 0.1


def test_reward_func_rewards_correct_submission() -> None:
    reset_reward_trace()
    env = make_env()
    env.reset()

    env.inspect_induction_scores(top_k=3)
    env.submit_circuit(["L1H3"])
    rewards = reward_func([env])
    records = consume_reward_trace()

    assert rewards[0] > 1.0
    assert records[0]["submitted"] is True
    assert records[0]["correct"] is True


def test_phase2_rewards_ablation_verified_submission_more_than_lookup_submission() -> None:
    lookup_env = make_phase2_env()
    lookup_env.reset()
    lookup_env.inspect_induction_scores(top_k=3)
    lookup_env.submit_circuit(["L1H3"])

    causal_env = make_phase2_env()
    causal_env.reset()
    causal_env.inspect_induction_scores(top_k=3)
    causal_env.ablate_head(layer=1, head=3)
    causal_env.submit_circuit(["L1H3"])

    assert causal_env.reward > lookup_env.reward
    assert lookup_env.reward < 0.0


def test_phase2_no_submit_after_ablation_is_hard_failure() -> None:
    reset_reward_trace()
    env = make_phase2_env()
    env.reset()

    env.inspect_induction_scores(top_k=3)
    env.ablate_head(layer=1, head=3)
    rewards = reward_func([env])
    records = consume_reward_trace()

    assert rewards[0] == -1.0
    assert records[0]["submitted"] is False


def test_phase2_wrong_submission_is_not_mildly_rewarded() -> None:
    reset_reward_trace()
    env = make_phase2_env()
    env.reset()

    env.inspect_induction_scores(top_k=3)
    env.submit_circuit(["L0H0"])
    rewards = reward_func([env])

    assert rewards[0] < -0.2


def test_phase2_reward_func_records_causal_diagnostics() -> None:
    reset_reward_trace()
    env = make_phase2_env()
    env.reset()

    env.inspect_induction_scores(top_k=3)
    env.ablate_head(layer=1, head=3)
    env.submit_circuit(["L1H3"])
    rewards = reward_func([env])
    records = consume_reward_trace()

    assert rewards[0] > 1.0
    assert records[0]["causal_success"] is True
    assert records[0]["ablate_submitted"] is True
    assert records[0]["ablation_faithfulness"] > 0.0


def test_planted_wrapper_rewards_ablate_then_submit_true_target() -> None:
    reset_reward_trace()
    env = make_planted_env()
    env.reset()
    target = env.env.ground_truth_heads()[0]

    inspect_payload = json.loads(env.inspect_induction_scores(top_k=3))
    decoy = Head.parse(str(inspect_payload["result"]["scores"][0]["head_id"]))
    planted = Head.parse(target)

    assert decoy.head_id != target

    decoy_payload = json.loads(env.ablate_head(layer=decoy.layer, head=decoy.head))
    assert decoy_payload["result"]["causal_verified"] is False

    target_payload = json.loads(env.ablate_head(layer=planted.layer, head=planted.head))
    assert target_payload["result"]["causal_verified"] is True

    env.submit_circuit([target])
    rewards = reward_func([env])
    records = consume_reward_trace()

    assert rewards[0] > 1.0
    assert records[0]["causal_success"] is True
    assert records[0]["ablate_submitted"] is True


def test_planted_lite_wrapper_rewards_full_causal_chain_only() -> None:
    reset_reward_trace()
    env = make_planted_lite_env()
    env.reset()
    target = env.env.ground_truth_heads()[0]

    inspect_payload = json.loads(env.inspect_induction_scores(top_k=2))
    candidates = [str(item["head_id"]) for item in inspect_payload["result"]["scores"]]
    decoy = Head.parse(candidates[0])
    planted = Head.parse(target)

    env.ablate_head(layer=decoy.layer, head=decoy.head)
    env.ablate_head(layer=planted.layer, head=planted.head)
    env.submit_circuit([target])
    rewards = reward_func([env])
    records = consume_reward_trace()

    assert rewards[0] == 5.0
    assert records[0]["causal_success"] is True
    assert records[0]["ablate_submitted"] is True


def test_planted_lite_reward_ladder_keeps_grpo_variance_before_success() -> None:
    reset_reward_trace()
    inspect_only = make_planted_lite_env(seed=41)
    inspect_only.reset()
    inspect_only.inspect_induction_scores(top_k=2)

    one_ablation = make_planted_lite_env(seed=41)
    one_ablation.reset()
    inspect_payload = json.loads(one_ablation.inspect_induction_scores(top_k=2))
    decoy = Head.parse(str(inspect_payload["result"]["scores"][0]["head_id"]))
    one_ablation.ablate_head(layer=decoy.layer, head=decoy.head)

    two_ablation = make_planted_lite_env(seed=41)
    two_ablation.reset()
    inspect_payload = json.loads(two_ablation.inspect_induction_scores(top_k=2))
    candidates = [Head.parse(str(item["head_id"])) for item in inspect_payload["result"]["scores"]]
    for candidate in candidates:
        two_ablation.ablate_head(layer=candidate.layer, head=candidate.head)

    rewards = reward_func([inspect_only, one_ablation, two_ablation])

    assert rewards == [-1.2, -1.0, -0.8]


def test_planted_lite_wrong_submit_after_full_evidence_is_less_bad_than_no_evidence() -> None:
    reset_reward_trace()
    env = make_planted_lite_env(seed=43)
    env.reset()
    target = env.env.ground_truth_heads()[0]

    inspect_payload = json.loads(env.inspect_induction_scores(top_k=2))
    candidates = [Head.parse(str(item["head_id"])) for item in inspect_payload["result"]["scores"]]
    for candidate in candidates:
        env.ablate_head(layer=candidate.layer, head=candidate.head)
    wrong = next(candidate.head_id for candidate in candidates if candidate.head_id != target)
    env.submit_circuit([wrong])

    rewards = reward_func([env])

    assert rewards[0] == 0.6


def test_planted_lite_correct_submit_after_one_ablation_gets_bridge_credit() -> None:
    reset_reward_trace()
    env = make_planted_lite_env(seed=47)
    env.reset()
    target = Head.parse(env.env.ground_truth_heads()[0])

    env.inspect_induction_scores(top_k=2)
    env.ablate_head(layer=target.layer, head=target.head)
    env.submit_circuit([target.head_id])
    rewards = reward_func([env])

    assert rewards[0] == 1.5


def test_planted_lite_strict_renderer_compacts_terminal_guidance() -> None:
    env = make_planted_lite_env(seed=53)
    reset_payload = json.loads(env.reset())

    assert "available_tools" not in reset_payload
    assert "candidate_heads" not in reset_payload["result"]
    assert reset_payload["result"]["next_required_tool"] == "inspect_induction_scores"

    inspect_payload = json.loads(env.inspect_induction_scores(top_k=2))
    candidates = [
        Head.parse(str(item["head_id"]))
        for item in inspect_payload["result"]["scores"]
    ]
    for candidate in candidates:
        ablation_payload = json.loads(
            env.ablate_head(layer=candidate.layer, head=candidate.head)
        )

    assert "available_tools" not in ablation_payload
    assert ablation_payload["result"]["next_required_tool"] == "submit_circuit"
    assert ablation_payload["result"]["next_required_arguments"] == {
        "heads": [ablation_payload["result"]["best_ablated_head_so_far"]]
    }
    assert ablation_payload["result"]["terminal_action_required"] is True


def test_ioi_wrapper_handles_multi_head_submission_and_rubric() -> None:
    reset_reward_trace()
    env = make_ioi_env()
    env.reset()
    targets = env.env.ground_truth_heads()

    env.inspect_induction_scores(top_k=8)
    env.ablate_head(layer=9, head=9)
    env.submit_circuit(targets)
    rewards = reward_func([env])
    records = consume_reward_trace()

    assert rewards[0] > 1.0
    assert records[0]["scenario_id"] == "ioi_gpt2_small_name_mover"
    assert records[0]["correct"] is True
    assert records[0]["causal_success"] is True
    assert records[0]["rubric"]["final_answer_f1"] == 1.0
    assert records[0]["rubric"]["causal_validation"] == 1.0


def test_curriculum_wrapper_cycles_between_planted_and_ioi() -> None:
    CurriculumCircuitToolEnv._instance_counter = 0

    first = CurriculumCircuitToolEnv()
    second = CurriculumCircuitToolEnv()

    first_payload = json.loads(first.reset() or "{}")
    second_payload = json.loads(second.reset() or "{}")

    assert first_payload["scenario_id"] == "planted_circuit_arena"
    assert second_payload["scenario_id"] == "ioi_gpt2_small_name_mover"


def test_real_ioi_wrapper_uses_real_scenario_without_loading_model_on_reset() -> None:
    env = RealIOICircuitToolEnv()

    payload = json.loads(env.reset() or "{}")

    assert payload["scenario_id"] == "ioi_gpt2_small_real"
