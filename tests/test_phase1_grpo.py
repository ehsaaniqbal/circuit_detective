from __future__ import annotations

import json

from circuit_detective.phase1_grpo import CircuitDetectiveToolEnv
from circuit_detective.phase1_grpo import Phase2CircuitDetectiveToolEnv
from circuit_detective.phase1_grpo import PlantedCircuitToolEnv
from circuit_detective.phase1_grpo import consume_reward_trace, reset_reward_trace, reward_func
from circuit_detective.server.backend import FakeInductionBackend, Head, RandomizedPlantedCircuitBackend


def make_env() -> CircuitDetectiveToolEnv:
    return CircuitDetectiveToolEnv(backend_factory=FakeInductionBackend)


def make_phase2_env() -> Phase2CircuitDetectiveToolEnv:
    return Phase2CircuitDetectiveToolEnv(backend_factory=FakeInductionBackend)


def make_planted_env(seed: int = 17) -> PlantedCircuitToolEnv:
    return PlantedCircuitToolEnv(
        backend_factory=lambda: RandomizedPlantedCircuitBackend(seed=seed)
    )


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
