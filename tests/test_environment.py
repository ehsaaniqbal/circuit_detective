from __future__ import annotations

from circuit_detective.models import CircuitDetectiveAction
from circuit_detective.server.backend import (
    FakeInductionBackend,
    Head,
    PublishedIOICircuitBackend,
    RandomizedPlantedCircuitBackend,
    RealIOITransformerLensBackend,
)
from circuit_detective.server.circuit_detective_environment import (
    CircuitDetectiveEnvironment,
    IOI_CAUSAL_DELTA_THRESHOLD,
    PLANTED_CAUSAL_DELTA_THRESHOLD,
)


def make_env() -> CircuitDetectiveEnvironment:
    return CircuitDetectiveEnvironment(backend=FakeInductionBackend())


def make_phase2_env() -> CircuitDetectiveEnvironment:
    return CircuitDetectiveEnvironment(
        backend=FakeInductionBackend(),
        require_ablation=True,
    )


def make_planted_env(seed: int = 7) -> CircuitDetectiveEnvironment:
    return CircuitDetectiveEnvironment(
        backend=RandomizedPlantedCircuitBackend(seed=seed),
        require_ablation=True,
        causal_delta_threshold=PLANTED_CAUSAL_DELTA_THRESHOLD,
    )


def make_ioi_env() -> CircuitDetectiveEnvironment:
    return CircuitDetectiveEnvironment(
        backend=PublishedIOICircuitBackend(),
        require_ablation=True,
        causal_delta_threshold=IOI_CAUSAL_DELTA_THRESHOLD,
    )


def test_reset_exposes_budget_and_tools() -> None:
    env = make_env()
    observation = env.reset()

    assert observation.scenario_id == "l1_induction_attn_only_2l"
    assert observation.remaining_budget == 12
    assert "inspect_induction_scores" in observation.available_tools


def test_list_tools_is_valid_nonterminal_step() -> None:
    env = make_env()
    env.reset()
    observation = env.step(CircuitDetectiveAction(tool_name="list_tools"))

    assert observation.done is False
    assert observation.reward == 0.01
    assert "tools" in observation.result


def test_submit_exact_match_terminates_with_positive_reward() -> None:
    env = make_env()
    env.reset()
    observation = env.step(
        CircuitDetectiveAction(
            tool_name="submit_circuit",
            arguments={"heads": ["L1H3"]},
        )
    )

    assert observation.done is True
    assert observation.reward > 0.8
    assert observation.result["score"]["f1"] == 1.0


def test_phase2_correct_submit_without_ablation_is_partial_credit() -> None:
    env = make_phase2_env()
    observation = env.reset()

    assert observation.scenario_id == "l2_ablation_required"
    observation = env.step(
        CircuitDetectiveAction(
            tool_name="submit_circuit",
            arguments={"heads": ["L1H3"]},
        )
    )

    assert observation.done is True
    assert observation.result["score"]["f1"] == 1.0
    assert observation.result["phase2"]["causal_success"] is False
    assert observation.reward < 0.5


def test_phase2_correct_submit_after_ablation_gets_full_credit() -> None:
    env = make_phase2_env()
    env.reset()
    env.step(
        CircuitDetectiveAction(
            tool_name="ablate_head",
            arguments={"layer": 1, "head": 3},
        )
    )
    observation = env.step(
        CircuitDetectiveAction(
            tool_name="submit_circuit",
            arguments={"heads": ["L1H3"]},
        )
    )

    assert observation.done is True
    assert observation.result["phase2"]["ablate_submitted"] is True
    assert observation.result["phase2"]["causal_success"] is True
    assert observation.reward > 0.8


def test_planted_backend_ranks_decoy_above_target_but_ablation_reveals_target() -> None:
    backend = RandomizedPlantedCircuitBackend(seed=11)

    target = backend.ground_truth_heads()[0]
    scores = backend.inspect_induction_scores(top_k=3)
    decoy = scores[0].head

    assert decoy != target
    assert target in [score.head for score in scores]
    assert backend.ablate_head(decoy).behavior_delta < PLANTED_CAUSAL_DELTA_THRESHOLD
    assert backend.ablate_head(target).behavior_delta >= PLANTED_CAUSAL_DELTA_THRESHOLD


def test_planted_env_requires_ablation_of_true_target_not_top_decoy() -> None:
    env = make_planted_env(seed=13)
    observation = env.reset()
    target = env.ground_truth_heads()[0]

    assert observation.scenario_id == "planted_circuit_arena"
    assert observation.remaining_budget == 10

    observation = env.step(
        CircuitDetectiveAction(
            tool_name="inspect_induction_scores",
            arguments={"top_k": 3},
        )
    )
    top_head = str(observation.result["scores"][0]["head_id"])
    assert top_head != target

    top = Head.parse(top_head)
    observation = env.step(
        CircuitDetectiveAction(
            tool_name="ablate_head",
            arguments={"layer": top.layer, "head": top.head},
        )
    )
    assert observation.result["causal_verified"] is False

    planted = Head.parse(target)
    observation = env.step(
        CircuitDetectiveAction(
            tool_name="ablate_head",
            arguments={"layer": planted.layer, "head": planted.head},
        )
    )
    assert observation.result["causal_verified"] is True

    observation = env.step(
        CircuitDetectiveAction(
            tool_name="submit_circuit",
            arguments={"heads": [target]},
        )
    )
    assert observation.done is True
    assert observation.result["phase2"]["causal_success"] is True
    assert observation.reward > 0.8


def test_ioi_env_scores_multi_head_name_mover_submission() -> None:
    env = make_ioi_env()
    observation = env.reset()
    targets = env.ground_truth_heads()

    assert observation.scenario_id == "ioi_gpt2_small_name_mover"
    assert targets == ["L9H9", "L9H6", "L10H0"]

    observation = env.step(
        CircuitDetectiveAction(
            tool_name="inspect_induction_scores",
            arguments={"top_k": 8},
        )
    )
    ranked = [str(item["head_id"]) for item in observation.result["scores"]]
    assert set(targets).issubset(ranked)

    head = Head.parse(targets[0])
    observation = env.step(
        CircuitDetectiveAction(
            tool_name="ablate_head",
            arguments={"layer": head.layer, "head": head.head},
        )
    )
    assert observation.result["causal_verified"] is True

    observation = env.step(
        CircuitDetectiveAction(
            tool_name="submit_circuit",
            arguments={"heads": targets},
        )
    )
    assert observation.done is True
    assert observation.result["score"]["f1"] == 1.0
    assert observation.result["phase2"]["causal_success"] is True


def test_real_ioi_backend_is_isolated_transformerlens_scenario() -> None:
    backend = RealIOITransformerLensBackend()

    try:
        assert backend.scenario_id == "ioi_gpt2_small_real"
        assert backend.max_steps == 12
        assert backend._client.worker_path.name == "ioi_worker.py"
    finally:
        backend.close()


def test_invalid_tool_gets_penalized() -> None:
    env = make_env()
    env.reset()
    observation = env.step(CircuitDetectiveAction(tool_name="does_not_exist"))

    assert observation.done is False
    assert observation.reward == -0.05
    assert "error" in observation.result
