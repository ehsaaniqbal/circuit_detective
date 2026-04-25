from __future__ import annotations

from circuit_detective.models import CircuitDetectiveAction
from circuit_detective.server.backend import FakeInductionBackend
from circuit_detective.server.circuit_detective_environment import CircuitDetectiveEnvironment


def make_env() -> CircuitDetectiveEnvironment:
    return CircuitDetectiveEnvironment(backend=FakeInductionBackend())


def make_phase2_env() -> CircuitDetectiveEnvironment:
    return CircuitDetectiveEnvironment(
        backend=FakeInductionBackend(),
        require_ablation=True,
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


def test_invalid_tool_gets_penalized() -> None:
    env = make_env()
    env.reset()
    observation = env.step(CircuitDetectiveAction(tool_name="does_not_exist"))

    assert observation.done is False
    assert observation.reward == -0.05
    assert "error" in observation.result
