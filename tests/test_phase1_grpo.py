from __future__ import annotations

import json

from circuit_detective.phase1_grpo import CircuitDetectiveToolEnv
from circuit_detective.phase1_grpo import consume_reward_trace, reset_reward_trace, reward_func
from circuit_detective.server.backend import FakeInductionBackend


def make_env() -> CircuitDetectiveToolEnv:
    return CircuitDetectiveToolEnv(backend_factory=FakeInductionBackend)


def test_wrapper_reset_returns_json_observation() -> None:
    env = make_env()

    payload = json.loads(env.reset() or "{}")

    assert payload["scenario_id"] == "l1_induction_attn_only_2l"
    assert payload["done"] is False


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

    assert rewards[0] > 0.0
    assert records[0]["used_inspect"] is True
    assert records[0]["submitted"] is False


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
