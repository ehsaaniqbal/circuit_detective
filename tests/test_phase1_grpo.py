from __future__ import annotations

import json

from circuit_detective.phase1_grpo import CircuitDetectiveToolEnv
from circuit_detective.server.backend import FakeInductionBackend


def make_env() -> CircuitDetectiveToolEnv:
    return CircuitDetectiveToolEnv(backend_factory=FakeInductionBackend)


def test_wrapper_reset_returns_json_observation() -> None:
    env = make_env()

    payload = json.loads(env.reset() or "{}")

    assert payload["scenario_id"] == "l1_induction_attn_only_2l"
    assert payload["done"] is False


def test_wrapper_ignores_positive_intermediate_rewards() -> None:
    env = make_env()
    env.reset()

    env.list_tools()

    assert env.reward == 0.0
    assert env.done is False


def test_wrapper_keeps_terminal_reward() -> None:
    env = make_env()
    env.reset()

    env.submit_circuit(["L1H3"])

    assert env.done is True
    assert env.reward > 0.8
