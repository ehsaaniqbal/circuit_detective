"""Client for the Circuit Detective OpenEnv environment."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CircuitDetectiveAction, CircuitDetectiveObservation


class CircuitDetectiveEnv(
    EnvClient[CircuitDetectiveAction, CircuitDetectiveObservation, State]
):
    """WebSocket client for a running Circuit Detective environment server."""

    def _step_payload(self, action: CircuitDetectiveAction) -> dict[str, Any]:
        return {
            "tool_name": action.tool_name,
            "arguments": action.arguments,
        }

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[CircuitDetectiveObservation]:
        obs_data = payload.get("observation", {})
        observation = CircuitDetectiveObservation(
            summary=obs_data.get("summary", ""),
            result=obs_data.get("result") or {},
            scenario_id=obs_data.get("scenario_id", ""),
            step_count=obs_data.get("step_count", 0),
            remaining_budget=obs_data.get("remaining_budget", 0),
            available_tools=obs_data.get("available_tools", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
