"""Data models for the Circuit Detective environment."""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class CircuitDetectiveAction(Action):
    """Generic tool-call action for the environment."""

    tool_name: str = Field(..., description="Tool to execute.")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-serializable tool arguments.",
    )


class CircuitDetectiveObservation(Observation):
    """Observation returned after each environment action."""

    summary: str = Field(default="", description="Short human-readable summary.")
    result: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured result payload for the last tool call.",
    )
    scenario_id: str = Field(default="", description="Active scenario identifier.")
    step_count: int = Field(default=0, description="Current episode step count.")
    remaining_budget: int = Field(default=0, description="Steps left in the episode.")
    available_tools: list[str] = Field(
        default_factory=list,
        description="Tools exposed in the current environment.",
    )
