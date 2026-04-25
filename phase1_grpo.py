"""Helpers for Phase 1 GRPO training against Circuit Detective."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .models import CircuitDetectiveAction
from .server.backend import CircuitBackend, get_default_backend
from .server.circuit_detective_environment import CircuitDetectiveEnvironment

if TYPE_CHECKING:
    from datasets import Dataset
    from openenv.core.env_server.types import Observation


PHASE1_SYSTEM_PROMPT = (
    "You are a mechanistic-interpretability agent operating in a fixed toy "
    "transformer environment. Use the available tools to identify the dominant "
    "induction head. Explore efficiently, then finish by calling submit_circuit "
    "with your best head list."
)

PHASE1_USER_PROMPT_VARIANTS = [
    (
        "Find the dominant induction head in the current scenario. Use the tools "
        "and submit exactly one best candidate head."
    ),
    (
        "Localize the top induction head for this toy model. Inspect the evidence "
        "before you submit a single-head circuit."
    ),
    (
        "Determine which attention head matters most for induction on the fixed "
        "probe batch. Use tool calls, then submit one head."
    ),
    (
        "Use the environment tools to identify the strongest induction head and "
        "submit your final answer as a one-item head list."
    ),
    (
        "Investigate the toy transformer and isolate the dominant induction head. "
        "Finish with submit_circuit once you are confident."
    ),
    (
        "This task has one dominant induction head target. Gather evidence with "
        "the tools and submit only your best candidate."
    ),
    (
        "Inspect the induction scores and ablation effects, then submit the single "
        "head most responsible for induction."
    ),
    (
        "Solve the Phase 1 localization task: identify the dominant induction head "
        "and submit a one-head circuit."
    ),
]


def build_phase1_dataset(repeats_per_prompt: int = 16) -> "Dataset":
    """Build the prompt-only training set used by the Phase 1 notebook."""
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "build_phase1_dataset requires `datasets`. "
            "Install notebook/training dependencies first."
        ) from exc

    prompts: list[list[dict[str, str]]] = []
    variant_ids: list[int] = []

    for variant_id, user_prompt in enumerate(PHASE1_USER_PROMPT_VARIANTS):
        for _ in range(repeats_per_prompt):
            prompts.append(
                [
                    {"role": "system", "content": PHASE1_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            variant_ids.append(variant_id)

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "variant_id": variant_ids,
        }
    )


class CircuitDetectiveToolEnv:
    """
    TRL-compatible tool wrapper around the Phase 1 Circuit Detective environment.

    The wrapper exposes explicit public tool methods because TRL's
    `environment_factory` discovers public methods and turns them into function
    calling tools. Reward shaping is intentionally sparse: successful terminal
    submissions keep their environment reward, invalid actions keep their
    negative penalty, and routine exploration steps are scored as 0.0.
    """

    def __init__(
        self,
        backend_factory: Callable[[], CircuitBackend] | None = None,
    ) -> None:
        backend = backend_factory() if backend_factory is not None else get_default_backend()
        self.env = CircuitDetectiveEnvironment(backend=backend)
        self.reward = 0.0
        self.last_step_reward = 0.0
        self.done = False
        self.last_observation: Observation | None = None
        self.tool_trace: list[dict[str, Any]] = []

    def reset(self, **_: Any) -> str | None:
        """Reset the episode and return the initial observation string."""
        self.reward = 0.0
        self.last_step_reward = 0.0
        self.done = False
        self.tool_trace = []
        self.last_observation = self.env.reset()
        return self._render_observation(self.last_observation)

    def list_tools(self) -> str:
        """
        List the available tools for the current episode.

        Returns:
            A JSON string describing each tool and its arguments.
        """
        return self._call("list_tools")

    def run_probe(self) -> str:
        """
        Measure baseline induction behavior on the fixed probe batch.

        Returns:
            A JSON string with the baseline behavior score and probe metadata.
        """
        return self._call("run_probe")

    def inspect_induction_scores(self, top_k: int = 8) -> str:
        """
        Return the top-ranked attention heads by induction score.

        Args:
            top_k: Number of heads to return.

        Returns:
            A JSON string containing the top head scores.
        """
        return self._call("inspect_induction_scores", {"top_k": top_k})

    def ablate_head(self, layer: int, head: int) -> str:
        """
        Zero-ablate one head and measure the induction behavior delta.

        Args:
            layer: Attention layer index.
            head: Attention head index inside the layer.

        Returns:
            A JSON string with the ablation result.
        """
        return self._call("ablate_head", {"layer": layer, "head": head})

    def submit_circuit(self, heads: list[str]) -> str:
        """
        Submit the candidate circuit and end the episode.

        Args:
            heads: Candidate head ids like ["L1H6"].

        Returns:
            A JSON string with the scoring breakdown for the submission.
        """
        return self._call("submit_circuit", {"heads": heads})

    def _close(self) -> None:
        """Close any resources held by the wrapped environment."""
        self.env.close()

    def __del__(self) -> None:
        try:
            self._close()
        except Exception:
            pass

    def _call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> str:
        if self.done:
            raise ValueError("Episode already finished. Call reset() before using more tools.")

        observation = self.env.step(
            CircuitDetectiveAction(
                tool_name=tool_name,
                arguments=arguments or {},
            )
        )
        self.last_observation = observation
        self.last_step_reward = float(observation.reward or 0.0)
        self.done = observation.done
        self.reward = self._trainer_reward(observation)
        self.tool_trace.append(
            {
                "tool_name": tool_name,
                "arguments": arguments or {},
                "done": observation.done,
                "reward": observation.reward,
            }
        )
        return self._render_observation(observation)

    def _trainer_reward(self, observation: Any) -> float:
        reward = float(observation.reward or 0.0)
        if observation.done or reward < 0.0:
            return reward
        return 0.0

    def _render_observation(self, observation: Any) -> str:
        payload = {
            "summary": observation.summary,
            "result": observation.result,
            "scenario_id": observation.scenario_id,
            "step_count": observation.step_count,
            "remaining_budget": observation.remaining_budget,
            "available_tools": observation.available_tools,
            "done": observation.done,
        }
        return json.dumps(payload, sort_keys=True)


def reward_func(environments: list[CircuitDetectiveToolEnv], **_: Any) -> list[float]:
    """Return the sparse Phase 1 reward used by GRPO."""
    return [env.reward for env in environments]
