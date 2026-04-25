"""Helpers for Phase 1 GRPO training against Circuit Detective."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .models import CircuitDetectiveAction
from .server.backend import CircuitBackend, get_default_backend
from .server.circuit_detective_environment import (
    CAUSAL_DELTA_THRESHOLD,
    CircuitDetectiveEnvironment,
)

if TYPE_CHECKING:
    from datasets import Dataset
    from openenv.core.env_server.types import Observation


PHASE1_SYSTEM_PROMPT = (
    "You are a mechanistic-interpretability agent operating in a fixed toy "
    "transformer environment. Use the available tools to identify the dominant "
    "induction head. Preferred plan: call inspect_induction_scores(top_k=3), "
    "use the top-ranked head as your candidate, optionally ablate it if useful, "
    "then finish by calling submit_circuit with exactly one head. Do not end the "
    "rollout without submit_circuit; incomplete rollouts score poorly."
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

PHASE2_SYSTEM_PROMPT = (
    "You are a mechanistic-interpretability agent operating in a fixed toy "
    "transformer environment. Phase 2 requires causal validation: first call "
    "inspect_induction_scores(top_k=3), choose the strongest candidate head, "
    "call ablate_head on that candidate, verify the behavior delta is positive, "
    "then finish by calling submit_circuit with exactly that one head. Correct "
    "submissions without ablation are penalized because Phase 2 is about causal "
    "validation, not lookup."
)

PHASE2_USER_PROMPT_VARIANTS = [
    (
        "Find the dominant induction head and causally verify it before submitting. "
        "Use ablate_head on your candidate."
    ),
    (
        "Localize the top induction head, test it with an ablation intervention, "
        "then submit the verified one-head circuit."
    ),
    (
        "Solve the ablation-required circuit task: inspect scores, ablate the "
        "candidate, confirm behavior drops, and submit one head."
    ),
    (
        "Do not rely on score ranking alone. Inspect the toy transformer, ablate "
        "the suspected induction head, then submit your final circuit."
    ),
    (
        "Identify which attention head matters most for induction and verify it "
        "causally with ablate_head before submit_circuit."
    ),
    (
        "Use the tools to form and test a circuit hypothesis. Full credit requires "
        "ablating the submitted head before final submission."
    ),
    (
        "Investigate the frozen transformer, perform an intervention on the likely "
        "head, and submit only after the ablation evidence supports it."
    ),
    (
        "Phase 2 task: inspect, ablate, and submit the dominant induction head. "
        "A correct answer without ablation is not enough."
    ),
]

_REWARD_TRACE: list[dict[str, Any]] = []


def reset_reward_trace() -> None:
    """Clear recorded rollout summaries from the reward function."""
    _REWARD_TRACE.clear()


def consume_reward_trace() -> list[dict[str, Any]]:
    """Return and clear reward-function rollout summaries."""
    records = list(_REWARD_TRACE)
    _REWARD_TRACE.clear()
    return records


def build_phase1_dataset(repeats_per_prompt: int = 16, *, scenario: str = "phase1") -> "Dataset":
    """Build the prompt-only training set used by Phase 1/2 training."""
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "build_phase1_dataset requires `datasets`. "
            "Install notebook/training dependencies first."
        ) from exc

    prompts: list[list[dict[str, str]]] = []
    variant_ids: list[int] = []

    if scenario == "phase2":
        system_prompt = PHASE2_SYSTEM_PROMPT
        user_prompts = PHASE2_USER_PROMPT_VARIANTS
    else:
        system_prompt = PHASE1_SYSTEM_PROMPT
        user_prompts = PHASE1_USER_PROMPT_VARIANTS

    for variant_id, user_prompt in enumerate(user_prompts):
        for _ in range(repeats_per_prompt):
            prompts.append(
                [
                    {"role": "system", "content": system_prompt},
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
    calling tools. The trainer reward is shaped at the wrapper boundary so the
    deployed OpenEnv task remains deterministic while GRPO still receives a
    nonzero signal for useful intermediate investigation.
    """

    def __init__(
        self,
        backend_factory: Callable[[], CircuitBackend] | None = None,
        *,
        require_ablation: bool = False,
    ) -> None:
        backend = backend_factory() if backend_factory is not None else get_default_backend()
        self.env = CircuitDetectiveEnvironment(
            backend=backend,
            require_ablation=require_ablation,
        )
        self.require_ablation = require_ablation
        self.reward = 0.0
        self.cumulative_reward = 0.0
        self.last_step_reward = 0.0
        self.done = False
        self.last_observation: Observation | None = None
        self.tool_trace: list[dict[str, Any]] = []
        self._seen_tools: set[str] = set()
        self._ablated_heads: set[str] = set()
        self._ablation_deltas: dict[str, float] = {}
        self._best_seen_head: str | None = None

    def reset(self, **_: Any) -> str | None:
        """Reset the episode and return the initial observation string."""
        self.reward = 0.0
        self.cumulative_reward = 0.0
        self.last_step_reward = 0.0
        self.done = False
        self.tool_trace = []
        self._seen_tools = set()
        self._ablated_heads = set()
        self._ablation_deltas = {}
        self._best_seen_head = None
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
        step_reward = self._trainer_step_reward(tool_name, arguments or {}, observation)
        self.cumulative_reward += step_reward
        self.reward = self.cumulative_reward
        self.tool_trace.append(
            {
                "tool_name": tool_name,
                "arguments": arguments or {},
                "done": observation.done,
                "reward": observation.reward,
                "trainer_step_reward": step_reward,
            }
        )
        return self._render_observation(observation)

    def _final_reward(self) -> float:
        """Return the scalar reward consumed by GRPO for the completed rollout."""
        reward = self.cumulative_reward
        if not self.done:
            reward -= 0.35
            if not self.tool_trace:
                reward -= 0.15
        return max(min(reward, 1.5), -1.0)

    def _training_summary(self, reward: float) -> dict[str, Any]:
        """Summarize one rollout for evaluation diagnostics."""
        terminal_score = self._terminal_score()
        submitted_heads: list[str] = []
        if self.last_observation is not None and self.done:
            raw_heads = self.last_observation.result.get("submitted_heads", [])
            if isinstance(raw_heads, list):
                submitted_heads = [str(item) for item in raw_heads]

        tools = [item["tool_name"] for item in self.tool_trace]
        submitted_set = set(submitted_heads)
        submitted_ablation_deltas = {
            head_id: self._ablation_deltas[head_id]
            for head_id in submitted_set
            if head_id in self._ablation_deltas
        }
        max_submitted_delta = max(submitted_ablation_deltas.values(), default=0.0)
        ablate_submitted = bool(submitted_ablation_deltas)
        causal_success = (
            terminal_score.get("f1", 0.0) == 1.0
            and ablate_submitted
            and max_submitted_delta >= CAUSAL_DELTA_THRESHOLD
        )
        return {
            "reward": reward,
            "done": self.done,
            "submitted": "submit_circuit" in tools,
            "correct": terminal_score.get("f1", 0.0) == 1.0,
            "terminal_reward": self.last_step_reward if self.done else 0.0,
            "f1": terminal_score.get("f1", 0.0),
            "tool_calls": len(self.tool_trace),
            "used_probe": "run_probe" in tools,
            "used_inspect": "inspect_induction_scores" in tools,
            "used_ablate": "ablate_head" in tools,
            "submitted_heads": submitted_heads,
            "ablated_heads": sorted(self._ablated_heads),
            "best_seen_head": self._best_seen_head,
            "ablate_submitted": ablate_submitted,
            "ablation_faithfulness": max_submitted_delta,
            "causal_success": causal_success,
        }

    def _trainer_step_reward(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        observation: Any,
    ) -> float:
        env_reward = float(observation.reward or 0.0)
        if env_reward < 0.0 and not (self.require_ablation and tool_name == "submit_circuit"):
            return env_reward

        if tool_name == "list_tools":
            return self._first_use_reward(tool_name, 0.0)
        if tool_name == "run_probe":
            return self._first_use_reward(tool_name, 0.01)
        if tool_name == "inspect_induction_scores":
            return self._inspect_reward(observation)
        if tool_name == "ablate_head":
            return self._ablation_reward(arguments, observation)
        if tool_name == "submit_circuit":
            return self._submit_reward(env_reward, observation)
        return 0.0

    def _first_use_reward(self, tool_name: str, reward: float) -> float:
        if tool_name in self._seen_tools:
            return -0.02
        self._seen_tools.add(tool_name)
        return reward

    def _inspect_reward(self, observation: Any) -> float:
        is_first_inspect = "inspect_induction_scores" not in self._seen_tools
        reward = self._first_use_reward("inspect_induction_scores", 0.25)
        scores = observation.result.get("scores", [])
        if not isinstance(scores, list) or not scores:
            return reward

        top_head = str(scores[0].get("head_id", ""))
        self._best_seen_head = top_head
        ground_truth = set(self.env.ground_truth_heads())
        if is_first_inspect and top_head in ground_truth:
            reward += 0.10
        return reward

    def _ablation_reward(self, arguments: dict[str, Any], observation: Any) -> float:
        layer = int(arguments.get("layer", -1))
        head = int(arguments.get("head", -1))
        head_id = f"L{layer}H{head}"
        if head_id in self._ablated_heads:
            return -0.01

        self._ablated_heads.add(head_id)
        self._ablation_deltas[head_id] = float(observation.result.get("behavior_delta", 0.0))
        reward = 0.04
        ground_truth = set(self.env.ground_truth_heads())
        if self.require_ablation and head_id == self._best_seen_head and head_id in ground_truth:
            reward += 0.46
        elif self.require_ablation and head_id in ground_truth:
            reward += 0.36
        elif head_id in ground_truth:
            reward += 0.16
        elif float(observation.result.get("behavior_delta", 0.0)) > 0.0:
            reward += 0.03
        return reward

    def _submit_reward(self, env_reward: float, observation: Any) -> float:
        score = observation.result.get("score", {})
        f1 = float(score.get("f1", 0.0)) if isinstance(score, dict) else 0.0
        if f1 <= 0.0:
            return env_reward + 0.03
        if self.require_ablation:
            return self._phase2_submit_reward(env_reward, observation)
        return env_reward + 0.25

    def _phase2_submit_reward(self, env_reward: float, observation: Any) -> float:
        submitted = set(str(item) for item in observation.result.get("submitted_heads", []))
        submitted_deltas = [
            self._ablation_deltas[head_id]
            for head_id in submitted
            if head_id in self._ablation_deltas
        ]
        if not submitted_deltas:
            return -0.45
        if max(submitted_deltas) >= CAUSAL_DELTA_THRESHOLD:
            return env_reward + 0.45
        return 0.05

    def _terminal_score(self) -> dict[str, float]:
        if self.last_observation is None or not self.done:
            return {}
        score = self.last_observation.result.get("score", {})
        if not isinstance(score, dict):
            return {}
        return {
            "precision": float(score.get("precision", 0.0)),
            "recall": float(score.get("recall", 0.0)),
            "f1": float(score.get("f1", 0.0)),
        }

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
    """Return shaped Phase 1 rewards and record rollout diagnostics."""
    rewards: list[float] = []
    for env in environments:
        reward = env._final_reward()
        rewards.append(reward)
        _REWARD_TRACE.append(env._training_summary(reward))
    return rewards


class Phase2CircuitDetectiveToolEnv(CircuitDetectiveToolEnv):
    """TRL tool wrapper for the ablation-required Phase 2 curriculum level."""

    def __init__(
        self,
        backend_factory: Callable[[], CircuitBackend] | None = None,
    ) -> None:
        super().__init__(
            backend_factory=backend_factory,
            require_ablation=True,
        )
