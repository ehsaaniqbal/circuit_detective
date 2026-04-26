"""Helpers for Phase 1 GRPO training against Circuit Detective."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .models import CircuitDetectiveAction
from .server.backend import (
    CircuitBackend,
    PlantedLiteCausalChainBackend,
    PublishedIOICircuitBackend,
    RandomizedPlantedCircuitBackend,
    RealIOITransformerLensBackend,
    get_default_backend,
    get_real_ioi_backend,
)
from .server.circuit_detective_environment import (
    CAUSAL_DELTA_THRESHOLD,
    CircuitDetectiveEnvironment,
    IOI_CAUSAL_DELTA_THRESHOLD,
    PLANTED_CAUSAL_DELTA_THRESHOLD,
    PLANTED_LITE_CAUSAL_DELTA_THRESHOLD,
    REAL_IOI_CAUSAL_DELTA_THRESHOLD,
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

PLANTED_SYSTEM_PROMPT = (
    "You are a mechanistic-interpretability agent in a planted circuit arena. "
    "Inspection scores are noisy and may rank a decoy head first. You must use "
    "ablate_head to test candidates, then submit the head with the largest "
    "causal behavior delta. Do not submit based on score ranking alone."
)

PLANTED_USER_PROMPT_VARIANTS = [
    (
        "Find the planted causal head. Scores may contain a decoy, so ablate "
        "candidate heads before submitting."
    ),
    (
        "This episode has one hidden planted circuit. Use inspection and ablation "
        "to identify the true causal head."
    ),
    (
        "Do not trust the top score. Intervene on candidate heads and submit the "
        "one whose ablation causes the largest behavior drop."
    ),
    (
        "Investigate the randomized planted circuit arena. Use ablate_head to "
        "separate decoys from the true target."
    ),
    (
        "Find the causal component in this synthetic transformer lab. Full credit "
        "requires submitting the ablated head with real behavior impact."
    ),
    (
        "Scores are only hints in this task. Test the likely heads with ablation "
        "and submit the verified planted head."
    ),
    (
        "Solve the planted circuit challenge: inspect, ablate candidates, compare "
        "behavior deltas, then submit one head."
    ),
    (
        "Use the tools like a circuit detective. The top-ranked head can be a "
        "decoy; the true answer is revealed by causal intervention."
    ),
]

PLANTED_LITE_SYSTEM_PROMPT = (
    "You are a mechanistic-interpretability agent in a two-candidate planted "
    "causal-chain curriculum. The top inspection score is deliberately a decoy. "
    "Required workflow: call inspect_induction_scores(top_k=2), ablate both "
    "candidate heads, compare behavior_delta, then call submit_circuit with "
    "exactly the best_ablated_head_so_far. Episodes that stop after ablation "
    "without submit_circuit are failures. When a tool response contains "
    "next_required_tool, follow it exactly. When a tool response contains "
    "must_submit, immediately call submit_circuit with that exact head."
)

PLANTED_LITE_USER_PROMPT_VARIANTS = [
    (
        "Solve the planted-lite causal task. Inspect two candidates, ablate both, "
        "and submit the candidate with the larger behavior_delta."
    ),
    (
        "The first inspected head is a decoy. Use ablate_head on every candidate "
        "before submitting best_ablated_head_so_far."
    ),
    (
        "Train the causal chain: inspect candidates, intervene on both heads, "
        "compare deltas, then submit one verified head."
    ),
    (
        "Do not guess from ranking. Full credit requires ablating both candidate "
        "heads and submitting the max-delta head."
    ),
    (
        "Terminal action matters: after both candidate heads are ablated and "
        "must_submit is populated, call submit_circuit immediately."
    ),
    (
        "Follow next_required_tool exactly. If it says submit_circuit, finish "
        "the episode with the provided heads argument."
    ),
]

IOI_SYSTEM_PROMPT = (
    "You are a mechanistic-interpretability agent investigating the IOI "
    "name-mover component in GPT-2 small. Inspect candidate head effects, "
    "ablate at least one supporting name-mover head, then submit exactly the "
    "name-mover heads as a list. This is a multi-head circuit task."
)

IOI_USER_PROMPT_VARIANTS = [
    (
        "Find the IOI name-mover component. Inspect candidate heads, verify with "
        "ablation, and submit the complete name-mover head set."
    ),
    (
        "Identify the GPT-2-small heads that move the indirect-object name in the "
        "IOI circuit. Submit all and only the name-mover heads."
    ),
    (
        "Solve the IOI stretch task: use the tools to localize the name-mover "
        "heads and submit the multi-head circuit."
    ),
    (
        "Do not submit a single head. The IOI name-mover component has multiple "
        "heads; inspect, ablate, and submit the full component."
    ),
]

CURRICULUM_SYSTEM_PROMPT = (
    "You are a mechanistic-interpretability agent. Each episode may be a toy "
    "induction, planted-circuit, or IOI component task. Read the reset "
    "observation, use tools to gather evidence, ablate when causal validation is "
    "required, and submit the requested circuit."
)

CURRICULUM_USER_PROMPT_VARIANTS = [
    (
        "Solve the current circuit task. Use the observation to infer whether this "
        "is a single-head or multi-head scenario."
    ),
    (
        "Investigate the active scenario with the tools, verify causal evidence "
        "when needed, and submit the final circuit."
    ),
    (
        "Generalize your circuit-detective policy across this episode. Inspect, "
        "intervene if required, and submit the supported heads."
    ),
    (
        "Do the current mechanistic-interpretability task end to end. Avoid fixed "
        "answers; follow the evidence in the tool outputs."
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

    if scenario == "curriculum":
        system_prompt = CURRICULUM_SYSTEM_PROMPT
        user_prompts = CURRICULUM_USER_PROMPT_VARIANTS
    elif scenario in {"ioi", "real_ioi"}:
        system_prompt = IOI_SYSTEM_PROMPT
        user_prompts = IOI_USER_PROMPT_VARIANTS
    elif scenario == "planted_lite":
        system_prompt = PLANTED_LITE_SYSTEM_PROMPT
        user_prompts = PLANTED_LITE_USER_PROMPT_VARIANTS
    elif scenario == "planted":
        system_prompt = PLANTED_SYSTEM_PROMPT
        user_prompts = PLANTED_USER_PROMPT_VARIANTS
    elif scenario == "phase2":
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
        causal_delta_threshold: float = CAUSAL_DELTA_THRESHOLD,
        strict_causal_chain: bool = False,
    ) -> None:
        backend = backend_factory() if backend_factory is not None else get_default_backend()
        self.env = CircuitDetectiveEnvironment(
            backend=backend,
            require_ablation=require_ablation,
            causal_delta_threshold=causal_delta_threshold,
        )
        self.require_ablation = require_ablation
        self.causal_delta_threshold = causal_delta_threshold
        self.strict_causal_chain = strict_causal_chain
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
        if self.strict_causal_chain:
            return self._strict_causal_chain_final_reward()

        if self.require_ablation and not self.done:
            return -2.0 if self.strict_causal_chain else -1.0

        reward = self.cumulative_reward
        if not self.done:
            reward -= 0.35
            if not self.tool_trace:
                reward -= 0.15
        return max(min(reward, 1.5), -1.0)

    def _strict_causal_chain_final_reward(self) -> float:
        candidate_heads = set(self.env.candidate_heads())
        ablated_candidates = candidate_heads.intersection(self._ablated_heads)
        ablated_all_candidates = bool(candidate_heads) and ablated_candidates == candidate_heads
        tools = [item["tool_name"] for item in self.tool_trace]
        terminal_score = self._terminal_score()
        f1 = float(terminal_score.get("f1", 0.0))

        if self.done:
            if f1 == 1.0 and ablated_all_candidates:
                return 5.0
            if f1 == 1.0 and ablated_candidates:
                return 1.5
            if "submit_circuit" in tools and ablated_all_candidates:
                return 0.6
            if "submit_circuit" in tools:
                return -0.1

        if ablated_all_candidates:
            return -0.8
        if len(ablated_candidates) == 1:
            return -1.0
        if "inspect_induction_scores" in tools:
            return -1.2
        if tools:
            return -1.6
        return -2.0

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
        candidate_heads = set(self.env.candidate_heads())
        ablated_candidate_heads = sorted(candidate_heads.intersection(self._ablated_heads))
        ablated_all_candidates = bool(candidate_heads) and set(ablated_candidate_heads) == candidate_heads
        max_submitted_delta = max(submitted_ablation_deltas.values(), default=0.0)
        ablate_submitted = bool(submitted_ablation_deltas)
        best_ablated_head = (
            max(self._ablation_deltas, key=self._ablation_deltas.get)
            if self._ablation_deltas
            else ""
        )
        causal_success = (
            terminal_score.get("f1", 0.0) == 1.0
            and ablate_submitted
            and max_submitted_delta >= self.causal_delta_threshold
            and (
                not self.strict_causal_chain
                or ablated_all_candidates
            )
        )
        scenario_id = ""
        if self.last_observation is not None:
            scenario_id = getattr(self.last_observation, "scenario_id", "")
        return {
            "scenario_id": scenario_id,
            "reward": reward,
            "rubric": self._rubric_breakdown(
                reward=reward,
                terminal_score=terminal_score,
                submitted=bool(submitted_heads),
                causal_success=causal_success,
                ablate_submitted=ablate_submitted,
                max_submitted_delta=max_submitted_delta,
            ),
            "done": self.done,
            "submitted": "submit_circuit" in tools,
            "correct": terminal_score.get("f1", 0.0) == 1.0,
            "terminal_reward": self.last_step_reward if self.done else 0.0,
            "f1": terminal_score.get("f1", 0.0),
            "tool_calls": len(self.tool_trace),
            "tool_sequence": tools,
            "tool_trace": self.tool_trace,
            "used_probe": "run_probe" in tools,
            "used_inspect": "inspect_induction_scores" in tools,
            "used_ablate": "ablate_head" in tools,
            "submitted_heads": submitted_heads,
            "ablated_heads": sorted(self._ablated_heads),
            "candidate_heads": sorted(candidate_heads),
            "ablated_candidate_heads": ablated_candidate_heads,
            "candidate_ablation_coverage": (
                len(ablated_candidate_heads) / len(candidate_heads)
                if candidate_heads
                else 0.0
            ),
            "all_candidates_ablated": ablated_all_candidates,
            "terminal_ready_no_submit": ablated_all_candidates and "submit_circuit" not in tools,
            "submitted_after_all_candidates": ablated_all_candidates and "submit_circuit" in tools,
            "best_ablated_head": best_ablated_head,
            "submitted_best_ablated_head": bool(best_ablated_head)
            and best_ablated_head in submitted_set,
            "best_seen_head": self._best_seen_head,
            "ablate_submitted": ablate_submitted,
            "ablation_faithfulness": max_submitted_delta,
            "causal_success": causal_success,
        }

    def _rubric_breakdown(
        self,
        *,
        reward: float,
        terminal_score: dict[str, float],
        submitted: bool,
        causal_success: bool,
        ablate_submitted: bool,
        max_submitted_delta: float,
    ) -> dict[str, float]:
        tools = [item["tool_name"] for item in self.tool_trace]
        evidence = 1.0 if "inspect_induction_scores" in tools else 0.0
        intervention = 1.0 if "ablate_head" in tools else 0.0
        final_answer = terminal_score.get("f1", 0.0) if submitted else 0.0
        causal = 0.0
        if not self.require_ablation:
            causal = 1.0 if submitted else 0.0
        elif causal_success:
            causal = 1.0
        elif ablate_submitted:
            causal = min(max_submitted_delta / self.causal_delta_threshold, 1.0)
        return {
            "tool_format": 1.0 if self.tool_trace else 0.0,
            "evidence_gathering": evidence,
            "intervention": intervention,
            "causal_validation": causal,
            "final_answer_f1": final_answer,
            "scalar_reward": reward,
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
        reward = self._first_use_reward(
            "inspect_induction_scores",
            0.02 if self.strict_causal_chain else 0.08 if self.require_ablation else 0.25,
        )
        scores = observation.result.get("scores", [])
        if not isinstance(scores, list) or not scores:
            return reward

        top_head = str(scores[0].get("head_id", ""))
        self._best_seen_head = top_head
        ground_truth = set(self.env.ground_truth_heads())
        if is_first_inspect and top_head in ground_truth:
            reward += 0.02 if self.require_ablation else 0.10
        return reward

    def _ablation_reward(self, arguments: dict[str, Any], observation: Any) -> float:
        layer = int(arguments.get("layer", -1))
        head = int(arguments.get("head", -1))
        head_id = f"L{layer}H{head}"
        if head_id in self._ablated_heads:
            return -0.01

        self._ablated_heads.add(head_id)
        self._ablation_deltas[head_id] = float(observation.result.get("behavior_delta", 0.0))
        if self.strict_causal_chain:
            candidate_heads = set(self.env.candidate_heads())
            if head_id not in candidate_heads:
                return -0.25
            ablated_candidates = candidate_heads.intersection(self._ablated_heads)
            if candidate_heads and ablated_candidates == candidate_heads:
                return 0.80
            return 0.25

        reward = 0.01 if self.require_ablation else 0.04
        ground_truth = set(self.env.ground_truth_heads())
        if self.require_ablation and head_id == self._best_seen_head and head_id in ground_truth:
            reward += 0.24
        elif self.require_ablation and head_id in ground_truth:
            reward += 0.24
        elif head_id in ground_truth:
            reward += 0.16
        elif (
            self.require_ablation
            and float(observation.result.get("behavior_delta", 0.0)) >= self.causal_delta_threshold
        ):
            reward += 0.02
        elif not self.require_ablation and float(observation.result.get("behavior_delta", 0.0)) > 0.0:
            reward += 0.03
        return reward

    def _submit_reward(self, env_reward: float, observation: Any) -> float:
        score = observation.result.get("score", {})
        f1 = float(score.get("f1", 0.0)) if isinstance(score, dict) else 0.0
        if f1 <= 0.0:
            if self.strict_causal_chain:
                candidate_heads = set(self.env.candidate_heads())
                ablated_candidates = candidate_heads.intersection(self._ablated_heads)
                if candidate_heads and ablated_candidates == candidate_heads:
                    return 0.6
                if ablated_candidates:
                    return -0.1
                return -1.2
            if self.require_ablation:
                return -0.40
            return env_reward + 0.03
        if self.require_ablation:
            return self._phase2_submit_reward(env_reward, observation)
        return env_reward + 0.25

    def _phase2_submit_reward(self, env_reward: float, observation: Any) -> float:
        raw_submitted = observation.result.get("submitted_heads", [])
        submitted = set(str(item) for item in raw_submitted if isinstance(item, str))
        score = observation.result.get("score", {})
        f1 = float(score.get("f1", 0.0)) if isinstance(score, dict) else 0.0
        ground_truth_count = len(self.env.ground_truth_heads())
        if len(submitted) > max(ground_truth_count + 1, 2):
            if self.strict_causal_chain:
                return -2.0
            return -0.55

        submitted_deltas = [
            self._ablation_deltas[head_id]
            for head_id in submitted
            if head_id in self._ablation_deltas
        ]
        if not submitted_deltas:
            if self.strict_causal_chain:
                return -1.5
            return -0.55
        if self.strict_causal_chain:
            phase2 = observation.result.get("phase2", {})
            ablated_all = bool(phase2.get("ablated_all_candidates", False)) if isinstance(phase2, dict) else False
            if f1 == 1.0 and ablated_all and max(submitted_deltas) >= self.causal_delta_threshold:
                return 5.0
            if f1 == 1.0:
                return 1.5
            if ablated_all:
                return 0.6
            return -0.1
        if f1 < 1.0:
            return -0.20 + (0.20 * f1)
        if max(submitted_deltas) >= self.causal_delta_threshold:
            bonus = 0.85
            if submitted and self._best_seen_head in submitted:
                bonus += 0.15
            return env_reward + bonus
        return -0.10

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
        if self.strict_causal_chain:
            return self._render_strict_causal_observation(observation)

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

    def _render_strict_causal_observation(self, observation: Any) -> str:
        keep_result_keys = {
            "ablation_faithfulness",
            "ablated_all_candidates",
            "ablated_candidate_heads",
            "ablated_head",
            "behavior_delta",
            "best_ablated_delta_so_far",
            "best_ablated_head_so_far",
            "candidate_heads",
            "causal_success",
            "causal_verified",
            "goal",
            "must_submit",
            "next_required_arguments",
            "next_required_tool",
            "phase2",
            "remaining_candidate_heads",
            "requires_ablation",
            "score",
            "scores",
            "submitted_heads",
            "terminal_action_required",
            "total_reward",
        }
        result = {
            key: value
            for key, value in observation.result.items()
            if key in keep_result_keys
        }
        if observation.step_count == 0:
            result.pop("candidate_heads", None)

        payload = {
            "summary": observation.summary,
            "result": result,
            "scenario_id": observation.scenario_id,
            "step_count": observation.step_count,
            "remaining_budget": observation.remaining_budget,
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


class PlantedCircuitToolEnv(CircuitDetectiveToolEnv):
    """TRL tool wrapper for randomized planted-circuit episodes."""

    def __init__(
        self,
        backend_factory: Callable[[], CircuitBackend] | None = None,
    ) -> None:
        super().__init__(
            backend_factory=backend_factory or RandomizedPlantedCircuitBackend,
            require_ablation=True,
            causal_delta_threshold=PLANTED_CAUSAL_DELTA_THRESHOLD,
        )


class PlantedLiteCircuitToolEnv(CircuitDetectiveToolEnv):
    """TRL tool wrapper for the two-candidate causal-chain curriculum."""

    def __init__(
        self,
        backend_factory: Callable[[], CircuitBackend] | None = None,
    ) -> None:
        super().__init__(
            backend_factory=backend_factory or PlantedLiteCausalChainBackend,
            require_ablation=True,
            causal_delta_threshold=PLANTED_LITE_CAUSAL_DELTA_THRESHOLD,
            strict_causal_chain=True,
        )


class IOICircuitToolEnv(CircuitDetectiveToolEnv):
    """TRL tool wrapper for the IOI name-mover stretch curriculum."""

    def __init__(
        self,
        backend_factory: Callable[[], CircuitBackend] | None = None,
    ) -> None:
        super().__init__(
            backend_factory=backend_factory or PublishedIOICircuitBackend,
            require_ablation=True,
            causal_delta_threshold=IOI_CAUSAL_DELTA_THRESHOLD,
        )


class RealIOICircuitToolEnv(CircuitDetectiveToolEnv):
    """TRL tool wrapper for real TransformerLens GPT-2-small IOI probes."""

    def __init__(
        self,
        backend_factory: Callable[[], CircuitBackend] | None = None,
    ) -> None:
        super().__init__(
            backend_factory=backend_factory or get_real_ioi_backend,
            require_ablation=True,
            causal_delta_threshold=REAL_IOI_CAUSAL_DELTA_THRESHOLD,
        )


class CurriculumCircuitToolEnv(CircuitDetectiveToolEnv):
    """Balanced planted-plus-IOI curriculum for multi-task GRPO."""

    _instance_counter = 0

    def __init__(self) -> None:
        index = CurriculumCircuitToolEnv._instance_counter
        CurriculumCircuitToolEnv._instance_counter += 1
        if index % 2 == 0:
            super().__init__(
                backend_factory=RandomizedPlantedCircuitBackend,
                require_ablation=True,
                causal_delta_threshold=PLANTED_CAUSAL_DELTA_THRESHOLD,
            )
        else:
            super().__init__(
                backend_factory=PublishedIOICircuitBackend,
                require_ablation=True,
                causal_delta_threshold=IOI_CAUSAL_DELTA_THRESHOLD,
            )
