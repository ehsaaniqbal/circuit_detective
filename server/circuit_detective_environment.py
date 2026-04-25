"""Phase 1 Circuit Detective environment implementation."""

from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CircuitDetectiveAction, CircuitDetectiveObservation
except ImportError:
    from models import CircuitDetectiveAction, CircuitDetectiveObservation

from .backend import CircuitBackend, Head, get_default_backend
from .rewards import compute_submission_score


PHASE2_SCENARIO_ID = "l2_ablation_required"
CAUSAL_DELTA_THRESHOLD = 0.05


class CircuitDetectiveEnvironment(Environment):
    """OpenEnv server implementation for the Phase 1 induction-localization task."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        backend: CircuitBackend | None = None,
        *,
        require_ablation: bool = False,
        causal_delta_threshold: float = CAUSAL_DELTA_THRESHOLD,
    ) -> None:
        self._backend = backend or get_default_backend()
        self._require_ablation = require_ablation
        self._causal_delta_threshold = causal_delta_threshold
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._submitted_heads: list[str] = []
        self._inspected_heads: set[str] = set()
        self._ablation_deltas: dict[str, float] = {}

    @property
    def state(self) -> State:
        return self._state

    def ground_truth_heads(self) -> list[str]:
        """Return the deterministic answer key for trainer-side diagnostics."""
        return [head.head_id for head in self._backend.ground_truth_heads()]

    def reset(self) -> CircuitDetectiveObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._submitted_heads = []
        self._inspected_heads = set()
        self._ablation_deltas = {}
        if self._require_ablation:
            summary = (
                "Phase 2: localize the dominant induction head, causally verify it "
                "with ablate_head, then submit_circuit. Correct submissions receive "
                "full credit only when the submitted head was ablated and the "
                "intervention produced a meaningful behavior drop."
            )
            goal = (
                "Inspect the dominant induction head, ablate that candidate, "
                "then submit the verified head as ['LxHy']."
            )
        else:
            summary = (
                "Phase 1: localize the dominant induction head in TransformerLens "
                "attn-only-2l. Use list_tools, run_probe, inspect_induction_scores, "
                "ablate_head, then submit_circuit. You must call submit_circuit for "
                "the episode to receive a high score."
            )
            goal = "Submit the dominant induction head as ['LxHy']."
        return self._make_observation(
            summary=summary,
            result={
                "scenario": self._backend.scenario_id,
                "goal": goal,
                "requires_ablation": self._require_ablation,
            },
            reward=0.0,
            done=False,
        )

    def step(self, action: CircuitDetectiveAction) -> CircuitDetectiveObservation:  # type: ignore[override]
        self._state.step_count += 1

        try:
            tool_name = action.tool_name
            arguments = action.arguments or {}

            if tool_name == "list_tools":
                return self._list_tools()
            if tool_name == "run_probe":
                return self._run_probe()
            if tool_name == "inspect_induction_scores":
                return self._inspect_induction_scores(arguments)
            if tool_name == "ablate_head":
                return self._ablate_head(arguments)
            if tool_name == "submit_circuit":
                return self._submit_circuit(arguments)
            raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as exc:
            exhausted = self._state.step_count >= self._backend.max_steps
            reward = -0.05 if not exhausted else -0.25
            summary = f"Invalid action: {exc}"
            if exhausted:
                summary = f"{summary}. Step budget exhausted."
            return self._make_observation(
                summary=summary,
                result={"error": str(exc)},
                reward=reward,
                done=exhausted,
            )

    def _list_tools(self) -> CircuitDetectiveObservation:
        return self._make_observation(
            summary="Listed available tools.",
            result={
                "tools": [
                    {
                        "name": "list_tools",
                        "arguments": {},
                        "description": "Return the available tool surface.",
                    },
                    {
                        "name": "run_probe",
                        "arguments": {},
                        "description": "Measure baseline induction behavior on the fixed probe batch.",
                    },
                    {
                        "name": "inspect_induction_scores",
                        "arguments": {"top_k": "int, optional"},
                        "description": (
                            "Return the top attention heads by induction score; "
                            "this is the main evidence for the final submission."
                        ),
                    },
                    {
                        "name": "ablate_head",
                        "arguments": {"layer": "int", "head": "int"},
                        "description": "Zero-ablate one head and measure the behavior delta.",
                    },
                    {
                        "name": "submit_circuit",
                        "arguments": {"heads": "list[str] like ['L1H3']"},
                        "description": (
                            "Submit your candidate circuit and end the episode; "
                            "required for high reward."
                        ),
                    },
                ]
            },
            reward=0.01,
            done=False,
        )

    def _run_probe(self) -> CircuitDetectiveObservation:
        probe = self._backend.run_probe()
        return self._make_observation(
            summary="Measured baseline induction behavior.",
            result=probe.to_dict(),
            reward=0.01,
            done=False,
        )

    def _inspect_induction_scores(
        self,
        arguments: dict[str, object],
    ) -> CircuitDetectiveObservation:
        top_k = int(arguments.get("top_k", 8))
        scores = [item.to_dict() for item in self._backend.inspect_induction_scores(top_k=top_k)]
        self._inspected_heads.update(str(item["head_id"]) for item in scores)
        return self._make_observation(
            summary=(
                f"Returned the top {len(scores)} heads ranked by induction score. "
                "Use the strongest supported head in submit_circuit before the budget ends."
            ),
            result={"scores": scores},
            reward=0.01,
            done=False,
        )

    def _ablate_head(self, arguments: dict[str, object]) -> CircuitDetectiveObservation:
        layer = int(arguments["layer"])
        head = int(arguments["head"])
        result = self._backend.ablate_head(Head(layer=layer, head=head))
        payload = result.to_dict()
        self._ablation_deltas[result.head.head_id] = result.behavior_delta
        payload.update(
            {
                "causal_delta_threshold": self._causal_delta_threshold,
                "causal_verified": result.behavior_delta >= self._causal_delta_threshold,
            }
        )
        return self._make_observation(
            summary=f"Ablated L{layer}H{head} and measured the behavior delta.",
            result=payload,
            reward=0.01,
            done=False,
        )

    def _submit_circuit(self, arguments: dict[str, object]) -> CircuitDetectiveObservation:
        raw_heads = arguments.get("heads")
        if not isinstance(raw_heads, list):
            raise ValueError("submit_circuit requires heads to be a list of strings.")

        submitted = {Head.parse(str(item)).head_id for item in raw_heads}
        ground_truth = set(self.ground_truth_heads())
        score = compute_submission_score(
            submitted=submitted,
            ground_truth=ground_truth,
            step_count=self._state.step_count,
            max_steps=self._backend.max_steps,
        )
        self._submitted_heads = sorted(submitted)
        phase2 = self._phase2_score(submitted=submitted, f1=score.f1)
        reward = score.total_reward if not self._require_ablation else phase2["total_reward"]

        return self._make_observation(
            summary=(
                "Submitted candidate circuit."
                if submitted == ground_truth
                else "Submitted candidate circuit with partial or incorrect overlap."
            ),
            result={
                "submitted_heads": sorted(submitted),
                "score": score.to_dict(),
                "phase2": phase2,
            },
            reward=reward,
            done=True,
        )

    def _phase2_score(self, *, submitted: set[str], f1: float) -> dict[str, object]:
        submitted_ablation_deltas = {
            head_id: self._ablation_deltas[head_id]
            for head_id in submitted
            if head_id in self._ablation_deltas
        }
        max_delta = max(submitted_ablation_deltas.values(), default=0.0)
        ablate_submitted = bool(submitted_ablation_deltas)
        faithful_ablation = max_delta >= self._causal_delta_threshold
        causal_success = f1 == 1.0 and ablate_submitted and faithful_ablation

        if not self._require_ablation:
            total_reward = 0.0
        elif causal_success:
            normalized_steps = min(self._state.step_count / self._backend.max_steps, 1.0)
            total_reward = 1.0 - (0.1 * normalized_steps)
        elif f1 > 0.0 and ablate_submitted:
            total_reward = 0.20 * f1
        elif f1 > 0.0:
            total_reward = -0.35
        else:
            total_reward = -0.10

        return {
            "requires_ablation": self._require_ablation,
            "ablate_submitted": ablate_submitted,
            "submitted_ablation_deltas": submitted_ablation_deltas,
            "ablation_faithfulness": max_delta,
            "causal_delta_threshold": self._causal_delta_threshold,
            "faithful_ablation": faithful_ablation,
            "causal_success": causal_success,
            "total_reward": total_reward,
        }

    def _make_observation(
        self,
        *,
        summary: str,
        result: dict[str, object],
        reward: float,
        done: bool,
    ) -> CircuitDetectiveObservation:
        remaining_budget = max(self._backend.max_steps - self._state.step_count, 0)
        return CircuitDetectiveObservation(
            summary=summary,
            result=result,
            scenario_id=PHASE2_SCENARIO_ID if self._require_ablation else self._backend.scenario_id,
            step_count=self._state.step_count,
            remaining_budget=remaining_budget,
            available_tools=[
                "list_tools",
                "run_probe",
                "inspect_induction_scores",
                "ablate_head",
                "submit_circuit",
            ],
            reward=reward,
            done=done,
            metadata={
                "submitted_heads": self._submitted_heads,
                "inspected_heads": sorted(self._inspected_heads),
                "ablation_deltas": self._ablation_deltas,
                "requires_ablation": self._require_ablation,
            },
        )
