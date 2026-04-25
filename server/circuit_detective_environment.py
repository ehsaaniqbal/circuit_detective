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


class CircuitDetectiveEnvironment(Environment):
    """OpenEnv server implementation for the Phase 1 induction-localization task."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, backend: CircuitBackend | None = None) -> None:
        self._backend = backend or get_default_backend()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._submitted_heads: list[str] = []

    @property
    def state(self) -> State:
        return self._state

    def reset(self) -> CircuitDetectiveObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._submitted_heads = []
        return self._make_observation(
            summary=(
                "Phase 1: localize the dominant induction head in TransformerLens "
                "attn-only-2l. Use list_tools, run_probe, inspect_induction_scores, "
                "ablate_head, then submit_circuit."
            ),
            result={
                "scenario": self._backend.scenario_id,
                "goal": "Submit the dominant induction head as ['LxHy'].",
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
                        "description": "Return the top attention heads by induction score.",
                    },
                    {
                        "name": "ablate_head",
                        "arguments": {"layer": "int", "head": "int"},
                        "description": "Zero-ablate one head and measure the behavior delta.",
                    },
                    {
                        "name": "submit_circuit",
                        "arguments": {"heads": "list[str] like ['L1H3']"},
                        "description": "Submit your candidate circuit and end the episode.",
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
        return self._make_observation(
            summary=f"Returned the top {len(scores)} heads ranked by induction score.",
            result={"scores": scores},
            reward=0.01,
            done=False,
        )

    def _ablate_head(self, arguments: dict[str, object]) -> CircuitDetectiveObservation:
        layer = int(arguments["layer"])
        head = int(arguments["head"])
        result = self._backend.ablate_head(Head(layer=layer, head=head))
        return self._make_observation(
            summary=f"Ablated L{layer}H{head} and measured the behavior delta.",
            result=result.to_dict(),
            reward=0.01,
            done=False,
        )

    def _submit_circuit(self, arguments: dict[str, object]) -> CircuitDetectiveObservation:
        raw_heads = arguments.get("heads")
        if not isinstance(raw_heads, list):
            raise ValueError("submit_circuit requires heads to be a list of strings.")

        submitted = {Head.parse(str(item)).head_id for item in raw_heads}
        ground_truth = {head.head_id for head in self._backend.ground_truth_heads()}
        score = compute_submission_score(
            submitted=submitted,
            ground_truth=ground_truth,
            step_count=self._state.step_count,
            max_steps=self._backend.max_steps,
        )
        self._submitted_heads = sorted(submitted)

        return self._make_observation(
            summary=(
                "Submitted candidate circuit."
                if submitted == ground_truth
                else "Submitted candidate circuit with partial or incorrect overlap."
            ),
            result={
                "submitted_heads": sorted(submitted),
                "score": score.to_dict(),
            },
            reward=score.total_reward,
            done=True,
        )

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
            scenario_id=self._backend.scenario_id,
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
            metadata={"submitted_heads": self._submitted_heads},
        )
