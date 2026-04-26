from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuit_detective.models import CircuitDetectiveAction
from circuit_detective.server.backend import Head, RealIOITransformerLensBackend
from circuit_detective.server.circuit_detective_environment import (
    CircuitDetectiveEnvironment,
    REAL_IOI_CAUSAL_DELTA_THRESHOLD,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the real GPT-2-small IOI backend.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--output",
        default="artifacts/real_ioi_smoke/real_ioi_smoke.json",
        help="Path to write the smoke-test transcript.",
    )
    return parser.parse_args()


def observation_payload(observation: object) -> dict[str, object]:
    return {
        "summary": getattr(observation, "summary"),
        "result": getattr(observation, "result"),
        "scenario_id": getattr(observation, "scenario_id"),
        "step_count": getattr(observation, "step_count"),
        "remaining_budget": getattr(observation, "remaining_budget"),
        "done": getattr(observation, "done"),
        "reward": getattr(observation, "reward", 0.0),
    }


def main() -> None:
    args = parse_args()
    backend = RealIOITransformerLensBackend()
    env = CircuitDetectiveEnvironment(
        backend=backend,
        require_ablation=True,
        causal_delta_threshold=REAL_IOI_CAUSAL_DELTA_THRESHOLD,
    )
    transcript: list[dict[str, object]] = []

    try:
        reset = env.reset()
        transcript.append({"action": "reset", "observation": observation_payload(reset)})

        probe = env.step(CircuitDetectiveAction(tool_name="run_probe"))
        transcript.append({"action": "run_probe", "observation": observation_payload(probe)})

        inspect = env.step(
            CircuitDetectiveAction(
                tool_name="inspect_induction_scores",
                arguments={"top_k": args.top_k},
            )
        )
        transcript.append({"action": "inspect_induction_scores", "observation": observation_payload(inspect)})

        scores = inspect.result["scores"]
        if not isinstance(scores, list) or not scores:
            raise RuntimeError("Real IOI inspect returned no candidate scores.")
        top_head = Head.parse(str(scores[0]["head_id"]))
        ablate_top = env.step(
            CircuitDetectiveAction(
                tool_name="ablate_head",
                arguments={"layer": top_head.layer, "head": top_head.head},
            )
        )
        transcript.append({"action": f"ablate_top:{top_head.head_id}", "observation": observation_payload(ablate_top)})

        ground_truth = env.ground_truth_heads()
        if top_head.head_id not in ground_truth:
            submitted_probe = Head.parse(ground_truth[0])
            ablate_submitted = env.step(
                CircuitDetectiveAction(
                    tool_name="ablate_head",
                    arguments={"layer": submitted_probe.layer, "head": submitted_probe.head},
                )
            )
            transcript.append(
                {
                    "action": f"ablate_submitted:{submitted_probe.head_id}",
                    "observation": observation_payload(ablate_submitted),
                }
            )

        submit = env.step(
            CircuitDetectiveAction(
                tool_name="submit_circuit",
                arguments={"heads": ground_truth},
            )
        )
        transcript.append({"action": "submit_ground_truth", "observation": observation_payload(submit)})

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(transcript, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(transcript[-1], indent=2, sort_keys=True), flush=True)
        print(f"saved: {output_path}", flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
