from __future__ import annotations

from circuit_detective.models import CircuitDetectiveAction
from circuit_detective.server.circuit_detective_environment import CircuitDetectiveEnvironment


def main() -> None:
    env = CircuitDetectiveEnvironment()
    reset_obs = env.reset()
    print("reset:", reset_obs.summary)

    tools_obs = env.step(CircuitDetectiveAction(tool_name="list_tools"))
    print("tools:", [tool["name"] for tool in tools_obs.result["tools"]])

    scores_obs = env.step(
        CircuitDetectiveAction(
            tool_name="inspect_induction_scores",
            arguments={"top_k": 3},
        )
    )
    top_head = scores_obs.result["scores"][0]["head_id"]
    print("top_head:", top_head)

    submit_obs = env.step(
        CircuitDetectiveAction(
            tool_name="submit_circuit",
            arguments={"heads": [top_head]},
        )
    )
    print("submit_reward:", submit_obs.reward)


if __name__ == "__main__":
    main()
