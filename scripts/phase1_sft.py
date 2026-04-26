from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from circuit_detective.phase1_grpo import (
    CURRICULUM_SYSTEM_PROMPT,
    CURRICULUM_USER_PROMPT_VARIANTS,
    IOI_SYSTEM_PROMPT,
    IOI_USER_PROMPT_VARIANTS,
    PHASE1_SYSTEM_PROMPT,
    PHASE1_USER_PROMPT_VARIANTS,
    PHASE2_SYSTEM_PROMPT,
    PHASE2_USER_PROMPT_VARIANTS,
    PLANTED_LITE_SYSTEM_PROMPT,
    PLANTED_LITE_USER_PROMPT_VARIANTS,
    PLANTED_SYSTEM_PROMPT,
    PLANTED_USER_PROMPT_VARIANTS,
)
from circuit_detective.phase1_grpo import CircuitDetectiveToolEnv
from circuit_detective.server.circuit_detective_environment import (
    PLANTED_CAUSAL_DELTA_THRESHOLD,
    PLANTED_LITE_CAUSAL_DELTA_THRESHOLD,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny Phase 1 SFT warm-start.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--output-dir", default="outputs/phase1_sft_warmup")
    parser.add_argument("--target-head", default="L1H6")
    parser.add_argument("--examples-per-prompt", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional PEFT adapter path or Hub model id to continue SFT from.",
    )
    parser.add_argument(
        "--scenario",
        choices=["phase1", "phase2", "planted", "planted_lite", "ioi", "curriculum", "real_ioi"],
        default="phase1",
        help="Warm-start curriculum level. phase2 examples always include ablation.",
    )
    return parser.parse_args()


def tool_call(name: str, parameters: dict[str, Any]) -> str:
    lines = ["<tool_call>", f"<function={name}>"]
    for key, value in parameters.items():
        rendered = json.dumps(value) if isinstance(value, list | dict) else str(value)
        lines.extend([f"<parameter={key}>", rendered, "</parameter>"])
    lines.extend([f"</function>", "</tool_call>"])
    return "\n".join(lines)


def synthetic_reset_observation(*, scenario: str) -> str:
    is_phase2 = scenario == "phase2"
    is_planted = scenario == "planted"
    is_planted_lite = scenario == "planted_lite"
    is_ioi = scenario in {"ioi", "real_ioi"}
    remaining_budget = 5 if is_planted_lite else 10 if is_planted else 12
    if is_ioi:
        remaining_budget = 12
    if is_ioi:
        goal = "Submit exactly the IOI name-mover heads as a multi-head circuit."
        summary = "IOI Name-Mover Arena: identify and submit the multi-head name-mover component."
    elif is_planted_lite:
        goal = "Ablate both candidate heads and submit best_ablated_head_so_far."
        summary = (
            "Planted-Lite Causal Chain: exactly two candidate heads are shown. "
            "Ablate both candidates and submit the larger behavior_delta."
        )
    elif is_planted:
        goal = (
            "Ablate candidate heads and submit the planted head with the largest "
            "behavior delta."
        )
        summary = (
            "Planted Circuit Arena: score rankings can be decoys. "
            "Use ablation to find the true causal head."
        )
    elif is_phase2:
        goal = (
            "Inspect the dominant induction head, ablate that candidate, "
            "then submit the verified head as ['LxHy']."
        )
        summary = (
            "Phase 2: localize and causally verify the dominant induction head. "
            "Use inspect_induction_scores, ablate_head, then submit_circuit."
        )
    else:
        goal = "Submit the dominant induction head as ['LxHy']."
        summary = (
            "Phase 1: localize the dominant induction head. "
            "Use inspect_induction_scores, then submit_circuit."
        )
    scenario_id = (
        "ioi_gpt2_small_real"
        if scenario == "real_ioi"
        else "ioi_gpt2_small_name_mover"
        if scenario == "ioi"
        else "planted_lite_causal_chain"
        if is_planted_lite
        else "planted_circuit_arena"
        if is_planted
        else "l2_ablation_required"
        if is_phase2
        else "l1_induction_attn_only_2l"
    )
    return json.dumps(
        {
            "available_tools": [
                "list_tools",
                "run_probe",
                "inspect_induction_scores",
                "ablate_head",
                "submit_circuit",
            ],
            "done": False,
            "remaining_budget": remaining_budget,
            "result": {
                "goal": goal,
                "requires_ablation": is_phase2 or is_planted or is_planted_lite or is_ioi,
                "scenario": scenario_id,
            },
            "scenario_id": scenario_id,
            "step_count": 0,
            "summary": summary,
        },
        sort_keys=True,
    )


def parse_head_id(head_id: str) -> tuple[int, int]:
    return (
        int(head_id.split("H", maxsplit=1)[0].removeprefix("L")),
        int(head_id.split("H", maxsplit=1)[1]),
    )


def all_synthetic_heads() -> list[str]:
    return [f"L{layer}H{head}" for layer in range(2) for head in range(8)]


def planted_heads_for_record(index: int) -> tuple[str, list[str]]:
    heads = all_synthetic_heads()
    target = heads[(3 * index + 5) % len(heads)]
    decoy_one = heads[(5 * index + 1) % len(heads)]
    decoy_two = heads[(7 * index + 2) % len(heads)]
    decoys = []
    for candidate in [decoy_one, decoy_two, *heads]:
        if candidate != target and candidate not in decoys:
            decoys.append(candidate)
        if len(decoys) == 2:
            break
    return target, decoys


def planted_lite_heads_for_record(index: int) -> tuple[str, str]:
    heads = all_synthetic_heads()
    target = heads[(5 * index + 3) % len(heads)]
    decoy = heads[(7 * index + 1) % len(heads)]
    if decoy == target:
        decoy = heads[(heads.index(decoy) + 1) % len(heads)]
    return target, decoy


def synthetic_planted_lite_inspect_response(
    *,
    target_head: str,
    decoy_head: str,
) -> str:
    scores: list[dict[str, float | int | str]] = []
    for score, head_id in [(0.96, decoy_head), (0.84, target_head)]:
        layer, head = parse_head_id(head_id)
        scores.append({"head": head, "head_id": head_id, "layer": layer, "score": score})
    return json.dumps(
        {
            "available_tools": [
                "list_tools",
                "run_probe",
                "inspect_induction_scores",
                "ablate_head",
                "submit_circuit",
            ],
            "done": False,
            "remaining_budget": 4,
            "result": {
                "scores": scores,
                "candidate_heads": [decoy_head, target_head],
            },
            "scenario_id": "planted_lite_causal_chain",
            "step_count": 1,
            "summary": (
                "Returned exactly two planted-lite candidate heads. Ablate each "
                "candidate, then submit best_ablated_head_so_far."
            ),
        },
        sort_keys=True,
    )


def synthetic_planted_lite_ablation_response(
    *,
    head_id: str,
    target_head: str,
    decoy_head: str,
    ablated_heads_so_far: list[str],
) -> str:
    layer, head = parse_head_id(head_id)
    candidate_heads = [decoy_head, target_head]
    ablated_candidate_heads = [
        candidate for candidate in candidate_heads if candidate in [*ablated_heads_so_far, head_id]
    ]
    deltas = {decoy_head: 0.018, target_head: 0.49}
    best_head = max(ablated_candidate_heads, key=lambda candidate: deltas[candidate])
    remaining_candidate_heads = [
        candidate for candidate in candidate_heads if candidate not in ablated_candidate_heads
    ]
    delta = deltas[head_id]
    return json.dumps(
        {
            "available_tools": [
                "list_tools",
                "run_probe",
                "inspect_induction_scores",
                "ablate_head",
                "submit_circuit",
            ],
            "done": False,
            "remaining_budget": 3 if remaining_candidate_heads else 2,
            "result": {
                "ablated_head": head_id,
                "behavior_delta": delta,
                "causal_delta_threshold": PLANTED_LITE_CAUSAL_DELTA_THRESHOLD,
                "causal_verified": delta >= PLANTED_LITE_CAUSAL_DELTA_THRESHOLD,
                "head": head,
                "layer": layer,
                "candidate_heads": candidate_heads,
                "ablated_candidate_heads": ablated_candidate_heads,
                "remaining_candidate_heads": remaining_candidate_heads,
                "best_ablated_head_so_far": best_head,
                "best_ablated_delta_so_far": deltas[best_head],
                "must_submit": best_head if not remaining_candidate_heads else None,
            },
            "scenario_id": "planted_lite_causal_chain",
            "step_count": 2 if remaining_candidate_heads else 3,
            "summary": f"Ablated {head_id}; behavior_delta={delta}.",
        },
        sort_keys=True,
    )


def planted_candidate_order(
    *,
    target_head: str,
    decoy_heads: list[str],
    target_position: int,
) -> list[str]:
    if len(decoy_heads) < 2:
        raise ValueError("Planted SFT examples require at least two decoy heads.")
    if target_position == 1:
        return [decoy_heads[0], target_head, decoy_heads[1]]
    if target_position == 2:
        return [decoy_heads[0], decoy_heads[1], target_head]
    raise ValueError("target_position must be 1 or 2 so the top-ranked head stays a decoy.")


def synthetic_planted_inspect_response(
    *,
    target_head: str,
    decoy_heads: list[str],
    target_position: int = 2,
) -> str:
    scores: list[dict[str, float | int | str]] = []
    candidate_order = planted_candidate_order(
        target_head=target_head,
        decoy_heads=decoy_heads,
        target_position=target_position,
    )
    for score, head_id in zip([0.96, 0.88, 0.82], candidate_order):
        layer, head = parse_head_id(head_id)
        scores.append(
            {
                "head": head,
                "head_id": head_id,
                "layer": layer,
                "score": score,
            }
        )
    return json.dumps(
        {
            "available_tools": [
                "list_tools",
                "run_probe",
                "inspect_induction_scores",
                "ablate_head",
                "submit_circuit",
            ],
            "done": False,
            "remaining_budget": 9,
            "result": {"scores": scores},
            "scenario_id": "planted_circuit_arena",
            "step_count": 1,
            "summary": (
                "Returned noisy planted-arena scores. The top-ranked head may be a decoy; "
                "use ablation deltas before submitting."
            ),
        },
        sort_keys=True,
    )


def synthetic_planted_ablation_response(
    *,
    head_id: str,
    is_target: bool,
) -> str:
    layer, head = parse_head_id(head_id)
    delta = 0.47 if is_target else 0.018
    causal_verified = delta >= PLANTED_CAUSAL_DELTA_THRESHOLD
    return json.dumps(
        {
            "available_tools": [
                "list_tools",
                "run_probe",
                "inspect_induction_scores",
                "ablate_head",
                "submit_circuit",
            ],
            "done": False,
            "remaining_budget": 8,
            "result": {
                "ablated_head": head_id,
                "behavior_delta": delta,
                "causal_delta_threshold": PLANTED_CAUSAL_DELTA_THRESHOLD,
                "causal_verified": causal_verified,
                "head": head,
                "layer": layer,
            },
            "scenario_id": "planted_circuit_arena",
            "step_count": 2,
            "summary": f"Ablated {head_id}; behavior_delta={delta}.",
        },
        sort_keys=True,
    )


def ioi_target_heads() -> list[str]:
    return ["L9H9", "L9H6", "L10H0"]


def real_ioi_expert_ablation_heads() -> list[str]:
    """Heads an expert trace should test in the real GPT-2 IOI backend."""
    return ["L8H10", "L10H0", "L9H9", "L9H6"]


def synthetic_ioi_inspect_response(*, scenario_id: str = "ioi_gpt2_small_name_mover") -> str:
    if scenario_id == "ioi_gpt2_small_real":
        ranked_heads = [
            ("L8H10", 0.6778),
            ("L8H6", 0.5857),
            ("L9H7", 0.3173),
            ("L7H3", 0.2739),
            ("L10H0", 0.2103),
            ("L10H6", 0.1353),
            ("L9H9", 0.1215),
            ("L10H10", -0.0033),
            ("L9H6", -0.0120),
            ("L10H7", -0.0190),
            ("L8H11", -0.0250),
            ("L7H9", -0.0310),
        ]
        summary = (
            "Returned real GPT-2 IOI ablation effects. The largest single-head "
            "effect is not the complete name-mover circuit; test and submit the "
            "name-mover heads."
        )
    else:
        ranked_heads = [
            ("L9H9", 0.97),
            ("L10H0", 0.94),
            ("L9H6", 0.92),
            ("L10H10", 0.74),
            ("L10H6", 0.71),
            ("L7H3", 0.68),
            ("L8H10", 0.65),
            ("L10H7", 0.61),
        ]
        summary = "Returned candidate IOI head effects for the name-mover component."
    scores = []
    for head_id, score in ranked_heads:
        layer, head = parse_head_id(head_id)
        scores.append(
            {
                "head": head,
                "head_id": head_id,
                "layer": layer,
                "score": score,
            }
        )
    return json.dumps(
        {
            "available_tools": [
                "list_tools",
                "run_probe",
                "inspect_induction_scores",
                "ablate_head",
                "submit_circuit",
            ],
            "done": False,
            "remaining_budget": 11,
            "result": {"scores": scores},
            "scenario_id": scenario_id,
            "step_count": 1,
            "summary": summary,
        },
        sort_keys=True,
    )


def synthetic_ioi_ablation_response(
    head_id: str,
    *,
    scenario_id: str = "ioi_gpt2_small_name_mover",
) -> str:
    layer, head = parse_head_id(head_id)
    if scenario_id == "ioi_gpt2_small_real":
        threshold = 0.01
        delta = {
            "L8H10": 0.6778,
            "L10H0": 0.2103,
            "L9H9": 0.1215,
            "L9H6": 0.0180,
        }.get(head_id, -0.01)
    else:
        threshold = 0.10
        delta = {"L9H9": 0.34, "L10H0": 0.31, "L9H6": 0.29}.get(head_id, 0.02)
    return json.dumps(
        {
            "available_tools": [
                "list_tools",
                "run_probe",
                "inspect_induction_scores",
                "ablate_head",
                "submit_circuit",
            ],
            "done": False,
            "remaining_budget": 10,
            "result": {
                "ablated_head": head_id,
                "behavior_delta": delta,
                "causal_delta_threshold": threshold,
                "causal_verified": delta >= threshold,
                "head": head,
                "layer": layer,
            },
            "scenario_id": scenario_id,
            "step_count": 2,
            "summary": f"Ablated IOI candidate {head_id}; behavior_delta={delta}.",
        },
        sort_keys=True,
    )


def synthetic_inspect_response(target_head: str, *, scenario: str = "phase1") -> str:
    return json.dumps(
        {
            "available_tools": [
                "list_tools",
                "run_probe",
                "inspect_induction_scores",
                "ablate_head",
                "submit_circuit",
            ],
            "done": False,
            "remaining_budget": 11,
            "result": {
                "scores": [
                    {
                        "head": int(target_head.split("H", maxsplit=1)[1]),
                        "head_id": target_head,
                        "induction_score": 0.42,
                        "layer": int(target_head.split("H", maxsplit=1)[0].removeprefix("L")),
                    },
                    {"head": 3, "head_id": "L1H3", "induction_score": 0.11, "layer": 1},
                    {"head": 0, "head_id": "L0H0", "induction_score": 0.04, "layer": 0},
                ]
            },
            "scenario_id": "l2_ablation_required" if scenario == "phase2" else "l1_induction_attn_only_2l",
            "step_count": 1,
            "summary": (
                "Returned the top heads ranked by induction score. "
                f"The strongest supported head is {target_head}."
            ),
        },
        sort_keys=True,
    )


def synthetic_ablation_response(target_head: str, *, scenario: str = "phase1") -> str:
    layer = int(target_head.split("H", maxsplit=1)[0].removeprefix("L"))
    head = int(target_head.split("H", maxsplit=1)[1])
    return json.dumps(
        {
            "available_tools": [
                "list_tools",
                "run_probe",
                "inspect_induction_scores",
                "ablate_head",
                "submit_circuit",
            ],
            "done": False,
            "remaining_budget": 10,
            "result": {
                "ablated_head": target_head,
                "behavior_delta": 0.31,
                "causal_verified": True,
                "head": head,
                "layer": layer,
            },
            "scenario_id": "l2_ablation_required" if scenario == "phase2" else "l1_induction_attn_only_2l",
            "step_count": 2,
            "summary": f"Ablated {target_head}; induction behavior dropped sharply.",
        },
        sort_keys=True,
    )


def build_sft_records(
    *,
    tokenizer: Any,
    examples_per_prompt: int,
    target_head: str,
    scenario: str = "phase1",
) -> list[dict[str, str]]:
    tool_env = CircuitDetectiveToolEnv()
    tools = [
        tool_env.list_tools,
        tool_env.run_probe,
        tool_env.inspect_induction_scores,
        tool_env.ablate_head,
        tool_env.submit_circuit,
    ]
    target_layer = int(target_head.split("H", maxsplit=1)[0].removeprefix("L"))
    target_head_index = int(target_head.split("H", maxsplit=1)[1])

    records: list[dict[str, str]] = []
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

    record_index = 0
    for user_prompt in user_prompts:
        for repeat in range(examples_per_prompt):
            record_scenario = scenario
            if scenario == "curriculum":
                record_scenario = "planted" if record_index % 2 == 0 else "ioi"
            reset_observation = synthetic_reset_observation(scenario=record_scenario)

            if record_scenario == "planted_lite":
                planted_target, planted_decoy = planted_lite_heads_for_record(record_index)
                planted_inspect = synthetic_planted_lite_inspect_response(
                    target_head=planted_target,
                    decoy_head=planted_decoy,
                )
                decoy_layer, decoy_head_index = parse_head_id(planted_decoy)
                target_layer_index, target_head_index = parse_head_id(planted_target)
                decoy_ablation = synthetic_planted_lite_ablation_response(
                    head_id=planted_decoy,
                    target_head=planted_target,
                    decoy_head=planted_decoy,
                    ablated_heads_so_far=[],
                )
                target_ablation = synthetic_planted_lite_ablation_response(
                    head_id=planted_target,
                    target_head=planted_target,
                    decoy_head=planted_decoy,
                    ablated_heads_so_far=[planted_decoy],
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{user_prompt}\n{reset_observation}"},
                    {
                        "role": "assistant",
                        "content": tool_call("inspect_induction_scores", {"top_k": 2}),
                    },
                    {"role": "user", "content": f"<tool_response>\n{planted_inspect}\n</tool_response>"},
                    {
                        "role": "assistant",
                        "content": tool_call(
                            "ablate_head",
                            {"layer": decoy_layer, "head": decoy_head_index},
                        ),
                    },
                    {"role": "user", "content": f"<tool_response>\n{decoy_ablation}\n</tool_response>"},
                    {
                        "role": "assistant",
                        "content": tool_call(
                            "ablate_head",
                            {"layer": target_layer_index, "head": target_head_index},
                        ),
                    },
                    {"role": "user", "content": f"<tool_response>\n{target_ablation}\n</tool_response>"},
                    {
                        "role": "assistant",
                        "content": tool_call("submit_circuit", {"heads": [planted_target]}),
                    },
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=False,
                    chat_template_kwargs={"enable_thinking": False},
                )
                records.append({"text": text})
                record_index += 1
                continue

            if record_scenario == "planted":
                planted_target, planted_decoys = planted_heads_for_record(record_index)
                target_position = 1 if record_index % 2 == 0 else 2
                candidate_order = planted_candidate_order(
                    target_head=planted_target,
                    decoy_heads=planted_decoys,
                    target_position=target_position,
                )
                planted_inspect = synthetic_planted_inspect_response(
                    target_head=planted_target,
                    decoy_heads=planted_decoys,
                    target_position=target_position,
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{user_prompt}\n{reset_observation}"},
                    {
                        "role": "assistant",
                        "content": tool_call("inspect_induction_scores", {"top_k": 3}),
                    },
                    {"role": "user", "content": f"<tool_response>\n{planted_inspect}\n</tool_response>"},
                ]
                for candidate_head in candidate_order:
                    candidate_layer, candidate_index = parse_head_id(candidate_head)
                    is_target = candidate_head == planted_target
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": tool_call(
                                    "ablate_head",
                                    {"layer": candidate_layer, "head": candidate_index},
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    "<tool_response>\n"
                                    f"{synthetic_planted_ablation_response(head_id=candidate_head, is_target=is_target)}\n"
                                    "</tool_response>"
                                ),
                            },
                        ]
                    )
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": tool_call("submit_circuit", {"heads": [planted_target]}),
                        },
                    ]
                )
                text = tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=False,
                    chat_template_kwargs={"enable_thinking": False},
                )
                records.append({"text": text})
                record_index += 1
                continue

            if record_scenario in {"ioi", "real_ioi"}:
                ioi_scenario_id = (
                    "ioi_gpt2_small_real"
                    if record_scenario == "real_ioi"
                    else "ioi_gpt2_small_name_mover"
                )
                inspect_top_k = 12 if record_scenario == "real_ioi" else 8
                ablation_heads = (
                    real_ioi_expert_ablation_heads()
                    if record_scenario == "real_ioi"
                    else ioi_target_heads()
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{user_prompt}\n{reset_observation}"},
                    {
                        "role": "assistant",
                        "content": tool_call("inspect_induction_scores", {"top_k": inspect_top_k}),
                    },
                    {
                        "role": "user",
                        "content": (
                            "<tool_response>\n"
                            f"{synthetic_ioi_inspect_response(scenario_id=ioi_scenario_id)}\n"
                            "</tool_response>"
                        ),
                    },
                ]
                for head_id in ablation_heads:
                    layer, head = parse_head_id(head_id)
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": tool_call(
                                    "ablate_head",
                                    {"layer": layer, "head": head},
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    "<tool_response>\n"
                                    f"{synthetic_ioi_ablation_response(head_id, scenario_id=ioi_scenario_id)}\n"
                                    "</tool_response>"
                                ),
                            },
                        ]
                    )
                messages.append(
                    {
                        "role": "assistant",
                        "content": tool_call("submit_circuit", {"heads": ioi_target_heads()}),
                    }
                )
                text = tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=False,
                    chat_template_kwargs={"enable_thinking": False},
                )
                records.append({"text": text})
                record_index += 1
                continue

            inspect_response = synthetic_inspect_response(target_head, scenario=record_scenario)
            ablation_response = synthetic_ablation_response(target_head, scenario=record_scenario)
            use_ablation = record_scenario == "phase2" or repeat % 2 == 1
            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prompt}\n{reset_observation}"},
                {
                    "role": "assistant",
                    "content": tool_call("inspect_induction_scores", {"top_k": 3}),
                },
                {"role": "user", "content": f"<tool_response>\n{inspect_response}\n</tool_response>"},
            ]

            if use_ablation:
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": tool_call(
                                "ablate_head",
                                {"layer": target_layer, "head": target_head_index},
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"<tool_response>\n{ablation_response}\n</tool_response>",
                        },
                    ]
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": tool_call("submit_circuit", {"heads": [target_head]}),
                }
            )
            text = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
                chat_template_kwargs={"enable_thinking": False},
            )
            records.append({"text": text})
            record_index += 1

    return records


def main() -> None:
    args = parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Phase 1 SFT warm-start requires a GPU because "
            "it uses 4-bit QLoRA and bitsandbytes. Try a different HF Jobs flavor "
            "or image; do not continue on CPU."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    records = build_sft_records(
        tokenizer=tokenizer,
        examples_per_prompt=args.examples_per_prompt,
        target_head=args.target_head,
        scenario=args.scenario,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "warmup_examples.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )

    peft_config = None
    if args.adapter_path:
        model = PeftModel.from_pretrained(
            model,
            args.adapter_path,
            is_trainable=True,
        )
    else:
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = SFTConfig(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_steps=max(args.max_steps, 1),
        max_length=args.max_seq_length,
        packing=False,
        dataset_text_field="text",
        optim="paged_adamw_8bit",
        bf16=bf16,
        fp16=not bf16,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(records),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final_adapter"))
    print(f"saved_sft_adapter: {output_dir / 'final_adapter'}", flush=True)


if __name__ == "__main__":
    main()
