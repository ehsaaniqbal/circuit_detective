from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from circuit_detective.phase1_grpo import (
    PHASE1_SYSTEM_PROMPT,
    PHASE1_USER_PROMPT_VARIANTS,
    PHASE2_SYSTEM_PROMPT,
    PHASE2_USER_PROMPT_VARIANTS,
)
from circuit_detective.phase1_grpo import CircuitDetectiveToolEnv


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
        choices=["phase1", "phase2"],
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
            "remaining_budget": 12,
            "result": {
                "goal": (
                    "Inspect the dominant induction head, ablate that candidate, "
                    "then submit the verified head as ['LxHy']."
                    if is_phase2
                    else "Submit the dominant induction head as ['LxHy']."
                ),
                "requires_ablation": is_phase2,
                "scenario": "l1_induction_attn_only_2l",
            },
            "scenario_id": "l2_ablation_required" if is_phase2 else "l1_induction_attn_only_2l",
            "step_count": 0,
            "summary": (
                "Phase 2: localize and causally verify the dominant induction head. "
                "Use inspect_induction_scores, ablate_head, then submit_circuit."
                if is_phase2
                else (
                    "Phase 1: localize the dominant induction head. "
                    "Use inspect_induction_scores, then submit_circuit."
                )
            ),
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
    reset_observation = synthetic_reset_observation(scenario=scenario)
    inspect_response = synthetic_inspect_response(target_head, scenario=scenario)
    ablation_response = synthetic_ablation_response(target_head, scenario=scenario)
    target_layer = int(target_head.split("H", maxsplit=1)[0].removeprefix("L"))
    target_head_index = int(target_head.split("H", maxsplit=1)[1])

    records: list[dict[str, str]] = []
    if scenario == "phase2":
        system_prompt = PHASE2_SYSTEM_PROMPT
        user_prompts = PHASE2_USER_PROMPT_VARIANTS
    else:
        system_prompt = PHASE1_SYSTEM_PROMPT
        user_prompts = PHASE1_USER_PROMPT_VARIANTS

    for user_prompt in user_prompts:
        for repeat in range(examples_per_prompt):
            use_ablation = scenario == "phase2" or repeat % 2 == 1
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
