from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from circuit_detective.phase1_grpo import (
    CircuitDetectiveToolEnv,
    CurriculumCircuitToolEnv,
    IOICircuitToolEnv,
    Phase2CircuitDetectiveToolEnv,
    PlantedCircuitToolEnv,
    PlantedLiteCircuitToolEnv,
    RealIOICircuitToolEnv,
    build_phase1_dataset,
    consume_reward_trace,
    reward_func,
    reset_reward_trace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 1 GRPO pilot.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--repeats-per-prompt", type=int, default=16)
    parser.add_argument("--output-dir", default="outputs/phase1_qwen35_2b_grpo")
    parser.add_argument("--artifact-dir", default="artifacts/phase1")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--eval-generations", type=int, default=4)
    parser.add_argument("--eval-prompts", type=int, default=2)
    parser.add_argument("--max-tool-calling-iterations", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=8e-6)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument(
        "--scenario",
        choices=["phase1", "phase2", "planted", "planted_lite", "ioi", "curriculum", "real_ioi"],
        default="phase1",
        help="Training curriculum level. phase2 requires ablation before full credit.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional PEFT adapter directory to continue training from.",
    )
    parser.add_argument(
        "--log-completions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log full completion tables during training. Useful for debugging, noisy for longer runs.",
    )
    parser.add_argument(
        "--eval-before-after",
        action="store_true",
        help="Run a small GRPO environment evaluation before and after training.",
    )
    parser.add_argument(
        "--backend",
        choices=["trl", "unsloth"],
        default="trl",
        help="Training backend. `trl` is the current smoke default; `unsloth` is optional.",
    )
    return parser.parse_args()


def save_curve(
    *,
    log_history: list[dict[str, object]],
    key: str,
    path: Path,
    title: str,
) -> None:
    xs = [
        int(entry["step"])
        for entry in log_history
        if "step" in entry and key in entry
    ]
    ys = [
        float(entry[key])
        for entry in log_history
        if "step" in entry and key in entry
    ]
    if not xs:
        available_keys = sorted({key for entry in log_history for key in entry})
        raise ValueError(f"No values found for {key!r}. Available keys: {available_keys}")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys)
    plt.xlabel("step")
    plt.ylabel(key)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"saved: {path}", flush=True)


def save_training_curves(log_history: list[dict[str, object]], artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_curve(
        log_history=log_history,
        key="loss",
        path=artifact_dir / "phase1_loss_curve.png",
        title="Phase 1 Loss",
    )

    reward_key = next(
        (
            key
            for key in [
                "reward",
                "rewards",
                "mean_reward",
                "objective/reward",
                "reward/mean",
            ]
            if any(key in entry for entry in log_history)
        ),
        None,
    )
    if reward_key is None:
        available_keys = sorted({key for entry in log_history for key in entry})
        raise ValueError(f"No reward-like key found. Available keys: {available_keys}")

    save_curve(
        log_history=log_history,
        key=reward_key,
        path=artifact_dir / "phase1_reward_curve.png",
        title=f"Phase 1 Reward ({reward_key})",
    )


def save_eval_metrics(
    *,
    before: dict[str, object] | None,
    after: dict[str, object] | None,
    artifact_dir: Path,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "before": before,
        "after": after,
    }
    (artifact_dir / "phase1_eval_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"saved: {artifact_dir / 'phase1_eval_metrics.json'}", flush=True)


def summarize_reward_trace(
    records: list[dict[str, object]],
    *,
    prefix: str,
) -> dict[str, float]:
    if not records:
        return {
            f"{prefix}_rollouts": 0.0,
            f"{prefix}_mean_reward": 0.0,
            f"{prefix}_submit_rate": 0.0,
            f"{prefix}_success_rate": 0.0,
            f"{prefix}_mean_f1": 0.0,
            f"{prefix}_mean_terminal_reward": 0.0,
            f"{prefix}_mean_tool_calls": 0.0,
            f"{prefix}_probe_rate": 0.0,
            f"{prefix}_inspect_rate": 0.0,
            f"{prefix}_ablate_rate": 0.0,
            f"{prefix}_ablate_submitted_rate": 0.0,
            f"{prefix}_causal_success_rate": 0.0,
            f"{prefix}_mean_ablation_faithfulness": 0.0,
            f"{prefix}_all_candidates_ablated_rate": 0.0,
            f"{prefix}_mean_candidate_ablation_coverage": 0.0,
        }

    count = float(len(records))

    def mean(key: str) -> float:
        return sum(float(record.get(key, 0.0)) for record in records) / count

    def rate(key: str) -> float:
        return sum(1.0 for record in records if bool(record.get(key))) / count

    return {
        f"{prefix}_rollouts": count,
        f"{prefix}_mean_reward": mean("reward"),
        f"{prefix}_submit_rate": rate("submitted"),
        f"{prefix}_success_rate": rate("correct"),
        f"{prefix}_mean_f1": mean("f1"),
        f"{prefix}_mean_terminal_reward": mean("terminal_reward"),
        f"{prefix}_mean_tool_calls": mean("tool_calls"),
        f"{prefix}_probe_rate": rate("used_probe"),
        f"{prefix}_inspect_rate": rate("used_inspect"),
        f"{prefix}_ablate_rate": rate("used_ablate"),
        f"{prefix}_ablate_submitted_rate": rate("ablate_submitted"),
        f"{prefix}_causal_success_rate": rate("causal_success"),
        f"{prefix}_mean_ablation_faithfulness": mean("ablation_faithfulness"),
        f"{prefix}_all_candidates_ablated_rate": rate("all_candidates_ablated"),
        f"{prefix}_mean_candidate_ablation_coverage": mean("candidate_ablation_coverage"),
    }


def evaluate_with_rollout_metrics(
    trainer: object,
    *,
    metric_key_prefix: str,
) -> dict[str, object]:
    reset_reward_trace()
    metrics = trainer.evaluate(metric_key_prefix=metric_key_prefix)
    rollout_records = consume_reward_trace()
    metrics.update(
        summarize_reward_trace(
            rollout_records,
            prefix=metric_key_prefix,
        )
    )
    metrics[f"{metric_key_prefix}_sample_rollouts"] = rollout_records[:8]
    return dict(metrics)


def main() -> None:
    args = parse_args()
    if args.eval_before_after and args.eval_generations != args.num_generations:
        print(
            "For TRL environment_factory evaluation, --eval-generations must match "
            "--num-generations. Overriding eval generations to "
            f"{args.num_generations}.",
            flush=True,
        )
        args.eval_generations = args.num_generations

    from trl import GRPOConfig, GRPOTrainer

    if args.backend == "unsloth":
        from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported

        PatchFastRL("GRPO", FastLanguageModel)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
            fast_inference=False,
            max_lora_rank=args.lora_rank,
            gpu_memory_utilization=0.6,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=args.lora_rank,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        bf16 = is_bfloat16_supported()
    else:
        import torch
        from peft import LoraConfig
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Phase 1 GRPO training requires a GPU "
                "because it uses 4-bit QLoRA and bitsandbytes. Try a different "
                "HF Jobs flavor or image; do not continue on CPU."
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
        if args.adapter_path:
            model = PeftModel.from_pretrained(
                model,
                args.adapter_path,
                is_trainable=True,
            )
            peft_config = None
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    train_dataset = build_phase1_dataset(
        repeats_per_prompt=args.repeats_per_prompt,
        scenario=args.scenario,
    )
    eval_count = max(args.eval_prompts, args.eval_generations)
    eval_count = math.ceil(eval_count / args.eval_generations) * args.eval_generations
    base_eval_dataset = build_phase1_dataset(
        repeats_per_prompt=1,
        scenario=args.scenario,
    )
    eval_repeats = math.ceil(eval_count / len(base_eval_dataset))
    eval_dataset = build_phase1_dataset(
        repeats_per_prompt=eval_repeats,
        scenario=args.scenario,
    )
    eval_dataset = eval_dataset.select(range(eval_count))
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_steps=max(args.max_steps // 10, 1),
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=bf16,
        fp16=not bf16,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=args.eval_generations,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        num_generations_eval=args.eval_generations,
        generation_batch_size=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        save_steps=max(args.max_steps // 2, 1),
        max_grad_norm=0.1,
        report_to="none",
        log_completions=args.log_completions,
        chat_template_kwargs={"enable_thinking": False},
        use_vllm=False,
        max_tool_calling_iterations=args.max_tool_calling_iterations,
    )
    trainer_kwargs = {}
    if args.backend == "trl" and peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        environment_factory={
            "phase1": CircuitDetectiveToolEnv,
            "phase2": Phase2CircuitDetectiveToolEnv,
            "planted": PlantedCircuitToolEnv,
            "planted_lite": PlantedLiteCircuitToolEnv,
            "ioi": IOICircuitToolEnv,
            "curriculum": CurriculumCircuitToolEnv,
            "real_ioi": RealIOICircuitToolEnv,
        }[args.scenario],
        **trainer_kwargs,
    )

    before_metrics = None
    if args.eval_before_after:
        before_metrics = evaluate_with_rollout_metrics(
            trainer,
            metric_key_prefix="eval_before",
        )
        print(json.dumps({"eval_before": before_metrics}, indent=2, sort_keys=True), flush=True)

    reset_reward_trace()
    trainer.train()

    after_metrics = None
    if args.eval_before_after:
        after_metrics = evaluate_with_rollout_metrics(
            trainer,
            metric_key_prefix="eval_after",
        )
        print(json.dumps({"eval_after": after_metrics}, indent=2, sort_keys=True), flush=True)

    trainer.save_model(f"{args.output_dir}/final_adapter")
    artifact_dir = Path(args.artifact_dir)
    save_training_curves(trainer.state.log_history, artifact_dir)
    if args.eval_before_after:
        save_eval_metrics(
            before=before_metrics,
            after=after_metrics,
            artifact_dir=artifact_dir,
        )


if __name__ == "__main__":
    main()
