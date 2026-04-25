from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from circuit_detective.phase1_grpo import (
    CircuitDetectiveToolEnv,
    build_phase1_dataset,
    reward_func,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 1 GRPO pilot.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--repeats-per-prompt", type=int, default=16)
    parser.add_argument("--output-dir", default="outputs/phase1_qwen35_2b_grpo")
    parser.add_argument("--artifact-dir", default="artifacts/phase1")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=384)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
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

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys)
    plt.xlabel("step")
    plt.ylabel(key)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


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


def main() -> None:
    args = parse_args()

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
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    train_dataset = build_phase1_dataset(repeats_per_prompt=args.repeats_per_prompt)
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=5e-6,
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
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        generation_batch_size=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        save_steps=max(args.max_steps // 2, 1),
        max_grad_norm=0.1,
        report_to="none",
        log_completions=True,
        chat_template_kwargs={"enable_thinking": False},
        use_vllm=False,
    )
    trainer_kwargs = {}
    if args.backend == "trl":
        trainer_kwargs["peft_config"] = peft_config

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=train_dataset,
        environment_factory=CircuitDetectiveToolEnv,
        **trainer_kwargs,
    )
    trainer.train()
    trainer.save_model(f"{args.output_dir}/final_adapter")
    save_training_curves(trainer.state.log_history, Path(args.artifact_dir))


if __name__ == "__main__":
    main()
