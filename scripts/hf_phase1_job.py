from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import create_repo, snapshot_download, upload_folder


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF Jobs launcher for Phase 1 smoke.")
    parser.add_argument("--repo-id", default="ehsaaniqbal/circuit-detective")
    parser.add_argument("--workdir", default="/tmp/circuit_detective")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--repeats-per-prompt", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/hf_phase1_smoke")
    parser.add_argument("--artifact-dir", default="artifacts/hf_phase1_smoke")
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--eval-generations", type=int, default=4)
    parser.add_argument("--eval-prompts", type=int, default=2)
    parser.add_argument("--max-tool-calling-iterations", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=8e-6)
    parser.add_argument("--log-completions", action="store_true")
    parser.add_argument("--sft-warmup", action="store_true")
    parser.add_argument("--sft-output-dir", default="outputs/phase1_sft_warmup")
    parser.add_argument("--sft-max-steps", type=int, default=32)
    parser.add_argument("--sft-examples-per-prompt", type=int, default=4)
    parser.add_argument("--sft-learning-rate", type=float, default=2e-5)
    parser.add_argument("--sft-target-head", default="L1H6")
    parser.add_argument(
        "--adapter-path",
        default=None,
        help=(
            "Optional PEFT adapter path or Hub model id to continue GRPO from. "
            "When --sft-warmup is set, SFT first continues from this adapter and "
            "GRPO then uses the freshly SFT-tuned adapter."
        ),
    )
    parser.add_argument("--backend", choices=["trl", "unsloth"], default="trl")
    parser.add_argument(
        "--scenario",
        choices=["phase1", "phase2", "planted", "ioi", "curriculum", "real_ioi"],
        default="phase1",
        help="Training curriculum level to pass through to SFT/GRPO scripts.",
    )
    parser.add_argument("--eval-before-after", action="store_true")
    parser.add_argument("--upload-artifacts", action="store_true")
    parser.add_argument("--artifact-repo-path", default="artifacts/hf_phase1_smoke")
    parser.add_argument(
        "--upload-adapter",
        action="store_true",
        help="Upload the final GRPO LoRA adapter to a Hugging Face model repo.",
    )
    parser.add_argument(
        "--adapter-repo-id",
        default=None,
        help="Model repo id for --upload-adapter, for example user/model-name.",
    )
    parser.add_argument(
        "--adapter-repo-path",
        default="",
        help="Optional folder path inside the adapter model repo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="space",
        local_dir=workdir,
        ignore_patterns=[
            ".git/*",
            ".venv*/*",
            "__pycache__/*",
            "*.pyc",
            "outputs/*",
            "checkpoints/*",
            "wandb/*",
        ],
    )

    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], cwd=workdir)
    run([sys.executable, "-m", "pip", "install", "uv"], cwd=workdir)
    run(["uv", "pip", "install", "--system", "-e", "."], cwd=workdir)
    run(["uv", "venv", ".venv-tlens", "--python", "3.11"], cwd=workdir)
    run(
        [
            "uv",
            "pip",
            "install",
            "--torch-backend",
            "cpu",
            "--python",
            ".venv-tlens/bin/python",
            "transformer-lens==2.18.0",
        ],
        cwd=workdir,
    )
    artifact_dir = Path(args.artifact_dir)
    adapter_args: list[str] = []
    if args.sft_warmup:
        if args.backend != "trl":
            raise ValueError("--sft-warmup currently supports --backend trl only.")
        sft_command = [
            sys.executable,
            "scripts/phase1_sft.py",
            "--output-dir",
            args.sft_output_dir,
            "--max-steps",
            str(args.sft_max_steps),
            "--examples-per-prompt",
            str(args.sft_examples_per_prompt),
            "--learning-rate",
            str(args.sft_learning_rate),
            "--target-head",
            args.sft_target_head,
            "--scenario",
            args.scenario,
        ]
        if args.adapter_path:
            sft_command.extend(["--adapter-path", args.adapter_path])
        run(sft_command, cwd=workdir)
        adapter_args = ["--adapter-path", str(Path(args.sft_output_dir) / "final_adapter")]
    elif args.adapter_path:
        adapter_args = ["--adapter-path", args.adapter_path]

    run(
        [
            sys.executable,
            "scripts/phase1_train.py",
            "--max-steps",
            str(args.max_steps),
            "--repeats-per-prompt",
            str(args.repeats_per_prompt),
            "--backend",
            args.backend,
            "--max-completion-length",
            str(args.max_completion_length),
            "--num-generations",
            str(args.num_generations),
            "--eval-generations",
            str(args.eval_generations),
            "--eval-prompts",
            str(args.eval_prompts),
            "--max-tool-calling-iterations",
            str(args.max_tool_calling_iterations),
            "--learning-rate",
            str(args.learning_rate),
            "--scenario",
            args.scenario,
            *adapter_args,
            "--output-dir",
            args.output_dir,
            "--artifact-dir",
            str(artifact_dir),
        ]
        + (["--eval-before-after"] if args.eval_before_after else [])
        + (["--log-completions"] if args.log_completions else []),
        cwd=workdir,
    )
    if args.upload_artifacts:
        uploaded = upload_folder(
            repo_id=args.repo_id,
            repo_type="space",
            folder_path=workdir / artifact_dir,
            path_in_repo=args.artifact_repo_path,
            commit_message="Upload Phase 1 HF job artifacts",
        )
        print(f"uploaded_artifacts: {uploaded}", flush=True)

    if args.upload_adapter:
        if not args.adapter_repo_id:
            raise ValueError("--upload-adapter requires --adapter-repo-id.")
        adapter_dir = workdir / args.output_dir / "final_adapter"
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Final adapter directory not found: {adapter_dir}")

        create_repo(
            repo_id=args.adapter_repo_id,
            repo_type="model",
            exist_ok=True,
            private=False,
        )
        upload_kwargs = {
            "repo_id": args.adapter_repo_id,
            "repo_type": "model",
            "folder_path": adapter_dir,
            "commit_message": "Upload Phase 1 GRPO LoRA adapter",
        }
        if args.adapter_repo_path:
            upload_kwargs["path_in_repo"] = args.adapter_repo_path
        uploaded = upload_folder(**upload_kwargs)
        print(f"uploaded_adapter: {uploaded}", flush=True)


if __name__ == "__main__":
    main()
