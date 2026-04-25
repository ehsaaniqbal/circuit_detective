from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download, upload_folder


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
    parser.add_argument("--backend", choices=["trl", "unsloth"], default="trl")
    parser.add_argument("--eval-before-after", action="store_true")
    parser.add_argument("--upload-artifacts", action="store_true")
    parser.add_argument("--artifact-repo-path", default="artifacts/hf_phase1_smoke")
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


if __name__ == "__main__":
    main()
