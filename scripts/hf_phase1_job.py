from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF Jobs launcher for Phase 1 smoke.")
    parser.add_argument("--repo-id", default="ehsaaniqbal/circuit-detective")
    parser.add_argument("--workdir", default="/tmp/circuit_detective")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--repeats-per-prompt", type=int, default=1)
    parser.add_argument("--backend", choices=["trl", "unsloth"], default="trl")
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
            "--python",
            ".venv-tlens/bin/python",
            "transformer-lens==2.18.0",
        ],
        cwd=workdir,
    )
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
            "--output-dir",
            "outputs/hf_phase1_smoke",
            "--artifact-dir",
            "artifacts/hf_phase1_smoke",
        ],
        cwd=workdir,
    )


if __name__ == "__main__":
    main()
