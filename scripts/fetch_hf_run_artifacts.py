from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download one HF Space artifact run and summarize eval metrics."
    )
    parser.add_argument("artifact_path", help="Path in the Space, e.g. artifacts/run_name")
    parser.add_argument("--repo-id", default="ehsaaniqbal/circuit-detective")
    parser.add_argument("--output-root", default="/tmp/circuit_detective_hf_metrics")
    parser.add_argument("--success-threshold", type=float, default=0.40)
    parser.add_argument("--min-rollouts", type=float, default=8.0)
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Only download artifacts; do not run scripts/analyze_phase1_run.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_path = args.artifact_path.strip("/")
    local_root = Path(args.output_root)
    local_dir = local_root / artifact_path

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="space",
        local_dir=local_root,
        allow_patterns=[f"{artifact_path}/*"],
    )

    metrics_path = local_dir / "phase1_eval_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Downloaded artifacts, but no phase1_eval_metrics.json found at {metrics_path}"
        )

    print(f"downloaded: {local_dir}", flush=True)
    if args.no_summary:
        return

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_phase1_run.py",
            str(local_dir),
            "--success-threshold",
            str(args.success_threshold),
            "--min-rollouts",
            str(args.min_rollouts),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
