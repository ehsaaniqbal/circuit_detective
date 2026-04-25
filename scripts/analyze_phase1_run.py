from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


METRIC_NAMES = [
    "success_rate",
    "submit_rate",
    "mean_reward",
    "mean_f1",
    "inspect_rate",
    "probe_rate",
    "ablate_rate",
    "rollouts",
]


@dataclass(frozen=True)
class Phase1Summary:
    name: str
    metrics_path: Path
    before: dict[str, float | None]
    after: dict[str, float | None]
    deltas: dict[str, float | None]
    has_loss_curve: bool
    has_reward_curve: bool
    gate_passed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Phase 1 eval metrics from one or more artifact directories."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["artifacts"],
        help="Artifact directory, phase1_eval_metrics.json, or parent directory containing runs.",
    )
    parser.add_argument("--success-threshold", type=float, default=0.40)
    parser.add_argument("--min-rollouts", type=float, default=32.0)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a compact text report.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if no summarized run passes the configured gate.",
    )
    return parser.parse_args()


def find_metric_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            if path.name != "phase1_eval_metrics.json":
                raise ValueError(f"Expected phase1_eval_metrics.json, got {path}")
            files.append(path)
        elif path.is_dir():
            direct = path / "phase1_eval_metrics.json"
            if direct.exists():
                files.append(direct)
            else:
                files.extend(sorted(path.glob("*/phase1_eval_metrics.json")))
        else:
            raise FileNotFoundError(path)

    return sorted(dict.fromkeys(files))


def metric_value(section: dict[str, Any] | None, prefix: str, metric: str) -> float | None:
    if not section:
        return None

    value = section.get(f"{prefix}_{metric}")
    if value is None:
        return None
    return float(value)


def summarize_file(
    metrics_path: Path,
    *,
    success_threshold: float,
    min_rollouts: float,
) -> Phase1Summary:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    before_section = payload.get("before")
    after_section = payload.get("after")

    before = {
        metric: metric_value(before_section, "eval_before", metric)
        for metric in METRIC_NAMES
    }
    after = {
        metric: metric_value(after_section, "eval_after", metric)
        for metric in METRIC_NAMES
    }
    deltas = {
        metric: None
        if before[metric] is None or after[metric] is None
        else after[metric] - before[metric]
        for metric in METRIC_NAMES
    }

    artifact_dir = metrics_path.parent
    after_success = after["success_rate"] or 0.0
    after_rollouts = after["rollouts"] or 0.0
    return Phase1Summary(
        name=artifact_dir.name,
        metrics_path=metrics_path,
        before=before,
        after=after,
        deltas=deltas,
        has_loss_curve=(artifact_dir / "phase1_loss_curve.png").exists(),
        has_reward_curve=(artifact_dir / "phase1_reward_curve.png").exists(),
        gate_passed=after_success >= success_threshold and after_rollouts >= min_rollouts,
    )


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def render_text(
    summaries: list[Phase1Summary],
    *,
    success_threshold: float,
    min_rollouts: float,
) -> str:
    lines = [
        f"Phase 1 gate: success >= {success_threshold:.0%} on >= {min_rollouts:.0f} eval rollouts",
        "",
    ]
    for summary in summaries:
        lines.extend(
            [
                summary.name,
                f"  metrics: {summary.metrics_path}",
                f"  success: {format_pct(summary.before['success_rate'])} -> {format_pct(summary.after['success_rate'])} ({format_pct(summary.deltas['success_rate'])})",
                f"  submit:  {format_pct(summary.before['submit_rate'])} -> {format_pct(summary.after['submit_rate'])} ({format_pct(summary.deltas['submit_rate'])})",
                f"  reward:  {format_float(summary.before['mean_reward'])} -> {format_float(summary.after['mean_reward'])} ({format_float(summary.deltas['mean_reward'])})",
                f"  f1:      {format_float(summary.before['mean_f1'])} -> {format_float(summary.after['mean_f1'])} ({format_float(summary.deltas['mean_f1'])})",
                f"  tools:   inspect {format_pct(summary.after['inspect_rate'])}, probe {format_pct(summary.after['probe_rate'])}, ablate {format_pct(summary.after['ablate_rate'])}",
                f"  rollouts: {format_float(summary.after['rollouts'])}",
                f"  plots: loss={'yes' if summary.has_loss_curve else 'no'}, reward={'yes' if summary.has_reward_curve else 'no'}",
                f"  gate: {'PASS' if summary.gate_passed else 'not yet'}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def as_jsonable(summary: Phase1Summary) -> dict[str, Any]:
    return {
        "name": summary.name,
        "metrics_path": str(summary.metrics_path),
        "before": summary.before,
        "after": summary.after,
        "deltas": summary.deltas,
        "has_loss_curve": summary.has_loss_curve,
        "has_reward_curve": summary.has_reward_curve,
        "gate_passed": summary.gate_passed,
    }


def main() -> None:
    args = parse_args()
    metric_files = find_metric_files(args.paths)
    if not metric_files:
        raise SystemExit("No phase1_eval_metrics.json files found.")

    summaries = [
        summarize_file(
            path,
            success_threshold=args.success_threshold,
            min_rollouts=args.min_rollouts,
        )
        for path in metric_files
    ]

    if args.json:
        print(json.dumps([as_jsonable(summary) for summary in summaries], indent=2))
    else:
        print(
            render_text(
                summaries,
                success_threshold=args.success_threshold,
                min_rollouts=args.min_rollouts,
            ),
            end="",
        )

    if args.strict and not any(summary.gate_passed for summary in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
