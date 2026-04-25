from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_analyzer_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_phase1_run.py"
    spec = importlib.util.spec_from_file_location("analyze_phase1_run", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_metrics(path: Path, *, after_success: float, after_rollouts: float) -> None:
    path.mkdir(parents=True)
    (path / "phase1_eval_metrics.json").write_text(
        json.dumps(
            {
                "before": {
                    "eval_before_success_rate": 0.125,
                    "eval_before_submit_rate": 0.125,
                    "eval_before_mean_reward": 0.1,
                    "eval_before_mean_f1": 0.125,
                    "eval_before_rollouts": 16.0,
                },
                "after": {
                    "eval_after_success_rate": after_success,
                    "eval_after_submit_rate": after_success,
                    "eval_after_mean_reward": 0.5,
                    "eval_after_mean_f1": after_success,
                    "eval_after_inspect_rate": 0.75,
                    "eval_after_probe_rate": 0.25,
                    "eval_after_ablate_rate": 0.125,
                    "eval_after_rollouts": after_rollouts,
                },
            }
        ),
        encoding="utf-8",
    )
    (path / "phase1_loss_curve.png").write_bytes(b"png")
    (path / "phase1_reward_curve.png").write_bytes(b"png")


def test_summarize_file_computes_gate_and_deltas(tmp_path: Path) -> None:
    analyzer = load_analyzer_module()
    run_dir = tmp_path / "run"
    write_metrics(run_dir, after_success=0.5, after_rollouts=32.0)

    summary = analyzer.summarize_file(
        run_dir / "phase1_eval_metrics.json",
        success_threshold=0.4,
        min_rollouts=32.0,
    )

    assert summary.gate_passed is True
    assert summary.deltas["success_rate"] == 0.375
    assert summary.has_loss_curve is True
    assert summary.has_reward_curve is True


def test_find_metric_files_accepts_parent_directory(tmp_path: Path) -> None:
    analyzer = load_analyzer_module()
    write_metrics(tmp_path / "run_a", after_success=0.25, after_rollouts=16.0)
    write_metrics(tmp_path / "run_b", after_success=0.5, after_rollouts=32.0)

    files = analyzer.find_metric_files([str(tmp_path)])

    assert [file.parent.name for file in files] == ["run_a", "run_b"]


def test_render_text_reports_gate_status(tmp_path: Path) -> None:
    analyzer = load_analyzer_module()
    run_dir = tmp_path / "run"
    write_metrics(run_dir, after_success=0.375, after_rollouts=16.0)
    summary = analyzer.summarize_file(
        run_dir / "phase1_eval_metrics.json",
        success_threshold=0.4,
        min_rollouts=32.0,
    )

    rendered = analyzer.render_text(
        [summary],
        success_threshold=0.4,
        min_rollouts=32.0,
    )

    assert "success: 12.5% -> 37.5% (25.0%)" in rendered
    assert "gate: not yet" in rendered
