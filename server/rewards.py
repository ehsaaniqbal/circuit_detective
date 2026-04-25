"""Deterministic reward helpers for Circuit Detective."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class SubmissionScore:
    precision: float
    recall: float
    f1: float
    step_penalty: float
    total_reward: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_f1(submitted: set[str], ground_truth: set[str]) -> tuple[float, float, float]:
    if not submitted and not ground_truth:
        return 1.0, 1.0, 1.0

    true_positives = len(submitted & ground_truth)
    precision = true_positives / len(submitted) if submitted else 0.0
    recall = true_positives / len(ground_truth) if ground_truth else 0.0

    if precision + recall == 0.0:
        return precision, recall, 0.0

    return precision, recall, (2.0 * precision * recall) / (precision + recall)


def compute_submission_score(
    *,
    submitted: set[str],
    ground_truth: set[str],
    step_count: int,
    max_steps: int,
) -> SubmissionScore:
    precision, recall, f1 = compute_f1(submitted, ground_truth)
    normalized_steps = min(step_count / max_steps, 1.0)
    step_penalty = 0.1 * normalized_steps
    total_reward = f1 - step_penalty
    return SubmissionScore(
        precision=precision,
        recall=recall,
        f1=f1,
        step_penalty=step_penalty,
        total_reward=total_reward,
    )
