"""Backends for Phase 1 induction localization."""

from __future__ import annotations

import atexit
import json
import os
import random
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Protocol


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TLENS_PYTHON = ROOT / ".venv-tlens" / "bin" / "python"
WORKER_PATH = ROOT / "server" / "tlens_worker.py"


@dataclass(frozen=True, slots=True)
class Head:
    layer: int
    head: int

    @property
    def head_id(self) -> str:
        return f"L{self.layer}H{self.head}"

    @classmethod
    def parse(cls, value: str) -> "Head":
        if not value.startswith("L") or "H" not in value:
            raise ValueError(f"Invalid head id: {value}")

        layer_part, head_part = value[1:].split("H", maxsplit=1)
        return cls(layer=int(layer_part), head=int(head_part))

    def to_dict(self) -> dict[str, int | str]:
        return {
            "layer": self.layer,
            "head": self.head,
            "head_id": self.head_id,
        }


@dataclass(frozen=True, slots=True)
class HeadScore:
    head: Head
    score: float

    def to_dict(self) -> dict[str, int | str | float]:
        return {
            **self.head.to_dict(),
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "HeadScore":
        return cls(
            head=Head(layer=int(payload["layer"]), head=int(payload["head"])),
            score=float(payload["score"]),
        )


@dataclass(frozen=True, slots=True)
class ProbeResult:
    baseline_behavior: float
    probe_batch_size: int
    probe_seq_len: int

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ProbeResult":
        return cls(
            baseline_behavior=float(payload["baseline_behavior"]),
            probe_batch_size=int(payload["probe_batch_size"]),
            probe_seq_len=int(payload["probe_seq_len"]),
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "baseline_behavior": self.baseline_behavior,
            "probe_batch_size": self.probe_batch_size,
            "probe_seq_len": self.probe_seq_len,
        }


@dataclass(frozen=True, slots=True)
class AblationResult:
    head: Head
    baseline_behavior: float
    ablated_behavior: float
    behavior_delta: float

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "AblationResult":
        return cls(
            head=Head(layer=int(payload["layer"]), head=int(payload["head"])),
            baseline_behavior=float(payload["baseline_behavior"]),
            ablated_behavior=float(payload["ablated_behavior"]),
            behavior_delta=float(payload["behavior_delta"]),
        )

    def to_dict(self) -> dict[str, int | str | float]:
        return {
            **self.head.to_dict(),
            "baseline_behavior": self.baseline_behavior,
            "ablated_behavior": self.ablated_behavior,
            "behavior_delta": self.behavior_delta,
        }


class CircuitBackend(Protocol):
    """Minimal backend contract the environment depends on."""

    scenario_id: str
    max_steps: int

    def run_probe(self) -> ProbeResult:
        """Return the baseline behavior score for the current scenario."""

    def inspect_induction_scores(self, top_k: int = 8) -> list[HeadScore]:
        """Return the top heads ranked by induction score."""

    def ablate_head(self, head: Head) -> AblationResult:
        """Return the effect of zero-ablating one head."""

    def ground_truth_heads(self) -> list[Head]:
        """Return the deterministic head set the agent is scored against."""


class FakeInductionBackend:
    """Small deterministic backend used by unit tests."""

    scenario_id = "l1_induction_attn_only_2l"
    max_steps = 12

    def __init__(self) -> None:
        self._scores = [
            HeadScore(Head(layer=1, head=3), 0.91),
            HeadScore(Head(layer=1, head=2), 0.44),
            HeadScore(Head(layer=0, head=1), 0.15),
        ]
        self._baseline_behavior = 0.73

    def run_probe(self) -> ProbeResult:
        return ProbeResult(
            baseline_behavior=self._baseline_behavior,
            probe_batch_size=8,
            probe_seq_len=24,
        )

    def inspect_induction_scores(self, top_k: int = 8) -> list[HeadScore]:
        return self._scores[:top_k]

    def ablate_head(self, head: Head) -> AblationResult:
        baseline = self._baseline_behavior
        if head == self.ground_truth_heads()[0]:
            ablated = 0.11
        else:
            ablated = 0.69
        return AblationResult(
            head=head,
            baseline_behavior=baseline,
            ablated_behavior=ablated,
            behavior_delta=baseline - ablated,
        )

    def ground_truth_heads(self) -> list[Head]:
        return [self._scores[0].head]


class RandomizedPlantedCircuitBackend:
    """
    Synthetic planted-circuit arena with exact ground truth.

    Each episode samples a hidden target head and a high-scoring decoy. Inspection
    scores are deliberately misleading; ablation deltas reveal the causal head.
    This is intended for curriculum training, not as a claim about a real model.
    """

    scenario_id = "planted_circuit_arena"
    max_steps = 10
    _instance_counter = 0

    def __init__(self, *, seed: int | None = None) -> None:
        if seed is None:
            seed = 20260426 + RandomizedPlantedCircuitBackend._instance_counter
            RandomizedPlantedCircuitBackend._instance_counter += 1
        self._rng = random.Random(seed)
        self._heads = [Head(layer=layer, head=head) for layer in range(2) for head in range(8)]
        self._baseline_behavior = 0.82
        self._target = self._heads[0]
        self._scores: list[HeadScore] = []
        self._deltas: dict[str, float] = {}
        self.reset_episode()

    def reset_episode(self) -> None:
        self._target = self._rng.choice(self._heads)
        candidates = [head for head in self._heads if head != self._target]
        decoy = self._rng.choice(candidates)
        distractors = [head for head in candidates if head != decoy]
        self._rng.shuffle(distractors)

        target_rank = self._rng.choice([1, 2])
        ranked_heads = [decoy]
        if target_rank == 1:
            ranked_heads.append(self._target)
            ranked_heads.extend(distractors[:6])
        else:
            ranked_heads.append(distractors[0])
            ranked_heads.append(self._target)
            ranked_heads.extend(distractors[1:6])

        score_by_rank = [0.96, 0.88, 0.82, 0.64, 0.57, 0.49, 0.41, 0.35]
        self._scores = [
            HeadScore(head=head, score=score_by_rank[index])
            for index, head in enumerate(ranked_heads[: len(score_by_rank)])
        ]
        self._deltas = {head.head_id: self._rng.uniform(0.0, 0.025) for head in self._heads}
        self._deltas[decoy.head_id] = self._rng.uniform(0.005, 0.035)
        self._deltas[self._target.head_id] = self._rng.uniform(0.38, 0.62)

    def run_probe(self) -> ProbeResult:
        return ProbeResult(
            baseline_behavior=self._baseline_behavior,
            probe_batch_size=8,
            probe_seq_len=24,
        )

    def inspect_induction_scores(self, top_k: int = 8) -> list[HeadScore]:
        return self._scores[:top_k]

    def ablate_head(self, head: Head) -> AblationResult:
        delta = self._deltas.get(head.head_id, 0.0)
        return AblationResult(
            head=head,
            baseline_behavior=self._baseline_behavior,
            ablated_behavior=self._baseline_behavior - delta,
            behavior_delta=delta,
        )

    def ground_truth_heads(self) -> list[Head]:
        return [self._target]


class PublishedIOICircuitBackend:
    """
    Fast IOI name-mover circuit arena using published GPT-2-small head ids.

    This backend is intentionally deterministic and lightweight for RL training:
    it scores the agent against the Name Mover heads reported in the IOI circuit
    paper, while exposing distractor heads from adjacent IOI circuit roles.
    It is not a live TransformerLens GPT-2-small forward pass.
    """

    scenario_id = "ioi_gpt2_small_name_mover"
    max_steps = 12

    NAME_MOVER_HEADS = [Head(9, 9), Head(9, 6), Head(10, 0)]
    S_INHIBITION_HEADS = [Head(7, 3), Head(7, 9), Head(8, 6), Head(8, 10)]
    NEGATIVE_NAME_MOVER_HEADS = [Head(10, 7), Head(11, 10)]
    BACKUP_NAME_MOVER_HEADS = [
        Head(9, 0),
        Head(9, 7),
        Head(10, 1),
        Head(10, 2),
        Head(10, 6),
        Head(10, 10),
        Head(11, 2),
        Head(11, 9),
    ]

    def __init__(self) -> None:
        self._baseline_behavior = 1.0
        self._scores = self._build_scores()
        self._deltas = self._build_deltas()

    def run_probe(self) -> ProbeResult:
        return ProbeResult(
            baseline_behavior=self._baseline_behavior,
            probe_batch_size=16,
            probe_seq_len=15,
        )

    def inspect_induction_scores(self, top_k: int = 8) -> list[HeadScore]:
        return self._scores[:top_k]

    def ablate_head(self, head: Head) -> AblationResult:
        delta = self._deltas.get(head.head_id, 0.005)
        return AblationResult(
            head=head,
            baseline_behavior=self._baseline_behavior,
            ablated_behavior=self._baseline_behavior - delta,
            behavior_delta=delta,
        )

    def ground_truth_heads(self) -> list[Head]:
        return self.NAME_MOVER_HEADS

    def _build_scores(self) -> list[HeadScore]:
        scored_heads: list[tuple[Head, float]] = [
            (Head(9, 9), 0.97),
            (Head(10, 0), 0.94),
            (Head(9, 6), 0.92),
            (Head(10, 10), 0.74),
            (Head(10, 6), 0.71),
            (Head(7, 3), 0.68),
            (Head(8, 10), 0.65),
            (Head(10, 7), 0.61),
            (Head(11, 10), 0.59),
            (Head(11, 2), 0.54),
            (Head(9, 7), 0.52),
            (Head(8, 6), 0.50),
        ]
        return [HeadScore(head=head, score=score) for head, score in scored_heads]

    def _build_deltas(self) -> dict[str, float]:
        deltas = {
            head.head_id: 0.015
            for layer in range(12)
            for head in [Head(layer=layer, head=index) for index in range(12)]
        }
        deltas.update(
            {
                "L9H9": 0.34,
                "L10H0": 0.31,
                "L9H6": 0.29,
                "L10H10": 0.12,
                "L10H6": 0.10,
                "L7H3": 0.09,
                "L8H10": 0.08,
                "L10H7": -0.08,
                "L11H10": -0.07,
            }
        )
        return deltas


class TransformerLensSubprocessBackend:
    """
    Phase 1 backend for induction localization on TransformerLens' attn-only-2l.

    `openenv-core` and `transformer-lens` do not currently resolve in the same
    Python environment because of incompatible `beartype` constraints, so the
    live backend runs in a dedicated sidecar virtual environment.
    """

    scenario_id = "l1_induction_attn_only_2l"
    max_steps = 12

    def __init__(self, *, python_executable: Path | None = None) -> None:
        self.python_executable = python_executable or DEFAULT_TLENS_PYTHON
        self._lock = RLock()
        self._process: subprocess.Popen[str] | None = None
        self._request_id = 0
        atexit.register(self.close)

    def close(self) -> None:
        if self._process is None:
            return
        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except Exception:
            self._process.kill()
        finally:
            self._process = None

    def run_probe(self) -> ProbeResult:
        payload = self._call_worker("run_probe")
        return ProbeResult.from_dict(payload)

    def inspect_induction_scores(self, top_k: int = 8) -> list[HeadScore]:
        payload = self._call_worker("inspect_induction_scores", {"top_k": top_k})
        return [HeadScore.from_dict(item) for item in payload["scores"]]

    def ablate_head(self, head: Head) -> AblationResult:
        payload = self._call_worker(
            "ablate_head",
            {"layer": head.layer, "head": head.head},
        )
        return AblationResult.from_dict(payload)

    def ground_truth_heads(self) -> list[Head]:
        payload = self._call_worker("ground_truth_heads")
        return [Head.parse(item) for item in payload["heads"]]

    def _call_worker(
        self,
        command: str,
        arguments: dict[str, object] | None = None,
    ) -> dict[str, object]:
        with self._lock:
            process = self._ensure_process()
            self._request_id += 1
            request = {
                "id": self._request_id,
                "command": command,
                "arguments": arguments or {},
            }

            assert process.stdin is not None
            assert process.stdout is not None
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()

            preamble: list[str] = []
            while True:
                response_line = process.stdout.readline()
                if not response_line:
                    stderr = ""
                    if process.stderr is not None:
                        stderr = process.stderr.read().strip()
                    preamble_text = " | ".join(item for item in preamble if item)
                    raise RuntimeError(
                        "TransformerLens worker exited unexpectedly."
                        + (f" stdout: {preamble_text}" if preamble_text else "")
                        + (f" stderr: {stderr}" if stderr else "")
                    )
                try:
                    response = json.loads(response_line)
                except json.JSONDecodeError:
                    preamble.append(response_line.strip())
                    continue
                if response.get("id") != self._request_id:
                    preamble.append(response_line.strip())
                    continue
                break
            if not response.get("ok", False):
                raise RuntimeError(str(response.get("error", "Unknown worker error")))
            return response["result"]

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process

        if not self.python_executable.exists():
            raise RuntimeError(
                "TransformerLens runtime is not bootstrapped. "
                "Run `uv venv .venv-tlens --python 3.11 && "
                "uv pip install --python .venv-tlens/bin/python transformer-lens==2.18.0`."
            )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        env.setdefault("HF_HUB_DISABLE_XET", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        self._process = subprocess.Popen(
            [str(self.python_executable), str(WORKER_PATH)],
            cwd=ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return self._process


@lru_cache(maxsize=1)
def get_default_backend() -> TransformerLensSubprocessBackend:
    return TransformerLensSubprocessBackend()
