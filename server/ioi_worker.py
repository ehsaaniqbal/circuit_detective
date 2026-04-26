"""Persistent TransformerLens worker for real GPT-2-small IOI probes."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import traceback
from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer


@dataclass(frozen=True, slots=True)
class Head:
    layer: int
    head: int

    @property
    def head_id(self) -> str:
        return f"L{self.layer}H{self.head}"


@dataclass(frozen=True, slots=True)
class IOIExample:
    prompt: str
    indirect_object: str
    subject: str


class Worker:
    """Serve real IOI logit-diff and head-ablation measurements over stdin/stdout."""

    MODEL_NAME = "gpt2-small"
    CANDIDATE_HEADS = [
        Head(9, 9),
        Head(10, 0),
        Head(9, 6),
        Head(10, 10),
        Head(10, 6),
        Head(7, 3),
        Head(8, 10),
        Head(10, 7),
        Head(11, 10),
        Head(11, 2),
        Head(9, 7),
        Head(8, 6),
    ]
    NAME_MOVER_HEADS = [Head(9, 9), Head(9, 6), Head(10, 0)]

    def __init__(self) -> None:
        self._model: HookedTransformer | None = None
        self._examples: list[IOIExample] | None = None

    @property
    def model(self) -> HookedTransformer:
        if self._model is None:
            device = os.getenv("CIRCUIT_DETECTIVE_TLENS_DEVICE", "cpu")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self._model = HookedTransformer.from_pretrained(
                    self.MODEL_NAME,
                    device=device,
                    dtype="float32",
                    default_prepend_bos=True,
                )
        return self._model

    def run(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            request = json.loads(line)
            try:
                command = request["command"]
                arguments = request.get("arguments", {})
                result = self.handle(command, arguments)
                response = {
                    "id": request["id"],
                    "ok": True,
                    "result": result,
                }
            except Exception as exc:  # pragma: no cover
                response = {
                    "id": request.get("id"),
                    "ok": False,
                    "error": f"{exc}\n{traceback.format_exc()}",
                }
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

    def handle(self, command: str, arguments: dict[str, object]) -> dict[str, object]:
        if command == "run_probe":
            return self.run_probe()
        if command == "inspect_induction_scores":
            top_k = int(arguments.get("top_k", 8))
            return self.inspect_head_effects(top_k=top_k)
        if command == "ablate_head":
            layer = int(arguments["layer"])
            head = int(arguments["head"])
            return self.ablate_head(layer=layer, head=head)
        if command == "ground_truth_heads":
            return self.ground_truth_heads()
        raise ValueError(f"Unknown worker command: {command}")

    def run_probe(self) -> dict[str, object]:
        return {
            "baseline_behavior": self.measure_logit_diff(),
            "probe_batch_size": len(self.examples()),
            "probe_seq_len": int(self.tokens().shape[1]),
            "metric": "mean_logit_diff_io_minus_subject",
        }

    def inspect_head_effects(self, top_k: int) -> dict[str, object]:
        baseline = self.measure_logit_diff()
        scores = []
        for head in self.CANDIDATE_HEADS:
            ablated = self.measure_logit_diff(head_to_ablate=head)
            scores.append(
                {
                    "layer": head.layer,
                    "head": head.head,
                    "head_id": head.head_id,
                    "score": baseline - ablated,
                    "baseline_behavior": baseline,
                    "ablated_behavior": ablated,
                    "metric": "logit_diff_delta",
                }
            )
        scores.sort(key=lambda item: float(item["score"]), reverse=True)
        return {"scores": scores[:top_k]}

    def ablate_head(self, *, layer: int, head: int) -> dict[str, object]:
        target = Head(layer=layer, head=head)
        baseline = self.measure_logit_diff()
        ablated = self.measure_logit_diff(head_to_ablate=target)
        return {
            "layer": layer,
            "head": head,
            "head_id": target.head_id,
            "baseline_behavior": baseline,
            "ablated_behavior": ablated,
            "behavior_delta": baseline - ablated,
            "metric": "mean_logit_diff_io_minus_subject",
        }

    def ground_truth_heads(self) -> dict[str, object]:
        return {"heads": [head.head_id for head in self.NAME_MOVER_HEADS]}

    def measure_logit_diff(self, head_to_ablate: Head | None = None) -> float:
        hooks = []
        if head_to_ablate is not None:

            def ablation_hook(z, hook):
                z = z.clone()
                z[:, :, head_to_ablate.head, :] = 0
                return z

            hooks = [(f"blocks.{head_to_ablate.layer}.attn.hook_z", ablation_hook)]

        with torch.no_grad():
            logits = self.model.run_with_hooks(
                self.tokens(),
                return_type="logits",
                fwd_hooks=hooks,
            )
        final_logits = logits[:, -1, :]
        io_tokens, subject_tokens = self.answer_tokens()
        diffs = final_logits.gather(-1, io_tokens[:, None]).squeeze(-1) - final_logits.gather(
            -1,
            subject_tokens[:, None],
        ).squeeze(-1)
        return float(diffs.mean().item())

    def tokens(self) -> torch.Tensor:
        prompts = [example.prompt for example in self.examples()]
        return self.model.to_tokens(
            prompts,
            prepend_bos=True,
            padding_side="right",
            move_to_device=True,
        )

    def answer_tokens(self) -> tuple[torch.Tensor, torch.Tensor]:
        io_tokens = [
            self.model.to_single_token(f" {example.indirect_object}") for example in self.examples()
        ]
        subject_tokens = [
            self.model.to_single_token(f" {example.subject}") for example in self.examples()
        ]
        device = self.model.cfg.device
        return (
            torch.tensor(io_tokens, device=device),
            torch.tensor(subject_tokens, device=device),
        )

    def examples(self) -> list[IOIExample]:
        if self._examples is not None:
            return self._examples

        raw_examples = [
            ("Mary", "John", "When {io} and {s} went to the store, {s} gave a bottle to"),
            ("John", "Mary", "When {io} and {s} went to the store, {s} gave a bottle to"),
            ("Alice", "Bob", "After {io} and {s} went to the park, {s} handed a book to"),
            ("Bob", "Alice", "After {io} and {s} went to the park, {s} handed a book to"),
            ("Tom", "Mary", "Then {io} and {s} had lunch together, and {s} passed a note to"),
            ("Mary", "Tom", "Then {io} and {s} had lunch together, and {s} passed a note to"),
        ]
        examples = []
        for indirect_object, subject, template in raw_examples:
            self.model.to_single_token(f" {indirect_object}")
            self.model.to_single_token(f" {subject}")
            examples.append(
                IOIExample(
                    prompt=template.format(io=indirect_object, s=subject),
                    indirect_object=indirect_object,
                    subject=subject,
                )
            )
        self._examples = examples
        return self._examples


def main() -> None:
    Worker().run()


if __name__ == "__main__":
    main()
