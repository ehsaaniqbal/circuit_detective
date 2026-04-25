"""Persistent TransformerLens worker process for Phase 1 induction analysis."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback
from dataclasses import dataclass

import einops
import torch
from transformer_lens import HookedTransformer

from server.probes import InductionProbeConfig, build_repeated_token_batch


@dataclass(frozen=True, slots=True)
class Head:
    layer: int
    head: int

    @property
    def head_id(self) -> str:
        return f"L{self.layer}H{self.head}"

    def to_dict(self) -> dict[str, int | str]:
        return {
            "layer": self.layer,
            "head": self.head,
            "head_id": self.head_id,
        }


class Worker:
    def __init__(self) -> None:
        self.model_name = "attn-only-2l"
        self.probe_config = InductionProbeConfig()
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = HookedTransformer.from_pretrained(self.model_name)

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
            return self.inspect_induction_scores(top_k=top_k)
        if command == "ablate_head":
            layer = int(arguments["layer"])
            head = int(arguments["head"])
            return self.ablate_head(layer=layer, head=head)
        if command == "ground_truth_heads":
            return self.ground_truth_heads()
        raise ValueError(f"Unknown worker command: {command}")

    def run_probe(self) -> dict[str, object]:
        return {
            "baseline_behavior": self.measure_behavior_score(),
            "probe_batch_size": self.probe_config.batch_size,
            "probe_seq_len": self.probe_config.seq_len,
        }

    def inspect_induction_scores(self, top_k: int) -> dict[str, object]:
        scores = [item for item in self.compute_induction_scores()[:top_k]]
        return {"scores": scores}

    def ablate_head(self, *, layer: int, head: int) -> dict[str, object]:
        baseline = self.measure_behavior_score()
        ablated = self.measure_behavior_score(head_to_ablate=Head(layer=layer, head=head))
        return {
            "layer": layer,
            "head": head,
            "head_id": f"L{layer}H{head}",
            "baseline_behavior": baseline,
            "ablated_behavior": ablated,
            "behavior_delta": baseline - ablated,
        }

    def ground_truth_heads(self) -> dict[str, object]:
        top_head = self.compute_induction_scores()[0]["head_id"]
        return {"heads": [top_head]}

    def compute_induction_scores(self) -> list[dict[str, object]]:
        tokens = self.tokens()
        seq_len = self.probe_config.seq_len
        score_store = torch.zeros(
            (self.model.cfg.n_layers, self.model.cfg.n_heads),
            device=self.model.cfg.device,
        )

        def induction_score_hook(pattern, hook) -> None:
            stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1 - seq_len)
            scores = einops.reduce(stripe, "batch head position -> head", "mean")
            score_store[hook.layer(), :] = scores

        self.model.run_with_hooks(
            tokens,
            return_type=None,
            fwd_hooks=[(lambda name: name.endswith("pattern"), induction_score_hook)],
        )

        ranked: list[dict[str, object]] = []
        for layer in range(self.model.cfg.n_layers):
            for head in range(self.model.cfg.n_heads):
                ranked.append(
                    {
                        "layer": layer,
                        "head": head,
                        "head_id": f"L{layer}H{head}",
                        "score": float(score_store[layer, head].item()),
                    }
                )
        ranked.sort(key=lambda item: float(item["score"]), reverse=True)
        return ranked

    def measure_behavior_score(self, head_to_ablate: Head | None = None) -> float:
        tokens = self.tokens()
        hooks = []
        if head_to_ablate is not None:

            def ablation_hook(z, hook):
                z = z.clone()
                z[:, :, head_to_ablate.head, :] = 0
                return z

            hooks = [(f"blocks.{head_to_ablate.layer}.attn.hook_z", ablation_hook)]

        logits = self.model.run_with_hooks(
            tokens,
            return_type="logits",
            fwd_hooks=hooks,
        )
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = tokens[:, 1:]

        start = self.probe_config.seq_len - 1
        end = (2 * self.probe_config.seq_len) - 1
        repeated_slice = log_probs[:, start:end, :]
        target_slice = targets[:, start:end]
        gathered = repeated_slice.gather(-1, target_slice.unsqueeze(-1)).squeeze(-1)
        return float(gathered.exp().mean().item())

    def tokens(self):
        return build_repeated_token_batch(
            d_vocab=self.model.cfg.d_vocab,
            device=str(self.model.cfg.device),
            config=self.probe_config,
        )


def main() -> None:
    Worker().run()


if __name__ == "__main__":
    main()
