"""Probe generation utilities for Phase 1 induction experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InductionProbeConfig:
    batch_size: int = 8
    seq_len: int = 24
    seed: int = 0


def build_repeated_token_batch(
    *,
    d_vocab: int,
    device: str,
    config: InductionProbeConfig,
):
    """Build a repeated-token batch used for induction-head measurements."""
    import torch

    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)

    prefix = torch.randint(
        low=0,
        high=d_vocab,
        size=(config.batch_size, config.seq_len),
        generator=generator,
        device=device,
    )
    return torch.cat([prefix, prefix], dim=-1)
