---
title: Circuit Detective
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - mechanistic-interpretability
  - reinforcement-learning
---

# Circuit Detective

Circuit Detective is an OpenEnv environment for training an agent to localize circuits in frozen transformers.

Public Space target: https://huggingface.co/spaces/ehsaaniqbal/circuit-detective

The current implementation is intentionally narrow: **Phase 1 only**. It targets induction localization in TransformerLens' `attn-only-2l` toy model and exposes a small, deterministic tool surface:

- `list_tools`
- `run_probe`
- `inspect_induction_scores`
- `ablate_head`
- `submit_circuit`

The environment package is OpenEnv-valid and runnable locally with Gym-style `reset`, `step`, and `state`. The training notebook and richer reward shaping will build on top of this Phase 1 server/client contract.

Notebook-first training entrypoint:

- `notebooks/phase1_qwen35_2b_grpo.ipynb`
- `scripts/phase1_train.py`
- `scripts/hf_phase1_job.py`

## Local Development

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync --extra dev
uv venv .venv-tlens --python 3.11
HF_HUB_DISABLE_XET=1 uv pip install --torch-backend cpu --python .venv-tlens/bin/python transformer-lens==2.18.0
uv run server --port 8000
```

In another shell:

```bash
uv run python scripts/validator_smoke.py
```

## Current Scope

- One scenario: `l1_induction_attn_only_2l`
- One dominant-head submission target derived deterministically from a fixed induction metric on the chosen checkpoint
- Deterministic reward only
- Split runtimes by necessity: `openenv-core` and `transformer-lens` currently have incompatible `beartype` constraints, so the live backend runs in a dedicated `.venv-tlens` sidecar process
- Training wrapper uses explicit public Python tool methods for TRL `environment_factory`, while the deployed environment keeps the OpenEnv `reset` / `step` / `state` surface

## Repository Layout

```text
circuit_detective/
├── __init__.py
├── client.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── scripts/
├── server/
└── tests/
```
