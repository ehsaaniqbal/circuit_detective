# Circuit Detective Architecture

This is a compact implementation map for the current repo.

## Layers

Circuit Detective has three main layers:

1. **OpenEnv environment**: `server/circuit_detective_environment.py` implements `CircuitDetectiveEnvironment(Environment)` with `reset`, `step`, and `state`.
2. **Circuit backends**: `server/backend.py` provides probe, inspection, ablation, and ground-truth scoring backends.
3. **Training wrappers**: `phase1_grpo.py` and `scripts/` adapt the same tool surface for TRL SFT/GRPO training.

Rewards are deterministic. The environment does not use LLM-as-judge scoring.

## Runtime Layout

```text
circuit_detective/
├── openenv.yaml
├── models.py
├── server/
│   ├── app.py
│   ├── circuit_detective_environment.py
│   ├── backend.py
│   ├── demo.py
│   ├── tlens_worker.py
│   └── ioi_worker.py
├── phase1_grpo.py
├── scripts/
│   ├── phase1_sft.py
│   ├── phase1_train.py
│   ├── hf_phase1_job.py
│   ├── validator_smoke.py
│   └── analyze_phase1_run.py
├── notebooks/
├── assets/
├── artifacts/
└── tests/
```

## OpenEnv Surface

`server/app.py` builds the FastAPI/OpenEnv app via `openenv.core.env_server.http_server.create_app`.

The agent interacts through five tools:

- `list_tools`
- `run_probe`
- `inspect_induction_scores`
- `ablate_head`
- `submit_circuit`

`submit_circuit` terminates the episode and computes the deterministic score.

## Backends

`server/backend.py` defines the backend contract:

- `run_probe()`
- `inspect_induction_scores(top_k)`
- `ablate_head(head)`
- `ground_truth_heads()`

Implemented paths include the Phase 1 TransformerLens induction backend, planted/planted-lite synthetic causal arenas, and IOI stretch backends. TransformerLens runs through a `.venv-tlens` sidecar process because `openenv-core` and `transformer-lens` currently have dependency constraints that are cleaner to isolate.

## Training

Canonical training used:

- Agent model: `Qwen/Qwen3.5-2B`
- Trainer: HF TRL `SFTTrainer` + `GRPOTrainer`
- Adapter stack: PEFT LoRA with bitsandbytes 4-bit loading
- Compute: HF Jobs `a10g-large`

Main entrypoints:

- `notebooks/phase1_qwen35_2b_grpo.ipynb`
- `scripts/phase1_sft.py`
- `scripts/phase1_train.py`
- `scripts/hf_phase1_job.py`

Unsloth was considered, but the canonical successful runs use HF TRL + PEFT/bitsandbytes.

## Current Evidence

Phase 1 canonical adapter:

- `ehsaaniqbal/circuit-detective-qwen35-2b-phase1-sft64-grpo200-lora`
- Artifacts: `artifacts/phase1_sft64_grpo200_a10g_large/`

Phase 2 planted-lite adapter:

- `ehsaaniqbal/circuit-detective-qwen35-2b-planted-lite-naive-max-lora`
- Artifacts: `artifacts/planted_lite_naive_max_sft1536_grpo300_ctx1024/`

Final planted-lite result on 256 eval rollouts:

| Metric | Before GRPO | After GRPO |
| --- | ---: | ---: |
| Success / causal success | 94.9% | 97.7% |
| Submit rate | 95.7% | 98.0% |
| Ablate rate | 99.2% | 100.0% |
| Submitted best ablated head | 95.3% | 97.7% |
| Mean reward | 4.70 | 4.87 |

Honest interpretation: targeted SFT solves most of the Phase 2 interaction protocol; GRPO improves an already-high policy. Do not claim RL discovered the whole behavior from scratch.

## Demo

`server/demo.py` adds judge-facing routes on top of the OpenEnv app:

- `/`: HTML demo UI
- `/demo/reset`: start a planted-lite case
- `/demo/step`: issue a tool action
- `/demo/baseline`: rank-only baseline trace
- `/demo/protocol`: causal protocol trace
- `/demo/results`: load result snapshots

The demo story is:

```text
inspect candidates -> top-ranked head is decoy
ablate both heads -> one has larger behavior_delta
submit max-delta head -> causal success
```

## Validation

Minimum local checks:

```bash
uv run python scripts/validator_smoke.py
uv run pytest
```

Training evidence lives in committed `.png` curves and JSON metrics under `artifacts/`, with polished README/blog plots under `assets/`.
