# Circuit Detective Architecture

This document describes the current implementation layout and runtime setup. It is intentionally implementation-facing rather than a submission writeup.

## High-Level Shape

Circuit Detective has three main layers:

1. **OpenEnv environment layer**: exposes a Gym-style `reset`, `step`, and `state` interface for validation and the Hugging Face Space.
2. **Circuit backend layer**: computes probe scores, head rankings, ablation deltas, and ground-truth scoring for frozen target models or synthetic arenas.
3. **Training layer**: wraps the same tool surface for HF TRL SFT/GRPO training on `Qwen/Qwen3.5-2B`.

The core design is deterministic: the environment computes rewards from known circuit targets and ablation evidence. There is no LLM-as-judge reward.

## Repository Layout

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
│   └── phase1_qwen35_2b_grpo.ipynb
├── tests/
└── artifacts/
```

Important docs:

- `docs/phase_plan.md`: execution plan and gates.
- `docs/log.md`: experiment log, including failed Phase 2 attempts.
- `docs/writeup.md`: draft narrative.
- `docs/circuit_detective_brief.md`: original project spec.

## OpenEnv Runtime

The OpenEnv server is built in `server/app.py`.

- `server/app.py` calls `openenv.core.env_server.http_server.create_app`.
- `server/circuit_detective_environment.py` defines `CircuitDetectiveEnvironment(Environment)`.
- `models.py` defines typed action/observation payloads.
- `openenv.yaml` is the parseable OpenEnv manifest.

The environment exposes this tool surface:

- `list_tools`
- `run_probe`
- `inspect_induction_scores`
- `ablate_head`
- `submit_circuit`

Each episode is a tool-use trajectory. The agent observes summaries/results, chooses tool calls, and receives deterministic reward. `submit_circuit` ends the episode and computes final score.

## Backend Layer

Backend contracts live in `server/backend.py`.

The environment depends on a small `CircuitBackend` protocol:

- `run_probe()`
- `inspect_induction_scores(top_k)`
- `ablate_head(head)`
- `ground_truth_heads()`

Implemented backend paths include:

- `FakeInductionBackend`: deterministic unit-test backend.
- TransformerLens induction backend: uses `server/tlens_worker.py`.
- Planted-circuit and planted-lite synthetic backends: exact ground truth with decoy heads.
- IOI stretch backends: fast deterministic IOI/name-mover and real TransformerLens GPT-2-small smoke path.

The deployed backend uses a sidecar Python environment for TransformerLens because `openenv-core` and `transformer-lens` currently have dependency constraints that do not cleanly resolve in one environment. The sidecar path is `.venv-tlens`, launched through worker scripts.

## Training Runtime

Training is separate from the OpenEnv HTTP server.

Main training files:

- `phase1_grpo.py`: TRL-compatible tool environment classes.
- `scripts/phase1_sft.py`: SFT warm-start generation/training.
- `scripts/phase1_train.py`: GRPO training/eval.
- `scripts/hf_phase1_job.py`: HF Jobs launcher that can run SFT then GRPO in one job.

The current successful path uses:

- Base agent model: `Qwen/Qwen3.5-2B`.
- Trainer: HF TRL `SFTTrainer` and `GRPOTrainer`.
- Adapter stack: PEFT LoRA with bitsandbytes 4-bit loading.
- Platform: Hugging Face Jobs, usually `a10g-large`.

Unsloth was considered for speed/memory, but the successful runs used the TRL + PEFT/bitsandbytes path.

## Scenarios

Current scenario IDs are defined in `server/circuit_detective_environment.py`.

- `l1_induction_attn_only_2l`: Phase 1 toy induction task on TransformerLens `attn-only-2l`.
- `l2_ablation_required`: Phase 2 variant requiring ablation evidence before full credit.
- `planted_circuit_arena`: randomized planted target and decoy heads.
- `planted_lite_causal_chain`: final minimal causal-chain Phase 2 task.
- `ioi_gpt2_small_name_mover`: fast IOI/name-mover stretch arena.
- `ioi_gpt2_small_real`: real TransformerLens GPT-2-small IOI smoke path.

The current strongest Phase 2 evidence is `planted_lite_causal_chain`, where the top inspected head is a decoy and the correct head is determined by ablation delta.

## Reward And Metrics

The OpenEnv reward is deterministic. Trainer-side wrappers add dense shaping so GRPO gets useful signal before terminal submission.

Important evaluation metrics:

- `success_rate`
- `causal_success_rate`
- `submit_rate`
- `ablate_rate`
- `ablate_submitted_rate`
- `submitted_after_all_candidates_rate`
- `submitted_best_ablated_head_rate`
- `terminal_ready_no_submit_rate`
- `mean_reward`
- `mean_f1`

Phase 2 failures taught us that ordinary success/F1 can be misleading. The key Phase 2 metric is `causal_success_rate`: did the agent actually ablate and submit the causally supported head?

## Current Evidence

Phase 1 canonical adapter:

- `ehsaaniqbal/circuit-detective-qwen35-2b-phase1-sft64-grpo200-lora`
- Artifacts: `artifacts/phase1_sft64_grpo200_a10g_large/`

Final planted-lite Phase 2 run:

- HF Job: `ehsaaniqbal/69edc4ddd2c8bd8662bcf87c`
- Adapter: `ehsaaniqbal/circuit-detective-qwen35-2b-planted-lite-naive-max-lora`
- Artifacts: `artifacts/planted_lite_naive_max_sft1536_grpo300_ctx1024/`

Final planted-lite result on 256 eval rollouts:

| Metric | Before GRPO | After GRPO |
| --- | ---: | ---: |
| Success / causal success | 94.9% | 97.7% |
| Submit rate | 95.7% | 98.0% |
| Ablate rate | 99.2% | 100.0% |
| Submitted best ablated head | 95.3% | 97.7% |
| Mean reward | 4.70 | 4.87 |

The honest interpretation is that targeted SFT solved most of the interaction protocol, and GRPO cleaned it up further. The result should not be described as RL discovering the full behavior from scratch.

## Demo And Deployment

The Hugging Face Space runs the Docker/OpenEnv server and registers a lightweight judge-facing demo in `server/demo.py`.

Demo routes:

- `/`: HTML demo UI.
- `/demo/reset`: start a planted-lite case.
- `/demo/step`: manually issue a tool action.
- `/demo/baseline`: show rank-only baseline behavior.
- `/demo/protocol`: show the causal protocol behavior.
- `/demo/results`: load result snapshots from artifacts.

The intended demo story is simple:

```text
inspect two candidate heads -> top-ranked head is decoy
ablate both heads -> one has larger behavior_delta
submit max-delta head -> causal success
```

## Validation Path

Key validation files:

- `scripts/validator_smoke.py`: simulates the validator path.
- `scripts/smoke_rollout.py`: quick local rollout.
- `tests/`: unit tests for environment, demo, SFT generation, and run analysis.

Minimum health check:

```bash
uv run python scripts/validator_smoke.py
uv run pytest
```

For training evidence, committed `.png` curves and JSON metrics live under `artifacts/`.
