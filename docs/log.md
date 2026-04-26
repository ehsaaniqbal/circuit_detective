# Circuit Detective Build Log

Generated: 2026-04-26 14:21 IST

This is a factual experiment log for blog/video drafting. It is intentionally not polished as a submission narrative. Metrics are copied from committed or downloaded `phase1_eval_metrics.json` artifacts unless a row is explicitly marked as a live/in-progress observation.

## Project Pivot And Scope

Circuit Detective started as a Round 2 pivot away from InvoiceOps/AP exceptions into a mechanistic-interpretability RL environment. The core bet was that OpenEnv can be used for agentic interpretability workflows: the agent receives a tool surface, investigates a frozen transformer, and submits a circuit.

The implementation target became deliberately narrow:

| Item | Decision |
| --- | --- |
| Agent model | `Qwen/Qwen3.5-2B` |
| Trainer | HF TRL `GRPOTrainer` |
| GPU path | HF Jobs, mostly `a10g-large` |
| Optional speed path | Unsloth considered, but not used for canonical runs |
| Phase 1 target | TransformerLens `attn-only-2l` induction head localization |
| Main tool surface | Direct Python tool methods through TRL `environment_factory` |
| OpenEnv surface | Standard `reset`, `step`, `state`, plus parseable `openenv.yaml` |

Key implementation constraint: keep the OpenEnv environment deterministic, and put dense trainer-side shaping in the TRL wrapper.

References:

- Design/plan: [`docs/phase_plan.md`](docs/phase_plan.md)
- README result summary: [`README.md`](README.md)
- Current writeup draft: [`docs/writeup.md`](docs/writeup.md)
- OpenEnv smoke validator: [`scripts/validator_smoke.py`](scripts/validator_smoke.py)
- Training scripts: [`scripts/phase1_sft.py`](scripts/phase1_sft.py), [`scripts/phase1_train.py`](scripts/phase1_train.py), [`scripts/hf_phase1_job.py`](scripts/hf_phase1_job.py)

## Infrastructure And Validation Work

Before training results mattered, the project had to pass the hackathon gates:

| Gate | Status |
| --- | --- |
| Public HF Space | `https://huggingface.co/spaces/ehsaaniqbal/circuit-detective` |
| Public GitHub repo | `https://github.com/ehsaaniqbal/circuit_detective` |
| OpenEnv validation | `uv run openenv validate` passed |
| Local validator smoke | `uv run python scripts/validator_smoke.py` passed |
| Training curves | `.png` curves committed/uploaded under `artifacts/*` |
| Runnable training entrypoints | notebook plus `scripts/phase1_sft.py`, `scripts/phase1_train.py`, `scripts/hf_phase1_job.py` |

Roadblocks:

| Roadblock | What happened | Resolution / learning |
| --- | --- | --- |
| HF Space builds were slow | Docker/HF Space rebuilds took longer than local tests. | Kept HF Space as deployment/validation target and used HF Jobs for training. |
| Dependency split | `openenv-core` and `transformer-lens` dependency constraints were awkward together. | Used a sidecar `.venv-tlens` path for TransformerLens backend work. |
| HF Jobs are ephemeral | Model/dependency downloads happen each job unless cached by platform. | Accepted download overhead; HF Jobs snapshot/push discipline became important. |
| H200 jobs failed | H200 was attempted but did not become a reliable training path in time. | Standardized on `a10g-large`, which consistently ran. |
| Logs sometimes lagged | HF Jobs status could show `RUNNING` while logs were delayed or empty. | Used `hf jobs inspect` plus later log polling; did not treat empty logs alone as failure. |

## Phase 1: Basic Induction Localization

Goal: prove that a small model can learn the minimum tool protocol:

```text
inspect_induction_scores(top_k=3) -> submit_circuit([top_head])
```

The stronger desired protocol included `ablate_head`, but Phase 1 success was defined as reliable circuit localization and submission.

### Phase 1 Experiment Table

| Run / artifact | Setup | Eval rollouts | Success before -> after | Submit before -> after | Mean reward before -> after | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `phase1_shaped_smoke_20b` | Early shaped GRPO smoke | 32 | 3.1% -> 0.0% | 3.1% -> 0.0% | 0.0032 -> -0.0256 | Failed; reward/tool shaping insufficient. |
| `phase1_submit_tuned_50` | Submit reward tuning, 50 GRPO steps | 16 | 0.0% -> 25.0% | 0.0% -> 25.0% | -0.0625 -> 0.3100 | First evidence that submit shaping could move behavior. |
| `phase1_submit_tuned_150` | Submit reward tuning, 150 GRPO steps | 32 | 12.5% -> 18.8% | 12.5% -> 18.8% | 0.1316 -> 0.2719 | Improvement too small. |
| `phase1_sft_grpo_75_a10g_large` | Tiny SFT warm-start + 75 GRPO | 16 | 18.8% -> 37.5% | 18.8% -> 37.5% | 0.2200 -> 0.4756 | Promising but below 40% gate. |
| `phase1_sft_grpo_150_a10g_large` | Tiny SFT warm-start + 150 GRPO | 32 | 12.5% -> 56.2% | 12.5% -> 59.4% | 0.0888 -> 0.8879 | Passed Phase 1 gate. |
| `phase1_sft64_grpo200_a10g_large` | SFT64 + 200 GRPO, canonical run | 48 | 10.4% -> 79.2% | 10.4% -> 81.2% | 0.0858 -> 1.2072 | Best Phase 1 result; canonical adapter uploaded. |

Canonical Phase 1 job:

- HF Job: `ehsaaniqbal/69ecd77ad2c8bd8662bcdd0b`
- Adapter: `ehsaaniqbal/circuit-detective-qwen35-2b-phase1-sft64-grpo200-lora`
- Metrics: [`artifacts/phase1_sft64_grpo200_a10g_large/phase1_eval_metrics.json`](artifacts/phase1_sft64_grpo200_a10g_large/phase1_eval_metrics.json)
- Reward curve: [`artifacts/phase1_sft64_grpo200_a10g_large/phase1_reward_curve.png`](artifacts/phase1_sft64_grpo200_a10g_large/phase1_reward_curve.png)
- Loss curve: [`artifacts/phase1_sft64_grpo200_a10g_large/phase1_loss_curve.png`](artifacts/phase1_sft64_grpo200_a10g_large/phase1_loss_curve.png)

Learning:

- Tiny SFT warm-start mattered. Raw GRPO/tool reward shaping was unstable at first.
- GRPO did improve the agent after SFT. The canonical result went from 10.4% success before GRPO to 79.2% after GRPO.
- Phase 1 was still mostly localization/submission, not rich causal interpretability. Ablation usage after the canonical run was only 6.2%.

## Phase 2: Ablation-Required Causal Validation

Goal: move beyond ranking and require causal evidence:

```text
inspect -> ablate candidate -> submit verified head
```

The first Phase 2 design reused the Phase 1 toy transformer and required ablation for full causal credit.

### Early Phase 2 Experiment Table

| Run / artifact | Eval rollouts | Success before -> after | Submit before -> after | Ablate after | Ablate submitted after | Causal success after | What happened |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `phase2_ablation_required_grpo120_a10g_large` | 48 | 14.6% -> 83.3% | 14.6% -> 85.4% | 12.5% | 0.0% | 0.0% | Looked good on success, but it solved by submitting without causal evidence. |
| `phase2_calibrated_sft96_grpo100_a10g_large` | 48 | 8.3% -> 77.1% | 10.4% -> 79.2% | 16.7% | 0.0% | 0.0% | Same shortcut persisted after reward calibration. |
| `phase2_strict_sft64_grpo160_a10g_large` | 48 | 18.8% -> 2.1% | 18.8% -> 2.1% | 95.8% | 2.1% | 0.0% | Strict reward made it ablate, but it mostly stopped before submitting. |

References:

- [`artifacts/phase2_ablation_required_grpo120_a10g_large/phase1_eval_metrics.json`](artifacts/phase2_ablation_required_grpo120_a10g_large/phase1_eval_metrics.json)
- [`artifacts/phase2_calibrated_sft96_grpo100_a10g_large/phase1_eval_metrics.json`](artifacts/phase2_calibrated_sft96_grpo100_a10g_large/phase1_eval_metrics.json)
- [`artifacts/phase2_strict_sft64_grpo160_a10g_large/phase1_eval_metrics.json`](artifacts/phase2_strict_sft64_grpo160_a10g_large/phase1_eval_metrics.json)

Learning:

- A high task success metric was misleading when causal success stayed at 0.0%.
- If reward allowed a shortcut, the model took it.
- If reward was too strict/sparse, the model learned the evidence-gathering action (`ablate_head`) but failed to complete the terminal action (`submit_circuit`).
- Phase 2 needed a cleaner toy causal chain before harder IOI/backdoor stories.

## Randomized Planted Circuit Attempts

Goal: prevent memorizing a single answer by using a planted target with decoy heads. Inspection rank was allowed to be misleading, and ablation was supposed to reveal the true causal head.

| Run / artifact | Eval rollouts | Success before -> after | Submit before -> after | Ablate after | Causal success after | What happened |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `planted_arena_sft128_grpo120_a10g_large` | 48 | 0.0% -> 0.0% | 4.2% -> 29.2% | 41.7% | 0.0% | More submissions, but no correct causal solves. |
| `planted_arena_rankvar_sft256_grpo240_a10g_large` | 128 | 0.0% -> 5.5% | 17.2% -> 85.9% | 9.4% | 0.0% | Learned to submit frequently, but mostly wrong; causal objective did not land. |
| `planted_expert_loopfix_sft384_grpo120` | 24 | 0.0% -> 0.0% | 4.2% -> 0.0% | 41.7% | 0.0% | Loop/submit failure persisted. |

Remote artifact references:

- `https://huggingface.co/spaces/ehsaaniqbal/circuit-detective/tree/main/artifacts/planted_arena_sft128_grpo120_a10g_large`
- `https://huggingface.co/spaces/ehsaaniqbal/circuit-detective/tree/main/artifacts/planted_arena_rankvar_sft256_grpo240_a10g_large`
- `https://huggingface.co/spaces/ehsaaniqbal/circuit-detective/tree/main/artifacts/planted_expert_loopfix_sft384_grpo120`

Learning:

- Randomizing targets made the environment much harder than Phase 1.
- SFT examples that showed the answer were not enough to produce robust interactive behavior.
- The model often learned one part of the protocol in isolation: submit frequently, or ablate frequently, but not the complete causal chain.

## IOI And Real TransformerLens Stretch Attempts

Goal: move from toy induction to GPT-2-small IOI/name-mover style circuit localization.

Implemented variants:

- Fast deterministic IOI name-mover arena using a published-style target set.
- Real TransformerLens GPT-2-small backend smoke path with fixed IOI prompts and real ablation deltas.
- Curriculum mixing planted and IOI tasks.

| Run / artifact | Eval rollouts | Success before -> after | Submit before -> after | Mean F1 before -> after | Causal success after | What happened |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `real_ioi_sft64_grpo40_a10g_large` | 8 | 0.0% -> 0.0% | 0.0% -> 0.0% | 0.0000 -> 0.0000 | 0.0% | Smoke run only; no learning result. |
| `ioi_name_mover_sft160_grpo160_a10g_large` | 128 | 0.0% -> 0.0% | 1.6% -> 5.5% | 0.0000 -> 0.0059 | 0.0% | Small increase in partial F1, no solved trajectories. |
| `real_ioi_expert_loopfix_sft256_grpo80` | 16 | 0.0% -> 0.0% | 0.0% -> 6.2% | 0.0000 -> 0.0000 | 0.0% | Some submit attempts after evidence, but no correct IOI solve. |
| `curriculum_planted_ioi_sft256_grpo220_a10g_large` | 128 | 0.0% -> 2.3% | 27.3% -> 32.8% | 0.0180 -> 0.0422 | 0.0% | Mixed curriculum gave tiny improvement, not enough for a credible success claim. |

Remote artifact references:

- `https://huggingface.co/spaces/ehsaaniqbal/circuit-detective/tree/main/artifacts/real_ioi_sft64_grpo40_a10g_large`
- `https://huggingface.co/spaces/ehsaaniqbal/circuit-detective/tree/main/artifacts/ioi_name_mover_sft160_grpo160_a10g_large`
- `https://huggingface.co/spaces/ehsaaniqbal/circuit-detective/tree/main/artifacts/real_ioi_expert_loopfix_sft256_grpo80`
- `https://huggingface.co/spaces/ehsaaniqbal/circuit-detective/tree/main/artifacts/curriculum_planted_ioi_sft256_grpo220_a10g_large`

Learning:

- IOI was too ambitious before the smaller causal-chain protocol worked.
- The real TransformerLens backend was useful technically, but not yet a training success.
- For the final story, IOI should be framed as a stretch/roadmap unless later results change materially.

## Planted-Lite Causal Chain

Goal: create the smallest Phase 2 task that still requires causal reasoning:

```text
inspect two candidates -> ablate both -> compare behavior_delta -> submit max-delta head
```

This environment intentionally makes the top inspected head a decoy. The correct answer should come from ablation, not ranking.

### Planted-Lite Experiment Table

| Run / artifact | Eval rollouts | Success before -> after | Submit before -> after | Ablate before -> after | Causal success after | What happened |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `planted_lite_causal_chain_sft512_grpo160` | 256 | 0.0% -> 0.0% | 0.0% -> 0.0% | 100.0% -> 100.0% | 0.0% | Model inspected/ablated but never submitted. Reward stayed at -2.0. |
| `planted_lite_sftfit_reward_ladder_sft512_grpo160` | 256 | 0.0% -> 0.0% | 0.0% -> 1.6% | 99.6% -> 98.0% | 0.0% | Partial breakthrough: it often reached ablation evidence, but terminal submit almost never happened. |

Known job references:

- First planted-lite strict run: `ehsaaniqbal/69ed97c8d2c8bd8662bcf2a7`
- Reward-ladder run: `ehsaaniqbal/69edab88d2c8bd8662bcf51d`

Remote artifact references:

- `https://huggingface.co/spaces/ehsaaniqbal/circuit-detective/tree/main/artifacts/planted_lite_causal_chain_sft512_grpo160`
- `https://huggingface.co/spaces/ehsaaniqbal/circuit-detective/tree/main/artifacts/planted_lite_sftfit_reward_ladder_sft512_grpo160`

Important failure mode:

- The agent learned the middle of the workflow: `inspect -> ablate -> ablate`.
- It did not reliably learn the terminal action: `submit_circuit`.
- Sample rollouts from the reward-ladder run showed all candidate heads ablated, no terminal submit, and no causal success.

Learning:

- The bottleneck was not simply "does the model know the answer?"
- The bottleneck was interactive control: after evidence appears, the model must choose the final tool call.
- Live GRPO observations had to match SFT examples. If SFT traces are compact but live observations are noisy/verbose, the model can fail at the exact transition.

## Repairs After Planted-Lite Failure

After the reward-ladder run, the code changed in three main ways.

### 1. Terminal bridge in the environment

Added explicit fields to planted-lite observations:

```json
{
  "next_required_tool": "submit_circuit",
  "next_required_arguments": {"heads": ["LxHy"]},
  "terminal_action_required": true,
  "must_submit": "LxHy"
}
```

Reference commit:

- `bdf120a` - `Tighten planted-lite terminal bridge`

### 2. Strict observation compaction

The strict planted-lite training wrapper now renders compact observations and removes clutter such as repeated full tool lists. This was intended to align live GRPO prompts with SFT-style observations.

Reference file:

- [`phase1_grpo.py`](phase1_grpo.py)

### 3. Terminal SFT clinic

The SFT data was expanded from generic full-chain traces to targeted terminal/recovery examples. With `--sft-examples-per-prompt 16`, the preflight now creates:

| SFT statistic | Value |
| --- | ---: |
| Total records | 1152 |
| Records containing `submit_circuit` | 864 |
| Terminal submit records | 864 |
| Longest tokenized record | 1673 / 2048 tokens |

Reference commit:

- `cbe8946` - `Add planted-lite terminal SFT clinic`

Learning:

- More SFT volume alone is less important than targeted SFT at the failing state.
- We need to train the exact transition: "the observation says `next_required_tool=submit_circuit`; now call `submit_circuit`."
- Preflight checks are mandatory. Earlier failures were made harder by long traces and possible truncation risk.

## Current Final Serious Phase 2 Run

Current running job:

- HF Job: `ehsaaniqbal/69edc4ddd2c8bd8662bcf87c`
- URL: `https://huggingface.co/jobs/ehsaaniqbal/69edc4ddd2c8bd8662bcf87c`
- Status at log generation: `RUNNING`
- Started: 2026-04-26 07:55:09 UTC

Config:

| Parameter | Value |
| --- | --- |
| Scenario | `planted_lite` |
| Start adapter | `ehsaaniqbal/circuit-detective-qwen35-2b-phase1-sft64-grpo200-lora` |
| SFT steps | 1536 |
| SFT examples per prompt | 16 |
| SFT max sequence length | 2048 |
| GRPO steps | 300 |
| Number of generations | 8 |
| Eval generations | 8 |
| Eval prompts | 32 |
| Max completion length | 1024 |
| Max tool-calling iterations | 4 |
| GPU flavor | `a10g-large` |
| Output artifacts | `artifacts/planted_lite_naive_max_sft1536_grpo300_ctx1024` |
| Adapter repo | `ehsaaniqbal/circuit-detective-qwen35-2b-planted-lite-naive-max-lora` |

Live SFT observation from logs shared during the run:

- SFT loss was around `0.02-0.05`.
- Mean token accuracy was around `0.99`.
- Epoch was around `1.03`, approximately 77% through the 1536-step SFT stage.

Interpretation:

- The SFT stage is fitting the terminal-clinic traces strongly.
- This does not prove Phase 2 success; the decisive metrics are still `eval_before` and `eval_after` from GRPO:
  - `submit_rate`
  - `all_candidates_ablated_rate`
  - `terminal_ready_no_submit_rate`
  - `submitted_after_all_candidates_rate`
  - `submitted_best_ablated_head_rate`
  - `causal_success_rate`

## Summary Of What Worked

1. The OpenEnv environment and validator path work.
2. The HF Space is public and cloneable.
3. The training pipeline works end to end on HF Jobs `a10g-large`.
4. A small Qwen3.5-2B agent can learn the basic Phase 1 induction localization protocol.
5. SFT warm-start plus GRPO gave a strong Phase 1 result: 10.4% -> 79.2% success on 48 eval rollouts.
6. Diagnostics are now much better: later runs expose submit rate, ablation rate, causal success, candidate ablation coverage, terminal-ready/no-submit, and tool sequences.

## Summary Of What Failed

1. Pure shaping without SFT did not solve Phase 1.
2. Simple Phase 2 "require ablation" rewards were hackable: the agent still submitted without causal evidence.
3. Strict ablation rewards caused a different failure: the agent ablated but did not submit.
4. Random planted circuits were too hard before the two-candidate causal chain was solved.
5. IOI/GPT-2-small was too ambitious during the hackathon window and should not be overclaimed.
6. The planted-lite causal chain has not yet produced nonzero causal success in completed runs.

## Main Technical Learnings

1. Phase 1 was a solvable RL environment; Phase 2 exposed a real agent-control bottleneck.
2. "Correct answer" and "correct interactive policy" are different. SFT can teach answers while the rollout policy still fails at terminal tool use.
3. Reward shaping needs to distinguish intermediate evidence gathering from terminal completion. Otherwise the agent can plateau at "almost solved."
4. Evaluation metrics must separate raw success from causal success. The Phase 2 runs with 77-83% raw success were still failures because causal success was 0.0%.
5. Compact observation design matters. Live observations should not bury the decisive action in verbose tool metadata.
6. The most promising path is curriculum: Phase 1 localization -> planted-lite causal chain -> randomized planted circuits -> IOI/stretch.

## Claims That Are Safe Today

Safe:

- Circuit Detective is a working OpenEnv environment for mechanistic-interpretability tool use.
- The project trained Qwen3.5-2B with SFT + TRL GRPO on HF Jobs.
- Phase 1 succeeded: success improved from 10.4% to 79.2% on 48 eval rollouts.
- Phase 2 is implemented and has exposed a concrete failure mode around causal evidence and terminal tool use.
- The project has honest negative results on stricter causal tasks.

Not safe unless the current job succeeds:

- Do not claim that the agent reliably performs causal circuit validation.
- Do not claim IOI is solved.
- Do not claim planted backdoor/circuit discovery succeeded.
- Do not claim Unsloth was used for the canonical run.

## Reference Index

Core repo references:

- [`README.md`](README.md)
- [`docs/phase_plan.md`](docs/phase_plan.md)
- [`docs/writeup.md`](docs/writeup.md)
- [`phase1_grpo.py`](phase1_grpo.py)
- [`scripts/phase1_sft.py`](scripts/phase1_sft.py)
- [`scripts/phase1_train.py`](scripts/phase1_train.py)
- [`scripts/hf_phase1_job.py`](scripts/hf_phase1_job.py)

Canonical Phase 1 artifacts:

- [`artifacts/phase1_sft64_grpo200_a10g_large/phase1_eval_metrics.json`](artifacts/phase1_sft64_grpo200_a10g_large/phase1_eval_metrics.json)
- [`artifacts/phase1_sft64_grpo200_a10g_large/phase1_reward_curve.png`](artifacts/phase1_sft64_grpo200_a10g_large/phase1_reward_curve.png)
- [`artifacts/phase1_sft64_grpo200_a10g_large/phase1_loss_curve.png`](artifacts/phase1_sft64_grpo200_a10g_large/phase1_loss_curve.png)

Remote artifact root:

- `https://huggingface.co/spaces/ehsaaniqbal/circuit-detective/tree/main/artifacts`

Important HF Jobs:

- Phase 1 canonical: `https://huggingface.co/jobs/ehsaaniqbal/69ecd77ad2c8bd8662bcdd0b`
- Planted-lite strict: `https://huggingface.co/jobs/ehsaaniqbal/69ed97c8d2c8bd8662bcf2a7`
- Planted-lite reward ladder: `https://huggingface.co/jobs/ehsaaniqbal/69edab88d2c8bd8662bcf51d`
- Current naive max planted-lite run: `https://huggingface.co/jobs/ehsaaniqbal/69edc4ddd2c8bd8662bcf87c`

Important model adapters:

- Phase 1 canonical adapter: `https://huggingface.co/ehsaaniqbal/circuit-detective-qwen35-2b-phase1-sft64-grpo200-lora`
- Current in-progress planted-lite adapter target: `https://huggingface.co/ehsaaniqbal/circuit-detective-qwen35-2b-planted-lite-naive-max-lora`
