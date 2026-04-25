# Circuit Detective Phase Plan

This is the working execution plan. It separates what is implemented from what is still a target.

## Current Status

- Implemented: Phase 1 trained; Phase 2 ablation-required mode implemented and locally smoke-tested.
- Agent model: `Qwen/Qwen3.5-2B`.
- Trainer path: HF TRL `GRPOTrainer` with PEFT/QLoRA and bitsandbytes.
- Frozen target: TransformerLens `attn-only-2l`.
- Current trained task: find the dominant induction head and call `submit_circuit`.
- Tool surface: direct Python methods exposed through TRL `environment_factory`.
- OpenEnv surface: deterministic `reset`, `step`, and `state`.
- Canonical evidence: SFT warm-start plus 200-step GRPO improved eval success from 10.4% to 79.2% on 48 rollouts.
- Current gate status: Phase 1 PASS. The canonical final LoRA adapter is uploaded at `ehsaaniqbal/circuit-detective-qwen35-2b-phase1-sft64-grpo200-lora`.

## Phase 1 - L1 Induction Pilot

Goal: prove that a small model can learn the basic circuit-investigation protocol on a two-layer attention-only transformer.

Implemented tools:

- `list_tools`
- `run_probe`
- `inspect_induction_scores`
- `ablate_head`
- `submit_circuit`

Current learned protocol target:

```text
inspect_induction_scores(top_k=3) -> submit_circuit([top_head])
```

Preferred stronger protocol:

```text
inspect_induction_scores(top_k=3) -> ablate_head(top_head) -> submit_circuit([top_head])
```

Exit gate:

- 150-step run completes successfully.
- `eval_after_success_rate` clearly beats `eval_before_success_rate`.
- Target threshold: `eval_after_success_rate >= 0.4` on at least 32 eval rollouts.
- Reward curve is not flat.
- Loss/gradients show nonzero training signal.
- Artifacts exist as local and HF Space files: loss curve, reward curve, eval JSON.

If Phase 1 stalls:

- Add tiny SFT warm-start for `inspect -> submit`.
- Then rerun GRPO.
- If still unstable, simplify further or step model size up.

## Phase 2 - Ablation-Required Causal Validation

Goal: make the task more mechanistic, not just score-ranking.

Implementation status: added as an explicit training/eval mode. Phase 1 defaults are unchanged.

Changes:

- Require ablation before full terminal reward.
- Reward causal validation via behavior drop after `ablate_head`.
- Track `causal_success_rate`, `ablate_submitted_rate`, and `ablation_faithfulness`.
- Add decoy heads where high induction score is not sufficient. Not implemented yet.

Exit gate:

- Agent uses `ablate_head` in a meaningful fraction of successful rollouts.
- Success stays above Phase 1 random/base baseline.
- We can show a trajectory where the agent verifies the candidate before submitting.

## Phase 3 - L2 IOI On GPT-2 Small

Goal: move from toy induction to a canonical published circuit.

Changes:

- Add GPT-2 small target backend.
- Add IOI probe prompts.
- Encode verified IOI ground-truth heads only after checking source material.
- Expand submission schema if needed for multiple heads.

Exit gate:

- OpenEnv smoke still passes.
- Agent achieves nonzero improvement over base model.
- README claims remain conservative if performance is partial.

## Phase 4 - Held-Out Generalization

Goal: show the agent is learning a reusable investigation policy, not only memorizing L1.

Candidate:

- Train on induction plus IOI.
- Evaluate on successor heads or a held-out scenario.

Exit gate:

- Baseline-vs-trained eval plot on held-out task.
- Honest result summary, even if partial.

## Stretch - Planted Backdoor Circuit

Goal: create an env-known target circuit where ground truth is exact and not dependent on hand-copying paper head indices.

Candidate:

- Pythia-70M or Pythia-160M.
- Fine-tune or construct a trigger behavior.
- Score whether the agent identifies the planted causal components.

Only do this if Phase 1 and submission packaging are already safe.

## Submission Packaging Track

This runs in parallel once Phase 1 has usable evidence.

- Keep HF Space public and cloneable.
- Keep `openenv.yaml` parseable.
- Keep `scripts/validator_smoke.py` passing.
- Commit `.png` training curves.
- Link HF Space, notebook/script, writeup/video/slides from README.
- Avoid committing large video files.

## RL Techniques: Implemented vs Planned

Implemented now:

- GRPO with HF TRL.
- Direct tool-calling environment via TRL `environment_factory`.
- Deterministic OpenEnv reward.
- Trainer-side dense reward shaping for Phase 1.
- Format/tool-surface discipline through explicit typed tool methods.
- Rollout diagnostics: submit rate, success rate, F1, tool-use rates.
- Phase 2 causal diagnostics: causal success, submitted-head ablation rate, ablation faithfulness.

Partially implemented:

- Rubric-style decomposition exists conceptually, but code is still a lean Phase 1 reward function rather than a full `RubricDict`.
- Ablation is available as a tool and is required for full credit in the explicit Phase 2 mode.

Not implemented yet:

- `<hypothesis>` / `<decision>` meta-reasoning tags.
- Rule-checked hypothesis discipline.
- Curriculum sampling across multiple tasks.

Implemented after initial GRPO-only runs:

- Optional tiny SFT warm-start for the Phase 1 tool protocol.
- The HF job launcher can run SFT and GRPO in one ephemeral job, avoiding a second model download.

Recommended order:

1. Keep Phase 1 evidence and adapter-freezing runs reproducible.
2. Start Phase 2 by requiring meaningful ablation before full reward.
3. Add meta-reasoning tags only after the base tool protocol works, because tag enforcement adds another failure mode.
4. Expand to harder circuits only after OpenEnv validation and submission packaging stay green.
