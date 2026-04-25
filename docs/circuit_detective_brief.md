# Circuit Detective — Round 2 Implementation Brief

**Audience:** a coding agent driving implementation. The human lead is highly technical but has never trained an RL model end-to-end. Be honest. Never fabricate APIs, flag numbers that need verification, and pause for decisions rather than guess.

**Pivot context:** the Round 1 submission (InvoiceOps AP Exception) is **not** the starting point for Round 2. That space turned out to be crowded (see the SF gallery: OpsGate, SentinelOps, Tryouts/InsureClaim, Proton-a-thon, Apex HarFeast — all adjacent executions of the same pattern). Round 2 is a fresh direction: **Circuit Detective — an OpenEnv where a small LLM learns to do mechanistic interpretability on frozen transformers.**

---

## 1. TL;DR

**The pitch:** An OpenEnv where an agent learns to *discover circuits* in frozen small transformers, using the same causal-tracing / ablation / logit-lens tools a human interp researcher uses. Ground truth is free: canonical circuits (IOI, induction, successor) are published in peer-reviewed papers. Reward combines circuit-match F1 with ablation faithfulness. The thesis: **mechanistic interpretability is bottlenecked by human researcher hours; we build the first OpenEnv where an automated investigator can be trained and measured.**

**One-liner for the README hook:** *"Wang et al. spent months localizing one circuit in GPT-2 small. We trained a 1.7B model to do the same job in 30 tool calls."* (Verify the "months" figure against the IOI paper's stated timeline before using — don't fabricate.)

---

## 2. Capability gap (why this matters)

Mech interp has infrastructure (TransformerLens, ACDC, SAELens) and published circuits (IOI, induction, successor heads, etc.), but the **investigator** — the thing that forms hypotheses, runs the right interventions, narrows down — is still a human researcher. Anthropic, FAIR, OpenAI superalignment, Redwood all want this automated. ACDC (Conmy et al. 2023, arxiv 2304.14997) is a greedy algorithm, not an agent that learns. We build the *learning* version.

Research lineage (verify every citation before putting it in the README):

- **Wang et al. 2022**, *Interpretability in the Wild: A Circuit for IOI in GPT-2 Small* — arxiv 2211.00593 (canonical ground-truth circuit)
- **Olsson et al. 2022**, *In-context Learning and Induction Heads* (Anthropic) — induction ground truth
- **Meng et al. 2022**, ROME — arxiv 2202.05262 (causal tracing / activation patching primitive)
- **Conmy et al. 2023**, *Towards Automated Circuit Discovery for Mechanistic Interpretability* — arxiv 2304.14997 (ACDC; baseline we compare against)
- **Gould et al. 2023**, *Successor Heads* — arxiv 2312.09230 (additional ground truth circuit for held-out)
- **Hubinger et al. 2024**, *Sleeper Agents* — arxiv 2401.05566 (planted-backdoor methodology)
- **Biderman et al. 2023**, *Pythia* — arxiv 2304.01373 (open small models with full training checkpoints; ideal targets)
- **Nanda & Bloom**, *TransformerLens* (library; verify current version + API on GitHub before wiring)
- **Bricken et al. 2023**; **Templeton et al. 2024**, Anthropic monosemanticity papers (SAE features as an optional richer feature basis)

**SF gallery check:** zero submissions in this lane. Closest adjacent work is "Bears" (Diplomacy overseer via behavior-based interpretability) — different angle. This is an empty corner.

---

## 3. Theme mapping

- **Primary: Theme 3.1 — World Modeling (Professional Tasks).** The frozen transformer *is* the partially observable world. The organizer's own example list includes *"scientific workflow loops (papers → code → experiments)"* and *"tool-discovery benchmarks"* — mech interp is the cleanest real-world instance of both.
- **Secondary: Theme 2 — Long-Horizon Planning & Instruction Following.** Episodes run 20-50 tool calls with sparse final reward. Agent must decompose (which layer? which head? which edge?), track state across the trajectory, and recover from wrong hypotheses.
- **Overlay: Theme 5 — Wild Card.** Auto-interpretability doesn't fit any standard RL-env taxonomy box cleanly. One README sentence acknowledging this is appropriate.

Do not try to claim all five themes. Focus.

---

## 4. Judging criteria mapping (40 / 30 / 20 / 10)

- **Innovation (40%)** — empty corner, frontier research direction, passes the "could a researcher write a paper about this?" test. Novelty is the strongest card here; lead with it.
- **Storytelling (30%)** — the README hook writes itself (*"trained a model to do interp"*). Concrete IOI walk-through before abstraction. `<hypothesis>` tags make trained-agent reasoning legible in transcripts.
- **Showing Improvement (20%)** — F1-vs-step plot, ablation-faithfulness-vs-step plot, held-out generalization (train on IOI, eval on successor), baseline-vs-trained trajectory side-by-side. Commit as `.png` files (see §12).
- **Pipeline (10%)** — verifiable deterministic rewards (no LLM-as-judge). TransformerLens is the infra trust anchor. Standard GRPO via HF TRL.

---

## 5. Positioning

Lead with the capability gap, not the methodology. The pitch is *"mech interp is bottlenecked by human hours, we built the first trainable investigator."* Not *"we built a tool-discovery RL environment."* The former gets clicks, the latter gets skimmed.

The poetic line (use exactly once, in README or video): *"An interpretable model learning to do interpretability."*

---

## 6. Architecture overview

Four components.

| # | Component | Role |
|---|---|---|
| A | **OpenEnv `Environment` subclass** | Wraps a frozen target transformer. Exposes Gym-style `reset / step / state`. |
| B | **Tool surface (MCP or direct)** | The agent's actions: run prompts, patch activations, ablate heads, logit-lens, submit circuit. Decide MCP vs. direct in open questions §13. |
| C | **Ground-truth verifier** | Deterministic rule engine. Holds published circuits + planted backdoor circuits. Scores submissions by F1 + ablation faithfulness. |
| D | **Scenario curriculum** | Behavior description + probe prompt set + ground-truth circuit. Difficulty-tiered. |

**Verifiability invariant (non-negotiable):** ground truth is deterministic — published circuits or env-planted backdoors. No LLM judges in the training reward path. An LLM may be used for behavior description paraphrasing at *scenario construction time* (offline), never in the reward loop.

---

## 7. Target models + behaviors (what to find)

Candidate frozen targets — open questions flag in §13:

- **GPT-2 small (124M)** — canonical, well-studied, Wang et al.'s IOI circuit is here.
- **Pythia-70M / Pythia-160M** — open, full training checkpoints, good for planted-backdoor fine-tuning.
- **2-layer attention-only toy** — trivially small, induction head is the archetype (Olsson et al.).

Behavior probes (start with these; add only if Phase 1 converges):

| Tier | Behavior | Target model | Ground truth source |
|---|---|---|---|
| L1 | Induction | 2L attention-only toy | Olsson et al. 2022 |
| L2 | Indirect Object Identification (IOI) | GPT-2 small | Wang et al. 2022 |
| L3 | Successor (increment day/month/ordinal) | GPT-2 small | Gould et al. 2023 |
| L4 | Planted backdoor (trigger → specific output) | Pythia-70M fine-tuned | Env-known |
| L5 (stretch) | Held-out novel behavior | GPT-2 small | Constructed eval set |

Encode each as `{behavior_name, probe_prompts: [...], ground_truth_heads: [...], ground_truth_edges: [...]}`. Do not attempt to encode full circuits from the papers by hand without reading them end-to-end — pull the head indices from the paper's published supplementary material or an existing open-source replication.

---

## 8. Tool surface (the agent's action space)

Minimum viable tool set:

- `run_on_prompt(prompt) → {logits, top_k_tokens}`
- `patch_activation(layer, position, src_prompt, dst_prompt, component) → {patched_logits}` — activation patching / causal tracing
- `ablate_head(layer, head) → {behavior_score_delta}` — zero-ablate a single head, re-run probes
- `ablate_mlp(layer) → {behavior_score_delta}`
- `zero_attention_edge(layer, head, src_token, dst_token) → {behavior_score_delta}` — edge patching (optional for Phase 1)
- `probe_logit_lens(layer, position) → {top_k_unembed_tokens}`
- `linear_probe(layer, train_prompts, target) → {accuracy}` — optional, only if Phase 1 shows agent can use simpler tools first
- `submit_circuit({heads: [(L,H), ...], mlps: [L, ...], edges: [...]}) → terminal`

**Meta-reasoning discipline** (carried over from InvoiceOps — this pattern is defensible here): require the agent to emit `<hypothesis>` before each experiment and `<decision>` before submit. Rubric can check that experiments correspond to hypotheses (rule-checkable heuristic, e.g. "if hypothesis mentions L8, the next tool call touches L8").

**MCP vs direct tool surface** — open question §13. MCP is organizer-preferred ("RFC #004 actions-as-tool-calls"); direct Python tool surface is simpler and faster to iterate. The InvoiceOps repo had MCP wiring we can reuse if we go that way.

---

## 9. Reward design (RubricDict)

Composable rubrics, all deterministic:

| Rubric | Weight (tentative) | Signal |
|---|---|---|
| `circuit_f1` | 0.40 | F1 of submitted head set vs. ground-truth head set |
| `ablation_faithfulness` | 0.35 | Behavior score drops when submitted circuit is ablated, preserved when preserved. Principled, hardest-to-game signal. |
| `experiment_efficiency` | 0.10 | Fewer unnecessary tool calls → higher. Trains hypothesis-driven inquiry. |
| `hypothesis_discipline` | 0.10 | `<hypothesis>` tags present + experiments correspond (rule-check). |
| `format_validity` | 0.05 | `submit_circuit` schema valid; no malformed actions. Graduated negative reward for invalid JSON / missing tags (cf. OpsGate's −0.5 → 0.0 → 1.0 shaping). |

Tune weights empirically after Phase 1 pilot. Do not commit to these numbers in code before seeing the first reward curves.

**Anti-gaming notes:**
- Ablation faithfulness prevents "submit all heads" degenerate strategy (ablating everything destroys *all* behavior, failing the "preserve what's unrelated" check).
- F1 prevents "submit nothing" degenerate strategy.
- Efficiency must be *bounded* — over-weighting it causes the agent to stop investigating prematurely.

---

## 10. Curriculum

- **Phase 1 (L1 only):** train to solve induction on 2L toy. This is the de-risk milestone. If L1 does not converge in ~2-4 hours of training, escalate.
- **Phase 2 (L1+L2):** add IOI on GPT-2 small. Headline result target.
- **Phase 3 (L1+L2+L3):** add successor as held-out *generalization* test (train only on IOI+induction, evaluate on successor).
- **Phase 4 (stretch, only if ≥1 day slack):** planted backdoor on Pythia-70M.

Curriculum controller can be as simple as random sampling weighted by per-skill success rate (cf. Kube SRE Gym's approach). Do not over-engineer.

---

## 11. Training plan

- **Base agent model:** open — Qwen3-1.7B-Instruct is the default candidate (matches Kube SRE Gym precedent). Gemma 3 small is an alternate. Verify latest HF-available versions and TRL compatibility before committing. See §13.
- **Trainer:** GRPO via HF TRL. Unsloth is the alternate if memory becomes tight. Verify current TRL version and GRPOTrainer API on HF docs before wiring.
- **Frozen target infra:** TransformerLens for patching/ablation. Verify current version compatible with the target models on GitHub.
- **Platform:** HF compute credits land onsite April 25-26. Before then, de-risk on a local GPU or small RunPod instance. A single A100 should be sufficient for Phase 1 pilot.
- **Episode length:** start with a 20-call cap. Raise to 40-50 only if Phase 1 converges and episodes are clearly starving for budget.
- **Batch / rollout shape:** follow TRL GRPOTrainer defaults initially. Tune after Phase 1.

---

## 12. Submission artifacts + hard validation rules

**Automated validation runs first. If any item below is missing/broken at the deadline, the submission never reaches a human judge, regardless of how good the idea is. Verify each explicitly before submitting.**

- [ ] **Public, cloneable HF Space** at the submitted URL. Test from a logged-out browser. Private spaces / dead links / 404s are an automatic out.
- [ ] **Valid OpenEnv structure:** proper `Environment` / `MCPEnvironment` base class, Gym-style `reset / step / state`, parseable `openenv.yaml`.
- [ ] **Training evidence committed to repo as `.png` / `.jpg`:** at minimum a loss curve and a reward curve. Wandb-only links and plots that live only in Colab cells do **not** count — they may not be reachable when validation runs.
- [ ] **Runnable training script** (Unsloth, HF TRL, or other), preferably as a Colab notebook re-executable end to end. A Python script is acceptable.
- [ ] **README links every deliverable:** HF Space, training notebook, writeup (blog / video / slides), with key plots embedded inline. If validation can't reach a deliverable from the README, it counts as missing.

**Beyond validation (scoring surface):**

- Mini-blog on HF OR <2-min video on YouTube OR short slide deck. Linked from README.
- Do not commit large video files to the HF Space (organizer request). Link externally.
- README narrative arc: Problem → Environment → Results → Why it matters.
- Embed plots with one-line captions. Label both axes with units.

---

## 13. Open questions (raise with the human lead before committing)

1. **Base agent model.** Qwen3-1.7B-Instruct vs Gemma 3 small vs SmolLM3. Verify latest HF availability + TRL GRPO compatibility for each.
2. **Trainer.** HF TRL GRPOTrainer (default) vs Unsloth. Unsloth gives memory savings but adds a compatibility surface.
3. **Tool surface.** MCP server (organizer-preferred per RFC #004) vs direct Python tool surface (simpler, faster to iterate). Which does current OpenEnv support best for non-chat agents?
4. **Frozen target scope.** Start with 2L toy + GPT-2 small, or include Pythia-70M from the start? Pythia brings planted-backdoor but costs setup time.
5. **Activation feature granularity.** Heads + MLPs only, or add SAE features (Anthropic-style monosemantic features) as an additional feature basis? SAE adds complexity but is closer to frontier work.
6. **Episode length cap.** Start at 20, 30, or 50? Trade-off: longer episodes give the agent room to reason but make the reward sparser.
7. **Ablation faithfulness numeric definition.** How much behavior-drop qualifies as "ablation destroys the circuit"? Threshold vs. continuous score.
8. **Training platform before onsite credits.** Local GPU / RunPod / HF Jobs for Phase 1 pilot?
9. **Agent system prompt / scratchpad format.** How much structure to impose? `<hypothesis>` / `<experiment>` / `<observation>` / `<decision>` tags, or looser?
10. **Eval set size.** For generalization plot, how many held-out scenarios? (Suggest 20-50 paired with seed variation.)

---

## 14. Risks + mitigations (honest)

1. **1.7B agent may not reason coherently about head/layer indices over long episodes.** This is the biggest risk. **Mitigation:** Phase 1 L1 pilot is a de-risk gate. If a 1.7B baseline can't even produce legal tool-call sequences on the 2L toy, either step up to 3-4B or simplify the tool surface before continuing.
2. **Engineering load is heavy.** TransformerLens wiring + OpenEnv wrapper + ground-truth circuit encoding + MCP (if chosen) is ~12-16h before training starts. **Mitigation:** scaffold in a sharp order — env skeleton first, then one tool at a time, with unit tests per tool. Don't write 5 tools and debug them together.
3. **Niche pitch.** Judge unfamiliar with mech interp needs ramp-up. **Mitigation:** README opens with one concrete IOI example (image of the circuit graph) *before* any abstraction. 2-min video teaches the concept in the first 30 seconds.
4. **Sparse rewards hurt GRPO.** Final-only rewards over 30+ tool calls mean high variance. **Mitigation:** dense shaping via `format_validity` and `hypothesis_discipline` early; bounded efficiency reward after format is solid.
5. **HF Space must actually run.** Async validation clones and executes. **Mitigation:** write a smoke-test script that simulates the validator's path (clone → install → `reset/step/state` → small rollout). Run it in CI.
6. **Claims must be honest.** If training only converges on L1, the README should say *"converges on induction in 2L toy, partial on IOI"* rather than overclaiming. Honest partial results are fine; overclaiming will be caught.

---

## 15. What carries over from Round 1 (and what does not)

**Carries over:**
- OpenEnv + HF TRL + GRPO scaffolding patterns
- RubricDict composition philosophy
- Meta-reasoning tag discipline (`<hypothesis>` / `<decision>`)
- Graduated-reward format-validity shaping (cf. OpsGate)
- Honest-claims / verify-before-citing posture

**Does not carry over:**
- The InvoiceOps domain (MCP servers, rubrics, scenarios — all discarded)
- The Claude-driven adversarial designer (Circuit Detective's "adversary" is the frozen target's complexity, not a live LLM)
- The expert-persona auditor curriculum (replaced by difficulty-tiered circuit curriculum)
- Any framing language about enterprise / AP / fraud

Start the fresh repo at `/Users/ehsaan/Documents/lab/circuit_detective` (or similar — confirm path with the human lead). The InvoiceOps repo is reference only.

---

## 16. References (verify each before citing in README / video / slides)

Papers:
1. Wang et al. 2022 — IOI in GPT-2 Small — arxiv 2211.00593
2. Olsson et al. 2022 — Induction Heads (Anthropic)
3. Meng et al. 2022 — ROME — arxiv 2202.05262
4. Conmy et al. 2023 — ACDC — arxiv 2304.14997
5. Gould et al. 2023 — Successor Heads — arxiv 2312.09230
6. Hubinger et al. 2024 — Sleeper Agents — arxiv 2401.05566
7. Biderman et al. 2023 — Pythia — arxiv 2304.01373
8. Bricken et al. 2023 / Templeton et al. 2024 — Anthropic monosemanticity (SAE line)
9. Nanda 2023 — *A Comprehensive Mechanistic Interpretability Explainer* (Neel Nanda's public notes; useful for README onboarding)

Libraries / infra (verify current version + API):
- **TransformerLens** (Nanda & Bloom) — GitHub
- **SAELens** — GitHub (optional, only if SAE features are in scope)
- **HF TRL** `GRPOTrainer` — HF docs
- **OpenEnv** latest release — HF Meta repo + organizer docs
- **Unsloth** — GitHub (alternate trainer)

Organizer docs:
- `/Users/ehsaan/Downloads/[External] Apr '26 OpenEnv Hackathon Themes & Judging Criteria.md`
- `/Users/ehsaan/Downloads/old_hackathon_ideas.md` (SF gallery — reference for empty-corner claims)

---

## Closing note

Circuit Detective is a research-flavored env. It wins on Innovation (empty corner, frontier topic) and Storytelling (poetic narrative + legible `<hypothesis>` tags in trained trajectories). The hardest risk is whether a 1.7B model can coherently reason about activations. **Phase 1 L1 is the de-risk milestone — budget it as the first 8-12 hours of work and do not move past it until it either converges or clearly won't.** Honest partial results on Phase 1 beat overclaimed full results every time.
