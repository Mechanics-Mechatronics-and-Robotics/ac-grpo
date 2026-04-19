# AGENTS.md

## Project

Implement one final LunarLander experiment grid that is maximally close to the SimpleVLA-RL workflow, while preserving the current README.md as the main human-readable project description.

README.md is the source of truth for:
- scientific motivation
- AC-GRPO theory
- prior experiment history
- naming of methods and modes

This file only defines execution rules for code changes.

---

## Goal

Run one clean, reproducible experiment grid with:

Methods:
- BASELINE
- AC_LITE
- AC_FULL

Modes:
- CLEAN
- OBS_NOISE
- REWARD_NOISE

Seeds:
- 42
- 0
- 17
- 9
- 3

All branches must start from the same pretrained clean checkpoint.

---

## Main Principle

Keep the implementation as close as possible to the current project structure and to the SimpleVLA-RL training style, but avoid broad refactoring.

Do not rewrite theory.
Do not expand the model family.
Do not add new reward terms.
Do not introduce large new abstractions unless required for correctness.

---

## Fixed Default Recipe

Unless clearly broken, use these defaults for the final grid:

- grouped_rollouts = true
- dynamic_sampling = true
- rollout_temperature = 1.0
- epsilon_low = 0.2
- epsilon_high = 0.2

All 9 experiment branches should use the same defaults unless mode-specific behavior already exists.

---

## Required Behavior

### 1. Shared pretrained anchor
All branches must load the same pretrained clean checkpoint.

### 2. One top-level run = one top-level output folder
A global experiment launch must create exactly one timestamped folder under `outputs/`.

Inside it create exactly these branch folders:

- baseline_clean
- baseline_obs_noise
- baseline_reward_noise
- ac_lite_clean
- ac_lite_obs_noise
- ac_lite_reward_noise
- ac_full_clean
- ac_full_obs_noise
- ac_full_reward_noise

Each branch folder must contain:
- config.yaml
- summary.json
- report.md
- plots/
- seed_42/
- seed_0/
- seed_17/
- seed_9/
- seed_3/

### 3. Checkpointed training
Save checkpoints at a fixed interval, default every 10k steps.

### 4. Evaluation by checkpoint
Evaluate saved checkpoints on fixed held-out seeds / episodes.
Do not assume the final checkpoint is best.

### 5. Greedy evaluation
Use greedy evaluation consistently.

---

## Toy-task safeguard

Dynamic sampling may produce batches with no mixed-outcome groups in LunarLander.

If a batch has no mixed-outcome groups:
- fall back to the unfiltered batch
- log that fallback clearly

Do not silently skip updates.

This safeguard is allowed because LunarLander is only a toy diagnostic task.

---

## Metrics

Per seed and branch, preserve existing metrics and logging where possible.

Minimum required metrics:
- return
- success
- episode length
- mixed-group fraction
- discarded-group fraction
- mean successes per group
- policy entropy

For AC methods also keep:
- certainty
- delta_t
- trajectory AUROC
- timestep AUROC

Aggregate over seeds using mean ± std.

---

## Failure checks

Watch for:
- NaN loss
- no learning progress
- entropy collapse
- certainty collapse
- zero certainty gradients
- all groups identical
- no mixed groups for long intervals
- checkpoint degradation after longer training

If failure occurs, only use minimal fixes:
1. verify checkpoint loading
2. verify grouped rollout and dynamic sampling logic
3. verify fallback-to-unfiltered-batch behavior
4. lower adaptation learning rate
5. shorten training horizon

Do not redesign the project.

---

## What not to do

- do not rewrite README.md content
- do not duplicate theory from README.md into new files
- do not add many hyperparameter sweeps
- do not optimize each branch separately
- do not remove existing AC_LITE / AC_FULL paths
- do not switch to a completely new RL framework

---

## Deliverable

Produce one reproducible final experiment grid with:
- one pretrained clean anchor
- 9 branches
- 5 seeds
- checkpoint-wise evaluation
- one detailed final report per experiment (one output folder - one report)