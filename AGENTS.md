# AGENTS.md

## Project

Implement and maintain a `LunarLander-v2` diagnostic that stays as close as practical to the **SimpleVLA-RL intuition** while remaining minimal, reproducible, and scientifically honest.

README.md is the source of truth for:

* task setup
* sparse reward semantics
* method definitions
* AC-models mathematical structure
* evaluation protocol

This file defines execution rules for code changes only.

---

## Goal

Run one reproducible experiment grid with:

Methods:

* BASELINE
* AC_LITE
* AC_FULL

Modes:

* CLEAN
* OBS_NOISE
* REWARD_NOISE

Seeds:

* 42
* 0
* 17
* 9
* 3

All branches must start from the same pretrained anchor checkpoint when a pretrained path is provided.

The pretrained anchor may come from a dense-reward baseline. That is acceptable. The sparse-reward phase is an adaptation study, not scratch learning.

---

## Main Principle

Stay close to the current project structure and to the SimpleVLA-style workflow:

1. common pretrained anchor
2. binary outcome-driven RL adaptation
3. grouped rollouts and dynamic sampling
4. checkpoint-wise greedy evaluation
5. best checkpoint selected by held-out evaluation, not by final training step

Avoid broad refactoring.

Do not rewrite theory.
Do not expand the model family.
Do not add new reward terms.
Do not introduce large abstractions unless required for correctness.

---

## Fixed Default Recipe

Unless clearly broken, use these defaults:

* sparse terminal reward for learning
* grouped_rollouts = true
* dynamic_sampling = true
* dynamic_sampling_fallback_on_empty = true
* rollout_temperature = 1.0
* epsilon_low = 0.2
* epsilon_high = 0.2

All experiment branches should use the same defaults unless a mode-specific corruption is already part of the implementation.

---

## Required Behavior

### 1. Shared pretrained anchor

All branches must load the same pretrained checkpoint when `pretrained_policy_path` is provided.

### 2. Checkpoint 0 is part of evaluation

The pretrained anchor must be treated as checkpoint 0 and must be eligible to win checkpoint selection.

Do not assume RL adaptation improves the model.

### 3. One top-level run = one top-level output folder

A global experiment launch must create exactly one timestamped folder under `outputs/`.

Inside it create exactly these branch folders:

* baseline_sparse_clean
* baseline_sparse_obs_noise
* baseline_sparse_reward_noise
* baseline_dense_clean
* baseline_dense_obs_noise
* baseline_dense_reward_noise
* ac_lite_sparse_clean
* ac_lite_sparse_obs_noise
* ac_lite_sparse_reward_noise
* ac_lite_dense_clean
* ac_lite_dense_obs_noise
* ac_lite_dense_reward_noise
* ac_full_sparse_clean
* ac_full_sparse_obs_noise
* ac_full_sparse_reward_noise

Each branch folder must contain:

* config.yaml
* summary.json
* report.md
* seed_42/
* seed_0/
* seed_17/
* seed_9/
* seed_3/

### 4. Checkpointed training

Save checkpoints at a fixed interval.
Default interval: every 10k steps.
If the trainer already supports another checkpoint cadence internally, preserve correctness and document the actual interval.

### 5. Evaluation by checkpoint

Evaluate saved checkpoints on fixed held-out episodes.
Do not assume the final checkpoint is best.

### 6. Greedy evaluation

Use greedy evaluation consistently for checkpoint selection and reporting.

### 7. Mode-matched evaluation

Checkpoint selection must use the matching evaluation mode:

* CLEAN branch → CLEAN eval
* OBS_NOISE branch → OBS_NOISE eval
* REWARD_NOISE branch → REWARD_NOISE eval

---

## Toy-task safeguard

Dynamic sampling may produce batches with no mixed-outcome groups in LunarLander.

If a batch has no mixed-outcome groups:

* fall back to the sampled unfiltered batch
* log that fallback clearly

Do not silently skip updates.

This safeguard is allowed because LunarLander is only a toy diagnostic task.

---

## Metrics

Per seed and branch, preserve existing metrics and logging where possible.

Minimum required metrics:

* raw environment return
* policy outcome / success
* episode length
* mixed-group fraction
* discarded-group fraction
* mean successes per group
* policy entropy

For AC methods also keep:

* certainty
* delta_t
* trajectory AUROC
* timestep AUROC

Aggregate over seeds using mean ± std.

---

## Failure checks

Watch for:

* NaN loss
* no learning progress
* entropy collapse
* certainty collapse
* zero certainty gradients
* all groups identical
* no mixed groups for long intervals
* checkpoint degradation after longer training
* accidental use of dense environment reward for optimization

If failure occurs, only use minimal fixes in this order:

1. verify pretrained checkpoint loading
2. verify sparse terminal reward is actually used for training
3. verify grouped rollout and dynamic sampling logic
4. verify fallback-to-unfiltered-batch behavior
5. lower adaptation learning rate
6. shorten training horizon

Do not redesign the project.

---

## What not to do

* do not rewrite README.md theory into other files
* do not change sparse reward into dense reward for one method only
* do not compare dense baseline against sparse AC as the main result
* do not add many hyperparameter sweeps
* do not optimize each branch separately
* do not remove existing AC_LITE / AC_FULL code paths
* do not switch to a completely new RL framework

---

## Deliverable

Produce one reproducible experiment grid with:

* one common pretrained anchor
* 9 branches
* 5 seeds
* sparse reward for all compared methods
* checkpoint-wise greedy evaluation
* one detailed final report per experiment folder

The code should remain simple enough that the later VLA pipeline can reuse the same logic:
common anchor → binary outcome RL → checkpoint selection.
