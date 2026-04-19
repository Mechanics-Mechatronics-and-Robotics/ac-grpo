# AGENTS.md

## Project

AC-GRPO pipeline repair and LunarLander-v2 diagnostic experiments.

Goal: keep the full experimental pipeline simple, reproducible, and honest while comparing BASELINE, AC_LITE, and AC_FULL under clean and noisy settings.

This phase is no longer baseline-only. It is allowed to fix the baseline trainer, AC trainer, run scripts, metrics, logging, plotting, and config plumbing when needed for a correct end-to-end experiment.

---

## Core Question

Does certainty-aware policy optimization improve robustness once the baseline pipeline is strong enough to learn LunarLander-v2 reliably?

---

## Environment

Gym: LunarLander-v2  
Discrete actions: 4  
Observation size: 8  
Max episode length: 1000  

Success = safe landing.

Modes:

- CLEAN
- REWARD_NOISE
- OBS_NOISE

Methods:

- BASELINE
- AC_LITE
- AC_FULL

---

## Metrics

Episode-level:

- return
- success
- episode length

Training diagnostics:

- fraction of mixed-outcome groups
- fraction of discarded groups
- mean group success count
- policy entropy
- gradient norm

AC diagnostics:

- certainty
- policy entropy
- delta_t
- trajectory AUROC
- timestep AUROC

---

## Output Organization

Each `run_all_seeds.py` launch must create exactly one folder in `outputs/`.

Inside it store:

- config.yaml
- aggregate_metrics.csv
- report.md
- plots/
- seed_42/
- seed_0/
- seed_17/
- seed_9/
- seed_3/

All artifacts for one experiment must stay inside one folder.

---

## Failure Checks

Monitor:

- NaN loss
- no learning progress
- entropy collapse
- all groups discarded
- all groups identical
- exploding gradients
- certainty collapse to 0 or 1
- zero certainty gradients

If failure occurs:

1. reduce learning rate
2. lower rollout temperature
3. disable dynamic sampling or use fallback
4. disable clip-higher
5. inspect certainty histogram and group statistics

---

## End
