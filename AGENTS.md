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

## Best Baseline Settings

Use the repaired baseline settings unless an experiment explicitly overrides them:

```python
{
    "dynamic_sampling": True,
    "grouped_rollouts": True,
    "rollout_temperature": 1.0,
    "epsilon_low": 0.2,
    "epsilon_high": 0.2,
    "total_steps": 60_000,
    "dynamic_sampling_warmup_steps": 150_000,
}
```

Note: if `total_steps < dynamic_sampling_warmup_steps`, dynamic sampling will not activate in that run. This is acceptable for short smoke/debug runs, but should be checked before final claims.

---

## Training Defaults

Steps per update: 2048  
Batch size: 64  
Optimizer: Adam  
Learning rate: 3e-4  
Gamma: 0.99  
Lambda: 0.95  
Clip default: low=0.2, high=0.2  

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
- summary.json
- aggregate_metrics.csv
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
