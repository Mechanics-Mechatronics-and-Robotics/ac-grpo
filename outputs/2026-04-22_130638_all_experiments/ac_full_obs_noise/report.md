# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-22_130638_all_experiments\ac_full_obs_noise`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Notes on experimental modes

- **Reward semantics**: logs include `return_train` (optimizer reward) and `return_env` (raw dense environment return). Plot/report curves use `return_env` unless stated otherwise.
- **REWARD_NOISE**: false-negative successes set terminal `policy_success` to `0` during training. Checkpoint selection evaluates on clean held-out episodes because reward noise is training-only corruption.
- **OBS_NOISE**: adds Gaussian noise \(\sigma=0.1\) to observations at every step.
- **AC v3**: AC methods use standard PPO with certainty-gated advantages; runner-up statistics supervise the certainty network only.

## Seed aggregation

Learning curves are computed **per seed** and then aggregated (mean ± std). This avoids interleaving seeds (each run resets `step` to 0).

## Summary table (mean ± std over 5 seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| OBS_NOISE | AC_FULL | 194.5 ± 47.7 | 0.710 ± 0.119 | 284.8 | 1.000 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| OBS_NOISE | AC_FULL | 0 | 224.6 | 0.800 |
| OBS_NOISE | AC_FULL | 3 | 153.7 | 0.650 |
| OBS_NOISE | AC_FULL | 9 | 134.9 | 0.550 |
| OBS_NOISE | AC_FULL | 17 | 245.2 | 0.850 |
| OBS_NOISE | AC_FULL | 42 | 214.0 | 0.700 |

## Best checkpoint by greedy held-out evaluation

Checkpoints are evaluated greedily on fixed held-out seeds. The final checkpoint is not assumed to be best.

| mode | method | seed | checkpoint | eval return | eval success |
|---|---|---:|---|---:|---:|
| OBS_NOISE | AC_FULL | 0 | AC_FULL_OBS_NOISE_seed0_step0330000_policy.pt | 286.3 | 1.000 |
| OBS_NOISE | AC_FULL | 3 | AC_FULL_OBS_NOISE_seed3_step0080000_policy.pt | 283.7 | 1.000 |
| OBS_NOISE | AC_FULL | 9 | AC_FULL_OBS_NOISE_seed9_step0430000_policy.pt | 288.2 | 1.000 |
| OBS_NOISE | AC_FULL | 17 | AC_FULL_OBS_NOISE_seed17_step0120000_policy.pt | 289.6 | 1.000 |
| OBS_NOISE | AC_FULL | 42 | AC_FULL_OBS_NOISE_seed42_step0260000_policy.pt | 288.7 | 1.000 |

## Certainty AUROC diagnostics

- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.
- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\) (diagnostic).

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| OBS_NOISE | AC_FULL | 0.740 | 0.311 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

