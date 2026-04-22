# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-21_193602_all_experiments\ac_full_obs_noise`

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
| OBS_NOISE | AC_FULL | 200.1 ± 54.3 | 0.720 ± 0.175 | 277.6 | 0.970 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| OBS_NOISE | AC_FULL | 0 | 210.0 | 0.750 |
| OBS_NOISE | AC_FULL | 3 | 133.1 | 0.500 |
| OBS_NOISE | AC_FULL | 9 | 265.0 | 0.950 |
| OBS_NOISE | AC_FULL | 17 | 157.5 | 0.600 |
| OBS_NOISE | AC_FULL | 42 | 234.7 | 0.800 |

## Best checkpoint by greedy held-out evaluation

Checkpoints are evaluated greedily on fixed held-out seeds. The final checkpoint is not assumed to be best.

| mode | method | seed | checkpoint | eval return | eval success |
|---|---|---:|---|---:|---:|
| OBS_NOISE | AC_FULL | 0 | AC_FULL_OBS_NOISE_seed0_final_policy.pt | 286.9 | 1.000 |
| OBS_NOISE | AC_FULL | 3 | AC_FULL_OBS_NOISE_seed3_step0080000_policy.pt | 283.7 | 1.000 |
| OBS_NOISE | AC_FULL | 9 | AC_FULL_OBS_NOISE_seed9_step0050000_policy.pt | 285.0 | 1.000 |
| OBS_NOISE | AC_FULL | 17 | AC_FULL_OBS_NOISE_seed17_step0120000_policy.pt | 289.6 | 1.000 |
| OBS_NOISE | AC_FULL | 42 | AC_FULL_OBS_NOISE_seed42_step0230000_policy.pt | 288.0 | 1.000 |

## Certainty AUROC diagnostics

- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.
- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\) (diagnostic).

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| OBS_NOISE | AC_FULL | 0.698 | 0.365 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

