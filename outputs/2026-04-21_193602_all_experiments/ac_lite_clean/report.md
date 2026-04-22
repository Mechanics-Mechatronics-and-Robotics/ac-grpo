# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-21_193602_all_experiments\ac_lite_clean`

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
| CLEAN | AC_LITE | 212.5 ± 35.1 | 0.700 ± 0.132 | 289.2 | 1.000 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_LITE | 0 | 258.8 | 0.900 |
| CLEAN | AC_LITE | 3 | 228.2 | 0.750 |
| CLEAN | AC_LITE | 9 | 182.3 | 0.650 |
| CLEAN | AC_LITE | 17 | 220.1 | 0.650 |
| CLEAN | AC_LITE | 42 | 172.9 | 0.550 |

## Best checkpoint by greedy held-out evaluation

Checkpoints are evaluated greedily on fixed held-out seeds. The final checkpoint is not assumed to be best.

| mode | method | seed | checkpoint | eval return | eval success |
|---|---|---:|---|---:|---:|
| CLEAN | AC_LITE | 0 | AC_LITE_CLEAN_seed0_step0020000_policy.pt | 288.0 | 1.000 |
| CLEAN | AC_LITE | 3 | AC_LITE_CLEAN_seed3_step0210000_policy.pt | 289.4 | 1.000 |
| CLEAN | AC_LITE | 9 | AC_LITE_CLEAN_seed9_step0030000_policy.pt | 289.7 | 1.000 |
| CLEAN | AC_LITE | 17 | AC_LITE_CLEAN_seed17_step0010000_policy.pt | 289.6 | 1.000 |
| CLEAN | AC_LITE | 42 | AC_LITE_CLEAN_seed42_step0050000_policy.pt | 289.2 | 1.000 |

## Certainty AUROC diagnostics

- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.
- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\) (diagnostic).

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_LITE | 0.382 | 0.447 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

