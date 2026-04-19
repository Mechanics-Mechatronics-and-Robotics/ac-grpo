# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-19_145000_all_experiments`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Notes on experimental modes

- **REWARD_NOISE**: false-negative successes also penalize the terminal rollout reward used by PPO/GAE, so the policy update sees the corruption.
- **OBS_NOISE**: adds Gaussian noise \(\sigma=0.1\) to observations at every step.

## Seed aggregation

Learning curves are computed **per seed** and then aggregated (mean ± std). This avoids interleaving seeds (each run resets `step` to 0).

## Summary table (mean ± std over 5 seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL | -66.8 ± 11.0 | 0.000 ± 0.000 | 237.1 | 0.790 |
| CLEAN | AC_LITE | -61.1 ± 12.3 | 0.000 ± 0.000 | 229.3 | 0.770 |
| CLEAN | BASELINE | -36.0 ± 4.6 | 0.000 ± 0.000 | 220.3 | 0.720 |
| OBS_NOISE | AC_FULL | -87.3 ± 10.2 | 0.000 ± 0.000 | 95.9 | 0.220 |
| OBS_NOISE | AC_LITE | -87.6 ± 9.4 | 0.000 ± 0.000 | 99.0 | 0.230 |
| OBS_NOISE | BASELINE | -60.9 ± 11.0 | 0.000 ± 0.000 | 87.3 | 0.160 |
| REWARD_NOISE | AC_FULL | -60.6 ± 8.9 | 0.000 ± 0.000 | 220.3 | 0.600 |
| REWARD_NOISE | AC_LITE | -70.6 ± 18.4 | 0.000 ± 0.000 | 213.1 | 0.570 |
| REWARD_NOISE | BASELINE | -41.4 ± 2.8 | 0.000 ± 0.000 | 215.1 | 0.570 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL | 0 | -74.3 | 0.000 |
| CLEAN | AC_FULL | 3 | -75.6 | 0.000 |
| CLEAN | AC_FULL | 9 | -53.8 | 0.000 |
| CLEAN | AC_FULL | 17 | -55.7 | 0.000 |
| CLEAN | AC_FULL | 42 | -74.6 | 0.000 |
| CLEAN | AC_LITE | 0 | -50.8 | 0.000 |
| CLEAN | AC_LITE | 3 | -46.7 | 0.000 |
| CLEAN | AC_LITE | 9 | -61.9 | 0.000 |
| CLEAN | AC_LITE | 17 | -71.0 | 0.000 |
| CLEAN | AC_LITE | 42 | -75.0 | 0.000 |
| CLEAN | BASELINE | 0 | -33.6 | 0.000 |
| CLEAN | BASELINE | 3 | -40.5 | 0.000 |
| CLEAN | BASELINE | 9 | -34.2 | 0.000 |
| CLEAN | BASELINE | 17 | -30.6 | 0.000 |
| CLEAN | BASELINE | 42 | -41.2 | 0.000 |
| OBS_NOISE | AC_FULL | 0 | -90.9 | 0.000 |
| OBS_NOISE | AC_FULL | 3 | -101.1 | 0.000 |
| OBS_NOISE | AC_FULL | 9 | -88.5 | 0.000 |
| OBS_NOISE | AC_FULL | 17 | -82.2 | 0.000 |
| OBS_NOISE | AC_FULL | 42 | -73.6 | 0.000 |
| OBS_NOISE | AC_LITE | 0 | -97.3 | 0.000 |
| OBS_NOISE | AC_LITE | 3 | -96.9 | 0.000 |
| OBS_NOISE | AC_LITE | 9 | -77.2 | 0.000 |
| OBS_NOISE | AC_LITE | 17 | -79.5 | 0.000 |
| OBS_NOISE | AC_LITE | 42 | -86.9 | 0.000 |
| OBS_NOISE | BASELINE | 0 | -49.1 | 0.000 |
| OBS_NOISE | BASELINE | 3 | -55.3 | 0.000 |
| OBS_NOISE | BASELINE | 9 | -54.9 | 0.000 |
| OBS_NOISE | BASELINE | 17 | -74.8 | 0.000 |
| OBS_NOISE | BASELINE | 42 | -70.2 | 0.000 |
| REWARD_NOISE | AC_FULL | 0 | -58.0 | 0.000 |
| REWARD_NOISE | AC_FULL | 3 | -50.6 | 0.000 |
| REWARD_NOISE | AC_FULL | 9 | -55.0 | 0.000 |
| REWARD_NOISE | AC_FULL | 17 | -66.9 | 0.000 |
| REWARD_NOISE | AC_FULL | 42 | -72.5 | 0.000 |
| REWARD_NOISE | AC_LITE | 0 | -50.8 | 0.000 |
| REWARD_NOISE | AC_LITE | 3 | -75.9 | 0.000 |
| REWARD_NOISE | AC_LITE | 9 | -63.1 | 0.000 |
| REWARD_NOISE | AC_LITE | 17 | -99.5 | 0.000 |
| REWARD_NOISE | AC_LITE | 42 | -63.7 | 0.000 |
| REWARD_NOISE | BASELINE | 0 | -38.2 | 0.000 |
| REWARD_NOISE | BASELINE | 3 | -41.4 | 0.000 |
| REWARD_NOISE | BASELINE | 9 | -45.7 | 0.000 |
| REWARD_NOISE | BASELINE | 17 | -39.9 | 0.000 |
| REWARD_NOISE | BASELINE | 42 | -41.7 | 0.000 |

## Best checkpoint by greedy held-out evaluation

Checkpoints are evaluated greedily on fixed held-out seeds. The final checkpoint is not assumed to be best.

| mode | method | seed | checkpoint | eval return | eval success |
|---|---|---:|---|---:|---:|
| CLEAN | AC_FULL | 0 | AC_FULL_CLEAN_seed0_step0040000_policy.pt | 278.0 | 1.000 |
| CLEAN | AC_FULL | 3 | AC_FULL_CLEAN_seed3_step0010000_policy.pt | 288.9 | 1.000 |
| CLEAN | AC_FULL | 9 | AC_FULL_CLEAN_seed9_step0010000_policy.pt | 288.4 | 1.000 |
| CLEAN | AC_FULL | 17 | AC_FULL_CLEAN_seed17_step0010000_policy.pt | 286.3 | 1.000 |
| CLEAN | AC_FULL | 42 | AC_FULL_CLEAN_seed42_step0030000_policy.pt | 285.1 | 1.000 |
| CLEAN | AC_LITE | 0 | AC_LITE_CLEAN_seed0_step0010000_policy.pt | 287.3 | 1.000 |
| CLEAN | AC_LITE | 3 | AC_LITE_CLEAN_seed3_step0040000_policy.pt | 280.5 | 1.000 |
| CLEAN | AC_LITE | 9 | AC_LITE_CLEAN_seed9_step0010000_policy.pt | 287.5 | 1.000 |
| CLEAN | AC_LITE | 17 | AC_LITE_CLEAN_seed17_step0030000_policy.pt | 287.2 | 1.000 |
| CLEAN | AC_LITE | 42 | AC_LITE_CLEAN_seed42_step0020000_policy.pt | 285.3 | 1.000 |
| CLEAN | BASELINE | 0 | BASELINE_CLEAN_seed0_step0030000.pt | 283.0 | 1.000 |
| CLEAN | BASELINE | 3 | BASELINE_CLEAN_seed3_step0010000.pt | 284.2 | 1.000 |
| CLEAN | BASELINE | 9 | BASELINE_CLEAN_seed9_step0010000.pt | 288.0 | 1.000 |
| CLEAN | BASELINE | 17 | BASELINE_CLEAN_seed17_step0010000.pt | 288.3 | 1.000 |
| CLEAN | BASELINE | 42 | BASELINE_CLEAN_seed42_step0030000.pt | 283.8 | 1.000 |
| OBS_NOISE | AC_FULL | 0 | AC_FULL_OBS_NOISE_seed0_step0030000_policy.pt | 280.1 | 1.000 |
| OBS_NOISE | AC_FULL | 3 | AC_FULL_OBS_NOISE_seed3_step0060000_policy.pt | 255.1 | 0.933 |
| OBS_NOISE | AC_FULL | 9 | AC_FULL_OBS_NOISE_seed9_step0010000_policy.pt | 258.4 | 0.800 |
| OBS_NOISE | AC_FULL | 17 | AC_FULL_OBS_NOISE_seed17_step0040000_policy.pt | 211.7 | 0.667 |
| OBS_NOISE | AC_FULL | 42 | AC_FULL_OBS_NOISE_seed42_step0040000_policy.pt | 248.3 | 0.933 |
| OBS_NOISE | AC_LITE | 0 | AC_LITE_OBS_NOISE_seed0_step0040000_policy.pt | 252.6 | 0.933 |
| OBS_NOISE | AC_LITE | 3 | AC_LITE_OBS_NOISE_seed3_step0060000_policy.pt | 221.0 | 0.933 |
| OBS_NOISE | AC_LITE | 9 | AC_LITE_OBS_NOISE_seed9_step0040000_policy.pt | 226.6 | 0.867 |
| OBS_NOISE | AC_LITE | 17 | AC_LITE_OBS_NOISE_seed17_step0030000_policy.pt | 254.8 | 0.867 |
| OBS_NOISE | AC_LITE | 42 | AC_LITE_OBS_NOISE_seed42_step0040000_policy.pt | 270.7 | 1.000 |
| OBS_NOISE | BASELINE | 0 | BASELINE_OBS_NOISE_seed0_step0050000.pt | 247.6 | 0.933 |
| OBS_NOISE | BASELINE | 3 | BASELINE_OBS_NOISE_seed3_step0030000.pt | 273.3 | 0.933 |
| OBS_NOISE | BASELINE | 9 | BASELINE_OBS_NOISE_seed9_step0030000.pt | 220.2 | 0.733 |
| OBS_NOISE | BASELINE | 17 | BASELINE_OBS_NOISE_seed17_step0040000.pt | 256.0 | 0.933 |
| OBS_NOISE | BASELINE | 42 | BASELINE_OBS_NOISE_seed42_step0050000.pt | 251.5 | 1.000 |
| REWARD_NOISE | AC_FULL | 0 | AC_FULL_REWARD_NOISE_seed0_step0040000_policy.pt | 283.3 | 1.000 |
| REWARD_NOISE | AC_FULL | 3 | AC_FULL_REWARD_NOISE_seed3_step0010000_policy.pt | 287.5 | 1.000 |
| REWARD_NOISE | AC_FULL | 9 | AC_FULL_REWARD_NOISE_seed9_step0040000_policy.pt | 287.5 | 1.000 |
| REWARD_NOISE | AC_FULL | 17 | AC_FULL_REWARD_NOISE_seed17_step0010000_policy.pt | 285.8 | 1.000 |
| REWARD_NOISE | AC_FULL | 42 | AC_FULL_REWARD_NOISE_seed42_step0020000_policy.pt | 289.5 | 1.000 |
| REWARD_NOISE | AC_LITE | 0 | AC_LITE_REWARD_NOISE_seed0_step0010000_policy.pt | 287.3 | 1.000 |
| REWARD_NOISE | AC_LITE | 3 | AC_LITE_REWARD_NOISE_seed3_step0010000_policy.pt | 286.3 | 1.000 |
| REWARD_NOISE | AC_LITE | 9 | AC_LITE_REWARD_NOISE_seed9_step0010000_policy.pt | 289.0 | 1.000 |
| REWARD_NOISE | AC_LITE | 17 | AC_LITE_REWARD_NOISE_seed17_step0030000_policy.pt | 285.0 | 1.000 |
| REWARD_NOISE | AC_LITE | 42 | AC_LITE_REWARD_NOISE_seed42_step0030000_policy.pt | 275.8 | 1.000 |
| REWARD_NOISE | BASELINE | 0 | BASELINE_REWARD_NOISE_seed0_step0010000.pt | 285.5 | 1.000 |
| REWARD_NOISE | BASELINE | 3 | BASELINE_REWARD_NOISE_seed3_step0010000.pt | 287.3 | 1.000 |
| REWARD_NOISE | BASELINE | 9 | BASELINE_REWARD_NOISE_seed9_step0010000.pt | 283.8 | 1.000 |
| REWARD_NOISE | BASELINE | 17 | BASELINE_REWARD_NOISE_seed17_step0010000.pt | 287.1 | 1.000 |
| REWARD_NOISE | BASELINE | 42 | BASELINE_REWARD_NOISE_seed42_step0010000.pt | 288.0 | 1.000 |

## Certainty AUROC diagnostics

- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.
- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\) (diagnostic).

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL | 0.979 | 0.356 |
| CLEAN | AC_LITE | 0.945 | 0.311 |
| OBS_NOISE | AC_FULL | 0.961 | 0.364 |
| OBS_NOISE | AC_LITE | 0.892 | 0.352 |
| REWARD_NOISE | AC_FULL | 0.981 | 0.352 |
| REWARD_NOISE | AC_LITE | 0.949 | 0.312 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

