# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `outputs\2026-04-18_180916_all_experiments`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Notes on experimental modes

- **REWARD_NOISE**: false-negative successes also penalize the terminal rollout reward used by PPO/GAE, so the policy update sees the corruption.
- **OBS_NOISE**: adds Gaussian noise \(\sigma=0.1\) to observations at every step.

## Seed aggregation

Learning curves are computed **per seed** and then aggregated (mean ± std). This avoids interleaving seeds (each run resets `step` to 0).

## Summary table (mean ± std over 5 seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL | 268.2 ± 13.9 | 0.917 ± 0.076 | 287.6 | 1.000 |
| CLEAN | AC_LITE | 261.5 ± 21.4 | 0.883 ± 0.104 | 288.7 | 1.000 |
| CLEAN | BASELINE | 277.7 ± 6.2 | 0.967 ± 0.029 | 288.3 | 1.000 |
| OBS_NOISE | AC_FULL | 239.2 ± 19.1 | 0.867 ± 0.076 | 273.9 | 1.000 |
| OBS_NOISE | AC_LITE | 239.6 ± 14.8 | 0.850 ± 0.050 | 273.8 | 0.983 |
| OBS_NOISE | BASELINE | 227.1 ± 35.1 | 0.817 ± 0.126 | 273.2 | 0.983 |
| REWARD_NOISE | AC_FULL | 270.1 ± 18.5 | 0.750 ± 0.100 | 290.1 | 0.917 |
| REWARD_NOISE | AC_LITE | 257.2 ± 23.1 | 0.667 ± 0.076 | 284.6 | 0.917 |
| REWARD_NOISE | BASELINE | 265.5 ± 12.3 | 0.700 ± 0.100 | 289.9 | 0.933 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL | 0 | 256.2 | 0.850 |
| CLEAN | AC_FULL | 17 | 283.4 | 1.000 |
| CLEAN | AC_FULL | 42 | 264.9 | 0.900 |
| CLEAN | AC_LITE | 0 | 285.1 | 1.000 |
| CLEAN | AC_LITE | 17 | 243.4 | 0.800 |
| CLEAN | AC_LITE | 42 | 256.1 | 0.850 |
| CLEAN | BASELINE | 0 | 284.8 | 1.000 |
| CLEAN | BASELINE | 17 | 275.2 | 0.950 |
| CLEAN | BASELINE | 42 | 273.2 | 0.950 |
| OBS_NOISE | AC_FULL | 0 | 246.6 | 0.850 |
| OBS_NOISE | AC_FULL | 17 | 253.5 | 0.950 |
| OBS_NOISE | AC_FULL | 42 | 217.5 | 0.800 |
| OBS_NOISE | AC_LITE | 0 | 241.0 | 0.900 |
| OBS_NOISE | AC_LITE | 17 | 224.0 | 0.800 |
| OBS_NOISE | AC_LITE | 42 | 253.6 | 0.850 |
| OBS_NOISE | BASELINE | 0 | 267.5 | 0.950 |
| OBS_NOISE | BASELINE | 17 | 205.1 | 0.800 |
| OBS_NOISE | BASELINE | 42 | 208.6 | 0.700 |
| REWARD_NOISE | AC_FULL | 0 | 248.8 | 0.850 |
| REWARD_NOISE | AC_FULL | 17 | 281.4 | 0.650 |
| REWARD_NOISE | AC_FULL | 42 | 280.2 | 0.750 |
| REWARD_NOISE | AC_LITE | 0 | 272.9 | 0.600 |
| REWARD_NOISE | AC_LITE | 17 | 268.0 | 0.750 |
| REWARD_NOISE | AC_LITE | 42 | 230.7 | 0.650 |
| REWARD_NOISE | BASELINE | 0 | 256.7 | 0.700 |
| REWARD_NOISE | BASELINE | 17 | 260.2 | 0.600 |
| REWARD_NOISE | BASELINE | 42 | 279.6 | 0.800 |

## Certainty AUROC diagnostics

- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.
- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\) (diagnostic).

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL | 0.824 | 0.240 |
| CLEAN | AC_LITE | 0.613 | 0.209 |
| OBS_NOISE | AC_FULL | 0.868 | 0.312 |
| OBS_NOISE | AC_LITE | 0.805 | 0.266 |
| REWARD_NOISE | AC_FULL | 0.609 | 0.295 |
| REWARD_NOISE | AC_LITE | 0.585 | 0.230 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

