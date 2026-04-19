# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-18_192448_all_experiments`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Notes on experimental modes

- **REWARD_NOISE**: false-negative successes also penalize the terminal rollout reward used by PPO/GAE, so the policy update sees the corruption.
- **OBS_NOISE**: adds Gaussian noise \(\sigma=0.1\) to observations at every step.

## Seed aggregation

Learning curves are computed **per seed** and then aggregated (mean ± std). This avoids interleaving seeds (each run resets `step` to 0).

## Summary table (mean ± std over 5 seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL | 268.1 ± 17.2 | 0.910 ± 0.082 | 297.2 | 1.000 |
| CLEAN | AC_LITE | 274.5 ± 2.7 | 0.950 ± 0.000 | 295.4 | 1.000 |
| CLEAN | BASELINE | 266.3 ± 14.7 | 0.910 ± 0.065 | 294.4 | 1.000 |
| OBS_NOISE | AC_FULL | 71.9 ± 68.7 | 0.120 ± 0.130 | 273.4 | 1.000 |
| OBS_NOISE | AC_LITE | -17.0 ± 46.2 | 0.000 ± 0.000 | 274.1 | 1.000 |
| OBS_NOISE | BASELINE | -7.7 ± 68.0 | 0.050 ± 0.050 | 278.2 | 1.000 |
| REWARD_NOISE | AC_FULL | 265.6 ± 22.1 | 0.750 ± 0.071 | 296.0 | 1.000 |
| REWARD_NOISE | AC_LITE | 266.3 ± 13.6 | 0.730 ± 0.076 | 294.5 | 0.990 |
| REWARD_NOISE | BASELINE | 274.2 ± 10.9 | 0.770 ± 0.104 | 296.1 | 1.000 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL | 0 | 277.8 | 0.950 |
| CLEAN | AC_FULL | 3 | 277.0 | 0.950 |
| CLEAN | AC_FULL | 9 | 244.3 | 0.800 |
| CLEAN | AC_FULL | 17 | 256.0 | 0.850 |
| CLEAN | AC_FULL | 42 | 285.2 | 1.000 |
| CLEAN | AC_LITE | 0 | 278.1 | 0.950 |
| CLEAN | AC_LITE | 3 | 271.3 | 0.950 |
| CLEAN | AC_LITE | 9 | 275.9 | 0.950 |
| CLEAN | AC_LITE | 17 | 274.3 | 0.950 |
| CLEAN | AC_LITE | 42 | 272.9 | 0.950 |
| CLEAN | BASELINE | 0 | 242.0 | 0.800 |
| CLEAN | BASELINE | 3 | 264.9 | 0.900 |
| CLEAN | BASELINE | 9 | 276.7 | 0.950 |
| CLEAN | BASELINE | 17 | 269.2 | 0.950 |
| CLEAN | BASELINE | 42 | 278.9 | 0.950 |
| OBS_NOISE | AC_FULL | 0 | -9.0 | 0.050 |
| OBS_NOISE | AC_FULL | 3 | 163.9 | 0.350 |
| OBS_NOISE | AC_FULL | 9 | 96.2 | 0.050 |
| OBS_NOISE | AC_FULL | 17 | 90.8 | 0.100 |
| OBS_NOISE | AC_FULL | 42 | 17.8 | 0.050 |
| OBS_NOISE | AC_LITE | 0 | -43.4 | 0.000 |
| OBS_NOISE | AC_LITE | 3 | -78.3 | 0.000 |
| OBS_NOISE | AC_LITE | 9 | -11.1 | 0.000 |
| OBS_NOISE | AC_LITE | 17 | 42.8 | 0.000 |
| OBS_NOISE | AC_LITE | 42 | 4.8 | 0.000 |
| OBS_NOISE | BASELINE | 0 | -12.5 | 0.100 |
| OBS_NOISE | BASELINE | 3 | -84.7 | 0.000 |
| OBS_NOISE | BASELINE | 9 | 45.1 | 0.100 |
| OBS_NOISE | BASELINE | 17 | 75.0 | 0.050 |
| OBS_NOISE | BASELINE | 42 | -61.5 | 0.000 |
| REWARD_NOISE | AC_FULL | 0 | 231.6 | 0.650 |
| REWARD_NOISE | AC_FULL | 3 | 265.5 | 0.800 |
| REWARD_NOISE | AC_FULL | 9 | 288.9 | 0.700 |
| REWARD_NOISE | AC_FULL | 17 | 260.9 | 0.800 |
| REWARD_NOISE | AC_FULL | 42 | 280.9 | 0.800 |
| REWARD_NOISE | AC_LITE | 0 | 268.9 | 0.800 |
| REWARD_NOISE | AC_LITE | 3 | 271.8 | 0.750 |
| REWARD_NOISE | AC_LITE | 9 | 242.3 | 0.600 |
| REWARD_NOISE | AC_LITE | 17 | 276.0 | 0.750 |
| REWARD_NOISE | AC_LITE | 42 | 272.3 | 0.750 |
| REWARD_NOISE | BASELINE | 0 | 278.8 | 0.850 |
| REWARD_NOISE | BASELINE | 3 | 262.3 | 0.650 |
| REWARD_NOISE | BASELINE | 9 | 271.3 | 0.750 |
| REWARD_NOISE | BASELINE | 17 | 290.5 | 0.900 |
| REWARD_NOISE | BASELINE | 42 | 268.0 | 0.700 |

## Certainty AUROC diagnostics

- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.
- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\) (diagnostic).

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL | 0.850 | 0.221 |
| CLEAN | AC_LITE | 0.704 | 0.182 |
| OBS_NOISE | AC_FULL | 0.857 | 0.386 |
| OBS_NOISE | AC_LITE | 0.728 | 0.337 |
| REWARD_NOISE | AC_FULL | 0.615 | 0.252 |
| REWARD_NOISE | AC_LITE | 0.591 | 0.238 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

