# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-18_175130_all_experiments`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Notes on experimental modes

- **REWARD_NOISE**: false-negative successes also penalize the terminal rollout reward used by PPO/GAE, so the policy update sees the corruption.
- **OBS_NOISE**: adds Gaussian noise \(\sigma=0.1\) to observations at every step.

## Seed aggregation

Learning curves are computed **per seed** and then aggregated (mean ± std). This avoids interleaving seeds (each run resets `step` to 0).

## Summary table (mean ± std over 5 seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL | 266.1 ± 13.3 | 0.933 ± 0.058 | 278.1 | 0.967 |
| CLEAN | AC_LITE | 258.1 ± 17.7 | 0.883 ± 0.115 | 282.8 | 0.983 |
| CLEAN | BASELINE | 268.2 ± 16.0 | 0.917 ± 0.076 | 287.9 | 1.000 |
| OBS_NOISE | AC_FULL | 173.0 ± 21.3 | 0.533 ± 0.076 | 183.2 | 0.583 |
| OBS_NOISE | AC_LITE | 163.7 ± 36.5 | 0.500 ± 0.180 | 192.9 | 0.633 |
| OBS_NOISE | BASELINE | 154.9 ± 19.9 | 0.467 ± 0.104 | 181.9 | 0.567 |
| REWARD_NOISE | AC_FULL | 252.6 ± 38.6 | 0.681 ± 0.211 | 258.7 | 0.748 |
| REWARD_NOISE | AC_LITE | 239.7 ± 43.6 | 0.663 ± 0.243 | 251.5 | 0.730 |
| REWARD_NOISE | BASELINE | 247.3 ± 33.3 | 0.650 ± 0.132 | 281.1 | 0.850 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL | 0 | 254.4 | 0.900 |
| CLEAN | AC_FULL | 17 | 263.4 | 0.900 |
| CLEAN | AC_FULL | 42 | 280.6 | 1.000 |
| CLEAN | AC_LITE | 0 | 265.5 | 0.950 |
| CLEAN | AC_LITE | 17 | 270.8 | 0.950 |
| CLEAN | AC_LITE | 42 | 237.8 | 0.750 |
| CLEAN | BASELINE | 0 | 285.7 | 1.000 |
| CLEAN | BASELINE | 17 | 264.8 | 0.900 |
| CLEAN | BASELINE | 42 | 254.3 | 0.850 |
| OBS_NOISE | AC_FULL | 0 | 176.3 | 0.550 |
| OBS_NOISE | AC_FULL | 17 | 192.4 | 0.600 |
| OBS_NOISE | AC_FULL | 42 | 150.2 | 0.450 |
| OBS_NOISE | AC_LITE | 0 | 132.9 | 0.350 |
| OBS_NOISE | AC_LITE | 17 | 154.1 | 0.450 |
| OBS_NOISE | AC_LITE | 42 | 204.0 | 0.700 |
| OBS_NOISE | BASELINE | 0 | 133.0 | 0.350 |
| OBS_NOISE | BASELINE | 17 | 172.0 | 0.550 |
| OBS_NOISE | BASELINE | 42 | 159.8 | 0.500 |
| REWARD_NOISE | AC_FULL | 0 | 279.2 | 0.850 |
| REWARD_NOISE | AC_FULL | 17 | 270.2 | 0.750 |
| REWARD_NOISE | AC_FULL | 42 | 208.3 | 0.444 |
| REWARD_NOISE | AC_LITE | 0 | 250.4 | 0.750 |
| REWARD_NOISE | AC_LITE | 17 | 277.0 | 0.850 |
| REWARD_NOISE | AC_LITE | 42 | 191.9 | 0.389 |
| REWARD_NOISE | BASELINE | 0 | 257.8 | 0.750 |
| REWARD_NOISE | BASELINE | 17 | 210.0 | 0.500 |
| REWARD_NOISE | BASELINE | 42 | 274.1 | 0.700 |

## Certainty AUROC diagnostics

- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.
- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\) (diagnostic).

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL | 0.667 | 0.232 |
| CLEAN | AC_LITE | 0.575 | 0.270 |
| OBS_NOISE | AC_FULL | 0.661 | 0.399 |
| OBS_NOISE | AC_LITE | 0.729 | 0.371 |
| REWARD_NOISE | AC_FULL | 0.575 | 0.370 |
| REWARD_NOISE | AC_LITE | 0.481 | 0.359 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_clean_vs_noisy_return_comparison.png`

