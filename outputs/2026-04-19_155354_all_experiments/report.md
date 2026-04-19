# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-19_155354_all_experiments`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Notes on experimental modes

- **REWARD_NOISE**: false-negative successes also penalize the terminal rollout reward used by PPO/GAE, so the policy update sees the corruption.
- **OBS_NOISE**: adds Gaussian noise \(\sigma=0.1\) to observations at every step.

## Seed aggregation

Learning curves are computed **per seed** and then aggregated (mean ± std). This avoids interleaving seeds (each run resets `step` to 0).

## Summary table (mean ± std over 5 seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL | 254.7 ± 10.2 | 0.860 ± 0.042 | 289.4 | 1.000 |
| CLEAN | AC_LITE | 265.6 ± 16.5 | 0.900 ± 0.079 | 288.7 | 1.000 |
| CLEAN | BASELINE | 263.1 ± 6.6 | 0.910 ± 0.022 | 286.6 | 1.000 |
| OBS_NOISE | AC_FULL | 225.4 ± 25.0 | 0.750 ± 0.071 | 255.4 | 0.880 |
| OBS_NOISE | AC_LITE | 241.2 ± 28.1 | 0.810 ± 0.147 | 273.7 | 0.960 |
| OBS_NOISE | BASELINE | 243.8 ± 26.7 | 0.830 ± 0.120 | 272.5 | 0.950 |
| REWARD_NOISE | AC_FULL | 265.6 ± 13.9 | 0.660 ± 0.082 | 287.6 | 0.910 |
| REWARD_NOISE | AC_LITE | 256.7 ± 25.9 | 0.720 ± 0.045 | 288.0 | 0.910 |
| REWARD_NOISE | BASELINE | 263.4 ± 5.2 | 0.760 ± 0.055 | 286.3 | 0.900 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL | 0 | 241.2 | 0.800 |
| CLEAN | AC_FULL | 3 | 262.8 | 0.900 |
| CLEAN | AC_FULL | 9 | 260.9 | 0.850 |
| CLEAN | AC_FULL | 17 | 246.0 | 0.850 |
| CLEAN | AC_FULL | 42 | 262.3 | 0.900 |
| CLEAN | AC_LITE | 0 | 244.4 | 0.800 |
| CLEAN | AC_LITE | 3 | 273.9 | 0.950 |
| CLEAN | AC_LITE | 9 | 273.2 | 0.900 |
| CLEAN | AC_LITE | 17 | 252.5 | 0.850 |
| CLEAN | AC_LITE | 42 | 284.2 | 1.000 |
| CLEAN | BASELINE | 0 | 257.4 | 0.900 |
| CLEAN | BASELINE | 3 | 270.6 | 0.950 |
| CLEAN | BASELINE | 9 | 268.1 | 0.900 |
| CLEAN | BASELINE | 17 | 264.0 | 0.900 |
| CLEAN | BASELINE | 42 | 255.2 | 0.900 |
| OBS_NOISE | AC_FULL | 0 | 210.5 | 0.750 |
| OBS_NOISE | AC_FULL | 3 | 224.8 | 0.750 |
| OBS_NOISE | AC_FULL | 9 | 201.0 | 0.650 |
| OBS_NOISE | AC_FULL | 17 | 224.5 | 0.750 |
| OBS_NOISE | AC_FULL | 42 | 266.4 | 0.850 |
| OBS_NOISE | AC_LITE | 0 | 274.8 | 0.950 |
| OBS_NOISE | AC_LITE | 3 | 211.1 | 0.650 |
| OBS_NOISE | AC_LITE | 9 | 258.3 | 0.900 |
| OBS_NOISE | AC_LITE | 17 | 213.5 | 0.650 |
| OBS_NOISE | AC_LITE | 42 | 248.3 | 0.900 |
| OBS_NOISE | BASELINE | 0 | 231.2 | 0.750 |
| OBS_NOISE | BASELINE | 3 | 233.8 | 0.800 |
| OBS_NOISE | BASELINE | 9 | 287.0 | 1.000 |
| OBS_NOISE | BASELINE | 17 | 217.4 | 0.700 |
| OBS_NOISE | BASELINE | 42 | 249.6 | 0.900 |
| REWARD_NOISE | AC_FULL | 0 | 268.8 | 0.700 |
| REWARD_NOISE | AC_FULL | 3 | 251.3 | 0.600 |
| REWARD_NOISE | AC_FULL | 9 | 251.8 | 0.550 |
| REWARD_NOISE | AC_FULL | 17 | 272.6 | 0.700 |
| REWARD_NOISE | AC_FULL | 42 | 283.4 | 0.750 |
| REWARD_NOISE | AC_LITE | 0 | 263.2 | 0.750 |
| REWARD_NOISE | AC_LITE | 3 | 254.3 | 0.750 |
| REWARD_NOISE | AC_LITE | 9 | 281.3 | 0.750 |
| REWARD_NOISE | AC_LITE | 17 | 271.0 | 0.700 |
| REWARD_NOISE | AC_LITE | 42 | 213.8 | 0.650 |
| REWARD_NOISE | BASELINE | 0 | 258.2 | 0.800 |
| REWARD_NOISE | BASELINE | 3 | 272.1 | 0.800 |
| REWARD_NOISE | BASELINE | 9 | 263.7 | 0.700 |
| REWARD_NOISE | BASELINE | 17 | 262.3 | 0.700 |
| REWARD_NOISE | BASELINE | 42 | 260.9 | 0.800 |

## Best checkpoint by greedy held-out evaluation

Checkpoints are evaluated greedily on fixed held-out seeds. The final checkpoint is not assumed to be best.

| mode | method | seed | checkpoint | eval return | eval success |
|---|---|---:|---|---:|---:|
| CLEAN | AC_FULL | 0 | AC_FULL_CLEAN_seed0_step0010000_policy.pt | 290.2 | 1.000 |
| CLEAN | AC_FULL | 3 | AC_FULL_CLEAN_seed3_step0030000_policy.pt | 289.7 | 1.000 |
| CLEAN | AC_FULL | 9 | AC_FULL_CLEAN_seed9_step0050000_policy.pt | 290.0 | 1.000 |
| CLEAN | AC_FULL | 17 | AC_FULL_CLEAN_seed17_step0020000_policy.pt | 289.1 | 1.000 |
| CLEAN | AC_FULL | 42 | AC_FULL_CLEAN_seed42_step0010000_policy.pt | 288.6 | 1.000 |
| CLEAN | AC_LITE | 0 | AC_LITE_CLEAN_seed0_step0030000_policy.pt | 288.9 | 1.000 |
| CLEAN | AC_LITE | 3 | AC_LITE_CLEAN_seed3_step0060000_policy.pt | 288.6 | 1.000 |
| CLEAN | AC_LITE | 9 | AC_LITE_CLEAN_seed9_step0030000_policy.pt | 289.6 | 1.000 |
| CLEAN | AC_LITE | 17 | AC_LITE_CLEAN_seed17_step0010000_policy.pt | 291.3 | 1.000 |
| CLEAN | AC_LITE | 42 | AC_LITE_CLEAN_seed42_step0050000_policy.pt | 288.1 | 1.000 |
| CLEAN | BASELINE | 0 | BASELINE_CLEAN_seed0_step0060000.pt | 288.4 | 1.000 |
| CLEAN | BASELINE | 3 | BASELINE_CLEAN_seed3_step0060000.pt | 289.1 | 1.000 |
| CLEAN | BASELINE | 9 | BASELINE_CLEAN_seed9_step0010000.pt | 289.8 | 1.000 |
| CLEAN | BASELINE | 17 | BASELINE_CLEAN_seed17_step0010000.pt | 287.2 | 1.000 |
| CLEAN | BASELINE | 42 | BASELINE_CLEAN_seed42_step0030000.pt | 285.3 | 1.000 |
| OBS_NOISE | AC_FULL | 0 | AC_FULL_OBS_NOISE_seed0_step0040000_policy.pt | 239.5 | 0.733 |
| OBS_NOISE | AC_FULL | 3 | AC_FULL_OBS_NOISE_seed3_step0060000_policy.pt | 287.5 | 1.000 |
| OBS_NOISE | AC_FULL | 9 | AC_FULL_OBS_NOISE_seed9_final_policy.pt | 272.9 | 1.000 |
| OBS_NOISE | AC_FULL | 17 | AC_FULL_OBS_NOISE_seed17_step0040000_policy.pt | 247.2 | 0.800 |
| OBS_NOISE | AC_FULL | 42 | AC_FULL_OBS_NOISE_seed42_step0040000_policy.pt | 278.1 | 0.933 |
| OBS_NOISE | AC_LITE | 0 | AC_LITE_OBS_NOISE_seed0_final_policy.pt | 286.0 | 1.000 |
| OBS_NOISE | AC_LITE | 3 | AC_LITE_OBS_NOISE_seed3_step0030000_policy.pt | 278.0 | 0.933 |
| OBS_NOISE | AC_LITE | 9 | AC_LITE_OBS_NOISE_seed9_step0030000_policy.pt | 263.9 | 0.867 |
| OBS_NOISE | AC_LITE | 17 | AC_LITE_OBS_NOISE_seed17_step0030000_policy.pt | 253.1 | 0.800 |
| OBS_NOISE | AC_LITE | 42 | AC_LITE_OBS_NOISE_seed42_step0040000_policy.pt | 264.8 | 0.867 |
| OBS_NOISE | BASELINE | 0 | BASELINE_OBS_NOISE_seed0_step0010000.pt | 227.4 | 0.600 |
| OBS_NOISE | BASELINE | 3 | BASELINE_OBS_NOISE_seed3_step0050000.pt | 287.0 | 1.000 |
| OBS_NOISE | BASELINE | 9 | BASELINE_OBS_NOISE_seed9_step0060000.pt | 276.8 | 1.000 |
| OBS_NOISE | BASELINE | 17 | BASELINE_OBS_NOISE_seed17_step0060000.pt | 230.9 | 0.733 |
| OBS_NOISE | BASELINE | 42 | BASELINE_OBS_NOISE_seed42_step0050000.pt | 287.2 | 1.000 |
| REWARD_NOISE | AC_FULL | 0 | AC_FULL_REWARD_NOISE_seed0_step0030000_policy.pt | 286.5 | 1.000 |
| REWARD_NOISE | AC_FULL | 3 | AC_FULL_REWARD_NOISE_seed3_final_policy.pt | 290.1 | 1.000 |
| REWARD_NOISE | AC_FULL | 9 | AC_FULL_REWARD_NOISE_seed9_step0010000_policy.pt | 289.7 | 1.000 |
| REWARD_NOISE | AC_FULL | 17 | AC_FULL_REWARD_NOISE_seed17_step0010000_policy.pt | 288.8 | 1.000 |
| REWARD_NOISE | AC_FULL | 42 | AC_FULL_REWARD_NOISE_seed42_step0050000_policy.pt | 287.5 | 1.000 |
| REWARD_NOISE | AC_LITE | 0 | AC_LITE_REWARD_NOISE_seed0_step0050000_policy.pt | 288.1 | 1.000 |
| REWARD_NOISE | AC_LITE | 3 | AC_LITE_REWARD_NOISE_seed3_step0020000_policy.pt | 289.7 | 1.000 |
| REWARD_NOISE | AC_LITE | 9 | AC_LITE_REWARD_NOISE_seed9_step0040000_policy.pt | 289.5 | 1.000 |
| REWARD_NOISE | AC_LITE | 17 | AC_LITE_REWARD_NOISE_seed17_step0050000_policy.pt | 288.7 | 1.000 |
| REWARD_NOISE | AC_LITE | 42 | AC_LITE_REWARD_NOISE_seed42_step0050000_policy.pt | 290.6 | 1.000 |
| REWARD_NOISE | BASELINE | 0 | BASELINE_REWARD_NOISE_seed0_step0020000.pt | 286.9 | 1.000 |
| REWARD_NOISE | BASELINE | 3 | BASELINE_REWARD_NOISE_seed3_step0050000.pt | 289.9 | 1.000 |
| REWARD_NOISE | BASELINE | 9 | BASELINE_REWARD_NOISE_seed9_step0020000.pt | 288.7 | 1.000 |
| REWARD_NOISE | BASELINE | 17 | BASELINE_REWARD_NOISE_seed17_step0010000.pt | 291.0 | 1.000 |
| REWARD_NOISE | BASELINE | 42 | BASELINE_REWARD_NOISE_seed42_step0060000.pt | 288.9 | 1.000 |

## Certainty AUROC diagnostics

- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.
- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\) (diagnostic).

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL | 0.514 | 0.322 |
| CLEAN | AC_LITE | 0.596 | 0.222 |
| OBS_NOISE | AC_FULL | 0.769 | 0.354 |
| OBS_NOISE | AC_LITE | 0.796 | 0.299 |
| REWARD_NOISE | AC_FULL | 0.534 | 0.270 |
| REWARD_NOISE | AC_LITE | 0.540 | 0.209 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

