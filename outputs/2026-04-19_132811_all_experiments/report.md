# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-19_132811_all_experiments`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Notes on experimental modes

- **REWARD_NOISE**: false-negative successes also penalize the terminal rollout reward used by PPO/GAE, so the policy update sees the corruption.
- **OBS_NOISE**: adds Gaussian noise \(\sigma=0.1\) to observations at every step.

## Seed aggregation

Learning curves are computed **per seed** and then aggregated (mean ± std). This avoids interleaving seeds (each run resets `step` to 0).

## Summary table (mean ± std over 5 seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL | 261.5 ± 13.8 | 0.890 ± 0.065 | 285.9 | 1.000 |
| CLEAN | AC_LITE | 272.9 ± 8.8 | 0.930 ± 0.045 | 284.8 | 1.000 |
| CLEAN | BASELINE | 252.4 ± 21.3 | 0.860 ± 0.108 | 285.7 | 1.000 |
| OBS_NOISE | AC_FULL | 191.2 ± 17.5 | 0.630 ± 0.057 | 234.0 | 0.790 |
| OBS_NOISE | AC_LITE | 233.7 ± 14.3 | 0.790 ± 0.065 | 256.1 | 0.880 |
| OBS_NOISE | BASELINE | 216.4 ± 40.0 | 0.720 ± 0.160 | 246.3 | 0.850 |
| REWARD_NOISE | AC_FULL | 250.1 ± 25.9 | 0.680 ± 0.135 | 285.1 | 0.920 |
| REWARD_NOISE | AC_LITE | 258.1 ± 15.5 | 0.660 ± 0.055 | 288.6 | 0.920 |
| REWARD_NOISE | BASELINE | 248.6 ± 22.0 | 0.650 ± 0.127 | 289.1 | 0.890 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL | 0 | 270.3 | 0.900 |
| CLEAN | AC_FULL | 3 | 277.9 | 0.950 |
| CLEAN | AC_FULL | 9 | 264.8 | 0.950 |
| CLEAN | AC_FULL | 17 | 246.1 | 0.850 |
| CLEAN | AC_FULL | 42 | 248.4 | 0.800 |
| CLEAN | AC_LITE | 0 | 273.3 | 0.900 |
| CLEAN | AC_LITE | 3 | 273.9 | 0.950 |
| CLEAN | AC_LITE | 9 | 284.6 | 1.000 |
| CLEAN | AC_LITE | 17 | 272.7 | 0.900 |
| CLEAN | AC_LITE | 42 | 259.9 | 0.900 |
| CLEAN | BASELINE | 0 | 228.9 | 0.750 |
| CLEAN | BASELINE | 3 | 258.0 | 0.900 |
| CLEAN | BASELINE | 9 | 260.8 | 0.900 |
| CLEAN | BASELINE | 17 | 233.4 | 0.750 |
| CLEAN | BASELINE | 42 | 280.6 | 1.000 |
| OBS_NOISE | AC_FULL | 0 | 188.4 | 0.650 |
| OBS_NOISE | AC_FULL | 3 | 187.2 | 0.600 |
| OBS_NOISE | AC_FULL | 9 | 166.3 | 0.550 |
| OBS_NOISE | AC_FULL | 17 | 213.5 | 0.700 |
| OBS_NOISE | AC_FULL | 42 | 200.6 | 0.650 |
| OBS_NOISE | AC_LITE | 0 | 239.3 | 0.800 |
| OBS_NOISE | AC_LITE | 3 | 221.6 | 0.750 |
| OBS_NOISE | AC_LITE | 9 | 225.7 | 0.750 |
| OBS_NOISE | AC_LITE | 17 | 256.4 | 0.900 |
| OBS_NOISE | AC_LITE | 42 | 225.4 | 0.750 |
| OBS_NOISE | BASELINE | 0 | 202.4 | 0.650 |
| OBS_NOISE | BASELINE | 3 | 243.8 | 0.850 |
| OBS_NOISE | BASELINE | 9 | 258.8 | 0.900 |
| OBS_NOISE | BASELINE | 17 | 156.2 | 0.500 |
| OBS_NOISE | BASELINE | 42 | 220.8 | 0.700 |
| REWARD_NOISE | AC_FULL | 0 | 278.4 | 0.850 |
| REWARD_NOISE | AC_FULL | 3 | 232.0 | 0.600 |
| REWARD_NOISE | AC_FULL | 9 | 262.8 | 0.550 |
| REWARD_NOISE | AC_FULL | 17 | 262.7 | 0.800 |
| REWARD_NOISE | AC_FULL | 42 | 214.9 | 0.600 |
| REWARD_NOISE | AC_LITE | 0 | 255.1 | 0.650 |
| REWARD_NOISE | AC_LITE | 3 | 254.3 | 0.750 |
| REWARD_NOISE | AC_LITE | 9 | 282.8 | 0.600 |
| REWARD_NOISE | AC_LITE | 17 | 240.0 | 0.650 |
| REWARD_NOISE | AC_LITE | 42 | 258.2 | 0.650 |
| REWARD_NOISE | BASELINE | 0 | 223.7 | 0.500 |
| REWARD_NOISE | BASELINE | 3 | 268.2 | 0.750 |
| REWARD_NOISE | BASELINE | 9 | 240.1 | 0.550 |
| REWARD_NOISE | BASELINE | 17 | 235.9 | 0.650 |
| REWARD_NOISE | BASELINE | 42 | 274.9 | 0.800 |

## Best checkpoint by greedy held-out evaluation

Checkpoints are evaluated greedily on fixed held-out seeds. The final checkpoint is not assumed to be best.

| mode | method | seed | checkpoint | eval return | eval success |
|---|---|---:|---|---:|---:|
| CLEAN | AC_FULL | 0 | AC_FULL_CLEAN_seed0_step0030000_policy.pt | 286.9 | 1.000 |
| CLEAN | AC_FULL | 3 | AC_FULL_CLEAN_seed3_step0030000_policy.pt | 287.6 | 1.000 |
| CLEAN | AC_FULL | 9 | AC_FULL_CLEAN_seed9_step0010000_policy.pt | 290.3 | 1.000 |
| CLEAN | AC_FULL | 17 | AC_FULL_CLEAN_seed17_step0010000_policy.pt | 288.8 | 1.000 |
| CLEAN | AC_FULL | 42 | AC_FULL_CLEAN_seed42_step0010000_policy.pt | 288.8 | 1.000 |
| CLEAN | AC_LITE | 0 | AC_LITE_CLEAN_seed0_step0030000_policy.pt | 287.9 | 1.000 |
| CLEAN | AC_LITE | 3 | AC_LITE_CLEAN_seed3_step0040000_policy.pt | 289.4 | 1.000 |
| CLEAN | AC_LITE | 9 | AC_LITE_CLEAN_seed9_step0040000_policy.pt | 287.3 | 1.000 |
| CLEAN | AC_LITE | 17 | AC_LITE_CLEAN_seed17_step0010000_policy.pt | 287.5 | 1.000 |
| CLEAN | AC_LITE | 42 | AC_LITE_CLEAN_seed42_step0030000_policy.pt | 287.8 | 1.000 |
| CLEAN | BASELINE | 0 | BASELINE_CLEAN_seed0_final.pt | 287.6 | 1.000 |
| CLEAN | BASELINE | 3 | BASELINE_CLEAN_seed3_step0040000.pt | 288.1 | 1.000 |
| CLEAN | BASELINE | 9 | BASELINE_CLEAN_seed9_final.pt | 288.7 | 1.000 |
| CLEAN | BASELINE | 17 | BASELINE_CLEAN_seed17_step0010000.pt | 287.2 | 1.000 |
| CLEAN | BASELINE | 42 | BASELINE_CLEAN_seed42_step0040000.pt | 285.9 | 1.000 |
| OBS_NOISE | AC_FULL | 0 | AC_FULL_OBS_NOISE_seed0_final_policy.pt | 251.5 | 0.800 |
| OBS_NOISE | AC_FULL | 3 | AC_FULL_OBS_NOISE_seed3_step0030000_policy.pt | 267.2 | 0.867 |
| OBS_NOISE | AC_FULL | 9 | AC_FULL_OBS_NOISE_seed9_step0020000_policy.pt | 265.3 | 0.867 |
| OBS_NOISE | AC_FULL | 17 | AC_FULL_OBS_NOISE_seed17_step0010000_policy.pt | 238.7 | 0.733 |
| OBS_NOISE | AC_FULL | 42 | AC_FULL_OBS_NOISE_seed42_step0040000_policy.pt | 259.1 | 0.800 |
| OBS_NOISE | AC_LITE | 0 | AC_LITE_OBS_NOISE_seed0_step0030000_policy.pt | 245.0 | 0.800 |
| OBS_NOISE | AC_LITE | 3 | AC_LITE_OBS_NOISE_seed3_step0030000_policy.pt | 278.0 | 0.933 |
| OBS_NOISE | AC_LITE | 9 | AC_LITE_OBS_NOISE_seed9_step0030000_policy.pt | 263.9 | 0.867 |
| OBS_NOISE | AC_LITE | 17 | AC_LITE_OBS_NOISE_seed17_final_policy.pt | 268.4 | 0.867 |
| OBS_NOISE | AC_LITE | 42 | AC_LITE_OBS_NOISE_seed42_final_policy.pt | 290.2 | 1.000 |
| OBS_NOISE | BASELINE | 0 | BASELINE_OBS_NOISE_seed0_step0020000.pt | 219.4 | 0.600 |
| OBS_NOISE | BASELINE | 3 | BASELINE_OBS_NOISE_seed3_final.pt | 262.8 | 0.867 |
| OBS_NOISE | BASELINE | 9 | BASELINE_OBS_NOISE_seed9_final.pt | 286.6 | 1.000 |
| OBS_NOISE | BASELINE | 17 | BASELINE_OBS_NOISE_seed17_final.pt | 229.7 | 0.667 |
| OBS_NOISE | BASELINE | 42 | BASELINE_OBS_NOISE_seed42_step0010000.pt | 245.4 | 0.733 |
| REWARD_NOISE | AC_FULL | 0 | AC_FULL_REWARD_NOISE_seed0_step0010000_policy.pt | 287.4 | 1.000 |
| REWARD_NOISE | AC_FULL | 3 | AC_FULL_REWARD_NOISE_seed3_step0020000_policy.pt | 290.4 | 1.000 |
| REWARD_NOISE | AC_FULL | 9 | AC_FULL_REWARD_NOISE_seed9_step0010000_policy.pt | 289.7 | 1.000 |
| REWARD_NOISE | AC_FULL | 17 | AC_FULL_REWARD_NOISE_seed17_step0010000_policy.pt | 288.8 | 1.000 |
| REWARD_NOISE | AC_FULL | 42 | AC_FULL_REWARD_NOISE_seed42_step0010000_policy.pt | 286.7 | 1.000 |
| REWARD_NOISE | AC_LITE | 0 | AC_LITE_REWARD_NOISE_seed0_step0010000_policy.pt | 286.9 | 1.000 |
| REWARD_NOISE | AC_LITE | 3 | AC_LITE_REWARD_NOISE_seed3_step0010000_policy.pt | 287.2 | 1.000 |
| REWARD_NOISE | AC_LITE | 9 | AC_LITE_REWARD_NOISE_seed9_step0040000_policy.pt | 289.5 | 1.000 |
| REWARD_NOISE | AC_LITE | 17 | AC_LITE_REWARD_NOISE_seed17_step0030000_policy.pt | 290.2 | 1.000 |
| REWARD_NOISE | AC_LITE | 42 | AC_LITE_REWARD_NOISE_seed42_step0030000_policy.pt | 289.6 | 1.000 |
| REWARD_NOISE | BASELINE | 0 | BASELINE_REWARD_NOISE_seed0_step0040000.pt | 288.5 | 1.000 |
| REWARD_NOISE | BASELINE | 3 | BASELINE_REWARD_NOISE_seed3_step0010000.pt | 286.5 | 1.000 |
| REWARD_NOISE | BASELINE | 9 | BASELINE_REWARD_NOISE_seed9_final.pt | 289.5 | 1.000 |
| REWARD_NOISE | BASELINE | 17 | BASELINE_REWARD_NOISE_seed17_step0010000.pt | 291.0 | 1.000 |
| REWARD_NOISE | BASELINE | 42 | BASELINE_REWARD_NOISE_seed42_step0030000.pt | 289.0 | 1.000 |

## Certainty AUROC diagnostics

- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.
- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\) (diagnostic).

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL | 0.555 | 0.294 |
| CLEAN | AC_LITE | 0.561 | 0.234 |
| OBS_NOISE | AC_FULL | 0.733 | 0.379 |
| OBS_NOISE | AC_LITE | 0.777 | 0.334 |
| REWARD_NOISE | AC_FULL | 0.513 | 0.304 |
| REWARD_NOISE | AC_LITE | 0.528 | 0.214 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

