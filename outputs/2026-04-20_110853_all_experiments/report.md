# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-20_110853_all_experiments`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Notes on experimental modes

- **Reward semantics**: PPO/GAE uses sparse terminal binary reward only (`0` before termination, terminal `policy_success` at episode end); dense LunarLander return is logged for diagnostics only.
- **REWARD_NOISE**: false-negative successes set terminal `policy_success` to `0`, so the sparse policy update sees the corrupted outcome directly.
- **OBS_NOISE**: adds Gaussian noise \(\sigma=0.1\) to observations at every step.

## Seed aggregation

Learning curves are computed **per seed** and then aggregated (mean ± std). This avoids interleaving seeds (each run resets `step` to 0).

## Summary table (mean ± std over 5 seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL | 86.8 ± 54.6 | 0.180 ± 0.125 | 290.9 | 1.000 |
| CLEAN | AC_LITE | 149.8 ± 69.2 | 0.460 ± 0.292 | 289.2 | 1.000 |
| CLEAN | BASELINE | 131.3 ± 63.8 | 0.380 ± 0.251 | 291.1 | 1.000 |
| OBS_NOISE | AC_FULL | 85.1 ± 73.5 | 0.190 ± 0.275 | 226.7 | 0.780 |
| OBS_NOISE | AC_LITE | 131.1 ± 100.8 | 0.420 ± 0.391 | 236.2 | 0.790 |
| OBS_NOISE | BASELINE | 175.3 ± 83.6 | 0.560 ± 0.363 | 254.4 | 0.870 |
| REWARD_NOISE | AC_FULL | 41.9 ± 39.4 | 0.080 ± 0.057 | 290.2 | 0.920 |
| REWARD_NOISE | AC_LITE | 89.5 ± 46.1 | 0.240 ± 0.124 | 290.8 | 0.950 |
| REWARD_NOISE | BASELINE | 68.2 ± 70.2 | 0.180 ± 0.175 | 290.4 | 0.950 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL | 0 | -5.4 | 0.000 |
| CLEAN | AC_FULL | 3 | 139.6 | 0.100 |
| CLEAN | AC_FULL | 9 | 92.8 | 0.250 |
| CLEAN | AC_FULL | 17 | 109.1 | 0.300 |
| CLEAN | AC_FULL | 42 | 97.8 | 0.250 |
| CLEAN | AC_LITE | 0 | 99.1 | 0.250 |
| CLEAN | AC_LITE | 3 | 194.8 | 0.650 |
| CLEAN | AC_LITE | 9 | 189.8 | 0.650 |
| CLEAN | AC_LITE | 17 | 54.1 | 0.050 |
| CLEAN | AC_LITE | 42 | 211.5 | 0.700 |
| CLEAN | BASELINE | 0 | 65.4 | 0.150 |
| CLEAN | BASELINE | 3 | 154.4 | 0.500 |
| CLEAN | BASELINE | 9 | 69.6 | 0.100 |
| CLEAN | BASELINE | 17 | 151.3 | 0.450 |
| CLEAN | BASELINE | 42 | 216.0 | 0.700 |
| OBS_NOISE | AC_FULL | 0 | 23.2 | 0.000 |
| OBS_NOISE | AC_FULL | 3 | 129.9 | 0.350 |
| OBS_NOISE | AC_FULL | 9 | 189.0 | 0.600 |
| OBS_NOISE | AC_FULL | 17 | 17.2 | 0.000 |
| OBS_NOISE | AC_FULL | 42 | 66.2 | 0.000 |
| OBS_NOISE | AC_LITE | 0 | 60.4 | 0.200 |
| OBS_NOISE | AC_LITE | 3 | 194.9 | 0.650 |
| OBS_NOISE | AC_LITE | 9 | 54.6 | 0.100 |
| OBS_NOISE | AC_LITE | 17 | 278.3 | 1.000 |
| OBS_NOISE | AC_LITE | 42 | 67.3 | 0.150 |
| OBS_NOISE | BASELINE | 0 | 175.2 | 0.550 |
| OBS_NOISE | BASELINE | 3 | 45.4 | 0.000 |
| OBS_NOISE | BASELINE | 9 | 162.2 | 0.500 |
| OBS_NOISE | BASELINE | 17 | 265.4 | 0.950 |
| OBS_NOISE | BASELINE | 42 | 228.1 | 0.800 |
| REWARD_NOISE | AC_FULL | 0 | 25.3 | 0.100 |
| REWARD_NOISE | AC_FULL | 3 | -19.6 | 0.000 |
| REWARD_NOISE | AC_FULL | 9 | 63.5 | 0.050 |
| REWARD_NOISE | AC_FULL | 17 | 63.6 | 0.100 |
| REWARD_NOISE | AC_FULL | 42 | 76.8 | 0.150 |
| REWARD_NOISE | AC_LITE | 0 | 144.7 | 0.350 |
| REWARD_NOISE | AC_LITE | 3 | 72.8 | 0.250 |
| REWARD_NOISE | AC_LITE | 9 | 111.3 | 0.350 |
| REWARD_NOISE | AC_LITE | 17 | 21.4 | 0.050 |
| REWARD_NOISE | AC_LITE | 42 | 97.5 | 0.200 |
| REWARD_NOISE | BASELINE | 0 | -19.2 | 0.000 |
| REWARD_NOISE | BASELINE | 3 | 96.2 | 0.200 |
| REWARD_NOISE | BASELINE | 9 | 18.4 | 0.050 |
| REWARD_NOISE | BASELINE | 17 | 160.6 | 0.450 |
| REWARD_NOISE | BASELINE | 42 | 85.0 | 0.200 |

## Best checkpoint by greedy held-out evaluation

Checkpoints are evaluated greedily on fixed held-out seeds. The final checkpoint is not assumed to be best.

| mode | method | seed | checkpoint | eval return | eval success |
|---|---|---:|---|---:|---:|
| CLEAN | AC_FULL | 0 | AC_FULL_CLEAN_seed0_step0120000_policy.pt | 288.5 | 1.000 |
| CLEAN | AC_FULL | 3 | AC_FULL_CLEAN_seed3_step0020000_policy.pt | 289.5 | 1.000 |
| CLEAN | AC_FULL | 9 | AC_FULL_CLEAN_seed9_step0030000_policy.pt | 289.4 | 1.000 |
| CLEAN | AC_FULL | 17 | AC_FULL_CLEAN_seed17_step0120000_policy.pt | 289.4 | 1.000 |
| CLEAN | AC_FULL | 42 | AC_FULL_CLEAN_seed42_step0090000_policy.pt | 289.3 | 1.000 |
| CLEAN | AC_LITE | 0 | AC_LITE_CLEAN_seed0_step0020000_policy.pt | 286.1 | 1.000 |
| CLEAN | AC_LITE | 3 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_LITE | 9 | AC_LITE_CLEAN_seed9_step0010000_policy.pt | 287.8 | 1.000 |
| CLEAN | AC_LITE | 17 | AC_LITE_CLEAN_seed17_step0010000_policy.pt | 286.0 | 1.000 |
| CLEAN | AC_LITE | 42 | AC_LITE_CLEAN_seed42_step0030000_policy.pt | 287.1 | 1.000 |
| CLEAN | BASELINE | 0 | BASELINE_CLEAN_seed0_step0180000.pt | 288.4 | 1.000 |
| CLEAN | BASELINE | 3 | BASELINE_CLEAN_seed3_step0050000.pt | 287.5 | 1.000 |
| CLEAN | BASELINE | 9 | BASELINE_CLEAN_seed9_step0010000.pt | 290.9 | 1.000 |
| CLEAN | BASELINE | 17 | BASELINE_CLEAN_seed17_step0030000.pt | 286.9 | 1.000 |
| CLEAN | BASELINE | 42 | BASELINE_CLEAN_seed42_step0050000.pt | 288.1 | 1.000 |
| OBS_NOISE | AC_FULL | 0 | AC_FULL_OBS_NOISE_seed0_step0160000_policy.pt | 225.0 | 0.667 |
| OBS_NOISE | AC_FULL | 3 | AC_FULL_OBS_NOISE_seed3_step0170000_policy.pt | 257.3 | 0.867 |
| OBS_NOISE | AC_FULL | 9 | AC_FULL_OBS_NOISE_seed9_step0150000_policy.pt | 268.6 | 0.933 |
| OBS_NOISE | AC_FULL | 17 | AC_FULL_OBS_NOISE_seed17_step0010000_policy.pt | 239.1 | 0.733 |
| OBS_NOISE | AC_FULL | 42 | AC_FULL_OBS_NOISE_seed42_step0040000_policy.pt | 237.4 | 0.733 |
| OBS_NOISE | AC_LITE | 0 | AC_LITE_OBS_NOISE_seed0_step0190000_policy.pt | 228.0 | 0.733 |
| OBS_NOISE | AC_LITE | 3 | AC_LITE_OBS_NOISE_seed3_step0210000_policy.pt | 288.8 | 1.000 |
| OBS_NOISE | AC_LITE | 9 | AC_LITE_OBS_NOISE_seed9_step0210000_policy.pt | 279.5 | 0.933 |
| OBS_NOISE | AC_LITE | 17 | AC_LITE_OBS_NOISE_seed17_step0120000_policy.pt | 287.6 | 1.000 |
| OBS_NOISE | AC_LITE | 42 | AC_LITE_OBS_NOISE_seed42_step0050000_policy.pt | 213.7 | 0.667 |
| OBS_NOISE | BASELINE | 0 | BASELINE_OBS_NOISE_seed0_step0050000.pt | 270.7 | 0.933 |
| OBS_NOISE | BASELINE | 3 | BASELINE_OBS_NOISE_seed3_step0030000.pt | 216.5 | 0.733 |
| OBS_NOISE | BASELINE | 9 | BASELINE_OBS_NOISE_seed9_step0060000.pt | 273.7 | 0.933 |
| OBS_NOISE | BASELINE | 17 | BASELINE_OBS_NOISE_seed17_step0200000.pt | 285.9 | 1.000 |
| OBS_NOISE | BASELINE | 42 | BASELINE_OBS_NOISE_seed42_step0250000.pt | 269.6 | 0.933 |
| REWARD_NOISE | AC_FULL | 0 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | AC_FULL | 3 | AC_FULL_REWARD_NOISE_seed3_step0070000_policy.pt | 290.5 | 1.000 |
| REWARD_NOISE | AC_FULL | 9 | AC_FULL_REWARD_NOISE_seed9_step0040000_policy.pt | 291.0 | 1.000 |
| REWARD_NOISE | AC_FULL | 17 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | AC_FULL | 42 | AC_FULL_REWARD_NOISE_seed42_step0100000_policy.pt | 291.6 | 1.000 |
| REWARD_NOISE | AC_LITE | 0 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE | 3 | AC_LITE_REWARD_NOISE_seed3_step0010000_policy.pt | 288.4 | 1.000 |
| REWARD_NOISE | AC_LITE | 9 | AC_LITE_REWARD_NOISE_seed9_step0010000_policy.pt | 289.7 | 1.000 |
| REWARD_NOISE | AC_LITE | 17 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE | 42 | AC_LITE_REWARD_NOISE_seed42_step0040000_policy.pt | 290.9 | 1.000 |
| REWARD_NOISE | BASELINE | 0 | BASELINE_REWARD_NOISE_seed0_step0110000.pt | 288.5 | 1.000 |
| REWARD_NOISE | BASELINE | 3 | BASELINE_REWARD_NOISE_seed3_step0210000.pt | 287.2 | 1.000 |
| REWARD_NOISE | BASELINE | 9 | BASELINE_REWARD_NOISE_seed9_step0090000.pt | 291.5 | 1.000 |
| REWARD_NOISE | BASELINE | 17 | BASELINE_REWARD_NOISE_seed17_step0010000.pt | 289.9 | 1.000 |
| REWARD_NOISE | BASELINE | 42 | BASELINE_REWARD_NOISE_seed42_step0020000.pt | 290.9 | 1.000 |

## Certainty AUROC diagnostics

- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.
- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\) (diagnostic).

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL | 0.919 | 0.238 |
| CLEAN | AC_LITE | 0.738 | 0.156 |
| OBS_NOISE | AC_FULL | 0.883 | 0.419 |
| OBS_NOISE | AC_LITE | 0.735 | 0.332 |
| REWARD_NOISE | AC_FULL | 0.863 | 0.259 |
| REWARD_NOISE | AC_LITE | 0.613 | 0.162 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

