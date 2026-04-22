# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-22_130638_all_experiments`

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
| CLEAN | AC_FULL | 270.8 ± 12.2 | 0.940 ± 0.065 | 292.2 | 1.000 |
| CLEAN | AC_LITE | 262.4 ± 11.7 | 0.910 ± 0.096 | 289.8 | 1.000 |
| CLEAN | BASELINE | 259.6 ± 8.7 | 0.910 ± 0.042 | 285.7 | 1.000 |
| OBS_NOISE | AC_FULL | 194.5 ± 47.7 | 0.710 ± 0.119 | 284.8 | 1.000 |
| OBS_NOISE | AC_LITE | 220.5 ± 34.7 | 0.820 ± 0.135 | 282.7 | 1.000 |
| OBS_NOISE | BASELINE | 225.2 ± 16.2 | 0.810 ± 0.096 | 284.8 | 1.000 |
| REWARD_NOISE | AC_FULL | 242.4 ± 13.7 | 0.610 ± 0.089 | 282.8 | 0.930 |
| REWARD_NOISE | AC_LITE | 239.9 ± 14.6 | 0.650 ± 0.079 | 285.6 | 0.950 |
| REWARD_NOISE | BASELINE | 253.1 ± 16.9 | 0.630 ± 0.076 | 286.1 | 0.960 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL | 0 | 288.0 | 1.000 |
| CLEAN | AC_FULL | 3 | 270.2 | 0.950 |
| CLEAN | AC_FULL | 9 | 254.5 | 0.850 |
| CLEAN | AC_FULL | 17 | 266.5 | 0.900 |
| CLEAN | AC_FULL | 42 | 274.7 | 1.000 |
| CLEAN | AC_LITE | 0 | 265.5 | 0.950 |
| CLEAN | AC_LITE | 3 | 263.5 | 0.950 |
| CLEAN | AC_LITE | 9 | 244.7 | 0.750 |
| CLEAN | AC_LITE | 17 | 277.3 | 1.000 |
| CLEAN | AC_LITE | 42 | 260.8 | 0.900 |
| CLEAN | BASELINE | 0 | 250.0 | 0.900 |
| CLEAN | BASELINE | 3 | 259.3 | 0.900 |
| CLEAN | BASELINE | 9 | 266.5 | 0.950 |
| CLEAN | BASELINE | 17 | 270.1 | 0.950 |
| CLEAN | BASELINE | 42 | 252.3 | 0.850 |
| OBS_NOISE | AC_FULL | 0 | 224.6 | 0.800 |
| OBS_NOISE | AC_FULL | 3 | 153.7 | 0.650 |
| OBS_NOISE | AC_FULL | 9 | 134.9 | 0.550 |
| OBS_NOISE | AC_FULL | 17 | 245.2 | 0.850 |
| OBS_NOISE | AC_FULL | 42 | 214.0 | 0.700 |
| OBS_NOISE | AC_LITE | 0 | 194.9 | 0.800 |
| OBS_NOISE | AC_LITE | 3 | 235.5 | 0.850 |
| OBS_NOISE | AC_LITE | 9 | 246.7 | 0.900 |
| OBS_NOISE | AC_LITE | 17 | 252.2 | 0.950 |
| OBS_NOISE | AC_LITE | 42 | 173.2 | 0.600 |
| OBS_NOISE | BASELINE | 0 | 245.9 | 0.850 |
| OBS_NOISE | BASELINE | 3 | 218.6 | 0.800 |
| OBS_NOISE | BASELINE | 9 | 231.7 | 0.900 |
| OBS_NOISE | BASELINE | 17 | 227.9 | 0.850 |
| OBS_NOISE | BASELINE | 42 | 202.2 | 0.650 |
| REWARD_NOISE | AC_FULL | 0 | 256.9 | 0.600 |
| REWARD_NOISE | AC_FULL | 3 | 247.1 | 0.750 |
| REWARD_NOISE | AC_FULL | 9 | 237.2 | 0.600 |
| REWARD_NOISE | AC_FULL | 17 | 221.5 | 0.500 |
| REWARD_NOISE | AC_FULL | 42 | 249.3 | 0.600 |
| REWARD_NOISE | AC_LITE | 0 | 242.9 | 0.650 |
| REWARD_NOISE | AC_LITE | 3 | 235.7 | 0.700 |
| REWARD_NOISE | AC_LITE | 9 | 263.2 | 0.750 |
| REWARD_NOISE | AC_LITE | 17 | 224.4 | 0.550 |
| REWARD_NOISE | AC_LITE | 42 | 233.5 | 0.600 |
| REWARD_NOISE | BASELINE | 0 | 243.7 | 0.700 |
| REWARD_NOISE | BASELINE | 3 | 240.4 | 0.550 |
| REWARD_NOISE | BASELINE | 9 | 245.7 | 0.650 |
| REWARD_NOISE | BASELINE | 17 | 282.2 | 0.700 |
| REWARD_NOISE | BASELINE | 42 | 253.6 | 0.550 |

## Best checkpoint by greedy held-out evaluation

Checkpoints are evaluated greedily on fixed held-out seeds. The final checkpoint is not assumed to be best.

| mode | method | seed | checkpoint | eval return | eval success |
|---|---|---:|---|---:|---:|
| CLEAN | AC_FULL | 0 | AC_FULL_CLEAN_seed0_step0030000_policy.pt | 288.5 | 1.000 |
| CLEAN | AC_FULL | 3 | AC_FULL_CLEAN_seed3_step0460000_policy.pt | 290.1 | 1.000 |
| CLEAN | AC_FULL | 9 | AC_FULL_CLEAN_seed9_step0440000_policy.pt | 288.9 | 1.000 |
| CLEAN | AC_FULL | 17 | AC_FULL_CLEAN_seed17_step0020000_policy.pt | 289.1 | 1.000 |
| CLEAN | AC_FULL | 42 | AC_FULL_CLEAN_seed42_step0010000_policy.pt | 289.0 | 1.000 |
| CLEAN | AC_LITE | 0 | AC_LITE_CLEAN_seed0_step0020000_policy.pt | 288.0 | 1.000 |
| CLEAN | AC_LITE | 3 | AC_LITE_CLEAN_seed3_step0210000_policy.pt | 289.4 | 1.000 |
| CLEAN | AC_LITE | 9 | AC_LITE_CLEAN_seed9_step0030000_policy.pt | 289.7 | 1.000 |
| CLEAN | AC_LITE | 17 | AC_LITE_CLEAN_seed17_step0010000_policy.pt | 289.6 | 1.000 |
| CLEAN | AC_LITE | 42 | AC_LITE_CLEAN_seed42_step0050000_policy.pt | 289.2 | 1.000 |
| CLEAN | BASELINE | 0 | BASELINE_CLEAN_seed0_step0010000.pt | 287.6 | 1.000 |
| CLEAN | BASELINE | 3 | BASELINE_CLEAN_seed3_step0020000.pt | 288.2 | 1.000 |
| CLEAN | BASELINE | 9 | BASELINE_CLEAN_seed9_step0070000.pt | 287.8 | 1.000 |
| CLEAN | BASELINE | 17 | BASELINE_CLEAN_seed17_step0050000.pt | 287.4 | 1.000 |
| CLEAN | BASELINE | 42 | BASELINE_CLEAN_seed42_step0030000.pt | 289.7 | 1.000 |
| OBS_NOISE | AC_FULL | 0 | AC_FULL_OBS_NOISE_seed0_step0330000_policy.pt | 286.3 | 1.000 |
| OBS_NOISE | AC_FULL | 3 | AC_FULL_OBS_NOISE_seed3_step0080000_policy.pt | 283.7 | 1.000 |
| OBS_NOISE | AC_FULL | 9 | AC_FULL_OBS_NOISE_seed9_step0430000_policy.pt | 288.2 | 1.000 |
| OBS_NOISE | AC_FULL | 17 | AC_FULL_OBS_NOISE_seed17_step0120000_policy.pt | 289.6 | 1.000 |
| OBS_NOISE | AC_FULL | 42 | AC_FULL_OBS_NOISE_seed42_step0260000_policy.pt | 288.7 | 1.000 |
| OBS_NOISE | AC_LITE | 0 | AC_LITE_OBS_NOISE_seed0_step0390000_policy.pt | 291.5 | 1.000 |
| OBS_NOISE | AC_LITE | 3 | AC_LITE_OBS_NOISE_seed3_step0120000_policy.pt | 283.8 | 1.000 |
| OBS_NOISE | AC_LITE | 9 | AC_LITE_OBS_NOISE_seed9_step0060000_policy.pt | 291.8 | 1.000 |
| OBS_NOISE | AC_LITE | 17 | AC_LITE_OBS_NOISE_seed17_step0480000_policy.pt | 281.9 | 1.000 |
| OBS_NOISE | AC_LITE | 42 | AC_LITE_OBS_NOISE_seed42_step0080000_policy.pt | 286.2 | 1.000 |
| OBS_NOISE | BASELINE | 0 | BASELINE_OBS_NOISE_seed0_step0440000.pt | 285.1 | 1.000 |
| OBS_NOISE | BASELINE | 3 | BASELINE_OBS_NOISE_seed3_step0410000.pt | 289.3 | 1.000 |
| OBS_NOISE | BASELINE | 9 | BASELINE_OBS_NOISE_seed9_step0090000.pt | 287.5 | 1.000 |
| OBS_NOISE | BASELINE | 17 | BASELINE_OBS_NOISE_seed17_step0330000.pt | 287.9 | 1.000 |
| OBS_NOISE | BASELINE | 42 | BASELINE_OBS_NOISE_seed42_step0120000.pt | 287.0 | 1.000 |
| REWARD_NOISE | AC_FULL | 0 | AC_FULL_REWARD_NOISE_seed0_step0010000_policy.pt | 287.5 | 1.000 |
| REWARD_NOISE | AC_FULL | 3 | AC_FULL_REWARD_NOISE_seed3_step0010000_policy.pt | 289.5 | 1.000 |
| REWARD_NOISE | AC_FULL | 9 | AC_FULL_REWARD_NOISE_seed9_step0020000_policy.pt | 289.2 | 1.000 |
| REWARD_NOISE | AC_FULL | 17 | AC_FULL_REWARD_NOISE_seed17_step0010000_policy.pt | 288.3 | 1.000 |
| REWARD_NOISE | AC_FULL | 42 | AC_FULL_REWARD_NOISE_seed42_step0030000_policy.pt | 288.1 | 1.000 |
| REWARD_NOISE | AC_LITE | 0 | AC_LITE_REWARD_NOISE_seed0_step0040000_policy.pt | 289.1 | 1.000 |
| REWARD_NOISE | AC_LITE | 3 | AC_LITE_REWARD_NOISE_seed3_step0010000_policy.pt | 289.2 | 1.000 |
| REWARD_NOISE | AC_LITE | 9 | AC_LITE_REWARD_NOISE_seed9_step0390000_policy.pt | 290.0 | 1.000 |
| REWARD_NOISE | AC_LITE | 17 | AC_LITE_REWARD_NOISE_seed17_step0010000_policy.pt | 289.8 | 1.000 |
| REWARD_NOISE | AC_LITE | 42 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE | 0 | BASELINE_REWARD_NOISE_seed0_step0170000.pt | 289.2 | 1.000 |
| REWARD_NOISE | BASELINE | 3 | BASELINE_REWARD_NOISE_seed3_step0040000.pt | 288.1 | 1.000 |
| REWARD_NOISE | BASELINE | 9 | BASELINE_REWARD_NOISE_seed9_step0010000.pt | 290.9 | 1.000 |
| REWARD_NOISE | BASELINE | 17 | BASELINE_REWARD_NOISE_seed17_step0010000.pt | 289.3 | 1.000 |
| REWARD_NOISE | BASELINE | 42 | BASELINE_REWARD_NOISE_seed42_step0460000.pt | 289.9 | 1.000 |

## Certainty AUROC diagnostics

- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.
- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\) (diagnostic).

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL | 0.627 | 0.531 |
| CLEAN | AC_LITE | 0.412 | 0.375 |
| OBS_NOISE | AC_FULL | 0.740 | 0.311 |
| OBS_NOISE | AC_LITE | 0.645 | 0.393 |
| REWARD_NOISE | AC_FULL | 0.532 | 0.327 |
| REWARD_NOISE | AC_LITE | 0.484 | 0.390 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

