# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-21_143935_all_experiments`

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
| CLEAN | AC_FULL | 260.0 ± 13.0 | 0.900 ± 0.050 | 287.2 | 1.000 |
| CLEAN | AC_LITE | 270.0 ± 5.7 | 0.920 ± 0.057 | 285.3 | 0.990 |
| CLEAN | BASELINE | 266.2 ± 20.2 | 0.930 ± 0.084 | 284.0 | 0.990 |
| OBS_NOISE | AC_FULL | 135.4 ± 29.0 | 0.420 ± 0.130 | 196.2 | 0.670 |
| OBS_NOISE | AC_LITE | 123.3 ± 44.3 | 0.370 ± 0.172 | 195.7 | 0.650 |
| OBS_NOISE | BASELINE | 149.3 ± 15.3 | 0.450 ± 0.035 | 190.8 | 0.610 |
| REWARD_NOISE | AC_FULL | 246.9 ± 9.4 | 0.680 ± 0.125 | 279.1 | 0.820 |
| REWARD_NOISE | AC_LITE | 245.4 ± 22.7 | 0.630 ± 0.067 | 281.0 | 0.830 |
| REWARD_NOISE | BASELINE | 250.5 ± 25.1 | 0.670 ± 0.125 | 279.6 | 0.860 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL | 0 | 267.8 | 0.950 |
| CLEAN | AC_FULL | 3 | 274.5 | 0.950 |
| CLEAN | AC_FULL | 9 | 263.1 | 0.900 |
| CLEAN | AC_FULL | 17 | 253.0 | 0.850 |
| CLEAN | AC_FULL | 42 | 241.3 | 0.850 |
| CLEAN | AC_LITE | 0 | 273.1 | 0.900 |
| CLEAN | AC_LITE | 3 | 276.9 | 1.000 |
| CLEAN | AC_LITE | 9 | 271.2 | 0.950 |
| CLEAN | AC_LITE | 17 | 266.4 | 0.900 |
| CLEAN | AC_LITE | 42 | 262.3 | 0.850 |
| CLEAN | BASELINE | 0 | 272.0 | 0.950 |
| CLEAN | BASELINE | 3 | 287.2 | 1.000 |
| CLEAN | BASELINE | 9 | 238.9 | 0.800 |
| CLEAN | BASELINE | 17 | 252.0 | 0.900 |
| CLEAN | BASELINE | 42 | 281.0 | 1.000 |
| OBS_NOISE | AC_FULL | 0 | 121.4 | 0.350 |
| OBS_NOISE | AC_FULL | 3 | 134.6 | 0.450 |
| OBS_NOISE | AC_FULL | 9 | 98.6 | 0.250 |
| OBS_NOISE | AC_FULL | 17 | 176.7 | 0.600 |
| OBS_NOISE | AC_FULL | 42 | 145.8 | 0.450 |
| OBS_NOISE | AC_LITE | 0 | 102.9 | 0.300 |
| OBS_NOISE | AC_LITE | 3 | 77.9 | 0.200 |
| OBS_NOISE | AC_LITE | 9 | 105.5 | 0.300 |
| OBS_NOISE | AC_LITE | 17 | 192.9 | 0.650 |
| OBS_NOISE | AC_LITE | 42 | 137.4 | 0.400 |
| OBS_NOISE | BASELINE | 0 | 146.1 | 0.450 |
| OBS_NOISE | BASELINE | 3 | 171.0 | 0.500 |
| OBS_NOISE | BASELINE | 9 | 150.3 | 0.450 |
| OBS_NOISE | BASELINE | 17 | 151.0 | 0.450 |
| OBS_NOISE | BASELINE | 42 | 128.0 | 0.400 |
| REWARD_NOISE | AC_FULL | 0 | 253.3 | 0.750 |
| REWARD_NOISE | AC_FULL | 3 | 255.6 | 0.750 |
| REWARD_NOISE | AC_FULL | 9 | 246.1 | 0.500 |
| REWARD_NOISE | AC_FULL | 17 | 231.6 | 0.600 |
| REWARD_NOISE | AC_FULL | 42 | 247.9 | 0.800 |
| REWARD_NOISE | AC_LITE | 0 | 216.0 | 0.550 |
| REWARD_NOISE | AC_LITE | 3 | 272.4 | 0.700 |
| REWARD_NOISE | AC_LITE | 9 | 253.1 | 0.600 |
| REWARD_NOISE | AC_LITE | 17 | 257.0 | 0.700 |
| REWARD_NOISE | AC_LITE | 42 | 228.6 | 0.600 |
| REWARD_NOISE | BASELINE | 0 | 260.7 | 0.750 |
| REWARD_NOISE | BASELINE | 3 | 271.4 | 0.700 |
| REWARD_NOISE | BASELINE | 9 | 216.4 | 0.450 |
| REWARD_NOISE | BASELINE | 17 | 272.3 | 0.750 |
| REWARD_NOISE | BASELINE | 42 | 231.8 | 0.700 |

## Best checkpoint by greedy held-out evaluation

Checkpoints are evaluated greedily on fixed held-out seeds. The final checkpoint is not assumed to be best.

| mode | method | seed | checkpoint | eval return | eval success |
|---|---|---:|---|---:|---:|
| CLEAN | AC_FULL | 0 | AC_FULL_CLEAN_seed0_step0010000_policy.pt | 288.1 | 1.000 |
| CLEAN | AC_FULL | 3 | AC_FULL_CLEAN_seed3_step0010000_policy.pt | 287.1 | 1.000 |
| CLEAN | AC_FULL | 9 | AC_FULL_CLEAN_seed9_step0020000_policy.pt | 286.4 | 1.000 |
| CLEAN | AC_FULL | 17 | AC_FULL_CLEAN_seed17_final_policy.pt | 290.1 | 1.000 |
| CLEAN | AC_FULL | 42 | AC_FULL_CLEAN_seed42_step0010000_policy.pt | 289.0 | 1.000 |
| CLEAN | AC_LITE | 0 | AC_LITE_CLEAN_seed0_step0020000_policy.pt | 288.0 | 1.000 |
| CLEAN | AC_LITE | 3 | AC_LITE_CLEAN_seed3_final_policy.pt | 286.7 | 1.000 |
| CLEAN | AC_LITE | 9 | AC_LITE_CLEAN_seed9_final_policy.pt | 286.5 | 1.000 |
| CLEAN | AC_LITE | 17 | AC_LITE_CLEAN_seed17_step0010000_policy.pt | 289.6 | 1.000 |
| CLEAN | AC_LITE | 42 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | BASELINE | 0 | BASELINE_CLEAN_seed0_step0010000.pt | 287.6 | 1.000 |
| CLEAN | BASELINE | 3 | BASELINE_CLEAN_seed3_final.pt | 288.4 | 1.000 |
| CLEAN | BASELINE | 9 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | BASELINE | 17 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | BASELINE | 42 | BASELINE_CLEAN_seed42_step0010000.pt | 288.4 | 1.000 |
| OBS_NOISE | AC_FULL | 0 | AC_FULL_OBS_NOISE_seed0_final_policy.pt | 222.4 | 0.667 |
| OBS_NOISE | AC_FULL | 3 | AC_FULL_OBS_NOISE_seed3_final_policy.pt | 221.2 | 0.667 |
| OBS_NOISE | AC_FULL | 9 | checkpoint_0_pretrained | 213.0 | 0.600 |
| OBS_NOISE | AC_FULL | 17 | AC_FULL_OBS_NOISE_seed17_step0010000_policy.pt | 253.3 | 0.800 |
| OBS_NOISE | AC_FULL | 42 | AC_FULL_OBS_NOISE_seed42_step0020000_policy.pt | 236.2 | 0.733 |
| OBS_NOISE | AC_LITE | 0 | AC_LITE_OBS_NOISE_seed0_final_policy.pt | 216.1 | 0.600 |
| OBS_NOISE | AC_LITE | 3 | checkpoint_0_pretrained | 213.0 | 0.600 |
| OBS_NOISE | AC_LITE | 9 | checkpoint_0_pretrained | 213.0 | 0.600 |
| OBS_NOISE | AC_LITE | 17 | AC_LITE_OBS_NOISE_seed17_step0010000_policy.pt | 238.3 | 0.667 |
| OBS_NOISE | AC_LITE | 42 | checkpoint_0_pretrained | 213.0 | 0.600 |
| OBS_NOISE | BASELINE | 0 | BASELINE_OBS_NOISE_seed0_step0010000.pt | 251.1 | 0.800 |
| OBS_NOISE | BASELINE | 3 | checkpoint_0_pretrained | 213.0 | 0.600 |
| OBS_NOISE | BASELINE | 9 | BASELINE_OBS_NOISE_seed9_step0010000.pt | 239.1 | 0.733 |
| OBS_NOISE | BASELINE | 17 | BASELINE_OBS_NOISE_seed17_step0010000.pt | 221.8 | 0.667 |
| OBS_NOISE | BASELINE | 42 | BASELINE_OBS_NOISE_seed42_step0020000.pt | 252.2 | 0.867 |
| REWARD_NOISE | AC_FULL | 0 | AC_FULL_REWARD_NOISE_seed0_final_policy.pt | 288.3 | 1.000 |
| REWARD_NOISE | AC_FULL | 3 | AC_FULL_REWARD_NOISE_seed3_step0010000_policy.pt | 289.5 | 1.000 |
| REWARD_NOISE | AC_FULL | 9 | AC_FULL_REWARD_NOISE_seed9_step0020000_policy.pt | 289.2 | 1.000 |
| REWARD_NOISE | AC_FULL | 17 | AC_FULL_REWARD_NOISE_seed17_step0010000_policy.pt | 288.3 | 1.000 |
| REWARD_NOISE | AC_FULL | 42 | AC_FULL_REWARD_NOISE_seed42_final_policy.pt | 288.3 | 1.000 |
| REWARD_NOISE | AC_LITE | 0 | AC_LITE_REWARD_NOISE_seed0_step0020000_policy.pt | 288.1 | 1.000 |
| REWARD_NOISE | AC_LITE | 3 | AC_LITE_REWARD_NOISE_seed3_step0010000_policy.pt | 289.2 | 1.000 |
| REWARD_NOISE | AC_LITE | 9 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE | 17 | AC_LITE_REWARD_NOISE_seed17_step0010000_policy.pt | 289.8 | 1.000 |
| REWARD_NOISE | AC_LITE | 42 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE | 0 | BASELINE_REWARD_NOISE_seed0_step0010000.pt | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE | 3 | BASELINE_REWARD_NOISE_seed3_final.pt | 288.8 | 1.000 |
| REWARD_NOISE | BASELINE | 9 | BASELINE_REWARD_NOISE_seed9_step0010000.pt | 290.9 | 1.000 |
| REWARD_NOISE | BASELINE | 17 | BASELINE_REWARD_NOISE_seed17_step0010000.pt | 289.3 | 1.000 |
| REWARD_NOISE | BASELINE | 42 | BASELINE_REWARD_NOISE_seed42_step0020000.pt | 289.3 | 1.000 |

## Certainty AUROC diagnostics

- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.
- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\) (diagnostic).

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL | 0.450 | 0.450 |
| CLEAN | AC_LITE | 0.491 | 0.443 |
| OBS_NOISE | AC_FULL | 0.527 | 0.458 |
| OBS_NOISE | AC_LITE | 0.387 | 0.507 |
| REWARD_NOISE | AC_FULL | 0.431 | 0.460 |
| REWARD_NOISE | AC_LITE | 0.426 | 0.464 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

