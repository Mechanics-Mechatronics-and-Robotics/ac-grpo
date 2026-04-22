# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-21_193602_all_experiments`

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
| CLEAN | AC_FULL | 258.7 ± 13.1 | 0.880 ± 0.076 | 291.4 | 1.000 |
| CLEAN | AC_LITE | 212.5 ± 35.1 | 0.700 ± 0.132 | 289.2 | 1.000 |
| CLEAN | BASELINE | 233.4 ± 17.0 | 0.780 ± 0.076 | 285.2 | 1.000 |
| OBS_NOISE | AC_FULL | 200.1 ± 54.3 | 0.720 ± 0.175 | 277.6 | 0.970 |
| OBS_NOISE | AC_LITE | 187.4 ± 40.7 | 0.630 ± 0.175 | 279.1 | 1.000 |
| OBS_NOISE | BASELINE | 233.3 ± 25.0 | 0.800 ± 0.094 | 280.4 | 0.990 |
| REWARD_NOISE | AC_FULL | 216.7 ± 22.9 | 0.540 ± 0.096 | 279.7 | 0.880 |
| REWARD_NOISE | AC_LITE | 214.8 ± 19.7 | 0.500 ± 0.061 | 284.1 | 0.910 |
| REWARD_NOISE | BASELINE | 218.2 ± 28.9 | 0.570 ± 0.115 | 285.0 | 0.950 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL | 0 | 267.0 | 0.950 |
| CLEAN | AC_FULL | 3 | 239.0 | 0.800 |
| CLEAN | AC_FULL | 9 | 268.4 | 0.900 |
| CLEAN | AC_FULL | 17 | 267.8 | 0.950 |
| CLEAN | AC_FULL | 42 | 251.1 | 0.800 |
| CLEAN | AC_LITE | 0 | 258.8 | 0.900 |
| CLEAN | AC_LITE | 3 | 228.2 | 0.750 |
| CLEAN | AC_LITE | 9 | 182.3 | 0.650 |
| CLEAN | AC_LITE | 17 | 220.1 | 0.650 |
| CLEAN | AC_LITE | 42 | 172.9 | 0.550 |
| CLEAN | BASELINE | 0 | 235.0 | 0.750 |
| CLEAN | BASELINE | 3 | 220.6 | 0.750 |
| CLEAN | BASELINE | 9 | 260.4 | 0.900 |
| CLEAN | BASELINE | 17 | 233.6 | 0.800 |
| CLEAN | BASELINE | 42 | 217.5 | 0.700 |
| OBS_NOISE | AC_FULL | 0 | 210.0 | 0.750 |
| OBS_NOISE | AC_FULL | 3 | 133.1 | 0.500 |
| OBS_NOISE | AC_FULL | 9 | 265.0 | 0.950 |
| OBS_NOISE | AC_FULL | 17 | 157.5 | 0.600 |
| OBS_NOISE | AC_FULL | 42 | 234.7 | 0.800 |
| OBS_NOISE | AC_LITE | 0 | 157.2 | 0.450 |
| OBS_NOISE | AC_LITE | 3 | 201.7 | 0.700 |
| OBS_NOISE | AC_LITE | 9 | 236.1 | 0.850 |
| OBS_NOISE | AC_LITE | 17 | 134.8 | 0.450 |
| OBS_NOISE | AC_LITE | 42 | 207.1 | 0.700 |
| OBS_NOISE | BASELINE | 0 | 225.4 | 0.800 |
| OBS_NOISE | BASELINE | 3 | 252.5 | 0.850 |
| OBS_NOISE | BASELINE | 9 | 235.9 | 0.800 |
| OBS_NOISE | BASELINE | 17 | 257.6 | 0.900 |
| OBS_NOISE | BASELINE | 42 | 194.9 | 0.650 |
| REWARD_NOISE | AC_FULL | 0 | 244.0 | 0.600 |
| REWARD_NOISE | AC_FULL | 3 | 229.5 | 0.500 |
| REWARD_NOISE | AC_FULL | 9 | 183.7 | 0.400 |
| REWARD_NOISE | AC_FULL | 17 | 206.7 | 0.550 |
| REWARD_NOISE | AC_FULL | 42 | 219.5 | 0.650 |
| REWARD_NOISE | AC_LITE | 0 | 203.8 | 0.450 |
| REWARD_NOISE | AC_LITE | 3 | 238.8 | 0.450 |
| REWARD_NOISE | AC_LITE | 9 | 232.8 | 0.500 |
| REWARD_NOISE | AC_LITE | 17 | 204.3 | 0.600 |
| REWARD_NOISE | AC_LITE | 42 | 194.2 | 0.500 |
| REWARD_NOISE | BASELINE | 0 | 263.2 | 0.650 |
| REWARD_NOISE | BASELINE | 3 | 193.9 | 0.400 |
| REWARD_NOISE | BASELINE | 9 | 191.8 | 0.550 |
| REWARD_NOISE | BASELINE | 17 | 222.8 | 0.550 |
| REWARD_NOISE | BASELINE | 42 | 219.2 | 0.700 |

## Best checkpoint by greedy held-out evaluation

Checkpoints are evaluated greedily on fixed held-out seeds. The final checkpoint is not assumed to be best.

| mode | method | seed | checkpoint | eval return | eval success |
|---|---|---:|---|---:|---:|
| CLEAN | AC_FULL | 0 | AC_FULL_CLEAN_seed0_step0030000_policy.pt | 288.5 | 1.000 |
| CLEAN | AC_FULL | 3 | AC_FULL_CLEAN_seed3_step0060000_policy.pt | 287.3 | 1.000 |
| CLEAN | AC_FULL | 9 | AC_FULL_CLEAN_seed9_step0240000_policy.pt | 287.6 | 1.000 |
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
| OBS_NOISE | AC_FULL | 0 | AC_FULL_OBS_NOISE_seed0_final_policy.pt | 286.9 | 1.000 |
| OBS_NOISE | AC_FULL | 3 | AC_FULL_OBS_NOISE_seed3_step0080000_policy.pt | 283.7 | 1.000 |
| OBS_NOISE | AC_FULL | 9 | AC_FULL_OBS_NOISE_seed9_step0050000_policy.pt | 285.0 | 1.000 |
| OBS_NOISE | AC_FULL | 17 | AC_FULL_OBS_NOISE_seed17_step0120000_policy.pt | 289.6 | 1.000 |
| OBS_NOISE | AC_FULL | 42 | AC_FULL_OBS_NOISE_seed42_step0230000_policy.pt | 288.0 | 1.000 |
| OBS_NOISE | AC_LITE | 0 | AC_LITE_OBS_NOISE_seed0_step0120000_policy.pt | 279.2 | 0.933 |
| OBS_NOISE | AC_LITE | 3 | AC_LITE_OBS_NOISE_seed3_step0120000_policy.pt | 283.8 | 1.000 |
| OBS_NOISE | AC_LITE | 9 | AC_LITE_OBS_NOISE_seed9_step0060000_policy.pt | 291.8 | 1.000 |
| OBS_NOISE | AC_LITE | 17 | AC_LITE_OBS_NOISE_seed17_step0180000_policy.pt | 281.5 | 1.000 |
| OBS_NOISE | AC_LITE | 42 | AC_LITE_OBS_NOISE_seed42_step0080000_policy.pt | 286.2 | 1.000 |
| OBS_NOISE | BASELINE | 0 | BASELINE_OBS_NOISE_seed0_step0250000.pt | 281.9 | 1.000 |
| OBS_NOISE | BASELINE | 3 | BASELINE_OBS_NOISE_seed3_step0150000.pt | 286.2 | 1.000 |
| OBS_NOISE | BASELINE | 9 | BASELINE_OBS_NOISE_seed9_step0090000.pt | 287.5 | 1.000 |
| OBS_NOISE | BASELINE | 17 | BASELINE_OBS_NOISE_seed17_step0180000.pt | 286.8 | 1.000 |
| OBS_NOISE | BASELINE | 42 | BASELINE_OBS_NOISE_seed42_step0120000.pt | 287.0 | 1.000 |
| REWARD_NOISE | AC_FULL | 0 | AC_FULL_REWARD_NOISE_seed0_step0010000_policy.pt | 287.5 | 1.000 |
| REWARD_NOISE | AC_FULL | 3 | AC_FULL_REWARD_NOISE_seed3_step0010000_policy.pt | 289.5 | 1.000 |
| REWARD_NOISE | AC_FULL | 9 | AC_FULL_REWARD_NOISE_seed9_step0020000_policy.pt | 289.2 | 1.000 |
| REWARD_NOISE | AC_FULL | 17 | AC_FULL_REWARD_NOISE_seed17_step0010000_policy.pt | 288.3 | 1.000 |
| REWARD_NOISE | AC_FULL | 42 | AC_FULL_REWARD_NOISE_seed42_step0030000_policy.pt | 288.1 | 1.000 |
| REWARD_NOISE | AC_LITE | 0 | AC_LITE_REWARD_NOISE_seed0_step0040000_policy.pt | 289.1 | 1.000 |
| REWARD_NOISE | AC_LITE | 3 | AC_LITE_REWARD_NOISE_seed3_step0010000_policy.pt | 289.2 | 1.000 |
| REWARD_NOISE | AC_LITE | 9 | AC_LITE_REWARD_NOISE_seed9_step0020000_policy.pt | 286.6 | 1.000 |
| REWARD_NOISE | AC_LITE | 17 | AC_LITE_REWARD_NOISE_seed17_step0010000_policy.pt | 289.8 | 1.000 |
| REWARD_NOISE | AC_LITE | 42 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE | 0 | BASELINE_REWARD_NOISE_seed0_step0170000.pt | 289.2 | 1.000 |
| REWARD_NOISE | BASELINE | 3 | BASELINE_REWARD_NOISE_seed3_step0040000.pt | 288.1 | 1.000 |
| REWARD_NOISE | BASELINE | 9 | BASELINE_REWARD_NOISE_seed9_step0010000.pt | 290.9 | 1.000 |
| REWARD_NOISE | BASELINE | 17 | BASELINE_REWARD_NOISE_seed17_step0010000.pt | 289.3 | 1.000 |
| REWARD_NOISE | BASELINE | 42 | BASELINE_REWARD_NOISE_seed42_step0020000.pt | 289.3 | 1.000 |

## Certainty AUROC diagnostics

- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.
- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\) (diagnostic).

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL | 0.545 | 0.459 |
| CLEAN | AC_LITE | 0.382 | 0.447 |
| OBS_NOISE | AC_FULL | 0.698 | 0.365 |
| OBS_NOISE | AC_LITE | 0.664 | 0.451 |
| REWARD_NOISE | AC_FULL | 0.509 | 0.347 |
| REWARD_NOISE | AC_LITE | 0.455 | 0.456 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

