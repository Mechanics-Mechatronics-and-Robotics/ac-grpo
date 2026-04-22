# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-21_150935_all_experiments`

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
| CLEAN | AC_FULL | 264.7 ± 18.2 | 0.920 ± 0.076 | 284.8 | 1.000 |
| CLEAN | AC_LITE | 264.7 ± 18.2 | 0.920 ± 0.076 | 284.8 | 1.000 |
| CLEAN | BASELINE | 264.5 ± 13.5 | 0.910 ± 0.089 | 284.4 | 0.990 |
| OBS_NOISE | AC_FULL | 198.5 ± 15.8 | 0.640 ± 0.042 | 233.4 | 0.790 |
| OBS_NOISE | AC_LITE | 198.5 ± 15.8 | 0.640 ± 0.042 | 233.4 | 0.790 |
| OBS_NOISE | BASELINE | 184.1 ± 31.3 | 0.590 ± 0.129 | 223.3 | 0.750 |
| REWARD_NOISE | AC_FULL | 265.9 ± 10.8 | 0.780 ± 0.084 | 286.1 | 0.890 |
| REWARD_NOISE | AC_LITE | 265.9 ± 10.8 | 0.780 ± 0.084 | 286.1 | 0.890 |
| REWARD_NOISE | BASELINE | 261.0 ± 7.9 | 0.790 ± 0.074 | 284.8 | 0.880 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL | 0 | 255.1 | 0.900 |
| CLEAN | AC_FULL | 3 | 277.5 | 0.950 |
| CLEAN | AC_FULL | 9 | 238.5 | 0.800 |
| CLEAN | AC_FULL | 17 | 284.0 | 1.000 |
| CLEAN | AC_FULL | 42 | 268.5 | 0.950 |
| CLEAN | AC_LITE | 0 | 255.1 | 0.900 |
| CLEAN | AC_LITE | 3 | 277.5 | 0.950 |
| CLEAN | AC_LITE | 9 | 238.5 | 0.800 |
| CLEAN | AC_LITE | 17 | 284.0 | 1.000 |
| CLEAN | AC_LITE | 42 | 268.5 | 0.950 |
| CLEAN | BASELINE | 0 | 280.4 | 1.000 |
| CLEAN | BASELINE | 3 | 255.0 | 0.850 |
| CLEAN | BASELINE | 9 | 265.3 | 0.900 |
| CLEAN | BASELINE | 17 | 247.3 | 0.800 |
| CLEAN | BASELINE | 42 | 274.3 | 1.000 |
| OBS_NOISE | AC_FULL | 0 | 220.0 | 0.700 |
| OBS_NOISE | AC_FULL | 3 | 208.2 | 0.650 |
| OBS_NOISE | AC_FULL | 9 | 195.3 | 0.600 |
| OBS_NOISE | AC_FULL | 17 | 180.0 | 0.600 |
| OBS_NOISE | AC_FULL | 42 | 189.2 | 0.650 |
| OBS_NOISE | AC_LITE | 0 | 220.0 | 0.700 |
| OBS_NOISE | AC_LITE | 3 | 208.2 | 0.650 |
| OBS_NOISE | AC_LITE | 9 | 195.3 | 0.600 |
| OBS_NOISE | AC_LITE | 17 | 180.0 | 0.600 |
| OBS_NOISE | AC_LITE | 42 | 189.2 | 0.650 |
| OBS_NOISE | BASELINE | 0 | 182.5 | 0.550 |
| OBS_NOISE | BASELINE | 3 | 180.5 | 0.550 |
| OBS_NOISE | BASELINE | 9 | 141.7 | 0.450 |
| OBS_NOISE | BASELINE | 17 | 185.8 | 0.600 |
| OBS_NOISE | BASELINE | 42 | 230.0 | 0.800 |
| REWARD_NOISE | AC_FULL | 0 | 281.5 | 0.850 |
| REWARD_NOISE | AC_FULL | 3 | 260.9 | 0.650 |
| REWARD_NOISE | AC_FULL | 9 | 270.5 | 0.850 |
| REWARD_NOISE | AC_FULL | 17 | 263.8 | 0.750 |
| REWARD_NOISE | AC_FULL | 42 | 252.9 | 0.800 |
| REWARD_NOISE | AC_LITE | 0 | 281.5 | 0.850 |
| REWARD_NOISE | AC_LITE | 3 | 260.9 | 0.650 |
| REWARD_NOISE | AC_LITE | 9 | 270.5 | 0.850 |
| REWARD_NOISE | AC_LITE | 17 | 263.8 | 0.750 |
| REWARD_NOISE | AC_LITE | 42 | 252.9 | 0.800 |
| REWARD_NOISE | BASELINE | 0 | 265.7 | 0.800 |
| REWARD_NOISE | BASELINE | 3 | 249.9 | 0.750 |
| REWARD_NOISE | BASELINE | 9 | 258.9 | 0.700 |
| REWARD_NOISE | BASELINE | 17 | 259.6 | 0.800 |
| REWARD_NOISE | BASELINE | 42 | 270.9 | 0.900 |

## Best checkpoint by greedy held-out evaluation

Checkpoints are evaluated greedily on fixed held-out seeds. The final checkpoint is not assumed to be best.

| mode | method | seed | checkpoint | eval return | eval success |
|---|---|---:|---|---:|---:|
| CLEAN | AC_FULL | 0 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_FULL | 3 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_FULL | 9 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_FULL | 17 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_FULL | 42 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_LITE | 0 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_LITE | 3 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_LITE | 9 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_LITE | 17 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_LITE | 42 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | BASELINE | 0 | BASELINE_CLEAN_seed0_step0010000.pt | 287.9 | 1.000 |
| CLEAN | BASELINE | 3 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | BASELINE | 9 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | BASELINE | 17 | BASELINE_CLEAN_seed17_step0020000.pt | 288.4 | 1.000 |
| CLEAN | BASELINE | 42 | checkpoint_0_pretrained | 285.9 | 1.000 |
| OBS_NOISE | AC_FULL | 0 | AC_FULL_OBS_NOISE_seed0_step0010000_policy.pt | 247.3 | 0.800 |
| OBS_NOISE | AC_FULL | 3 | AC_FULL_OBS_NOISE_seed3_step0010000_policy.pt | 214.5 | 0.600 |
| OBS_NOISE | AC_FULL | 9 | AC_FULL_OBS_NOISE_seed9_final_policy.pt | 238.8 | 0.733 |
| OBS_NOISE | AC_FULL | 17 | AC_FULL_OBS_NOISE_seed17_final_policy.pt | 234.5 | 0.667 |
| OBS_NOISE | AC_FULL | 42 | AC_FULL_OBS_NOISE_seed42_final_policy.pt | 226.8 | 0.667 |
| OBS_NOISE | AC_LITE | 0 | AC_LITE_OBS_NOISE_seed0_step0010000_policy.pt | 247.3 | 0.800 |
| OBS_NOISE | AC_LITE | 3 | AC_LITE_OBS_NOISE_seed3_step0010000_policy.pt | 214.5 | 0.600 |
| OBS_NOISE | AC_LITE | 9 | AC_LITE_OBS_NOISE_seed9_step0020000_policy.pt | 238.8 | 0.733 |
| OBS_NOISE | AC_LITE | 17 | AC_LITE_OBS_NOISE_seed17_step0020000_policy.pt | 234.5 | 0.667 |
| OBS_NOISE | AC_LITE | 42 | AC_LITE_OBS_NOISE_seed42_step0020000_policy.pt | 226.8 | 0.667 |
| OBS_NOISE | BASELINE | 0 | BASELINE_OBS_NOISE_seed0_final.pt | 250.9 | 0.800 |
| OBS_NOISE | BASELINE | 3 | BASELINE_OBS_NOISE_seed3_final.pt | 225.8 | 0.667 |
| OBS_NOISE | BASELINE | 9 | BASELINE_OBS_NOISE_seed9_step0020000.pt | 215.8 | 0.600 |
| OBS_NOISE | BASELINE | 17 | BASELINE_OBS_NOISE_seed17_step0020000.pt | 237.4 | 0.667 |
| OBS_NOISE | BASELINE | 42 | BASELINE_OBS_NOISE_seed42_step0010000.pt | 219.1 | 0.600 |
| REWARD_NOISE | AC_FULL | 0 | AC_FULL_REWARD_NOISE_seed0_step0010000_policy.pt | 287.7 | 1.000 |
| REWARD_NOISE | AC_FULL | 3 | AC_FULL_REWARD_NOISE_seed3_step0010000_policy.pt | 286.2 | 1.000 |
| REWARD_NOISE | AC_FULL | 9 | AC_FULL_REWARD_NOISE_seed9_step0010000_policy.pt | 286.6 | 1.000 |
| REWARD_NOISE | AC_FULL | 17 | AC_FULL_REWARD_NOISE_seed17_step0010000_policy.pt | 288.4 | 1.000 |
| REWARD_NOISE | AC_FULL | 42 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE | 0 | AC_LITE_REWARD_NOISE_seed0_step0010000_policy.pt | 287.7 | 1.000 |
| REWARD_NOISE | AC_LITE | 3 | AC_LITE_REWARD_NOISE_seed3_step0010000_policy.pt | 286.2 | 1.000 |
| REWARD_NOISE | AC_LITE | 9 | AC_LITE_REWARD_NOISE_seed9_step0010000_policy.pt | 286.6 | 1.000 |
| REWARD_NOISE | AC_LITE | 17 | AC_LITE_REWARD_NOISE_seed17_step0010000_policy.pt | 288.4 | 1.000 |
| REWARD_NOISE | AC_LITE | 42 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE | 0 | BASELINE_REWARD_NOISE_seed0_final.pt | 288.6 | 1.000 |
| REWARD_NOISE | BASELINE | 3 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE | 9 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE | 17 | BASELINE_REWARD_NOISE_seed17_final.pt | 288.6 | 1.000 |
| REWARD_NOISE | BASELINE | 42 | checkpoint_0_pretrained | 285.9 | 1.000 |

## Certainty AUROC diagnostics

- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.
- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\) (diagnostic).

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL | 0.464 | 0.443 |
| CLEAN | AC_LITE | 0.464 | 0.443 |
| OBS_NOISE | AC_FULL | 0.540 | 0.483 |
| OBS_NOISE | AC_LITE | 0.540 | 0.483 |
| REWARD_NOISE | AC_FULL | 0.483 | 0.446 |
| REWARD_NOISE | AC_LITE | 0.483 | 0.446 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

