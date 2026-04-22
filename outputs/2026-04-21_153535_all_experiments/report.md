# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-21_153535_all_experiments`

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
| CLEAN | AC_FULL | 253.4 ± 34.2 | 0.890 ± 0.147 | 286.7 | 1.000 |
| CLEAN | AC_LITE | 253.4 ± 34.2 | 0.890 ± 0.147 | 286.7 | 1.000 |
| CLEAN | BASELINE | 240.4 ± 22.8 | 0.830 ± 0.084 | 289.4 | 1.000 |
| OBS_NOISE | AC_FULL | 256.5 ± 15.4 | 0.900 ± 0.061 | 285.9 | 1.000 |
| OBS_NOISE | AC_LITE | 256.5 ± 15.4 | 0.900 ± 0.061 | 285.9 | 1.000 |
| OBS_NOISE | BASELINE | 240.8 ± 22.8 | 0.870 ± 0.104 | 282.7 | 1.000 |
| REWARD_NOISE | AC_FULL | 228.9 ± 26.6 | 0.620 ± 0.205 | 288.6 | 0.930 |
| REWARD_NOISE | AC_LITE | 228.9 ± 26.6 | 0.620 ± 0.205 | 288.6 | 0.930 |
| REWARD_NOISE | BASELINE | 212.9 ± 23.9 | 0.460 ± 0.074 | 287.8 | 0.970 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL | 0 | 271.4 | 1.000 |
| CLEAN | AC_FULL | 3 | 198.3 | 0.650 |
| CLEAN | AC_FULL | 9 | 242.6 | 0.850 |
| CLEAN | AC_FULL | 17 | 273.8 | 0.950 |
| CLEAN | AC_FULL | 42 | 281.2 | 1.000 |
| CLEAN | AC_LITE | 0 | 271.4 | 1.000 |
| CLEAN | AC_LITE | 3 | 198.3 | 0.650 |
| CLEAN | AC_LITE | 9 | 242.6 | 0.850 |
| CLEAN | AC_LITE | 17 | 273.8 | 0.950 |
| CLEAN | AC_LITE | 42 | 281.2 | 1.000 |
| CLEAN | BASELINE | 0 | 234.1 | 0.850 |
| CLEAN | BASELINE | 3 | 220.0 | 0.750 |
| CLEAN | BASELINE | 9 | 222.7 | 0.750 |
| CLEAN | BASELINE | 17 | 275.5 | 0.950 |
| CLEAN | BASELINE | 42 | 249.6 | 0.850 |
| OBS_NOISE | AC_FULL | 0 | 237.7 | 0.850 |
| OBS_NOISE | AC_FULL | 3 | 259.5 | 0.900 |
| OBS_NOISE | AC_FULL | 9 | 253.6 | 0.900 |
| OBS_NOISE | AC_FULL | 17 | 251.5 | 0.850 |
| OBS_NOISE | AC_FULL | 42 | 280.0 | 1.000 |
| OBS_NOISE | AC_LITE | 0 | 237.7 | 0.850 |
| OBS_NOISE | AC_LITE | 3 | 259.5 | 0.900 |
| OBS_NOISE | AC_LITE | 9 | 253.6 | 0.900 |
| OBS_NOISE | AC_LITE | 17 | 251.5 | 0.850 |
| OBS_NOISE | AC_LITE | 42 | 280.0 | 1.000 |
| OBS_NOISE | BASELINE | 0 | 268.0 | 0.950 |
| OBS_NOISE | BASELINE | 3 | 212.2 | 0.700 |
| OBS_NOISE | BASELINE | 9 | 251.7 | 0.900 |
| OBS_NOISE | BASELINE | 17 | 222.5 | 0.850 |
| OBS_NOISE | BASELINE | 42 | 249.4 | 0.950 |
| REWARD_NOISE | AC_FULL | 0 | 218.3 | 0.400 |
| REWARD_NOISE | AC_FULL | 3 | 249.0 | 0.700 |
| REWARD_NOISE | AC_FULL | 9 | 186.9 | 0.400 |
| REWARD_NOISE | AC_FULL | 17 | 249.1 | 0.800 |
| REWARD_NOISE | AC_FULL | 42 | 241.2 | 0.800 |
| REWARD_NOISE | AC_LITE | 0 | 218.3 | 0.400 |
| REWARD_NOISE | AC_LITE | 3 | 249.0 | 0.700 |
| REWARD_NOISE | AC_LITE | 9 | 186.9 | 0.400 |
| REWARD_NOISE | AC_LITE | 17 | 249.1 | 0.800 |
| REWARD_NOISE | AC_LITE | 42 | 241.2 | 0.800 |
| REWARD_NOISE | BASELINE | 0 | 236.1 | 0.450 |
| REWARD_NOISE | BASELINE | 3 | 189.1 | 0.350 |
| REWARD_NOISE | BASELINE | 9 | 237.9 | 0.500 |
| REWARD_NOISE | BASELINE | 17 | 212.2 | 0.450 |
| REWARD_NOISE | BASELINE | 42 | 189.4 | 0.550 |

## Best checkpoint by greedy held-out evaluation

Checkpoints are evaluated greedily on fixed held-out seeds. The final checkpoint is not assumed to be best.

| mode | method | seed | checkpoint | eval return | eval success |
|---|---|---:|---|---:|---:|
| CLEAN | AC_FULL | 0 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_FULL | 3 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_FULL | 9 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_FULL | 17 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_FULL | 42 | AC_FULL_CLEAN_seed42_step0230000_policy.pt | 288.4 | 1.000 |
| CLEAN | AC_LITE | 0 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_LITE | 3 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_LITE | 9 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_LITE | 17 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_LITE | 42 | AC_LITE_CLEAN_seed42_step0230000_policy.pt | 288.4 | 1.000 |
| CLEAN | BASELINE | 0 | BASELINE_CLEAN_seed0_step0010000.pt | 287.9 | 1.000 |
| CLEAN | BASELINE | 3 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | BASELINE | 9 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | BASELINE | 17 | BASELINE_CLEAN_seed17_step0020000.pt | 288.4 | 1.000 |
| CLEAN | BASELINE | 42 | checkpoint_0_pretrained | 285.9 | 1.000 |
| OBS_NOISE | AC_FULL | 0 | AC_FULL_OBS_NOISE_seed0_step0190000_policy.pt | 282.4 | 1.000 |
| OBS_NOISE | AC_FULL | 3 | AC_FULL_OBS_NOISE_seed3_step0090000_policy.pt | 282.8 | 1.000 |
| OBS_NOISE | AC_FULL | 9 | AC_FULL_OBS_NOISE_seed9_step0130000_policy.pt | 287.1 | 1.000 |
| OBS_NOISE | AC_FULL | 17 | AC_FULL_OBS_NOISE_seed17_step0120000_policy.pt | 285.7 | 1.000 |
| OBS_NOISE | AC_FULL | 42 | AC_FULL_OBS_NOISE_seed42_step0090000_policy.pt | 283.1 | 1.000 |
| OBS_NOISE | AC_LITE | 0 | AC_LITE_OBS_NOISE_seed0_step0190000_policy.pt | 282.4 | 1.000 |
| OBS_NOISE | AC_LITE | 3 | AC_LITE_OBS_NOISE_seed3_step0090000_policy.pt | 282.8 | 1.000 |
| OBS_NOISE | AC_LITE | 9 | AC_LITE_OBS_NOISE_seed9_step0130000_policy.pt | 287.1 | 1.000 |
| OBS_NOISE | AC_LITE | 17 | AC_LITE_OBS_NOISE_seed17_step0120000_policy.pt | 285.7 | 1.000 |
| OBS_NOISE | AC_LITE | 42 | AC_LITE_OBS_NOISE_seed42_step0090000_policy.pt | 283.1 | 1.000 |
| OBS_NOISE | BASELINE | 0 | BASELINE_OBS_NOISE_seed0_step0110000.pt | 283.8 | 1.000 |
| OBS_NOISE | BASELINE | 3 | BASELINE_OBS_NOISE_seed3_step0040000.pt | 275.1 | 0.933 |
| OBS_NOISE | BASELINE | 9 | BASELINE_OBS_NOISE_seed9_step0170000.pt | 278.0 | 1.000 |
| OBS_NOISE | BASELINE | 17 | BASELINE_OBS_NOISE_seed17_step0090000.pt | 282.2 | 1.000 |
| OBS_NOISE | BASELINE | 42 | BASELINE_OBS_NOISE_seed42_step0090000.pt | 280.3 | 1.000 |
| REWARD_NOISE | AC_FULL | 0 | AC_FULL_REWARD_NOISE_seed0_step0010000_policy.pt | 287.7 | 1.000 |
| REWARD_NOISE | AC_FULL | 3 | AC_FULL_REWARD_NOISE_seed3_step0010000_policy.pt | 286.2 | 1.000 |
| REWARD_NOISE | AC_FULL | 9 | AC_FULL_REWARD_NOISE_seed9_step0010000_policy.pt | 286.6 | 1.000 |
| REWARD_NOISE | AC_FULL | 17 | AC_FULL_REWARD_NOISE_seed17_step0010000_policy.pt | 288.4 | 1.000 |
| REWARD_NOISE | AC_FULL | 42 | AC_FULL_REWARD_NOISE_seed42_step0020000_policy.pt | 287.8 | 1.000 |
| REWARD_NOISE | AC_LITE | 0 | AC_LITE_REWARD_NOISE_seed0_step0010000_policy.pt | 287.7 | 1.000 |
| REWARD_NOISE | AC_LITE | 3 | AC_LITE_REWARD_NOISE_seed3_step0010000_policy.pt | 286.2 | 1.000 |
| REWARD_NOISE | AC_LITE | 9 | AC_LITE_REWARD_NOISE_seed9_step0010000_policy.pt | 286.6 | 1.000 |
| REWARD_NOISE | AC_LITE | 17 | AC_LITE_REWARD_NOISE_seed17_step0010000_policy.pt | 288.4 | 1.000 |
| REWARD_NOISE | AC_LITE | 42 | AC_LITE_REWARD_NOISE_seed42_step0020000_policy.pt | 287.8 | 1.000 |
| REWARD_NOISE | BASELINE | 0 | BASELINE_REWARD_NOISE_seed0_step0010000.pt | 287.9 | 1.000 |
| REWARD_NOISE | BASELINE | 3 | BASELINE_REWARD_NOISE_seed3_step0050000.pt | 288.0 | 1.000 |
| REWARD_NOISE | BASELINE | 9 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE | 17 | BASELINE_REWARD_NOISE_seed17_step0030000.pt | 288.3 | 1.000 |
| REWARD_NOISE | BASELINE | 42 | checkpoint_0_pretrained | 285.9 | 1.000 |

## Certainty AUROC diagnostics

- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.
- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\) (diagnostic).

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL | 0.363 | 0.424 |
| CLEAN | AC_LITE | 0.363 | 0.424 |
| OBS_NOISE | AC_FULL | 0.660 | 0.442 |
| OBS_NOISE | AC_LITE | 0.660 | 0.442 |
| REWARD_NOISE | AC_FULL | 0.430 | 0.437 |
| REWARD_NOISE | AC_LITE | 0.430 | 0.437 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

