# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-23_002939_all_experiments\ac_lite_dense_obs_noise`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Experiment protocol

- Implemented method variants in this run: AC_LITE_DENSE
- Training modes: OBS_NOISE
- Reward modes represented: DENSE
- Training seeds: 0, 3, 9, 17, 42
- All branches start from the shared pretrained anchor when a pretrained path is provided.
- Training uses grouped rollouts with dynamic sampling fallback when no mixed-outcome groups are available.
- Checkpoints are saved during training and the pretrained anchor is treated as checkpoint 0.
- Checkpoint selection uses greedy held-out evaluation under the branch's primary selection condition.
- Additional challenge tests evaluate selected checkpoints under CLEAN, OBS_NOISE (typical), and OBS_NOISE (hard).
- Challenge tests currently use up to 200 episodes per evaluation seed in the generated logs.

## Summary table (mean ± std over seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| OBS_NOISE | AC_LITE_DENSE | 198.4 ± 40.8 | 0.630 ± 0.179 | 245.3 | 0.830 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| OBS_NOISE | AC_LITE_DENSE | 0 | 197.4 | 0.650 |
| OBS_NOISE | AC_LITE_DENSE | 3 | 209.0 | 0.600 |
| OBS_NOISE | AC_LITE_DENSE | 9 | 137.9 | 0.400 |
| OBS_NOISE | AC_LITE_DENSE | 17 | 252.1 | 0.900 |
| OBS_NOISE | AC_LITE_DENSE | 42 | 195.7 | 0.600 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| OBS_NOISE | AC_LITE_DENSE | 0 | AC_LITE_DENSE_OBS_NOISE_seed0_step0020000_policy.pt | OBS_NOISE | 254.8 | 0.800 |
| OBS_NOISE | AC_LITE_DENSE | 3 | AC_LITE_DENSE_OBS_NOISE_seed3_step0020000_policy.pt | OBS_NOISE | 253.8 | 0.800 |
| OBS_NOISE | AC_LITE_DENSE | 9 | AC_LITE_DENSE_OBS_NOISE_seed9_step0010000_policy.pt | OBS_NOISE | 221.6 | 0.667 |
| OBS_NOISE | AC_LITE_DENSE | 17 | AC_LITE_DENSE_OBS_NOISE_seed17_step0020000_policy.pt | OBS_NOISE | 222.7 | 0.600 |
| OBS_NOISE | AC_LITE_DENSE | 42 | AC_LITE_DENSE_OBS_NOISE_seed42_final_policy.pt | OBS_NOISE | 249.0 | 0.800 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| OBS_NOISE | AC_LITE_DENSE | test_clean | CLEAN | 0.10 | 275.6 ± 20.0 | 1.000 ± 0.000 |
| OBS_NOISE | AC_LITE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 252.8 ± 75.0 | 0.873 ± 0.334 |
| OBS_NOISE | AC_LITE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 36.8 ± 49.2 | 0.010 ± 0.100 |

## Episode-level certainty summary

| mode | method | mean c_i | corr(c, delta) | corr(c, action prob) | corr(c, runner-up prob) |
|---|---|---:|---:|---:|---:|
| OBS_NOISE | AC_LITE_DENSE | 0.565 | 0.037 | 0.042 | -0.035 |

## Certainty AUROC diagnostics

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| OBS_NOISE | AC_LITE_DENSE | 0.557 | 0.488 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- OBS_NOISE: `AC_LITE_DENSE` has the highest mean final return (198.4) with mean final success 0.630.

Checkpoint selection versus the pretrained anchor:

- OBS_NOISE / AC_LITE_DENSE: checkpoint 0 wins in 0 of 5 seeds (0.00).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `AC_LITE_DENSE` in `OBS_NOISE` with 275.6 ± 20.0 and success 1.000 ± 0.000.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `AC_LITE_DENSE` in `OBS_NOISE` with 252.8 ± 75.0 and success 0.873 ± 0.334.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `AC_LITE_DENSE` in `OBS_NOISE` with 36.8 ± 49.2 and success 0.010 ± 0.100.

Episode-level certainty behavior:

- OBS_NOISE / AC_LITE_DENSE: mean episode certainty 0.565, mean corr(certainty, delta) 0.037, mean corr(certainty, action_prob) 0.042, mean corr(certainty, runner_up_prob) -0.035.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

