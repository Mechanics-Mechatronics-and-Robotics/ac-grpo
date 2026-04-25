# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-23_235329_all_experiments\ac_lite_sparse_obs_noise`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Experiment protocol

- Implemented method variants in this run: AC_LITE_SPARSE
- Training modes: OBS_NOISE
- Reward modes represented: SPARSE
- Training seeds: 0, 3, 9, 17, 42
- All branches start from the shared pretrained anchor when a pretrained path is provided.
- Training uses grouped rollouts with dynamic sampling fallback when no mixed-outcome groups are available.
- Checkpoints are saved during training and the pretrained anchor is treated as checkpoint 0.
- Checkpoint selection uses greedy held-out evaluation under the branch's primary selection condition.
- Additional challenge tests evaluate selected checkpoints under CLEAN, OBS_NOISE (typical), and OBS_NOISE (hard).
- Challenge tests currently use up to 200 episodes per evaluation seed in the generated logs.

## Summary table (mean ± std over seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | reward AUC | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|---:|
| OBS_NOISE | AC_LITE_SPARSE | 209.9 ± 28.4 | 0.760 ± 0.114 | 221.8 ± 5.3 | 285.8 | 1.000 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| OBS_NOISE | AC_LITE_SPARSE | 0 | 240.1 | 0.900 |
| OBS_NOISE | AC_LITE_SPARSE | 3 | 174.3 | 0.650 |
| OBS_NOISE | AC_LITE_SPARSE | 9 | 195.3 | 0.750 |
| OBS_NOISE | AC_LITE_SPARSE | 17 | 237.8 | 0.850 |
| OBS_NOISE | AC_LITE_SPARSE | 42 | 202.0 | 0.650 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| OBS_NOISE | AC_LITE_SPARSE | 0 | AC_LITE_SPARSE_OBS_NOISE_seed0_step0400000_policy.pt | OBS_NOISE | 287.7 | 1.000 |
| OBS_NOISE | AC_LITE_SPARSE | 3 | AC_LITE_SPARSE_OBS_NOISE_seed3_step0750000_policy.pt | OBS_NOISE | 281.4 | 1.000 |
| OBS_NOISE | AC_LITE_SPARSE | 9 | AC_LITE_SPARSE_OBS_NOISE_seed9_step0100000_policy.pt | OBS_NOISE | 285.7 | 1.000 |
| OBS_NOISE | AC_LITE_SPARSE | 17 | AC_LITE_SPARSE_OBS_NOISE_seed17_step0700000_policy.pt | OBS_NOISE | 282.5 | 1.000 |
| OBS_NOISE | AC_LITE_SPARSE | 42 | AC_LITE_SPARSE_OBS_NOISE_seed42_step0150000_policy.pt | OBS_NOISE | 276.9 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| OBS_NOISE | AC_LITE_SPARSE | test_clean | CLEAN | 0.10 | 258.2 ± 35.6 | 0.926 ± 0.262 |
| OBS_NOISE | AC_LITE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 262.3 ± 70.6 | 0.950 ± 0.218 |
| OBS_NOISE | AC_LITE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 154.3 ± 131.4 | 0.534 ± 0.499 |

## Cross-test summary on OBS evaluation

| Method | Train: CLEAN → Test OBS | Train: OBS → Test OBS | Train: REWARD → Test OBS |
|---|---:|---:|---:|
| Baseline (Sparse) | n/a | n/a | n/a |
| Baseline (Dense) | n/a | n/a | n/a |
| AC-LITE (Sparse) | n/a | 262.3 ± 70.6 / 0.950 ± 0.218 | n/a |
| AC-LITE (Dense) | n/a | n/a | n/a |
| AC-FULL (Sparse) | n/a | n/a | n/a |

## Episode-level certainty summary

| mode | method | mean c_i | corr(c, delta) | corr(c, action prob) | corr(c, runner-up prob) |
|---|---|---:|---:|---:|---:|
| OBS_NOISE | AC_LITE_SPARSE | 0.910 | 0.298 | 0.324 | -0.290 |

## Certainty AUROC diagnostics

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| OBS_NOISE | AC_LITE_SPARSE | 0.598 | 0.313 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- OBS_NOISE: `AC_LITE_SPARSE` has the highest mean final return (209.9) with mean final success 0.760.

Checkpoint selection versus the pretrained anchor:

- OBS_NOISE / AC_LITE_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `AC_LITE_SPARSE` in `OBS_NOISE` with 258.2 ± 35.6 and success 0.926 ± 0.262.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `AC_LITE_SPARSE` in `OBS_NOISE` with 262.3 ± 70.6 and success 0.950 ± 0.218.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `AC_LITE_SPARSE` in `OBS_NOISE` with 154.3 ± 131.4 and success 0.534 ± 0.499.

Episode-level certainty behavior:

- OBS_NOISE / AC_LITE_SPARSE: mean episode certainty 0.910, mean corr(certainty, delta) 0.298, mean corr(certainty, action_prob) 0.324, mean corr(certainty, runner_up_prob) -0.290.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

