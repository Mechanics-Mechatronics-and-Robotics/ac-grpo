# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-23_013020_all_experiments\ac_full_sparse_obs_noise`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Experiment protocol

- Implemented method variants in this run: AC_FULL_SPARSE
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

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| OBS_NOISE | AC_FULL_SPARSE | 194.5 ± 47.7 | 0.710 ± 0.119 | 284.8 | 1.000 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| OBS_NOISE | AC_FULL_SPARSE | 0 | 224.6 | 0.800 |
| OBS_NOISE | AC_FULL_SPARSE | 3 | 153.7 | 0.650 |
| OBS_NOISE | AC_FULL_SPARSE | 9 | 134.9 | 0.550 |
| OBS_NOISE | AC_FULL_SPARSE | 17 | 245.2 | 0.850 |
| OBS_NOISE | AC_FULL_SPARSE | 42 | 214.0 | 0.700 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| OBS_NOISE | AC_FULL_SPARSE | 0 | AC_FULL_SPARSE_OBS_NOISE_seed0_step0330000_policy.pt | OBS_NOISE | 286.3 | 1.000 |
| OBS_NOISE | AC_FULL_SPARSE | 3 | AC_FULL_SPARSE_OBS_NOISE_seed3_step0080000_policy.pt | OBS_NOISE | 283.7 | 1.000 |
| OBS_NOISE | AC_FULL_SPARSE | 9 | AC_FULL_SPARSE_OBS_NOISE_seed9_step0430000_policy.pt | OBS_NOISE | 288.2 | 1.000 |
| OBS_NOISE | AC_FULL_SPARSE | 17 | AC_FULL_SPARSE_OBS_NOISE_seed17_step0120000_policy.pt | OBS_NOISE | 289.6 | 1.000 |
| OBS_NOISE | AC_FULL_SPARSE | 42 | AC_FULL_SPARSE_OBS_NOISE_seed42_step0260000_policy.pt | OBS_NOISE | 288.7 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| OBS_NOISE | AC_FULL_SPARSE | test_clean | CLEAN | 0.10 | 253.5 ± 65.0 | 0.884 ± 0.321 |
| OBS_NOISE | AC_FULL_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 266.7 ± 53.5 | 0.956 ± 0.205 |
| OBS_NOISE | AC_FULL_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 162.6 ± 131.6 | 0.568 ± 0.496 |

## Episode-level certainty summary

| mode | method | mean c_i | corr(c, delta) | corr(c, action prob) | corr(c, runner-up prob) |
|---|---|---:|---:|---:|---:|
| OBS_NOISE | AC_FULL_SPARSE | 0.717 | 0.358 | 0.390 | -0.346 |

## Certainty AUROC diagnostics

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| OBS_NOISE | AC_FULL_SPARSE | 0.740 | 0.311 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- OBS_NOISE: `AC_FULL_SPARSE` has the highest mean final return (194.5) with mean final success 0.710.

Checkpoint selection versus the pretrained anchor:

- OBS_NOISE / AC_FULL_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `AC_FULL_SPARSE` in `OBS_NOISE` with 253.5 ± 65.0 and success 0.884 ± 0.321.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `AC_FULL_SPARSE` in `OBS_NOISE` with 266.7 ± 53.5 and success 0.956 ± 0.205.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `AC_FULL_SPARSE` in `OBS_NOISE` with 162.6 ± 131.6 and success 0.568 ± 0.496.

Episode-level certainty behavior:

- OBS_NOISE / AC_FULL_SPARSE: mean episode certainty 0.717, mean corr(certainty, delta) 0.358, mean corr(certainty, action_prob) 0.390, mean corr(certainty, runner_up_prob) -0.346.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

