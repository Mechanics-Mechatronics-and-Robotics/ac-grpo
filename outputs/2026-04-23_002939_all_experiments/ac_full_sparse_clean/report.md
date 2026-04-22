# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-23_002939_all_experiments\ac_full_sparse_clean`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Experiment protocol

- Implemented method variants in this run: AC_FULL_SPARSE
- Training modes: CLEAN
- Reward modes represented: SPARSE
- Training seeds: 0, 3, 9, 17, 42
- All branches start from the shared pretrained anchor when a pretrained path is provided.
- Training uses grouped rollouts with dynamic sampling fallback when no mixed-outcome groups are available.
- Checkpoints are saved during training and the pretrained anchor is treated as checkpoint 0.
- Checkpoint selection uses greedy held-out evaluation under the branch's primary selection condition.
- Additional challenge tests evaluate selected checkpoints under CLEAN, OBS_NOISE (typical), and OBS_NOISE (hard).
- Challenge tests currently use up to 180 episodes per evaluation seed in the generated logs.

## Summary table (mean ± std over seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 269.7 ± 12.2 | 0.940 ± 0.055 | 282.2 | 0.990 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0 | 280.3 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 3 | 264.5 | 0.900 |
| CLEAN | AC_FULL_SPARSE | 9 | 284.6 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 17 | 262.6 | 0.900 |
| CLEAN | AC_FULL_SPARSE | 42 | 256.2 | 0.900 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0 | AC_FULL_SPARSE_CLEAN_seed0_step0010000_policy.pt | CLEAN | 286.1 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 3 | AC_FULL_SPARSE_CLEAN_seed3_step0020000_policy.pt | CLEAN | 286.4 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 9 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 17 | AC_FULL_SPARSE_CLEAN_seed17_step0010000_policy.pt | CLEAN | 287.0 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 42 | AC_FULL_SPARSE_CLEAN_seed42_step0010000_policy.pt | CLEAN | 288.8 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | test_clean | CLEAN | 0.10 | 277.8 ± 20.6 | 1.000 ± 0.000 |
| CLEAN | AC_FULL_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 206.4 ± 110.7 | 0.690 ± 0.463 |
| CLEAN | AC_FULL_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 23.9 ± 35.2 | 0.000 ± 0.000 |

## Episode-level certainty summary

| mode | method | mean c_i | corr(c, delta) | corr(c, action prob) | corr(c, runner-up prob) |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0.564 | 0.147 | 0.167 | -0.147 |

## Certainty AUROC diagnostics

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0.491 | 0.465 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- CLEAN: `AC_FULL_SPARSE` has the highest mean final return (269.7) with mean final success 0.940.

Checkpoint selection versus the pretrained anchor:

- CLEAN / AC_FULL_SPARSE: checkpoint 0 wins in 1 of 5 seeds (0.20).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `AC_FULL_SPARSE` in `CLEAN` with 277.8 ± 20.6 and success 1.000 ± 0.000.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `AC_FULL_SPARSE` in `CLEAN` with 206.4 ± 110.7 and success 0.690 ± 0.463.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `AC_FULL_SPARSE` in `CLEAN` with 23.9 ± 35.2 and success 0.000 ± 0.000.

Episode-level certainty behavior:

- CLEAN / AC_FULL_SPARSE: mean episode certainty 0.564, mean corr(certainty, delta) 0.147, mean corr(certainty, action_prob) 0.167, mean corr(certainty, runner_up_prob) -0.147.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

