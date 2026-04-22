# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-22_231909_all_experiments\ac_full_sparse_clean`

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
- Challenge tests currently use up to 200 episodes per evaluation seed in the generated logs.

## Summary table (mean ± std over seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 268.6 ± 17.7 | 0.940 ± 0.089 | 288.1 | 1.000 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0 | 280.5 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 3 | 254.7 | 0.900 |
| CLEAN | AC_FULL_SPARSE | 9 | 280.9 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 17 | 282.5 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 42 | 244.6 | 0.800 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0 | AC_FULL_SPARSE_CLEAN_seed0_final_policy.pt | CLEAN | 288.5 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 3 | AC_FULL_SPARSE_CLEAN_seed3_final_policy.pt | CLEAN | 288.0 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 9 | AC_FULL_SPARSE_CLEAN_seed9_step0010000_policy.pt | CLEAN | 286.2 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 17 | AC_FULL_SPARSE_CLEAN_seed17_step0020000_policy.pt | CLEAN | 289.1 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 42 | AC_FULL_SPARSE_CLEAN_seed42_step0010000_policy.pt | CLEAN | 289.0 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | test_clean | CLEAN | 0.10 | 278.6 ± 19.2 | 1.000 ± 0.000 |
| CLEAN | AC_FULL_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 211.0 ± 108.3 | 0.713 ± 0.453 |
| CLEAN | AC_FULL_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 24.0 ± 35.1 | 0.000 ± 0.000 |

## Episode-level certainty summary

| mode | method | mean c_i | corr(c, delta) | corr(c, action prob) | corr(c, runner-up prob) |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0.534 | 0.097 | 0.112 | -0.100 |

## Certainty AUROC diagnostics

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0.485 | 0.459 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- CLEAN: `AC_FULL_SPARSE` has the highest mean final return (268.6) with mean final success 0.940.

Checkpoint selection versus the pretrained anchor:

- CLEAN / AC_FULL_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `AC_FULL_SPARSE` in `CLEAN` with 278.6 ± 19.2 and success 1.000 ± 0.000.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `AC_FULL_SPARSE` in `CLEAN` with 211.0 ± 108.3 and success 0.713 ± 0.453.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `AC_FULL_SPARSE` in `CLEAN` with 24.0 ± 35.1 and success 0.000 ± 0.000.

Episode-level certainty behavior:

- CLEAN / AC_FULL_SPARSE: mean episode certainty 0.534, mean corr(certainty, delta) 0.097, mean corr(certainty, action_prob) 0.112, mean corr(certainty, runner_up_prob) -0.100.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

