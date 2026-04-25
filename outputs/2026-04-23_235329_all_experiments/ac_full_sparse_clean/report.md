# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-23_235329_all_experiments\ac_full_sparse_clean`

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

| mode | method | final return (last 20 eps) | final success (last 20 eps) | reward AUC | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 273.4 ± 14.9 | 0.940 ± 0.082 | 266.1 ± 5.8 | 293.7 | 1.000 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0 | 280.7 | 0.950 |
| CLEAN | AC_FULL_SPARSE | 3 | 285.9 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 9 | 247.8 | 0.800 |
| CLEAN | AC_FULL_SPARSE | 17 | 274.9 | 0.950 |
| CLEAN | AC_FULL_SPARSE | 42 | 277.9 | 1.000 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0 | AC_FULL_SPARSE_CLEAN_seed0_step0400000_policy.pt | CLEAN | 287.7 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 3 | AC_FULL_SPARSE_CLEAN_seed3_step0600000_policy.pt | CLEAN | 289.0 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 9 | AC_FULL_SPARSE_CLEAN_seed9_step0650000_policy.pt | CLEAN | 288.0 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 17 | AC_FULL_SPARSE_CLEAN_seed17_step0650000_policy.pt | CLEAN | 289.9 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 42 | AC_FULL_SPARSE_CLEAN_seed42_step0300000_policy.pt | CLEAN | 286.9 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | test_clean | CLEAN | 0.10 | 275.0 ± 29.2 | 0.980 ± 0.140 |
| CLEAN | AC_FULL_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 156.8 ± 81.7 | 0.308 ± 0.462 |
| CLEAN | AC_FULL_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 36.0 ± 49.5 | 0.000 ± 0.000 |

## Cross-test summary on OBS evaluation

| Method | Train: CLEAN → Test OBS | Train: OBS → Test OBS | Train: REWARD → Test OBS |
|---|---:|---:|---:|
| Baseline (Sparse) | n/a | n/a | n/a |
| Baseline (Dense) | n/a | n/a | n/a |
| AC-LITE (Sparse) | n/a | n/a | n/a |
| AC-LITE (Dense) | n/a | n/a | n/a |
| AC-FULL (Sparse) | 156.8 ± 81.7 / 0.308 ± 0.462 | n/a | n/a |

## Episode-level certainty summary

| mode | method | mean c_i | corr(c, delta) | corr(c, action prob) | corr(c, runner-up prob) |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0.867 | -0.079 | -0.071 | 0.092 |

## Certainty AUROC diagnostics

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0.706 | 0.613 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- CLEAN: `AC_FULL_SPARSE` has the highest mean final return (273.4) with mean final success 0.940.

Checkpoint selection versus the pretrained anchor:

- CLEAN / AC_FULL_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `AC_FULL_SPARSE` in `CLEAN` with 275.0 ± 29.2 and success 0.980 ± 0.140.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `AC_FULL_SPARSE` in `CLEAN` with 156.8 ± 81.7 and success 0.308 ± 0.462.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `AC_FULL_SPARSE` in `CLEAN` with 36.0 ± 49.5 and success 0.000 ± 0.000.

Episode-level certainty behavior:

- CLEAN / AC_FULL_SPARSE: mean episode certainty 0.867, mean corr(certainty, delta) -0.079, mean corr(certainty, action_prob) -0.071, mean corr(certainty, runner_up_prob) 0.092.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

