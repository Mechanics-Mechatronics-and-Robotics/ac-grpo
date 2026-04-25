# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-23_235329_all_experiments\ac_full_sparse_reward_noise`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Experiment protocol

- Implemented method variants in this run: AC_FULL_SPARSE
- Training modes: REWARD_NOISE
- Reward modes represented: SPARSE
- Training seeds: 0, 3, 9, 17, 42
- All branches start from the shared pretrained anchor when a pretrained path is provided.
- Training uses grouped rollouts with dynamic sampling fallback when no mixed-outcome groups are available.
- Checkpoints are saved during training and the pretrained anchor is treated as checkpoint 0.
- Checkpoint selection uses greedy held-out evaluation under the branch's primary selection condition.
- Additional challenge tests evaluate selected checkpoints under CLEAN, OBS_NOISE (typical), and OBS_NOISE (hard).
- Challenge tests currently use up to 180 episodes per evaluation seed in the generated logs.

## Summary table (mean ± std over seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | reward AUC | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|---:|
| REWARD_NOISE | AC_FULL_SPARSE | 225.9 ± 43.3 | 0.540 ± 0.134 | 231.9 ± 12.7 | 290.2 | 0.950 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| REWARD_NOISE | AC_FULL_SPARSE | 0 | 194.3 | 0.450 |
| REWARD_NOISE | AC_FULL_SPARSE | 3 | 275.1 | 0.600 |
| REWARD_NOISE | AC_FULL_SPARSE | 9 | 228.7 | 0.750 |
| REWARD_NOISE | AC_FULL_SPARSE | 17 | 171.7 | 0.450 |
| REWARD_NOISE | AC_FULL_SPARSE | 42 | 259.7 | 0.450 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| REWARD_NOISE | AC_FULL_SPARSE | 0 | AC_FULL_SPARSE_REWARD_NOISE_seed0_final_policy.pt | REWARD_NOISE | 291.8 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 3 | AC_FULL_SPARSE_REWARD_NOISE_seed3_step0850000_policy.pt | REWARD_NOISE | 286.6 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 9 | AC_FULL_SPARSE_REWARD_NOISE_seed9_step0750000_policy.pt | REWARD_NOISE | 286.2 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 17 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 42 | AC_FULL_SPARSE_REWARD_NOISE_seed42_step0600000_policy.pt | REWARD_NOISE | 293.3 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| REWARD_NOISE | AC_FULL_SPARSE | test_clean | CLEAN | 0.10 | 276.7 ± 25.6 | 0.988 ± 0.111 |
| REWARD_NOISE | AC_FULL_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 95.2 ± 132.4 | 0.210 ± 0.408 |
| REWARD_NOISE | AC_FULL_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | -6.2 ± 72.0 | 0.000 ± 0.000 |

## Cross-test summary on OBS evaluation

| Method | Train: CLEAN → Test OBS | Train: OBS → Test OBS | Train: REWARD → Test OBS |
|---|---:|---:|---:|
| Baseline (Sparse) | n/a | n/a | n/a |
| Baseline (Dense) | n/a | n/a | n/a |
| AC-LITE (Sparse) | n/a | n/a | n/a |
| AC-LITE (Dense) | n/a | n/a | n/a |
| AC-FULL (Sparse) | n/a | n/a | 95.2 ± 132.4 / 0.210 ± 0.408 |

## Episode-level certainty summary

| mode | method | mean c_i | corr(c, delta) | corr(c, action prob) | corr(c, runner-up prob) |
|---|---|---:|---:|---:|---:|
| REWARD_NOISE | AC_FULL_SPARSE | 0.678 | 0.408 | 0.454 | -0.381 |

## Certainty AUROC diagnostics

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| REWARD_NOISE | AC_FULL_SPARSE | 0.577 | 0.290 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- REWARD_NOISE: `AC_FULL_SPARSE` has the highest mean final return (225.9) with mean final success 0.540.

Checkpoint selection versus the pretrained anchor:

- REWARD_NOISE / AC_FULL_SPARSE: checkpoint 0 wins in 1 of 5 seeds (0.20).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `AC_FULL_SPARSE` in `REWARD_NOISE` with 276.7 ± 25.6 and success 0.988 ± 0.111.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `AC_FULL_SPARSE` in `REWARD_NOISE` with 95.2 ± 132.4 and success 0.210 ± 0.408.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `AC_FULL_SPARSE` in `REWARD_NOISE` with -6.2 ± 72.0 and success 0.000 ± 0.000.

Episode-level certainty behavior:

- REWARD_NOISE / AC_FULL_SPARSE: mean episode certainty 0.678, mean corr(certainty, delta) 0.408, mean corr(certainty, action_prob) 0.454, mean corr(certainty, runner_up_prob) -0.381.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

