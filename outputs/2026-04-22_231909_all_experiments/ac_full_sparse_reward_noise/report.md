# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-22_231909_all_experiments\ac_full_sparse_reward_noise`

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
- Challenge tests currently use up to 200 episodes per evaluation seed in the generated logs.

## Summary table (mean ± std over seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| REWARD_NOISE | AC_FULL_SPARSE | 232.8 ± 37.3 | 0.640 ± 0.147 | 279.7 | 0.840 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| REWARD_NOISE | AC_FULL_SPARSE | 0 | 281.5 | 0.700 |
| REWARD_NOISE | AC_FULL_SPARSE | 3 | 242.8 | 0.850 |
| REWARD_NOISE | AC_FULL_SPARSE | 9 | 187.6 | 0.500 |
| REWARD_NOISE | AC_FULL_SPARSE | 17 | 204.4 | 0.500 |
| REWARD_NOISE | AC_FULL_SPARSE | 42 | 248.0 | 0.650 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| REWARD_NOISE | AC_FULL_SPARSE | 0 | AC_FULL_SPARSE_REWARD_NOISE_seed0_step0010000_policy.pt | REWARD_NOISE | 287.5 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 3 | AC_FULL_SPARSE_REWARD_NOISE_seed3_step0010000_policy.pt | REWARD_NOISE | 289.5 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 9 | AC_FULL_SPARSE_REWARD_NOISE_seed9_step0020000_policy.pt | REWARD_NOISE | 289.2 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 17 | AC_FULL_SPARSE_REWARD_NOISE_seed17_step0010000_policy.pt | REWARD_NOISE | 288.3 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 42 | AC_FULL_SPARSE_REWARD_NOISE_seed42_final_policy.pt | REWARD_NOISE | 288.1 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| REWARD_NOISE | AC_FULL_SPARSE | test_clean | CLEAN | 0.10 | 275.0 ± 34.1 | 0.975 ± 0.156 |
| REWARD_NOISE | AC_FULL_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 172.7 ± 122.1 | 0.555 ± 0.498 |
| REWARD_NOISE | AC_FULL_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 17.1 ± 32.6 | 0.000 ± 0.000 |

## Episode-level certainty summary

| mode | method | mean c_i | corr(c, delta) | corr(c, action prob) | corr(c, runner-up prob) |
|---|---|---:|---:|---:|---:|
| REWARD_NOISE | AC_FULL_SPARSE | 0.532 | 0.091 | 0.105 | -0.095 |

## Certainty AUROC diagnostics

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| REWARD_NOISE | AC_FULL_SPARSE | 0.413 | 0.459 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- REWARD_NOISE: `AC_FULL_SPARSE` has the highest mean final return (232.8) with mean final success 0.640.

Checkpoint selection versus the pretrained anchor:

- REWARD_NOISE / AC_FULL_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `AC_FULL_SPARSE` in `REWARD_NOISE` with 275.0 ± 34.1 and success 0.975 ± 0.156.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `AC_FULL_SPARSE` in `REWARD_NOISE` with 172.7 ± 122.1 and success 0.555 ± 0.498.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `AC_FULL_SPARSE` in `REWARD_NOISE` with 17.1 ± 32.6 and success 0.000 ± 0.000.

Episode-level certainty behavior:

- REWARD_NOISE / AC_FULL_SPARSE: mean episode certainty 0.532, mean corr(certainty, delta) 0.091, mean corr(certainty, action_prob) 0.105, mean corr(certainty, runner_up_prob) -0.095.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

