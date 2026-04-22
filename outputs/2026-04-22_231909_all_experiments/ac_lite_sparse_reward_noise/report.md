# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-22_231909_all_experiments\ac_lite_sparse_reward_noise`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Experiment protocol

- Implemented method variants in this run: AC_LITE_SPARSE
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

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| REWARD_NOISE | AC_LITE_SPARSE | 248.0 ± 24.2 | 0.670 ± 0.115 | 282.8 | 0.860 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| REWARD_NOISE | AC_LITE_SPARSE | 0 | 268.5 | 0.650 |
| REWARD_NOISE | AC_LITE_SPARSE | 3 | 275.0 | 0.800 |
| REWARD_NOISE | AC_LITE_SPARSE | 9 | 246.6 | 0.750 |
| REWARD_NOISE | AC_LITE_SPARSE | 17 | 233.3 | 0.650 |
| REWARD_NOISE | AC_LITE_SPARSE | 42 | 216.8 | 0.500 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| REWARD_NOISE | AC_LITE_SPARSE | 0 | AC_LITE_SPARSE_REWARD_NOISE_seed0_step0020000_policy.pt | REWARD_NOISE | 288.1 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 3 | AC_LITE_SPARSE_REWARD_NOISE_seed3_step0010000_policy.pt | REWARD_NOISE | 289.2 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 9 | AC_LITE_SPARSE_REWARD_NOISE_seed9_step0020000_policy.pt | REWARD_NOISE | 286.6 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 17 | AC_LITE_SPARSE_REWARD_NOISE_seed17_step0010000_policy.pt | REWARD_NOISE | 289.8 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 42 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| REWARD_NOISE | AC_LITE_SPARSE | test_clean | CLEAN | 0.10 | 278.8 ± 19.4 | 1.000 ± 0.000 |
| REWARD_NOISE | AC_LITE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 166.6 ± 124.2 | 0.526 ± 0.500 |
| REWARD_NOISE | AC_LITE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 12.9 ± 29.1 | 0.000 ± 0.000 |

## Episode-level certainty summary

| mode | method | mean c_i | corr(c, delta) | corr(c, action prob) | corr(c, runner-up prob) |
|---|---|---:|---:|---:|---:|
| REWARD_NOISE | AC_LITE_SPARSE | 0.534 | 0.065 | 0.078 | -0.067 |

## Certainty AUROC diagnostics

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| REWARD_NOISE | AC_LITE_SPARSE | 0.461 | 0.468 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- REWARD_NOISE: `AC_LITE_SPARSE` has the highest mean final return (248.0) with mean final success 0.670.

Checkpoint selection versus the pretrained anchor:

- REWARD_NOISE / AC_LITE_SPARSE: checkpoint 0 wins in 1 of 5 seeds (0.20).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `AC_LITE_SPARSE` in `REWARD_NOISE` with 278.8 ± 19.4 and success 1.000 ± 0.000.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `AC_LITE_SPARSE` in `REWARD_NOISE` with 166.6 ± 124.2 and success 0.526 ± 0.500.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `AC_LITE_SPARSE` in `REWARD_NOISE` with 12.9 ± 29.1 and success 0.000 ± 0.000.

Episode-level certainty behavior:

- REWARD_NOISE / AC_LITE_SPARSE: mean episode certainty 0.534, mean corr(certainty, delta) 0.065, mean corr(certainty, action_prob) 0.078, mean corr(certainty, runner_up_prob) -0.067.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

