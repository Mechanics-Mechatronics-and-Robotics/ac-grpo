# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-23_002939_all_experiments\ac_full_sparse_obs_noise`

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
- Challenge tests currently use up to 160 episodes per evaluation seed in the generated logs.

## Summary table (mean ± std over seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| OBS_NOISE | AC_FULL_SPARSE | 107.0 ± 17.7 | 0.340 ± 0.074 | 175.7 | 0.590 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| OBS_NOISE | AC_FULL_SPARSE | 0 | 134.8 | 0.450 |
| OBS_NOISE | AC_FULL_SPARSE | 3 | 97.9 | 0.350 |
| OBS_NOISE | AC_FULL_SPARSE | 9 | 98.9 | 0.300 |
| OBS_NOISE | AC_FULL_SPARSE | 17 | 113.7 | 0.350 |
| OBS_NOISE | AC_FULL_SPARSE | 42 | 89.9 | 0.250 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| OBS_NOISE | AC_FULL_SPARSE | 0 | AC_FULL_SPARSE_OBS_NOISE_seed0_step0020000_policy.pt | OBS_NOISE | 228.1 | 0.800 |
| OBS_NOISE | AC_FULL_SPARSE | 3 | checkpoint_0_pretrained | OBS_NOISE | 213.0 | 0.600 |
| OBS_NOISE | AC_FULL_SPARSE | 9 | checkpoint_0_pretrained | OBS_NOISE | 213.0 | 0.600 |
| OBS_NOISE | AC_FULL_SPARSE | 17 | AC_FULL_SPARSE_OBS_NOISE_seed17_final_policy.pt | OBS_NOISE | 280.9 | 1.000 |
| OBS_NOISE | AC_FULL_SPARSE | 42 | AC_FULL_SPARSE_OBS_NOISE_seed42_final_policy.pt | OBS_NOISE | 226.0 | 0.800 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| OBS_NOISE | AC_FULL_SPARSE | test_clean | CLEAN | 0.10 | 276.6 ± 27.0 | 0.977 ± 0.151 |
| OBS_NOISE | AC_FULL_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 207.3 ± 113.5 | 0.687 ± 0.465 |
| OBS_NOISE | AC_FULL_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 26.7 ± 42.3 | 0.017 ± 0.128 |

## Episode-level certainty summary

| mode | method | mean c_i | corr(c, delta) | corr(c, action prob) | corr(c, runner-up prob) |
|---|---|---:|---:|---:|---:|
| OBS_NOISE | AC_FULL_SPARSE | 0.526 | 0.176 | 0.191 | -0.171 |

## Certainty AUROC diagnostics

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| OBS_NOISE | AC_FULL_SPARSE | 0.664 | 0.376 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- OBS_NOISE: `AC_FULL_SPARSE` has the highest mean final return (107.0) with mean final success 0.340.

Checkpoint selection versus the pretrained anchor:

- OBS_NOISE / AC_FULL_SPARSE: checkpoint 0 wins in 2 of 5 seeds (0.40).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `AC_FULL_SPARSE` in `OBS_NOISE` with 276.6 ± 27.0 and success 0.977 ± 0.151.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `AC_FULL_SPARSE` in `OBS_NOISE` with 207.3 ± 113.5 and success 0.687 ± 0.465.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `AC_FULL_SPARSE` in `OBS_NOISE` with 26.7 ± 42.3 and success 0.017 ± 0.128.

Episode-level certainty behavior:

- OBS_NOISE / AC_FULL_SPARSE: mean episode certainty 0.526, mean corr(certainty, delta) 0.176, mean corr(certainty, action_prob) 0.191, mean corr(certainty, runner_up_prob) -0.171.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

