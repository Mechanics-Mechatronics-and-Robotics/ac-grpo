# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-23_002939_all_experiments\ac_lite_sparse_obs_noise`

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
- Challenge tests currently use up to 140 episodes per evaluation seed in the generated logs.

## Summary table (mean ± std over seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| OBS_NOISE | AC_LITE_SPARSE | 104.8 ± 52.5 | 0.300 ± 0.209 | 186.6 | 0.620 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| OBS_NOISE | AC_LITE_SPARSE | 0 | 67.5 | 0.150 |
| OBS_NOISE | AC_LITE_SPARSE | 3 | 58.4 | 0.100 |
| OBS_NOISE | AC_LITE_SPARSE | 9 | 155.3 | 0.500 |
| OBS_NOISE | AC_LITE_SPARSE | 17 | 168.1 | 0.550 |
| OBS_NOISE | AC_LITE_SPARSE | 42 | 74.6 | 0.200 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| OBS_NOISE | AC_LITE_SPARSE | 0 | checkpoint_0_pretrained | OBS_NOISE | 213.0 | 0.600 |
| OBS_NOISE | AC_LITE_SPARSE | 3 | checkpoint_0_pretrained | OBS_NOISE | 213.0 | 0.600 |
| OBS_NOISE | AC_LITE_SPARSE | 9 | AC_LITE_SPARSE_OBS_NOISE_seed9_step0020000_policy.pt | OBS_NOISE | 236.5 | 0.733 |
| OBS_NOISE | AC_LITE_SPARSE | 17 | AC_LITE_SPARSE_OBS_NOISE_seed17_final_policy.pt | OBS_NOISE | 243.8 | 0.867 |
| OBS_NOISE | AC_LITE_SPARSE | 42 | checkpoint_0_pretrained | OBS_NOISE | 213.0 | 0.600 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| OBS_NOISE | AC_LITE_SPARSE | test_clean | CLEAN | 0.10 | 278.1 ± 20.4 | 1.000 ± 0.000 |
| OBS_NOISE | AC_LITE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 215.4 ± 109.3 | 0.723 ± 0.448 |
| OBS_NOISE | AC_LITE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 25.2 ± 41.9 | 0.015 ± 0.122 |

## Episode-level certainty summary

| mode | method | mean c_i | corr(c, delta) | corr(c, action prob) | corr(c, runner-up prob) |
|---|---|---:|---:|---:|---:|
| OBS_NOISE | AC_LITE_SPARSE | 0.575 | -0.046 | -0.050 | 0.043 |

## Certainty AUROC diagnostics

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| OBS_NOISE | AC_LITE_SPARSE | 0.451 | 0.515 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- OBS_NOISE: `AC_LITE_SPARSE` has the highest mean final return (104.8) with mean final success 0.300.

Checkpoint selection versus the pretrained anchor:

- OBS_NOISE / AC_LITE_SPARSE: checkpoint 0 wins in 3 of 5 seeds (0.60).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `AC_LITE_SPARSE` in `OBS_NOISE` with 278.1 ± 20.4 and success 1.000 ± 0.000.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `AC_LITE_SPARSE` in `OBS_NOISE` with 215.4 ± 109.3 and success 0.723 ± 0.448.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `AC_LITE_SPARSE` in `OBS_NOISE` with 25.2 ± 41.9 and success 0.015 ± 0.122.

Episode-level certainty behavior:

- OBS_NOISE / AC_LITE_SPARSE: mean episode certainty 0.575, mean corr(certainty, delta) -0.046, mean corr(certainty, action_prob) -0.050, mean corr(certainty, runner_up_prob) 0.043.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

