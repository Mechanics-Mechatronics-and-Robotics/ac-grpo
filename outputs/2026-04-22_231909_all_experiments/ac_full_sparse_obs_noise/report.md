# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-22_231909_all_experiments\ac_full_sparse_obs_noise`

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
- Challenge tests currently use up to 180 episodes per evaluation seed in the generated logs.

## Summary table (mean ± std over seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| OBS_NOISE | AC_FULL_SPARSE | 164.8 ± 29.6 | 0.520 ± 0.115 | 198.6 | 0.680 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| OBS_NOISE | AC_FULL_SPARSE | 0 | 169.6 | 0.500 |
| OBS_NOISE | AC_FULL_SPARSE | 3 | 171.4 | 0.550 |
| OBS_NOISE | AC_FULL_SPARSE | 9 | 124.2 | 0.400 |
| OBS_NOISE | AC_FULL_SPARSE | 17 | 205.4 | 0.700 |
| OBS_NOISE | AC_FULL_SPARSE | 42 | 153.3 | 0.450 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| OBS_NOISE | AC_FULL_SPARSE | 0 | AC_FULL_SPARSE_OBS_NOISE_seed0_final_policy.pt | OBS_NOISE | 233.7 | 0.733 |
| OBS_NOISE | AC_FULL_SPARSE | 3 | AC_FULL_SPARSE_OBS_NOISE_seed3_final_policy.pt | OBS_NOISE | 240.0 | 0.800 |
| OBS_NOISE | AC_FULL_SPARSE | 9 | checkpoint_0_pretrained | OBS_NOISE | 213.0 | 0.600 |
| OBS_NOISE | AC_FULL_SPARSE | 17 | AC_FULL_SPARSE_OBS_NOISE_seed17_final_policy.pt | OBS_NOISE | 265.3 | 0.867 |
| OBS_NOISE | AC_FULL_SPARSE | 42 | AC_FULL_SPARSE_OBS_NOISE_seed42_step0020000_policy.pt | OBS_NOISE | 218.2 | 0.733 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| OBS_NOISE | AC_FULL_SPARSE | test_clean | CLEAN | 0.10 | 266.9 ± 46.7 | 0.950 ± 0.218 |
| OBS_NOISE | AC_FULL_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 221.0 ± 106.0 | 0.745 ± 0.437 |
| OBS_NOISE | AC_FULL_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 34.8 ± 54.1 | 0.035 ± 0.184 |

## Episode-level certainty summary

| mode | method | mean c_i | corr(c, delta) | corr(c, action prob) | corr(c, runner-up prob) |
|---|---|---:|---:|---:|---:|
| OBS_NOISE | AC_FULL_SPARSE | 0.522 | 0.124 | 0.133 | -0.122 |

## Certainty AUROC diagnostics

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| OBS_NOISE | AC_FULL_SPARSE | 0.602 | 0.429 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- OBS_NOISE: `AC_FULL_SPARSE` has the highest mean final return (164.8) with mean final success 0.520.

Checkpoint selection versus the pretrained anchor:

- OBS_NOISE / AC_FULL_SPARSE: checkpoint 0 wins in 1 of 5 seeds (0.20).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `AC_FULL_SPARSE` in `OBS_NOISE` with 266.9 ± 46.7 and success 0.950 ± 0.218.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `AC_FULL_SPARSE` in `OBS_NOISE` with 221.0 ± 106.0 and success 0.745 ± 0.437.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `AC_FULL_SPARSE` in `OBS_NOISE` with 34.8 ± 54.1 and success 0.035 ± 0.184.

Episode-level certainty behavior:

- OBS_NOISE / AC_FULL_SPARSE: mean episode certainty 0.522, mean corr(certainty, delta) 0.124, mean corr(certainty, action_prob) 0.133, mean corr(certainty, runner_up_prob) -0.122.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

