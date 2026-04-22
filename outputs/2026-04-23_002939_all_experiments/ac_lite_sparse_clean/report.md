# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-23_002939_all_experiments\ac_lite_sparse_clean`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Experiment protocol

- Implemented method variants in this run: AC_LITE_SPARSE
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
| CLEAN | AC_LITE_SPARSE | 262.8 ± 6.5 | 0.920 ± 0.027 | 287.4 | 1.000 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_LITE_SPARSE | 0 | 254.5 | 0.900 |
| CLEAN | AC_LITE_SPARSE | 3 | 264.4 | 0.950 |
| CLEAN | AC_LITE_SPARSE | 9 | 267.2 | 0.900 |
| CLEAN | AC_LITE_SPARSE | 17 | 257.7 | 0.900 |
| CLEAN | AC_LITE_SPARSE | 42 | 270.0 | 0.950 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| CLEAN | AC_LITE_SPARSE | 0 | AC_LITE_SPARSE_CLEAN_seed0_step0010000_policy.pt | CLEAN | 287.3 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 3 | AC_LITE_SPARSE_CLEAN_seed3_step0010000_policy.pt | CLEAN | 288.0 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 9 | AC_LITE_SPARSE_CLEAN_seed9_final_policy.pt | CLEAN | 287.9 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 17 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 42 | AC_LITE_SPARSE_CLEAN_seed42_step0020000_policy.pt | CLEAN | 286.0 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| CLEAN | AC_LITE_SPARSE | test_clean | CLEAN | 0.10 | 272.9 ± 31.8 | 0.988 ± 0.111 |
| CLEAN | AC_LITE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 222.3 ± 101.9 | 0.770 ± 0.421 |
| CLEAN | AC_LITE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 24.8 ± 37.3 | 0.003 ± 0.050 |

## Episode-level certainty summary

| mode | method | mean c_i | corr(c, delta) | corr(c, action prob) | corr(c, runner-up prob) |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_LITE_SPARSE | 0.565 | 0.151 | 0.170 | -0.150 |

## Certainty AUROC diagnostics

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_LITE_SPARSE | 0.480 | 0.462 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- CLEAN: `AC_LITE_SPARSE` has the highest mean final return (262.8) with mean final success 0.920.

Checkpoint selection versus the pretrained anchor:

- CLEAN / AC_LITE_SPARSE: checkpoint 0 wins in 1 of 5 seeds (0.20).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `AC_LITE_SPARSE` in `CLEAN` with 272.9 ± 31.8 and success 0.988 ± 0.111.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `AC_LITE_SPARSE` in `CLEAN` with 222.3 ± 101.9 and success 0.770 ± 0.421.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `AC_LITE_SPARSE` in `CLEAN` with 24.8 ± 37.3 and success 0.003 ± 0.050.

Episode-level certainty behavior:

- CLEAN / AC_LITE_SPARSE: mean episode certainty 0.565, mean corr(certainty, delta) 0.151, mean corr(certainty, action_prob) 0.170, mean corr(certainty, runner_up_prob) -0.150.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

