# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-24_183542_ac_lite_sparse_clean`

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
- Challenge tests currently use up to 200 episodes per evaluation seed in the generated logs.

## Summary table (mean ± std over seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | reward AUC | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|---:|
| CLEAN | AC_LITE_SPARSE | 212.5 ± 35.1 | 0.700 ± 0.132 | 247.0 ± 9.2 | 289.2 | 1.000 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_LITE_SPARSE | 0 | 258.8 | 0.900 |
| CLEAN | AC_LITE_SPARSE | 3 | 228.2 | 0.750 |
| CLEAN | AC_LITE_SPARSE | 9 | 182.3 | 0.650 |
| CLEAN | AC_LITE_SPARSE | 17 | 220.1 | 0.650 |
| CLEAN | AC_LITE_SPARSE | 42 | 172.9 | 0.550 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| CLEAN | AC_LITE_SPARSE | 0 | AC_LITE_SPARSE_CLEAN_seed0_step0020000_policy.pt | CLEAN | 288.0 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 3 | AC_LITE_SPARSE_CLEAN_seed3_step0210000_policy.pt | CLEAN | 289.4 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 9 | AC_LITE_SPARSE_CLEAN_seed9_step0030000_policy.pt | CLEAN | 289.7 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 17 | AC_LITE_SPARSE_CLEAN_seed17_step0010000_policy.pt | CLEAN | 289.6 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 42 | AC_LITE_SPARSE_CLEAN_seed42_step0050000_policy.pt | CLEAN | 289.2 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| CLEAN | AC_LITE_SPARSE | test_clean | CLEAN | 0.10 | 278.8 ± 19.4 | 1.000 ± 0.000 |
| CLEAN | AC_LITE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 200.1 ± 112.2 | 0.654 ± 0.476 |
| CLEAN | AC_LITE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 21.4 ± 32.9 | 0.000 ± 0.000 |

## Cross-test summary on OBS evaluation

| Method | Train: CLEAN → Test OBS | Train: OBS → Test OBS | Train: REWARD → Test OBS |
|---|---:|---:|---:|
| Baseline (Sparse) | n/a | n/a | n/a |
| Baseline (Dense) | n/a | n/a | n/a |
| AC-LITE (Sparse) | 200.1 ± 112.2 / 0.654 ± 0.476 | n/a | n/a |
| AC-LITE (Dense) | n/a | n/a | n/a |
| AC-FULL (Sparse) | n/a | n/a | n/a |

## Episode-level certainty summary

| mode | method | mean c_i | corr(c, delta) | corr(c, action prob) | corr(c, runner-up prob) |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_LITE_SPARSE | 0.771 | 0.176 | 0.206 | -0.171 |

## Certainty AUROC diagnostics

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_LITE_SPARSE | 0.382 | 0.447 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- CLEAN: `AC_LITE_SPARSE` has the highest mean final return (212.5) with mean final success 0.700.

Checkpoint selection versus the pretrained anchor:

- CLEAN / AC_LITE_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `AC_LITE_SPARSE` in `CLEAN` with 278.8 ± 19.4 and success 1.000 ± 0.000.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `AC_LITE_SPARSE` in `CLEAN` with 200.1 ± 112.2 and success 0.654 ± 0.476.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `AC_LITE_SPARSE` in `CLEAN` with 21.4 ± 32.9 and success 0.000 ± 0.000.

Episode-level certainty behavior:

- CLEAN / AC_LITE_SPARSE: mean episode certainty 0.771, mean corr(certainty, delta) 0.176, mean corr(certainty, action_prob) 0.206, mean corr(certainty, runner_up_prob) -0.171.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

