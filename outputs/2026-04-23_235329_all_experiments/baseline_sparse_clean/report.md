# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-23_235329_all_experiments\baseline_sparse_clean`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Experiment protocol

- Implemented method variants in this run: BASELINE_SPARSE
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
| CLEAN | BASELINE_SPARSE | 272.1 ± 11.3 | 0.930 ± 0.045 | 249.6 ± 4.1 | 294.2 | 1.000 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | BASELINE_SPARSE | 0 | 262.1 | 0.900 |
| CLEAN | BASELINE_SPARSE | 3 | 282.0 | 0.950 |
| CLEAN | BASELINE_SPARSE | 9 | 261.9 | 0.900 |
| CLEAN | BASELINE_SPARSE | 17 | 286.2 | 1.000 |
| CLEAN | BASELINE_SPARSE | 42 | 268.3 | 0.900 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| CLEAN | BASELINE_SPARSE | 0 | BASELINE_SPARSE_CLEAN_seed0_step0900000.pt | CLEAN | 290.7 | 1.000 |
| CLEAN | BASELINE_SPARSE | 3 | BASELINE_SPARSE_CLEAN_seed3_step0750000.pt | CLEAN | 288.7 | 1.000 |
| CLEAN | BASELINE_SPARSE | 9 | BASELINE_SPARSE_CLEAN_seed9_step0850000.pt | CLEAN | 289.7 | 1.000 |
| CLEAN | BASELINE_SPARSE | 17 | BASELINE_SPARSE_CLEAN_seed17_step0900000.pt | CLEAN | 292.0 | 1.000 |
| CLEAN | BASELINE_SPARSE | 42 | BASELINE_SPARSE_CLEAN_seed42_step0950000.pt | CLEAN | 289.4 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| CLEAN | BASELINE_SPARSE | test_clean | CLEAN | 0.10 | 274.0 ± 30.5 | 0.968 ± 0.176 |
| CLEAN | BASELINE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 79.7 ± 82.1 | 0.096 ± 0.295 |
| CLEAN | BASELINE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | -2.1 ± 32.5 | 0.000 ± 0.000 |

## Cross-test summary on OBS evaluation

| Method | Train: CLEAN → Test OBS | Train: OBS → Test OBS | Train: REWARD → Test OBS |
|---|---:|---:|---:|
| Baseline (Sparse) | 79.7 ± 82.1 / 0.096 ± 0.295 | n/a | n/a |
| Baseline (Dense) | n/a | n/a | n/a |
| AC-LITE (Sparse) | n/a | n/a | n/a |
| AC-LITE (Dense) | n/a | n/a | n/a |
| AC-FULL (Sparse) | n/a | n/a | n/a |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- CLEAN: `BASELINE_SPARSE` has the highest mean final return (272.1) with mean final success 0.930.

Checkpoint selection versus the pretrained anchor:

- CLEAN / BASELINE_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `BASELINE_SPARSE` in `CLEAN` with 274.0 ± 30.5 and success 0.968 ± 0.176.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `BASELINE_SPARSE` in `CLEAN` with 79.7 ± 82.1 and success 0.096 ± 0.295.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `BASELINE_SPARSE` in `CLEAN` with -2.1 ± 32.5 and success 0.000 ± 0.000.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

