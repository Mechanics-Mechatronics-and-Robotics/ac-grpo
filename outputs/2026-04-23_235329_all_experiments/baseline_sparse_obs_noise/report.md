# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-23_235329_all_experiments\baseline_sparse_obs_noise`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Experiment protocol

- Implemented method variants in this run: BASELINE_SPARSE
- Training modes: OBS_NOISE
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
| OBS_NOISE | BASELINE_SPARSE | 206.2 ± 40.0 | 0.770 ± 0.125 | 225.1 ± 9.9 | 287.6 | 1.000 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| OBS_NOISE | BASELINE_SPARSE | 0 | 254.9 | 0.900 |
| OBS_NOISE | BASELINE_SPARSE | 3 | 167.3 | 0.650 |
| OBS_NOISE | BASELINE_SPARSE | 9 | 184.4 | 0.750 |
| OBS_NOISE | BASELINE_SPARSE | 17 | 243.6 | 0.900 |
| OBS_NOISE | BASELINE_SPARSE | 42 | 180.6 | 0.650 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| OBS_NOISE | BASELINE_SPARSE | 0 | BASELINE_SPARSE_OBS_NOISE_seed0_step0450000.pt | OBS_NOISE | 283.9 | 1.000 |
| OBS_NOISE | BASELINE_SPARSE | 3 | BASELINE_SPARSE_OBS_NOISE_seed3_step0150000.pt | OBS_NOISE | 286.2 | 1.000 |
| OBS_NOISE | BASELINE_SPARSE | 9 | BASELINE_SPARSE_OBS_NOISE_seed9_step0700000.pt | OBS_NOISE | 279.4 | 1.000 |
| OBS_NOISE | BASELINE_SPARSE | 17 | BASELINE_SPARSE_OBS_NOISE_seed17_step0200000.pt | OBS_NOISE | 286.0 | 1.000 |
| OBS_NOISE | BASELINE_SPARSE | 42 | BASELINE_SPARSE_OBS_NOISE_seed42_step0300000.pt | OBS_NOISE | 283.7 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| OBS_NOISE | BASELINE_SPARSE | test_clean | CLEAN | 0.10 | 260.4 ± 47.4 | 0.946 ± 0.226 |
| OBS_NOISE | BASELINE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 260.5 ± 72.3 | 0.936 ± 0.245 |
| OBS_NOISE | BASELINE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 154.7 ± 136.4 | 0.538 ± 0.499 |

## Cross-test summary on OBS evaluation

| Method | Train: CLEAN → Test OBS | Train: OBS → Test OBS | Train: REWARD → Test OBS |
|---|---:|---:|---:|
| Baseline (Sparse) | n/a | 260.5 ± 72.3 / 0.936 ± 0.245 | n/a |
| Baseline (Dense) | n/a | n/a | n/a |
| AC-LITE (Sparse) | n/a | n/a | n/a |
| AC-LITE (Dense) | n/a | n/a | n/a |
| AC-FULL (Sparse) | n/a | n/a | n/a |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- OBS_NOISE: `BASELINE_SPARSE` has the highest mean final return (206.2) with mean final success 0.770.

Checkpoint selection versus the pretrained anchor:

- OBS_NOISE / BASELINE_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `BASELINE_SPARSE` in `OBS_NOISE` with 260.4 ± 47.4 and success 0.946 ± 0.226.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `BASELINE_SPARSE` in `OBS_NOISE` with 260.5 ± 72.3 and success 0.936 ± 0.245.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `BASELINE_SPARSE` in `OBS_NOISE` with 154.7 ± 136.4 and success 0.538 ± 0.499.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

