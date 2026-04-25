# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-23_235329_all_experiments\baseline_dense_obs_noise`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Experiment protocol

- Implemented method variants in this run: BASELINE_DENSE
- Training modes: OBS_NOISE
- Reward modes represented: DENSE
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
| OBS_NOISE | BASELINE_DENSE | 51.2 ± 34.4 | 0.060 ± 0.134 | 148.4 ± 20.1 | 282.7 | 1.000 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| OBS_NOISE | BASELINE_DENSE | 0 | 64.4 | 0.000 |
| OBS_NOISE | BASELINE_DENSE | 3 | 84.1 | 0.300 |
| OBS_NOISE | BASELINE_DENSE | 9 | 22.8 | 0.000 |
| OBS_NOISE | BASELINE_DENSE | 17 | 78.0 | 0.000 |
| OBS_NOISE | BASELINE_DENSE | 42 | 7.0 | 0.000 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| OBS_NOISE | BASELINE_DENSE | 0 | BASELINE_DENSE_OBS_NOISE_seed0_step0100000.pt | OBS_NOISE | 277.7 | 0.933 |
| OBS_NOISE | BASELINE_DENSE | 3 | BASELINE_DENSE_OBS_NOISE_seed3_step0250000.pt | OBS_NOISE | 269.6 | 0.933 |
| OBS_NOISE | BASELINE_DENSE | 9 | BASELINE_DENSE_OBS_NOISE_seed9_step0200000.pt | OBS_NOISE | 275.2 | 1.000 |
| OBS_NOISE | BASELINE_DENSE | 17 | BASELINE_DENSE_OBS_NOISE_seed17_step0200000.pt | OBS_NOISE | 274.5 | 1.000 |
| OBS_NOISE | BASELINE_DENSE | 42 | BASELINE_DENSE_OBS_NOISE_seed42_step0100000.pt | OBS_NOISE | 277.8 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| OBS_NOISE | BASELINE_DENSE | test_clean | CLEAN | 0.10 | 253.4 ± 26.4 | 0.970 ± 0.171 |
| OBS_NOISE | BASELINE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 264.6 ± 42.3 | 0.962 ± 0.191 |
| OBS_NOISE | BASELINE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 175.6 ± 112.2 | 0.558 ± 0.497 |

## Cross-test summary on OBS evaluation

| Method | Train: CLEAN → Test OBS | Train: OBS → Test OBS | Train: REWARD → Test OBS |
|---|---:|---:|---:|
| Baseline (Sparse) | n/a | n/a | n/a |
| Baseline (Dense) | n/a | 264.6 ± 42.3 / 0.962 ± 0.191 | n/a |
| AC-LITE (Sparse) | n/a | n/a | n/a |
| AC-LITE (Dense) | n/a | n/a | n/a |
| AC-FULL (Sparse) | n/a | n/a | n/a |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- OBS_NOISE: `BASELINE_DENSE` has the highest mean final return (51.2) with mean final success 0.060.

Checkpoint selection versus the pretrained anchor:

- OBS_NOISE / BASELINE_DENSE: checkpoint 0 wins in 0 of 5 seeds (0.00).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `BASELINE_DENSE` in `OBS_NOISE` with 253.4 ± 26.4 and success 0.970 ± 0.171.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `BASELINE_DENSE` in `OBS_NOISE` with 264.6 ± 42.3 and success 0.962 ± 0.191.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `BASELINE_DENSE` in `OBS_NOISE` with 175.6 ± 112.2 and success 0.558 ± 0.497.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

