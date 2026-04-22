# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-23_002939_all_experiments\baseline_sparse_obs_noise`

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

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| OBS_NOISE | BASELINE_SPARSE | 163.8 ± 34.0 | 0.520 ± 0.152 | 194.3 | 0.630 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| OBS_NOISE | BASELINE_SPARSE | 0 | 133.6 | 0.400 |
| OBS_NOISE | BASELINE_SPARSE | 3 | 137.4 | 0.350 |
| OBS_NOISE | BASELINE_SPARSE | 9 | 192.2 | 0.650 |
| OBS_NOISE | BASELINE_SPARSE | 17 | 208.0 | 0.700 |
| OBS_NOISE | BASELINE_SPARSE | 42 | 147.9 | 0.500 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| OBS_NOISE | BASELINE_SPARSE | 0 | BASELINE_SPARSE_OBS_NOISE_seed0_final.pt | OBS_NOISE | 248.2 | 0.800 |
| OBS_NOISE | BASELINE_SPARSE | 3 | BASELINE_SPARSE_OBS_NOISE_seed3_final.pt | OBS_NOISE | 230.4 | 0.733 |
| OBS_NOISE | BASELINE_SPARSE | 9 | BASELINE_SPARSE_OBS_NOISE_seed9_step0020000.pt | OBS_NOISE | 270.3 | 0.867 |
| OBS_NOISE | BASELINE_SPARSE | 17 | BASELINE_SPARSE_OBS_NOISE_seed17_step0020000.pt | OBS_NOISE | 221.2 | 0.667 |
| OBS_NOISE | BASELINE_SPARSE | 42 | BASELINE_SPARSE_OBS_NOISE_seed42_step0020000.pt | OBS_NOISE | 217.4 | 0.667 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| OBS_NOISE | BASELINE_SPARSE | test_clean | CLEAN | 0.10 | 274.2 ± 24.0 | 0.983 ± 0.128 |
| OBS_NOISE | BASELINE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 218.1 ± 106.0 | 0.740 ± 0.439 |
| OBS_NOISE | BASELINE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 48.0 ± 75.3 | 0.077 ± 0.267 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- OBS_NOISE: `BASELINE_SPARSE` has the highest mean final return (163.8) with mean final success 0.520.

Checkpoint selection versus the pretrained anchor:

- OBS_NOISE / BASELINE_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `BASELINE_SPARSE` in `OBS_NOISE` with 274.2 ± 24.0 and success 0.983 ± 0.128.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `BASELINE_SPARSE` in `OBS_NOISE` with 218.1 ± 106.0 and success 0.740 ± 0.439.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `BASELINE_SPARSE` in `OBS_NOISE` with 48.0 ± 75.3 and success 0.077 ± 0.267.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

