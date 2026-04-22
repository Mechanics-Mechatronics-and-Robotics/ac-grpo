# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-22_231909_all_experiments\baseline_dense_obs_noise`

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

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| OBS_NOISE | BASELINE_DENSE | 188.0 ± 27.5 | 0.600 ± 0.100 | 244.4 | 0.840 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| OBS_NOISE | BASELINE_DENSE | 0 | 182.7 | 0.550 |
| OBS_NOISE | BASELINE_DENSE | 3 | 215.9 | 0.700 |
| OBS_NOISE | BASELINE_DENSE | 9 | 143.9 | 0.450 |
| OBS_NOISE | BASELINE_DENSE | 17 | 203.2 | 0.650 |
| OBS_NOISE | BASELINE_DENSE | 42 | 194.2 | 0.650 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| OBS_NOISE | BASELINE_DENSE | 0 | BASELINE_DENSE_OBS_NOISE_seed0_step0020000.pt | OBS_NOISE | 250.9 | 0.800 |
| OBS_NOISE | BASELINE_DENSE | 3 | BASELINE_DENSE_OBS_NOISE_seed3_final.pt | OBS_NOISE | 258.3 | 0.867 |
| OBS_NOISE | BASELINE_DENSE | 9 | BASELINE_DENSE_OBS_NOISE_seed9_final.pt | OBS_NOISE | 222.1 | 0.600 |
| OBS_NOISE | BASELINE_DENSE | 17 | BASELINE_DENSE_OBS_NOISE_seed17_step0020000.pt | OBS_NOISE | 254.3 | 0.800 |
| OBS_NOISE | BASELINE_DENSE | 42 | BASELINE_DENSE_OBS_NOISE_seed42_final.pt | OBS_NOISE | 220.2 | 0.600 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| OBS_NOISE | BASELINE_DENSE | test_clean | CLEAN | 0.10 | 273.7 ± 21.1 | 1.000 ± 0.000 |
| OBS_NOISE | BASELINE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 242.0 ± 86.9 | 0.830 ± 0.377 |
| OBS_NOISE | BASELINE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 40.3 ± 44.1 | 0.000 ± 0.000 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- OBS_NOISE: `BASELINE_DENSE` has the highest mean final return (188.0) with mean final success 0.600.

Checkpoint selection versus the pretrained anchor:

- OBS_NOISE / BASELINE_DENSE: checkpoint 0 wins in 0 of 5 seeds (0.00).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `BASELINE_DENSE` in `OBS_NOISE` with 273.7 ± 21.1 and success 1.000 ± 0.000.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `BASELINE_DENSE` in `OBS_NOISE` with 242.0 ± 86.9 and success 0.830 ± 0.377.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `BASELINE_DENSE` in `OBS_NOISE` with 40.3 ± 44.1 and success 0.000 ± 0.000.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

