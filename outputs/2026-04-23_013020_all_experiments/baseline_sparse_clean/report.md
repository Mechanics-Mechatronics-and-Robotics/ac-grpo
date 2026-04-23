# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-23_013020_all_experiments\baseline_sparse_clean`

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

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| CLEAN | BASELINE_SPARSE | 259.6 ± 8.7 | 0.910 ± 0.042 | 285.7 | 1.000 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | BASELINE_SPARSE | 0 | 250.0 | 0.900 |
| CLEAN | BASELINE_SPARSE | 3 | 259.3 | 0.900 |
| CLEAN | BASELINE_SPARSE | 9 | 266.5 | 0.950 |
| CLEAN | BASELINE_SPARSE | 17 | 270.1 | 0.950 |
| CLEAN | BASELINE_SPARSE | 42 | 252.3 | 0.850 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| CLEAN | BASELINE_SPARSE | 0 | BASELINE_SPARSE_CLEAN_seed0_step0010000.pt | CLEAN | 287.6 | 1.000 |
| CLEAN | BASELINE_SPARSE | 3 | BASELINE_SPARSE_CLEAN_seed3_step0020000.pt | CLEAN | 288.2 | 1.000 |
| CLEAN | BASELINE_SPARSE | 9 | BASELINE_SPARSE_CLEAN_seed9_step0070000.pt | CLEAN | 287.8 | 1.000 |
| CLEAN | BASELINE_SPARSE | 17 | BASELINE_SPARSE_CLEAN_seed17_step0050000.pt | CLEAN | 287.4 | 1.000 |
| CLEAN | BASELINE_SPARSE | 42 | BASELINE_SPARSE_CLEAN_seed42_step0030000.pt | CLEAN | 289.7 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| CLEAN | BASELINE_SPARSE | test_clean | CLEAN | 0.10 | 276.5 ± 22.4 | 0.990 ± 0.100 |
| CLEAN | BASELINE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 173.6 ± 121.7 | 0.574 ± 0.495 |
| CLEAN | BASELINE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 16.2 ± 30.8 | 0.000 ± 0.000 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- CLEAN: `BASELINE_SPARSE` has the highest mean final return (259.6) with mean final success 0.910.

Checkpoint selection versus the pretrained anchor:

- CLEAN / BASELINE_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `BASELINE_SPARSE` in `CLEAN` with 276.5 ± 22.4 and success 0.990 ± 0.100.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `BASELINE_SPARSE` in `CLEAN` with 173.6 ± 121.7 and success 0.574 ± 0.495.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `BASELINE_SPARSE` in `CLEAN` with 16.2 ± 30.8 and success 0.000 ± 0.000.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

