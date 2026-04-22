# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-22_231909_all_experiments\baseline_sparse_obs_noise`

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
| OBS_NOISE | BASELINE_SPARSE | 150.8 ± 38.4 | 0.460 ± 0.164 | 198.0 | 0.660 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| OBS_NOISE | BASELINE_SPARSE | 0 | 190.1 | 0.650 |
| OBS_NOISE | BASELINE_SPARSE | 3 | 96.8 | 0.250 |
| OBS_NOISE | BASELINE_SPARSE | 9 | 184.5 | 0.600 |
| OBS_NOISE | BASELINE_SPARSE | 17 | 148.8 | 0.400 |
| OBS_NOISE | BASELINE_SPARSE | 42 | 133.6 | 0.400 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| OBS_NOISE | BASELINE_SPARSE | 0 | BASELINE_SPARSE_OBS_NOISE_seed0_final.pt | OBS_NOISE | 264.9 | 0.867 |
| OBS_NOISE | BASELINE_SPARSE | 3 | BASELINE_SPARSE_OBS_NOISE_seed3_final.pt | OBS_NOISE | 213.6 | 0.733 |
| OBS_NOISE | BASELINE_SPARSE | 9 | BASELINE_SPARSE_OBS_NOISE_seed9_final.pt | OBS_NOISE | 260.3 | 0.867 |
| OBS_NOISE | BASELINE_SPARSE | 17 | BASELINE_SPARSE_OBS_NOISE_seed17_final.pt | OBS_NOISE | 232.2 | 0.733 |
| OBS_NOISE | BASELINE_SPARSE | 42 | BASELINE_SPARSE_OBS_NOISE_seed42_final.pt | OBS_NOISE | 261.3 | 0.867 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- OBS_NOISE: `BASELINE_SPARSE` has the highest mean final return (150.8) with mean final success 0.460.

Checkpoint selection versus the pretrained anchor:

- OBS_NOISE / BASELINE_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

