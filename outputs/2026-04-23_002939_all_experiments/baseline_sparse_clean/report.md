# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-23_002939_all_experiments\baseline_sparse_clean`

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
- Challenge tests currently use up to 140 episodes per evaluation seed in the generated logs.

## Summary table (mean ± std over seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| CLEAN | BASELINE_SPARSE | 244.1 ± 22.6 | 0.840 ± 0.096 | 277.4 | 0.990 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | BASELINE_SPARSE | 0 | 276.4 | 0.950 |
| CLEAN | BASELINE_SPARSE | 3 | 257.1 | 0.900 |
| CLEAN | BASELINE_SPARSE | 9 | 230.3 | 0.800 |
| CLEAN | BASELINE_SPARSE | 17 | 219.8 | 0.700 |
| CLEAN | BASELINE_SPARSE | 42 | 237.0 | 0.850 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| CLEAN | BASELINE_SPARSE | 0 | BASELINE_SPARSE_CLEAN_seed0_step0010000.pt | CLEAN | 287.4 | 1.000 |
| CLEAN | BASELINE_SPARSE | 3 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_SPARSE | 9 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_SPARSE | 17 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_SPARSE | 42 | BASELINE_SPARSE_CLEAN_seed42_final.pt | CLEAN | 288.6 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| CLEAN | BASELINE_SPARSE | test_clean | CLEAN | 0.10 | 277.6 ± 20.2 | 1.000 ± 0.000 |
| CLEAN | BASELINE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 207.3 ± 112.3 | 0.690 ± 0.463 |
| CLEAN | BASELINE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 22.4 ± 32.8 | 0.003 ± 0.050 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- CLEAN: `BASELINE_SPARSE` has the highest mean final return (244.1) with mean final success 0.840.

Checkpoint selection versus the pretrained anchor:

- CLEAN / BASELINE_SPARSE: checkpoint 0 wins in 3 of 5 seeds (0.60).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `BASELINE_SPARSE` in `CLEAN` with 277.6 ± 20.2 and success 1.000 ± 0.000.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `BASELINE_SPARSE` in `CLEAN` with 207.3 ± 112.3 and success 0.690 ± 0.463.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `BASELINE_SPARSE` in `CLEAN` with 22.4 ± 32.8 and success 0.003 ± 0.050.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

