# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-23_002939_all_experiments\baseline_dense_clean`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Experiment protocol

- Implemented method variants in this run: BASELINE_DENSE
- Training modes: CLEAN
- Reward modes represented: DENSE
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
| CLEAN | BASELINE_DENSE | 253.0 ± 7.4 | 0.880 ± 0.057 | 283.0 | 1.000 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | BASELINE_DENSE | 0 | 257.3 | 0.900 |
| CLEAN | BASELINE_DENSE | 3 | 251.2 | 0.850 |
| CLEAN | BASELINE_DENSE | 9 | 260.2 | 0.950 |
| CLEAN | BASELINE_DENSE | 17 | 241.1 | 0.800 |
| CLEAN | BASELINE_DENSE | 42 | 255.1 | 0.900 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| CLEAN | BASELINE_DENSE | 0 | BASELINE_DENSE_CLEAN_seed0_step0010000.pt | CLEAN | 287.8 | 1.000 |
| CLEAN | BASELINE_DENSE | 3 | BASELINE_DENSE_CLEAN_seed3_final.pt | CLEAN | 286.5 | 1.000 |
| CLEAN | BASELINE_DENSE | 9 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_DENSE | 17 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_DENSE | 42 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| CLEAN | BASELINE_DENSE | test_clean | CLEAN | 0.10 | 277.3 ± 20.2 | 1.000 ± 0.000 |
| CLEAN | BASELINE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 219.5 ± 106.2 | 0.740 ± 0.439 |
| CLEAN | BASELINE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 25.2 ± 35.6 | 0.003 ± 0.050 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- CLEAN: `BASELINE_DENSE` has the highest mean final return (253.0) with mean final success 0.880.

Checkpoint selection versus the pretrained anchor:

- CLEAN / BASELINE_DENSE: checkpoint 0 wins in 3 of 5 seeds (0.60).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `BASELINE_DENSE` in `CLEAN` with 277.3 ± 20.2 and success 1.000 ± 0.000.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `BASELINE_DENSE` in `CLEAN` with 219.5 ± 106.2 and success 0.740 ± 0.439.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `BASELINE_DENSE` in `CLEAN` with 25.2 ± 35.6 and success 0.003 ± 0.050.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

