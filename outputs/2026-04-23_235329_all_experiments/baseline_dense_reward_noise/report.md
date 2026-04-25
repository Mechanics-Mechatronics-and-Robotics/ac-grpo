# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-23_235329_all_experiments\baseline_dense_reward_noise`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Experiment protocol

- Implemented method variants in this run: BASELINE_DENSE
- Training modes: REWARD_NOISE
- Reward modes represented: DENSE
- Training seeds: 0, 3, 9, 17, 42
- All branches start from the shared pretrained anchor when a pretrained path is provided.
- Training uses grouped rollouts with dynamic sampling fallback when no mixed-outcome groups are available.
- Checkpoints are saved during training and the pretrained anchor is treated as checkpoint 0.
- Checkpoint selection uses greedy held-out evaluation under the branch's primary selection condition.
- Additional challenge tests evaluate selected checkpoints under CLEAN, OBS_NOISE (typical), and OBS_NOISE (hard).
- Challenge tests currently use up to 140 episodes per evaluation seed in the generated logs.

## Summary table (mean ± std over seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | reward AUC | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|---:|
| REWARD_NOISE | BASELINE_DENSE | 235.0 ± 47.3 | 0.630 ± 0.164 | 232.6 ± 11.8 | 289.2 | 0.970 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| REWARD_NOISE | BASELINE_DENSE | 0 | 244.2 | 0.600 |
| REWARD_NOISE | BASELINE_DENSE | 3 | 277.5 | 0.850 |
| REWARD_NOISE | BASELINE_DENSE | 9 | 244.2 | 0.600 |
| REWARD_NOISE | BASELINE_DENSE | 17 | 153.9 | 0.400 |
| REWARD_NOISE | BASELINE_DENSE | 42 | 255.2 | 0.700 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| REWARD_NOISE | BASELINE_DENSE | 0 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 3 | BASELINE_DENSE_REWARD_NOISE_seed3_step0050000.pt | REWARD_NOISE | 288.0 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 9 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 17 | BASELINE_DENSE_REWARD_NOISE_seed17_step0050000.pt | REWARD_NOISE | 287.2 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 42 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| REWARD_NOISE | BASELINE_DENSE | test_clean | CLEAN | 0.10 | 277.2 ± 20.5 | 1.000 ± 0.000 |
| REWARD_NOISE | BASELINE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 214.4 ± 108.7 | 0.718 ± 0.450 |
| REWARD_NOISE | BASELINE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 24.7 ± 33.3 | 0.000 ± 0.000 |

## Cross-test summary on OBS evaluation

| Method | Train: CLEAN → Test OBS | Train: OBS → Test OBS | Train: REWARD → Test OBS |
|---|---:|---:|---:|
| Baseline (Sparse) | n/a | n/a | n/a |
| Baseline (Dense) | n/a | n/a | 214.4 ± 108.7 / 0.718 ± 0.450 |
| AC-LITE (Sparse) | n/a | n/a | n/a |
| AC-LITE (Dense) | n/a | n/a | n/a |
| AC-FULL (Sparse) | n/a | n/a | n/a |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- REWARD_NOISE: `BASELINE_DENSE` has the highest mean final return (235.0) with mean final success 0.630.

Checkpoint selection versus the pretrained anchor:

- REWARD_NOISE / BASELINE_DENSE: checkpoint 0 wins in 3 of 5 seeds (0.60).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `BASELINE_DENSE` in `REWARD_NOISE` with 277.2 ± 20.5 and success 1.000 ± 0.000.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `BASELINE_DENSE` in `REWARD_NOISE` with 214.4 ± 108.7 and success 0.718 ± 0.450.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `BASELINE_DENSE` in `REWARD_NOISE` with 24.7 ± 33.3 and success 0.000 ± 0.000.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

