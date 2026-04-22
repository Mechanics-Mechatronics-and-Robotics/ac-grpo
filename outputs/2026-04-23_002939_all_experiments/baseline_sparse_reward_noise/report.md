# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-23_002939_all_experiments\baseline_sparse_reward_noise`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Experiment protocol

- Implemented method variants in this run: BASELINE_SPARSE
- Training modes: REWARD_NOISE
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
| REWARD_NOISE | BASELINE_SPARSE | 252.5 ± 19.0 | 0.730 ± 0.125 | 279.6 | 0.890 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| REWARD_NOISE | BASELINE_SPARSE | 0 | 256.2 | 0.700 |
| REWARD_NOISE | BASELINE_SPARSE | 3 | 279.3 | 0.900 |
| REWARD_NOISE | BASELINE_SPARSE | 9 | 243.1 | 0.750 |
| REWARD_NOISE | BASELINE_SPARSE | 17 | 227.8 | 0.550 |
| REWARD_NOISE | BASELINE_SPARSE | 42 | 255.9 | 0.750 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| REWARD_NOISE | BASELINE_SPARSE | 0 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 3 | BASELINE_SPARSE_REWARD_NOISE_seed3_step0010000.pt | REWARD_NOISE | 288.5 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 9 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 17 | BASELINE_SPARSE_REWARD_NOISE_seed17_step0010000.pt | REWARD_NOISE | 290.0 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 42 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| REWARD_NOISE | BASELINE_SPARSE | test_clean | CLEAN | 0.10 | 278.1 ± 21.1 | 1.000 ± 0.000 |
| REWARD_NOISE | BASELINE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 204.5 ± 112.9 | 0.684 ± 0.465 |
| REWARD_NOISE | BASELINE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 21.1 ± 33.8 | 0.000 ± 0.000 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- REWARD_NOISE: `BASELINE_SPARSE` has the highest mean final return (252.5) with mean final success 0.730.

Checkpoint selection versus the pretrained anchor:

- REWARD_NOISE / BASELINE_SPARSE: checkpoint 0 wins in 3 of 5 seeds (0.60).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `BASELINE_SPARSE` in `REWARD_NOISE` with 278.1 ± 21.1 and success 1.000 ± 0.000.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `BASELINE_SPARSE` in `REWARD_NOISE` with 204.5 ± 112.9 and success 0.684 ± 0.465.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `BASELINE_SPARSE` in `REWARD_NOISE` with 21.1 ± 33.8 and success 0.000 ± 0.000.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

