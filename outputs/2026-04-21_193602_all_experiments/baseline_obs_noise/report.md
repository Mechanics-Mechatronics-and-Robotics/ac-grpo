# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-21_193602_all_experiments\baseline_obs_noise`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Notes on experimental modes

- **Reward semantics**: logs include `return_train` (optimizer reward) and `return_env` (raw dense environment return). Plot/report curves use `return_env` unless stated otherwise.
- **REWARD_NOISE**: false-negative successes set terminal `policy_success` to `0` during training. Checkpoint selection evaluates on clean held-out episodes because reward noise is training-only corruption.
- **OBS_NOISE**: adds Gaussian noise \(\sigma=0.1\) to observations at every step.
- **AC v3**: AC methods use standard PPO with certainty-gated advantages; runner-up statistics supervise the certainty network only.

## Seed aggregation

Learning curves are computed **per seed** and then aggregated (mean ± std). This avoids interleaving seeds (each run resets `step` to 0).

## Summary table (mean ± std over 5 seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| OBS_NOISE | BASELINE | 233.3 ± 25.0 | 0.800 ± 0.094 | 280.4 | 0.990 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| OBS_NOISE | BASELINE | 0 | 225.4 | 0.800 |
| OBS_NOISE | BASELINE | 3 | 252.5 | 0.850 |
| OBS_NOISE | BASELINE | 9 | 235.9 | 0.800 |
| OBS_NOISE | BASELINE | 17 | 257.6 | 0.900 |
| OBS_NOISE | BASELINE | 42 | 194.9 | 0.650 |

## Best checkpoint by greedy held-out evaluation

Checkpoints are evaluated greedily on fixed held-out seeds. The final checkpoint is not assumed to be best.

| mode | method | seed | checkpoint | eval return | eval success |
|---|---|---:|---|---:|---:|
| OBS_NOISE | BASELINE | 0 | BASELINE_OBS_NOISE_seed0_step0250000.pt | 281.9 | 1.000 |
| OBS_NOISE | BASELINE | 3 | BASELINE_OBS_NOISE_seed3_step0150000.pt | 286.2 | 1.000 |
| OBS_NOISE | BASELINE | 9 | BASELINE_OBS_NOISE_seed9_step0090000.pt | 287.5 | 1.000 |
| OBS_NOISE | BASELINE | 17 | BASELINE_OBS_NOISE_seed17_step0180000.pt | 286.8 | 1.000 |
| OBS_NOISE | BASELINE | 42 | BASELINE_OBS_NOISE_seed42_step0120000.pt | 287.0 | 1.000 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

