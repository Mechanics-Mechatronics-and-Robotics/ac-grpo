# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-20_205419_all_experiments\baseline_obs_noise`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Notes on experimental modes

- **Reward semantics**: PPO/GAE uses sparse terminal binary reward only (`0` before termination, terminal `policy_success` at episode end); dense LunarLander return is logged for diagnostics only.
- **REWARD_NOISE**: false-negative successes set terminal `policy_success` to `0`, so the sparse policy update sees the corrupted outcome directly.
- **OBS_NOISE**: adds Gaussian noise \(\sigma=0.1\) to observations at every step.
- **AC v3**: AC methods use runner-up mixture PPO; `delta` is the normalized executed-vs-runner-up margin, and `mixture_prob` is the likelihood used by the AC ratio.

## Seed aggregation

Learning curves are computed **per seed** and then aggregated (mean ± std). This avoids interleaving seeds (each run resets `step` to 0).

## Summary table (mean ± std over 5 seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| OBS_NOISE | BASELINE | 145.9 ± 49.2 | 0.410 ± 0.204 | 229.0 | 0.780 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| OBS_NOISE | BASELINE | 0 | 201.6 | 0.600 |
| OBS_NOISE | BASELINE | 3 | 97.4 | 0.150 |
| OBS_NOISE | BASELINE | 9 | 184.6 | 0.600 |
| OBS_NOISE | BASELINE | 17 | 94.0 | 0.250 |
| OBS_NOISE | BASELINE | 42 | 151.9 | 0.450 |

## Best checkpoint by greedy held-out evaluation

Checkpoints are evaluated greedily on fixed held-out seeds. The final checkpoint is not assumed to be best.

| mode | method | seed | checkpoint | eval return | eval success |
|---|---|---:|---|---:|---:|
| OBS_NOISE | BASELINE | 0 | BASELINE_OBS_NOISE_seed0_final.pt | 270.7 | 0.933 |
| OBS_NOISE | BASELINE | 3 | BASELINE_OBS_NOISE_seed3_step0030000.pt | 216.5 | 0.733 |
| OBS_NOISE | BASELINE | 9 | BASELINE_OBS_NOISE_seed9_step0040000.pt | 244.0 | 0.800 |
| OBS_NOISE | BASELINE | 17 | BASELINE_OBS_NOISE_seed17_step0010000.pt | 230.3 | 0.667 |
| OBS_NOISE | BASELINE | 42 | BASELINE_OBS_NOISE_seed42_step0030000.pt | 260.8 | 0.867 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

