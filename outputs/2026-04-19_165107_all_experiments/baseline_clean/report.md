# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-19_165107_all_experiments\baseline_clean`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Notes on experimental modes

- **REWARD_NOISE**: false-negative successes also penalize the terminal rollout reward used by PPO/GAE, so the policy update sees the corruption.
- **OBS_NOISE**: adds Gaussian noise \(\sigma=0.1\) to observations at every step.

## Seed aggregation

Learning curves are computed **per seed** and then aggregated (mean ± std). This avoids interleaving seeds (each run resets `step` to 0).

## Summary table (mean ± std over 5 seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| CLEAN | BASELINE | 282.4 ± 9.0 | 0.970 ± 0.045 | 294.1 | 1.000 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | BASELINE | 0 | 288.3 | 1.000 |
| CLEAN | BASELINE | 3 | 286.1 | 1.000 |
| CLEAN | BASELINE | 9 | 279.2 | 0.950 |
| CLEAN | BASELINE | 17 | 290.3 | 1.000 |
| CLEAN | BASELINE | 42 | 268.1 | 0.900 |

## Best checkpoint by greedy held-out evaluation

Checkpoints are evaluated greedily on fixed held-out seeds. The final checkpoint is not assumed to be best.

| mode | method | seed | checkpoint | eval return | eval success |
|---|---|---:|---|---:|---:|
| CLEAN | BASELINE | 0 | BASELINE_CLEAN_seed0_step0480000.pt | 291.7 | 1.000 |
| CLEAN | BASELINE | 3 | BASELINE_CLEAN_seed3_step0390000.pt | 291.6 | 1.000 |
| CLEAN | BASELINE | 9 | BASELINE_CLEAN_seed9_step0200000.pt | 291.2 | 1.000 |
| CLEAN | BASELINE | 17 | BASELINE_CLEAN_seed17_step0410000.pt | 292.0 | 1.000 |
| CLEAN | BASELINE | 42 | BASELINE_CLEAN_seed42_step0480000.pt | 290.1 | 1.000 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

