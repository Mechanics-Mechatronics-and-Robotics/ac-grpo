# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-20_205419_all_experiments\baseline_clean`

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
| CLEAN | BASELINE | 274.2 ± 18.6 | 0.970 ± 0.067 | 290.9 | 1.000 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | BASELINE | 0 | 279.6 | 1.000 |
| CLEAN | BASELINE | 3 | 242.1 | 0.850 |
| CLEAN | BASELINE | 9 | 289.9 | 1.000 |
| CLEAN | BASELINE | 17 | 276.6 | 1.000 |
| CLEAN | BASELINE | 42 | 282.9 | 1.000 |

## Best checkpoint by greedy held-out evaluation

Checkpoints are evaluated greedily on fixed held-out seeds. The final checkpoint is not assumed to be best.

| mode | method | seed | checkpoint | eval return | eval success |
|---|---|---:|---|---:|---:|
| CLEAN | BASELINE | 0 | BASELINE_CLEAN_seed0_step0010000.pt | 288.0 | 1.000 |
| CLEAN | BASELINE | 3 | BASELINE_CLEAN_seed3_final.pt | 287.5 | 1.000 |
| CLEAN | BASELINE | 9 | BASELINE_CLEAN_seed9_step0010000.pt | 290.9 | 1.000 |
| CLEAN | BASELINE | 17 | BASELINE_CLEAN_seed17_step0030000.pt | 286.9 | 1.000 |
| CLEAN | BASELINE | 42 | BASELINE_CLEAN_seed42_step0050000.pt | 288.1 | 1.000 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

