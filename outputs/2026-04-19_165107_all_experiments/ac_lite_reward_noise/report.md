# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-19_165107_all_experiments\ac_lite_reward_noise`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Notes on experimental modes

- **REWARD_NOISE**: false-negative successes also penalize the terminal rollout reward used by PPO/GAE, so the policy update sees the corruption.
- **OBS_NOISE**: adds Gaussian noise \(\sigma=0.1\) to observations at every step.

## Seed aggregation

Learning curves are computed **per seed** and then aggregated (mean ± std). This avoids interleaving seeds (each run resets `step` to 0).

## Summary table (mean ± std over 5 seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| REWARD_NOISE | AC_LITE | 270.2 ± 9.2 | 0.760 ± 0.129 | 293.7 | 0.980 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| REWARD_NOISE | AC_LITE | 0 | 264.8 | 0.600 |
| REWARD_NOISE | AC_LITE | 3 | 281.6 | 0.800 |
| REWARD_NOISE | AC_LITE | 9 | 278.1 | 0.850 |
| REWARD_NOISE | AC_LITE | 17 | 266.3 | 0.900 |
| REWARD_NOISE | AC_LITE | 42 | 260.0 | 0.650 |

## Best checkpoint by greedy held-out evaluation

Checkpoints are evaluated greedily on fixed held-out seeds. The final checkpoint is not assumed to be best.

| mode | method | seed | checkpoint | eval return | eval success |
|---|---|---:|---|---:|---:|
| REWARD_NOISE | AC_LITE | 0 | AC_LITE_REWARD_NOISE_seed0_step0280000_policy.pt | 291.2 | 1.000 |
| REWARD_NOISE | AC_LITE | 3 | AC_LITE_REWARD_NOISE_seed3_step0270000_policy.pt | 292.0 | 1.000 |
| REWARD_NOISE | AC_LITE | 9 | AC_LITE_REWARD_NOISE_seed9_step0410000_policy.pt | 292.5 | 1.000 |
| REWARD_NOISE | AC_LITE | 17 | AC_LITE_REWARD_NOISE_seed17_step0430000_policy.pt | 290.7 | 1.000 |
| REWARD_NOISE | AC_LITE | 42 | AC_LITE_REWARD_NOISE_seed42_step0450000_policy.pt | 293.0 | 1.000 |

## Certainty AUROC diagnostics

- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.
- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\) (diagnostic).

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| REWARD_NOISE | AC_LITE | 0.522 | 0.195 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

