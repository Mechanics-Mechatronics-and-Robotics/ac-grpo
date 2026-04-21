# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-21_015400_ac_lite_clean\ac_lite_clean`

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
| CLEAN | AC_LITE | -17.0 ± 76.6 | 0.000 ± 0.000 | 242.3 | 0.840 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_LITE | 0 | 27.0 | 0.000 |
| CLEAN | AC_LITE | 3 | -102.1 | 0.000 |
| CLEAN | AC_LITE | 9 | 78.7 | 0.000 |
| CLEAN | AC_LITE | 17 | -87.1 | 0.000 |
| CLEAN | AC_LITE | 42 | -1.5 | 0.000 |

## Best checkpoint by greedy held-out evaluation

Checkpoints are evaluated greedily on fixed held-out seeds. The final checkpoint is not assumed to be best.

| mode | method | seed | checkpoint | eval return | eval success |
|---|---|---:|---|---:|---:|
| CLEAN | AC_LITE | 0 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_LITE | 3 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_LITE | 9 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_LITE | 17 | AC_LITE_CLEAN_seed17_step0020000_policy.pt | 289.4 | 1.000 |
| CLEAN | AC_LITE | 42 | checkpoint_0_pretrained | 285.9 | 1.000 |

## Certainty AUROC diagnostics

- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.
- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\) (diagnostic).

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_LITE | 0.123 | 0.499 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

