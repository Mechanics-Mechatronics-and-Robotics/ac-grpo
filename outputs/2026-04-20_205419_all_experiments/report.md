# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-20_205419_all_experiments`

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
| CLEAN | AC_FULL | -152.8 ± 90.3 | 0.000 ± 0.000 | 222.0 | 0.760 |
| CLEAN | AC_LITE | -56.2 ± 95.4 | 0.010 ± 0.022 | 219.1 | 0.740 |
| CLEAN | BASELINE | 274.2 ± 18.6 | 0.970 ± 0.067 | 290.9 | 1.000 |
| OBS_NOISE | AC_FULL | -136.3 ± 17.3 | 0.000 ± 0.000 | 78.4 | 0.220 |
| OBS_NOISE | AC_LITE | -152.2 ± 45.1 | 0.000 ± 0.000 | 77.6 | 0.230 |
| OBS_NOISE | BASELINE | 145.9 ± 49.2 | 0.410 ± 0.204 | 229.0 | 0.780 |
| REWARD_NOISE | AC_FULL | -215.0 ± 198.2 | 0.000 ± 0.000 | 221.2 | 0.630 |
| REWARD_NOISE | AC_LITE | -22.9 ± 98.9 | 0.000 ± 0.000 | 226.8 | 0.650 |
| REWARD_NOISE | BASELINE | 263.8 ± 14.0 | 0.750 ± 0.112 | 289.9 | 0.940 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL | 0 | -18.5 | 0.000 |
| CLEAN | AC_FULL | 3 | -111.1 | 0.000 |
| CLEAN | AC_FULL | 9 | -208.1 | 0.000 |
| CLEAN | AC_FULL | 17 | -177.9 | 0.000 |
| CLEAN | AC_FULL | 42 | -248.5 | 0.000 |
| CLEAN | AC_LITE | 0 | -81.4 | 0.050 |
| CLEAN | AC_LITE | 3 | -111.8 | 0.000 |
| CLEAN | AC_LITE | 9 | -166.9 | 0.000 |
| CLEAN | AC_LITE | 17 | 7.3 | 0.000 |
| CLEAN | AC_LITE | 42 | 71.8 | 0.000 |
| CLEAN | BASELINE | 0 | 279.6 | 1.000 |
| CLEAN | BASELINE | 3 | 242.1 | 0.850 |
| CLEAN | BASELINE | 9 | 289.9 | 1.000 |
| CLEAN | BASELINE | 17 | 276.6 | 1.000 |
| CLEAN | BASELINE | 42 | 282.9 | 1.000 |
| OBS_NOISE | AC_FULL | 0 | -144.7 | 0.000 |
| OBS_NOISE | AC_FULL | 3 | -115.1 | 0.000 |
| OBS_NOISE | AC_FULL | 9 | -151.0 | 0.000 |
| OBS_NOISE | AC_FULL | 17 | -120.1 | 0.000 |
| OBS_NOISE | AC_FULL | 42 | -150.5 | 0.000 |
| OBS_NOISE | AC_LITE | 0 | -199.2 | 0.000 |
| OBS_NOISE | AC_LITE | 3 | -102.4 | 0.000 |
| OBS_NOISE | AC_LITE | 9 | -110.6 | 0.000 |
| OBS_NOISE | AC_LITE | 17 | -193.4 | 0.000 |
| OBS_NOISE | AC_LITE | 42 | -155.4 | 0.000 |
| OBS_NOISE | BASELINE | 0 | 201.6 | 0.600 |
| OBS_NOISE | BASELINE | 3 | 97.4 | 0.150 |
| OBS_NOISE | BASELINE | 9 | 184.6 | 0.600 |
| OBS_NOISE | BASELINE | 17 | 94.0 | 0.250 |
| OBS_NOISE | BASELINE | 42 | 151.9 | 0.450 |
| REWARD_NOISE | AC_FULL | 0 | -112.3 | 0.000 |
| REWARD_NOISE | AC_FULL | 3 | 3.0 | 0.000 |
| REWARD_NOISE | AC_FULL | 9 | -262.6 | 0.000 |
| REWARD_NOISE | AC_FULL | 17 | -179.2 | 0.000 |
| REWARD_NOISE | AC_FULL | 42 | -524.0 | 0.000 |
| REWARD_NOISE | AC_LITE | 0 | 38.3 | 0.000 |
| REWARD_NOISE | AC_LITE | 3 | 46.3 | 0.000 |
| REWARD_NOISE | AC_LITE | 9 | -161.7 | 0.000 |
| REWARD_NOISE | AC_LITE | 17 | -94.0 | 0.000 |
| REWARD_NOISE | AC_LITE | 42 | 56.5 | 0.000 |
| REWARD_NOISE | BASELINE | 0 | 285.3 | 0.900 |
| REWARD_NOISE | BASELINE | 3 | 254.6 | 0.750 |
| REWARD_NOISE | BASELINE | 9 | 264.5 | 0.600 |
| REWARD_NOISE | BASELINE | 17 | 265.9 | 0.800 |
| REWARD_NOISE | BASELINE | 42 | 248.5 | 0.700 |

## Best checkpoint by greedy held-out evaluation

Checkpoints are evaluated greedily on fixed held-out seeds. The final checkpoint is not assumed to be best.

| mode | method | seed | checkpoint | eval return | eval success |
|---|---|---:|---|---:|---:|
| CLEAN | AC_FULL | 0 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_FULL | 3 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_FULL | 9 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_FULL | 17 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_FULL | 42 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_LITE | 0 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_LITE | 3 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_LITE | 9 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_LITE | 17 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_LITE | 42 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | BASELINE | 0 | BASELINE_CLEAN_seed0_step0010000.pt | 288.0 | 1.000 |
| CLEAN | BASELINE | 3 | BASELINE_CLEAN_seed3_step0050000.pt | 287.5 | 1.000 |
| CLEAN | BASELINE | 9 | BASELINE_CLEAN_seed9_step0010000.pt | 290.9 | 1.000 |
| CLEAN | BASELINE | 17 | BASELINE_CLEAN_seed17_step0030000.pt | 286.9 | 1.000 |
| CLEAN | BASELINE | 42 | BASELINE_CLEAN_seed42_final.pt | 288.1 | 1.000 |
| OBS_NOISE | AC_FULL | 0 | checkpoint_0_pretrained | 213.0 | 0.600 |
| OBS_NOISE | AC_FULL | 3 | checkpoint_0_pretrained | 213.0 | 0.600 |
| OBS_NOISE | AC_FULL | 9 | checkpoint_0_pretrained | 213.0 | 0.600 |
| OBS_NOISE | AC_FULL | 17 | checkpoint_0_pretrained | 213.0 | 0.600 |
| OBS_NOISE | AC_FULL | 42 | checkpoint_0_pretrained | 213.0 | 0.600 |
| OBS_NOISE | AC_LITE | 0 | checkpoint_0_pretrained | 213.0 | 0.600 |
| OBS_NOISE | AC_LITE | 3 | checkpoint_0_pretrained | 213.0 | 0.600 |
| OBS_NOISE | AC_LITE | 9 | checkpoint_0_pretrained | 213.0 | 0.600 |
| OBS_NOISE | AC_LITE | 17 | checkpoint_0_pretrained | 213.0 | 0.600 |
| OBS_NOISE | AC_LITE | 42 | checkpoint_0_pretrained | 213.0 | 0.600 |
| OBS_NOISE | BASELINE | 0 | BASELINE_OBS_NOISE_seed0_step0050000.pt | 270.7 | 0.933 |
| OBS_NOISE | BASELINE | 3 | BASELINE_OBS_NOISE_seed3_step0030000.pt | 216.5 | 0.733 |
| OBS_NOISE | BASELINE | 9 | BASELINE_OBS_NOISE_seed9_step0040000.pt | 244.0 | 0.800 |
| OBS_NOISE | BASELINE | 17 | BASELINE_OBS_NOISE_seed17_step0010000.pt | 230.3 | 0.667 |
| OBS_NOISE | BASELINE | 42 | BASELINE_OBS_NOISE_seed42_step0030000.pt | 260.8 | 0.867 |
| REWARD_NOISE | AC_FULL | 0 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | AC_FULL | 3 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | AC_FULL | 9 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | AC_FULL | 17 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | AC_FULL | 42 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE | 0 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE | 3 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE | 9 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE | 17 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE | 42 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE | 0 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE | 3 | BASELINE_REWARD_NOISE_seed3_step0030000.pt | 286.8 | 1.000 |
| REWARD_NOISE | BASELINE | 9 | BASELINE_REWARD_NOISE_seed9_final.pt | 289.4 | 1.000 |
| REWARD_NOISE | BASELINE | 17 | BASELINE_REWARD_NOISE_seed17_step0010000.pt | 289.9 | 1.000 |
| REWARD_NOISE | BASELINE | 42 | BASELINE_REWARD_NOISE_seed42_step0020000.pt | 290.9 | 1.000 |

## Certainty AUROC diagnostics

- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.
- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\) (diagnostic).

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL | 0.749 | 0.450 |
| CLEAN | AC_LITE | 0.257 | 0.506 |
| OBS_NOISE | AC_FULL | 0.865 | 0.467 |
| OBS_NOISE | AC_LITE | 0.067 | 0.490 |
| REWARD_NOISE | AC_FULL | 0.733 | 0.445 |
| REWARD_NOISE | AC_LITE | 0.261 | 0.514 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

