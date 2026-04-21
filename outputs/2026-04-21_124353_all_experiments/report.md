# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-21_124353_all_experiments`

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
| CLEAN | AC_FULL | 51.6 ± 70.2 | 0.220 ± 0.225 | 240.0 | 0.830 |
| CLEAN | AC_LITE | 150.9 ± 117.3 | 0.510 ± 0.429 | 243.5 | 0.840 |
| CLEAN | BASELINE | 243.2 ± 18.0 | 0.840 ± 0.082 | 283.1 | 0.990 |
| OBS_NOISE | AC_FULL | -80.0 ± 33.3 | 0.000 ± 0.000 | 53.0 | 0.220 |
| OBS_NOISE | AC_LITE | -45.7 ± 41.7 | 0.010 ± 0.022 | 56.7 | 0.230 |
| OBS_NOISE | BASELINE | 131.2 ± 54.1 | 0.370 ± 0.231 | 195.6 | 0.640 |
| REWARD_NOISE | AC_FULL | 16.8 ± 117.5 | 0.150 ± 0.212 | 193.1 | 0.540 |
| REWARD_NOISE | AC_LITE | 5.2 ± 65.6 | 0.070 ± 0.157 | 191.8 | 0.540 |
| REWARD_NOISE | BASELINE | 241.2 ± 22.7 | 0.640 ± 0.185 | 276.1 | 0.860 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL | 0 | -48.6 | 0.000 |
| CLEAN | AC_FULL | 3 | 71.2 | 0.100 |
| CLEAN | AC_FULL | 9 | 75.8 | 0.350 |
| CLEAN | AC_FULL | 17 | 19.9 | 0.100 |
| CLEAN | AC_FULL | 42 | 139.5 | 0.550 |
| CLEAN | AC_LITE | 0 | 8.1 | 0.000 |
| CLEAN | AC_LITE | 3 | 172.6 | 0.550 |
| CLEAN | AC_LITE | 9 | 250.8 | 0.900 |
| CLEAN | AC_LITE | 17 | 52.1 | 0.150 |
| CLEAN | AC_LITE | 42 | 271.1 | 0.950 |
| CLEAN | BASELINE | 0 | 238.0 | 0.800 |
| CLEAN | BASELINE | 3 | 238.9 | 0.800 |
| CLEAN | BASELINE | 9 | 259.3 | 0.900 |
| CLEAN | BASELINE | 17 | 217.8 | 0.750 |
| CLEAN | BASELINE | 42 | 261.8 | 0.950 |
| OBS_NOISE | AC_FULL | 0 | -114.3 | 0.000 |
| OBS_NOISE | AC_FULL | 3 | -101.6 | 0.000 |
| OBS_NOISE | AC_FULL | 9 | -95.0 | 0.000 |
| OBS_NOISE | AC_FULL | 17 | -50.2 | 0.000 |
| OBS_NOISE | AC_FULL | 42 | -39.1 | 0.000 |
| OBS_NOISE | AC_LITE | 0 | -20.0 | 0.050 |
| OBS_NOISE | AC_LITE | 3 | -8.4 | 0.000 |
| OBS_NOISE | AC_LITE | 9 | -115.3 | 0.000 |
| OBS_NOISE | AC_LITE | 17 | -47.3 | 0.000 |
| OBS_NOISE | AC_LITE | 42 | -37.7 | 0.000 |
| OBS_NOISE | BASELINE | 0 | 111.6 | 0.300 |
| OBS_NOISE | BASELINE | 3 | 141.7 | 0.400 |
| OBS_NOISE | BASELINE | 9 | 99.2 | 0.250 |
| OBS_NOISE | BASELINE | 17 | 220.1 | 0.750 |
| OBS_NOISE | BASELINE | 42 | 83.4 | 0.150 |
| REWARD_NOISE | AC_FULL | 0 | -48.6 | 0.000 |
| REWARD_NOISE | AC_FULL | 3 | 133.6 | 0.300 |
| REWARD_NOISE | AC_FULL | 9 | -74.4 | 0.000 |
| REWARD_NOISE | AC_FULL | 17 | 155.4 | 0.450 |
| REWARD_NOISE | AC_FULL | 42 | -82.0 | 0.000 |
| REWARD_NOISE | AC_LITE | 0 | 8.1 | 0.000 |
| REWARD_NOISE | AC_LITE | 3 | 111.4 | 0.350 |
| REWARD_NOISE | AC_LITE | 9 | -58.4 | 0.000 |
| REWARD_NOISE | AC_LITE | 17 | -37.8 | 0.000 |
| REWARD_NOISE | AC_LITE | 42 | 2.6 | 0.000 |
| REWARD_NOISE | BASELINE | 0 | 270.4 | 0.800 |
| REWARD_NOISE | BASELINE | 3 | 253.1 | 0.600 |
| REWARD_NOISE | BASELINE | 9 | 214.0 | 0.350 |
| REWARD_NOISE | BASELINE | 17 | 223.3 | 0.650 |
| REWARD_NOISE | BASELINE | 42 | 245.2 | 0.800 |

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
| CLEAN | BASELINE | 0 | BASELINE_CLEAN_seed0_step0020000.pt | 286.5 | 1.000 |
| CLEAN | BASELINE | 3 | BASELINE_CLEAN_seed3_step0020000.pt | 288.1 | 1.000 |
| CLEAN | BASELINE | 9 | BASELINE_CLEAN_seed9_step0020000.pt | 289.3 | 1.000 |
| CLEAN | BASELINE | 17 | BASELINE_CLEAN_seed17_step0010000.pt | 287.8 | 1.000 |
| CLEAN | BASELINE | 42 | BASELINE_CLEAN_seed42_step0020000.pt | 286.7 | 1.000 |
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
| OBS_NOISE | BASELINE | 0 | BASELINE_OBS_NOISE_seed0_step0020000.pt | 230.4 | 0.733 |
| OBS_NOISE | BASELINE | 3 | checkpoint_0_pretrained | 213.0 | 0.600 |
| OBS_NOISE | BASELINE | 9 | BASELINE_OBS_NOISE_seed9_step0010000.pt | 244.8 | 0.733 |
| OBS_NOISE | BASELINE | 17 | BASELINE_OBS_NOISE_seed17_step0010000.pt | 223.9 | 0.667 |
| OBS_NOISE | BASELINE | 42 | checkpoint_0_pretrained | 213.0 | 0.600 |
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
| REWARD_NOISE | BASELINE | 0 | BASELINE_REWARD_NOISE_seed0_step0010000.pt | 286.8 | 1.000 |
| REWARD_NOISE | BASELINE | 3 | BASELINE_REWARD_NOISE_seed3_step0020000.pt | 287.6 | 1.000 |
| REWARD_NOISE | BASELINE | 9 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE | 17 | BASELINE_REWARD_NOISE_seed17_step0010000.pt | 287.0 | 1.000 |
| REWARD_NOISE | BASELINE | 42 | BASELINE_REWARD_NOISE_seed42_step0020000.pt | 287.6 | 1.000 |

## Certainty AUROC diagnostics

- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.
- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\) (diagnostic).

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL | 0.682 | 0.470 |
| CLEAN | AC_LITE | 0.552 | 0.442 |
| OBS_NOISE | AC_FULL | 0.796 | 0.449 |
| OBS_NOISE | AC_LITE | 0.103 | 0.508 |
| REWARD_NOISE | AC_FULL | 0.745 | 0.445 |
| REWARD_NOISE | AC_LITE | 0.273 | 0.451 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

