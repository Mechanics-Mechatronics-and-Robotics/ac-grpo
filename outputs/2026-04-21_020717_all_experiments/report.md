# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-21_020717_all_experiments`

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
| CLEAN | AC_FULL | 182.0 ± 103.2 | 0.600 ± 0.381 | 242.2 | 0.840 |
| CLEAN | AC_LITE | 185.8 ± 94.7 | 0.610 ± 0.349 | 242.2 | 0.840 |
| CLEAN | BASELINE | 263.2 ± 14.1 | 0.890 ± 0.065 | 282.6 | 0.980 |
| OBS_NOISE | AC_FULL | 6.8 ± 9.9 | 0.010 ± 0.022 | 101.4 | 0.300 |
| OBS_NOISE | AC_LITE | -13.1 ± 31.9 | 0.010 ± 0.022 | 113.7 | 0.340 |
| OBS_NOISE | BASELINE | 148.7 ± 16.8 | 0.450 ± 0.035 | 187.5 | 0.590 |
| REWARD_NOISE | AC_FULL | 45.3 ± 80.7 | 0.150 ± 0.224 | 202.0 | 0.570 |
| REWARD_NOISE | AC_LITE | 38.8 ± 53.4 | 0.100 ± 0.170 | 206.8 | 0.580 |
| REWARD_NOISE | BASELINE | 250.6 ± 16.2 | 0.670 ± 0.097 | 278.1 | 0.820 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL | 0 | 88.1 | 0.250 |
| CLEAN | AC_FULL | 3 | 56.7 | 0.150 |
| CLEAN | AC_FULL | 9 | 278.6 | 0.950 |
| CLEAN | AC_FULL | 17 | 218.0 | 0.700 |
| CLEAN | AC_FULL | 42 | 268.4 | 0.950 |
| CLEAN | AC_LITE | 0 | 132.5 | 0.400 |
| CLEAN | AC_LITE | 3 | 58.2 | 0.150 |
| CLEAN | AC_LITE | 9 | 278.6 | 0.950 |
| CLEAN | AC_LITE | 17 | 183.4 | 0.600 |
| CLEAN | AC_LITE | 42 | 276.0 | 0.950 |
| CLEAN | BASELINE | 0 | 273.0 | 0.950 |
| CLEAN | BASELINE | 3 | 282.2 | 0.950 |
| CLEAN | BASELINE | 9 | 250.3 | 0.800 |
| CLEAN | BASELINE | 17 | 250.3 | 0.850 |
| CLEAN | BASELINE | 42 | 259.9 | 0.900 |
| OBS_NOISE | AC_FULL | 0 | 2.1 | 0.000 |
| OBS_NOISE | AC_FULL | 3 | 8.4 | 0.000 |
| OBS_NOISE | AC_FULL | 9 | 13.0 | 0.050 |
| OBS_NOISE | AC_FULL | 17 | -7.5 | 0.000 |
| OBS_NOISE | AC_FULL | 42 | 17.9 | 0.000 |
| OBS_NOISE | AC_LITE | 0 | 12.4 | 0.000 |
| OBS_NOISE | AC_LITE | 3 | -18.9 | 0.000 |
| OBS_NOISE | AC_LITE | 9 | -18.6 | 0.000 |
| OBS_NOISE | AC_LITE | 17 | -60.5 | 0.000 |
| OBS_NOISE | AC_LITE | 42 | 20.1 | 0.050 |
| OBS_NOISE | BASELINE | 0 | 154.5 | 0.450 |
| OBS_NOISE | BASELINE | 3 | 129.3 | 0.400 |
| OBS_NOISE | BASELINE | 9 | 132.5 | 0.450 |
| OBS_NOISE | BASELINE | 17 | 160.3 | 0.450 |
| OBS_NOISE | BASELINE | 42 | 166.8 | 0.500 |
| REWARD_NOISE | AC_FULL | 0 | 88.1 | 0.250 |
| REWARD_NOISE | AC_FULL | 3 | 166.8 | 0.500 |
| REWARD_NOISE | AC_FULL | 9 | -2.0 | 0.000 |
| REWARD_NOISE | AC_FULL | 17 | -27.1 | 0.000 |
| REWARD_NOISE | AC_FULL | 42 | 0.8 | 0.000 |
| REWARD_NOISE | AC_LITE | 0 | 132.5 | 0.400 |
| REWARD_NOISE | AC_LITE | 3 | 21.5 | 0.050 |
| REWARD_NOISE | AC_LITE | 9 | 29.1 | 0.050 |
| REWARD_NOISE | AC_LITE | 17 | 5.1 | 0.000 |
| REWARD_NOISE | AC_LITE | 42 | 5.6 | 0.000 |
| REWARD_NOISE | BASELINE | 0 | 262.1 | 0.750 |
| REWARD_NOISE | BASELINE | 3 | 253.6 | 0.700 |
| REWARD_NOISE | BASELINE | 9 | 222.6 | 0.500 |
| REWARD_NOISE | BASELINE | 17 | 261.1 | 0.700 |
| REWARD_NOISE | BASELINE | 42 | 253.8 | 0.700 |

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
| CLEAN | BASELINE | 0 | BASELINE_CLEAN_seed0_final.pt | 287.0 | 1.000 |
| CLEAN | BASELINE | 3 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | BASELINE | 9 | BASELINE_CLEAN_seed9_step0010000.pt | 286.6 | 1.000 |
| CLEAN | BASELINE | 17 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | BASELINE | 42 | BASELINE_CLEAN_seed42_final.pt | 288.4 | 1.000 |
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
| OBS_NOISE | BASELINE | 0 | BASELINE_OBS_NOISE_seed0_step0010000.pt | 251.1 | 0.800 |
| OBS_NOISE | BASELINE | 3 | checkpoint_0_pretrained | 213.0 | 0.600 |
| OBS_NOISE | BASELINE | 9 | BASELINE_OBS_NOISE_seed9_final.pt | 239.1 | 0.733 |
| OBS_NOISE | BASELINE | 17 | checkpoint_0_pretrained | 213.0 | 0.600 |
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
| REWARD_NOISE | BASELINE | 0 | BASELINE_REWARD_NOISE_seed0_step0010000.pt | 286.6 | 1.000 |
| REWARD_NOISE | BASELINE | 3 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE | 9 | BASELINE_REWARD_NOISE_seed9_final.pt | 290.9 | 1.000 |
| REWARD_NOISE | BASELINE | 17 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE | 42 | BASELINE_REWARD_NOISE_seed42_step0010000.pt | 287.8 | 1.000 |

## Certainty AUROC diagnostics

- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.
- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\) (diagnostic).

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL | 0.409 | 0.467 |
| CLEAN | AC_LITE | 0.361 | 0.476 |
| OBS_NOISE | AC_FULL | 0.312 | 0.462 |
| OBS_NOISE | AC_LITE | 0.126 | 0.515 |
| REWARD_NOISE | AC_FULL | 0.300 | 0.476 |
| REWARD_NOISE | AC_LITE | 0.197 | 0.511 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

