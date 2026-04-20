# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-20_102459_all_experiments`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Notes on experimental modes

- **Reward semantics**: PPO/GAE uses sparse terminal binary reward only (`0` before termination, terminal `policy_success` at episode end); dense LunarLander return is logged for diagnostics only.
- **REWARD_NOISE**: false-negative successes set terminal `policy_success` to `0`, so the sparse policy update sees the corrupted outcome directly.
- **OBS_NOISE**: adds Gaussian noise \(\sigma=0.1\) to observations at every step.

## Seed aggregation

Learning curves are computed **per seed** and then aggregated (mean ± std). This avoids interleaving seeds (each run resets `step` to 0).

## Summary table (mean ± std over 5 seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL | 271.8 ± 19.5 | 0.950 ± 0.061 | 290.1 | 1.000 |
| CLEAN | AC_LITE | 268.8 ± 6.1 | 0.940 ± 0.042 | 288.1 | 1.000 |
| CLEAN | BASELINE | 269.6 ± 12.4 | 0.950 ± 0.035 | 289.0 | 1.000 |
| OBS_NOISE | AC_FULL | 136.8 ± 30.6 | 0.320 ± 0.135 | 206.1 | 0.690 |
| OBS_NOISE | AC_LITE | 114.8 ± 27.6 | 0.310 ± 0.119 | 210.3 | 0.710 |
| OBS_NOISE | BASELINE | 150.8 ± 38.5 | 0.410 ± 0.164 | 229.0 | 0.780 |
| REWARD_NOISE | AC_FULL | 261.7 ± 13.9 | 0.720 ± 0.144 | 286.3 | 0.900 |
| REWARD_NOISE | AC_LITE | 271.5 ± 9.1 | 0.720 ± 0.084 | 286.0 | 0.910 |
| REWARD_NOISE | BASELINE | 260.0 ± 12.9 | 0.670 ± 0.104 | 289.5 | 0.940 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL | 0 | 287.8 | 1.000 |
| CLEAN | AC_FULL | 3 | 273.9 | 0.950 |
| CLEAN | AC_FULL | 9 | 238.9 | 0.850 |
| CLEAN | AC_FULL | 17 | 273.3 | 0.950 |
| CLEAN | AC_FULL | 42 | 285.2 | 1.000 |
| CLEAN | AC_LITE | 0 | 271.9 | 0.950 |
| CLEAN | AC_LITE | 3 | 260.8 | 0.900 |
| CLEAN | AC_LITE | 9 | 263.9 | 0.900 |
| CLEAN | AC_LITE | 17 | 275.1 | 1.000 |
| CLEAN | AC_LITE | 42 | 272.1 | 0.950 |
| CLEAN | BASELINE | 0 | 286.8 | 1.000 |
| CLEAN | BASELINE | 3 | 252.1 | 0.900 |
| CLEAN | BASELINE | 9 | 267.8 | 0.950 |
| CLEAN | BASELINE | 17 | 272.7 | 0.950 |
| CLEAN | BASELINE | 42 | 268.3 | 0.950 |
| OBS_NOISE | AC_FULL | 0 | 98.1 | 0.100 |
| OBS_NOISE | AC_FULL | 3 | 158.9 | 0.400 |
| OBS_NOISE | AC_FULL | 9 | 116.1 | 0.350 |
| OBS_NOISE | AC_FULL | 17 | 137.7 | 0.300 |
| OBS_NOISE | AC_FULL | 42 | 173.4 | 0.450 |
| OBS_NOISE | AC_LITE | 0 | 105.2 | 0.300 |
| OBS_NOISE | AC_LITE | 3 | 132.3 | 0.400 |
| OBS_NOISE | AC_LITE | 9 | 149.4 | 0.450 |
| OBS_NOISE | AC_LITE | 17 | 110.1 | 0.250 |
| OBS_NOISE | AC_LITE | 42 | 77.1 | 0.150 |
| OBS_NOISE | BASELINE | 0 | 180.8 | 0.550 |
| OBS_NOISE | BASELINE | 3 | 116.7 | 0.250 |
| OBS_NOISE | BASELINE | 9 | 193.2 | 0.600 |
| OBS_NOISE | BASELINE | 17 | 105.7 | 0.250 |
| OBS_NOISE | BASELINE | 42 | 157.7 | 0.400 |
| REWARD_NOISE | AC_FULL | 0 | 265.0 | 0.550 |
| REWARD_NOISE | AC_FULL | 3 | 281.4 | 0.900 |
| REWARD_NOISE | AC_FULL | 9 | 242.7 | 0.600 |
| REWARD_NOISE | AC_FULL | 17 | 260.8 | 0.800 |
| REWARD_NOISE | AC_FULL | 42 | 258.7 | 0.750 |
| REWARD_NOISE | AC_LITE | 0 | 265.0 | 0.650 |
| REWARD_NOISE | AC_LITE | 3 | 271.7 | 0.700 |
| REWARD_NOISE | AC_LITE | 9 | 286.3 | 0.650 |
| REWARD_NOISE | AC_LITE | 17 | 263.3 | 0.750 |
| REWARD_NOISE | AC_LITE | 42 | 271.3 | 0.850 |
| REWARD_NOISE | BASELINE | 0 | 274.9 | 0.600 |
| REWARD_NOISE | BASELINE | 3 | 252.4 | 0.650 |
| REWARD_NOISE | BASELINE | 9 | 252.7 | 0.550 |
| REWARD_NOISE | BASELINE | 17 | 247.0 | 0.750 |
| REWARD_NOISE | BASELINE | 42 | 272.7 | 0.800 |

## Best checkpoint by greedy held-out evaluation

Checkpoints are evaluated greedily on fixed held-out seeds. The final checkpoint is not assumed to be best.

| mode | method | seed | checkpoint | eval return | eval success |
|---|---|---:|---|---:|---:|
| CLEAN | AC_FULL | 0 | AC_FULL_CLEAN_seed0_step0020000_policy.pt | 287.9 | 1.000 |
| CLEAN | AC_FULL | 3 | AC_FULL_CLEAN_seed3_step0020000_policy.pt | 289.5 | 1.000 |
| CLEAN | AC_FULL | 9 | AC_FULL_CLEAN_seed9_step0030000_policy.pt | 289.4 | 1.000 |
| CLEAN | AC_FULL | 17 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_FULL | 42 | AC_FULL_CLEAN_seed42_step0020000_policy.pt | 286.8 | 1.000 |
| CLEAN | AC_LITE | 0 | AC_LITE_CLEAN_seed0_step0020000_policy.pt | 286.1 | 1.000 |
| CLEAN | AC_LITE | 3 | checkpoint_0_pretrained | 285.9 | 1.000 |
| CLEAN | AC_LITE | 9 | AC_LITE_CLEAN_seed9_step0010000_policy.pt | 287.8 | 1.000 |
| CLEAN | AC_LITE | 17 | AC_LITE_CLEAN_seed17_step0010000_policy.pt | 286.0 | 1.000 |
| CLEAN | AC_LITE | 42 | AC_LITE_CLEAN_seed42_step0030000_policy.pt | 287.1 | 1.000 |
| CLEAN | BASELINE | 0 | BASELINE_CLEAN_seed0_step0010000.pt | 288.0 | 1.000 |
| CLEAN | BASELINE | 3 | BASELINE_CLEAN_seed3_step0020000.pt | 286.3 | 1.000 |
| CLEAN | BASELINE | 9 | BASELINE_CLEAN_seed9_step0010000.pt | 290.9 | 1.000 |
| CLEAN | BASELINE | 17 | BASELINE_CLEAN_seed17_step0030000.pt | 286.9 | 1.000 |
| CLEAN | BASELINE | 42 | checkpoint_0_pretrained | 285.9 | 1.000 |
| OBS_NOISE | AC_FULL | 0 | checkpoint_0_pretrained | 213.0 | 0.600 |
| OBS_NOISE | AC_FULL | 3 | AC_FULL_OBS_NOISE_seed3_step0010000_policy.pt | 254.8 | 0.867 |
| OBS_NOISE | AC_FULL | 9 | AC_FULL_OBS_NOISE_seed9_step0010000_policy.pt | 219.3 | 0.667 |
| OBS_NOISE | AC_FULL | 17 | AC_FULL_OBS_NOISE_seed17_step0010000_policy.pt | 239.1 | 0.733 |
| OBS_NOISE | AC_FULL | 42 | AC_FULL_OBS_NOISE_seed42_step0040000_policy.pt | 237.4 | 0.733 |
| OBS_NOISE | AC_LITE | 0 | checkpoint_0_pretrained | 213.0 | 0.600 |
| OBS_NOISE | AC_LITE | 3 | AC_LITE_OBS_NOISE_seed3_final_policy.pt | 250.7 | 0.867 |
| OBS_NOISE | AC_LITE | 9 | AC_LITE_OBS_NOISE_seed9_step0020000_policy.pt | 231.8 | 0.733 |
| OBS_NOISE | AC_LITE | 17 | AC_LITE_OBS_NOISE_seed17_step0010000_policy.pt | 223.1 | 0.600 |
| OBS_NOISE | AC_LITE | 42 | checkpoint_0_pretrained | 213.0 | 0.600 |
| OBS_NOISE | BASELINE | 0 | BASELINE_OBS_NOISE_seed0_step0030000.pt | 264.5 | 0.867 |
| OBS_NOISE | BASELINE | 3 | BASELINE_OBS_NOISE_seed3_step0030000.pt | 216.5 | 0.733 |
| OBS_NOISE | BASELINE | 9 | BASELINE_OBS_NOISE_seed9_step0040000.pt | 244.0 | 0.800 |
| OBS_NOISE | BASELINE | 17 | BASELINE_OBS_NOISE_seed17_step0010000.pt | 230.3 | 0.667 |
| OBS_NOISE | BASELINE | 42 | BASELINE_OBS_NOISE_seed42_step0030000.pt | 260.8 | 0.867 |
| REWARD_NOISE | AC_FULL | 0 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | AC_FULL | 3 | AC_FULL_REWARD_NOISE_seed3_step0020000_policy.pt | 289.0 | 1.000 |
| REWARD_NOISE | AC_FULL | 9 | AC_FULL_REWARD_NOISE_seed9_final_policy.pt | 291.0 | 1.000 |
| REWARD_NOISE | AC_FULL | 17 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | AC_FULL | 42 | AC_FULL_REWARD_NOISE_seed42_step0020000_policy.pt | 290.4 | 1.000 |
| REWARD_NOISE | AC_LITE | 0 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE | 3 | AC_LITE_REWARD_NOISE_seed3_step0010000_policy.pt | 288.4 | 1.000 |
| REWARD_NOISE | AC_LITE | 9 | AC_LITE_REWARD_NOISE_seed9_step0010000_policy.pt | 289.7 | 1.000 |
| REWARD_NOISE | AC_LITE | 17 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE | 42 | AC_LITE_REWARD_NOISE_seed42_step0010000_policy.pt | 289.4 | 1.000 |
| REWARD_NOISE | BASELINE | 0 | checkpoint_0_pretrained | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE | 3 | BASELINE_REWARD_NOISE_seed3_step0030000.pt | 286.8 | 1.000 |
| REWARD_NOISE | BASELINE | 9 | BASELINE_REWARD_NOISE_seed9_step0010000.pt | 288.5 | 1.000 |
| REWARD_NOISE | BASELINE | 17 | BASELINE_REWARD_NOISE_seed17_step0010000.pt | 289.9 | 1.000 |
| REWARD_NOISE | BASELINE | 42 | BASELINE_REWARD_NOISE_seed42_step0020000.pt | 290.9 | 1.000 |

## Certainty AUROC diagnostics

- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.
- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\) (diagnostic).

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL | 0.595 | 0.264 |
| CLEAN | AC_LITE | 0.608 | 0.191 |
| OBS_NOISE | AC_FULL | 0.753 | 0.446 |
| OBS_NOISE | AC_LITE | 0.739 | 0.415 |
| REWARD_NOISE | AC_FULL | 0.512 | 0.282 |
| REWARD_NOISE | AC_LITE | 0.536 | 0.215 |

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

