# RL Experiment Report

This report summarizes the selected sweep from the CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-19_165107_all_experiments`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Notes on experimental modes

- **REWARD_NOISE**: false-negative successes also penalize the terminal rollout reward used by PPO/GAE, so the policy update sees the corruption.
- **OBS_NOISE**: adds Gaussian noise \(\sigma=0.1\) to observations at every step.

## Seed aggregation

Learning curves are computed **per seed** and then aggregated (mean ± std). This avoids interleaving seeds (each run resets `step` to 0).

## Summary table (mean ± std over 5 seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL | 279.2 ± 8.2 | 0.980 ± 0.027 | 294.8 | 1.000 |
| CLEAN | AC_LITE | 266.1 ± 17.5 | 0.890 ± 0.074 | 294.6 | 1.000 |
| CLEAN | BASELINE | 282.4 ± 9.0 | 0.970 ± 0.045 | 294.1 | 1.000 |
| OBS_NOISE | AC_FULL | 172.8 ± 44.6 | 0.610 ± 0.152 | 282.9 | 1.000 |
| OBS_NOISE | AC_LITE | 212.0 ± 32.4 | 0.780 ± 0.168 | 286.1 | 1.000 |
| OBS_NOISE | BASELINE | 211.4 ± 35.0 | 0.700 ± 0.294 | 285.2 | 1.000 |
| REWARD_NOISE | AC_FULL | 259.5 ± 19.5 | 0.670 ± 0.152 | 294.4 | 0.980 |
| REWARD_NOISE | AC_LITE | 270.2 ± 9.2 | 0.760 ± 0.129 | 293.7 | 0.980 |
| REWARD_NOISE | BASELINE | 272.8 ± 17.9 | 0.780 ± 0.115 | 292.9 | 0.990 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL | 0 | 270.4 | 0.950 |
| CLEAN | AC_FULL | 3 | 271.0 | 0.950 |
| CLEAN | AC_FULL | 9 | 288.6 | 1.000 |
| CLEAN | AC_FULL | 17 | 281.0 | 1.000 |
| CLEAN | AC_FULL | 42 | 284.8 | 1.000 |
| CLEAN | AC_LITE | 0 | 262.2 | 0.850 |
| CLEAN | AC_LITE | 3 | 270.4 | 0.900 |
| CLEAN | AC_LITE | 9 | 289.3 | 1.000 |
| CLEAN | AC_LITE | 17 | 240.6 | 0.800 |
| CLEAN | AC_LITE | 42 | 268.0 | 0.900 |
| CLEAN | BASELINE | 0 | 288.3 | 1.000 |
| CLEAN | BASELINE | 3 | 286.1 | 1.000 |
| CLEAN | BASELINE | 9 | 279.2 | 0.950 |
| CLEAN | BASELINE | 17 | 290.3 | 1.000 |
| CLEAN | BASELINE | 42 | 268.1 | 0.900 |
| OBS_NOISE | AC_FULL | 0 | 160.5 | 0.550 |
| OBS_NOISE | AC_FULL | 3 | 191.9 | 0.650 |
| OBS_NOISE | AC_FULL | 9 | 110.1 | 0.450 |
| OBS_NOISE | AC_FULL | 17 | 231.8 | 0.850 |
| OBS_NOISE | AC_FULL | 42 | 169.7 | 0.550 |
| OBS_NOISE | AC_LITE | 0 | 235.2 | 0.900 |
| OBS_NOISE | AC_LITE | 3 | 184.4 | 0.650 |
| OBS_NOISE | AC_LITE | 9 | 186.2 | 0.600 |
| OBS_NOISE | AC_LITE | 17 | 197.3 | 0.750 |
| OBS_NOISE | AC_LITE | 42 | 257.0 | 1.000 |
| OBS_NOISE | BASELINE | 0 | 192.7 | 0.700 |
| OBS_NOISE | BASELINE | 3 | 251.2 | 0.950 |
| OBS_NOISE | BASELINE | 9 | 161.1 | 0.200 |
| OBS_NOISE | BASELINE | 17 | 223.6 | 0.800 |
| OBS_NOISE | BASELINE | 42 | 228.2 | 0.850 |
| REWARD_NOISE | AC_FULL | 0 | 237.4 | 0.400 |
| REWARD_NOISE | AC_FULL | 3 | 273.2 | 0.700 |
| REWARD_NOISE | AC_FULL | 9 | 261.5 | 0.750 |
| REWARD_NOISE | AC_FULL | 17 | 242.5 | 0.750 |
| REWARD_NOISE | AC_FULL | 42 | 282.8 | 0.750 |
| REWARD_NOISE | AC_LITE | 0 | 264.8 | 0.600 |
| REWARD_NOISE | AC_LITE | 3 | 281.6 | 0.800 |
| REWARD_NOISE | AC_LITE | 9 | 278.1 | 0.850 |
| REWARD_NOISE | AC_LITE | 17 | 266.3 | 0.900 |
| REWARD_NOISE | AC_LITE | 42 | 260.0 | 0.650 |
| REWARD_NOISE | BASELINE | 0 | 251.1 | 0.650 |
| REWARD_NOISE | BASELINE | 3 | 255.5 | 0.700 |
| REWARD_NOISE | BASELINE | 9 | 287.4 | 0.800 |
| REWARD_NOISE | BASELINE | 17 | 283.5 | 0.800 |
| REWARD_NOISE | BASELINE | 42 | 286.4 | 0.950 |

## Best checkpoint by greedy held-out evaluation

Checkpoints are evaluated greedily on fixed held-out seeds. The final checkpoint is not assumed to be best.

| mode | method | seed | checkpoint | eval return | eval success |
|---|---|---:|---|---:|---:|
| CLEAN | AC_FULL | 0 | AC_FULL_CLEAN_seed0_step0330000_policy.pt | 291.8 | 1.000 |
| CLEAN | AC_FULL | 3 | AC_FULL_CLEAN_seed3_step0470000_policy.pt | 291.6 | 1.000 |
| CLEAN | AC_FULL | 9 | AC_FULL_CLEAN_seed9_step0070000_policy.pt | 292.5 | 1.000 |
| CLEAN | AC_FULL | 17 | AC_FULL_CLEAN_seed17_step0320000_policy.pt | 290.7 | 1.000 |
| CLEAN | AC_FULL | 42 | AC_FULL_CLEAN_seed42_step0100000_policy.pt | 290.4 | 1.000 |
| CLEAN | AC_LITE | 0 | AC_LITE_CLEAN_seed0_step0470000_policy.pt | 290.8 | 1.000 |
| CLEAN | AC_LITE | 3 | AC_LITE_CLEAN_seed3_step0390000_policy.pt | 292.8 | 1.000 |
| CLEAN | AC_LITE | 9 | AC_LITE_CLEAN_seed9_step0230000_policy.pt | 291.9 | 1.000 |
| CLEAN | AC_LITE | 17 | AC_LITE_CLEAN_seed17_step0290000_policy.pt | 291.7 | 1.000 |
| CLEAN | AC_LITE | 42 | AC_LITE_CLEAN_seed42_step0380000_policy.pt | 291.0 | 1.000 |
| CLEAN | BASELINE | 0 | BASELINE_CLEAN_seed0_step0480000.pt | 291.7 | 1.000 |
| CLEAN | BASELINE | 3 | BASELINE_CLEAN_seed3_step0390000.pt | 291.6 | 1.000 |
| CLEAN | BASELINE | 9 | BASELINE_CLEAN_seed9_step0200000.pt | 291.2 | 1.000 |
| CLEAN | BASELINE | 17 | BASELINE_CLEAN_seed17_step0410000.pt | 292.0 | 1.000 |
| CLEAN | BASELINE | 42 | BASELINE_CLEAN_seed42_step0480000.pt | 290.1 | 1.000 |
| OBS_NOISE | AC_FULL | 0 | AC_FULL_OBS_NOISE_seed0_step0130000_policy.pt | 283.1 | 1.000 |
| OBS_NOISE | AC_FULL | 3 | AC_FULL_OBS_NOISE_seed3_step0140000_policy.pt | 278.4 | 1.000 |
| OBS_NOISE | AC_FULL | 9 | AC_FULL_OBS_NOISE_seed9_step0130000_policy.pt | 284.4 | 1.000 |
| OBS_NOISE | AC_FULL | 17 | AC_FULL_OBS_NOISE_seed17_step0180000_policy.pt | 281.3 | 1.000 |
| OBS_NOISE | AC_FULL | 42 | AC_FULL_OBS_NOISE_seed42_step0040000_policy.pt | 278.1 | 0.933 |
| OBS_NOISE | AC_LITE | 0 | AC_LITE_OBS_NOISE_seed0_step0090000_policy.pt | 273.5 | 0.933 |
| OBS_NOISE | AC_LITE | 3 | AC_LITE_OBS_NOISE_seed3_step0180000_policy.pt | 284.4 | 1.000 |
| OBS_NOISE | AC_LITE | 9 | AC_LITE_OBS_NOISE_seed9_step0150000_policy.pt | 280.3 | 1.000 |
| OBS_NOISE | AC_LITE | 17 | AC_LITE_OBS_NOISE_seed17_step0150000_policy.pt | 284.4 | 1.000 |
| OBS_NOISE | AC_LITE | 42 | AC_LITE_OBS_NOISE_seed42_step0160000_policy.pt | 287.1 | 1.000 |
| OBS_NOISE | BASELINE | 0 | BASELINE_OBS_NOISE_seed0_step0170000.pt | 276.8 | 0.933 |
| OBS_NOISE | BASELINE | 3 | BASELINE_OBS_NOISE_seed3_step0050000.pt | 287.0 | 1.000 |
| OBS_NOISE | BASELINE | 9 | BASELINE_OBS_NOISE_seed9_step0160000.pt | 276.1 | 1.000 |
| OBS_NOISE | BASELINE | 17 | BASELINE_OBS_NOISE_seed17_step0150000.pt | 285.1 | 1.000 |
| OBS_NOISE | BASELINE | 42 | BASELINE_OBS_NOISE_seed42_step0050000.pt | 287.2 | 1.000 |
| REWARD_NOISE | AC_FULL | 0 | AC_FULL_REWARD_NOISE_seed0_step0240000_policy.pt | 291.2 | 1.000 |
| REWARD_NOISE | AC_FULL | 3 | AC_FULL_REWARD_NOISE_seed3_step0490000_policy.pt | 291.5 | 1.000 |
| REWARD_NOISE | AC_FULL | 9 | AC_FULL_REWARD_NOISE_seed9_step0420000_policy.pt | 292.9 | 1.000 |
| REWARD_NOISE | AC_FULL | 17 | AC_FULL_REWARD_NOISE_seed17_step0010000_policy.pt | 288.8 | 1.000 |
| REWARD_NOISE | AC_FULL | 42 | AC_FULL_REWARD_NOISE_seed42_step0440000_policy.pt | 289.5 | 1.000 |
| REWARD_NOISE | AC_LITE | 0 | AC_LITE_REWARD_NOISE_seed0_step0280000_policy.pt | 291.2 | 1.000 |
| REWARD_NOISE | AC_LITE | 3 | AC_LITE_REWARD_NOISE_seed3_step0270000_policy.pt | 292.0 | 1.000 |
| REWARD_NOISE | AC_LITE | 9 | AC_LITE_REWARD_NOISE_seed9_step0410000_policy.pt | 292.5 | 1.000 |
| REWARD_NOISE | AC_LITE | 17 | AC_LITE_REWARD_NOISE_seed17_step0430000_policy.pt | 290.7 | 1.000 |
| REWARD_NOISE | AC_LITE | 42 | AC_LITE_REWARD_NOISE_seed42_step0450000_policy.pt | 293.0 | 1.000 |
| REWARD_NOISE | BASELINE | 0 | BASELINE_REWARD_NOISE_seed0_step0210000.pt | 290.6 | 1.000 |
| REWARD_NOISE | BASELINE | 3 | BASELINE_REWARD_NOISE_seed3_step0270000.pt | 291.2 | 1.000 |
| REWARD_NOISE | BASELINE | 9 | BASELINE_REWARD_NOISE_seed9_step0090000.pt | 290.9 | 1.000 |
| REWARD_NOISE | BASELINE | 17 | BASELINE_REWARD_NOISE_seed17_step0410000.pt | 291.2 | 1.000 |
| REWARD_NOISE | BASELINE | 42 | BASELINE_REWARD_NOISE_seed42_step0420000.pt | 292.5 | 1.000 |

## Certainty AUROC diagnostics

- **Trajectory AUROC**: episode success predicted by mean certainty over the trajectory.
- **Timestep AUROC**: late-phase indicator predicted by \(1 - certainty\) (diagnostic).

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL | 0.721 | 0.207 |
| CLEAN | AC_LITE | 0.587 | 0.173 |
| OBS_NOISE | AC_FULL | 0.769 | 0.272 |
| OBS_NOISE | AC_LITE | 0.656 | 0.233 |
| REWARD_NOISE | AC_FULL | 0.608 | 0.264 |
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

