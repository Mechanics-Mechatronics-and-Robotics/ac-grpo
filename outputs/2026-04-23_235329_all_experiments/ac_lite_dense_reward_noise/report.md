# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-23_235329_all_experiments\ac_lite_dense_reward_noise`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Experiment protocol

- Implemented method variants in this run: AC_LITE_DENSE
- Training modes: REWARD_NOISE
- Reward modes represented: DENSE
- Training seeds: 0, 3, 9, 17, 42
- All branches start from the shared pretrained anchor when a pretrained path is provided.
- Training uses grouped rollouts with dynamic sampling fallback when no mixed-outcome groups are available.
- Checkpoints are saved during training and the pretrained anchor is treated as checkpoint 0.
- Checkpoint selection uses greedy held-out evaluation under the branch's primary selection condition.
- Additional challenge tests evaluate selected checkpoints under CLEAN, OBS_NOISE (typical), and OBS_NOISE (hard).
- Challenge tests currently use up to 100 episodes per evaluation seed in the generated logs.

## Summary table (mean ± std over seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | reward AUC | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|---:|
| REWARD_NOISE | AC_LITE_DENSE | 242.0 ± 31.3 | 0.690 ± 0.139 | 233.7 ± 18.9 | 289.8 | 0.970 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| REWARD_NOISE | AC_LITE_DENSE | 0 | 218.2 | 0.550 |
| REWARD_NOISE | AC_LITE_DENSE | 3 | 261.1 | 0.700 |
| REWARD_NOISE | AC_LITE_DENSE | 9 | 274.1 | 0.800 |
| REWARD_NOISE | AC_LITE_DENSE | 17 | 256.7 | 0.850 |
| REWARD_NOISE | AC_LITE_DENSE | 42 | 200.2 | 0.550 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| REWARD_NOISE | AC_LITE_DENSE | 0 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 3 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 9 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 17 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 42 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| REWARD_NOISE | AC_LITE_DENSE | test_clean | CLEAN | 0.10 | 277.4 ± 20.6 | 1.000 ± 0.000 |
| REWARD_NOISE | AC_LITE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 212.1 ± 110.7 | 0.710 ± 0.454 |
| REWARD_NOISE | AC_LITE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 21.7 ± 30.8 | 0.000 ± 0.000 |

## Cross-test summary on OBS evaluation

| Method | Train: CLEAN → Test OBS | Train: OBS → Test OBS | Train: REWARD → Test OBS |
|---|---:|---:|---:|
| Baseline (Sparse) | n/a | n/a | n/a |
| Baseline (Dense) | n/a | n/a | n/a |
| AC-LITE (Sparse) | n/a | n/a | n/a |
| AC-LITE (Dense) | n/a | n/a | 212.1 ± 110.7 / 0.710 ± 0.454 |
| AC-FULL (Sparse) | n/a | n/a | n/a |

## Episode-level certainty summary

| mode | method | mean c_i | corr(c, delta) | corr(c, action prob) | corr(c, runner-up prob) |
|---|---|---:|---:|---:|---:|
| REWARD_NOISE | AC_LITE_DENSE | 0.866 | 0.315 | 0.379 | -0.287 |

## Certainty AUROC diagnostics

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| REWARD_NOISE | AC_LITE_DENSE | 0.419 | 0.309 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- REWARD_NOISE: `AC_LITE_DENSE` has the highest mean final return (242.0) with mean final success 0.690.

Checkpoint selection versus the pretrained anchor:

- REWARD_NOISE / AC_LITE_DENSE: checkpoint 0 wins in 5 of 5 seeds (1.00).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `AC_LITE_DENSE` in `REWARD_NOISE` with 277.4 ± 20.6 and success 1.000 ± 0.000.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `AC_LITE_DENSE` in `REWARD_NOISE` with 212.1 ± 110.7 and success 0.710 ± 0.454.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `AC_LITE_DENSE` in `REWARD_NOISE` with 21.7 ± 30.8 and success 0.000 ± 0.000.

Episode-level certainty behavior:

- REWARD_NOISE / AC_LITE_DENSE: mean episode certainty 0.866, mean corr(certainty, delta) 0.315, mean corr(certainty, action_prob) 0.379, mean corr(certainty, runner_up_prob) -0.287.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

