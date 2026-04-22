# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-22_231316_ac_full_sparse_clean\ac_full_sparse_clean`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Experiment protocol

- Implemented method variants in this run: AC_FULL_SPARSE
- Training modes: CLEAN
- Reward modes represented: SPARSE
- Training seeds: 42
- All branches start from the shared pretrained anchor when a pretrained path is provided.
- Training uses grouped rollouts with dynamic sampling fallback when no mixed-outcome groups are available.
- Checkpoints are saved during training and the pretrained anchor is treated as checkpoint 0.
- Checkpoint selection uses greedy held-out evaluation under the branch's primary selection condition.
- Additional challenge tests evaluate selected checkpoints under CLEAN, OBS_NOISE (typical), and OBS_NOISE (hard).
- Challenge tests currently use up to 20 episodes per evaluation seed in the generated logs.

## Summary table (mean ± std over seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | n/a | n/a | 278.2 | 1.000 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 42 | 278.2 | 1.000 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| CLEAN | AC_FULL_SPARSE | 42 | AC_FULL_SPARSE_CLEAN_seed42_final_policy.pt | CLEAN | 285.9 | 1.000 |

## Episode-level certainty summary

| mode | method | mean c_i | corr(c, delta) | corr(c, action prob) | corr(c, runner-up prob) |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0.497 | 0.434 | 0.482 | -0.439 |

## Certainty AUROC diagnostics

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL_SPARSE | n/a | 0.082 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- CLEAN: `AC_FULL_SPARSE` has the highest mean final return (278.2) with mean final success 1.000.

Checkpoint selection versus the pretrained anchor:

- CLEAN / AC_FULL_SPARSE: checkpoint 0 wins in 0 of 1 seeds (0.00).

Episode-level certainty behavior:

- CLEAN / AC_FULL_SPARSE: mean episode certainty 0.497, mean corr(certainty, delta) 0.434, mean corr(certainty, action_prob) 0.482, mean corr(certainty, runner_up_prob) -0.439.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

