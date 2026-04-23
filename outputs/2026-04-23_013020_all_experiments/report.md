# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-23_013020_all_experiments`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Experiment protocol

- Implemented method variants in this run: AC_FULL_SPARSE, AC_LITE_DENSE, AC_LITE_SPARSE, BASELINE_DENSE, BASELINE_SPARSE
- Training modes: CLEAN, OBS_NOISE, REWARD_NOISE
- Reward modes represented: DENSE, SPARSE
- Training seeds: 0, 3, 9, 17, 42
- All branches start from the shared pretrained anchor when a pretrained path is provided.
- Training uses grouped rollouts with dynamic sampling fallback when no mixed-outcome groups are available.
- Checkpoints are saved during training and the pretrained anchor is treated as checkpoint 0.
- Checkpoint selection uses greedy held-out evaluation under the branch's primary selection condition.
- Additional challenge tests evaluate selected checkpoints under CLEAN, OBS_NOISE (typical), and OBS_NOISE (hard).
- Challenge tests currently use up to 2800 episodes per evaluation seed in the generated logs.

## Summary table (mean ± std over seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 270.8 ± 12.2 | 0.940 ± 0.065 | 292.2 | 1.000 |
| CLEAN | AC_LITE_DENSE | 243.2 ± 26.3 | 0.870 ± 0.115 | 286.7 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 262.4 ± 11.7 | 0.910 ± 0.096 | 289.8 | 1.000 |
| CLEAN | BASELINE_DENSE | 208.1 ± 19.1 | 0.720 ± 0.130 | 289.4 | 1.000 |
| CLEAN | BASELINE_SPARSE | 259.6 ± 8.7 | 0.910 ± 0.042 | 285.7 | 1.000 |
| OBS_NOISE | AC_FULL_SPARSE | 194.5 ± 47.7 | 0.710 ± 0.119 | 284.8 | 1.000 |
| OBS_NOISE | AC_LITE_DENSE | 254.5 ± 7.2 | 0.950 ± 0.035 | 285.9 | 1.000 |
| OBS_NOISE | AC_LITE_SPARSE | 220.5 ± 34.7 | 0.820 ± 0.135 | 282.7 | 1.000 |
| OBS_NOISE | BASELINE_DENSE | 181.6 ± 19.8 | 0.640 ± 0.156 | 282.7 | 1.000 |
| OBS_NOISE | BASELINE_SPARSE | 225.2 ± 16.2 | 0.810 ± 0.096 | 284.8 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 242.4 ± 13.7 | 0.610 ± 0.089 | 282.8 | 0.930 |
| REWARD_NOISE | AC_LITE_DENSE | 226.4 ± 25.7 | 0.580 ± 0.084 | 288.6 | 0.950 |
| REWARD_NOISE | AC_LITE_SPARSE | 239.9 ± 14.6 | 0.650 ± 0.079 | 285.6 | 0.950 |
| REWARD_NOISE | BASELINE_DENSE | 226.8 ± 10.7 | 0.660 ± 0.089 | 287.8 | 0.970 |
| REWARD_NOISE | BASELINE_SPARSE | 253.1 ± 16.9 | 0.630 ± 0.076 | 286.1 | 0.960 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0 | 288.0 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 3 | 270.2 | 0.950 |
| CLEAN | AC_FULL_SPARSE | 9 | 254.5 | 0.850 |
| CLEAN | AC_FULL_SPARSE | 17 | 266.5 | 0.900 |
| CLEAN | AC_FULL_SPARSE | 42 | 274.7 | 1.000 |
| CLEAN | AC_LITE_DENSE | 0 | 261.5 | 0.950 |
| CLEAN | AC_LITE_DENSE | 3 | 209.3 | 0.700 |
| CLEAN | AC_LITE_DENSE | 9 | 221.0 | 0.850 |
| CLEAN | AC_LITE_DENSE | 17 | 267.7 | 1.000 |
| CLEAN | AC_LITE_DENSE | 42 | 256.7 | 0.850 |
| CLEAN | AC_LITE_SPARSE | 0 | 265.5 | 0.950 |
| CLEAN | AC_LITE_SPARSE | 3 | 263.5 | 0.950 |
| CLEAN | AC_LITE_SPARSE | 9 | 244.7 | 0.750 |
| CLEAN | AC_LITE_SPARSE | 17 | 277.3 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 42 | 260.8 | 0.900 |
| CLEAN | BASELINE_DENSE | 0 | 224.5 | 0.900 |
| CLEAN | BASELINE_DENSE | 3 | 194.8 | 0.650 |
| CLEAN | BASELINE_DENSE | 9 | 181.2 | 0.550 |
| CLEAN | BASELINE_DENSE | 17 | 219.1 | 0.750 |
| CLEAN | BASELINE_DENSE | 42 | 220.8 | 0.750 |
| CLEAN | BASELINE_SPARSE | 0 | 250.0 | 0.900 |
| CLEAN | BASELINE_SPARSE | 3 | 259.3 | 0.900 |
| CLEAN | BASELINE_SPARSE | 9 | 266.5 | 0.950 |
| CLEAN | BASELINE_SPARSE | 17 | 270.1 | 0.950 |
| CLEAN | BASELINE_SPARSE | 42 | 252.3 | 0.850 |
| OBS_NOISE | AC_FULL_SPARSE | 0 | 224.6 | 0.800 |
| OBS_NOISE | AC_FULL_SPARSE | 3 | 153.7 | 0.650 |
| OBS_NOISE | AC_FULL_SPARSE | 9 | 134.9 | 0.550 |
| OBS_NOISE | AC_FULL_SPARSE | 17 | 245.2 | 0.850 |
| OBS_NOISE | AC_FULL_SPARSE | 42 | 214.0 | 0.700 |
| OBS_NOISE | AC_LITE_DENSE | 0 | 256.7 | 0.950 |
| OBS_NOISE | AC_LITE_DENSE | 3 | 263.2 | 0.950 |
| OBS_NOISE | AC_LITE_DENSE | 9 | 245.8 | 0.900 |
| OBS_NOISE | AC_LITE_DENSE | 17 | 258.4 | 1.000 |
| OBS_NOISE | AC_LITE_DENSE | 42 | 248.5 | 0.950 |
| OBS_NOISE | AC_LITE_SPARSE | 0 | 194.9 | 0.800 |
| OBS_NOISE | AC_LITE_SPARSE | 3 | 235.5 | 0.850 |
| OBS_NOISE | AC_LITE_SPARSE | 9 | 246.7 | 0.900 |
| OBS_NOISE | AC_LITE_SPARSE | 17 | 252.2 | 0.950 |
| OBS_NOISE | AC_LITE_SPARSE | 42 | 173.2 | 0.600 |
| OBS_NOISE | BASELINE_DENSE | 0 | 191.8 | 0.650 |
| OBS_NOISE | BASELINE_DENSE | 3 | 173.2 | 0.600 |
| OBS_NOISE | BASELINE_DENSE | 9 | 205.3 | 0.800 |
| OBS_NOISE | BASELINE_DENSE | 17 | 184.9 | 0.750 |
| OBS_NOISE | BASELINE_DENSE | 42 | 152.8 | 0.400 |
| OBS_NOISE | BASELINE_SPARSE | 0 | 245.9 | 0.850 |
| OBS_NOISE | BASELINE_SPARSE | 3 | 218.6 | 0.800 |
| OBS_NOISE | BASELINE_SPARSE | 9 | 231.7 | 0.900 |
| OBS_NOISE | BASELINE_SPARSE | 17 | 227.9 | 0.850 |
| OBS_NOISE | BASELINE_SPARSE | 42 | 202.2 | 0.650 |
| REWARD_NOISE | AC_FULL_SPARSE | 0 | 256.9 | 0.600 |
| REWARD_NOISE | AC_FULL_SPARSE | 3 | 247.1 | 0.750 |
| REWARD_NOISE | AC_FULL_SPARSE | 9 | 237.2 | 0.600 |
| REWARD_NOISE | AC_FULL_SPARSE | 17 | 221.5 | 0.500 |
| REWARD_NOISE | AC_FULL_SPARSE | 42 | 249.3 | 0.600 |
| REWARD_NOISE | AC_LITE_DENSE | 0 | 203.3 | 0.650 |
| REWARD_NOISE | AC_LITE_DENSE | 3 | 246.9 | 0.650 |
| REWARD_NOISE | AC_LITE_DENSE | 9 | 252.4 | 0.600 |
| REWARD_NOISE | AC_LITE_DENSE | 17 | 233.8 | 0.450 |
| REWARD_NOISE | AC_LITE_DENSE | 42 | 195.5 | 0.550 |
| REWARD_NOISE | AC_LITE_SPARSE | 0 | 242.9 | 0.650 |
| REWARD_NOISE | AC_LITE_SPARSE | 3 | 235.7 | 0.700 |
| REWARD_NOISE | AC_LITE_SPARSE | 9 | 263.2 | 0.750 |
| REWARD_NOISE | AC_LITE_SPARSE | 17 | 224.4 | 0.550 |
| REWARD_NOISE | AC_LITE_SPARSE | 42 | 233.5 | 0.600 |
| REWARD_NOISE | BASELINE_DENSE | 0 | 234.8 | 0.650 |
| REWARD_NOISE | BASELINE_DENSE | 3 | 210.4 | 0.550 |
| REWARD_NOISE | BASELINE_DENSE | 9 | 226.3 | 0.650 |
| REWARD_NOISE | BASELINE_DENSE | 17 | 225.0 | 0.650 |
| REWARD_NOISE | BASELINE_DENSE | 42 | 237.8 | 0.800 |
| REWARD_NOISE | BASELINE_SPARSE | 0 | 243.7 | 0.700 |
| REWARD_NOISE | BASELINE_SPARSE | 3 | 240.4 | 0.550 |
| REWARD_NOISE | BASELINE_SPARSE | 9 | 245.7 | 0.650 |
| REWARD_NOISE | BASELINE_SPARSE | 17 | 282.2 | 0.700 |
| REWARD_NOISE | BASELINE_SPARSE | 42 | 253.6 | 0.550 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0 | AC_FULL_SPARSE_CLEAN_seed0_step0030000_policy.pt | CLEAN | 288.5 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 3 | AC_FULL_SPARSE_CLEAN_seed3_step0460000_policy.pt | CLEAN | 290.1 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 9 | AC_FULL_SPARSE_CLEAN_seed9_step0440000_policy.pt | CLEAN | 288.9 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 17 | AC_FULL_SPARSE_CLEAN_seed17_step0020000_policy.pt | CLEAN | 289.1 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 42 | AC_FULL_SPARSE_CLEAN_seed42_step0010000_policy.pt | CLEAN | 289.0 | 1.000 |
| CLEAN | AC_LITE_DENSE | 0 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | AC_LITE_DENSE | 3 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | AC_LITE_DENSE | 9 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | AC_LITE_DENSE | 17 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | AC_LITE_DENSE | 42 | AC_LITE_DENSE_CLEAN_seed42_step0230000_policy.pt | CLEAN | 288.4 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 0 | AC_LITE_SPARSE_CLEAN_seed0_step0020000_policy.pt | CLEAN | 288.0 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 3 | AC_LITE_SPARSE_CLEAN_seed3_step0210000_policy.pt | CLEAN | 289.4 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 9 | AC_LITE_SPARSE_CLEAN_seed9_step0030000_policy.pt | CLEAN | 289.7 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 17 | AC_LITE_SPARSE_CLEAN_seed17_step0010000_policy.pt | CLEAN | 289.6 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 42 | AC_LITE_SPARSE_CLEAN_seed42_step0050000_policy.pt | CLEAN | 289.2 | 1.000 |
| CLEAN | BASELINE_DENSE | 0 | BASELINE_DENSE_CLEAN_seed0_step0010000.pt | CLEAN | 287.9 | 1.000 |
| CLEAN | BASELINE_DENSE | 3 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_DENSE | 9 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_DENSE | 17 | BASELINE_DENSE_CLEAN_seed17_step0020000.pt | CLEAN | 288.4 | 1.000 |
| CLEAN | BASELINE_DENSE | 42 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_SPARSE | 0 | BASELINE_SPARSE_CLEAN_seed0_step0010000.pt | CLEAN | 287.6 | 1.000 |
| CLEAN | BASELINE_SPARSE | 3 | BASELINE_SPARSE_CLEAN_seed3_step0020000.pt | CLEAN | 288.2 | 1.000 |
| CLEAN | BASELINE_SPARSE | 9 | BASELINE_SPARSE_CLEAN_seed9_step0070000.pt | CLEAN | 287.8 | 1.000 |
| CLEAN | BASELINE_SPARSE | 17 | BASELINE_SPARSE_CLEAN_seed17_step0050000.pt | CLEAN | 287.4 | 1.000 |
| CLEAN | BASELINE_SPARSE | 42 | BASELINE_SPARSE_CLEAN_seed42_step0030000.pt | CLEAN | 289.7 | 1.000 |
| OBS_NOISE | AC_FULL_SPARSE | 0 | AC_FULL_SPARSE_OBS_NOISE_seed0_step0330000_policy.pt | OBS_NOISE | 286.3 | 1.000 |
| OBS_NOISE | AC_FULL_SPARSE | 3 | AC_FULL_SPARSE_OBS_NOISE_seed3_step0080000_policy.pt | OBS_NOISE | 283.7 | 1.000 |
| OBS_NOISE | AC_FULL_SPARSE | 9 | AC_FULL_SPARSE_OBS_NOISE_seed9_step0430000_policy.pt | OBS_NOISE | 288.2 | 1.000 |
| OBS_NOISE | AC_FULL_SPARSE | 17 | AC_FULL_SPARSE_OBS_NOISE_seed17_step0120000_policy.pt | OBS_NOISE | 289.6 | 1.000 |
| OBS_NOISE | AC_FULL_SPARSE | 42 | AC_FULL_SPARSE_OBS_NOISE_seed42_step0260000_policy.pt | OBS_NOISE | 288.7 | 1.000 |
| OBS_NOISE | AC_LITE_DENSE | 0 | AC_LITE_DENSE_OBS_NOISE_seed0_step0190000_policy.pt | OBS_NOISE | 282.4 | 1.000 |
| OBS_NOISE | AC_LITE_DENSE | 3 | AC_LITE_DENSE_OBS_NOISE_seed3_step0090000_policy.pt | OBS_NOISE | 282.8 | 1.000 |
| OBS_NOISE | AC_LITE_DENSE | 9 | AC_LITE_DENSE_OBS_NOISE_seed9_step0130000_policy.pt | OBS_NOISE | 287.1 | 1.000 |
| OBS_NOISE | AC_LITE_DENSE | 17 | AC_LITE_DENSE_OBS_NOISE_seed17_step0120000_policy.pt | OBS_NOISE | 285.7 | 1.000 |
| OBS_NOISE | AC_LITE_DENSE | 42 | AC_LITE_DENSE_OBS_NOISE_seed42_step0090000_policy.pt | OBS_NOISE | 283.1 | 1.000 |
| OBS_NOISE | AC_LITE_SPARSE | 0 | AC_LITE_SPARSE_OBS_NOISE_seed0_step0390000_policy.pt | OBS_NOISE | 291.5 | 1.000 |
| OBS_NOISE | AC_LITE_SPARSE | 3 | AC_LITE_SPARSE_OBS_NOISE_seed3_step0120000_policy.pt | OBS_NOISE | 283.8 | 1.000 |
| OBS_NOISE | AC_LITE_SPARSE | 9 | AC_LITE_SPARSE_OBS_NOISE_seed9_step0060000_policy.pt | OBS_NOISE | 291.8 | 1.000 |
| OBS_NOISE | AC_LITE_SPARSE | 17 | AC_LITE_SPARSE_OBS_NOISE_seed17_step0480000_policy.pt | OBS_NOISE | 281.9 | 1.000 |
| OBS_NOISE | AC_LITE_SPARSE | 42 | AC_LITE_SPARSE_OBS_NOISE_seed42_step0080000_policy.pt | OBS_NOISE | 286.2 | 1.000 |
| OBS_NOISE | BASELINE_DENSE | 0 | BASELINE_DENSE_OBS_NOISE_seed0_step0110000.pt | OBS_NOISE | 283.8 | 1.000 |
| OBS_NOISE | BASELINE_DENSE | 3 | BASELINE_DENSE_OBS_NOISE_seed3_step0260000.pt | OBS_NOISE | 278.9 | 1.000 |
| OBS_NOISE | BASELINE_DENSE | 9 | BASELINE_DENSE_OBS_NOISE_seed9_step0170000.pt | OBS_NOISE | 278.0 | 1.000 |
| OBS_NOISE | BASELINE_DENSE | 17 | BASELINE_DENSE_OBS_NOISE_seed17_step0090000.pt | OBS_NOISE | 282.2 | 1.000 |
| OBS_NOISE | BASELINE_DENSE | 42 | BASELINE_DENSE_OBS_NOISE_seed42_step0090000.pt | OBS_NOISE | 280.3 | 1.000 |
| OBS_NOISE | BASELINE_SPARSE | 0 | BASELINE_SPARSE_OBS_NOISE_seed0_step0440000.pt | OBS_NOISE | 285.1 | 1.000 |
| OBS_NOISE | BASELINE_SPARSE | 3 | BASELINE_SPARSE_OBS_NOISE_seed3_step0410000.pt | OBS_NOISE | 289.3 | 1.000 |
| OBS_NOISE | BASELINE_SPARSE | 9 | BASELINE_SPARSE_OBS_NOISE_seed9_step0090000.pt | OBS_NOISE | 287.5 | 1.000 |
| OBS_NOISE | BASELINE_SPARSE | 17 | BASELINE_SPARSE_OBS_NOISE_seed17_step0330000.pt | OBS_NOISE | 287.9 | 1.000 |
| OBS_NOISE | BASELINE_SPARSE | 42 | BASELINE_SPARSE_OBS_NOISE_seed42_step0120000.pt | OBS_NOISE | 287.0 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 0 | AC_FULL_SPARSE_REWARD_NOISE_seed0_step0010000_policy.pt | REWARD_NOISE | 287.5 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 3 | AC_FULL_SPARSE_REWARD_NOISE_seed3_step0010000_policy.pt | REWARD_NOISE | 289.5 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 9 | AC_FULL_SPARSE_REWARD_NOISE_seed9_step0020000_policy.pt | REWARD_NOISE | 289.2 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 17 | AC_FULL_SPARSE_REWARD_NOISE_seed17_step0010000_policy.pt | REWARD_NOISE | 288.3 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 42 | AC_FULL_SPARSE_REWARD_NOISE_seed42_step0030000_policy.pt | REWARD_NOISE | 288.1 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 0 | AC_LITE_DENSE_REWARD_NOISE_seed0_step0010000_policy.pt | REWARD_NOISE | 287.7 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 3 | AC_LITE_DENSE_REWARD_NOISE_seed3_step0010000_policy.pt | REWARD_NOISE | 286.2 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 9 | AC_LITE_DENSE_REWARD_NOISE_seed9_step0010000_policy.pt | REWARD_NOISE | 286.6 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 17 | AC_LITE_DENSE_REWARD_NOISE_seed17_step0010000_policy.pt | REWARD_NOISE | 288.4 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 42 | AC_LITE_DENSE_REWARD_NOISE_seed42_step0020000_policy.pt | REWARD_NOISE | 287.8 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 0 | AC_LITE_SPARSE_REWARD_NOISE_seed0_step0040000_policy.pt | REWARD_NOISE | 289.1 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 3 | AC_LITE_SPARSE_REWARD_NOISE_seed3_step0010000_policy.pt | REWARD_NOISE | 289.2 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 9 | AC_LITE_SPARSE_REWARD_NOISE_seed9_step0390000_policy.pt | REWARD_NOISE | 290.0 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 17 | AC_LITE_SPARSE_REWARD_NOISE_seed17_step0010000_policy.pt | REWARD_NOISE | 289.8 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 42 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 0 | BASELINE_DENSE_REWARD_NOISE_seed0_step0010000.pt | REWARD_NOISE | 287.9 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 3 | BASELINE_DENSE_REWARD_NOISE_seed3_step0050000.pt | REWARD_NOISE | 288.0 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 9 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 17 | BASELINE_DENSE_REWARD_NOISE_seed17_step0030000.pt | REWARD_NOISE | 288.3 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 42 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 0 | BASELINE_SPARSE_REWARD_NOISE_seed0_step0170000.pt | REWARD_NOISE | 289.2 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 3 | BASELINE_SPARSE_REWARD_NOISE_seed3_step0040000.pt | REWARD_NOISE | 288.1 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 9 | BASELINE_SPARSE_REWARD_NOISE_seed9_step0010000.pt | REWARD_NOISE | 290.9 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 17 | BASELINE_SPARSE_REWARD_NOISE_seed17_step0010000.pt | REWARD_NOISE | 289.3 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 42 | BASELINE_SPARSE_REWARD_NOISE_seed42_step0460000.pt | REWARD_NOISE | 289.9 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | test_clean | CLEAN | 0.10 | 276.8 ± 21.8 | 0.990 ± 0.100 |
| CLEAN | AC_FULL_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 210.1 ± 99.1 | 0.676 ± 0.468 |
| CLEAN | AC_FULL_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 29.8 ± 39.0 | 0.000 ± 0.000 |
| CLEAN | AC_LITE_DENSE | test_clean | CLEAN | 0.10 | 276.7 ± 20.3 | 1.000 ± 0.000 |
| CLEAN | AC_LITE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 218.7 ± 106.4 | 0.736 ± 0.441 |
| CLEAN | AC_LITE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 26.2 ± 36.3 | 0.000 ± 0.000 |
| CLEAN | AC_LITE_SPARSE | test_clean | CLEAN | 0.10 | 278.8 ± 19.4 | 1.000 ± 0.000 |
| CLEAN | AC_LITE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 200.1 ± 112.2 | 0.654 ± 0.476 |
| CLEAN | AC_LITE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 21.4 ± 32.9 | 0.000 ± 0.000 |
| CLEAN | BASELINE_DENSE | test_clean | CLEAN | 0.10 | 277.9 ± 20.8 | 1.000 ± 0.000 |
| CLEAN | BASELINE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 220.2 ± 105.3 | 0.746 ± 0.436 |
| CLEAN | BASELINE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 24.8 ± 36.3 | 0.000 ± 0.000 |
| CLEAN | BASELINE_SPARSE | test_clean | CLEAN | 0.10 | 276.5 ± 22.4 | 0.990 ± 0.100 |
| CLEAN | BASELINE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 173.6 ± 121.7 | 0.574 ± 0.495 |
| CLEAN | BASELINE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 16.2 ± 30.8 | 0.000 ± 0.000 |
| OBS_NOISE | AC_FULL_SPARSE | test_clean | CLEAN | 0.10 | 253.5 ± 65.0 | 0.884 ± 0.321 |
| OBS_NOISE | AC_FULL_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 266.7 ± 53.5 | 0.956 ± 0.205 |
| OBS_NOISE | AC_FULL_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 162.6 ± 131.6 | 0.568 ± 0.496 |
| OBS_NOISE | AC_LITE_DENSE | test_clean | CLEAN | 0.10 | 256.2 ± 37.6 | 0.956 ± 0.205 |
| OBS_NOISE | AC_LITE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 271.9 ± 35.8 | 0.974 ± 0.159 |
| OBS_NOISE | AC_LITE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 118.5 ± 95.2 | 0.230 ± 0.421 |
| OBS_NOISE | AC_LITE_SPARSE | test_clean | CLEAN | 0.10 | 264.5 ± 33.4 | 0.932 ± 0.252 |
| OBS_NOISE | AC_LITE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 262.2 ± 60.9 | 0.924 ± 0.265 |
| OBS_NOISE | AC_LITE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 139.3 ± 124.2 | 0.454 ± 0.498 |
| OBS_NOISE | BASELINE_DENSE | test_clean | CLEAN | 0.10 | 249.2 ± 48.1 | 0.908 ± 0.289 |
| OBS_NOISE | BASELINE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 262.0 ± 50.4 | 0.936 ± 0.245 |
| OBS_NOISE | BASELINE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 153.6 ± 113.5 | 0.446 ± 0.498 |
| OBS_NOISE | BASELINE_SPARSE | test_clean | CLEAN | 0.10 | 262.4 ± 42.5 | 0.934 ± 0.249 |
| OBS_NOISE | BASELINE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 260.0 ± 67.5 | 0.914 ± 0.281 |
| OBS_NOISE | BASELINE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 116.1 ± 123.8 | 0.362 ± 0.481 |
| REWARD_NOISE | AC_FULL_SPARSE | test_clean | CLEAN | 0.10 | 275.8 ± 31.7 | 0.980 ± 0.140 |
| REWARD_NOISE | AC_FULL_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 166.3 ± 123.4 | 0.530 ± 0.500 |
| REWARD_NOISE | AC_FULL_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 15.7 ± 32.2 | 0.000 ± 0.000 |
| REWARD_NOISE | AC_LITE_DENSE | test_clean | CLEAN | 0.10 | 276.7 ± 21.3 | 1.000 ± 0.000 |
| REWARD_NOISE | AC_LITE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 230.9 ± 94.7 | 0.792 ± 0.406 |
| REWARD_NOISE | AC_LITE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 29.7 ± 42.1 | 0.006 ± 0.077 |
| REWARD_NOISE | AC_LITE_SPARSE | test_clean | CLEAN | 0.10 | 277.7 ± 19.1 | 1.000 ± 0.000 |
| REWARD_NOISE | AC_LITE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 166.7 ± 120.1 | 0.510 ± 0.500 |
| REWARD_NOISE | AC_LITE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 13.1 ± 32.3 | 0.000 ± 0.000 |
| REWARD_NOISE | BASELINE_DENSE | test_clean | CLEAN | 0.10 | 277.2 ± 21.1 | 1.000 ± 0.000 |
| REWARD_NOISE | BASELINE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 219.7 ± 105.4 | 0.742 ± 0.438 |
| REWARD_NOISE | BASELINE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 26.8 ± 37.2 | 0.000 ± 0.000 |
| REWARD_NOISE | BASELINE_SPARSE | test_clean | CLEAN | 0.10 | 274.3 ± 28.6 | 0.970 ± 0.171 |
| REWARD_NOISE | BASELINE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 170.0 ± 113.1 | 0.508 ± 0.500 |
| REWARD_NOISE | BASELINE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 19.1 ± 31.5 | 0.000 ± 0.000 |

## Episode-level certainty summary

| mode | method | mean c_i | corr(c, delta) | corr(c, action prob) | corr(c, runner-up prob) |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0.800 | 0.022 | 0.036 | -0.012 |
| CLEAN | AC_LITE_DENSE | 0.814 | 0.308 | 0.359 | -0.290 |
| CLEAN | AC_LITE_SPARSE | 0.846 | 0.243 | 0.281 | -0.234 |
| OBS_NOISE | AC_FULL_SPARSE | 0.717 | 0.358 | 0.390 | -0.346 |
| OBS_NOISE | AC_LITE_DENSE | 0.856 | 0.183 | 0.205 | -0.171 |
| OBS_NOISE | AC_LITE_SPARSE | 0.865 | 0.201 | 0.220 | -0.194 |
| REWARD_NOISE | AC_FULL_SPARSE | 0.651 | 0.333 | 0.378 | -0.314 |
| REWARD_NOISE | AC_LITE_DENSE | 0.819 | 0.277 | 0.337 | -0.254 |
| REWARD_NOISE | AC_LITE_SPARSE | 0.844 | 0.185 | 0.211 | -0.186 |

## Certainty AUROC diagnostics

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0.627 | 0.531 |
| CLEAN | AC_LITE_DENSE | 0.350 | 0.338 |
| CLEAN | AC_LITE_SPARSE | 0.412 | 0.375 |
| OBS_NOISE | AC_FULL_SPARSE | 0.740 | 0.311 |
| OBS_NOISE | AC_LITE_DENSE | 0.559 | 0.390 |
| OBS_NOISE | AC_LITE_SPARSE | 0.645 | 0.393 |
| REWARD_NOISE | AC_FULL_SPARSE | 0.532 | 0.327 |
| REWARD_NOISE | AC_LITE_DENSE | 0.419 | 0.365 |
| REWARD_NOISE | AC_LITE_SPARSE | 0.484 | 0.390 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- CLEAN: `AC_FULL_SPARSE` has the highest mean final return (270.8) with mean final success 0.940.
- OBS_NOISE: `AC_LITE_DENSE` has the highest mean final return (254.5) with mean final success 0.950.
- REWARD_NOISE: `BASELINE_SPARSE` has the highest mean final return (253.1) with mean final success 0.630.

Checkpoint selection versus the pretrained anchor:

- CLEAN / AC_FULL_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- CLEAN / AC_LITE_DENSE: checkpoint 0 wins in 4 of 5 seeds (0.80).
- CLEAN / AC_LITE_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- CLEAN / BASELINE_DENSE: checkpoint 0 wins in 3 of 5 seeds (0.60).
- CLEAN / BASELINE_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- OBS_NOISE / AC_FULL_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- OBS_NOISE / AC_LITE_DENSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- OBS_NOISE / AC_LITE_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- OBS_NOISE / BASELINE_DENSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- OBS_NOISE / BASELINE_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- REWARD_NOISE / AC_FULL_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- REWARD_NOISE / AC_LITE_DENSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- REWARD_NOISE / AC_LITE_SPARSE: checkpoint 0 wins in 1 of 5 seeds (0.20).
- REWARD_NOISE / BASELINE_DENSE: checkpoint 0 wins in 2 of 5 seeds (0.40).
- REWARD_NOISE / BASELINE_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `AC_LITE_SPARSE` in `CLEAN` with 278.8 ± 19.4 and success 1.000 ± 0.000.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `AC_LITE_DENSE` in `OBS_NOISE` with 271.9 ± 35.8 and success 0.974 ± 0.159.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `AC_FULL_SPARSE` in `OBS_NOISE` with 162.6 ± 131.6 and success 0.568 ± 0.496.

Episode-level certainty behavior:

- CLEAN / AC_FULL_SPARSE: mean episode certainty 0.800, mean corr(certainty, delta) 0.022, mean corr(certainty, action_prob) 0.036, mean corr(certainty, runner_up_prob) -0.012.
- CLEAN / AC_LITE_DENSE: mean episode certainty 0.814, mean corr(certainty, delta) 0.308, mean corr(certainty, action_prob) 0.359, mean corr(certainty, runner_up_prob) -0.290.
- CLEAN / AC_LITE_SPARSE: mean episode certainty 0.846, mean corr(certainty, delta) 0.243, mean corr(certainty, action_prob) 0.281, mean corr(certainty, runner_up_prob) -0.234.
- OBS_NOISE / AC_FULL_SPARSE: mean episode certainty 0.717, mean corr(certainty, delta) 0.358, mean corr(certainty, action_prob) 0.390, mean corr(certainty, runner_up_prob) -0.346.
- OBS_NOISE / AC_LITE_DENSE: mean episode certainty 0.856, mean corr(certainty, delta) 0.183, mean corr(certainty, action_prob) 0.205, mean corr(certainty, runner_up_prob) -0.171.
- OBS_NOISE / AC_LITE_SPARSE: mean episode certainty 0.865, mean corr(certainty, delta) 0.201, mean corr(certainty, action_prob) 0.220, mean corr(certainty, runner_up_prob) -0.194.
- REWARD_NOISE / AC_FULL_SPARSE: mean episode certainty 0.651, mean corr(certainty, delta) 0.333, mean corr(certainty, action_prob) 0.378, mean corr(certainty, runner_up_prob) -0.314.
- REWARD_NOISE / AC_LITE_DENSE: mean episode certainty 0.819, mean corr(certainty, delta) 0.277, mean corr(certainty, action_prob) 0.337, mean corr(certainty, runner_up_prob) -0.254.
- REWARD_NOISE / AC_LITE_SPARSE: mean episode certainty 0.844, mean corr(certainty, delta) 0.185, mean corr(certainty, action_prob) 0.211, mean corr(certainty, runner_up_prob) -0.186.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

