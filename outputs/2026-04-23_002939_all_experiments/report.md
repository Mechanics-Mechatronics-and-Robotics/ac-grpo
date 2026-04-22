# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-23_002939_all_experiments`

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
- Challenge tests currently use up to 2440 episodes per evaluation seed in the generated logs.

## Summary table (mean ± std over seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 269.7 ± 12.2 | 0.940 ± 0.055 | 282.2 | 0.990 |
| CLEAN | AC_LITE_DENSE | 260.5 ± 22.2 | 0.910 ± 0.074 | 288.9 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 262.8 ± 6.5 | 0.920 ± 0.027 | 287.4 | 1.000 |
| CLEAN | BASELINE_DENSE | 253.0 ± 7.4 | 0.880 ± 0.057 | 283.0 | 1.000 |
| CLEAN | BASELINE_SPARSE | 244.1 ± 22.6 | 0.840 ± 0.096 | 277.4 | 0.990 |
| OBS_NOISE | AC_FULL_SPARSE | 107.0 ± 17.7 | 0.340 ± 0.074 | 175.7 | 0.590 |
| OBS_NOISE | AC_LITE_DENSE | 198.4 ± 40.8 | 0.630 ± 0.179 | 245.3 | 0.830 |
| OBS_NOISE | AC_LITE_SPARSE | 104.8 ± 52.5 | 0.300 ± 0.209 | 186.6 | 0.620 |
| OBS_NOISE | BASELINE_DENSE | 198.4 ± 33.3 | 0.650 ± 0.150 | 241.4 | 0.820 |
| OBS_NOISE | BASELINE_SPARSE | 163.8 ± 34.0 | 0.520 ± 0.152 | 194.3 | 0.630 |
| REWARD_NOISE | AC_FULL_SPARSE | 260.6 ± 20.8 | 0.760 ± 0.102 | 280.6 | 0.890 |
| REWARD_NOISE | AC_LITE_DENSE | 262.6 ± 9.9 | 0.720 ± 0.130 | 284.9 | 0.900 |
| REWARD_NOISE | AC_LITE_SPARSE | 242.1 ± 9.9 | 0.640 ± 0.082 | 280.8 | 0.880 |
| REWARD_NOISE | BASELINE_DENSE | 267.4 ± 6.5 | 0.750 ± 0.100 | 281.9 | 0.890 |
| REWARD_NOISE | BASELINE_SPARSE | 252.5 ± 19.0 | 0.730 ± 0.125 | 279.6 | 0.890 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0 | 280.3 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 3 | 264.5 | 0.900 |
| CLEAN | AC_FULL_SPARSE | 9 | 284.6 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 17 | 262.6 | 0.900 |
| CLEAN | AC_FULL_SPARSE | 42 | 256.2 | 0.900 |
| CLEAN | AC_LITE_DENSE | 0 | 265.0 | 0.900 |
| CLEAN | AC_LITE_DENSE | 3 | 278.9 | 1.000 |
| CLEAN | AC_LITE_DENSE | 9 | 269.9 | 0.900 |
| CLEAN | AC_LITE_DENSE | 17 | 266.6 | 0.950 |
| CLEAN | AC_LITE_DENSE | 42 | 222.0 | 0.800 |
| CLEAN | AC_LITE_SPARSE | 0 | 254.5 | 0.900 |
| CLEAN | AC_LITE_SPARSE | 3 | 264.4 | 0.950 |
| CLEAN | AC_LITE_SPARSE | 9 | 267.2 | 0.900 |
| CLEAN | AC_LITE_SPARSE | 17 | 257.7 | 0.900 |
| CLEAN | AC_LITE_SPARSE | 42 | 270.0 | 0.950 |
| CLEAN | BASELINE_DENSE | 0 | 257.3 | 0.900 |
| CLEAN | BASELINE_DENSE | 3 | 251.2 | 0.850 |
| CLEAN | BASELINE_DENSE | 9 | 260.2 | 0.950 |
| CLEAN | BASELINE_DENSE | 17 | 241.1 | 0.800 |
| CLEAN | BASELINE_DENSE | 42 | 255.1 | 0.900 |
| CLEAN | BASELINE_SPARSE | 0 | 276.4 | 0.950 |
| CLEAN | BASELINE_SPARSE | 3 | 257.1 | 0.900 |
| CLEAN | BASELINE_SPARSE | 9 | 230.3 | 0.800 |
| CLEAN | BASELINE_SPARSE | 17 | 219.8 | 0.700 |
| CLEAN | BASELINE_SPARSE | 42 | 237.0 | 0.850 |
| OBS_NOISE | AC_FULL_SPARSE | 0 | 134.8 | 0.450 |
| OBS_NOISE | AC_FULL_SPARSE | 3 | 97.9 | 0.350 |
| OBS_NOISE | AC_FULL_SPARSE | 9 | 98.9 | 0.300 |
| OBS_NOISE | AC_FULL_SPARSE | 17 | 113.7 | 0.350 |
| OBS_NOISE | AC_FULL_SPARSE | 42 | 89.9 | 0.250 |
| OBS_NOISE | AC_LITE_DENSE | 0 | 197.4 | 0.650 |
| OBS_NOISE | AC_LITE_DENSE | 3 | 209.0 | 0.600 |
| OBS_NOISE | AC_LITE_DENSE | 9 | 137.9 | 0.400 |
| OBS_NOISE | AC_LITE_DENSE | 17 | 252.1 | 0.900 |
| OBS_NOISE | AC_LITE_DENSE | 42 | 195.7 | 0.600 |
| OBS_NOISE | AC_LITE_SPARSE | 0 | 67.5 | 0.150 |
| OBS_NOISE | AC_LITE_SPARSE | 3 | 58.4 | 0.100 |
| OBS_NOISE | AC_LITE_SPARSE | 9 | 155.3 | 0.500 |
| OBS_NOISE | AC_LITE_SPARSE | 17 | 168.1 | 0.550 |
| OBS_NOISE | AC_LITE_SPARSE | 42 | 74.6 | 0.200 |
| OBS_NOISE | BASELINE_DENSE | 0 | 145.0 | 0.400 |
| OBS_NOISE | BASELINE_DENSE | 3 | 231.5 | 0.800 |
| OBS_NOISE | BASELINE_DENSE | 9 | 206.4 | 0.700 |
| OBS_NOISE | BASELINE_DENSE | 17 | 191.4 | 0.650 |
| OBS_NOISE | BASELINE_DENSE | 42 | 217.9 | 0.700 |
| OBS_NOISE | BASELINE_SPARSE | 0 | 133.6 | 0.400 |
| OBS_NOISE | BASELINE_SPARSE | 3 | 137.4 | 0.350 |
| OBS_NOISE | BASELINE_SPARSE | 9 | 192.2 | 0.650 |
| OBS_NOISE | BASELINE_SPARSE | 17 | 208.0 | 0.700 |
| OBS_NOISE | BASELINE_SPARSE | 42 | 147.9 | 0.500 |
| REWARD_NOISE | AC_FULL_SPARSE | 0 | 283.6 | 0.850 |
| REWARD_NOISE | AC_FULL_SPARSE | 3 | 237.6 | 0.800 |
| REWARD_NOISE | AC_FULL_SPARSE | 9 | 271.0 | 0.850 |
| REWARD_NOISE | AC_FULL_SPARSE | 17 | 271.5 | 0.650 |
| REWARD_NOISE | AC_FULL_SPARSE | 42 | 239.4 | 0.650 |
| REWARD_NOISE | AC_LITE_DENSE | 0 | 273.0 | 0.600 |
| REWARD_NOISE | AC_LITE_DENSE | 3 | 270.0 | 0.900 |
| REWARD_NOISE | AC_LITE_DENSE | 9 | 264.1 | 0.800 |
| REWARD_NOISE | AC_LITE_DENSE | 17 | 257.9 | 0.600 |
| REWARD_NOISE | AC_LITE_DENSE | 42 | 248.3 | 0.700 |
| REWARD_NOISE | AC_LITE_SPARSE | 0 | 236.4 | 0.550 |
| REWARD_NOISE | AC_LITE_SPARSE | 3 | 236.7 | 0.700 |
| REWARD_NOISE | AC_LITE_SPARSE | 9 | 237.0 | 0.600 |
| REWARD_NOISE | AC_LITE_SPARSE | 17 | 240.9 | 0.600 |
| REWARD_NOISE | AC_LITE_SPARSE | 42 | 259.4 | 0.750 |
| REWARD_NOISE | BASELINE_DENSE | 0 | 262.4 | 0.700 |
| REWARD_NOISE | BASELINE_DENSE | 3 | 268.8 | 0.850 |
| REWARD_NOISE | BASELINE_DENSE | 9 | 269.7 | 0.800 |
| REWARD_NOISE | BASELINE_DENSE | 17 | 276.2 | 0.800 |
| REWARD_NOISE | BASELINE_DENSE | 42 | 259.9 | 0.600 |
| REWARD_NOISE | BASELINE_SPARSE | 0 | 256.2 | 0.700 |
| REWARD_NOISE | BASELINE_SPARSE | 3 | 279.3 | 0.900 |
| REWARD_NOISE | BASELINE_SPARSE | 9 | 243.1 | 0.750 |
| REWARD_NOISE | BASELINE_SPARSE | 17 | 227.8 | 0.550 |
| REWARD_NOISE | BASELINE_SPARSE | 42 | 255.9 | 0.750 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0 | AC_FULL_SPARSE_CLEAN_seed0_step0010000_policy.pt | CLEAN | 286.1 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 3 | AC_FULL_SPARSE_CLEAN_seed3_step0020000_policy.pt | CLEAN | 286.4 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 9 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 17 | AC_FULL_SPARSE_CLEAN_seed17_step0010000_policy.pt | CLEAN | 287.0 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 42 | AC_FULL_SPARSE_CLEAN_seed42_step0010000_policy.pt | CLEAN | 288.8 | 1.000 |
| CLEAN | AC_LITE_DENSE | 0 | AC_LITE_DENSE_CLEAN_seed0_step0020000_policy.pt | CLEAN | 287.4 | 1.000 |
| CLEAN | AC_LITE_DENSE | 3 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | AC_LITE_DENSE | 9 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | AC_LITE_DENSE | 17 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | AC_LITE_DENSE | 42 | AC_LITE_DENSE_CLEAN_seed42_step0010000_policy.pt | CLEAN | 288.0 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 0 | AC_LITE_SPARSE_CLEAN_seed0_step0010000_policy.pt | CLEAN | 287.3 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 3 | AC_LITE_SPARSE_CLEAN_seed3_step0010000_policy.pt | CLEAN | 288.0 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 9 | AC_LITE_SPARSE_CLEAN_seed9_final_policy.pt | CLEAN | 287.9 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 17 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 42 | AC_LITE_SPARSE_CLEAN_seed42_step0020000_policy.pt | CLEAN | 286.0 | 1.000 |
| CLEAN | BASELINE_DENSE | 0 | BASELINE_DENSE_CLEAN_seed0_step0010000.pt | CLEAN | 287.8 | 1.000 |
| CLEAN | BASELINE_DENSE | 3 | BASELINE_DENSE_CLEAN_seed3_final.pt | CLEAN | 286.5 | 1.000 |
| CLEAN | BASELINE_DENSE | 9 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_DENSE | 17 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_DENSE | 42 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_SPARSE | 0 | BASELINE_SPARSE_CLEAN_seed0_step0010000.pt | CLEAN | 287.4 | 1.000 |
| CLEAN | BASELINE_SPARSE | 3 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_SPARSE | 9 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_SPARSE | 17 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_SPARSE | 42 | BASELINE_SPARSE_CLEAN_seed42_final.pt | CLEAN | 288.6 | 1.000 |
| OBS_NOISE | AC_FULL_SPARSE | 0 | AC_FULL_SPARSE_OBS_NOISE_seed0_step0020000_policy.pt | OBS_NOISE | 228.1 | 0.800 |
| OBS_NOISE | AC_FULL_SPARSE | 3 | checkpoint_0_pretrained | OBS_NOISE | 213.0 | 0.600 |
| OBS_NOISE | AC_FULL_SPARSE | 9 | checkpoint_0_pretrained | OBS_NOISE | 213.0 | 0.600 |
| OBS_NOISE | AC_FULL_SPARSE | 17 | AC_FULL_SPARSE_OBS_NOISE_seed17_final_policy.pt | OBS_NOISE | 280.9 | 1.000 |
| OBS_NOISE | AC_FULL_SPARSE | 42 | AC_FULL_SPARSE_OBS_NOISE_seed42_final_policy.pt | OBS_NOISE | 226.0 | 0.800 |
| OBS_NOISE | AC_LITE_DENSE | 0 | AC_LITE_DENSE_OBS_NOISE_seed0_step0020000_policy.pt | OBS_NOISE | 254.8 | 0.800 |
| OBS_NOISE | AC_LITE_DENSE | 3 | AC_LITE_DENSE_OBS_NOISE_seed3_step0020000_policy.pt | OBS_NOISE | 253.8 | 0.800 |
| OBS_NOISE | AC_LITE_DENSE | 9 | AC_LITE_DENSE_OBS_NOISE_seed9_step0010000_policy.pt | OBS_NOISE | 221.6 | 0.667 |
| OBS_NOISE | AC_LITE_DENSE | 17 | AC_LITE_DENSE_OBS_NOISE_seed17_step0020000_policy.pt | OBS_NOISE | 222.7 | 0.600 |
| OBS_NOISE | AC_LITE_DENSE | 42 | AC_LITE_DENSE_OBS_NOISE_seed42_final_policy.pt | OBS_NOISE | 249.0 | 0.800 |
| OBS_NOISE | AC_LITE_SPARSE | 0 | checkpoint_0_pretrained | OBS_NOISE | 213.0 | 0.600 |
| OBS_NOISE | AC_LITE_SPARSE | 3 | checkpoint_0_pretrained | OBS_NOISE | 213.0 | 0.600 |
| OBS_NOISE | AC_LITE_SPARSE | 9 | AC_LITE_SPARSE_OBS_NOISE_seed9_step0020000_policy.pt | OBS_NOISE | 236.5 | 0.733 |
| OBS_NOISE | AC_LITE_SPARSE | 17 | AC_LITE_SPARSE_OBS_NOISE_seed17_final_policy.pt | OBS_NOISE | 243.8 | 0.867 |
| OBS_NOISE | AC_LITE_SPARSE | 42 | checkpoint_0_pretrained | OBS_NOISE | 213.0 | 0.600 |
| OBS_NOISE | BASELINE_DENSE | 0 | checkpoint_0_pretrained | OBS_NOISE | 213.0 | 0.600 |
| OBS_NOISE | BASELINE_DENSE | 3 | BASELINE_DENSE_OBS_NOISE_seed3_step0020000.pt | OBS_NOISE | 222.0 | 0.600 |
| OBS_NOISE | BASELINE_DENSE | 9 | BASELINE_DENSE_OBS_NOISE_seed9_final.pt | OBS_NOISE | 245.3 | 0.800 |
| OBS_NOISE | BASELINE_DENSE | 17 | BASELINE_DENSE_OBS_NOISE_seed17_final.pt | OBS_NOISE | 261.4 | 0.867 |
| OBS_NOISE | BASELINE_DENSE | 42 | BASELINE_DENSE_OBS_NOISE_seed42_step0020000.pt | OBS_NOISE | 228.2 | 0.667 |
| OBS_NOISE | BASELINE_SPARSE | 0 | BASELINE_SPARSE_OBS_NOISE_seed0_final.pt | OBS_NOISE | 248.2 | 0.800 |
| OBS_NOISE | BASELINE_SPARSE | 3 | BASELINE_SPARSE_OBS_NOISE_seed3_final.pt | OBS_NOISE | 230.4 | 0.733 |
| OBS_NOISE | BASELINE_SPARSE | 9 | BASELINE_SPARSE_OBS_NOISE_seed9_step0020000.pt | OBS_NOISE | 270.3 | 0.867 |
| OBS_NOISE | BASELINE_SPARSE | 17 | BASELINE_SPARSE_OBS_NOISE_seed17_step0020000.pt | OBS_NOISE | 221.2 | 0.667 |
| OBS_NOISE | BASELINE_SPARSE | 42 | BASELINE_SPARSE_OBS_NOISE_seed42_step0020000.pt | OBS_NOISE | 217.4 | 0.667 |
| REWARD_NOISE | AC_FULL_SPARSE | 0 | AC_FULL_SPARSE_REWARD_NOISE_seed0_step0020000_policy.pt | REWARD_NOISE | 288.3 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 3 | AC_FULL_SPARSE_REWARD_NOISE_seed3_step0010000_policy.pt | REWARD_NOISE | 289.4 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 9 | AC_FULL_SPARSE_REWARD_NOISE_seed9_final_policy.pt | REWARD_NOISE | 287.2 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 17 | AC_FULL_SPARSE_REWARD_NOISE_seed17_step0010000_policy.pt | REWARD_NOISE | 287.1 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 42 | AC_FULL_SPARSE_REWARD_NOISE_seed42_step0010000_policy.pt | REWARD_NOISE | 288.3 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 0 | AC_LITE_DENSE_REWARD_NOISE_seed0_step0020000_policy.pt | REWARD_NOISE | 286.8 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 3 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 9 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 17 | AC_LITE_DENSE_REWARD_NOISE_seed17_step0020000_policy.pt | REWARD_NOISE | 287.0 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 42 | AC_LITE_DENSE_REWARD_NOISE_seed42_step0010000_policy.pt | REWARD_NOISE | 287.9 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 0 | AC_LITE_SPARSE_REWARD_NOISE_seed0_step0010000_policy.pt | REWARD_NOISE | 288.6 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 3 | AC_LITE_SPARSE_REWARD_NOISE_seed3_step0010000_policy.pt | REWARD_NOISE | 289.7 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 9 | AC_LITE_SPARSE_REWARD_NOISE_seed9_step0010000_policy.pt | REWARD_NOISE | 287.0 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 17 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 42 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 0 | BASELINE_DENSE_REWARD_NOISE_seed0_step0010000.pt | REWARD_NOISE | 287.8 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 3 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 9 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 17 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 42 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 0 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 3 | BASELINE_SPARSE_REWARD_NOISE_seed3_step0010000.pt | REWARD_NOISE | 288.5 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 9 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 17 | BASELINE_SPARSE_REWARD_NOISE_seed17_step0010000.pt | REWARD_NOISE | 290.0 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 42 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | test_clean | CLEAN | 0.10 | 277.8 ± 20.6 | 1.000 ± 0.000 |
| CLEAN | AC_FULL_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 206.4 ± 110.7 | 0.690 ± 0.463 |
| CLEAN | AC_FULL_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 23.9 ± 35.2 | 0.000 ± 0.000 |
| CLEAN | AC_LITE_DENSE | test_clean | CLEAN | 0.10 | 277.2 ± 20.7 | 1.000 ± 0.000 |
| CLEAN | AC_LITE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 217.4 ± 107.4 | 0.738 ± 0.440 |
| CLEAN | AC_LITE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 23.7 ± 33.9 | 0.000 ± 0.000 |
| CLEAN | AC_LITE_SPARSE | test_clean | CLEAN | 0.10 | 272.9 ± 31.8 | 0.988 ± 0.111 |
| CLEAN | AC_LITE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 222.3 ± 101.9 | 0.770 ± 0.421 |
| CLEAN | AC_LITE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 24.8 ± 37.3 | 0.003 ± 0.050 |
| CLEAN | BASELINE_DENSE | test_clean | CLEAN | 0.10 | 277.3 ± 20.2 | 1.000 ± 0.000 |
| CLEAN | BASELINE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 219.5 ± 106.2 | 0.740 ± 0.439 |
| CLEAN | BASELINE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 25.2 ± 35.6 | 0.003 ± 0.050 |
| CLEAN | BASELINE_SPARSE | test_clean | CLEAN | 0.10 | 277.6 ± 20.2 | 1.000 ± 0.000 |
| CLEAN | BASELINE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 207.3 ± 112.3 | 0.690 ± 0.463 |
| CLEAN | BASELINE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 22.4 ± 32.8 | 0.003 ± 0.050 |
| OBS_NOISE | AC_FULL_SPARSE | test_clean | CLEAN | 0.10 | 276.6 ± 27.0 | 0.977 ± 0.151 |
| OBS_NOISE | AC_FULL_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 207.3 ± 113.5 | 0.687 ± 0.465 |
| OBS_NOISE | AC_FULL_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 26.7 ± 42.3 | 0.017 ± 0.128 |
| OBS_NOISE | AC_LITE_DENSE | test_clean | CLEAN | 0.10 | 275.6 ± 20.0 | 1.000 ± 0.000 |
| OBS_NOISE | AC_LITE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 252.8 ± 75.0 | 0.873 ± 0.334 |
| OBS_NOISE | AC_LITE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 36.8 ± 49.2 | 0.010 ± 0.100 |
| OBS_NOISE | AC_LITE_SPARSE | test_clean | CLEAN | 0.10 | 278.1 ± 20.4 | 1.000 ± 0.000 |
| OBS_NOISE | AC_LITE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 215.4 ± 109.3 | 0.723 ± 0.448 |
| OBS_NOISE | AC_LITE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 25.2 ± 41.9 | 0.015 ± 0.122 |
| OBS_NOISE | BASELINE_DENSE | test_clean | CLEAN | 0.10 | 271.3 ± 26.0 | 0.983 ± 0.128 |
| OBS_NOISE | BASELINE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 236.6 ± 91.1 | 0.817 ± 0.388 |
| OBS_NOISE | BASELINE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 34.3 ± 48.3 | 0.003 ± 0.058 |
| OBS_NOISE | BASELINE_SPARSE | test_clean | CLEAN | 0.10 | 274.2 ± 24.0 | 0.983 ± 0.128 |
| OBS_NOISE | BASELINE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 218.1 ± 106.0 | 0.740 ± 0.439 |
| OBS_NOISE | BASELINE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 48.0 ± 75.3 | 0.077 ± 0.267 |
| REWARD_NOISE | AC_FULL_SPARSE | test_clean | CLEAN | 0.10 | 277.2 ± 24.9 | 0.988 ± 0.111 |
| REWARD_NOISE | AC_FULL_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 204.2 ± 112.5 | 0.680 ± 0.467 |
| REWARD_NOISE | AC_FULL_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 21.5 ± 33.4 | 0.000 ± 0.000 |
| REWARD_NOISE | AC_LITE_DENSE | test_clean | CLEAN | 0.10 | 277.3 ± 20.4 | 1.000 ± 0.000 |
| REWARD_NOISE | AC_LITE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 233.9 ± 93.2 | 0.794 ± 0.405 |
| REWARD_NOISE | AC_LITE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 31.0 ± 39.1 | 0.004 ± 0.063 |
| REWARD_NOISE | AC_LITE_SPARSE | test_clean | CLEAN | 0.10 | 274.4 ± 36.3 | 0.980 ± 0.140 |
| REWARD_NOISE | AC_LITE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 192.5 ± 117.2 | 0.630 ± 0.483 |
| REWARD_NOISE | AC_LITE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 18.0 ± 30.5 | 0.000 ± 0.000 |
| REWARD_NOISE | BASELINE_DENSE | test_clean | CLEAN | 0.10 | 277.4 ± 20.3 | 1.000 ± 0.000 |
| REWARD_NOISE | BASELINE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 218.0 ± 107.1 | 0.734 ± 0.442 |
| REWARD_NOISE | BASELINE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 24.5 ± 34.7 | 0.002 ± 0.045 |
| REWARD_NOISE | BASELINE_SPARSE | test_clean | CLEAN | 0.10 | 278.1 ± 21.1 | 1.000 ± 0.000 |
| REWARD_NOISE | BASELINE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 204.5 ± 112.9 | 0.684 ± 0.465 |
| REWARD_NOISE | BASELINE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 21.1 ± 33.8 | 0.000 ± 0.000 |

## Episode-level certainty summary

| mode | method | mean c_i | corr(c, delta) | corr(c, action prob) | corr(c, runner-up prob) |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0.564 | 0.147 | 0.167 | -0.147 |
| CLEAN | AC_LITE_DENSE | 0.560 | 0.159 | 0.182 | -0.153 |
| CLEAN | AC_LITE_SPARSE | 0.565 | 0.151 | 0.170 | -0.150 |
| OBS_NOISE | AC_FULL_SPARSE | 0.526 | 0.176 | 0.191 | -0.171 |
| OBS_NOISE | AC_LITE_DENSE | 0.565 | 0.037 | 0.042 | -0.035 |
| OBS_NOISE | AC_LITE_SPARSE | 0.575 | -0.046 | -0.050 | 0.043 |
| REWARD_NOISE | AC_FULL_SPARSE | 0.559 | 0.093 | 0.108 | -0.097 |
| REWARD_NOISE | AC_LITE_DENSE | 0.561 | 0.181 | 0.205 | -0.177 |
| REWARD_NOISE | AC_LITE_SPARSE | 0.565 | 0.072 | 0.082 | -0.082 |

## Certainty AUROC diagnostics

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0.491 | 0.465 |
| CLEAN | AC_LITE_DENSE | 0.355 | 0.460 |
| CLEAN | AC_LITE_SPARSE | 0.480 | 0.462 |
| OBS_NOISE | AC_FULL_SPARSE | 0.664 | 0.376 |
| OBS_NOISE | AC_LITE_DENSE | 0.557 | 0.488 |
| OBS_NOISE | AC_LITE_SPARSE | 0.451 | 0.515 |
| REWARD_NOISE | AC_FULL_SPARSE | 0.463 | 0.464 |
| REWARD_NOISE | AC_LITE_DENSE | 0.463 | 0.449 |
| REWARD_NOISE | AC_LITE_SPARSE | 0.440 | 0.473 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- CLEAN: `AC_FULL_SPARSE` has the highest mean final return (269.7) with mean final success 0.940.
- OBS_NOISE: `BASELINE_DENSE` has the highest mean final return (198.4) with mean final success 0.650.
- REWARD_NOISE: `BASELINE_DENSE` has the highest mean final return (267.4) with mean final success 0.750.

Checkpoint selection versus the pretrained anchor:

- CLEAN / AC_FULL_SPARSE: checkpoint 0 wins in 1 of 5 seeds (0.20).
- CLEAN / AC_LITE_DENSE: checkpoint 0 wins in 3 of 5 seeds (0.60).
- CLEAN / AC_LITE_SPARSE: checkpoint 0 wins in 1 of 5 seeds (0.20).
- CLEAN / BASELINE_DENSE: checkpoint 0 wins in 3 of 5 seeds (0.60).
- CLEAN / BASELINE_SPARSE: checkpoint 0 wins in 3 of 5 seeds (0.60).
- OBS_NOISE / AC_FULL_SPARSE: checkpoint 0 wins in 2 of 5 seeds (0.40).
- OBS_NOISE / AC_LITE_DENSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- OBS_NOISE / AC_LITE_SPARSE: checkpoint 0 wins in 3 of 5 seeds (0.60).
- OBS_NOISE / BASELINE_DENSE: checkpoint 0 wins in 1 of 5 seeds (0.20).
- OBS_NOISE / BASELINE_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- REWARD_NOISE / AC_FULL_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- REWARD_NOISE / AC_LITE_DENSE: checkpoint 0 wins in 2 of 5 seeds (0.40).
- REWARD_NOISE / AC_LITE_SPARSE: checkpoint 0 wins in 2 of 5 seeds (0.40).
- REWARD_NOISE / BASELINE_DENSE: checkpoint 0 wins in 4 of 5 seeds (0.80).
- REWARD_NOISE / BASELINE_SPARSE: checkpoint 0 wins in 3 of 5 seeds (0.60).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `BASELINE_SPARSE` in `REWARD_NOISE` with 278.1 ± 21.1 and success 1.000 ± 0.000.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `AC_LITE_DENSE` in `OBS_NOISE` with 252.8 ± 75.0 and success 0.873 ± 0.334.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `BASELINE_SPARSE` in `OBS_NOISE` with 48.0 ± 75.3 and success 0.077 ± 0.267.

Episode-level certainty behavior:

- CLEAN / AC_FULL_SPARSE: mean episode certainty 0.564, mean corr(certainty, delta) 0.147, mean corr(certainty, action_prob) 0.167, mean corr(certainty, runner_up_prob) -0.147.
- CLEAN / AC_LITE_DENSE: mean episode certainty 0.560, mean corr(certainty, delta) 0.159, mean corr(certainty, action_prob) 0.182, mean corr(certainty, runner_up_prob) -0.153.
- CLEAN / AC_LITE_SPARSE: mean episode certainty 0.565, mean corr(certainty, delta) 0.151, mean corr(certainty, action_prob) 0.170, mean corr(certainty, runner_up_prob) -0.150.
- OBS_NOISE / AC_FULL_SPARSE: mean episode certainty 0.526, mean corr(certainty, delta) 0.176, mean corr(certainty, action_prob) 0.191, mean corr(certainty, runner_up_prob) -0.171.
- OBS_NOISE / AC_LITE_DENSE: mean episode certainty 0.565, mean corr(certainty, delta) 0.037, mean corr(certainty, action_prob) 0.042, mean corr(certainty, runner_up_prob) -0.035.
- OBS_NOISE / AC_LITE_SPARSE: mean episode certainty 0.575, mean corr(certainty, delta) -0.046, mean corr(certainty, action_prob) -0.050, mean corr(certainty, runner_up_prob) 0.043.
- REWARD_NOISE / AC_FULL_SPARSE: mean episode certainty 0.559, mean corr(certainty, delta) 0.093, mean corr(certainty, action_prob) 0.108, mean corr(certainty, runner_up_prob) -0.097.
- REWARD_NOISE / AC_LITE_DENSE: mean episode certainty 0.561, mean corr(certainty, delta) 0.181, mean corr(certainty, action_prob) 0.205, mean corr(certainty, runner_up_prob) -0.177.
- REWARD_NOISE / AC_LITE_SPARSE: mean episode certainty 0.565, mean corr(certainty, delta) 0.072, mean corr(certainty, action_prob) 0.082, mean corr(certainty, runner_up_prob) -0.082.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

